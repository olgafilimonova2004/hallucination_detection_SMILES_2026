from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SAPLMA_SLICE = slice(0, 896)
ICR_SLICE = slice(896, 920)
LLM_SLICE = slice(920, 947)
RANDOM_STATE = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SAPLMAProbe(nn.Module):
    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        super().__init__()
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = []
        prev_dim = 896
        for hidden_dim in (256, 128, 64):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SAPLMAProbe":
        _set_seed(self.random_state)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        X_t = torch.from_numpy(X_arr).to(self.device)
        y_t = torch.from_numpy(y_arr).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.train()
        for _ in range(5):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self(batch_x)
                loss = criterion(logits, batch_y)
                l1_penalty = torch.zeros((), device=self.device)
                for parameter in self.net.parameters():
                    l1_penalty = l1_penalty + parameter.abs().sum()
                loss = loss + 1e-5 * l1_penalty
                loss.backward()
                optimizer.step()
        self.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        self.eval()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self(X_t)).detach().cpu().numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


class _ICRNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in (128, 64, 32):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class _ICRProbe(nn.Module):
    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        super().__init__()
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _ICRNetwork | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("ICR network is not fitted.")
        return self.net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ICRProbe":
        _set_seed(self.random_state)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.net = _ICRNetwork(X_arr.shape[1]).to(self.device)
        X_t = torch.from_numpy(X_arr).to(self.device)
        y_t = torch.from_numpy(y_arr).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.train()
        for _ in range(25):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                probs = self(batch_x)
                loss = criterion(probs, batch_y)
                loss.backward()
                optimizer.step()
        self.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        self.eval()
        with torch.no_grad():
            prob_pos = self(X_t).detach().cpu().numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


class _LLMCheckProbe(nn.Module):
    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        super().__init__()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LLMCheckProbe":
        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=np.float32))
        self.model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=10.0,
            class_weight=None,
            max_iter=2000,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LLM-Check model is not fitted.")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float32))
        prob_pos = self.model.predict_proba(X_scaled)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.meta_model = LogisticRegression(
            penalty="l2",
            C=0.01,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        self.threshold = 0.5
        self.icr_model: _ICRProbe | None = None
        self.llm_model: _LLMCheckProbe | None = None
        self.saplma_model: _SAPLMAProbe | None = None

    def _split_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        return X_arr[:, ICR_SLICE], X_arr[:, LLM_SLICE], X_arr[:, SAPLMA_SLICE]

    def _new_models(self, seed: int) -> tuple[_ICRProbe, _LLMCheckProbe, _SAPLMAProbe]:
        return _ICRProbe(seed), _LLMCheckProbe(seed), _SAPLMAProbe(seed)

    def _fit_models(self, X: np.ndarray, y: np.ndarray, seed: int) -> tuple[_ICRProbe, _LLMCheckProbe, _SAPLMAProbe]:
        X_icr, X_llm, X_saplma = self._split_features(X)
        icr_model, llm_model, saplma_model = self._new_models(seed)
        icr_model.fit(X_icr, y)
        llm_model.fit(X_llm, y)
        saplma_model.fit(X_saplma, y)
        return icr_model, llm_model, saplma_model

    def _base_probabilities(self, X: np.ndarray, icr_model: _ICRProbe, llm_model: _LLMCheckProbe, saplma_model: _SAPLMAProbe) -> np.ndarray:
        X_icr, X_llm, X_saplma = self._split_features(X)
        p_icr = icr_model.predict_proba(X_icr)[:, 1]
        p_llm = llm_model.predict_proba(X_llm)[:, 1]
        p_saplma = saplma_model.predict_proba(X_saplma)[:, 1]
        return np.column_stack([p_icr, p_llm, p_saplma])

    def _oof_probabilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=int)
        class_counts = np.bincount(y_arr, minlength=2)
        n_splits = int(min(5, class_counts[class_counts > 0].min()))
        if n_splits < 2:
            icr_model, llm_model, saplma_model = self._fit_models(X, y_arr, RANDOM_STATE)
            return self._base_probabilities(X, icr_model, llm_model, saplma_model)
        oof = np.zeros((len(y_arr), 3), dtype=np.float32)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for fold_idx, (idx_train, idx_holdout) in enumerate(skf.split(np.arange(len(y_arr)), y_arr)):
            icr_model, llm_model, saplma_model = self._fit_models(X[idx_train], y_arr[idx_train], RANDOM_STATE + fold_idx)
            oof[idx_holdout] = self._base_probabilities(X[idx_holdout], icr_model, llm_model, saplma_model)
        return oof

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        meta_X = self._oof_probabilities(X_arr, y_arr)
        self.meta_model.fit(meta_X, y_arr)
        self.icr_model, self.llm_model, self.saplma_model = self._fit_models(X_arr, y_arr, RANDOM_STATE + 100)
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_threshold = 0.5
        best_score = -1.0
        for threshold in candidates:
            score = accuracy_score(y_val, (probs >= threshold).astype(int))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        self.threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.icr_model is None or self.llm_model is None or self.saplma_model is None:
            raise RuntimeError("Stacking probe is not fitted.")
        meta_X = self._base_probabilities(np.asarray(X, dtype=np.float32), self.icr_model, self.llm_model, self.saplma_model)
        prob_pos = self.meta_model.predict_proba(meta_X)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
