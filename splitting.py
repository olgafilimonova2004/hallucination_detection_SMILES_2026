from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    del df, test_size
    idx = np.arange(len(y))
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if val_size <= 0.0:
        return [(idx_train_val, None, idx_test) for idx_train_val, idx_test in skf.split(idx, y)]
    relative_val = val_size / (1.0 - 1.0 / n_splits)
    relative_val = min(max(relative_val, 0.05), 0.5)
    splits = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(skf.split(idx, y)):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state + fold_idx,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))
    return splits
