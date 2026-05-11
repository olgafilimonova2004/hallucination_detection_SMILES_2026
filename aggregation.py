from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import model as _model_module


MAX_SEQUENCE_LENGTH = 32768
SAPLMA_LAYER = 15
ICR_TOP_K = 10
HIDDEN_ALPHA = 1e-3
LOGIT_TOP_K = 50
ASSISTANT_MARKER = "<|im_start|>assistant\n"
USER_MARKER = "<|im_start|>user\n"
USER_END_MARKER = "\n<|im_end|>\n<|im_start|>assistant\n"
_FEATURE_QUEUE: deque[torch.Tensor] = deque()


def _find_last_subsequence(values: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return -1


def _response_span(input_ids: torch.Tensor, tokenizer) -> tuple[int, int]:
    ids = input_ids.tolist()
    marker_ids = tokenizer(ASSISTANT_MARKER, add_special_tokens=False)["input_ids"]
    marker_start = _find_last_subsequence(ids, marker_ids)
    if marker_start >= 0:
        return marker_start + len(marker_ids), len(ids)
    return max(0, len(ids) - 1), len(ids)


def _user_span(input_ids: torch.Tensor, tokenizer, response_start: int) -> tuple[int, int]:
    ids = input_ids.tolist()
    user_ids = tokenizer(USER_MARKER, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(USER_END_MARKER, add_special_tokens=False)["input_ids"]
    user_start = _find_last_subsequence(ids[:response_start], user_ids)
    user_end = _find_last_subsequence(ids[:response_start], end_ids)
    if user_start >= 0:
        user_start += len(user_ids)
    if user_start >= 0 and user_end >= 0 and user_end > user_start:
        return user_start, user_end
    return 0, response_start


def _clamp_log(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values.clamp_min(1e-12))


def _logit_features(logits: torch.Tensor, input_ids: torch.Tensor, response_span: tuple[int, int]) -> torch.Tensor:
    response_start, response_end = response_span
    effective_start = max(response_start, 1)
    if response_end <= effective_start:
        return logits.new_zeros(3)
    response_logits = logits[effective_start:response_end]
    response_input_ids = input_ids[effective_start:response_end]
    positions = torch.arange(effective_start, response_end, device=logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs[positions - 1, response_input_ids]
    perplexity = torch.exp(-target_log_probs.mean())
    full_probs = torch.softmax(response_logits, dim=-1)
    token_entropy = -(full_probs * _clamp_log(full_probs)).mean(dim=-1)
    window_entropy = token_entropy.max()
    k = min(LOGIT_TOP_K, response_logits.size(-1))
    top_probs = torch.softmax(torch.topk(response_logits, k, dim=-1).values, dim=-1)
    logit_entropy = -(top_probs * _clamp_log(top_probs)).mean()
    return torch.stack([perplexity, window_entropy, logit_entropy])


def _hidden_score(layer_hidden: torch.Tensor, response_span: tuple[int, int]) -> torch.Tensor:
    response_start, response_end = response_span
    response_hidden = layer_hidden[response_start:response_end]
    if response_hidden.size(0) == 0:
        return layer_hidden.new_tensor(0.0)
    centered = response_hidden - response_hidden.mean(dim=-1, keepdim=True)
    sigma = centered @ centered.transpose(0, 1)
    sigma = sigma + HIDDEN_ALPHA * torch.eye(sigma.size(0), dtype=sigma.dtype, device=sigma.device)
    return torch.log(torch.linalg.svdvals(sigma).clamp_min(1e-12)).mean()


def _llm_check_features(logits: torch.Tensor, input_ids: torch.Tensor, hidden_states: list[torch.Tensor], response_span: tuple[int, int]) -> torch.Tensor:
    logit_vector = _logit_features(logits, input_ids, response_span)
    hidden_vector = torch.stack([_hidden_score(hidden_states[layer_idx], response_span) for layer_idx in range(1, len(hidden_states))])
    return torch.cat([logit_vector, hidden_vector], dim=0)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=True)
    if not torch.isfinite(std) or float(std) < 1e-8:
        std = values.std(unbiased=False)
    return (values - values.mean()) / std.clamp_min(1e-8)


def _js_divergence(hidden_scores: torch.Tensor, attention_scores: torch.Tensor) -> torch.Tensor:
    p = F.softmax(_standardize(hidden_scores), dim=0).clamp_min(1e-12)
    q = F.softmax(_standardize(attention_scores), dim=0).clamp_min(1e-12)
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * torch.log(p / m)) + 0.5 * torch.sum(q * torch.log(q / m))


def _masked_attention_row(attention_row: torch.Tensor, user_span: tuple[int, int], response_start: int) -> torch.Tensor:
    user_start, user_end = user_span
    mask = torch.zeros_like(attention_row, dtype=torch.bool)
    if user_end > user_start:
        mask[user_start:user_end] = True
    if response_start < attention_row.numel():
        mask[response_start:] = True
    output = torch.zeros_like(attention_row)
    output[mask] = attention_row[mask]
    return output


def _icr_features(hidden_states: list[torch.Tensor], attentions: list[torch.Tensor], user_span: tuple[int, int], response_span: tuple[int, int]) -> torch.Tensor:
    response_start, response_end = response_span
    values = hidden_states[0].new_zeros(len(attentions))
    for layer_idx in range(len(attentions)):
        pooled_attention = attentions[layer_idx].mean(dim=0)
        previous_layer = hidden_states[layer_idx]
        current_layer = hidden_states[layer_idx + 1]
        token_scores = []
        for token_idx in range(response_start, response_end):
            attention_row = _masked_attention_row(pooled_attention[token_idx], user_span, response_start)
            top_k = min(ICR_TOP_K, attention_row.numel())
            top_attention, top_indices = torch.topk(attention_row, k=top_k)
            residual = current_layer[token_idx] - previous_layer[token_idx]
            attended = previous_layer.index_select(0, top_indices)
            norms = torch.linalg.vector_norm(attended, dim=-1).clamp_min(1e-8)
            projections = (attended * residual.unsqueeze(0)).sum(dim=-1) / norms
            token_scores.append(_js_divergence(projections, top_attention))
        if token_scores:
            values[layer_idx] = torch.stack(token_scores).mean()
    return values


class _ForwardWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, tokenizer) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = True
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs["use_cache"] = False
        outputs = self.base_model(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None and attention_mask is not None:
            _FEATURE_QUEUE.clear()
            for sample_idx in range(input_ids.size(0)):
                seq_len = int(attention_mask[sample_idx].sum().item())
                sample_ids = input_ids[sample_idx, :seq_len].detach().cpu()
                response_span = _response_span(sample_ids, self.tokenizer)
                user_span = _user_span(sample_ids, self.tokenizer, response_span[0])
                hidden_states = [layer[sample_idx, :seq_len].detach().to(device="cpu", dtype=torch.float32) for layer in outputs.hidden_states]
                attentions = [layer[sample_idx, :, :seq_len, :seq_len].detach().to(device="cpu", dtype=torch.float32) for layer in outputs.attentions]
                logits = outputs.logits[sample_idx, :seq_len].detach().to(device="cpu", dtype=torch.float32)
                icr = _icr_features(hidden_states, attentions, user_span, response_span)
                llm = _llm_check_features(logits, sample_ids, hidden_states, response_span)
                _FEATURE_QUEUE.append(torch.cat([icr, llm], dim=0))
        return outputs

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def _get_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return _ForwardWrapper(model, tokenizer), tokenizer


_model_module.MAX_LENGTH = MAX_SEQUENCE_LENGTH
_model_module.get_model_and_tokenizer = _get_model_and_tokenizer


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    last_pos = int(real_positions[-1].item())
    saplma = hidden_states[SAPLMA_LAYER, last_pos, :].detach().to(dtype=torch.float32, device="cpu")
    queued = _FEATURE_QUEUE.popleft() if _FEATURE_QUEUE else hidden_states.new_zeros(51).to(dtype=torch.float32, device="cpu")
    return torch.cat([saplma, queued], dim=0)


def extract_geometric_features(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return hidden_states.new_zeros(0)


def aggregation_and_feature_extraction(hidden_states: torch.Tensor, attention_mask: torch.Tensor, use_geometric: bool = False) -> torch.Tensor:
    features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        return torch.cat([features, extract_geometric_features(hidden_states, attention_mask).cpu()], dim=0)
    return features
