#!/usr/bin/env python3
"""research — SCMLM-R: State-Counterfactual MLM Ranking loss module.

Uses the SAME ordinary MLM head for both training and evaluation.
No side readout, no trained probe, exportable as ordinary DeBERTa MLM.

Loss components:
1. Direction: softplus(-m_A/tau) + softplus(-m_B/tau)
   Forces both counterfactual contexts to prefer their own correct answer.
2. Interaction: softplus((gamma - (m_A + m_B))/tau)
   Forces the sum of margins positive, eliminating fixed answer bias.
3. Multi-candidate: cross-entropy over true answer vs {counterfactual, other locations}
   Prevents solving by only comparing two candidates.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


def span_logprob(
    model,
    input_ids: torch.Tensor,      # (1, seq_len)
    attention_mask: torch.Tensor,  # (1, seq_len)
    mask_positions: torch.Tensor,  # (n_mask,)
    answer_token_ids: list[int],   # len == n_mask
) -> torch.Tensor:
    """Compute sum of log-probs at mask positions for given answer tokens.
    Returns a scalar tensor with grad."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]  # (seq, vocab)
    log_probs = F.log_softmax(logits[mask_positions], dim=-1)  # (n_mask, vocab)
    ans_ids = torch.tensor(answer_token_ids, device=log_probs.device)
    return log_probs.gather(1, ans_ids.unsqueeze(1)).sum()


def make_masked_input(
    passage: str,
    answer: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 256,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Replace the LAST occurrence of answer with [MASK] tokens and return tensors."""
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    n_mask = len(answer_ids)
    mask_phrase = " ".join([tokenizer.mask_token] * n_mask)
    idx = passage.rfind(answer)
    if idx < 0:
        raise ValueError(f"Answer '{answer}' not found in passage")
    masked_text = passage[:idx] + mask_phrase + passage[idx + len(answer):]
    enc = tokenizer(
        masked_text, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).view(-1)
    if len(mask_positions) != n_mask:
        raise ValueError(f"Mask count mismatch: {len(mask_positions)} vs {n_mask}")
    return input_ids, attn, mask_positions, answer_ids


def scmlm_r_loss(
    model,
    tokenizer: PreTrainedTokenizerFast,
    passage_a: str,
    passage_b: str,
    answer_a: str,
    answer_b: str,
    other_candidates: list[str] | None = None,
    tau: float = 1.0,
    gamma: float = 0.5,
    lambda_dir: float = 1.0,
    lambda_inter: float = 0.5,
    lambda_multi: float = 0.3,
    max_length: int = 256,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Compute full SCMLM-R loss for one counterfactual pair.
    
    Returns dict with 'loss', 'L_dir', 'L_inter', 'L_multi', 'm_A', 'm_B', 'I'.
    All are differentiable tensors.
    """
    # Tokenize answers (must have equal length — enforced by data generation)
    a_ids = tokenizer(answer_a, add_special_tokens=False)["input_ids"]
    b_ids = tokenizer(answer_b, add_special_tokens=False)["input_ids"]
    assert len(a_ids) == len(b_ids), f"Answer token length mismatch: {len(a_ids)} vs {len(b_ids)}"
    
    # Context A: mask answer_a position, score both candidates
    ids_a, attn_a, mask_a, _ = make_masked_input(passage_a, answer_a, tokenizer, max_length, device)
    s_A_a = span_logprob(model, ids_a, attn_a, mask_a, a_ids)
    s_A_b = span_logprob(model, ids_a, attn_a, mask_a, b_ids)
    
    # Context B: mask answer_b position, score both candidates
    ids_b, attn_b, mask_b, _ = make_masked_input(passage_b, answer_b, tokenizer, max_length, device)
    s_B_b = span_logprob(model, ids_b, attn_b, mask_b, b_ids)
    s_B_a = span_logprob(model, ids_b, attn_b, mask_b, a_ids)
    
    m_A = s_A_a - s_A_b  # should be positive
    m_B = s_B_b - s_B_a  # should be positive
    I = m_A + m_B         # interaction: should be positive
    
    # 1. Direction loss
    L_dir = F.softplus(-m_A / tau) + F.softplus(-m_B / tau)
    
    # 2. Interaction loss
    L_inter = F.softplus((gamma - I) / tau)
    
    # 3. Multi-candidate loss (optional)
    L_multi = torch.tensor(0.0, device=device)
    if other_candidates:
        # Context A: softmax over {a, b, others}
        other_scores_a = []
        other_scores_b = []
        for cand in other_candidates:
            c_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if len(c_ids) == len(a_ids):  # only use equal-length candidates
                other_scores_a.append(span_logprob(model, ids_a, attn_a, mask_a, c_ids))
                other_scores_b.append(span_logprob(model, ids_b, attn_b, mask_b, c_ids))
        
        if other_scores_a:
            # Context A: true is answer_a
            all_scores_a = torch.stack([s_A_a, s_A_b] + other_scores_a) / tau
            L_multi_a = -F.log_softmax(all_scores_a, dim=0)[0]
            # Context B: true is answer_b
            all_scores_b = torch.stack([s_B_b, s_B_a] + other_scores_b) / tau
            L_multi_b = -F.log_softmax(all_scores_b, dim=0)[0]
            L_multi = L_multi_a + L_multi_b
    
    # Total
    loss = lambda_dir * L_dir + lambda_inter * L_inter + lambda_multi * L_multi
    
    return {
        "loss": loss,
        "L_dir": L_dir.detach(),
        "L_inter": L_inter.detach(),
        "L_multi": L_multi.detach() if isinstance(L_multi, torch.Tensor) else torch.tensor(0.0),
        "m_A": m_A.detach(),
        "m_B": m_B.detach(),
        "I": I.detach(),
    }
