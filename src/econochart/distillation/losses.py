from __future__ import annotations

from typing import Any


def sparse_forward_kl_loss(
    student_logits: Any,
    teacher_topk_token_ids: Any,
    teacher_topk_logprobs: Any,
    *,
    valid_mask: Any | None = None,
    temperature: float = 1.0,
) -> Any:
    """Forward KL on teacher top-k support plus an exact residual-mass bucket."""
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.ndim != 3 or teacher_topk_token_ids.ndim != 3 or teacher_topk_logprobs.ndim != 3:
        raise ValueError("student logits and teacher top-k tensors must be rank 3")
    if teacher_topk_token_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError("teacher top-k token IDs and logprobs must have identical shapes")
    if student_logits.shape[:2] != teacher_topk_token_ids.shape[:2]:
        raise ValueError("student and teacher batch/token dimensions must match")
    if teacher_topk_token_ids.numel() and (
        teacher_topk_token_ids.min() < 0 or teacher_topk_token_ids.max() >= student_logits.shape[-1]
    ):
        raise ValueError("teacher top-k token ID is outside the student vocabulary")

    student_logprobs = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    student_topk_logprobs = student_logprobs.gather(-1, teacher_topk_token_ids.long())
    teacher_topk_logprobs = teacher_topk_logprobs.float()
    teacher_topk_probabilities = teacher_topk_logprobs.exp()
    teacher_topk_mass = teacher_topk_probabilities.sum(dim=-1)
    if torch.any(teacher_topk_mass > 1.0001):
        raise ValueError("teacher top-k probability mass exceeds 1")

    epsilon = torch.finfo(student_logprobs.dtype).eps
    teacher_tail = (1.0 - teacher_topk_mass).clamp_min(epsilon)
    student_topk_mass = student_topk_logprobs.exp().sum(dim=-1).clamp_max(1.0 - epsilon)
    student_tail_logprob = torch.log1p(-student_topk_mass)
    token_loss = (
        teacher_topk_probabilities * (teacher_topk_logprobs - student_topk_logprobs)
    ).sum(dim=-1) + teacher_tail * (teacher_tail.log() - student_tail_logprob)

    if valid_mask is not None:
        if valid_mask.shape != token_loss.shape:
            raise ValueError("valid_mask must match batch/token dimensions")
        mask = valid_mask.to(device=token_loss.device, dtype=token_loss.dtype)
        denominator = mask.sum()
        if denominator.item() <= 0:
            raise ValueError("valid_mask selects no tokens")
        loss = (token_loss * mask).sum() / denominator
    else:
        loss = token_loss.mean()
    return loss * (temperature**2)
