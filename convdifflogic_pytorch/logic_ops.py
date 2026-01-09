
from __future__ import annotations

import torch


# Gate IDs follow the mapping used in the original DiffLogic code:
# 0: False
# 1: AND
# 2: not(A => B)  == A - A*B
# 3: A
# 4: not(B => A)  == B - A*B
# 5: B
# 6: XOR
# 7: OR
# 8: NOR
# 9: XNOR
# 10: not(B)    
# 11: (B => A)
# 12: not(A)
# 13: (A => B)
# 14: NAND
# 15: True


def bin_op(a: torch.Tensor, b: torch.Tensor, gate_id: int) -> torch.Tensor:
    """Apply one relaxed binary logic gate (probabilistic semantics) to tensors a and b."""
    if gate_id == 0:
        return torch.zeros_like(a)
    elif gate_id == 1:
        return a * b
    elif gate_id == 2:
        return a - a * b
    elif gate_id == 3:
        return a
    elif gate_id == 4:
        return b - a * b
    elif gate_id == 5:
        return b
    elif gate_id == 6:
        return a + b - 2 * a * b
    elif gate_id == 7:
        return a + b - a * b
    elif gate_id == 8:
        return 1 - (a + b - a * b)
    elif gate_id == 9:
        return 1 - (a + b - 2 * a * b)
    elif gate_id == 10:
        return 1 - b
    elif gate_id == 11:
        return 1 - b + a * b
    elif gate_id == 12:
        return 1 - a
    elif gate_id == 13:
        return 1 - a + a * b
    elif gate_id == 14:
        return 1 - a * b
    elif gate_id == 15:
        return torch.ones_like(a)
    else:
        raise ValueError(f"gate_id must be in [0, 15], got {gate_id}.")


def bin_op_s(a: torch.Tensor, b: torch.Tensor, gate_probs: torch.Tensor) -> torch.Tensor:
    """
    Apply a *mixture* (expectation) of the 16 relaxed logic gates.

    Args:
        a, b: tensors of identical shape.
        gate_probs: tensor whose last dimension is 16 and that broadcasts to a/b.
                   Typically this comes from softmax(logits) or one_hot(argmax).

    Returns:
        Tensor with the same shape as a/b.
    """
    # LogicTreeConv2d의 gate_probs의 shape은 (1, out_channels, node_count, 1, 16)
    if gate_probs.shape[-1] != 16:
        raise ValueError(f"gate_probs last dim must be 16, got {gate_probs.shape[-1]}")

    # LogicTreeConv2d의 a의 shape은 (B, out_channels, node_count, L)
    r = torch.zeros_like(a)
    # Not optimized: explicit loop is fine for a reference implementation.
    for i in range(16):
        # gate_probs[..., i]는 마지막 차원(16)에서 i만 뽑으므로 (1, out_channels, node_count, 1) 이 된다.
        # gate_probs가 boradcast된다.
        r = r + gate_probs[..., i] * bin_op(a, b, i)
    return r


def logits_to_gate_probs(
    logits: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """
    Convert per-gate logits to either soft gate probabilities (training) or
    hard one-hot selections (eval/inference), matching the discretization
    described in the papers.
    """
    if training:
        return torch.softmax(logits, dim=-1)
    # Hard discretization (argmax -> one hot). Still differentiable wrt inputs, but not logits.
    idx = logits.argmax(dim=-1)
    return torch.nn.functional.one_hot(idx, num_classes=16).to(dtype=logits.dtype)


def residual_init_logits(
    # LogicTreeConv2d 에서는 shape이 (out_channels, num_nodes, 16)
    shape: tuple[int, ...],
    *,  # 여기부터 뒤에 나오는 파라미터들은 위치 인자로 못 받고 키워드 인자로만 받는다.
    gate_id: int = 3,   # 3번 로직 게이트는 입력 a를 그대로 흘려 보낸다.
    z_value: float = 5.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Residual initialization from the NeurIPS'24 paper:
      set one gate (default 'A', id=3) to a large logit z_value
      and the others to 0.

    After softmax this yields a highly peaked distribution (roughly 90% on 'A'
    for z_value=5 with 16 gates).
    """
    logits = torch.zeros(*shape, device=device, dtype=dtype)
    logits[..., gate_id] = z_value
    return logits