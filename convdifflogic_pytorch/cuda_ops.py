"""CUDA-accelerated ops for Differentiable Logic Gate Networks.

This module mirrors the CUDA extension approach used in the original difflogic
repository, but is packaged for this C-LGN reference implementation.

If the CUDA extension is not available (e.g. NVCC not installed), the rest of the
library will automatically fall back to the pure PyTorch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import os
import torch


try:
    import convdifflogic_cuda  # type: ignore

    _CUDA_EXT_AVAILABLE = True
except Exception:  # pragma: no cover
    convdifflogic_cuda = None
    _CUDA_EXT_AVAILABLE = False

# Allow force-disabling the extension (useful for debugging / benchmarking).
#   CONVDIFFLOGIC_DISABLE_CUDA_EXT=1
if os.environ.get("CONVDIFFLOGIC_DISABLE_CUDA_EXT", "0") in {"1", "true", "True"}:
    _CUDA_EXT_AVAILABLE = False


def cuda_extension_available() -> bool:
    """Return True if the custom CUDA extension could be imported."""
    return bool(_CUDA_EXT_AVAILABLE)


class LogicLayerCudaFunction(torch.autograd.Function):
    """Autograd wrapper around the fused CUDA logic-layer kernels."""

    @staticmethod
    def forward(
        ctx,
        x_t: torch.Tensor,
        a_idx: torch.Tensor,
        b_idx: torch.Tensor,
        w: torch.Tensor,
        given_x_indices_of_y_start: torch.Tensor,
        given_x_indices_of_y: torch.Tensor,
    ) -> torch.Tensor:
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError(
                "convdifflogic_cuda extension is not available. "
                "Please build it (see README) or use the pure PyTorch fallback."
            )

        # Save tensors for backward.
        ctx.save_for_backward(x_t, a_idx, b_idx, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return convdifflogic_cuda.forward(x_t, a_idx, b_idx, w)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("convdifflogic_cuda extension is not available")

        x_t, a_idx, b_idx, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_x_t = None
        grad_w = None

        if ctx.needs_input_grad[0]:
            grad_x_t = convdifflogic_cuda.backward_x(
                x_t,
                a_idx,
                b_idx,
                w,
                grad_y,
                given_x_indices_of_y_start,
                given_x_indices_of_y,
            )
        if ctx.needs_input_grad[3]:
            grad_w = convdifflogic_cuda.backward_w(x_t, a_idx, b_idx, grad_y)

        return grad_x_t, None, None, grad_w, None, None


@dataclass(frozen=True)
class CSRConnections:
    """CSR-style adjacency from input indices -> output indices.

    Used by the CUDA backward kernel to efficiently accumulate gradients w.r.t. inputs.
    """

    start: torch.Tensor  # (in_dim+1,)
    indices: torch.Tensor  # (num_edges,)


def build_csr_connections(
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    *,
    in_dim: int,
    out_dim: int,
    device: Optional[torch.device] = None,
) -> CSRConnections:
    """Build CSR adjacency lists from (a_idx, b_idx).

    This is a vectorized alternative to the Python list-of-lists approach used in
    the original difflogic repo.

    Args:
        a_idx, b_idx: int64 tensors of shape (out_dim,)
        in_dim: number of inputs
        out_dim: number of outputs
        device: device for returned tensors (defaults to a_idx.device)

    Returns:
        CSRConnections with:
          - start: (in_dim+1,)
          - indices: (num_edges,) where num_edges == 2*out_dim
    """
    if device is None:
        device = a_idx.device

    a_idx = a_idx.to(device=device, dtype=torch.int64)
    b_idx = b_idx.to(device=device, dtype=torch.int64)

    # Each output y uses two inputs (a[y], b[y]). Build edge list of length 2*out_dim.
    y = torch.arange(out_dim, device=device, dtype=torch.int64)
    edge_inputs = torch.cat([a_idx, b_idx], dim=0)
    edge_outputs = torch.cat([y, y], dim=0)

    # Sort edges by input index so we can build CSR.
    order = torch.argsort(edge_inputs)
    edge_inputs_sorted = edge_inputs[order]
    edge_outputs_sorted = edge_outputs[order]

    # Count edges per input.
    counts = torch.bincount(edge_inputs_sorted, minlength=in_dim)
    start = torch.empty(in_dim + 1, device=device, dtype=torch.int64)
    start[0] = 0
    start[1:] = torch.cumsum(counts, dim=0)

    return CSRConnections(start=start.contiguous(), indices=edge_outputs_sorted.contiguous())


def logic_layer_cuda(
    x_t: torch.Tensor,
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    w: torch.Tensor,
    csr: CSRConnections,
) -> torch.Tensor:
    """Convenience wrapper around the CUDA autograd function."""
    return LogicLayerCudaFunction.apply(x_t, a_idx, b_idx, w, csr.start, csr.indices)


def prepare_pairwise_connections(
    *,
    out_groups: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create (a_idx, b_idx) for a grouped pairwise pattern.

    This matches the tree-level pairing used in LogicTreeConv2d:
      for each group g in [0,out_groups):
        outputs i in [0, width/2):
          a = g*width + 2*i
          b = g*width + 2*i + 1

    Args:
        out_groups: number of independent groups (e.g. out_channels)
        width: number of inputs per group (must be even)

    Returns:
        (a_idx, b_idx): int64 tensors of shape (out_groups*(width/2),)
    """
    if width % 2 != 0:
        raise ValueError(f"width must be even, got {width}")

    node_count = width // 2
    # For one group: a=[0,2,4,...], b=[1,3,5,...]
    base = torch.arange(node_count, dtype=torch.int64)
    a_one = 2 * base
    b_one = 2 * base + 1

    # Repeat for all groups with offset g*width.
    g = torch.arange(out_groups, dtype=torch.int64).unsqueeze(1)  # (G,1)
    offsets = g * int(width)
    a = (offsets + a_one.unsqueeze(0)).reshape(-1)
    b = (offsets + b_one.unsqueeze(0)).reshape(-1)
    return a.contiguous(), b.contiguous()