
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logic_ops import bin_op_s, logits_to_gate_probs, residual_init_logits
from .cuda_ops import (
    CSRConnections,
    build_csr_connections,
    cuda_extension_available,
    logic_layer_cuda,
    prepare_pairwise_connections,
)


def _pairwise_random_connections(in_dim: int, out_dim: int, *, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random fixed connections for a logic-gate layer: each output gate gets exactly 2 inputs.

    This follows the spirit of the original DiffLogic implementation:
    - choose 2*out_dim indices (with a permutation trick),
    - reshape into (2, out_dim).

    Note: out_dim*2 should be >= in_dim to ensure all inputs can appear at least once.
    """
    if out_dim * 2 < in_dim:
        raise ValueError(
            f"out_dim*2 must be >= in_dim so all inputs can be considered at least once. "
            f"Got in_dim={in_dim}, out_dim={out_dim}."
        )

    # We mimic the original code pattern: two permutations for mixing.
    perm = torch.randperm(2 * out_dim, generator=generator) % in_dim
    perm = torch.randperm(in_dim, generator=generator)[perm]
    c = perm.reshape(2, out_dim)
    a, b = c[0].long(), c[1].long()
    return a, b


class DifferentiableLogicLayer(nn.Module):
    """
    Fully-connected Differentiable Logic Gate layer (2-input sparse layer).

    - Fixed random connectivity: each output node selects 2 inputs.
    - Trainable parameters: logits over 16 logic gates per node.
    - Training forward uses softmax(logits) (expected relaxed gate).
    - Eval forward uses argmax(logits) (hard discretization).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        residual_init: bool = True,
        residual_gate_id: int = 3,  # 'A'
        residual_z: float = 5.0,
        init_std: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))

        a_idx, b_idx = _pairwise_random_connections(self.in_dim, self.out_dim, generator=gen)
        self.register_buffer("a_idx", a_idx)
        self.register_buffer("b_idx", b_idx)

        # Precompute CSR adjacency lists used by the CUDA backward kernel.
        # These will be moved automatically with .to(device) because they are buffers.
        csr = build_csr_connections(self.a_idx, self.b_idx, in_dim=self.in_dim, out_dim=self.out_dim)
        self.register_buffer("given_x_indices_of_y_start", csr.start)
        self.register_buffer("given_x_indices_of_y", csr.indices)

        if residual_init:
            logits = residual_init_logits((self.out_dim, 16), gate_id=residual_gate_id, z_value=residual_z, device=device, dtype=dtype)
        else:
            logits = torch.randn(self.out_dim, 16, device=device, dtype=dtype) * float(init_std)

        self.logits = nn.Parameter(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) with values in [0,1] during differentiable training,
               and ideally {0,1} during discretized inference.

        Returns:
            (B, out_dim)
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D (B, in_dim), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.shape[-1]}")

        # --- CUDA fast path -------------------------------------------------
        # Uses the fused CUDA kernel from the original difflogic implementation.
        if (
            x.is_cuda
            and cuda_extension_available()
            and x.dtype in (torch.float16, torch.float32)
        ):
            # CUDA kernel expects (in_dim, batch) layout.
            x_t = x.transpose(0, 1).contiguous()  # (in_dim, B)
            w = logits_to_gate_probs(self.logits, training=self.training).to(dtype=x_t.dtype).contiguous()  # (out_dim,16)
            csr = CSRConnections(self.given_x_indices_of_y_start, self.given_x_indices_of_y)
            y_t = logic_layer_cuda(x_t, self.a_idx, self.b_idx, w, csr)  # (out_dim, B)
            return y_t.transpose(0, 1)

        # --- Pure PyTorch fallback -----------------------------------------
        a = x[:, self.a_idx]
        b = x[:, self.b_idx]
        gate_probs = logits_to_gate_probs(self.logits, training=self.training)  # (out_dim, 16)
        gate_probs = gate_probs.unsqueeze(0)  # (1, out_dim, 16)
        return bin_op_s(a, b, gate_probs)


class GroupSum(nn.Module):
    """
    Group-sum readout used in the DiffLogic papers.

    Splits the last dimension into k groups and sums within each group to yield k logits.
    """

    def __init__(self, k: int, tau: float = 1.0):
        super().__init__()
        self.k = int(k)
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.k != 0:
            raise ValueError(f"Last dim {x.shape[-1]} must be divisible by k={self.k}")
        g = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(dim=-1)
        return g / self.tau


@dataclass(frozen=True)
class LogicTreeConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int] = (3, 3)
    tree_depth: int = 3  # depth d: leaves=2^d, nodes=2^d-1
    stride: int = 1
    padding: int = 1  # like conv2d padding
    in_channels_per_tree: Optional[int] = 2  # paper uses 2 for routing/inductive bias, can set None for "all"
    residual_init: bool = True
    residual_gate_id: int = 3
    residual_z: float = 5.0
    init_std: float = 1.0
    seed: Optional[int] = None


class LogicTreeConv2d(nn.Module):
    """
    Convolutional logic gate tree layer:

    - Each output channel has a *fixed* random selection of inputs (leaves) from the receptive field.
    - Each output channel is parameterized by a complete binary tree of depth d:
        leaves = 2^d inputs, nodes = 2^d - 1 learnable logic gates.
    - Parameters are *shared across spatial locations* (i.e., convolutional weight sharing).
    - Training uses differentiable relaxation (softmax mixture of relaxed gates).
    - Eval uses hard discretization (argmax gate per node).

    This implementation prioritizes clarity over performance.
    """

    def __init__(self, cfg: LogicTreeConvConfig):
        super().__init__()
        self.cfg = cfg

        kh, kw = cfg.kernel_size
        if kh <= 0 or kw <= 0:
            raise ValueError(f"kernel_size must be positive, got {cfg.kernel_size}")
        if cfg.tree_depth <= 0:
            raise ValueError(f"tree_depth must be >=1, got {cfg.tree_depth}")

        self.in_channels = int(cfg.in_channels)
        self.out_channels = int(cfg.out_channels)
        self.kernel_size = (int(kh), int(kw))
        self.tree_depth = int(cfg.tree_depth)
        self.stride = int(cfg.stride)
        self.padding = int(cfg.padding)

        self.num_leaves = 2 ** self.tree_depth
        self.num_nodes = self.num_leaves - 1

        # --- Random fixed leaf connections (which inputs inside the patch are used as leaves) ---
        # leaf_indices의 shape은 (out_channels, num_leaves) 형태이다.
        leaf_indices = self._init_leaf_indices(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kh=self.kernel_size[0],
            kw=self.kernel_size[1],
            num_leaves=self.num_leaves,
            in_channels_per_tree=cfg.in_channels_per_tree,
            seed=cfg.seed,
        )
        self.register_buffer("leaf_indices", leaf_indices)  # (out_channels, num_leaves), indices into unfolded patches

        # --- Learnable gate logits for each node in the tree ---
        if cfg.residual_init:
            logits = residual_init_logits(
                (self.out_channels, self.num_nodes, 16),
                gate_id=cfg.residual_gate_id,
                z_value=cfg.residual_z,
                device=None,
                dtype=torch.float32,
            )
        else:
            logits = torch.randn(self.out_channels, self.num_nodes, 16) * float(cfg.init_std)

        self.logits = nn.Parameter(logits)

        # --- Precompute tree-level pairwise connections for the CUDA fast path ---
        # Each tree level is a grouped 2-input logic-gate layer with a deterministic
        # "(0,1), (2,3), ..." pairing within each output channel.
        #
        # The fused CUDA kernel expects a CSR adjacency representation for the backward.
        width = self.num_leaves
        for level in range(self.tree_depth):
            node_count = width // 2
            in_dim_level = self.out_channels * width
            out_dim_level = self.out_channels * node_count

            a_idx_l, b_idx_l = prepare_pairwise_connections(out_groups=self.out_channels, width=width)
            csr_l = build_csr_connections(a_idx_l, b_idx_l, in_dim=in_dim_level, out_dim=out_dim_level)

            self.register_buffer(f"cuda_a_idx_l{level}", a_idx_l)
            self.register_buffer(f"cuda_b_idx_l{level}", b_idx_l)
            self.register_buffer(f"cuda_given_start_l{level}", csr_l.start)
            self.register_buffer(f"cuda_given_indices_l{level}", csr_l.indices)

            width = node_count

    @staticmethod
    def _init_leaf_indices(
        *,
        in_channels: int,
        out_channels: int,
        kh: int,
        kw: int,
        num_leaves: int,
        in_channels_per_tree: Optional[int],
        seed: Optional[int],
    ) -> torch.Tensor:
        """
        Create random indices into an unfolded patch vector of length in_channels*kh*kw.

        The unfolded patch is assumed to be ordered as:
            [c0 (kh*kw entries), c1 (kh*kw entries), ..., c_{C-1} (...)]
        with per-channel row-major ordering within the spatial kernel.
        """
        gen = torch.Generator()
        if seed is None:
            # No deterministic seed -> still create generator but leave random.
            seed = torch.seed()
        gen.manual_seed(int(seed))

        patch_len = in_channels * kh * kw
        # leaf_indices[oc, leaf_id]는 oc번째 출력 트리(채널)의 leaf_id번째 leaf가 볼 입력 위치 의미
        leaf_indices = torch.empty(out_channels, num_leaves, dtype=torch.long)

        # 각 출력 채널마다 1개의 로직 트리가 생긴다.
        for oc in range(out_channels):
            if in_channels_per_tree is None or in_channels_per_tree >= in_channels:
                allowed_channels = torch.arange(in_channels, dtype=torch.long)
            else:
                # sample a subset of channels for this tree
                # 각 출력 마다 로직 트리가 선택하는 receptive field의 in_channels 개수는 in_channels_per_tree만큼 고정되어있다.
                # 예를 들어 in_channels의 개수가 8개이면 그 중 2개(inchannels_per_tree)만 뽑혀서 로직 트리 리프 노드의 인덱스 후보가 된다.
                perm = torch.randperm(in_channels, generator=gen)[: int(in_channels_per_tree)]
                allowed_channels = perm.to(torch.long)

            # For each leaf pick (channel, dy, dx).
            # (Channel is chosen from allowed_channels.)
            # torch.randint(low, high, size) 가 기본이고 low <= 값 < high 범위의 정수를 size만큼 뽑아서 1D 텐서를 반환한다.
            # torch.randint(high, sizem, ...) 처럼 low를 생략하면 자동으로 0부터 뽑는다. 아래를 그런 방식으로 사용했다.
            ch_sel = allowed_channels[torch.randint(len(allowed_channels), (num_leaves,), generator=gen)]
            dy = torch.randint(kh, (num_leaves,), generator=gen)
            dx = torch.randint(kw, (num_leaves,), generator=gen)

            # patch를 펼친 순서는  채널 0의 (kh*kw)개 먼저 그 다음 채널 1의 (kh*kw) ... 식으로 펼쳐진다.
            # 또 각 채널 내부의 (kh,kw) 는 row-major (dy, dx) --> dy*kw + dx
            # 즉 어떤 (channel, dy, dx)가 펼친 벡터에서 차지하는 index = channel*(kw*kh) + dy*kw + dx
            leaf_indices[oc] = ch_sel * (kh * kw) + dy * kw + dx

        # sanity
        if leaf_indices.min() < 0 or leaf_indices.max() >= patch_len:
            raise RuntimeError("Leaf index generation produced out-of-range indices.")
        return leaf_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W), values in [0,1] during training.

        Returns:
            (B, C_out, H_out, W_out)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        B, C, H, W = x.shape
        kh, kw = self.kernel_size

        # Pad like conv2d.
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Unfold into patches: (B, C*kh*kw, L) where L = H_out * W_out
        patches = F.unfold(x, kernel_size=(kh, kw), stride=self.stride)  # (B, C*kh*kw, L)
        B2, patch_len, L = patches.shape

        # Gather leaves: advanced indexing yields (B, out_channels, num_leaves, L)
        # patches : (B, C*kh*kw, L)
        # self.leaf_indices의 shape은 (out_channels, num_leaves) 형태이다.
        leaves = patches[:, self.leaf_indices, :]

        # --- CUDA fast path -------------------------------------------------
        # The expensive part of this layer is repeatedly applying the 16-gate mixture
        # at every tree node. The original difflogic repo provides a fused CUDA kernel
        # for a sparse 2-input logic layer. We can map each tree level to such a logic
        # layer by treating each (out_channel, node) as an independent neuron and
        # flattening the spatial positions (L) into the batch dimension.
        if (
            leaves.is_cuda
            and cuda_extension_available()
            and leaves.dtype in (torch.float16, torch.float32)
        ):
            OC = self.out_channels
            # Keep the intermediate in (OC, width, B, L) contiguous layout so that
            # each feature (oc, width_idx) is contiguous across the flattened samples.
            cur = leaves.permute(1, 2, 0, 3).contiguous()  # (OC, width, B, L)
            node_ptr = 0
            width = self.num_leaves

            for level in range(self.tree_depth):
                node_count = width // 2

                level_logits = self.logits[:, node_ptr : node_ptr + node_count, :]  # (OC, node_count, 16)
                w = logits_to_gate_probs(level_logits, training=self.training)
                w = w.reshape(OC * node_count, 16).to(dtype=cur.dtype).contiguous()

                # Flatten (B, L) into the batch dimension.
                x_t = cur.reshape(OC * width, B * L)  # (in_dim, batch)

                a_idx = getattr(self, f"cuda_a_idx_l{level}")
                b_idx = getattr(self, f"cuda_b_idx_l{level}")
                start = getattr(self, f"cuda_given_start_l{level}")
                idx = getattr(self, f"cuda_given_indices_l{level}")
                csr = CSRConnections(start, idx)

                y_t = logic_layer_cuda(x_t, a_idx, b_idx, w, csr)  # (OC*node_count, B*L)
                cur = y_t.reshape(OC, node_count, B, L)  # (OC, node_count, B, L)

                node_ptr += node_count
                width = node_count

            # Root: (OC, 1, B, L) -> (B, OC, L)
            y = cur.squeeze(1).permute(2, 0, 1).contiguous()

            H_padded = H + 2 * self.padding
            W_padded = W + 2 * self.padding
            H_out = (H_padded - kh) // self.stride + 1
            W_out = (W_padded - kw) // self.stride + 1
            return y.reshape(B, OC, H_out, W_out)

        # Compute tree bottom-up. We store logits in "level order" from bottom to top:
        # first num_leaves/2 nodes, then /4, ..., root.
        cur = leaves  # (B, out_channels, num_leaves, L)
        node_ptr = 0
        width = self.num_leaves

        for _level in range(self.tree_depth):
            node_count = width // 2
            a = cur[:, :, 0::2, :]  # (B, out_channels, node_count, L)
            b = cur[:, :, 1::2, :]
            # self.logits의 shape은 (out_channels, num_nodes, 16)
            level_logits = self.logits[:, node_ptr: node_ptr + node_count, :]  # (out_channels, node_count, 16)

            gate_probs = logits_to_gate_probs(level_logits, training=self.training)
            # Broadcast to (B, out_channels, node_count, L, 16):
            gate_probs = gate_probs.unsqueeze(0).unsqueeze(-2)  # (1, out_channels, node_count, 1, 16) --> unsqueeze(dim)은 해당 위치에 크기 1인 차원을 넣는다.

            # bin_op_s expects gate_probs broadcastable to a/b.
            out = bin_op_s(a, b, gate_probs)  # (B, out_channels, node_count, L)

            cur = out
            node_ptr += node_count
            width = node_count

        # Root output: (B, out_channels, 1, L) -> (B, out_channels, L)
        y = cur.squeeze(2)
        # Reshape L back to spatial map.
        # Compute output spatial size for stride=1 (with padding already done in unfold):
        # H_out = floor((H_padded - kh)/stride) + 1
        H_padded = H + 2 * self.padding
        W_padded = W + 2 * self.padding
        H_out = (H_padded - kh) // self.stride + 1
        W_out = (W_padded - kw) // self.stride + 1
        y = y.reshape(B, self.out_channels, H_out, W_out)
        return y


class OrPool2d(nn.Module):
    """
    Logical OR pooling with max t-conorm relaxation (NeurIPS'24):
        OR(a,b) relaxed as max(a,b).

    In practice this is just MaxPool2d.
    """

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)