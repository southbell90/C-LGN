
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DifferentiableLogicLayer, GroupSum, LogicTreeConv2d, LogicTreeConvConfig, OrPool2d


def binarize_with_thresholds(x: torch.Tensor, thresholds: Sequence[float]) -> torch.Tensor:
    """
    Expand an input tensor into multiple binary channels by thresholding.

    Args:
        x: (B, C, H, W) float tensor, typically in [0,1].
        thresholds: list of thresholds t; output has C*len(thresholds) channels.

    Returns:
        (B, C*len(thresholds), H, W) float tensor in {0,1}.
    """
    outs = [(x > float(t)).to(dtype=x.dtype) for t in thresholds]
    return torch.cat(outs, dim=1)


@dataclass(frozen=True)
class LogicTreeNetCIFARConfig:
    # Width multiplier "k" from the paper; final channel sizes are multiples of k.
    k: int = 32
    tree_depth: int = 3
    # Input binarization thresholds for small models (2-bit precision => 3 thresholds).
    input_thresholds: Tuple[float, ...] = (0.25, 0.5, 0.75)
    # Softmax temperature used in the final GroupSum (controls logit scale).
    tau: float = 100.0
    # Convolutional layer kernel size (paper uses 3x3).
    conv_kernel: Tuple[int, int] = (3, 3)
    conv_padding: int = 1
    in_channels_per_tree: Optional[int] = 2
    residual_init: bool = True
    residual_z: float = 5.0
    seed: Optional[int] = None
    num_classes: int = 10
    # Number of output neurons per class in last logic layer (n_ll/c in paper).
    outputs_per_class: int = 1000


class LogicTreeNetCIFAR10(nn.Module):
    def __init__(self, cfg: LogicTreeNetCIFARConfig):
        super().__init__()
        self.cfg = cfg

        # Input binarization expands channels. CIFAR has 3 RGB channels -> 3 * len(thresholds)
        self.input_thresholds = cfg.input_thresholds
        in_ch = 3 * len(cfg.input_thresholds)

        def conv(out_ch: int, seed_offset: int) -> LogicTreeConv2d:
            conv_cfg = LogicTreeConvConfig(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=cfg.conv_kernel,
                tree_depth=cfg.tree_depth,
                stride=1,
                padding=cfg.conv_padding,
                in_channels_per_tree=cfg.in_channels_per_tree,
                residual_init=cfg.residual_init,
                residual_gate_id=3,
                residual_z=cfg.residual_z,
                seed=None if cfg.seed is None else int(cfg.seed) + seed_offset,
            )
            return LogicTreeConv2d(conv_cfg)

        k = int(cfg.k)

        # Convolutional trunk
        self.conv1 = conv(k, 1)
        self.pool1 = OrPool2d(2, 2)

        in_ch = k
        self.conv2 = LogicTreeConv2d(LogicTreeConvConfig(
            in_channels=in_ch, out_channels=4*k, kernel_size=cfg.conv_kernel, tree_depth=cfg.tree_depth,
            stride=1, padding=cfg.conv_padding, in_channels_per_tree=cfg.in_channels_per_tree,
            residual_init=cfg.residual_init, residual_gate_id=3, residual_z=cfg.residual_z,
            seed=None if cfg.seed is None else int(cfg.seed) + 2,
        ))
        self.pool2 = OrPool2d(2, 2)

        in_ch = 4*k
        self.conv3 = LogicTreeConv2d(LogicTreeConvConfig(
            in_channels=in_ch, out_channels=16*k, kernel_size=cfg.conv_kernel, tree_depth=cfg.tree_depth,
            stride=1, padding=cfg.conv_padding, in_channels_per_tree=cfg.in_channels_per_tree,
            residual_init=cfg.residual_init, residual_gate_id=3, residual_z=cfg.residual_z,
            seed=None if cfg.seed is None else int(cfg.seed) + 3,
        ))
        self.pool3 = OrPool2d(2, 2)

        in_ch = 16*k
        self.conv4 = LogicTreeConv2d(LogicTreeConvConfig(
            in_channels=in_ch, out_channels=32*k, kernel_size=cfg.conv_kernel, tree_depth=cfg.tree_depth,
            stride=1, padding=cfg.conv_padding, in_channels_per_tree=cfg.in_channels_per_tree,
            residual_init=cfg.residual_init, residual_gate_id=3, residual_z=cfg.residual_z,
            seed=None if cfg.seed is None else int(cfg.seed) + 4,
        ))
        self.pool4 = OrPool2d(2, 2)

        # Head: 32k x 2 x 2 = 128k features
        self.flatten_dim = 32 * k * 2 * 2  # 128k

        self.logic1 = DifferentiableLogicLayer(self.flatten_dim, 1280 * k, residual_init=cfg.residual_init, residual_z=cfg.residual_z, seed=None if cfg.seed is None else int(cfg.seed) + 10)
        self.logic2 = DifferentiableLogicLayer(1280 * k, 640 * k, residual_init=cfg.residual_init, residual_z=cfg.residual_z, seed=None if cfg.seed is None else int(cfg.seed) + 11)

        self.logic3 = DifferentiableLogicLayer(640 * k, 320 * k, residual_init=cfg.residual_init, residual_z=cfg.residual_z, seed=None if cfg.seed is None else int(cfg.seed) + 12)

        self.group_sum = GroupSum(cfg.num_classes, tau=cfg.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize input into multiple channels.
        x = binarize_with_thresholds(x, self.input_thresholds)

        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))

        x = x.reshape(x.shape[0], -1)  # flatten

        x = self.logic1(x)
        x = self.logic2(x)
        x = self.logic3(x)

        logits = self.group_sum(x)
        return logits


@dataclass(frozen=True)
class LogicTreeNetMNISTConfig:
    k: int = 16
    tree_depth: int = 3
    # MNIST is already grayscale; we binarize at 0.5 by default.
    input_thresholds: Tuple[float, ...] = (0.5,)
    tau: float = 6.5
    residual_init: bool = True
    residual_z: float = 5.0
    seed: Optional[int] = None
    num_classes: int = 10
    outputs_per_class: int = 1000


class LogicTreeNetMNIST(nn.Module):
    """
    Reference PyTorch implementation of the MNIST LogicTreeNet from the paper Appendix A.1.2:

      - Conv tree: k kernels, receptive field 5x5, tree depth d=3, *no padding*
      - OR-pool 2x2 stride 2 -> k x 12 x 12
      - Conv tree: 3k kernels, 3x3, padding=1
      - OR-pool -> 3k x 6 x 6
      - Conv tree: 9k kernels, 3x3, padding=1
      - OR-pool -> 9k x 3 x 3
      - Flatten -> 81k
      - Logic layers -> GroupSum
    """

    def __init__(self, cfg: LogicTreeNetMNISTConfig):
        super().__init__()
        self.cfg = cfg
        self.input_thresholds = cfg.input_thresholds
        in_ch = 1 * len(cfg.input_thresholds)

        k = int(cfg.k)

        self.conv1 = LogicTreeConv2d(LogicTreeConvConfig(
            in_channels=in_ch, out_channels=k, kernel_size=(5, 5), tree_depth=cfg.tree_depth,
            stride=1, padding=0, in_channels_per_tree=2,
            residual_init=cfg.residual_init, residual_gate_id=3, residual_z=cfg.residual_z,
            seed=None if cfg.seed is None else int(cfg.seed) + 1,
        ))
        self.pool1 = OrPool2d(2, 2)

        self.conv2 = LogicTreeConv2d(LogicTreeConvConfig(
            in_channels=k, out_channels=3*k, kernel_size=(3, 3), tree_depth=cfg.tree_depth,
            stride=1, padding=1, in_channels_per_tree=2,
            residual_init=cfg.residual_init, residual_gate_id=3, residual_z=cfg.residual_z,
            seed=None if cfg.seed is None else int(cfg.seed) + 2,
        ))
        self.pool2 = OrPool2d(2, 2)

        self.conv3 = LogicTreeConv2d(LogicTreeConvConfig(
            in_channels=3*k, out_channels=9*k, kernel_size=(3, 3), tree_depth=cfg.tree_depth,
            stride=1, padding=1, in_channels_per_tree=2,
            residual_init=cfg.residual_init, residual_gate_id=3, residual_z=cfg.residual_z,
            seed=None if cfg.seed is None else int(cfg.seed) + 3,
        ))
        self.pool3 = OrPool2d(2, 2)

        self.flatten_dim = 9 * k * 3 * 3  # 81k

        self.logic1 = DifferentiableLogicLayer(self.flatten_dim, 1280 * k, residual_init=cfg.residual_init, residual_z=cfg.residual_z, seed=None if cfg.seed is None else int(cfg.seed) + 10)
        self.logic2 = DifferentiableLogicLayer(1280 * k, 640 * k, residual_init=cfg.residual_init, residual_z=cfg.residual_z, seed=None if cfg.seed is None else int(cfg.seed) + 11)

        # out_dim = cfg.outputs_per_class * cfg.num_classes
        self.logic3 = DifferentiableLogicLayer(640 * k, 320 * k, residual_init=cfg.residual_init, residual_z=cfg.residual_z, seed=None if cfg.seed is None else int(cfg.seed) + 12)

        self.group_sum = GroupSum(cfg.num_classes, tau=cfg.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,1,28,28)

        Returns:
            logits: (B, num_classes)
        """
        x = binarize_with_thresholds(x, self.input_thresholds)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.logic1(x)
        x = self.logic2(x)
        x = self.logic3(x)
        return self.group_sum(x)