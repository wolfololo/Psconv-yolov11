"""Feature Enhancement Module (FEM) shown in the supplied architecture figure.

The module implements the four branches in Eqs. (12)--(15): a 3x3 branch,
two asymmetric-convolution branches with dilation 5, and a 1x1 residual
branch.  It preserves spatial resolution and emits ``c2`` channels.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvBNAct(nn.Sequential):
    """Conv-BN-SiLU with tuple-kernel support for 3x1 and 1x3 convolutions."""

    def __init__(self, c1: int, c2: int, k=3, d=1):
        if isinstance(k, int):
            k = (k, k)
        if isinstance(d, int):
            d = (d, d)
        p = (d[0] * (k[0] - 1) // 2, d[1] * (k[1] - 1) // 2)
        super().__init__(
            nn.Conv2d(c1, c2, k, stride=1, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class FEM(nn.Module):
    """Four-branch FEM: standard, asymmetric+dilated, asymmetric+dilated, residual.

    Args:
        c1: Input channels (injected by Ultralytics ``parse_model``).
        c2: Output channels.
        reduction: Kept for YAML/API compatibility; branch widths are derived
            from ``c2`` so that concatenation can be added to the residual.

    Input and output tensors have the same ``H`` and ``W``.  The residual path
    is a learned 1x1 projection so it works for both ``c1 == c2`` and channel
    changes.
    """

    def __init__(self, c1: int, c2: int, reduction: int = 4):
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"FEM channels must be positive, got c1={c1}, c2={c2}")
        del reduction  # The reference equations do not use a bottleneck ratio.
        widths = (c2 // 3, c2 // 3, c2 - 2 * (c2 // 3))

        # W1 = conv3x3(conv1x1(F))
        self.w1 = nn.Sequential(_ConvBNAct(c1, widths[0], 1), _ConvBNAct(widths[0], widths[0], 3))

        # W2 = dilated3x3(conv1x3(conv3x1(conv1x1(F))))
        self.w2 = nn.Sequential(
            _ConvBNAct(c1, widths[1], 1),
            _ConvBNAct(widths[1], widths[1], (3, 1)),
            _ConvBNAct(widths[1], widths[1], (1, 3)),
            _ConvBNAct(widths[1], widths[1], 3, d=5),
        )

        # W3 = dilated3x3(conv1x3(conv3x1(dilated3x3(conv1x1(F)))))
        self.w3 = nn.Sequential(
            _ConvBNAct(c1, widths[2], 1),
            _ConvBNAct(widths[2], widths[2], 3, d=5),
            _ConvBNAct(widths[2], widths[2], (3, 1)),
            _ConvBNAct(widths[2], widths[2], (1, 3)),
            _ConvBNAct(widths[2], widths[2], 3, d=5),
        )

        # Residual f1x1(F), then Cat(W1,W2,W3) ⊕ residual.
        self.residual = _ConvBNAct(c1, c2, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat((self.w1(x), self.w2(x), self.w3(x)), dim=1) + self.residual(x))


__all__ = ["FEM"]

