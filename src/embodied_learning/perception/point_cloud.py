"""Fixed-size batched point clouds with explicit coordinate frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class PointCloud:
    """Points [B, N, 3] in meters and a boolean valid mask [B, N].

    Invalid points contain zeros and must be excluded using valid_mask.
    Optical coordinates have X right, Y down, and Z forward.
    Frozen fields do not make the underlying tensors immutable.
    """

    points: Tensor
    valid_mask: Tensor
    frame: Literal["optical", "world", "yaw_local"]
