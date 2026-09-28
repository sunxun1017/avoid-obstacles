from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class ProjectionResult:
    elevation: NDArray[np.float32]
    distance: NDArray[np.float32]

class DualProjector:
    def __init__(self,
                 num_envs: int,
                 elevation_shape: tuple[int, int] = (21, 17),
                 distance_shape: tuple[int, int] = (13, 17)): # 论文的默认值
        self.elevation_map.shape = (num_envs, *elevation_shape)
        self.distance_map.shape  = (num_envs, *distance_shape)

        # 送进论文的MLP之前需要展平
        self.elevation_flat = self.elevation_map.flatten(start_dim=1)
        self.distance_flat  = self.distance_map.flatten(start_dim=1)
