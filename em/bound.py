from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Bound:
    x: float = 3
    y: float = 3

    def meshgrid(self, nx=500, ny=500) -> Tuple[np.ndarray, np.ndarray]:
        # noinspection PyTypeChecker
        return np.meshgrid(np.linspace(-self.x, self.x, nx),
                           np.linspace(-self.y, self.y, ny))

    def random(self, count=1):
        size = 2
        if count > 1:
            size = (count, 2)
        return (np.random.random(size) * 2 - 1.0) * (self.x, self.y)

    def get_mesh_coord(self, pos: np.ndarray, nx=500, ny=500):
        return (pos / (self.x, self.y) + 1) / 2 * (nx, ny)
