# -*- coding: utf-8 -*-

from typing import Tuple, Union, Sequence

import numpy as np

# Types
Number = Union[float, int]
Vec2f = Union[Tuple[Number, Number], np.ndarray, Sequence[Number]]

Points = Union[Vec2f, Sequence[Vec2f]]
Vec2fArray = Union[Sequence[Vec2f], np.ndarray]


def rotate_points(pts: Points, radians) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    return np.dot(pts, [(c, s), (-s, c)])


def rotate_points_degree(pts: Points, degree) -> np.ndarray:
    return rotate_points(pts, np.radians(degree))
