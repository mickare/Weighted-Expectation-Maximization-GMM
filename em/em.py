from abc import abstractmethod, ABC
from typing import Union, Sequence

import numpy as np

from mathlib.vector import Vec2fArray
from em.gmm import GMM

FloatArray = Union[Sequence[float], np.ndarray]


class EM(ABC):
    @abstractmethod
    def compute(self, points: Vec2fArray, prob: FloatArray) -> GMM:
        ...
