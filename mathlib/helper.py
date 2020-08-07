"""
Library module for helper methods for vector / matrix operations

@author: Michal Käser <michael.kaeser@fzi.de>
"""
__version__ = '0.1'
__author__ = 'Michael Käser'

import math
from typing import Union, Iterable, Tuple, List, TypeVar, Sequence

import numpy as np

from mathlib.vector import Vec2f

T = TypeVar("T")


def assert_vector(vec: np.ndarray):
    assert vec.shape == (2,), "wrong vector shape: expected (2,), actual %s" % vec.shape
    return vec


def assert_vector_list(pts: np.ndarray, name):
    assert pts.shape[0] == 2, "%s wrong shape: expected (2, N), actual %s" % (name, pts.shape)
    return pts


def rotate_vector(vectors: np.ndarray, radians) -> np.ndarray:
    """
    Rotates vector counter-clockwise
    :param vectors: vector with the shape (N, 2)
    :param radians: rotation in radians
    :return: List of vectors rotated
    """
    assert_vector_list(vectors, "vectors")
    c, s = np.cos(radians), np.sin(radians)
    return np.dot([(c, s), (-s, c)], vectors)


def interspere_separator(data: Iterable[T], separator: T) -> Iterable[T]:
    first = True
    for d in data:
        if first:
            first = False
        else:
            yield separator
        yield d


class VectorHelper:
    @classmethod
    def join_vectors(cls, vectors: List[np.ndarray]):
        return np.concatenate(vectors, axis=0)


class MatrixMath:

    @classmethod
    def new_rotation(self, rotation: float = 0.0):
        c, s = np.cos(rotation), np.sin(rotation)
        return np.array([(c, -s, 0), (s, c, 0), (0, 0, 1)])

    @classmethod
    def new_translation(cls, translation=(0, 0)):
        return np.array([(1, 0, translation[0]), (0, 1, translation[1]), (0, 0, 1)])

    @classmethod
    def new_transform(cls, rotation: float = 0.0, scale: float = 1.0, translation: Vec2f = (0, 0)) -> np.ndarray:
        """
        Creates a transformation matrix
        :param rotation: radians counter-clockwise (default: 0)
        :param scale: scaling factor (default: 1)
        :param translation: offset movement (default (0,0)
        :return: Transformation matrix
        """
        c = np.cos(rotation)
        s = np.sin(rotation)
        return np.array([(c, -s, translation[0]), (s, c, translation[1]), (0, 0, 1)]) * scale

    @classmethod
    def make_quaternions(cls, vectors) -> np.ndarray:
        assert vectors.ndim == 2 and vectors.shape[1] == 2
        return np.pad(vectors, ((0, 0), (0, 1)), constant_values=((1, 1), (1, 1)))

    @classmethod
    def transform(cls, vectors: np.ndarray, transf: np.ndarray) -> np.ndarray:
        return cls.transform_quat(cls.make_quaternions(vectors), transf)[:, :2]

    @classmethod
    def transform_quat(cls, quaternions: np.ndarray, transf: np.ndarray) -> np.ndarray:
        assert transf.shape == (3, 3)
        result = transf.dot(quaternions.T).T
        return result


class MathHelper:
    @staticmethod
    def mitternacht(a: float, b: float, c: float) -> Tuple[float, ...]:
        if a == 0:
            if b != 0:
                return (-c / b),
            else:
                return ()

        d = (b ** 2) - (4 * a * c)
        if d > 0:
            root = math.sqrt(d)
            return (-b - root) / 2 / a, (-b + root) / 2 / a
        elif d == 0:
            return -b / 2 / a,
        else:
            return ()