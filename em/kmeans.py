from abc import ABC, abstractmethod
from typing import Iterator, Union, Sequence, Optional, Callable

import numpy as np

from em.em import FloatArray
from mathlib.vector import Vec2f, Vec2fArray

ClusterIdArray = Union[Sequence[int], np.ndarray]


class KMeansInit(ABC):
    @abstractmethod
    def __call__(self, nCluster: int, data: Vec2fArray, weights: FloatArray):
        ...


KMeansInitFunc = Callable[[int, Vec2fArray, FloatArray], Vec2fArray]


class KMeans:
    def __init__(self, nCluster: int, initial: Optional[KMeansInitFunc] = None):
        assert nCluster > 0
        self.nCluster = nCluster
        self.initializer: KMeansInitFunc = initial or RandomKMeansInit()

    @classmethod
    def mean_weighted(cls, data: Vec2fArray, weights: FloatArray, cluster: ClusterIdArray, nCluster) -> Iterator[Vec2f]:
        for c in range(0, nCluster):
            sel = cluster == c
            w = weights[sel]
            yield np.sum((data[sel].T * w).T, axis=0) / np.sum(w)

    @classmethod
    def _assign_cluster(cls, data: Vec2fArray, weights: FloatArray, centroids: Vec2fArray) -> ClusterIdArray:
        sqrd_dist = [np.sum(weights * np.square(data - c).T, axis=0) for c in centroids]
        return np.argmin(sqrd_dist, axis=0)

    def __call__(self, data: Vec2fArray, weights: FloatArray, max_steps: int = 10, epsilon: float = 1e-5):
        assert max_steps > 0
        assert len(data) > self.nCluster

        # Take n+1 initial centroids
        # +1, so we can break any equilibrium that may occur
        centroids = self.initializer(self.nCluster + 1, data, weights)
        assert len(centroids) == self.nCluster + 1

        # Cluster n+1
        cluster = self._assign_cluster(data, weights, centroids)
        # Remove the additional one
        centroids = np.array(list(self.mean_weighted(data, weights, cluster, self.nCluster + 1)))[:-1]
        assert len(centroids) == self.nCluster

        # Keep going with n clusters
        cluster = self._assign_cluster(data, weights, centroids)
        for s in range(0, max_steps):
            centroids_old = centroids
            centroids = np.array(list(self.mean_weighted(data, weights, cluster, self.nCluster)))
            cluster = self._assign_cluster(data, weights, centroids)
            if np.linalg.norm(centroids - centroids_old) < epsilon:
                break
        return centroids, cluster


class RandomKMeansInit(KMeansInit):
    def __call__(self, nCluster: int, data: Vec2fArray, weights: FloatArray):
        probs = np.array(weights) / np.sum(weights)
        index = np.random.choice(len(data), nCluster, replace=False, p=probs)
        return data[index]


class DividingMeanKMeansInit(KMeansInit):
    def __call__(self, nCluster: int, data: Vec2fArray, weights: FloatArray):
        return list(self._initializer_mean_divide(nCluster, data, axis=self._calc_max_axis(data)))

    @classmethod
    def _calc_max_axis(cls, data: Vec2fArray) -> int:
        diff = np.max(data, axis=0) - np.min(data, axis=0)
        return int(diff[0] < diff[1])

    @classmethod
    def _initializer_mean_divide(cls, k: int, data: Vec2fArray, axis: int = 0) -> Iterator[Vec2f]:
        assert len(data) > k
        mean = np.mean(data, axis=0)
        if k > 1:
            left_sel = data[:, axis] <= mean[axis]
            right_sel = data[:, axis] >= mean[axis]
            left_data = data[left_sel]
            right_data = data[right_sel]
            left_k = int(np.ceil(k * len(left_data) / len(data)))
            right_k = k - left_k
            nextaxis = (axis + 1) % 2
            yield from cls._initializer_mean_divide(left_k, left_data, axis=nextaxis)
            yield from cls._initializer_mean_divide(right_k, right_data, axis=nextaxis)
        elif k == 1:
            yield mean


class SplitKMeansInit(KMeansInit):
    def __call__(self, nCluster: int, data: Vec2fArray, weights: FloatArray):
        return list(self._initializer_split_divide(nCluster, data, axis=self._calc_max_axis(data)))

    @classmethod
    def _calc_max_axis(cls, data: Vec2fArray) -> int:
        diff = np.max(data, axis=0) - np.min(data, axis=0)
        return int(diff[0] < diff[1])

    @classmethod
    def _initializer_split_divide(cls, k: int, data: Vec2fArray, axis: int = 0) -> Iterator[Vec2f]:
        if k == 1:
            yield np.mean(data, axis=0)
        elif k > 0:
            sdata = sorted(data, key=lambda e: e[axis])
            half = len(data) // 2
            left = sdata[:half]
            right = sdata[half:]
            nextaxis = (axis + 1) % 2
            k2 = k // 2
            yield from cls._initializer_split_divide(k2, left, axis=nextaxis)
            yield from cls._initializer_split_divide(k - k2, right, axis=nextaxis)
