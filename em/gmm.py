from typing import Union, Sequence, Optional, List, Dict

import numpy as np
from matplotlib import pyplot
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen

from em.bound import Bound
from mathlib.vector import Vec2f


class NormalDist:
    @classmethod
    def random_variance(cls, mean: Vec2f, bound: Bound):
        variance: np.ndarray = (np.random.random((2, 2)) - 0.5) * 2 * (bound.x, bound.y)
        return cls(mean, np.dot(variance, variance.transpose()) / 2)

    def __init__(self, mean: Vec2f, cov: Union[Sequence[Sequence[float]], np.ndarray]):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        assert self.cov.shape == (2, 2)
        # assert self.is_positive_definite(self.cov)

    @classmethod
    def is_positive_definite(cls, arr):
        return np.all(np.linalg.eigvals(arr) > 0)

    def multi_normal(self) -> multivariate_normal_frozen:
        return multivariate_normal(mean=self.mean, cov=self.cov)

    def pdf(self, x, **kwargs):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov, **kwargs)

    def rv(self, size: int = 1):
        return multivariate_normal.rvs(mean=self.mean, cov=self.cov, size=size)

    @classmethod
    def load(cls, data: Dict):
        return cls(mean=np.array(data["mean"]),
                   cov=np.array(data["cov"]))

    def store(self) -> Dict:
        return {'mean': np.array(self.mean).tolist(),
                'cov': np.array(self.cov).tolist()}


def kullback_leibler_divergence(norm0: NormalDist, norm1: NormalDist):
    c0 = norm0.cov
    c1 = norm1.cov
    c1_inv = np.linalg.inv(c1)
    dm = norm1.mean - norm0.mean
    dmT = dm.reshape((-1, 1))
    n = len(dm)
    return 0.5 * (
            np.trace(c1_inv * c0) + dmT * c1_inv * dm - n + np.log(np.linalg.det(c1) / np.linalg.det(c0))
    )


class GMM:
    def __init__(self, normals: Sequence[NormalDist], weights: Optional[Sequence[float]] = None):
        assert normals
        if weights is None:
            weights = np.ones(len(normals)) / len(normals)
        assert len(weights) == len(normals)
        assert abs(1 - np.sum(weights)) < 1e-10
        self.normals: List[NormalDist] = list(normals)
        self.weights: List[float] = list(weights)

    @classmethod
    def load(cls, data: Dict):
        return cls(
            normals=[NormalDist.load(d) for d in data.get('normals')],
            weights=np.array(data.get('weights')) if 'weights' in data else None
        )

    def store(self) -> Dict:
        return {'normals': [n.store() for n in self.normals],
                'weights': np.array(self.weights).tolist()}

    def pdf(self, x, **kwargs):
        return np.sum([norm.pdf(x, **kwargs) * w for norm, w in zip(self.normals, self.weights)], axis=0)

    def _sample_ids(self, size):
        return np.random.choice(len(self.weights), size, replace=True, p=self.weights)

    def rv(self, size: int = 1):
        return np.array([self.normals[i].rv(1) for i in self._sample_ids(size)])

    def means(self) -> List[Vec2f]:
        return [n.mean for n in self.normals]

    def plot_pdf_2d(self, axs: Sequence[pyplot.Axes], xs, size=(500, 500), title: Optional[str] = None,
                    plot_kwargs: Optional[Dict] = None):
        count = len(self.normals)
        assert len(axs) >= count + 1
        plot_kwargs = plot_kwargs or dict()
        plot_kwargs.setdefault('cmap', 'viridis')
        results = []
        for n, (normal, ax) in enumerate(zip(self.normals, axs)):
            if title:
                ax.set_title(f"{title} - {n}")
            p = ax.imshow(normal.pdf(xs, allow_singular=True).reshape(*size), **plot_kwargs)
            results.append(p)

        if title:
            axs[count].set_title(f"{title} - GMM")
        p = axs[count].imshow(self.pdf(xs, allow_singular=True).reshape(*size), **plot_kwargs)
        results.append(p)
        return results
