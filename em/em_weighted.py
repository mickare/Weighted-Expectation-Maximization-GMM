from typing import List, Optional

import numpy as np

from em.em import EM, FloatArray
from em.gmm import GMM, NormalDist
from em.kmeans import KMeans
from mathlib.vector import Vec2fArray


class WeightedEM(EM):
    def __init__(self, ncluster: int, max_iter: int, epsilon: float = 1.0):
        self.ncluster = ncluster
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute(self, points: Vec2fArray, prob: FloatArray) -> GMM:
        trainer = WeightedEmTrainer(self.ncluster, points, prob)
        for it in range(0, self.max_iter):
            trainer.step()

        return trainer.gmm()


class WeightedEmTrainer:
    @classmethod
    def _calc_covariance(cls, data, mean, prob):
        dmean = data - mean
        return np.sum([(np.outer(x, x) * p) for x, p in zip(dmean, prob)], axis=0) / np.sum(prob)

    def __init__(self, ncluster: int, points: Vec2fArray, prob: FloatArray, phis: Optional[FloatArray] = None):
        assert ncluster > 0
        assert len(points) == len(prob)
        assert len(points[prob > 0]) > ncluster
        self.points = points[prob > 0]
        self.prob = prob[prob > 0]
        # self.prob /= np.sum(self.prob)
        self.ncluster = ncluster

        # Initial configuration
        kmeans = KMeans(self.ncluster)
        self.means, cluster = kmeans(self.points, self.prob, max_steps=5, epsilon=1e-3)
        assert len(self.means) == self.ncluster

        length = len(self.points)
        self.phis = np.array(phis or [1 / ncluster] * ncluster)
        assert len(self.phis) == ncluster and np.sum(self.phis) == 1.0

        mean0 = np.sum(self.prob * self.points.T, axis=1) / np.sum(self.prob)
        self.covs = [self._calc_covariance(self.points, mean0, self.prob)] * self.ncluster
        # self.covs = [self._calc_covariance(self.points[cluster == k], mean, self.prob[cluster == k])
        #              for k, mean in enumerate(self.means)]
        assert len(self.covs) == self.ncluster

        self.steps = 0

    def normals(self) -> List[NormalDist]:
        return [NormalDist(mean=m, cov=c) for m, c in zip(self.means, self.covs)]

    def gmm(self) -> GMM:
        return GMM(self.normals())

    def _step_expectation(self) -> np.ndarray:
        """Expectation step"""
        normals: List[NormalDist] = self.normals()
        gammas = np.array([(
                phi * normal.pdf(self.points, allow_singular=True)
        ) for phi, normal in zip(self.phis, normals)])

        gammas_denom = np.sum(gammas, axis=0)
        # Prevent division by zero, set numerator to 0 and denominator to 1
        _gammas_repair = gammas_denom == 0
        gammas[:, _gammas_repair] = 0
        gammas_denom[_gammas_repair] = 1
        # Apply denominator to gammas
        gammas /= gammas_denom
        return gammas

    def _step_maximization(self, gammas: np.ndarray, rate: float = 1.0):
        assert gammas.shape[1] == len(self.points)

        """Maximization step"""
        gamma_probs = gammas * self.prob
        gamma_probs_sum = np.sum(gamma_probs, axis=1)

        phis = gamma_probs_sum / np.sum(self.prob)
        self.phis = phis / np.sum(phis)

        self.means = [np.sum(gp * self.points.T, axis=1) / gps for gp, gps in zip(gamma_probs, gamma_probs_sum)]

        self.covs = [
            (np.sum([gpx * np.outer(dx, dx) for dx, gpx in zip(self.points - mean, gp)], axis=0)
             / gp_sum)
            for mean, gp, gp_sum in zip(self.means, gamma_probs, gamma_probs_sum)
        ]

    def step(self, rate: float = 0.9):
        # Expectation step
        gammas = self._step_expectation()

        # Maximization step
        self._step_maximization(gammas, rate)

        self.steps += 1
