import math
from enum import Enum
from typing import TypeVar, Generic, List

import cv2
import cv2.ml as cml
import numpy as np
from matplotlib import pyplot

from mathlib.vector import Vec2fArray
from em.bound import Bound
from em.em import EM, FloatArray
from em.gmm import GMM, NormalDist


class OpenCvEM(EM):
    def __init__(self, ncluster: int, max_iter: int, epsilon: float = 0.0):
        self.ncluster = ncluster
        self.max_iter = max_iter
        self.epsilon = epsilon

    def compute(self, points: Vec2fArray, prob: FloatArray) -> GMM:
        samples = np.array(points, dtype=np.float32)
        probs = np.array([prob] * self.ncluster, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.epsilon)
        em: cv2.ml_EM = cv2.ml.EM_create()
        em.setClustersNumber(self.ncluster)
        em.setTermCriteria(criteria)
        em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)
        em.trainEM(samples, probs=probs)

        return GMM(
            normals=[NormalDist(mean=mean, cov=cov) for mean, cov in zip(em.getMeans(), em.getCovs())],
            weights=em.getWeights().flatten()
        )


def main():
    bound = Bound()
    xs, ys = bound.meshgrid()
    pos = np.array([xs.flatten(), ys.flatten()], dtype=np.float32).T

    number = 2
    weights_random = np.random.random(number) + 0.2
    initial = GMM(
        normals=[NormalDist.random_variance(bound.random(), bound) for n in range(number)],
        weights=weights_random / np.sum(weights_random)
    )

    points_ids = np.random.choice(pos.shape[0], size=2000, replace=False)
    points = pos[points_ids]
    probs = initial.pdf(points)
    probs /= np.sum(probs)

    result = OpenCvEM(2, 20, replace=True, replication_factor=0.1).compute(points, probs)

    #################

    def hide_axis_label(ax: pyplot.Axes):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    fig: pyplot.Figure = pyplot.figure(figsize=((number + 1) * 4, 8))
    axs: List[List[pyplot.Axes]] = fig.subplots(2, number + 1, squeeze=False)

    for ax in [a for axx in axs for a in axx]:
        hide_axis_label(ax)

    for n, ax in enumerate(axs[0][:-1]):
        ax.set_title(f"Initial {n}")
        ax.imshow(initial.normals[n].pdf(pos).reshape(500, 500), cmap='viridis')

    ax = axs[0][-1]
    ax.set_title(f"Initial GMM")
    ax.imshow(initial.pdf(pos).reshape(500, 500), cmap='viridis')

    for n, ax in enumerate(axs[1][:-1]):
        ax.set_title(f"Result {n}")
        ax.imshow(result.normals[n].pdf(pos).reshape(500, 500), cmap='viridis')
    ax = axs[1][-1]
    ax.set_title(f"Result GMM")
    ax.imshow(result.pdf(pos).reshape(500, 500), cmap='viridis')

    fig.show()


if __name__ == '__main__':
    main()
