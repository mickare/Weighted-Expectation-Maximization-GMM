#!/usr/bin/env python3
from typing import List, Callable, Tuple

import numpy as np

from em.bound import Bound
from em.em import FloatArray
from em.em_opencv import OpenCvEM
from em.em_weighted import WeightedEM
from em.gmm import NormalDist, GMM
from main import sample_rv
from mathlib.vector import Vec2fArray


def compare_main(runs: int = 500):
    n_cluster = 2
    sample_count = 20
    max_iter = 5

    bound = Bound()
    xs, ys = bound.meshgrid()
    pos = np.array([xs.flatten(), ys.flatten()]).T

    cv_em = OpenCvEM(n_cluster, max_iter)
    wh_em = WeightedEM(n_cluster, max_iter)
    ems_compute: List[Tuple[str, Callable[[Vec2fArray, FloatArray], GMM]]] = [
        ('CV-EM', cv_em.compute),
        ('WH-EM', wh_em.compute),
        ('WH-EM-sqrt', lambda d, p: wh_em.compute(d, np.sqrt(p)))
    ]

    print(f"Comparing {runs} random...")

    gmms_initial = []
    gmms: List[List[GMM]] = []

    for run in range(runs):
        prog_perc = run / runs * 100
        if prog_perc % 5 == 0:
            print(f"{prog_perc:0.0f}%", end=', ', flush=True)

        n_cluster = 2
        weights_random = np.random.random(n_cluster) + 0.2
        gmm_initial = GMM(
            normals=[NormalDist.random_variance(bound.random(), bound) for n in range(n_cluster)],
            weights=weights_random / np.sum(weights_random)
        )
        gmms_initial.append(gmm_initial)
        sample_pos, sample_pdf = sample_rv(gmm_initial, sample_count, bound)
        gmms.append([em_comp(sample_pos, sample_pdf) for name, em_comp in ems_compute])
    print("100%")

    def rel_entr(x, y):
        xx = np.array(x)
        yy = np.array(y)
        assert xx.shape == yy.shape
        yy[y <= 0] = 1
        xx[x <= 0] = 1
        result = xx * np.log(xx / yy)
        result[np.logical_and(x == 0, y >= 0)] = 0
        result[np.logical_or(x < 0, y < 0)] = np.inf
        return np.sum(result)

    print("Computing pdfs...")
    i_pdfs = [g.pdf(pos, allow_singular=True) for g in gmms_initial]
    g_pdfs = [[g.pdf(pos, allow_singular=True) for g in gs] for gs in gmms]

    print("Computing KL-Divergence...")
    kls = [
        [rel_entr(ipdf, gpdf) for gpdf in gs]
        for ipdf, gs in zip(i_pdfs, g_pdfs)
        if np.all(ipdf > 0) and np.all(np.array(gs) > 0)
    ]

    mean_kls = np.mean(kls, axis=1)

    for (name, f), mkl in zip(ems_compute, mean_kls):
        print(f"Mean Relative Entropy D(P|{name}): ", mkl)


if __name__ == '__main__':
    compare_main()
