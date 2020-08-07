#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from typing import List, Optional, Dict

import numpy as np
from matplotlib import pyplot
from scipy.special import rel_entr

from mathlib.vector import Vec2f, Vec2fArray
from em.bound import Bound
from em.em_weighted import WeightedEM
from em.em_opencv import OpenCvEM
from em.gmm import NormalDist, GMM


def sample_random(gmm: GMM, count: int, bound: Bound):
    pos = bound.random(count)
    print(pos.shape)
    pdf = gmm.pdf(pos)
    return pos, pdf


def sample_rv(gmm: GMM, count: int, bound: Bound):
    xs, ys = bound.meshgrid()
    pos = np.array([xs.flatten(), ys.flatten()]).T
    pdf = gmm.pdf(pos)
    index = np.random.choice(len(pos), count, replace=False, p=pdf / np.sum(pdf))
    return pos[index], pdf[index]


def plot_samples(ax: pyplot.Axes, gmm: GMM, size: Vec2f,
                 pos: Vec2fArray, sample_pos: Vec2fArray, bound: Bound,
                 plot_args: Optional[Dict] = None):
    ax.imshow(gmm.pdf(pos, allow_singular=True).reshape(*size), cmap='viridis')
    plot_args = plot_args or dict()
    plot_args.setdefault('marker', '.')
    plot_args.setdefault('color', 'r')
    ax.plot(*bound.get_mesh_coord(sample_pos).T, '.', **plot_args)
    ax.set_xlim(0, 500)
    ax.set_ylim(500, 0)


def compare_main():
    default_n_cluster = 2
    default_sample_count = 20
    default_max_iter = 5

    parser = ArgumentParser()
    parser.add_argument("-n", "--ncluster", dest="ncluster", type=int, default=default_n_cluster,
                        help="Number of clusters / gmms to detect.")
    parser.add_argument("-s", "--samples", dest="samples", type=int, default=default_sample_count,
                        help="Number of samples to train the GMM with EM")
    parser.add_argument("-I", "--maxiter", dest="maxiter", type=int, default=default_max_iter,
                        help="Number of maximum iterations of EM steps")
    parser.add_argument("--save", type=str, help="Save the initial gmm to file")
    parser.add_argument("--load", type=str, help="Load the initial gmm from file")
    args = parser.parse_args()

    n_cluster = args.ncluster
    sample_count = args.samples
    max_iter = args.maxiter

    bound = Bound()
    xs, ys = bound.meshgrid()
    pos = np.array([xs.flatten(), ys.flatten()]).T

    gmm_initial: Optional[GMM] = None
    if args.load:
        print(f"Loading Initial-GMM from '{args.load}'")
        with open(args.load, 'rt', encoding='utf-8')as fp:
            gmm_initial = GMM.load(json.load(fp))

    else:
        weights_random = np.random.random(n_cluster) + 0.2
        gmm_initial = GMM(
            normals=[NormalDist.random_variance(bound.random(), bound) for n in range(n_cluster)],
            weights=weights_random / np.sum(weights_random)
        )
        if args.save:
            print(f"Storing Initial-GMM to '{args.save}'")
            with open(args.save, 'wt', encoding='utf-8') as fp:
                json.dump(gmm_initial.store(), fp)

    # Assert that there is a initial GMM that can provide sample data
    assert gmm_initial
    n_cluster = len(gmm_initial.normals)

    print(f"Training OpenCV-EM & Weighted-EM with configuration:\n"
          f"\tn_cluster: {n_cluster}\n"
          f"\tsamples:  {sample_count}\n"
          f"\tmax_iter: {max_iter}")

    # sample_pos, sample_pdf = sample_random(gmm_initial, sample_count, bound)
    sample_pos, sample_pdf = sample_rv(gmm_initial, sample_count, bound)

    cv_em = OpenCvEM(n_cluster, max_iter)
    wh_em = WeightedEM(n_cluster, max_iter)
    gmms = [
        ('CV-EM', cv_em.compute(sample_pos, sample_pdf)),
        ('WH-EM', wh_em.compute(sample_pos, sample_pdf)),
        ('WH-EM-sqrt', wh_em.compute(sample_pos, np.sqrt(sample_pdf)))
    ]

    def relative_entropy():
        points_count = len(pos) // 10
        points = gmm_initial.rv(points_count)
        ref_pdf = gmm_initial.pdf(points, allow_singular=True)
        for name, gmm in gmms:
            gmm_pdf = gmm.pdf(points, allow_singular=True)
            print(f"Relative Entropy D(P|{name}): ", np.sum(rel_entr(ref_pdf, gmm_pdf)))

    def mean_sqrd_error():
        points_count = len(pos) // 10
        points = gmm_initial.rv(points_count)
        ref_pdf = gmm_initial.pdf(points, allow_singular=True)
        for name, gmm in gmms:
            gmm_pdf = gmm.pdf(points, allow_singular=True)
            print(f"MSE {name}: ", np.mean((ref_pdf - gmm_pdf) ** 2))

    relative_entropy()
    mean_sqrd_error()

    #################

    def hide_axis_label(ax: pyplot.Axes):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    fig_size = 4
    rows = 1 + len(gmms)
    cols = n_cluster + 2
    fig: pyplot.Figure = pyplot.figure(figsize=(cols * fig_size, rows * fig_size))
    axs: List[List[pyplot.Axes]] = fig.subplots(rows, cols, squeeze=False)

    for ax in [a for axx in axs for a in axx]:
        hide_axis_label(ax)

    size = (500, 500)

    sample_plot_args = {
        # 'markersize': 1
    }

    gmm_initial.plot_pdf_2d(axs[0], pos, size=size, title="Initial")
    plot_samples(axs[0][-1], gmm_initial, size, pos, sample_pos, bound, sample_plot_args)
    axs[0][-1].set_title("Samples (red)")

    for axrow, (name, gmm) in zip(axs[1:], gmms):
        gmm.plot_pdf_2d(axrow, pos, size=size, title=name)
        plot_samples(axrow[-1], gmm, size, pos, sample_pos, bound, sample_plot_args)
        axrow[-1].set_title("Samples (red)")

    pyplot.show()


if __name__ == '__main__':
    compare_main()
