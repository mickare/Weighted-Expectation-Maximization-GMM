# Weighted Expectation-Maximization for sparse GMM Training

Proof of concept EM-Algorithm implementation that uses
prior knowledge of probabilities on 2D points to train
a multivariate Gaussian Mixture Model (GMM).


## TL;DR
Basically the probability is used for normalization
during the maximization step. 
When the sampling count is low, the square root 
of the probability p can be used instead of p
as an optimization.

![sigma](img/img_phi.png)

![sigma](img/img_mean.png)

![sigma](img/img_cov.png)

The expectation step is not changed:

![sigma](img/img_gamma.png)

## Summary
- KMeans with random point initialization
- Low Max-Iterations (default: 5)
- Low count of training points (default: 20).
- Comparison and reference of EM implementation in OpenCV

## Usage

- Run:    `./main.py` 
- Store:  `./main.py --save test.json`
- Replay: `./main.py --load test.json`


- Run large test: `./compare.py`

### What can be observed?
More information is used to approximate the incomplete data.
It shows slightly better results than the reference algorithm,
especially in a sparse sampled environment.

But keep in mind that with a low iteration count
the initial guess via K-Means plays a big role.

#### Example 1
`Initial` is the desired distribution that was used to sample the red dots.
`OpenCV-EM` is the reference algorithm by OpenCV.
`Weighted-EM` is the enhancement by using the probabilities in normalization.
![sigma](img/plot_example.1.png)

#### Example 2
![sigma](img/plot_example.2.png)

## Optimizations
On low sampling rate some normals can become faint. 
The square-root on the probability can bring them to the front again.

![sigma](img/plot_sqrt_effect.png)