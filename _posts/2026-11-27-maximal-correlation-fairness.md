---
title: "Maximal correlation for fairness, seven years later"
permalink: /blog/maximal-correlation-fairness/
excerpt: "In 2019 we penalized dependence on a continuous sensitive attribute with a χ² upper bound on the HGR maximal correlation, estimated by KDE. What would a neural version look like, and which parts of it are still open?"
tags:
  - fairness
  - maximal correlation
  - research notes
---

In 2019, with Clément Calauzènes and Noureddine El Karoui, we published [Fairness-aware learning for continuous attributes and treatments](https://proceedings.mlr.press/v97/mary19a.html) at ICML. I have been thinking about what a follow-up would look like today, with neural estimators in place of kernel density estimates. A lot has happened since: maximal correlation came back through neural spectral methods, representation learning theory and fairness on embeddings. These notes go through where that work has taken the idea, and what I think is still missing.

## What the 2019 paper did

The setting is a model whose prediction should not depend on a sensitive attribute that is continuous, rather than a binary group label. To measure dependence we used the Hirschfeld-Gebelein-Rényi (HGR) maximal correlation,

$$
\operatorname{HGR}(X,Y)=\sup_{f,g}\ \operatorname{corr}\big(f(X),g(Y)\big),
$$

which is zero exactly when $$X$$ and $$Y$$ are independent. HGR has a spectral reading. Write $$P=P_{XY}$$ for the joint law, $$Q=P_X\otimes P_Y$$ for the product of the marginals, and consider the kernel

$$
K(x,y)=\frac{p_{XY}(x,y)}{p_X(x)\,p_Y(y)}-1 .
$$

Its singular values $$\rho_1\ge\rho_2\ge\dots$$ are the dependence modes of the pair. The largest one is HGR, and the sum of their squares is the $$\chi^2$$ divergence between $$P$$ and $$Q$$:

$$
\operatorname{HGR}(X,Y)=\rho_1,\qquad \chi^2(P\,\Vert\,Q)=\sum_{i\ge 1}\rho_i^2 .
$$

So $$\chi^2$$ gives an upper bound on HGR², and it is a smooth quantity you can add to a training loss. We estimated the densities with a KDE on each minibatch and used the result as a fairness penalty, with spectral guarantees behind it.

The weak point is the KDE. It works for a scalar prediction against a scalar attribute and becomes useless as the dimension grows. Today the things we would want to decorrelate are embeddings, sequences and multimodal representations, so the obvious move is to replace the density estimate with a neural estimate of the density ratio and try to keep the guarantees.

## Going neural

Three routes replace the density estimate by networks. All three have been explored since 2019, from different angles.

### Estimating the χ² variationally

The $$\chi^2$$ divergence has an exact variational form, like every $$f$$-divergence:

$$
\chi^2(P\,\Vert\,Q)=\sup_{T}\ \Big\{\,2\,\mathbb E_P[T]-\mathbb E_Q[T^2]-1\,\Big\},
$$

and the supremum is reached at the density ratio $$T^\star(x,y)=p_{XY}(x,y)/\big(p_X(x)p_Y(y)\big)$$. A network $$T_\phi(x,y)$$ can be trained on this objective. Positive pairs come from the joint distribution, which is just the batch as it is. Negative pairs, which must follow $$Q$$, come from permuting $$Y$$ inside the batch. The loss is quadratic, so there is no explicit density and no exponential term to normalize.

{% include posts/hgr-shuffle.html %}

The catch is in the direction of the inequality. With a network of finite capacity, the objective gives a lower bound on $$\chi^2$$. It no longer gives an upper bound on HGR, which was the property that made the 2019 penalty safe to use.

### Learning the dependence spectrum

The second route learns the singular functions themselves. Take two networks $$f_\theta:\mathcal X\to\mathbb R^k$$ and $$g_\phi:\mathcal Y\to\mathbb R^k$$ with centered outputs, and maximize

$$
\Big\lVert\, C_{ff}^{-1/2}\,C_{fg}\,C_{gg}^{-1/2} \Big\rVert_F^2 ,
$$

where the $$C$$ are the covariance and cross-covariance matrices of the features. At the functional optimum this equals $$\sum_{i=1}^{k}\rho_i^2$$, so the rank $$k$$ interpolates between HGR and the 2019 penalty:

| Rank | What the objective estimates |
| --- | --- |
| $$k=1$$ | HGR² |
| $$k=2$$ | the first two modes |
| finite $$k$$ | truncated spectral energy |
| $$k\to\infty$$ | the $$\chi^2$$ divergence |

This is Deep CCA read as a dependence measure. The most complete version of the idea today is [Operator SVD with Neural Networks via Nested Low-Rank Approximation](https://arxiv.org/abs/2402.03655) (Ryu, Xu, Erol, Bu, Zheng and Wornell, ICML 2024, [code](https://github.com/jongharyu/neural-svd)). Their NeuralSVD learns the singular functions of an operator with neural networks, orthogonalizes them implicitly, learns the modes in order, and treats the HGR dependence kernel explicitly.

### Fitting a low-rank ratio

The third route parametrizes the ratio as a low-rank product, $$r(x,y)=1+f_\theta(x)^\top g_\phi(y)$$, or with centering and orthonormality constraints,

$$
r(x,y)=1+\sum_{i=1}^{k} s_i\, f_i(x)\, g_i(y).
$$

The quadratic objective then becomes

$$
\max_{f,g,s}\ \sum_{i=1}^{k}\Big(2\,s_i\,\mathbb E\big[f_i(X)\,g_i(Y)\big]-s_i^2\Big),
$$

which learns the singular functions and the singular values together, without running an SVD on a density estimate. The same idea, without the explicit $$s_i$$, appeared in the same year as our paper: Soft-HGR (Wang et al., 2019), later used by Lee et al. (2022, eq. 19), maximizes

$$
\mathbb E\big[f(X)^\top g(Y)\big]-\tfrac12\operatorname{Tr}\big(C_{ff}\,C_{gg}\big)
$$

under a centering constraint only. Expanding the approximation error of the kernel shows why it works:

$$
\big\lVert K-f^\top g\big\rVert^2_{L^2(P_X\otimes P_Y)}
=\chi^2(P\,\Vert\,Q)-2\,\mathbb E_{XY}\big[f(X)^\top g(Y)\big]+\operatorname{Tr}\big(C_{ff}\,C_{gg}\big),
$$

so the optimal Soft-HGR value is $$\tfrac12\sum_{i\le k}\sigma_i^2$$. It is a low-rank neural version of the 2019 penalty, and it needs no covariance inverse.

### Representations and fairness

On the fairness side, [A Maximal Correlation Framework for Fair Machine Learning](https://www.mdpi.com/1099-4300/24/4/461) (2022) uses neural networks to estimate HGR and regularize multidimensional representations, and touches conditional fairness. Maximal correlation has also become a tool for understanding representation learning in general: [Neural Feature Learning in Function Space](https://arxiv.org/abs/2309.10140) (2023) builds a theory of feature learning on the spectral decomposition of the dependence kernel and the H-score, and [Dependence Induced Representations](https://arxiv.org/abs/2411.15328) (2024) relates maximal correlation functions to sufficient statistics, the usual classification losses and neural collapse. [SHGR](https://mlanthology.org/neurips/2025/stocksieker2025neurips-shgr/) (Stocksieker and Pommeret, NeurIPS 2025) generalizes HGR with Spearman correlation and a neural estimator for multivariate dependence.

Put together, this is how I read the state of things:

| Question | Where it stands |
| --- | --- |
| HGR with two networks | covered |
| rank-$$k$$ estimation | covered |
| quadratic objective without whitening | Soft-HGR, 2019 |
| ordered learning of the spectrum | NeuralSVD, 2024 |
| fairness on embeddings | covered in 2022 |
| conditional fairness | touched in 2022 |
| adaptive rank with statistical guarantees | open as far as I can tell |
| neural certification of independence | promising |
| conditional spectrum in high dimension | open as far as I can tell |

Replacing the KDE by an MLP is no longer a contribution in itself. The guarantee is another matter.

## Keeping the upper bound

What made the 2019 paper specific was that it gave an upper bound on HGR, where the neural methods above give estimates. That matters when the quantity is a regularizer. A neural estimator that returns a small value cannot tell you whether the dependence is small or whether the network failed to find it, and a model trained against it can learn to hide dependence where the critic does not look.

Take a rank-$$k$$ approximation $$K_k(x,y)=\sum_{i=1}^{k} s_i f_i(x) g_i(y)$$ of the kernel. Since HGR is the operator norm of $$K$$,

$$
\operatorname{HGR}(X,Y)\ \le\ \lVert K_k\rVert_{\mathrm{op}}+\lVert K-K_k\rVert_{\mathrm{HS}},
$$

and when the $$f_i$$ and $$g_i$$ are orthonormal this reads

$$
\operatorname{HGR}(X,Y)\ \le\ \max_i \lvert s_i\rvert+\sqrt{\chi^2(P\,\Vert\,Q)-\mathcal J(K_k)}
$$

with

$$
\mathcal J(K_k)=2\,\mathbb E_P[K_k]-\mathbb E_Q[K_k^2].
$$

The first term is what the networks found. The second is the energy they missed. With $$k=0$$ you get back $$\sqrt{\chi^2}$$, the 2019 bound. The bound is not monotone in $$k$$: at $$k=1$$ it is always looser, since $$\rho_1+\sqrt{\chi^2-\rho_1^2}\ge\sqrt{\chi^2}$$, and it only beats $$\sqrt{\chi^2}$$ once the learned modes have taken most of the energy, so in practice you would keep the minimum over $$k$$. How many modes that takes depends on how fast the spectrum decays:

{% include posts/hgr-spectrum.html %}

The hard part is the second term. We need a statistical upper bound on the residual energy $$\chi^2-\mathcal J(K_k)$$, and estimating $$\chi^2$$ by the variational objective above gives a lower bound, which goes the wrong way. Getting the right direction without falling back on density estimation in high dimension will take assumptions: smoothness of the density ratio, a known rate of spectral decay, or a bound on the complexity of the function class. Without some assumption of that kind I don't expect a non-trivial, distribution-free certificate on HGR to exist.

## Other open questions

Conditional dependence is the one I would most like to see solved. Estimating $$\operatorname{HGR}(X,Y\mid Z=z)$$ efficiently when $$Z$$ is itself high-dimensional is still hard.

Smaller but useful: choosing the rank adaptively, with guarantees that relate approximation error, estimation error and the tail of the spectrum; and a proper analysis of the bias that minibatch estimation introduces, which the shuffle figure above makes visible on independent data.

A more speculative idea is to use a generative model to move between $$P_0=P_X\otimes P_Y$$ and $$P_1=P_{XY}$$. An exact continuous normalizing flow gives the log ratio along a trajectory $$z_t$$ with velocity field $$v_t$$:

$$
\log\frac{p_1(z_1)}{p_0(z_1)}=\log\frac{p_0(z_0)}{p_0(z_1)}-\int_0^1 \nabla\!\cdot v_t(z_t)\,dt .
$$

That is a density ratio estimator, so in principle a route to the dependence spectrum. I am not convinced it beats estimating the ratio directly, which is cheaper and needs no ODE integration, and since $$\chi^2$$ is sensitive to the tails of the ratio, the flow would have to be accurate there.

## The follow-up I would write

The working title I keep coming back to is *Certified Neural Maximal Correlation for High-Dimensional Representations*: neural spectral estimation, a statistical control of the residual dependence, adaptive choice of the rank, and a regularizer that comes with a guarantee on the maximal correlation. That would keep the part of the ICML 2019 paper that was specific to it, the upper bound, instead of only replacing its KDE. The uses go beyond fairness, to disentanglement, invariant representations and independence testing.

Whether this is new is still an open question. I have not yet gone carefully through the work on neural independence tests and on nonparametric operator estimation, and a certified version could already exist there.
