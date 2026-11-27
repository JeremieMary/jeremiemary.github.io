---
title: "Telling time series apart with a classifier"
permalink: /blog/telescope-distance/
excerpt: "The telescope distance turns questions about highly dependent time series into a sequence of binary classification problems, and it does not need the classifiers to be good. A look back at the JMLR 2013 paper with Daniil Ryabko, and a correction to two of its lemmas found by Jesse Geneson."
tags:
  - time series
  - two-sample tests
  - research notes
---

In 2013 Daniil Ryabko and I published [A Binary-Classification-Based Metric between Time-Series Distributions and Its Use in Statistical and Learning Problems](https://jmlr.org/papers/volume14/ryabko13a/ryabko13a.pdf) in JMLR, an extended version of our NIPS 2012 paper. This summer [Jesse Geneson](https://www.sjsu.edu/math/about-us/faculty/jesse-geneson.php) (San José State University) wrote to us about a problem in two of its lemmas. He is right, and the second half of this post records the correction. The first half explains what the paper does, because the correction only makes sense once you see which step it touches.

## Two series that agree one point at a time

The synthetic experiment in the paper uses a family of processes from ergodic theory. Pick a phase $$r_0$$ uniformly on $$[0,1)$$ and move it by a fixed $$\alpha$$ at each step, $$r_{i+1}=r_i+\alpha \bmod 1$$. The observation $$X_i$$ is Gaussian noise around one mean when $$r_i<1/2$$ and around another when $$r_i\ge 1/2$$. For irrational $$\alpha$$ the process is stationary ergodic, it has no finite-state description, and it is not even a B-process. Whatever $$\alpha$$ is, the phase spends half its time in each half of the circle, so every value of $$\alpha$$ gives the same distribution for a single $$X_i$$.

Take two such series with different rotations. Any test that looks at the values one at a time sees two samples from the same mixture of Gaussians. The difference is in the order: with a small $$\alpha$$ the series stays at one level for long runs, and with $$\alpha$$ close to $$1/2$$ it switches almost every step. Pairs $$(X_i,X_{i+1})$$ already show this. In other cases you may need triples, or windows of length 50, and you do not know in advance which.

## Telescoping

The paper starts from a classical distance between two distributions $$P$$ and $$Q$$ on a space $$\mathcal X$$, indexed by a class $$\mathcal H$$ of sets:

$$
d_{\mathcal H}(P,Q)=\sup_{h\in\mathcal H}\ \lvert P(h)-Q(h)\rvert .
$$

Kolmogorov–Smirnov is the case of half-lines, and total variation is the case of all measurable sets. For time series we apply it to the $$k$$-dimensional marginals, with a class $$\mathcal H_k$$ of subsets of $$\mathcal X^k$$ for each $$k$$, and add up the results with summable weights $$w_k$$ (we used $$w_k=1/k^2$$ in the experiments and $$2^{-k}$$ for the finite-sample bounds):

$$
D_{\mathbf H}(\rho_1,\rho_2)=\sum_{k=1}^{\infty} w_k\,\sup_{h\in\mathcal H_k}\Big\lvert\, \mathbb E_{\rho_1} h(X_1,\dots,X_k)-\mathbb E_{\rho_2} h(Y_1,\dots,Y_k)\Big\rvert .
$$

That is the telescope distance. The empirical version replaces each expectation by a frequency over the sliding windows of length $$k$$,

$$
\hat P_k(h)=\frac{1}{n-k+1}\sum_{i=1}^{n-k+1}h(X_{i..i+k-1}),\qquad \hat Q_k(h)=\frac{1}{m-k+1}\sum_{i=1}^{m-k+1}h(Y_{i..i+k-1}),
$$

$$
\hat D_{\mathbf H}(X_{1..n},Y_{1..m})=\sum_{k=1}^{\min(n,m)} w_k\,\sup_{h\in\mathcal H_k}\big\lvert \hat P_k(h)-\hat Q_k(h)\big\rvert ,
$$

and each supremum is a binary classification problem. Label the windows of the first series 1 and the windows of the second series 0. The class-weighted empirical risk of a set $$h$$, the fraction of windows of the first series it misses plus the fraction of windows of the second series it catches, is

$$
\big(1-\hat P_k(h)\big)+\hat Q_k(h)=1-\big(\hat P_k(h)-\hat Q_k(h)\big),
$$

so when $$\mathcal H_k$$ is closed under complements, as halfspaces are, the $$k$$-th term is one minus the smallest weighted risk reachable in $$\mathcal H_k$$. Equivalently, it is twice the best balanced accuracy minus one. In the paper we used an SVM with a Gaussian kernel and libsvm's default parameters.

The weights resolve the problem of not knowing which window length matters. Short windows that cannot tell the processes apart give terms that go to zero. Long windows have too few samples to be estimated well, but they are multiplied by small weights. The sum is dominated by the window lengths that both separate the processes and are estimated well, and it goes to zero if the processes are the same. In practice we stopped the sum at $$k=\log n$$, since higher-order terms do not have enough data to converge, and this changes none of the asymptotic results.

{% include posts/tel-window.html %}

The figure runs the construction on the rotation example. At $$k=1$$ the two CDFs lie on top of each other. At $$k=2$$ the best direction separates them, and later terms add a little more. Switching B to the same rotation as A shows the other side: the terms that remain are pure estimation noise, and they grow with $$k$$ as the same 500 points are spread over more dimensions.

## The classifier does not have to be good

This was, to me, the most interesting part of the paper. In binary classification, if the Bayes classifier is not in your class, your error does not go to zero no matter how much data you have. Here we never need the classification error to be small. We only need the empirical supremum to converge to the true supremum over the same class. The approximation error of $$\mathcal H_k$$ is irrelevant; only its estimation error counts.

The convergence result (Theorem 5) needs very little. If each $$\mathcal H_k$$ is a separable class of indicator functions with finite VC dimension, then for any two stationary ergodic processes

$$
\lim_{n,m\to\infty}\hat D_{\mathbf H}(X_{1..n},Y_{1..m})=D_{\mathbf H}(\rho_X,\rho_Y)\quad\text{a.s.}
$$

There is no mixing condition and no assumption on memory. The heavy lifting is a result of Adams and Nobel (Bernoulli, 2012): for stationary ergodic processes, empirical frequencies converge uniformly over VC classes. The rest of the proof cuts the sum at a finite $$T$$ where the tail weight is below $$\varepsilon$$, and applies that result to each of the $$T$$ remaining terms. Halfspaces in $$\mathbb R^k$$ already qualify, so the classifier can be as weak as a linear one. A weak class gives a weaker distance, but it is still a consistent estimate of that distance.

## What a consistent distance buys

Once a distance between process distributions can be estimated consistently and is zero only for identical processes, some problems become corollaries. In the three-sample problem, you assign a third series $$Z$$ to whichever of $$X$$ and $$Y$$ is closer. Clustering with a known number of clusters works the same way: eventually the empirical distance has the strict separation property, and average linkage or farthest-point clustering recover the target partition. Both results hold for arbitrary stationary ergodic processes, and in the clustering case the samples may even depend on each other.

Homogeneity testing (are $$X$$ and $$Y$$ from the same process?) and clustering with an unknown number of clusters are harder: for stationary ergodic processes no consistent procedure exists at all. Under geometric $$\beta$$-mixing, the paper gives finite-sample bounds through Karandikar and Vidyasagar's inequality. With halfspaces one can take the threshold $$\varepsilon_n=n^{-1/8}$$, and the probability of a type I error then decays like $$\exp(-n^{1/4})$$.

On the brain-computer interface data from BCI competition III, unsupervised clustering with the telescope distance reached 84%, 81% and 61% accuracy on the three subjects, against 46%, 41% and 36% for dynamic time warping. It used only the first 3 of the 96 channels, with sequences of about 300 points, and the whole experiment took about six minutes on a laptop.

## A correction, thanks to Jesse Geneson

The results of the previous section lean on one property: $$D_{\mathbf H}(\rho_1,\rho_2)=0$$ only when $$\rho_1=\rho_2$$. The paper establishes it with two lemmas, and Jesse pointed out that both are stated too strongly.

**Lemma 1** says that $$d_{\mathcal H}$$ is a metric on the distributions over $$\mathcal X$$ if and only if $$\mathcal H$$ generates the σ-algebra. Jesse's counterexample is four points. Take $$\mathcal X=\{1,2,3,4\}$$ and $$\mathcal H=\{\{1,2\},\{1,3\}\}$$. The σ-algebra generated by $$\mathcal H$$ contains every subset: $$\{1\}$$ is the intersection, $$\{2\}$$ and $$\{3\}$$ are differences, $$\{4\}$$ is the complement of the union. Now let

$$
P=\big(\tfrac12,0,0,\tfrac12\big),\qquad Q=\big(0,\tfrac12,\tfrac12,0\big).
$$

Both give mass $$1/2$$ to $$\{1,2\}$$ and to $$\{1,3\}$$, so $$d_{\mathcal H}(P,Q)=0$$ while $$P\ne Q$$.

The gap is a standard one. Two measures that agree on a generating class agree on the whole σ-algebra when the class is closed under intersections; that is Dynkin's π–λ theorem. Without that closure they need not agree. Adding the intersection $$\{1\}$$ to $$\mathcal H$$ in the example separates $$P$$ from $$Q$$ immediately. The condition that does the work is that $$\mathcal H$$ be *measure-determining*: $$P(h)=Q(h)$$ for all $$h\in\mathcal H$$ implies $$P=Q$$. Generating the σ-algebra is not enough on its own. A generating class closed under intersections is enough.

**Lemma 3** says that $$D_{\mathbf H}$$ is a metric if and only if every $$d_{\mathcal H_k}$$ is. The "if" direction is fine. The "only if" direction fails. Let $$\mathcal H_1=\{\emptyset,\mathcal X\}$$ and $$\mathcal H_k$$ the halfspaces of $$\mathbb R^k$$ for $$k\ge 2$$. Then $$d_{\mathcal H_1}$$ is not a metric, but $$D_{\mathbf H}$$ is one, because the two-dimensional marginals already determine the one-dimensional ones. And by the four-point example, asking each $$\mathcal H_k$$ to generate $$\mathcal F_k$$ does not make $$D_{\mathbf H}$$ separating either. The correct statement is about the family of processes you are working with: $$D_{\mathbf H}$$ is a metric on a class $$\mathcal C$$ of process distributions exactly when, for any two distinct $$\rho_1,\rho_2\in\mathcal C$$, some $$k$$ has $$d_{\mathcal H_k}(\rho_1^{(k)},\rho_2^{(k)})>0$$.

What this changes:

- **Theorem 5 is unaffected.** Its proof uses finite VC dimension, separability and the summability of the weights. It never uses the generating assumption, so $$\hat D_{\mathbf H}\to D_{\mathbf H}$$ holds as stated.
- **The results that use $$D_{\mathbf H}$$ to distinguish processes need the separation property as an explicit assumption.** These are the three-sample test, the clustering theorems and the homogeneity test. In their hypotheses, "$$\mathcal H_k$$ generates $$\mathcal F_k$$" should read "$$\mathcal H_k$$ is measure-determining on $$\mathcal X^k$$", or the separation property should be assumed directly on the class of processes.
- **The main example still works.** Halfspaces in $$\mathbb R^k$$ are not closed under intersection, but they are measure-determining by the Cramér–Wold theorem. If $$P$$ and $$Q$$ agree on every halfspace $$\{x:\langle a,x\rangle\le t\}$$, then every one-dimensional projection has the same distribution under both, and so $$P=Q$$. This covers Theorem 20 as well, which compares the telescope distance over halfspaces with the distributional distance.

The paper also claims, right after Lemma 1, that halfspaces in a kernel feature space generate the σ-algebra. That sentence needs the same repair. For a characteristic kernel such as the Gaussian one, the Cramér–Wold argument carries over: agreeing on every set $$\{f\ge t\}$$, with $$f$$ in the RKHS, forces each $$f(X)$$ to have the same law, so the kernel mean embeddings coincide and $$P=Q$$.

Jesse sent us a short note with the details and a corrected formulation. The statements above follow his diagnosis, and any slip in how I have written them up is mine. The addresses printed on the paper are INRIA ones that I no longer have access to, and I don't have a clean way to attach an erratum to the JMLR page, so this post is where the correction lives for now. I am grateful to Jesse for reading a thirteen-year-old paper this closely and for taking the time to write.

## Reading it again in 2026

The reduction from distances to classifiers has become common since then. The identity above, with $$d_{\mathcal H}$$ as twice the balanced accuracy minus one, is the statistic behind [classifier two-sample tests](https://arxiv.org/abs/1610.06545) (Lopez-Paz and Oquab, 2017), and $$d_{\mathcal H}$$ is, up to a factor of two, the $$\mathcal H$$-divergence of the domain adaptation bounds of Ben-David and colleagues. Kifer, Ben-David and Gehrke had already used it in 2004 to detect change in data streams. All of these work with i.i.d. samples. What the telescope construction adds is a way to keep the reduction valid under arbitrary dependence, at the price of giving up rates unless you assume mixing.

It also connects to my [previous post on maximal correlation](/blog/maximal-correlation-fairness/). There the two distributions to separate were a joint law and the product of its marginals, and the second sample came from shuffling within a batch. Here both samples are given, and the dependence that matters sits inside each series, along the time axis. In both cases a classifier, or a critic, is asked to tell two samples apart, and the quality of the answer depends on how well its estimation error is controlled, not on whether it is a good classifier.

The paper's outlook left one question open, and I still find it the most natural follow-up: extending the consistency results from empirical risk minimizers to the classifiers people actually use, such as nearest neighbours or today's neural networks. That would require proving their consistency for stationary ergodic data, not just i.i.d. data, and Jesse's note is a reminder to state the separation assumption carefully when doing it.
