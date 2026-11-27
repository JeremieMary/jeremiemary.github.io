---
title: "Semantic IDs that know what people buy together"
permalink: /blog/generative-recommendation-semantic-ids/
excerpt: "In August 2025, with Simon Lepage and David Picard, we argued that generative recommenders lagged behind a well-trained SASRec for two fixable reasons: tokenizers that ignore behavior, and an encoder-decoder that reads the history far too often. The paper was not accepted. A year later, both ideas are standard in industry."
tags:
  - recommender systems
  - semantic IDs
  - generative recommendation
  - research notes
---

In August 2025 Simon Lepage, David Picard and I posted [Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling](https://arxiv.org/abs/2508.14910) on arXiv. It came out of Simon's PhD at Criteo AI Lab and École des Ponts, and it was rejected at the conference we sent it to. It is still a single arXiv version.

I want to come back to it for two reasons. The first is timing. The paper was written before Google posted [PLUM](https://arxiv.org/abs/2510.07784) in October 2025, and PLUM's tokenizer adds to RQ-VAE almost the same co-occurrence contrastive term as ours. The second is that 2025 and 2026 turned semantic IDs from an academic curiosity into something many large recommender teams now run in production. The two questions we asked, how to put behavior into the codes and how to stop paying for the history at every decoding step, are the ones those teams ended up answering.

## The setting

A generative recommender does not score every item of the catalog. It turns each item into a short tuple of discrete codes, its semantic ID (SID), and trains a sequence model to write the codes of the next item, one token at a time. [TIGER](https://arxiv.org/abs/2305.05065) (Rajput et al., NeurIPS 2023) set the template: embed the item's text with a sentence encoder, compress the embedding with a residual quantizer (RQ-VAE), and train a T5 encoder-decoder on the resulting token sequences.

The argument for this design is mostly about scale. There is no embedding table that grows with the catalog, no separate approximate nearest-neighbor index, and items that share a prefix share statistical strength, which helps the long tail. The argument against it, in 2024, was that the published numbers were not very good once you looked at the baselines.

## The baseline problem

Most generative recommendation papers compared against SASRec trained the way the original code trains it: binary cross-entropy with a single sampled negative. That is a weak way to train SASRec. Moving to a full softmax cross-entropy, which [Klenitskiy and Vasilev](https://arxiv.org/abs/2309.07602) had already shown to matter more than the choice between SASRec and BERT4Rec, and adding two cheap augmentations (random cropping of timelines, and shuffling items that share a timestamp) changes the picture. We called the result SASRec++. Test R@10, mean over 5 runs:

| SASRec variant | Beauty | Sports | Toys |
| --- | --- | --- | --- |
| BCE, one negative | 5.48% | 3.81% | 5.86% |
| + filtering seen items (the official code) | 6.33% | 4.19% | 6.49% |
| + cross-entropy | 8.70% | 5.53% | 8.84% |
| + crop and shuffle | 9.73% | 6.44% | 9.94% |

That is more than 50% relative gain over the official implementation, with no change to the model, while TIGER reports 6.42% on Beauty. At that point the honest summary was that generative retrieval on these benchmarks was behind a properly trained ID model. We blamed two things for it.

## COSETTE: putting behavior into the codes

A standard RQ-VAE is trained to reconstruct a frozen text embedding. Each item is processed on its own, so the codes describe what the product page says. Recommendation needs something else: two products should share a prefix when the same people buy them, whatever their descriptions say. Running shoes and energy gels have nothing in common as text.

Earlier work had tried to fix this in two ways. [CoST](https://arxiv.org/abs/2404.14774) replaced reconstruction with a contrastive loss, but still on content embeddings. [LETTER](https://arxiv.org/abs/2405.07314) and [EAGER](https://arxiv.org/abs/2406.14017) aligned the codes with embeddings from a pretrained SASRec, which adds a whole training stage and ties the tokenizer to another model's geometry.

COSETTE (COllaborative and SEmantic Tokenization of Text Embeddings) takes the co-occurrence signal directly. Sample a batch of user timelines, collect the $$B'$$ distinct items in it, quantize each of them to $$\hat z_i$$, and call a pair positive when the two items appear together in at least one timeline of the batch. Then apply a sigmoid contrastive loss in the style of [SigLIP](https://arxiv.org/abs/2303.15343) on the quantized latents:

$$
\mathcal L_C=\frac{1}{B'}\sum_{i=1}^{B'}\frac{1}{Y_i}\sum_{j=1}^{B'}\log\Big(1+e^{\,y_{ij}\,(-t\,\langle \hat z_i,\hat z_j\rangle+b)}\Big),
$$

with $$y_{ij}=1$$ for co-occurring pairs and $$-1$$ otherwise, $$\langle\cdot,\cdot\rangle$$ the cosine similarity, $$Y_i$$ the number of positives of item $$i$$, and a learned temperature $$t$$ and bias $$b$$. The full objective keeps the usual quantization and reconstruction terms:

$$
\mathcal L_{\text{COSETTE}}=\mathcal L_Q+\mathcal L_R+\mathcal L_{QC}+\lambda\,\mathcal L_C .
$$

The contrastive term acts on the quantized vectors, so the gradient reaches the codebooks through the straight-through estimator. No pretrained recommender is involved. The figure below shows the effect on a toy catalog.

{% include posts/sid-cosette.html %}

Two details mattered in practice. The reconstruction loss works best on items sampled uniformly from the catalog, while the contrastive loss sees items in proportion to their popularity; using the popularity-weighted items for both hurt the small datasets. And the contrastive loss is a cross-entropy while the others are MSE terms around $$10^{-3}$$, so $$\lambda=10^{-3}$$ is what keeps it from flattening the codebooks.

A side effect we had not planned: COSETTE produces fewer collisions (distinct items with the same tuple) than RQ-VAE, because the contrastive term pushes apart items that do not co-occur even when their texts are close.

The gains hold across generators. Validation R@10 with codebooks of 256 and tuples of length 4, everything else fixed:

| Generator | Tokenizer | Video Games (11k items) | Movies & TV (50k) | Pet Supplies (115k) |
| --- | --- | --- | --- | --- |
| TIGER | RQ-VAE | 9.79% | 5.03% | 2.57% |
| TIGER | LETTER | 10.33% | 5.05% | 2.48% |
| TIGER | COSETTE | 10.89% | 6.44% | 3.30% |
| MARIUS | RQ-VAE | 14.24% | 9.16% | 5.01% |
| MARIUS | LETTER | 14.84% | 9.17% | 4.77% |
| MARIUS | COSETTE | 15.02% | 9.90% | 5.32% |

The largest relative gains are with TIGER on the two bigger catalogs (+28% on both), where the text alone has to tell apart many similar products.

## MARIUS: stop rereading the history

The second problem is the architecture. TIGER flattens a history of $$N$$ items into $$N\cdot L$$ code tokens and feeds them to a bidirectional encoder. A bidirectional encoder cannot use a KV cache, so the whole history is re-encoded for every request. Then each of the $$L$$ decoding steps cross-attends to all $$N\cdot L$$ tokens, for each beam.

Audio models had already met the same structure. A neural audio codec turns each frame into a stack of residual codes, and models such as [Moshi](https://arxiv.org/abs/2410.00037) and UniAudio handle it with the RQ-Transformer of [Lee et al. (2022)](https://arxiv.org/abs/2203.01941): a large transformer along time, with one token per frame, and a small transformer along the depth of the code stack. MARIUS (Multi-scale Attention as Recommendation Index with fUSion) applies this to recommendation. Each item becomes one token, the sum of the embeddings of its $$L$$ codes. A causal temporal transformer runs over the $$N$$ items and produces a context vector $$h$$. A depth transformer, applied independently at each position, decodes the codes of the next item from $$h$$ and the codes already written.

{% include posts/sid-marius.html %}

The temporal transformer plays the role SASRec plays in an ID model, and in our sizing experiments it behaved like one: it liked a lot of dropout (0.4) and did not need to grow with the catalog. The depth transformer plays the role of the item index, and it does need to grow with the number of items. In the paper's complexity model, with $$B$$ beams and $$T_E$$, $$T_D$$ the per-layer costs of the two stacks, inference costs

$$
\begin{aligned}
\text{TIGER:}\quad & (NL)^2\,T_E+B\,(NL^2+L^2)\,T_D,\\
\text{MARIUS:}\quad & N^2\,T_E+B\,L^2\,T_D,
\end{aligned}
$$

so the decoding cost of MARIUS does not depend on the history length at all. In training, the causal structure lets every position be supervised in parallel. Measured on one H200 with similar model sizes and $$N=50$$, MARIUS processes 33 times more training items per second than TIGER and runs beam search 3 times faster.

## What the numbers say

On the small Amazon 2014 benchmarks, MARIUS with a plain RQ-VAE already beats all the T5-based generative models, and COSETTE brings it level with SASRec++. Test R@10:

| Model | Beauty | Sports | Toys |
| --- | --- | --- | --- |
| SASRec (as usually reported) | 6.05% | 3.50% | 6.75% |
| SASRec++ | 9.73% | 6.44% | 9.94% |
| TIGER | 6.42% | 4.19% | 6.98% |
| LETTER-TIGER | 7.39% | 4.36% | 6.84% |
| MARIUS + RQ-VAE | 9.71% | 6.10% | 9.13% |
| MARIUS + COSETTE | 10.02% | 6.72% | 9.51% |

These datasets have 12k to 18k items, which is where an ID model is at its most comfortable. The more interesting test is the 2023 version of the Amazon data, an order of magnitude larger, with up to 267k items and 796k users. There TIGER falls far behind, and MARIUS with COSETTE (now on NV-Embed-v2 embeddings) beats SASRec++ on R@10 in nine categories out of ten.

{% include posts/sid-results.html %}

One caveat I would add today: the large-scale numbers are single runs after a grid search, whereas the small-benchmark ones are averages over five seeds. The R@10 margins are consistent across categories, but some of them are small.

The more useful result, to me, is where the difference comes from. MARIUS is very slightly behind SASRec++ for $$K\le 3$$ and pulls ahead as $$K$$ grows. Split by item popularity, it predicts the most popular decile less often and deciles 3 to 7 more often, and its recall gains are concentrated in those mid-popularity deciles. In other words it trades wrong bestseller predictions for correct mid-catalog ones. It does slightly worse on the rarest items, which remain hard for everyone. We also ran beam search without any constraint to valid tuples and counted the invalid ones: 14 out of almost 6 million top-10 predictions on Pet Supplies, against 261 for TIGER. In practice constrained decoding was not needed.

## Then semantic IDs went mainstream

When we wrote the paper, semantic IDs were mostly an academic topic, with YouTube's SID ranking features and Kuaishou's OneRec as the main industrial exceptions. Fifteen months later the list looks very different.

{% include posts/sid-timeline.html %}

### Behavior moved into the tokenizer

Most of the large deployments now feed engagement data into the codes in one way or another. Kuaishou's [OneRec report](https://arxiv.org/abs/2506.13695) (June 2025) aligns the item embeddings with co-click pairs before quantizing them, as a separate stage. PLUM, posted on 9 October 2025, two months after us, trains an RQ-VAE whose loss includes a contrastive term on items that co-occur in user sequences. That is COSETTE's construction, with a softmax instead of a sigmoid and multi-resolution codebooks, and it is in production for YouTube retrieval. The two pieces of work are independent and close to concurrent, and PLUM goes much further than we did on the LLM side. Since then [LinkedIn](https://arxiv.org/abs/2606.01396) reported that SIDs built from behavioral embeddings beat SIDs built from profile text, [Meta](https://engineering.fb.com/2026/10/19/ml-applications/genrecs-tokenization-transforming-ads-recommendation-at-meta-with-stable-model-vocabulary/) described an ads tokenizer that "combines that content signal with representations learned from how people engage with the items", and [Pinterest's PinGER](https://arxiv.org/abs/2610.33435) distills its SIDs toward engagement graph embeddings. On the academic side, [SCRec](https://arxiv.org/abs/2609.13678) (RecSys 2026) cites COSETTE and works on the same question of how to get collaborative signal into the codes.

### The history is read once

The MARIUS split shows up under different names. Kuaishou's [OneRec-V2](https://arxiv.org/abs/2508.20900) dropped the encoder for a "lazy" decoder-only design that encodes the context once. Tencent's [GPR](https://arxiv.org/abs/2511.10138) separates user-intent modeling from ad generation in its decoder. PinGER serves one token per history item and decodes the next item's codes with per-level heads. The closest academic version is [Deco](https://arxiv.org/abs/2610.32980) by Gusak, Volodkevich, Oseledets and Frolov: a backbone over one token per item produces a "next-item concept", and a small KV-cached decoder runs the beam search over codes, with 3.2 to 8.6 times higher throughput. Industry has tended to replace the depth transformer with multi-token prediction heads, as in Shopify's SID-LLM; Netflix's [comparison](https://arxiv.org/abs/2610.15864) found that autoregressive decoding of the codes still gives better recommendations than parallel heads, while the heads win on throughput.

### People started checking the baselines

A [SIGIR-AP 2026 reproducibility study](https://arxiv.org/abs/2609.24430) finds that SASRec beats several generative methods on some datasets, and [another paper](https://arxiv.org/abs/2605.25330) shows that SID collisions can inflate Hit@10 by up to 103% when they are not handled in evaluation. I have not seen a published version of the plain SASRec++ argument (cross-entropy plus two augmentations), and I still think it is the first thing a reviewer should ask for in a generative recommendation paper.

## Looking back

The paper was rejected, and I understand some of the reasons: the small benchmarks are saturated and the large-scale runs are single seeds. The two claims held up all the same. A tokenizer trained only to reconstruct content leaves a lot on the table, and the cheapest fix is to tell it which items people consume together. An encoder-decoder that rereads the flattened history for every code is the wrong shape for recommendation, and a model that summarizes each item once and decodes the codes with something small is both faster and more accurate than TIGER.

What I would do now is mostly about scale. The interesting regime is millions of items, where the depth decoder carries the whole index and has to grow with it, and where the tokenizer has to be retrained as the catalog changes without reshuffling every code. Those are the problems the production systems in the timeline are now working on.

Simon's PhD thesis is *Recommendation with Transformers: from dense to generative retrieval* (École des Ponts, 2025). The paper is on [arXiv](https://arxiv.org/abs/2508.14910).

*A note on how this post was made. This is material I would not have had the time to produce: the write-up, the literature follow-up and the interactive figures were made with an LLM, which I prompted and whose output I read and corrected. I believe some uses of LLMs can be positive in this way, when an expert does the prompting and the reading.*
