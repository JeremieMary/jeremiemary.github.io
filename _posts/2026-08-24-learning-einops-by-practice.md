---
title: "Learning einsum and einops by doing it"
permalink: /blog/learning-einops-by-practice/
excerpt: "A small interactive site that makes you write einsum and einops patterns instead of reading them. What it does and what I would still change."
tags:
  - teaching
  - einops
  - tensors
---

Teaching material takes me a long time to produce, and it usually ends up as slides the audience watches. That format suits einsum and einops badly. Reading `b c h w -> b h w c` on a slide feels easy; writing it yourself and getting the order inside a parenthesis right takes a few dozen tries. So I tried a faster way of making material and built a small site where you have to write the patterns: [ein.jeremie-mary.chatgpt.site](https://ein.jeremie-mary.chatgpt.site/).

## What is on the site

There are two tracks of eight short lessons, one for einsum (from `ij->ij` to batched products like `bij,bjk->bik`) and one for einops (`rearrange`, `reduce` and `repeat`, up to cutting an image into patches). Each lesson animates a tiny tensor, a 2 × 3 matrix or a strip of six values, so you can follow one number from input to output.

The lesson I care most about is the order inside a group. `h w -> (h w)` and `h w -> (w h)` have the same shape and give `[1, 2, 3, 4, 5, 6]` against `[1, 4, 2, 5, 3, 6]`. In real code nothing crashes; the model just trains on scrambled data.

## Practice

You write a pattern from a description, fix one that runs but computes the wrong thing, or find the pattern behind a given output. The broken patterns are generated to be valid and wrong (reversed axes, a dropped axis), because a syntax error teaches much less. A scheduler picks the next exercise from due reviews, skills where you needed help, and new skills whose prerequisites you have done. Reviews come back after 1, 3, 7, 14 and 30 days, and einsum and einops exercises are mixed so you cannot coast on the previous answer. The site cites the work on retrieval practice, spacing and interleaving, and says the intervals are heuristics, not a fitted model.

The last exercise is causal self-attention in seven steps, from splitting the heads with `rearrange` to merging them back. Each step is checked against its own input, so one mistake does not fail every later step.

## What does not work yet

Feedback on a wrong answer only compares shapes, so for the interesting exercises (right shape, wrong order) it says nothing useful. Showing the learner's output next to the expected one, with the first misplaced value highlighted, would do more.

The first exercise after a lesson is close to the lesson's own example and the hints are generous, so the first level is too easy.

Everything runs on a small JavaScript reimplementation, not NumPy or PyTorch. Tensors stay tiny, there are no dtype errors, and the jump from 2 × 3 toys to `(b, 197, 768)` is left to the learner.

Progress lives in the browser's local storage, so I cannot see where people get stuck, and the 30-day reviews only help people who come back. The lessons also stop before `pack`, `unpack` and ellipsis, which show up often in model code.

As a way to produce teaching material quickly, it worked: nobody gets through the site without writing patterns. The next version needs better feedback on wrong answers and exercises on realistic shapes.
