---
title: "Learning einsum and einops by doing it"
permalink: /blog/learning-einops-by-practice/
excerpt: "A small interactive site for learning einsum and einops, built to try a faster way of producing teaching material. The aim was to make learners write patterns instead of reading them. What it does, how the practice mode works, and what I would still change."
tags:
  - teaching
  - einops
  - tensors
---

Teaching material usually takes me a long time to produce, and it often ends up as slides that I understand and the audience mostly watches. For einsum and einops this is a bad format. The notation is short, and reading `b c h w -> b h w c` on a slide feels easy. Writing it yourself, and getting the order inside a parenthesis right, is a different skill, and you only learn it by doing it a few dozen times. So I used einops to explore a different way of producing teaching material, faster than writing slides, and built a small site that forces the reader to write the patterns: [ein.jeremie-mary.chatgpt.site](https://ein.jeremie-mary.chatgpt.site/).

The idea was to get active engagement built into the material from the start. It is still rough in places. Below is what it does and where it falls short.

## What is on the site

There are two tracks of eight short lessons each. The einsum track goes from reading the notation (`ij->ij`) to sums over an axis, transpose, dot and outer products, matrix multiplication, diagonals and trace, and batched products (`bij,bjk->bik`). The einops track covers `rearrange` (moving, merging and splitting axes, and the order inside a group), `reduce` and `repeat`, channels-first to channels-last, and cutting an image into patches with `(h ph) (w pw) -> (h w) (ph pw)`.

Each lesson shows a small tensor with actual numbers and animates where every value goes. The tensors are tiny on purpose, a 2 × 3 matrix or a strip of six values, so you can follow one number from input to output. Each lesson ends with a challenge and links to an exercise on the same idea, and there is a cheatsheet for later.

The lesson I care most about is "Order inside a group". `h w -> (h w)` and `h w -> (w h)` produce the same shape and different vectors, `[1, 2, 3, 4, 5, 6]` against `[1, 4, 2, 5, 3, 6]`. This mistake is easy to miss in real code, because nothing crashes: the shapes match and the model trains on scrambled data. The same point comes back with `repeat`, where `w -> (copy w)` tiles the whole strip and `w -> (w copy)` repeats each value next to itself.

## The practice mode

Practice is where most of the work went. Exercises come in three formats. You either write a pattern from a description, fix a pattern that runs but computes the wrong thing, or find the pattern that produces a given output. The broken patterns are generated so that they are valid and wrong (reversed output axes, a reversed group, a dropped axis), since a syntax error teaches much less than a pattern that runs and returns the wrong values.

The patterns stay hidden until you try or ask for a hint. Skills have prerequisites (outer products before matrix products, splitting axes before patches), and a scheduler picks the next exercise: a review that is due, then a skill you needed help on, then a new skill whose prerequisites you have done. Reviews come back after 1, 3, 7, 14 and 30 days. By default the practice mode mixes einsum and einops, so you do not solve five reductions in a row by pattern matching on the previous answer.

Each skill gets a label rather than a score. "Independent" means two first-try successes on different variants. "Retained" also needs a transfer problem and two successful reviews at least a day apart, and asking for help resets the review count. The site links the papers behind these choices (Karpicke and Roediger on retrieval practice, Cepeda et al. on spacing, Rohrer et al. on interleaving, Atkinson, Renkl and Merrill on faded examples) and says plainly that the intervals are heuristics and not a fitted model. I wanted that caveat on the page, because a progress bar invites learners to trust it more than it deserves.

At the end there is an attention exercise that puts the two libraries together. You implement causal self-attention or cross-attention in seven steps: split the heads with `rearrange`, contract Q and K with `einsum`, scale by $$\sqrt{d_k}$$, mask, take a softmax over the right axis, combine with V, and merge the heads back. Each step is checked against its own input layout, so a mistake in the head split does not make every later step fail.

## What does not work so well

Einops is mostly about practice, and the site gives a lot of it, but several things are not right yet.

The feedback on a wrong answer is weak. It compares output shapes: "Your output shape is 6; the requested shape is 6." For the exercises where the shape is right and the order is wrong, which are the interesting ones, that sentence says nothing. The einops message adds that matching shapes are not enough, which is true but still does not point to the axis that is wrong. Showing the learner's output next to the expected one, with the first misplaced value highlighted, would help more than any text.

The first exercise after each lesson is close to the lesson's own example, often with the same tensor, and the hints are generous ("Start from `ij,jk->ik`. Change only the order of the two output labels."). A learner can pass the first level by editing what they just saw. The scheduler then asks for variants and spaced reviews, so this matters less over time, but the first impression is easier than it should be.

Everything runs on a small JavaScript reimplementation of einsum and einops, not on NumPy or PyTorch. It is enough for the visualisations, but it limits tensors to a few entries per axis, the einsum playground takes at most two inputs and refuses ellipsis, and a learner never sees a real dtype error. In PyTorch, `reduce(x, 'h w -> h', 'mean')` on an integer tensor fails, and the site only mentions this in a lesson note. Real code also has shapes like `(b, 197, 768)`, and the step from 2 × 3 toys to those is left to the learner.

Progress lives in the browser's local storage. It is lost when someone switches machine or clears their data, and as the author I cannot see which exercises people get stuck on, which is the information I would need to improve the material. The review cycle, which goes up to 30 days, also only helps people who come back; someone who opens the site once gets none of it.

Some of the prose is written in a product voice ("brings matrix multiplication to life") that I would rewrite. The lessons also stop before `pack` and `unpack` and before ellipsis in einops patterns, both of which show up often in model code.

As an experiment in producing teaching material quickly, it worked: the site exists, and nobody can get through it without writing patterns. Before it gets any new topic, the next version needs better feedback on wrong answers and some exercises on realistic tensor shapes.
