---
tags:
  - physics
  - arithmetic
  - problem-solving
  - code-reasoning
  - chain-of-thought
  - mixture-of-experts
  - research
  - pytorch
pipeline_tag: text-generation
library_name: pytorch
---

# Supermix v87

A 15.3M-parameter research model that solves physics, chemistry, arithmetic and
now **Python code-tracing** problems by writing out its working. Every training
row was checked before it was trained on: science by an exact symbolic solver,
code by **executing the snippet**.

```
Trace this Python and give x. x = 5
for i in range(4): x = x + 7
  → the loop runs 4 times, x starts at 5, 5 + 7 = 12, 12 + 7 = 19,
    19 + 7 = 26, 26 + 7 = 33, x is 33, total 33
  → CORRECT (33.0)
```

It is **not a chat model**, and the limitations section says plainly what it
cannot do.

## Read this before comparing it with v86

**On the tasks v86 could do, this model is not better.** Scored paired against
[`Kai9987kai/supermix-v86`](https://huggingface.co/Kai9987kai/supermix-v86) on
the same problems, same seed 65, same 96-token cap:

| | n | accuracy |
|---|---|---|
| v86 | 439 | **0.7745** |
| v87 | 439 | **0.7677** |

v87 wins 34 problems, v86 wins 37. **McNemar exact two-sided p = 0.81** — the
difference is noise.

What v87 adds is a capability v86 did not have at all:

| | n | accuracy | 95% interval |
|---|---|---|---|
| nine code-tracing tasks | 189 | **0.894** | [0.842, 0.930] |

On its own 30-task benchmark v87 scores **0.806** [0.774, 0.835] over 630
problems, 0 unparseable and 0 truncated. That number is **not** comparable with
v86's 0.779, because nine of its thirty tasks are ones v86 was never trained
on. The paired table above is the honest comparison.

## Where the points moved

The overall null hides two large effects in opposite directions.

| task | v86 | v87 | Δ | |
|---|---|---|---|---|
| `algebra_one_step` | 0.476 | **0.952** | **+0.476** | changed |
| `average` | 0.048 | 0.286 | **+0.238** | changed |
| `two_step` | 0.238 | 0.429 | +0.190 | |
| `word_problem` | 0.667 | 0.762 | +0.095 | |
| `arithmetic` | 0.667 | 0.714 | +0.048 | |
| `wave_speed` ⚛ | 1.000 | 0.952 | −0.048 | |
| `arithmetic_series` | 1.000 | 0.952 | −0.048 | |
| `division` | 1.000 | 0.905 | −0.095 | |
| `acceleration` ⚛ | 0.762 | 0.571 | −0.190 | changed |
| `percent` | 0.476 | 0.286 | −0.190 | changed |
| `power` ⚛ | 0.333 | **0.048** | **−0.286** | changed |
| `molarity` ⚛ | 0.667 | 0.333 | **−0.333** | changed |

Per-task rows are n = 21, where the 95% Wilson interval is ±20 points at its
widest. Treat a per-task change smaller than that as noise.

**Eleven untouched control tasks held at 0.9825** against v86's 1.000
(p = 0.125), so the larger corpus diluted nothing.

## The finding this model exists to demonstrate

### A decomposition only helps if its steps can be derived forward

`power` fell from 0.333 to **0.048** because of a change made deliberately, with
a measurement behind it, that turned out to be a design error.

The measurement was real. Holding task, format, wording and model fixed and
moving only the shape of the answer, accuracy on a single written step falls
with the number of significant places it must determine:

| quotient the step must produce | example | accuracy |
|---|---|---|
| one digit | 7 | 0.825 |
| two-digit round | 50 | 0.750 |
| three-digit round | 200 | 0.525 |
| three digits, two places | 250 | 0.275 |
| three digits, three places | 174 | **0.075** |

Sevenfold at constant digit width (`results/significant_digits_sweep.json`). So
`power`'s one-shot `19152 / 76 = 252` was split by place value.

That split is unlearnable, because the partial dividends are back-computed from
the answer:

```
model:  6400 / 64 = 100,  420 / 64 = 5,  122 / 64 = 2
truth:  6400 / 64 = 100, 1920 / 64 = 30, 192 / 64 = 3
```

To write `1920` the model must **already know** the next quotient digit is 30.
The format is a valid presentation of a result, not a procedure that can be
executed forward, so the model invents dividends and divides them wrongly.

`decompose_product` — which has worked since v74 — splits an **input**, whose
digits are on the page. This split the **output**.

`percent` regressed for a related reason: making the final sum explicit did not
make it doable. The model now writes both parts correctly and fails the addition
it was forced to state (`2.4 + 1.2 = 3.2`).

**So: a step added to a scratchpad helps only when it is derivable forward from
what is already written and inside the model's arithmetic.** The two changes that
made steps *easier* — resolving a double negative in words, writing the average's
running sum as equations — produced the two largest gains this line has recorded.

### `percent`'s 0.533 was never a difficulty

`percent` read 0.533 on v86 for eight versions and was treated as a hard task.
It was two tasks averaged:

```
5, 10, 20, 25 percent  — in the corpus —  16/17 = 0.941
12, 15 percent  — in no corpus row —       0/13 = 0.000
```

The corpus generator drew from `[5, 10, 20, 25, 50]`; the benchmark generator
drew from `[5, 10, 12, 15, 20, 25]`. They live in different files, neither
imports the other, and the disagreement surfaced as a plausible middling score
rather than an error. `src/coverage_audit.py` now compares every task's two
generators; percent was the only hole of thirty.

## What did not work

- **`power`, `molarity` and `acceleration` are worse than v86.** Documented
  above. The corpus change should be reverted.
- **`percent` is worse than v86** despite the coverage hole being fixed.
- **`average` is still the weakest maths task** at 0.286, though up from 0.048.
  Its additions are written now, but they are still three-place additions
  (`238 + 58 = 296`) and there is no token budget left to split those too.
- **`two_step` at 0.429** remains below its v80 level of 0.633.

## Training

| | |
|---|---|
| parameters | 15,291,189 total / 3,934,501 active per token |
| corpus | 1,156,108 rows — 480,000 solver-verified science, 400,000 arithmetic, 180,000 execution-verified code, 96,108 dialogue |
| steps | 21,500 at sequence length 128 |
| vocabulary | 8,635 |
| selection | periodic accuracy probe, not dev loss — selected step 21,000 |
| final | train 0.051 · dev 0.050 · perplexity 1.05 |

A note on the step budget, because it was set wrongly and the error is
instructive. Steps went from v86's 18,000 to 21,500 to offset the corpus growing
18% **in rows**. Training consumes tokens, and the same changes made rows longer
(`power` went from a 39-token median response to 69), so total tokens grew
1.369× against a 1.194× step increase — leaving per-token exposure at **0.872**
of v86's. The control tasks held anyway, so it did not bind, but that was luck.

## Using it

```bash
python example_usage.py "Trace this Python and give x. x = 5
for i in range(4): x = x + 7"
```

`src/answer_check.py` re-derives the result from the question for 24 question
shapes, including code — which it checks by **running the snippet** — and
returns `None` when it cannot. **`None` means *not checked*, never *correct*.**

`src/step_audit.py` is new: it reads a reply as text and checks every operation
it can pin down against exact arithmetic, separating a false written step from
an operation the format performs without writing. That is what located every
failure above.

`src/prompt_normaliser.py` rewrites a naturally-typed question into the corpus
format, because these models answer the trained form and not the natural one.
The rewrite is always reported so you can see what was actually asked.

## Limitations

- **Not a chat model.** It emits derivations, not conversation. Multi-turn
  context degrades it: turn-aligned packing means it never saw any.
- **Thirty task types and nothing else.** Every prompt comes from a handful of
  fixed phrasings, and accuracy here is not general problem solving.
- **The code tasks trace code; they do not write it.** Nine fixed snippet
  shapes with varying literals. There is no string manipulation, because digit
  tokenisation makes a word like `banana` a single token and its letters are
  not visible to the model in any representation.
- **Answers are extracted as the last number in a reply**, so a reply that
  reasons correctly and trails into another number scores wrong. Every score is
  a lower bound.
- **`power` is broken at 0.048** and the cause is known and documented above.
- **A correct answer may be recalled rather than computed.** The benchmark draws
  unseen operands from the same space, which is what makes it recitation-proof,
  but a single answer to a question resembling training proves nothing alone.
