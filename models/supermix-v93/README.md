---
tags:
  - physics
  - arithmetic
  - problem-solving
  - code-reasoning
  - chain-of-thought
  - mixture-of-experts
  - connectome
  - neurogenesis
  - research
  - pytorch
pipeline_tag: text-generation
library_name: pytorch
base_model: Kai9987kai/supermix-v89
---

# Supermix v93

A 36.6M-parameter research model, warm-started from
[`Kai9987kai/supermix-v89`](https://huggingface.co/Kai9987kai/supermix-v89),
that solves physics, chemistry, arithmetic and Python code-tracing problems by
writing out its working. It carries two things v89 did not:

* **two fly brains** — the left and right hemispheres of the Janelia male
  *Drosophila* central nervous system connectome (384 modules each, 16,373
  real synaptic edges of which 34% are commissural bonds between the two),
  grafted onto the transformer as a recurrent branch that reads from three
  layers and writes into three layers and the model's thinking core;
* **neurogenesis** — during training the model grew and pruned its own
  structure: new vocabulary rows, new experts in the mixture-of-experts
  layers, new modules and synapses in the brains, new brain-to-trunk taps,
  and one appended layer.

```
What is the impulse from a force of 46 N acting for 7 s?
  → impulse = force x time, 40 x 7 = 280, 6 x 7 = 42,
    the impulse is 322 newton seconds, total 322
  → CORRECT (322.0)
```

It is **not a chat model**. Read the results before drawing conclusions
from the architecture: the causal tests below say the brains are not used.

## Results, paired against its predecessors

Same 630 novel problems (seed 65, 96-token cap, generator fingerprint
`3b99a446…`), exact McNemar on the 628 shared inputs:

| model | correct / 630 | acc | v93 vs it (wins / losses, p) |
|---|---|---|---|
| **v93** (this checkpoint) | **571** | **0.906** | — |
| v89 | 577 | 0.919 | 17 / 25, p = 0.28 |
| v91 C (v89 + 2,500 more steps, no graft) | 580 | 0.924 | 16 / 27, p = 0.13 |
| v91 A (v89 + one connectome graft) | 587 | 0.935 | 10 / 28, p = 0.005 |
| v91 B (same graft, rewired null wiring) | 592 | 0.943 | 8 / 31, p = 0.0003 |

**On the tasks v89 already knew, v93 is 1–2 points worse than v89 and its
continued-training control (not significant at n = 628) and 3–4 points worse
than the two v91 grafted arms (significant).** The losses are concentrated in
the long-division and multi-step tasks — `acceleration` 0.476, `power` 0.667,
`two_step` 0.571, `average` 0.476 — while 20 of the 30 tasks are at or above
0.95. The likely cause is the training mix: 48% of every batch was new
material, at twice v91's learning rate.

### Eleven new task families (231 problems; v89 cannot answer these)

| task | acc | task | acc |
|---|---|---|---|
| impulse | 1.000 | code_list_count | 1.000 |
| ohms_current | 1.000 | code_neg_index | 1.000 |
| spring_energy | 1.000 | code_range_sum | 0.857 |
| permutations | 1.000 | cns_type_count | 0.191 |
| final_velocity | 1.000 | cns_side_count | 0.191 |
| | | cns_pair_synapses | 0.000 |

The five new science tasks and two of the three code tasks were learned to
1.000. The three connectome-knowledge tasks ("how many neurons of type DNa02
are in the male CNS?") were **not** learned: each of those facts was seen
about 0.2 times in training, far below what a recall task needs.

### The causal tests: are the brains used?

Dev loss on the same weights with parts of the graft switched off (4,000 dev
rows; > 0 means the model relies on it):

| ablation | cost, nats |
|---|---|
| every brain-to-trunk write gate and the thinking bond closed | −0.00005 |
| the commissural bonds between the two hemispheres masked | +0.000001 |
| left hemisphere only / right hemisphere only | −0.00004 / −0.00002 |
| everything that grew during training removed | **−0.00027** |

**Closing the brains changes nothing, and removing what grew slightly
helps.** The gates opened wide during training (per-site means 0.04–0.07, the
thinking-core bond 0.46; 65% of modules active), and the transformer trunk,
trained jointly, still routed around the branch. This repeats
[v91's finding](https://huggingface.co/Kai9987kai/supermix-v89) with three
write sites, a thinking-core bond and temporal recurrence added, and it
matches v91's own completed readout: A's gate-off copy answers all 630
problems identically, and the rewired null (B) scored above the real wiring
(A). Whether the branch is redundant with attention or merely re-absorbed by a
jointly trained trunk is the question a frozen-trunk experiment answers; that
has not been run.

### What neurogenesis did (8 events, `results/neurogenesis.jsonl`)

48 module splits (816 alive at the end: 421 left, 395 right), 256 synapses
opened (138 commissural) and 104 pruned, 48 afferent and 48 efferent taps, 32
experts born and 56 killed (each MoE layer reshaped from 64 to 58 experts —
v89's starved experts were recycled), vocabulary 8,679 → 9,415, one appended
identity layer. Every event is logged with the loss on a fixed witness batch
before and after; four of eight exceeded the 1e-3-nat flag (+0.052, +0.006,
+0.001, −0.029). None destabilised training, and a crash-resume replays them
bit-for-bit. The machinery works; with the branch unused, growing it could
not matter.

## Training

| | |
|---|---|
| parameters | 36,594,245 total / 9,127,349 active per token |
| warm start | v89 (24,250 steps), tokenizer extended by appending 736 ids |
| corpus | 1,546,091 rows: v89's 1,156,108 + 200,000 new solver-verified science + 60,000 execution-verified code + 90,000 connectome lookups + 30,000 connectome statements + 9,983 english-foundations rows; new rows weighted 3× |
| brains | `data/malecns_hemispheres_384.npz`: 768 modules, 16,373 edges (LL 5,391 / RR 5,336 / LR 2,826 / RL 2,820), Dale signs fixed, magnitudes learned, temporal recurrence with 2 inner updates per token |
| steps | 8,000 at sequence length 128, peak LR 3e-4 (grafts 3e-3), OneCycle, 17.3 h on CPU |
| growth | every 1,000 steps: 6 splits, 32 synapses (≥ half commissural), 6 + 6 taps, expert births above 2× fair share, apoptosis after two weak readings |
| selection | 205-problem accuracy probe every 1,000 steps — selected step 6,000 |
| final | train 0.046 · dev 0.058 · perplexity 1.06 |

Design, contracts and the pre-registered readout are in
`docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md`; the v91 experiment this
builds on, with its completed readout, in `docs/V91_MALECNS_CONNECTOME.md`.

## Using it

```bash
python example_usage.py "What current flows through 7 ohm at 315 V?"
```

`src/answer_check.py` re-derives the result from the question for 35
question shapes (code by running the snippet, connectome counts by lookup
when the data is present) and returns `None` when it cannot. **`None` means
*not checked*, never *correct*.** `src/step_audit.py` reports the first
written step that disagrees with exact arithmetic. `src/prompt_normaliser.py`
rewrites a naturally-typed question into the trained format and always says
what it sent. The checkpoint carries its own wiring buffers, so no connectome
data is needed to load or run it.

## Limitations

- **Not a chat model.** It emits derivations, not conversation, and never saw
  multi-turn context.
- **Forty-one task types and nothing else.**
- **The brains are decorative in this checkpoint** — every ablation above is
  within ±3e-4 nats of zero. Do not cite this model as evidence that fly wiring
  helps a language model; its own tests say it did not.
- **The connectome-knowledge tasks are at 0.0–0.19.** The type names tokenise
  and are produced; the facts were not learned.
- **Worse than v89 on `acceleration`, `average`, `power` and `two_step`.**
- **Answers are extracted as the last number in a reply.** Every score is a
  lower bound.

## Attribution

Male CNS connectome data: male-cns v1.0 (https://male-cns.janelia.org/),
FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research,
licensed CC BY 4.0. Cite: Berg S, Beckett IR, Costa M, Schlegel P, et al.
*Sexual dimorphism in the complete Drosophila male central nervous system
connectome.* Cell (2026). Changes made: neurons split by hemisphere
(`somaSide`, else `rootSide`; midline and unknown-side neurons dropped and
counted), 11,751 cell types clustered into 384 mirrored modules per side, a
1% input-fraction threshold on module edges, neurotransmitter signs assigned
per type, and the connectome-knowledge corpus derived from per-type neuron
counts and synapse counts. The data providers do not endorse this work.
