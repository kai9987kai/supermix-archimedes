# Supermix v93: two fly hemispheres, bonded to the trunk, with neurogenesis

**Status:** design fixed 2026-09-20 01:20 (before any code). The "Preflight"
and "Pre-registered readout" sections are filled in before the run starts;
"Results" is appended after. Nothing above "Results" is edited once training
has begun.

## What v91 found, and what v93 changes

v91 grafted one per-token ConnectomeCore (512 male-CNS modules, after block 2
of 5, zero-initialised gate) onto v89 and trained every parameter for 2,500
steps. The gate opened (`gate_mean_abs` 0.014, max 0.18) and the trunk then
ignored the branch: forcing the gate to zero moved dev loss by **-6e-6 nats**,
and the no-graft control finished 0.00026 nats *ahead*. The branch was
redundant with what the trunk already computes at that one site, and the
trunk could route around it (docs/V91_MALECNS_CONNECTOME.md, "Results").

v93 changes four things at once, so it is an engineering release with a
pre-registered readout, not a single-factor experiment:

1. **Two fly brains.** The male CNS is bilateral: 163,718 of the 164,506 typed
   neurons carry a soma side (L 80,786 / R 82,932 after the `rootSide`
   fallback for sensory neurons; 375 midline `M` and 413 unknown are dropped
   and counted). Each hemisphere becomes its own module graph, and the real
   **commissural** type-to-type synapses (10.6% of L/R synapse mass in each
   direction) become the bonds between the two brains. Both live in one
   block-structured core:

       W = [[W_LL, W_LR],
            [W_RL, W_RR]]         (W[post, pre], Dale sign per pre column)

2. **More connections between the brains and the trunk.** The brains read
   from several trunk layers and write into several, and also into the
   recursive thinking core: read taps at blocks {0, 1, 2}, write sites after
   blocks {2, 3, 4} plus the thinking-core input. Every write site has its own
   zero-initialised gate, so the grafted model is bit-identical to v89 at
   step 0 (the v91 contract, kept).
3. **Neurogenesis.** The whole model can grow, prune and rewire *during*
   training: new tokens (vocabulary), new experts (MoE), new modules and new
   synapses in the brains (including new commissural bonds and new
   brain-to-trunk taps), and an appended identity layer. Growth is slot-based:
   every tensor is allocated at its capacity up front and an `alive` mask
   says which slots are in use, so no parameter changes shape mid-run and
   the optimiser, scheduler, checkpoint, resume and strict-load machinery are
   untouched. Every event is function-preserving by construction or its
   effect is measured on a fixed witness batch and logged.
4. **A bigger corpus.** Five new solver-verified science tasks, three new
   execution-verified code tasks, a connectome-knowledge family generated
   from the CC-BY male-CNS data, and 9,983 english-foundations rows. The
   vocabulary grows by appending tokens after v89's 8,679 ids (never
   rebuilt), which is what makes a warm start from v89 legal.

## Design contracts (what the implementers build against)

### D1. Hemisphere data (`source/malecns_connectome.py`, new command `hemispheres`)

* Node = (type, side); side = `somaSide` if in {L, R} else `rootSide` if in
  {L, R}; other typed bodies are dropped and counted in the receipt.
* One streaming pass over the 1.05 GB edge file with the existing
  `aggregate_edges` over 22,871 sided nodes; output
  `datasets/v93_malecns/malecns_sided_types.npz` (node_type, node_side,
  node_n_neurons, per-type superclass / class / nt / sign copied from
  `malecns_types.npz` **by type name**, pre / post / weight over sided nodes).
* Modules: the **mirror** scheme. Cluster the type graph once into N modules
  (the existing `cluster_modules`, same role/sign strata), then
  `module_of_node = module_of_type[type] + N * side`. Left module i and right
  module i are homologous.
* Edges: `module_matrix` over the 2N id space, `threshold_input_fraction` at
  1% of **total** (ipsi + contra) input, then sliced by block. Arrays in
  `datasets/v93_malecns/malecns_hemispheres_{N}.npz`: everything
  `ConnectomeCore.load_graph` reads today (`module_sign`, `module_role`,
  `edge_post`, `edge_pre`, `edge_fraction`, `rewired_*`), plus `module_side`
  (int8), `module_label`, `homolog`, `edge_block` (0 LL, 1 RR, 2 LR, 3 RL),
  and `null_kind`.
* Null: `stratified_rewire` gains a keyword-only `edge_strata` so an edge's
  stratum is (side_pre, role_pre, sign_pre, side_post): LL swaps with LL,
  LR with LR. Existing positional callers are unchanged.
* Diagnostics per block (edge counts, self-loops, density), side modularity,
  homolog edge symmetry, commissural input share per module, per-side and
  joint spectral radii.
* Built for N in {256, 384} per side; the preflight picks one on step time.
* `datasets/v91_malecns/*` is never modified (v91/v92 hash their files).

### D2. The core (`mimomix_core.ConnectomeCore`, kept under attribute `cns_core`)

The attribute name, the parameter prefix `cns_core.`, the v91 state-dict keys
and `load_graph`'s behaviour on a v91 npz are unchanged, so v91 checkpoints
still load and `test_v91_connectome_core.py` still passes. New, default-off:

* `hemisphere` buffer (n,) int8 from `module_side` (all 0 for a v91 graph).
* **Capacity slots.** `cns_nodes` is the capacity; `cns_spare_nodes` of them
  start dead. Buffers `alive` (n,) uint8, `born_step` (n,) int32, `edge_grown`
  (n, n) uint8. A dead slot has mask row and column 0, in/out masks 0,
  `read_in` row 0 and `read_out` column 0, so it is exactly inert.
* **Multi-site taps.** `cns_read_layers` (tuple) and `cns_write_layers`
  (tuple). The primary read/write site keeps `read_in` / `read_out` / `gate`;
  additional sites live in `extra_read_in` (ModuleList of Linear), and
  `extra_read_out` / `extra_gates` (ModuleList / ParameterList). Drive is the
  sum over read taps of `in_mask * read_in_i(norm_i(h_i))`; the core runs
  once after the deepest read layer; each write site adds
  `gate_j * read_out_j(rms(out_mask * r))` after block j. All gates start at
  zero.
* **Thinking-core bond.** `cns_to_thinking`: `gate_t * to_thinking(read)` is
  added to the thinking core's input (per position, zero gate).
* **Temporal recurrence** (`cns_temporal`): the rate vector is carried along
  the sequence as a causal scan with `cns_steps` inner updates per position.
  In generation the per-position state history rides in `past_key_values`
  as one extra rank-3 entry appended after the layer entries, so `trim_past`
  and speculative rollback work unchanged and cached decode equals the full
  forward. Whether v93 trains temporal or per-token is decided by the
  preflight timing (see "Preflight").
* Growth primitives on the module (used by the neurogenesis controller):
  `split_module(parent, step)`, `grow_edge(post, pre, step, logit)`,
  `open_tap(module, kind)`, `prune_edges(threshold)`, `kill_module(i)`.
  `reset_special_parameters` is never called after growth.
* Statistics for growth decisions are accumulated **only in eval mode**
  when `collect_stats` is on: mean rate per module, rate covariance
  (`E[r_pre r_post] - E[r_pre] E[r_post]`) over dev tokens.

### D3. MoE slots (`mimomix_core.SparseMoEFeedForward`)

`moe_spare_experts` extra expert slots per MoE layer; persistent `expert_alive`
buffer; dead logits are set to -inf before the softmax so selection, gate
weights and the balance loss (over alive experts) are exact. Birth copies the
highest-load alive expert (+ small noise), its router row and its bias.
Death masks the slot and resets it. With `moe_spare_experts = 0` the module
is byte-identical to v89's.

### D4. Depth growth

`--grow_layers K` appends K blocks at the end at construction, pins
`global_layers` so the existing layout does not re-derive, and after the
warm-start load zeroes each new block's `o_proj` and every `down_proj` so the
block is exactly the identity. v93 uses one dense block.

### D5. Vocabulary neurogenesis

`WordTokenizer.extend(base, texts, max_new, min_count)` copies the base token
list as a prefix and appends new (token, lstripped) pairs in frequency order.
`load_initial_weights` accepts a checkpoint whose token list is a prefix of
the live one (same `digit_tokens` / `reverse_digits`), grows the single tied
embedding matrix by copying the old rows and initialising new rows as
`mean(old) + 0.1 * std(old) * N(0, 1)` with a recorded seed, and still raises
`different vocabulary` for a non-prefix list. Everything else keeps the
identical-shape rule.

### D6. The neurogenesis controller (`source/neurogenesis.py`)

Runs inside the trainer's eval branch, after the dev eval and before the
checkpoint write, every `--grow_every` steps (a multiple of `--eval_every`),
from statistics gathered during that dev pass. Per event, in this order:

1. **Apoptosis.** Grown edges whose strength fell below `1e-4` for two
   consecutive events are masked off; experts with dev load below
   `0.1 / n_alive` for two consecutive events are killed. Real connectome
   edges are pruned only when their strength fell below `1e-4` (logged
   separately as `real_edges_pruned`).
2. **Module mitosis.** The `--grow_modules` alive modules with the highest
   `mean_rate * (1 + out_share)` are split into free slots. The child copies
   the parent's incoming edges, tap masks, sign, side, leak and bias;
   parent and child each keep **half** the parent's outgoing strengths and
   half the parent's `read_out` column, and the child copies the parent's
   `read_in` row; the read-out RMS count changes by one, and the measured
   witness-batch loss delta is logged.
3. **Synaptogenesis.** The `--grow_edges` unconnected, Dale-legal pairs with
   the largest rate covariance are opened at logit -7 (strength 9e-4);
   at least half the quota is reserved for cross-hemisphere pairs
   (bonding the brains further), and at most `--grow_edges` in total.
4. **New taps.** `--grow_taps` modules gain an afferent tap (in_mask 1 with a
   zeroed `read_in` row) and `--grow_taps` gain an efferent tap (out_mask 1
   with a zeroed `read_out` column), chosen by rate covariance with the
   trunk residual norm; exact at birth.
5. **Expert birth.** For each MoE layer whose highest alive load exceeds
   `2 / n_alive`, one spare slot is born from that expert.

AdamW moments for every slice an event wrote are zeroed in place (shapes do
not change). Every event is appended to `output/<run>/neurogenesis.jsonl`
with before/after counts and the witness-batch loss before and after, and
summarised in the receipt under `neurogenesis`.

### D7. Corpus (`source/build_v93_corpus.py`)

| family | tasks | rows | verifier |
|---|---|---|---|
| v89 omni (12), scratchpad (10), code (9) | unchanged generators | as v89 | as v89 |
| new omni | impulse, ohms_current, spring_energy, permutations, final_velocity | 40,000 each | `nexus_solver` / `science_plan` |
| new code | code_range_sum, code_list_count, code_neg_index | 20,000 each | execution |
| connectome knowledge | cns_type_count, cns_side_count, cns_pair_synapses | 30,000 each | lookup in the CC-BY data |
| connectome language | neurotransmitter / superclass / partner statements | 30,000 | lookup (not benchmarked) |
| dialogue (v86 rows) | -- | 96,108 | -- |
| english foundations (v62) | -- | 9,983 | -- |

Every task keeps the `... total N` reply convention, a prompt under 32
tokens and a reply under 96 tokens (measured with `--token_budget_report`),
and the existing generators are byte-unchanged so the 30-task v89 benchmark
fingerprint `3b99a446cd533be9bc5f8ae57d1310b4` still reproduces through
`eval_problem_solving.py --task_set v89`.

### D8. Training

Warm start from `output/v89_corpus/v89_corpus.pt` through
`train_supervised.py`, extended vocabulary, hemisphere core with the taps
above, spare slots, neurogenesis every 500 steps, one appended dense layer.
Step budget is set on **tokens**: the v93 corpus's supervised tokens per row
are measured and steps are scaled so each v89 task keeps at least v91's
exposure while new families get enough of their own. Exact command and
numbers are in "Preflight" below.

## Preflight (2026-09-20, before launch)

**Data.** `datasets/v93_malecns/malecns_hemispheres_384.npz`: 384 modules
per side, 16,373 edges (LL 5,391 / RR 5,336 / LR 2,826 / RL 2,820 -- 34% of
the edges are commissural), 669 self-loops, density 2.8%, no empty module on
either side, homolog edge symmetry 0.935, |W| spectral radius L 0.924 / R
0.926 / joint 0.962 (scaled to 0.9 at load), 104 afferent and 27 efferent
modules per side. The 256-per-side file (10,553 edges) was built as well and
not used. The mirror clustering at 512 reproduces v91's partition exactly.
Sided build: 163,718 of 164,506 typed neurons (rootSide decided 16,551
sensory neurons), 375 midline + 413 unknown dropped (1.5% of typed-typed
synapses); the whole build takes 3 minutes cold.

**Corpus.** `datasets/v93/v93_combined.jsonl`: 1,546,091 rows = v89's
1,156,108 + 200,000 new science + 60,000 new code + 90,000 connectome lookups
+ 30,000 connectome statements + 9,983 english-foundations rows. Every new
task: prompt <= 32 tokens, reply p95 <= 85, nothing dropped by packing. New
vocabulary: 736 ids appended to v89's 8,679 (the connectome type names are
most of it). Training mix `v93_train_mix.jsonl` duplicates every new row
three times (2,326,057 rows, new families 48% of a batch): at 8,000 steps a
new science task is drawn ~6,600 times, half of what v89 needed per task
from scratch, on a warm-started model.

**Core cost, idle box, trainer-equivalent step (forward + backward + clip +
AdamW, 16 x 128):** no core 5.1-5.5 s; temporal 2 inner steps 6.6 s;
temporal 1 step 5.9 s; per-token 6 steps 6.6 s. **Temporal at 2 inner steps
is used** -- it costs the same as v91's per-token core and gives the
brains what v91's core lacked, state across positions (the v92 rationale).

**The 40-step preflight** (`output/v93_preflight/`, the production command
at `--steps 40 --eval_every 20 --grow_every 20`): warm start grew the
vocabulary 8,679 -> 9,415, appended one identity block, padded 8 spare
expert slots per layer; 36,594,245 parameters (9,127,349 active per token).
Two growth events ran at the real shape in 2.7 s each: 12 splits, 64
synapses (35 commissural), 24 taps, 8 expert births, witness deltas -4e-5
and +3e-5 nats. The first attempt failed at once on `--cns_write_layers 5`
(validated before the block is appended), so the write sites are blocks 2-4.
The uncapped apoptosis rule culled 72 starved experts at the second event;
the run below caps deaths at 2 per layer per event. Speculative decoding
matched greedy; all five receipt checks passed. The v89 tasks were still at
1.0 on 27 of 30 probe tasks after 40 steps at the new learning rate.

**The run.** `output/v93_neurogenesis/run_v93.sh`: 8,000 steps, peak LR 3e-4
(trunk) and 3e-3 (grafts and the new block, x10), OneCycle, eval / probe /
growth every 1,000 steps, dev 0.003 of the mix (5,441 rows), 205-problem
probe (5 per task over all 41), selection on the probe, ablations over
4,000 dev rows. Expected ~18 h: 6.6 s x 8,000 plus ~16 min of dev + probe
per interval. Tripwires, checked at step 1,000 and 2,000 from the log: any
v89 task family falling to 0 on the probe, or train loss rising for two
consecutive evals, stops the run for a learning-rate review.

## Pre-registered readout

**Primary (v89-comparable).** `eval_problem_solving.py --task_set v89 --novel
630 --seen 0 --seed 65` on the final checkpoint, paired with v89's dump
(`output/v87_measurements/v89_replies.jsonl`) and with v91 arm C
(`output/v91_malecns/C_replies.jsonl`, the continued-training control) by
exact McNemar. v93 has more steps, more data and more parameters than C, so a
win over C is "the v93 package helps on the v89 tasks", not a wiring claim.

**New families.** The same eval over the new tasks (`--task_set new`) reports
per-task accuracy with Wilson intervals; there is no baseline (v89 cannot
answer them), so the number is descriptive.

**Mechanism (the causal tests).**

* gate-off ablation of *all* write sites and the thinking bond
  (`ablation_cost_nats`), read as v91's D3 table;
* **commissure ablation**: mask the LR and RL blocks only -- the cost of the
  bonds between the two brains;
* **hemisphere ablation**: each side alone;
* **neurogenesis ablation**: every grown slot, edge and tap returned to its
  birth state (dead / masked), so the cost of what grew is measured
  separately from the cost of the wiring that was there at step 0.

Each ablation is a dev-loss difference on the same weights with a paired
row bootstrap, exactly as in v91's E1.

**Neurogenesis log.** Counts of modules, edges (by block), taps and experts
born and killed per event, with the witness-batch loss before and after each
event. An event whose witness delta exceeds 1e-3 nats is flagged.

**What would count as failure.** Ablation cost of the whole graft below
1e-4 nats again, or a v89-task accuracy below v91 C's 0.935 minus the McNemar
margin. Either is reported as such.

## Attribution

Male CNS connectome data: male-cns v1.0 (https://male-cns.janelia.org/),
FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research,
CC BY 4.0; Berg et al., *Sexual dimorphism in the complete Drosophila male
central nervous system connectome*, Cell (2026). Changes made: hemispheres
split by `somaSide` / `rootSide`, midline and unknown-side neurons dropped,
types clustered into modules, and the connectome-knowledge corpus derived
from per-type neuron counts, synapse counts and neurotransmitter
predictions. The data providers do not endorse this work.

## Results

*(appended after the run)*

### The run (2026-09-20 13:25 -> 2026-09-21 08:59, one leg, no restarts)

17.3 h for 8,000 steps including eight evals, probes and growth events
(6,600-6,800 s per 1,000 steps; the last interval 20,000 s because the
end-of-run tiers over 66,500 rows and five ablations are inside it).
Selected step 6,000 on the 205-problem probe (0.898 over 41 tasks; 0.96 on
the 30 v89 tasks alone). Dev loss 0.151 -> 0.102 -> 0.094 -> 0.078 ->
0.071 -> 0.062 -> 0.058 -> 0.058. All five receipt checks pass, including
speculative == greedy with the temporal core.

### Primary readout -- v89 tasks, 630 novel problems, fingerprint 3b99a446

| model | correct | acc | v93 vs it: wins / losses, exact McNemar p |
|---|---|---|---|
| **v93** (selected step 6,000) | **571** | **0.906** | -- |
| v89 | 577 | 0.919 | 17 / 25, p = 0.28 |
| v91 C (continued training) | 580 | 0.924 | 16 / 27, p = 0.13 |
| v91 A (connectome graft) | 587 | 0.935 | 10 / 28, p = 0.005 |
| v91 B (rewired null) | 592 | 0.943 | 8 / 31, p = 0.0003 |

v93 is 1-2 points below v89 and C (not significant, n = 628) and 3-4 points
below the two v91 grafted arms (significant). The losses sit in four
families: `acceleration` 0.476 (A: 0.762), `power` 0.667 (0.857), `two_step`
0.571 (0.619) and `average` 0.476 (0.524) -- the long-division and
multi-step tasks -- while 20 of 30 tasks stay at or above 0.95. The
pre-registered accuracy criterion (below C by more than the McNemar
margin) is **not** met at p = 0.13, but the direction is a regression, and
the most likely cause is the data mix: 48% of every batch was new material
at a peak LR twice v91's, and the hardest old formats paid for it.

### New families -- 231 problems, fingerprint c1966c74 (no baseline exists)

| task | acc | task | acc |
|---|---|---|---|
| impulse | 1.000 | code_list_count | 1.000 |
| ohms_current | 1.000 | code_neg_index | 1.000 |
| spring_energy | 1.000 | code_range_sum | 0.857 |
| permutations | 1.000 | cns_type_count | 0.191 |
| final_velocity | 1.000 | cns_side_count | 0.191 |
| -- | -- | cns_pair_synapses | 0.000 |

The five new science tasks and two of three code tasks are solved from
~6,600 draws each on the warm-started model. The connectome lookups were
not learned: ~5,000 draws over 30,000 rows of four-digit facts about
thousands of cell types is far below the repetition a recall task needs
(docs/V81: 712 distinct x 56 repetitions reached 0.93; here each fact was
seen ~0.2 times). The vocabulary growth itself worked -- every type name
tokenises and is produced -- the exposure did not.

### Mechanism -- the causal tests (4,000 dev rows, same weights)

| ablation | cost, nats | reading |
|---|---|---|
| all write gates + thinking bond off | -0.000053 | the brains are not load-bearing |
| commissural blocks (LR, RL) off | +0.000001 | the bonds between the brains carry nothing |
| left hemisphere only | -0.000038 | -- |
| right hemisphere only | -0.000023 | -- |
| everything that grew, off | **-0.000272** | the grown structure slightly *hurts* |

Full-dev gate-off cost +0.000032 (v91's D3 floor is 1e-4). **This is the
pre-registered failure on the mechanism: v93 repeats v91's finding with
three write sites, a thinking-core bond and temporal recurrence added.**
The gates opened far wider than v91's (site means 0.035 / 0.039 / 0.069,
maxima 0.42-0.52; the thinking bond's gate mean 0.46; read-out activity
39, 65% of modules active) and the trunk still routes around the branch --
"open but ignored downstream", now at every site. Two readings are
consistent with this and with v91: the recurrent branch is redundant with
what attention plus the MoE already compute on these templated tasks, and
the trunk, being trained jointly, can always re-absorb the branch's
contribution. The frozen-trunk rig v92 was written for is the experiment
that separates them; it has still not been run.

### Neurogenesis -- what grew (8 events, `neurogenesis.jsonl`)

48 module splits (final 816 alive: 421 left, 395 right), 256 synapses
opened (138 commissural) and 104 pruned (92 grown, 12 real), 48 afferent
and 48 efferent taps, 32 experts born and 56 killed (each MoE layer went
64 -> 58: the starved experts v89 carried were recycled, and their slots
did not all refill). Alive edges 16,373 -> 22,131, mostly through splits
inheriting their parent's connections. Four events were flagged: witness
deltas +0.052 (step 1,000), +0.006, +0.001 and **-0.029** (step 7,000 --
an event that *improved* the witness batch); the rest were within
+/-1e-3. The machinery is sound -- every event was replayable, the crash
resume path is bit-identical (tested), and no event destabilised
training -- but with the branch unused, growing it could not matter, and
the `grown_off` ablation says the grown MoE/brain structure cost 2.7e-4
nats net.

### Verdict

Engineering: delivered and verified (two-hemisphere core, multi-site
bonds, temporal cache, slot-based growth of tokens / experts / modules /
synapses / taps / depth, extended vocabulary, corpus of 41 tasks). Science:
the brains are still not used by the trunk (all ablations ~0), the v89
tasks slipped 1-2 points, the new science and code families were learned
to ~1.0, and the connectome-knowledge family was not. The next experiment
this points at is the one v92 pre-registered: freeze the trunk so the loss
has to go through the wiring, and compare real against rewired there.
