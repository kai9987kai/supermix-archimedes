# Supermix v91: wiring v89 like a male fly's central nervous system

**Status:** training (arms launched 2026-09-19 02:09). Results are appended
below the pre-registration; nothing above the "Results" heading is edited after
the first arm finishes.

## What v91 asks

Supermix v89 is a 30.2M-parameter MiMoMix (h320, 5 layers, 64 routed experts
top-2, recursive thinking core), trained 47 hours on this CPU to 0.919 on the
630-problem novel benchmark. v91 grafts onto it a recurrent branch whose
**wiring, Dale signs and initial strengths come from the Janelia male CNS v1.0
connectome** -- the whole-animal dataset the `natverse/malecns` R package reads
through neuPrint -- and continues training.

Two questions, answered by three matched arms:

| arm | run | what differs |
|---|---|---|
| A | `v91_cns_connectome` | ConnectomeCore with the real male-CNS module wiring |
| C | `v91_control` | no graft -- plain continued training of v89 |
| B | `v91_cns_rewired` | the same core with a degree-preserving rewired wiring |

* **A vs C:** does grafting a connectome-shaped recurrent branch help at all?
* **A vs B:** if it helps, is it the *fly's* wiring that helps, or any sparse
  signed recurrent graph with the same degrees, signs, self-loops and
  spectral radius?

## The data

`natverse/malecns` (GPL-3, cloned to `external/malecns`) is a thin R wrapper:
every data call goes to neuPrint and needs a personal token. The same dataset
is published as public CC-BY bulk files, which is what v91 uses:

| file | size | MD5 (base64) |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.5 MB | `UKdxh3DFciDxYLpPQxq4ng==` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.3 MB | `PYQrEv5cSe763lKNfdJKHw==` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.05 GB | `8w6dzKJc/QIb8eez2XVZng==` |

from `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
(download page: https://male-cns.janelia.org/download/). Stored in
`external/malecns_data/`, outside the repo.

### Cell-type graph (`source/malecns_connectome.py build`)

The 151,856,684-row body-to-body edge list is memory-mapped and streamed one
Arrow record batch at a time (~2 minutes, well under 1.5 GB RAM). Edges whose
two bodies are both typed neurons are summed per (pre type, post type).

| | |
|---|---|
| typed neurons | 164,506 |
| cell types | 11,751 |
| type -> type edges | 3,830,931 |
| synapses kept | 122,318,556 of 311,833,243 (39.2%) |

The other 60.8% of synapses touch untyped fragments, orphans or glia. They
are counted in the receipt, not silently dropped.

Neurotransmitter per type is the mode of the per-neuron `consensus_nt`,
falling back to `celltype_predicted_nt`. The sign convention follows the fly
whole-brain models: acetylcholine +1; GABA, glutamate (GluCl) and histamine
-1; dopamine, serotonin and octopamine +1 but flagged modulatory. By type:
5,983 ACh, 2,778 GABA, 2,429 Glu, 363 unclear, 86 5-HT, 52 OA, 44 DA,
16 histamine.

### Module graph (`source/malecns_connectome.py modules`)

11,751 nodes cannot be a per-token recurrence on this CPU, so types are
clustered into **512 modules** that never mix a Dale sign or a functional role:

* role = sensory / visual / central / vnc / ascending / descending / output,
  from the superclass;
* each (role, sign) group gets a module budget proportional to
  sqrt(types x synapse mass), then is split by k-means on a 32-dimensional
  spectral embedding of the log-weighted, symmetrised type graph.

Module edges survive if they carry **at least 1% of the postsynaptic module's
input**, the usual threshold for a "strong" connection in the FlyWire and
hemibrain analyses. Result: 10,509 edges (4.0% density), 467 self-loops,
294 excitatory and 218 inhibitory modules.

**The null (arm B), as amended before B started.** This is a Maslov-Sneppen
rewire that swaps the targets of edge pairs *only between source modules of
the same role and Dale sign*. Each edge's input fraction moves with its
**target**, because a fraction is a share of the postsynaptic module's input.
It makes 10 accepted swaps per edge, rejects duplicates and new self-loops,
and holds the self-loops fixed (`malecns_connectome.stratified_rewire`). What
each wiring preserves:

| | real (A) | **stratified null (B)** | unstratified null (first build, not run) |
|---|---|---|---|
| in/out degree, self-loops | -- | identical | identical |
| max per-module input-fraction sum | 0.99 | **0.99** | 1.95 (116 modules > 1) |
| modules with a changed count of inhibitory inputs | -- | **0** | 429 |
| role-block edge counts, L1 difference from real | -- | **0** | 9,152 |
| role modularity | 0.328 | 0.339 | -0.012 |
| linearised 6-step input-to-output gain (Frobenius) | 0.568 | 0.609 | 1.885 |
| reciprocity | 0.451 | 0.084 | 0.041 |
| clustering | 0.431 | 0.178 | 0.095 |
| sensory -> efferent hops (synaptic direction) | **2.81** | 2.34 | 2.38 |
| edges in the same place as real | -- | 21.7% | 10.7% |

The first build shipped the unstratified null. The v91 validity review showed
that it changed each module's input overload, its E/I input balance, the role
layering, and the input-to-output gain (3.3x). An A-vs-B difference could then
have come from any of those, so B was switched **before it started**. The
unstratified arrays are kept in the npz as `unstratified_rewired_*` for the
null-ensemble follow-up. The connectome arrays A trains on were copied
through byte-for-byte; that was checked, and matters because a crash-resume of
A re-reads the file.

The first receipt quoted sensory-to-output paths of "3.03 vs 2.60 hops". That
walked every edge backwards (fixed in `malecns_connectome.mean_hops`). In the
synaptic direction the real graph averages **2.81 hops against 2.34 for the
stratified null**, with 100% of pairs reachable. So the real CNS is more
layered and more reciprocal than a graph matched on everything above; that
fine structure is what A-vs-B tests.

A 64-module graph (912 edges) is also built. It is not used by any arm (see
"Considered and not built").

## The graft (`mimomix_core.ConnectomeCore`)

Per token, after block 2 of 5:

    u   = in_mask * W_in RMSNorm(x)
    r_k = (1 - a) r_{k-1} + a relu(W r_{k-1} + u + b)        k = 1..6
    x'  = x + g * W_out rms(out_mask * r_6)

* `W[post, pre] = mask * softplus(edge_logit) * sign[pre]`. Topology is fixed,
  Dale signs are fixed, magnitudes are learned. This is the
  connectome-constrained recipe of Lappalainen et al. (2024).
* `edge_logit` starts at the measured input fractions, scaled so the spectral
  radius of |W| is 0.9 **for each wiring separately**. The scale is 0.930 for
  the real graph and 1.048 for the stratified null. **The match is on |W|, not
  on the signed matrix the core runs.** The signed spectral radius is 0.547
  (real) against 0.690 (null), and the largest real eigenvalue part is 0.534
  against 0.670. So B starts slightly more recurrent. This is disclosed rather
  than fixed, because A had already started under the |W| rule. Its effect is
  measured post hoc from the trained weights (`v91_analysis.py weights`).
* **Input and output follow the fly.** Text enters through the 138 sensory,
  visual and ascending modules and is read out of the 34 descending and motor
  modules, so the signal has to cross the wiring.
* `g` is a per-channel gate initialised at exactly zero, so the grafted model
  is **bit-identical to v89 at step 0**. This is verified on the real
  checkpoint and in `test_v91_connectome_core.py`.
* The read-out is RMS-normalised. Six steps through the real wiring deliver a
  mean efferent rate of ~0.01 against a drive of ~0.36. Unnormalised, the
  zero gate's gradient would be ~30x too small to open. That is the failure
  mode `docs/V59_MECHANISM_CAUSALITY.md` documents for the thinking core.

The graft adds 591,488 parameters (+2.0%) and 10% step time: 4.72 against
4.28 s/step, measured back to back.

## Training

All arms use the identical v89 command (corpus, tokenizer, split, seed 57,
batch order), with these changes:

    --init_from output/v89_corpus/v89_corpus.pt --steps 2500 --lr 0.00015
    --new_param_lr_mult 20 --eval_every 625 --accuracy_every 2500

* **Peak LR 1.5e-4**, a tenth of v89's. v89 ended fully annealed; a 20-step
  pre-flight re-warmed to 3e-4 moved dev loss from 0.02477 to 0.0260.
* **Grafted parameters get 20x** (3e-3, v80's from-scratch rate at this
  scale). Gates, edge logits, biases and leaks are not weight-decayed: decay
  would pull every edge toward softplus(0) = 0.69, a strength nobody measured.
* **Fresh AdamW moments in every arm.** v89's `.pt` carries no optimiser
  state, so nothing is restored in any arm.
* **The control's optimiser is byte-identical to v89's layout** (one group).
  A and B add two groups for the graft and leave the other 872 tensors on the
  same curve.
* **Selection is the final step.** The accuracy probe runs only at step 2,500,
  so no arm is picked on a noisy mid-run probe.

Exact commands: `output/v91_malecns/run_chain.sh`.

## Pre-registered readout (written before any arm finished)

**Primary.** Each arm's final checkpoint is scored on the v86-v89 benchmark:
`eval_problem_solving.py --novel 630 --seen 0 --seed 65`, which has the same
generator fingerprint as v89's dump. Pairs are compared with exact McNemar
(`compare_problem_transcripts.py`), which re-scores every row and includes the
code tasks.

* **"The graft helps"** needs A > C at p < 0.05.
* **"The fly's wiring matters"** needs A > B at p < 0.05, **and** A's ablation
  cost larger than B's.
* **Anything else is reported as no difference.** No per-task p-value is a
  finding without a multiplicity correction: 30 tasks means Bonferroni
  p < 0.0017.

**Power, stated up front.** v88 against v89 had 51 discordant problems of
439. At that rate, p < 0.05 needs roughly a 33/17 split, a net 16 problems
(about 2.5 points). A 2,500-step continuation is unlikely to move a converged
model that far, so **a null primary result is the most likely outcome**, and
the secondary measures are there to say *why*.

**Secondary.**

1. **Mechanism.** `cns_core.ablation_cost_nats` is dev loss with the gate
   forced to 0, minus dev loss as trained. Above 0.001 nats (4% of v89's dev
   loss) the branch is load-bearing. Near 0, it trained and was not used.
2. **Dev loss at 625 / 1,250 / 1,875 / 2,500** for all arms. It is the same
   dev set and the same batch order, so a gap is attributable to the graft;
   the gap has no confidence interval and is reported as descriptive.
3. **Gate magnitude** (`gate_mean_abs`) and read-out activity per arm.

**Tripwire.** If any arm's dev loss at step 1,250 is above 0.030, the
learning rate is re-examined before the next arm starts.

### Amendment 1 -- 2026-09-19 02:46, before any arm had a result

This was written after an adversarial validity review and a literature sweep,
while arm A was training and before its first evaluation (step 625) had been
logged. It
tightens the rules above and never loosens them.

**What the literature predicts.** When the connectome's weights are *trained*,
real wiring's advantage over a degree-preserving null has repeatedly shrunk to
nothing:

* Goulas, Damicelli & Hilgetag 2021, *Neural Networks* 142:608: human,
  macaque and marmoset RNNs matched randomly wired ones.
* Damicelli et al. 2022, *PLOS Comput Biol* 18:e1010639.
* Dhiman 2026, arXiv:2604.04033: the flyvis connectome's advantage vanished
  under a shared initialisation and a degree-preserving null.
* Suárez et al. 2024, *Nat Commun* 15:656: the fruit fly was the one species
  whose connectome reservoir did *not* beat 500 rewired nulls (p = 0.11).
* The MaleCNS projects `nfly` and FLM report no advantage from fly anatomy.

Positive results sit in regimes v91 lacks: frozen reservoirs near
criticality, temporal memory, embodied sensorimotor control. **Prior:**
P(A > C at p < 0.05) about 5-10%; P(A > B) about 5%.

**Why power is low.** Only **67,213 of the 591,488 grafted parameters can
ever receive a gradient**, because the in/out masks and the edge mask zero the
rest. Exact McNemar at n = 630 needs a net of about 19 problems (3 points) for
p < 0.05, with about 55% power at a true +3 points.

**D0 -- validity gates, checked first.** A contrast is *invalid*, not null,
unless every arm involved meets all of these:

* 2,500 steps completed per `supervised_run.json` (chain.log's exit codes are
  wrong: `$?` there reports `date`);
* `initialised_from.missing_keys` holds exactly the `cns_core.*` keys (A, B)
  or nothing (C), and `unexpected_keys` is empty;
* the source hashes in `output/v91_malecns/hashes_*.txt` differ only where
  this amendment says they do. Two things changed after A started, both for
  B's null only:
  * `mimomix_core.py` `6d022c7b…` -> `3e1c7221…`: `load_graph` reads
    `rewired_fraction` on the rewired path; the connectome path is unchanged;
  * `malecns_modules_512.npz` `10397de4…` -> `46319e74…`: new null arrays,
    connectome arrays byte-identical.

  Nothing else may change before B finishes;
* `truncated_replies == 0` at the v89 dump's generation budget.

**D1 -- "the graft helps".** All of the following must hold:

* A > C by exact McNemar, Holm-corrected over {A-C, A-B} at family
  alpha 0.05;
* A's gate-zero ablation cost is >= 1e-3 nats, with a paired row-bootstrap
  95% CI above 0;
* on the 630 problems, A with the gate on is not worse than A with it off.

If A > C but the ablation cost is < 1e-4, the gain is training-path
divergence, not the branch. A non-significant result is reported as
**"inconclusive, minimum detectable effect about 3 points"**, never as "no
effect".

**D2 -- "the fly's wiring matters".** This needs A > B (Holm), and
cost_A > cost_B with a paired-bootstrap CI excluding 0. Even then the claim is
capped at: "the male-CNS module graph beats one degree-, sign-, role-,
input-fraction- and self-loop-matched random graph at matched rho(|W|)". A
fly-specific claim needs a null *ensemble*. If B > A significantly, it is
reported as consistent with B's larger signed gain until a signed-rho-matched
null resolves it.

**D3 -- reading the ablation** (together with `gate_mean_abs`):

| ablation cost | gate | reading |
|---|---|---|
| < 1e-4 | < 1e-2 | the gate never opened; an optimisation outcome, silent on the connectome |
| < 1e-4 | >= 1e-2 | open but ignored downstream |
| 1e-4 to 1e-3 | any | used, not load-bearing; descriptive only |
| >= 1e-3 | any | load-bearing; then mean-ablation decides whether it is computation or a learned constant offset (mean-ablation < 25% of zero-ablation => constant offset) |
| < -1e-4 | any | the branch hurts |

**E1 -- the post-hoc battery (no training; run first, serially after the
chain).** These tools are in `source/v91_analysis.py`:

* Per-row dev loss on all 11,230 dev rows for v89, A, B, C, A-off, A-mean,
  B-off and B-mean, with a paired bootstrap (10,000 resamples). Gate-on vs
  gate-off on the *same weights* carries no training-path noise.
* Exact match on the 630 problems for gate-zeroed copies of A and B,
  McNemar on vs off, and the difference of those differences.
* Weight diagnostics: gate size, edge drift from the connectome init, and
  signed spectral radius after training.

Arm-vs-v89 numbers are **descriptive only**. Every arm is v89 plus a re-warm
and 2,500 more steps, and re-warming on the same data first raises loss
(Gupta et al. 2023; Ibrahim et al. 2024).

**Known, disclosed confounds.** Both are part of "the graft" treatment:

* global gradient clipping couples the trunk's update to the graft's
  gradients;
* the ColEI (column-sign) Dale's form is known to learn worse than DANNs
  (Cornford et al. 2021; Li et al. NeurIPS 2023).

## Considered and not built

* **Connectome-routed MoE (expert-transition prior).** Map the 64 experts to
  the 64-module graph and bias layer l+1's router toward the modules that
  layer l's modules project to. Rejected for v91 for two reasons. Expert
  identities are not shared across layers, so one expert-to-module alignment is
  ill-posed; it would need a multi-layer quadratic assignment. And under
  v89's softmax top-2 with `norm_topk_prob`, a router prior gets task gradient
  only through the two chosen experts (4.6e-18 against 0.26 for unselected
  versus selected, measured by the code-mapping pass), so it can reweight a
  pair but barely promote a new expert.
* **A connectome-knowledge corpus** (Q&A about cell types, neurotransmitters,
  2-hop paths). `--init_from` requires a byte-identical vocabulary, and names
  like `DNa02` are new tokens. It needs vocabulary growth first.
* **Recurrence across positions** (the connectome as a state-space transition
  over the sequence). It would need recurrent state in the KV cache and
  speculative decoder; v91 keeps the core per token so decoding stays exact.

## Attribution

Male CNS connectome data: male-cns v1.0 (https://male-cns.janelia.org/), a
collaboration between FlyEM (HHMI Janelia), the University of Cambridge, the
MRC Laboratory of Molecular Biology and Google Research, licensed under CC BY
4.0. Cite: Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, Marin EC,
Nern A, Preibisch S, Qiu W, Takemura S, et al. *Sexual dimorphism in the
complete Drosophila male central nervous system connectome.* Cell (2026),
doi:10.1016/j.cell.2026.08.015. Volume and pages are from a search result and
not checked against the article. Preprint: bioRxiv (2025),
doi:10.1101/2025.10.09.680999.

Changes made to the data:

* the minconf-0.5 flat connectome was aggregated to the 11,751 cell types of
  the v1.0 annotation release (the paper reports 11,710; the release carries
  provisional types);
* synapses touching untyped bodies (60.8%) were excluded;
* neurotransmitter signs were assigned per type;
* types were clustered into 512 modules.

The data providers do not endorse this work. The `natverse/malecns` package
is by the natverse authors (GPL-3); v91 uses its documentation and dataset
conventions, not its code.

## Results

*(appended as arms finish; nothing above this heading has changed since
Amendment 1)*

### Arm A -- `v91_cns_connectome` (finished 2026-09-19 11:00)

**D0 validity gates: pass.**

* One supervised leg, completed, no restarts, 31,837 s of wall time. That
  includes about 2.7 h during which the laptop hibernated on low battery, from
  04:29 to 07:12; the process survived it.
* `missing_keys` is exactly the 13 `cns_core.*` tensors; `unexpected_keys` is
  empty.
* All five receipt checks pass.

| step | train loss | dev loss |
|---|---|---|
| 625 | 0.0226 | 0.0272 |
| 1,250 | 0.0240 | 0.0262 |
| 1,875 | 0.0242 | 0.0249 |
| 2,500 | 0.0231 | 0.0247 |

The step-625 bump is the re-warm (peak LR at step 250). The 1,250 tripwire
(0.030) was not tripped.

| | arm A | v89 |
|---|---|---|
| dev loss, gate as trained | 0.024657 | 0.02477 (its best) |
| dev loss, gate forced to 0 | 0.024651 | -- |
| **ablation cost** | **-0.000006 nats** | -- |
| `gate_mean_abs` / `gate_max_abs` | 0.0140 / 0.180 | -- |
| read-out activity / active fraction (last dev batch) | 0.052 / 0.40 | -- |
| tier-1 / tier-2 / tier-3 loss | 0.0116 / 0.1890 / 0.2511 | 0.0117 / 0.1868 / 0.2499 |
| in-training probe (n = 100, seed 957) | 0.92 | 0.94 (selected on it) |

**D3 reading: "open but ignored downstream".** The gate opened past the 1e-2
threshold, and its largest channel reached 0.18. But closing it moves dev loss
by -6e-6 nats, which is under the 1e-4 floor and in the wrong direction. After
2,500 steps the network does not rely on the connectome branch at this graft
site. The dev-loss improvement over v89 is not attributed to the connectome;
arm C measures how much of it is simply the continued training. The
paired-bootstrap interval on the ablation and the gate-off exact-match eval
come from the post-hoc battery.

### Arm C -- `v91_control` (finished 2026-09-19 15:13)

**D0 validity gates: pass.**

* One supervised leg, completed, no restarts, 15,187 s, with no sleep this
  time.
* `missing_keys` and `unexpected_keys` are both empty.
* One optimiser group of 872 tensors -- v89's layout.
* All five receipt checks pass.
* Source hashes are identical at C's start and end
  (`hashes_at_arm_C_*.txt`).

| step | A dev loss | C dev loss |
|---|---|---|
| 625 | 0.027167 | 0.027376 |
| 1,250 | 0.026202 | 0.026253 |
| 1,875 | 0.024911 | 0.024713 |
| 2,500 | **0.024657** | **0.024394** |

| | arm A | arm C | v89 |
|---|---|---|---|
| tier-1 / tier-2 / tier-3 loss | 0.011576 / 0.188954 / 0.251149 | 0.011486 / 0.189042 / 0.250820 | 0.011670 / 0.186808 / 0.249947 |
| in-training probe (n = 100) | 0.92 | 0.95 | 0.94 |

Descriptively: **the continued training accounts for the improvement over
v89, and the graft does not add to it.** C ends 0.00026 nats below A on dev
loss and three probe points above. Two facts fit together: A's zero ablation
cost, and C being ahead from step 1,875 on. The branch went unused, and its
presence only perturbed the trunk's trajectory, probably through the shared
gradient clip (a disclosed confound). The paired-bootstrap CI on A-vs-C dev
rows and the 630-problem McNemar test come from the post-hoc battery; under
D1, nothing here is yet a significance claim.

Both models are also below v89 on tier-2 and tier-3, the unseen responses
and unseen sentences. The continued training bought seen-distribution loss
and slightly spent generalisation loss, as re-warming on the same data is
known to do.

### Arm B -- `v91_cns_rewired` (finished 2026-09-19 19:24) and the primary readout (2026-09-21)

The post-hoc battery died after scoring arm A on 2026-09-19 and was
finished on 2026-09-21 (`output/v91_malecns/analysis.log`; the first
re-scoring of C and B on 2026-09-20 used a 41-task list and was set aside
as `*_replies.41tasks_misscoped.*`). Every receipt below carries the v89
generator fingerprint `3b99a446cd533be9bc5f8ae57d1310b4`.

**D0:** all three arms pass (2,500 steps, one leg each, receipt checks
green, `truncated_replies == 0`).

**Primary, 630 novel problems, exact McNemar on the 628 shared inputs:**

| arm | correct | acc | vs C (wins / losses, p) | vs v89 (wins / losses, p) |
|---|---|---|---|---|
| v89 | 577 | 0.919 | -- | -- |
| C control | 580 | 0.924 | -- | 12 / 9, p = 0.66 |
| A connectome | 587 | 0.935 | 10 / 3, **p = 0.092** | 15 / 5, p = 0.041 |
| B rewired null | 592 | 0.943 | 15 / 3, p = 0.0075 | 19 / 4, p = 0.0026 |

A vs B: 2 / 7 in B's favour, p = 0.18.

**D1 ("the graft helps"): not met.** A > C reaches p = 0.092, above the
Holm-corrected 0.025 for the first of {A-C, A-B}, and A's gate-off copy
answers the 630 problems *identically* (587 = 587; likewise B_off 592 =
592). On the 11,230 paired dev rows, closing A's gate costs +6e-6 nats
(CI [+1e-6, +1.4e-5]) and closing B's -1e-6: both branches are "open but
ignored downstream" (D3). Reported as **inconclusive, minimum detectable
effect about 3 points**.

**D2 ("the fly's wiring matters"): not met.** B, the degree-, sign-, role-
and input-fraction-matched random wiring, scored *above* A, and neither
branch carries any load, so the A-B gap is training-path divergence, which
is what the literature prior (Amendment 1) predicted.

**Dev rows (token-mean, 10,000-resample paired bootstrap):** C - A =
-0.000263 [-0.000410, -0.000124] (C better); B - A = -0.000165 (B better);
C - v89 = -0.000376 [-0.000596, -0.000163]; A - v89 = -0.000113 [-0.000348,
+0.000113]. The continued training, not the graft, is what moved.
