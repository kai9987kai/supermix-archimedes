"""Generate a connectome-knowledge corpus from the CC-BY male CNS data.

## Why this exists

v93 grafts two hemispheres of the male *Drosophila* central nervous system onto
the trunk as module graphs (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md). Every
other family in the corpus teaches a procedure the model can generalise; this
one teaches **facts about the very data the graft is built from**: how many
neurons a cell type has, how many sit on each side, and how many synapses one
type makes onto another. The answers are integers, so the `... total N` reply
convention and the benchmark's last-number scoring apply unchanged, and every
row is verified by looking it up again in the source arrays rather than by
trusting the generator.

## Where the numbers come from

`datasets/v91_malecns/malecns_types.npz` (type_names, n_neurons, superclass,
nt, pre/post/weight) is the per-type table v91 derived from the FlyEM male-cns
v1.0 flat connectome at minconf 0.5. Per-side counts follow D1 of the v93
design: a neuron's side is `somaSide` when that is L or R, else `rootSide`
when that is L or R, and midline (M) or unknown-side neurons are dropped and
counted. They are read from `datasets/v93_malecns/malecns_sided_types.npz`
when the hemisphere build has produced it, and derived directly from the
annotations feather otherwise; the receipt records which. Measured on the
annotations file: 164,506 typed neurons, L 80,786 / R 82,932, 375 midline and
413 unknown -- the numbers in the design document.

## The population, and why it is capped

A knowledge task is learned by repetition, not by coverage.
V81_WHAT_THE_MODEL_CAN_LEARN.md measured 712 distinct problems repeated 56x at
0.93 against 24,000 unique ones at 1.7x scoring 0.03, and a fact has to be
memorised outright where a procedure only has to be followed. There are 11,215
types with at least two neurons and a plain name; 30,000 rows over all of them
would show each fact 2.7 times, and 47% of those answers would be `2`, so a
model that always said `total 2` would score 0.47 without knowing anything.

The population is therefore the `MAX_TYPES` largest of those types by neuron
count (ties by name), which at the default 1,000 covers 77.3% of typed neurons,
shows each type-count fact ~30 times at 30,000 rows, has a smallest count of 11
and a majority-class share of 14%. Pairs are the `MAX_PAIRS` strongest
type-to-type connections within that population at or above `MIN_PAIR_WEIGHT`
synapses, so every answer is real signal rather than a one-synapse
coincidence. Both caps are CLI flags and the receipt reports the coverage they
give. The benchmark for these tasks draws from the same population -- that is
recall, and `NON_CLAIMS` says so.

## The language family

A fourth kind of row carries no `task` key: statements about a type's
predicted neurotransmitter, its superclass and its strongest downstream
partner, in plain prose. They are not benchmarked. They exist so the new
vocabulary (type names, neurotransmitters, superclasses) is seen in sentences
rather than only inside a `total N` template, which is what the dialogue rows
do for the rest of the corpus.

## Attribution

Male CNS connectome data: male-cns v1.0 (https://male-cns.janelia.org/), FlyEM
(HHMI Janelia), University of Cambridge, MRC LMB and Google Research, CC BY
4.0; Berg et al., *Sexual dimorphism in the complete Drosophila male central
nervous system connectome*, Cell (2026). Changes made: per-type neuron
counts, per-side counts and type-to-type synapse counts were derived from the
flat connectome, and the questions and answers here were generated from them.
The data providers do not endorse this work.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from build_omni_corpus import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    extract_answer,
    token_budget_report,
)

#: The per-type table v91 derived (never modified here; read only).
DEFAULT_TYPES = ROOT / "datasets" / "v91_malecns" / "malecns_types.npz"

#: The sided node table the v93 hemisphere build writes (D1). Optional.
DEFAULT_SIDED = ROOT / "datasets" / "v93_malecns" / "malecns_sided_types.npz"

#: The CC-BY annotations file, used for sides when the sided table is absent.
DEFAULT_ANNOTATIONS = (
    ROOT.parent / "external" / "malecns_data"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

#: Population caps. See the module docstring for the measurement behind them.
MAX_TYPES = 1000
MAX_PAIRS = 2000
MIN_PAIR_WEIGHT = 50

#: A type must have at least this many neurons to be asked about.
MIN_NEURONS = 2

#: A "reasonable" type name: letters, digits, underscore and hyphen, starting
#: with a letter, at most 16 characters. Names with spaces, commas or dots
#: (`DNp51,DNpe019`, `Acc. ti flexor MN`: 218 of 11,751) are merged labels
#: rather than types and are left out.
NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{0,15}")

DOMAIN = "connectome"

#: Plain words for the superclass codes in the annotations.
SUPERCLASS_WORDS: Dict[str, str] = {
    "cb_intrinsic": "central brain intrinsic",
    "vnc_intrinsic": "ventral nerve cord intrinsic",
    "ascending_neuron": "ascending",
    "descending_neuron": "descending",
    "visual_projection": "visual projection",
    "visual_centrifugal": "visual centrifugal",
    "ol_intrinsic": "optic lobe intrinsic",
    "ol_sensory": "optic lobe sensory",
    "cb_sensory": "central brain sensory",
    "vnc_sensory": "ventral nerve cord sensory",
    "cb_motor": "central brain motor",
    "vnc_motor": "ventral nerve cord motor",
    "cb_efferent": "central brain efferent",
    "vnc_efferent": "ventral nerve cord efferent",
    "cb_endocrine": "central brain endocrine",
    "vnc_endocrine": "ventral nerve cord endocrine",
    "sensory_ascending": "sensory ascending",
    "sensory_descending": "sensory descending",
    "efferent_ascending": "efferent ascending",
    "efferent_descending": "efferent descending",
}


def superclass_words(code: str) -> str:
    """The prose form of a superclass code; unknown codes are spelled out."""

    if code in SUPERCLASS_WORDS:
        return SUPERCLASS_WORDS[code]
    return (code.replace("cb_", "central brain ").replace("vnc_", "ventral nerve cord ")
            .replace("ol_", "optic lobe ").replace("_", " "))


# -- the population ----------------------------------------------------------


@dataclass
class Population:
    """Everything a generator can ask about, fixed before any row is drawn."""

    names: List[str]
    n_neurons: Dict[str, int]
    left: Dict[str, int]
    right: Dict[str, int]
    nt: Dict[str, str]
    superclass: Dict[str, str]
    #: (pre, post, weight), strongest first.
    pairs: List[Tuple[str, str, int]]
    #: Strongest downstream partner within the population, per type.
    partner: Dict[str, Tuple[str, int]]
    receipt: Dict[str, Any] = field(default_factory=dict)

    @property
    def nt_names(self) -> List[str]:
        return [n for n in self.names if self.nt[n] != "unclear"]

    @property
    def partner_names(self) -> List[str]:
        return [n for n in self.names if n in self.partner]


def load_type_table(path: Path = DEFAULT_TYPES) -> Dict[str, Any]:
    """The v91 per-type arrays, as plain Python containers."""

    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


#: How `malecns_connectome.build_sided_nodes` encodes `node_side`: L is 0 so
#: the left hemisphere holds module ids 0..N-1 (its `SIDE_NAMES`).
SIDE_NAMES = ("L", "R")


def side_counts_from_sided_table(path: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
    """Per-type L/R neuron counts from the D1 sided node table.

    The table stores `node_type` as an int32 index into its own `type_names`
    (copied from `malecns_types.npz` by name) and `node_side` as int8 0/1 in
    `SIDE_NAMES` order, not as strings. The first version of this reader
    compared `str(side)` with ``"L"``, matched nothing, and returned every
    per-side count as 0 -- 1,000 of 1,000 population types "had zero on one
    side" and the v93 audit's 300 side-count draws all answered ``total 0``.
    A table that yields no sided neurons at all is refused rather than
    returned, so that shape of misread cannot ship again silently.
    """

    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        type_names = [str(t) for t in data["type_names"]]
        node_type = data["node_type"].astype(int)
        node_side = data["node_side"].astype(int)
        node_n = data["node_n_neurons"].astype(int)
    left: Dict[str, int] = {}
    right: Dict[str, int] = {}
    for type_index, side, count in zip(node_type, node_side, node_n):
        name = type_names[int(type_index)]
        side_name = SIDE_NAMES[int(side)] if 0 <= int(side) < len(SIDE_NAMES) else None
        target = left if side_name == "L" else right if side_name == "R" else None
        if target is not None:
            target[name] = target.get(name, 0) + int(count)
    if not left or not right:
        raise ValueError(
            f"{path} yields no neurons on one side (L {sum(left.values())}, "
            f"R {sum(right.values())}); the table is not in the D1 layout this reads"
        )
    receipt = {
        "source": str(path),
        "kind": "malecns_sided_types.npz (D1 hemisphere build)",
        "sided_nodes": int(len(node_type)),
        "neurons_left": int(sum(left.values())),
        "neurons_right": int(sum(right.values())),
    }
    return left, right, receipt


def side_counts_from_annotations(path: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Any]]:
    """Per-type L/R neuron counts derived from the annotations feather.

    The D1 rule exactly: `somaSide` if in {L, R}, else `rootSide` if in
    {L, R}; anything else is dropped and counted. This is the fallback for a
    checkout where the hemisphere build has not run yet, and it reads the
    14 MB annotations file only (never the 1 GB edge file).
    """

    import pyarrow.feather as feather

    table = feather.read_table(path, columns=["type", "somaSide", "rootSide"])
    types = table.column("type").to_pylist()
    soma = table.column("somaSide").to_pylist()
    root = table.column("rootSide").to_pylist()
    left: Dict[str, int] = {}
    right: Dict[str, int] = {}
    dropped = Counter()
    typed = 0
    for name, s, r in zip(types, soma, root):
        if not name:
            continue
        typed += 1
        side = s if s in ("L", "R") else r if r in ("L", "R") else None
        if side is None:
            dropped["midline" if s == "M" else "unknown"] += 1
            continue
        target = left if side == "L" else right
        target[name] = target.get(name, 0) + 1
    receipt = {
        "source": str(path),
        "kind": "annotations feather, somaSide-else-rootSide (D1 rule)",
        "typed_neurons": typed,
        "neurons_left": int(sum(left.values())),
        "neurons_right": int(sum(right.values())),
        "dropped_midline": int(dropped["midline"]),
        "dropped_unknown_side": int(dropped["unknown"]),
    }
    return left, right, receipt


def build_population(
    types_path: Path = DEFAULT_TYPES,
    sided_path: Path = DEFAULT_SIDED,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    max_types: int = MAX_TYPES,
    max_pairs: int = MAX_PAIRS,
    min_pair_weight: int = MIN_PAIR_WEIGHT,
) -> Population:
    """Fix the population the generators draw from, and its receipt."""

    import numpy as np

    table = load_type_table(types_path)
    names = [str(n) for n in table["type_names"]]
    counts = table["n_neurons"].astype(int)
    eligible = [
        i for i, name in enumerate(names)
        if counts[i] >= MIN_NEURONS and NAME_PATTERN.fullmatch(name)
    ]
    # Largest first, then by name, so the population is a deterministic
    # function of the table and nothing else.
    eligible.sort(key=lambda i: (-int(counts[i]), names[i]))
    chosen = eligible[:max_types]
    chosen_names = [names[i] for i in chosen]
    index_of = {names[i]: i for i in chosen}
    in_population = np.zeros(len(names), dtype=bool)
    in_population[chosen] = True

    if sided_path.is_file():
        left, right, side_receipt = side_counts_from_sided_table(sided_path)
    elif annotations_path.is_file():
        left, right, side_receipt = side_counts_from_annotations(annotations_path)
    else:
        raise FileNotFoundError(
            f"no side source: neither {sided_path} nor {annotations_path} exists"
        )

    pre = table["pre"].astype(int)
    post = table["post"].astype(int)
    weight = table["weight"].astype(int)
    keep = (weight >= min_pair_weight) & in_population[pre] & in_population[post] & (pre != post)
    kept = np.flatnonzero(keep)
    order = kept[np.lexsort((post[kept], pre[kept], -weight[kept]))]
    pairs = [(names[pre[i]], names[post[i]], int(weight[i])) for i in order[:max_pairs]]

    # Strongest downstream partner within the population, over every kept
    # edge (not only the capped pair list), so the language rows and the pair
    # task agree on what "strongest" means.
    partner: Dict[str, Tuple[str, int]] = {}
    for i in order:
        a = names[pre[i]]
        if a not in partner:
            partner[a] = (names[post[i]], int(weight[i]))

    population = Population(
        names=chosen_names,
        n_neurons={n: int(counts[index_of[n]]) for n in chosen_names},
        left={n: int(left.get(n, 0)) for n in chosen_names},
        right={n: int(right.get(n, 0)) for n in chosen_names},
        nt={n: str(table["nt"][index_of[n]]) for n in chosen_names},
        superclass={n: str(table["superclass"][index_of[n]]) for n in chosen_names},
        pairs=pairs,
        partner=partner,
    )
    typed_total = int(counts.sum())
    covered = int(sum(population.n_neurons.values()))
    population.receipt = {
        "types_path": str(types_path),
        "types_total": len(names),
        "types_eligible": len(eligible),
        "eligibility": f"n_neurons >= {MIN_NEURONS} and name matches {NAME_PATTERN.pattern}",
        "max_types": max_types,
        "population_size": len(chosen_names),
        "smallest_type_in_population": int(min(population.n_neurons.values())),
        "largest_type_in_population": int(max(population.n_neurons.values())),
        "neurons_typed": typed_total,
        "neurons_covered": covered,
        "coverage_fraction": round(covered / max(1, typed_total), 4),
        "sides": side_receipt,
        "types_with_zero_on_one_side": sum(
            1 for n in chosen_names if population.left[n] == 0 or population.right[n] == 0
        ),
        "min_pair_weight": min_pair_weight,
        "pairs_available": int(len(kept)),
        "max_pairs": max_pairs,
        "pairs_kept": len(pairs),
        "pair_weight_range": [pairs[-1][2], pairs[0][2]] if pairs else [],
        "types_with_partner": len(partner),
        "types_with_unclear_nt": sum(1 for n in chosen_names if population.nt[n] == "unclear"),
    }
    return population


_POPULATION: Optional[Population] = None
_SETTINGS: Dict[str, Any] = {}


def configure(**settings: Any) -> None:
    """Set the population's paths and caps before the first draw.

    The CLI calls this; the benchmark and tests use the defaults. A change
    after the population is built is refused, because two callers in one
    process must be asking about the same facts.
    """

    global _SETTINGS
    if _POPULATION is not None and settings != _SETTINGS:
        raise RuntimeError("the connectome population is already built; configure() must come first")
    _SETTINGS = dict(settings)


def population() -> Population:
    """The shared population, built on first use."""

    global _POPULATION
    if _POPULATION is None:
        _POPULATION = build_population(**_SETTINGS)
    return _POPULATION


def data_available(types_path: Path = DEFAULT_TYPES, sided_path: Path = DEFAULT_SIDED,
                   annotations_path: Path = DEFAULT_ANNOTATIONS) -> bool:
    """Whether the generators could run here. Cheap: existence checks only."""

    return types_path.is_file() and (sided_path.is_file() or annotations_path.is_file())


# -- problems ----------------------------------------------------------------


@dataclass
class ConnectomeProblem:
    """Mirrors `OmniProblem`, so `eval_problem_solving` can adapt it unchanged."""

    task: str
    domain: str
    prompt: str
    response: str
    answer: float
    unit: str
    canonical: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_row(self, keep_canonical: bool = False) -> Dict[str, str]:
        row = {
            "user": self.prompt,
            "assistant": self.response,
            "domain": self.domain,
            "task": self.task,
        }
        if keep_canonical:
            row["canonical"] = self.canonical
        return row


def _pick(rng: random.Random, templates: Sequence[str], **values: Any) -> str:
    return rng.choice(list(templates)).format(**values)


TYPE_COUNT_TEMPLATES = (
    "How many neurons of type {t} are in the male CNS?",
    "In the male CNS, how many neurons have the cell type {t}?",
    "type {t} neuron count in the male CNS",
    "Count the neurons of cell type {t} in the male CNS connectome.",
    "How many {t} neurons does the male CNS contain?",
)

SIDE_COUNT_TEMPLATES = (
    "How many neurons of type {t} are on the {side} side of the male CNS?",
    "In the male CNS, how many {t} neurons have their soma on the {side}?",
    "type {t} {side} side neuron count in the male CNS",
    "Count the {side} side neurons of cell type {t} in the male CNS.",
    "How many {t} neurons are in the {side} hemisphere of the male CNS?",
)

PAIR_TEMPLATES = (
    "How many synapses go from type {a} to type {b} in the male CNS?",
    "In the male CNS, how many synapses does type {a} make onto type {b}?",
    "synapses from {a} to {b} in the male CNS",
    "Count the synapses from cell type {a} onto cell type {b} in the male CNS connectome.",
    "What is the synapse count from {a} to {b} in the male CNS?",
)


def _type_count(rng: random.Random) -> ConnectomeProblem:
    pop = population()
    name = rng.choice(pop.names)
    count = pop.n_neurons[name]
    prompt = _pick(rng, TYPE_COUNT_TEMPLATES, t=name)
    response = f"type {name} has {count} neurons, total {count}"
    return ConnectomeProblem("cns_type_count", DOMAIN, prompt, response, float(count),
                             "neurons", f"type_count {name}", {"type": name})


def _side_count(rng: random.Random) -> ConnectomeProblem:
    pop = population()
    name = rng.choice(pop.names)
    side = rng.choice(("left", "right"))
    count = pop.left[name] if side == "left" else pop.right[name]
    prompt = _pick(rng, SIDE_COUNT_TEMPLATES, t=name, side=side)
    response = f"type {name} has {count} neurons on the {side} side, total {count}"
    return ConnectomeProblem("cns_side_count", DOMAIN, prompt, response, float(count),
                             "neurons", f"side_count {name} {side}",
                             {"type": name, "side": side})


def _pair_synapses(rng: random.Random) -> ConnectomeProblem:
    pop = population()
    a, b, weight = rng.choice(pop.pairs)
    prompt = _pick(rng, PAIR_TEMPLATES, a=a, b=b)
    response = f"type {a} to type {b} has {weight} synapses, total {weight}"
    return ConnectomeProblem("cns_pair_synapses", DOMAIN, prompt, response, float(weight),
                             "synapses", f"pair_synapses {a} {b}", {"pre": a, "post": b})


#: Every benchmarked generator, by task name. `cns_` keeps them distinct from
#: every arithmetic, omni and code task in a shared registry.
TASKS: Dict[str, Callable[[random.Random], ConnectomeProblem]] = {
    "cns_type_count": _type_count,
    "cns_side_count": _side_count,
    "cns_pair_synapses": _pair_synapses,
}


# -- the language family -----------------------------------------------------

NT_TEMPLATES = (
    "What neurotransmitter does type {t} use?",
    "Which neurotransmitter is predicted for the cell type {t} in the male CNS?",
    "neurotransmitter of type {t}",
    "Is type {t} cholinergic, GABAergic or something else?",
)

SUPERCLASS_TEMPLATES = (
    "What kind of neuron is type {t} in the male CNS?",
    "Which superclass does the cell type {t} belong to?",
    "superclass of type {t}",
    "Where in the male CNS does type {t} sit?",
)

PARTNER_TEMPLATES = (
    "Which cell type receives the most synapses from type {t}?",
    "What is the strongest downstream partner of type {t} in the male CNS?",
    "top output target of type {t}",
    "Where does type {t} send most of its synapses?",
)

PROFILE_TEMPLATES = (
    "Tell me about the cell type {t} in the male CNS.",
    "Describe type {t}.",
    "What do we know about type {t} in the male CNS connectome?",
)


def language_row(rng: random.Random) -> Dict[str, str]:
    """One prose statement about a type. No `task` key: these are not scored."""

    pop = population()
    kind = rng.choice(("nt", "superclass", "partner", "profile"))
    if kind == "nt":
        name = rng.choice(pop.nt_names)
        prompt = _pick(rng, NT_TEMPLATES, t=name)
        reply = f"In the male CNS, type {name} is predicted to use {pop.nt[name]}."
    elif kind == "superclass":
        name = rng.choice(pop.names)
        prompt = _pick(rng, SUPERCLASS_TEMPLATES, t=name)
        reply = (f"Type {name} is a {superclass_words(pop.superclass[name])} type "
                 f"in the male CNS.")
    elif kind == "partner":
        name = rng.choice(pop.partner_names)
        target, weight = pop.partner[name]
        prompt = _pick(rng, PARTNER_TEMPLATES, t=name)
        reply = (f"The strongest downstream partner of type {name} in the male CNS "
                 f"is type {target}, with {weight} synapses.")
    else:
        name = rng.choice(pop.names)
        prompt = _pick(rng, PROFILE_TEMPLATES, t=name)
        pieces = [f"Type {name} is a {superclass_words(pop.superclass[name])} type "
                  f"in the male CNS with {pop.n_neurons[name]} neurons, "
                  f"{pop.left[name]} on the left and {pop.right[name]} on the right."]
        if pop.nt[name] != "unclear":
            pieces.append(f"It is predicted to use {pop.nt[name]}.")
        if name in pop.partner:
            target, weight = pop.partner[name]
            pieces.append(f"Its strongest downstream partner is type {target}, "
                          f"with {weight} synapses.")
        reply = " ".join(pieces)
    return {"user": prompt, "assistant": reply, "domain": DOMAIN, "kind": kind}


# -- verification ------------------------------------------------------------


def lookup(canonical: str, pop: Optional[Population] = None) -> Optional[int]:
    """The answer to a canonical query, read back from the population."""

    pop = pop or population()
    parts = canonical.split()
    if len(parts) == 2 and parts[0] == "type_count":
        return pop.n_neurons.get(parts[1])
    if len(parts) == 3 and parts[0] == "side_count":
        table = pop.left if parts[2] == "left" else pop.right if parts[2] == "right" else None
        return None if table is None else table.get(parts[1])
    if len(parts) == 3 and parts[0] == "pair_synapses":
        for a, b, weight in pop.pairs:
            if a == parts[1] and b == parts[2]:
                return weight
        return None
    return None


def verify(problem: ConnectomeProblem) -> bool:
    """A row ships only if its answer is what the population says it is."""

    value = lookup(problem.canonical)
    return value is not None and float(value) == problem.answer


# -- corpus construction ----------------------------------------------------


def build(per_task: int, seed: int, tasks: Optional[Sequence[str]] = None,
          language_rows: int = 0, keep_canonical: bool = False,
          sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
          token_budget: bool = False) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Generate, verify and return (rows, report).

    Sampling is uniform over the population with replacement: repetition is
    the point for a knowledge task, and the report says how much of it each
    task got (`repetition`) and how guessable its answers are
    (`majority_answer_share`).
    """

    rng = random.Random(seed)
    chosen = list(tasks or TASKS)
    unknown = [name for name in chosen if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    pop = population()

    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    distinct: Dict[str, int] = {}
    answers: Dict[str, Counter] = {}
    for name in chosen:
        seen = set()
        made = 0
        tally: Counter = Counter()
        for _ in range(per_task):
            problem = TASKS[name](rng)
            if not verify(problem) or extract_answer(problem.response) != problem.answer:
                dropped[name] = dropped.get(name, 0) + 1
                continue
            seen.add(problem.prompt)
            tally[int(problem.answer)] += 1
            rows.append(problem.to_row(keep_canonical))
            made += 1
        counts[name] = made
        distinct[name] = len(seen)
        answers[name] = tally

    language_kinds: Counter = Counter()
    language_distinct = set()
    for _ in range(language_rows):
        row = language_row(rng)
        language_kinds[row.pop("kind")] += 1
        language_distinct.add(row["user"])
        rows.append(row)

    facts = {
        "cns_type_count": len(pop.names),
        "cns_side_count": 2 * len(pop.names),
        "cns_pair_synapses": len(pop.pairs),
    }
    report: Dict[str, Any] = {
        "schema": "supermix-v93-connectome-corpus-v1",
        "seed": seed,
        "rows": len(rows),
        "task_rows": sum(counts.values()),
        "language_rows": language_rows,
        "language_kinds": dict(language_kinds),
        "language_distinct_prompts": len(language_distinct),
        "per_task": counts,
        "dropped_failing_lookup": dropped,
        "distinct_prompts": distinct,
        "distinct_facts": {k: facts[k] for k in counts},
        "repetition": {k: round(counts[k] / facts[k], 1) for k in counts if facts[k]},
        "majority_answer_share": {
            k: round(tally.most_common(1)[0][1] / max(1, counts[k]), 4)
            for k, tally in answers.items() if tally
        },
        "answer_digits_max": {
            k: len(str(max(tally))) for k, tally in answers.items() if tally
        },
        "population": pop.receipt,
        "verified_by": "lookup in the population built from malecns_types.npz and the side source",
        "options": {"keep_canonical": keep_canonical},
        "non_claims": NON_CLAIMS,
    }
    if token_budget:
        labelled = [dict(row, task=row.get("task", "connectome_language")) for row in rows]
        report["token_budget"] = token_budget_report(labelled, sequence_length)
    return rows, report


NON_CLAIMS: List[str] = [
    "The three cns_* tasks are recall. Their benchmark draws from the same "
    "population of types and pairs the corpus was built from, so a score "
    "measures how much of the table the model retained, not whether it knows "
    "anything about a type it never saw. There is no held-out population.",
    "The population is the largest types by neuron count, capped, and pairs "
    "are the strongest connections within it; the receipt states the caps and "
    "the neuron coverage they give. Nothing here says anything about the "
    "10,000-odd smaller types.",
    "Per-side counts follow the somaSide-else-rootSide rule and drop midline "
    "and unknown-side neurons, so left + right can be less than the type's "
    "neuron count. The counts are facts about the v1.0 release at minconf "
    "0.5, not about the animal.",
    "The language rows are not scored anywhere. They are there so the new "
    "vocabulary appears in sentences; whether the model learns them is not "
    "measured by any receipt in this repository.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per_task", type=int, default=30000)
    parser.add_argument("--language_rows", type=int, default=30000,
                        help="prose rows with no task key (0 to omit)")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--output", default="datasets/v93/v93_connectome.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--task", action="append", default=[],
                        help="restrict to these tasks; repeatable")
    parser.add_argument("--types", default=str(DEFAULT_TYPES))
    parser.add_argument("--sided", default=str(DEFAULT_SIDED),
                        help="D1 sided node table; the annotations file is used if absent")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--max_types", type=int, default=MAX_TYPES)
    parser.add_argument("--max_pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--min_pair_weight", type=int, default=MIN_PAIR_WEIGHT)
    parser.add_argument("--keep_canonical", action="store_true")
    parser.add_argument("--token_budget_report", action="store_true")
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure(types_path=Path(args.types), sided_path=Path(args.sided),
              annotations_path=Path(args.annotations), max_types=args.max_types,
              max_pairs=args.max_pairs, min_pair_weight=args.min_pair_weight)
    rows, report = build(args.per_task, args.seed, args.task or None,
                         language_rows=args.language_rows,
                         keep_canonical=args.keep_canonical,
                         sequence_length=args.sequence_length,
                         token_budget=args.token_budget_report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report["output"] = str(output)
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pop = report["population"]
    print(f"wrote {len(rows):,} rows to {output}")
    print(f"population {pop['population_size']:,} types covering "
          f"{pop['coverage_fraction']:.1%} of typed neurons; "
          f"{pop['pairs_kept']:,} pairs at weight >= {pop['min_pair_weight']}; "
          f"sides from {pop['sides']['kind']}")
    print(f"  {'task':<20} {'rows':>7} {'facts':>6} {'x':>6} {'majority':>9} {'dropped':>8}")
    for name, count in report["per_task"].items():
        print(f"  {name:<20} {count:>7,} {report['distinct_facts'][name]:>6,} "
              f"{report['repetition'].get(name, 0):>6.1f} "
              f"{report['majority_answer_share'].get(name, 0):>9.3f} "
              f"{report['dropped_failing_lookup'].get(name, 0):>8,}")
    if report["language_rows"]:
        print(f"  {'language':<20} {report['language_rows']:>7,}  "
              f"{dict(report['language_kinds'])}")
    budget = report.get("token_budget")
    if budget:
        print(f"\ntoken budget at sequence_length {budget['sequence_length']}")
        for name, stats in budget["tasks"].items():
            print(f"  {name:<20} resp med {stats['response_median']:>3} p95 "
                  f"{stats['response_p95']:>3} max {stats['response_max']:>3}  "
                  f"turn max {stats['turn_max']:>3}  dropped {stats['dropped_fraction']}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
