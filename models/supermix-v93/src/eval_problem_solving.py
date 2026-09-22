"""Measure problem solving by whether the answer is *right*.

Every metric in this repo so far scores how probable the training text was. V64
showed why that is not enough: verbatim reproduction of training data is the
lowest-loss behaviour available, so perplexity actively prefers a model that
recites over one that reasons.

Arithmetic does not have that problem. The answer to "what is 617 + 288" is
checkable, and a model that recites a remembered answer to a *novel* problem is
simply wrong. Exact-match accuracy on freshly generated problems is therefore a
recitation-proof measure of problem solving, which is what this module provides.

The decisive comparison is the pair:

* **seen** -- problems drawn verbatim from the training corpus.
* **novel** -- problems in identical phrasing with operands generated here.

A model that has learned arithmetic scores similarly on both. A model that has
memorised the corpus scores well on *seen* and near zero on *novel*, and the gap
between them is the measurement. Aggregate accuracy alone cannot tell those two
models apart, which is the whole point.

    python source/eval_problem_solving.py --checkpoint output/v64_meaning/v64_meaning.partial.pt

v82 changed three things about *how* this measures, none of them about the
model:

1. **Each task draws from its own RNG.** Until v82 `generate_novel` round-robined
   over `GENERATORS` with one shared `random.Random`, so adding a task,
   reordering `build_omni_corpus.TASKS`, or changing how many draws a generator
   makes shifted every *later* problem for the same seed. The commit before this
   one removed `combination`'s `k` draw, which means the pre- and post-commit
   n=630 numbers were never paired. `--legacy_shared_rng` reproduces the old
   draw exactly, and that is how the v80 receipt is still checkable.
2. **The generation cap rose from 40 to 96 tokens.** Measured with the v80
   tokenizer over `datasets/v80/v80_combined.jsonl` (400 rows per task), reply
   lengths run to a median of 92 tokens for `arithmetic_series` and 88 for
   `work`; eleven of twenty-two tasks exceed 40 tokens and seven exceed 64. At
   40 the benchmark was scoring the cap, not the model.
3. **Two scores are reported, not one.** Exact match is kept unchanged, and an
   abstention-aware score (correct +1, no answer 0, wrong -1) is reported
   beside it, because a binary-graded benchmark rewards confident guessing
   (OpenAI, "Why Language Models Hallucinate", arXiv 2509.04664). Wilson score
   intervals accompany every rate: at n=30 per task the 95% interval is
   [0.332, 0.668] at its widest (p=0.5), so +-17 points -- which is the
   difference between a real per-task change and noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

#: v1 receipts (v65..v81) carry only exact match and no settings block. They
#: are still readable; they are not comparable to a v2 receipt without checking
#: the generator fingerprint by hand, which is exactly why v2 records one.
RECEIPT_SCHEMA_V1 = "supermix-v65-problem-solving-accuracy-v1"
RECEIPT_SCHEMA = "supermix-v82-problem-solving-accuracy-v2"

#: Answers are floats; comparison needs a tolerance rather than equality.
TOLERANCE = 1e-6

#: The cap in force from v65 to v81. Kept as a name so a re-run against an old
#: receipt is a flag rather than a magic number.
LEGACY_MAX_NEW_TOKENS = 40

#: Measured, not chosen for roundness. Two constraints bracket it:
#:
#: * **Below**: the longest replies the model was trained to produce. Over 400
#:   rows per task of `datasets/v80/v80_combined.jsonl`, encoded with the v80
#:   tokenizer, the longest reply is 99 tokens (`momentum`), the 95th
#:   percentile per task tops out at 97 (`arithmetic_series`, `work`); 8 of the
#:   22 task labels in that corpus have a *median* above 40, and 11 produce
#:   replies over 40 at all.
#: * **Above**: `max_position_embeddings` is 128 in the v80 config, and the
#:   longest benchmark prompt measured across all 21 tasks is 30 tokens
#:   (`average`, `word_problem`). 30 + 96 = 126 fits; 30 + 112 would not.
#:
#: 96 is therefore the largest cap that never runs the model past the context
#: it was trained on. It does not eliminate truncation -- a 99-token
#: `momentum` reply still hits it -- which is why truncation is now counted and
#: reported rather than silently scored wrong.
DEFAULT_MAX_NEW_TOKENS = 96

#: Every corpus reply ends "... total <number>". A reply that stops without one
#: was cut off rather than finished.
_TERMINAL = re.compile(r"total\s+-?\d+(?:\.\d+)?\s*\.?\s*$", re.I)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


@dataclass
class Problem:
    task: str
    prompt: str
    answer: float
    source: str  # "novel" or "seen"


# -- generators ------------------------------------------------------------
#
# Phrasing is copied from `build_english_math_dataset.py` so the *format* is
# in-distribution and only the operands are new. A model that failed here purely
# because the wording was unfamiliar would be measuring the wrong thing.


def _arithmetic(rng: random.Random) -> Problem:
    a, b = rng.randint(100, 999), rng.randint(10, 999)
    op = rng.choice(["+", "-"])
    answer = a + b if op == "+" else a - b
    return Problem(
        "arithmetic", f"Solve this basic math problem: {a} {op} {b}", float(answer), "novel"
    )


def _percent(rng: random.Random) -> Problem:
    pct, base = rng.choice([5, 10, 12, 15, 20, 25]), rng.randint(20, 2000)
    return Problem("percent", f"What is {pct}% of {base}?", pct * base / 100.0, "novel")


def _average(rng: random.Random) -> Problem:
    values = [rng.randint(5, 99) for _ in range(rng.choice([4, 5, 6]))]
    joined = ", ".join(str(v) for v in values)
    return Problem(
        "average",
        f"Find the average (mean) of these numbers: {joined}",
        sum(values) / len(values),
        "novel",
    )


def _algebra(rng: random.Random) -> Problem:
    x, b = rng.randint(-30, 30), rng.randint(-30, 30)
    return Problem("algebra_one_step", f"Solve for x: x + {b} = {x + b}", float(x), "novel")


def _word_problem(rng: random.Random) -> Problem:
    start, gain, lose = rng.randint(20, 99), rng.randint(5, 60), rng.randint(5, 60)
    item = rng.choice(["notebooks", "cookies", "marbles", "stickers"])
    return Problem(
        "word_problem",
        f"A student has {start} {item}. They get {gain} more and then give away "
        f"{lose}. How many {item} do they have now?",
        float(start + gain - lose),
        "novel",
    )



# -- v74 task types ---------------------------------------------------------
#
# Added alongside the corpus generators, not after the fact. A capability the
# benchmark cannot see is a capability nobody can claim.


def _multiplication(rng: random.Random) -> Problem:
    a, b = rng.randint(11, 99), rng.randint(2, 9)
    return Problem("multiplication", f"Solve this basic math problem: {a} x {b}",
                   float(a * b), "novel")


def _division(rng: random.Random) -> Problem:
    b, quotient = rng.randint(2, 9), rng.randint(11, 60)
    return Problem("division", f"Solve this basic math problem: {b * quotient} / {b}",
                   float(quotient), "novel")


def _sequence(rng: random.Random) -> Problem:
    start, step = rng.randint(3, 40), rng.randint(2, 12)
    terms = [start + step * i for i in range(4)]
    joined = ", ".join(str(t) for t in terms)
    return Problem("sequence", f"What comes next in the sequence: {joined}?",
                   float(terms[-1] + step), "novel")


def _two_step(rng: random.Random) -> Problem:
    pct = rng.choice([10, 20, 25, 50])
    base = rng.choice([x for x in range(40, 900) if (x * pct) % 100 == 0])
    first = base * pct // 100
    delta = rng.randint(5, 60)
    add = rng.random() < 0.5
    answer = first + delta if add else first - delta
    word = "add" if add else "subtract"
    return Problem("two_step", f"What is {pct}% of {base}, then {word} {delta}?",
                   float(answer), "novel")


GENERATORS: Dict[str, Callable[[random.Random], Problem]] = {
    "arithmetic": _arithmetic,
    "percent": _percent,
    "average": _average,
    "algebra_one_step": _algebra,
    "word_problem": _word_problem,
    "multiplication": _multiplication,
    "division": _division,
    "sequence": _sequence,
    "two_step": _two_step,
}


def _register_omni_generators() -> List[str]:
    """Add the solver-verified science tasks to this benchmark.

    v79 trains on physics, chemistry and mathematics generated by
    `build_omni_corpus`, every row checked against `nexus_solver`. Those tasks
    need measuring on the *same* benchmark as the arithmetic ones, or v79
    cannot be compared with v74 at all -- two report formats would make the
    comparison a matter of interpretation rather than a number.

    The adapter is thin on purpose: the omni generator already produces a
    prompt and an exactly-verified answer, so this only reshapes it. Import
    failures are swallowed and the science tasks simply do not appear, because
    a benchmark that cannot run at all is worse than one measuring less.
    """

    try:
        import build_omni_corpus as omni
    except Exception:  # noqa: BLE001 - optional; arithmetic tasks still run
        return []

    def adapt(name):
        def generate(rng: random.Random) -> Problem:
            problem = omni.TASKS[name](rng)
            return Problem(name, problem.prompt, problem.answer, "novel")
        return generate

    added = []
    for name in omni.TASKS:
        if name not in GENERATORS:      # never shadow an arithmetic task
            GENERATORS[name] = adapt(name)
            added.append(name)
    return added


#: Task names contributed by the solver-verified science corpus.
OMNI_TASKS: List[str] = _register_omni_generators()


def _register_code_generators() -> List[str]:
    """Add the execution-verified code-tracing tasks to this benchmark.

    Same argument as the omni adapter above, and the same shape. What differs
    is the oracle: an omni row is checked against `nexus_solver`, a code row is
    checked by **running the snippet**. The interpreter is the exact checker,
    which is why this family belongs in a corpus whose whole premise is that
    every row is verified before it ships.

    The `code_` prefix means these can never shadow an arithmetic or omni task,
    so the guard below is belt-and-braces rather than load-bearing.

    Registered unconditionally, unlike the corpus-side flags. A benchmark task
    is not a training decision: leaving it out would mean a corpus built with
    code rows had no matching benchmark, which is exactly the silent
    train/eval split `--combination_in_envelope` exists to prevent.
    """

    try:
        import build_code_corpus as code
    except Exception:  # noqa: BLE001 - optional; every other task still runs
        return []

    def adapt(name):
        def generate(rng: random.Random) -> Problem:
            problem = code.TASKS[name](rng)
            return Problem(name, problem.prompt, problem.answer, "novel")
        return generate

    added = []
    for name in code.TASKS:
        if name not in GENERATORS:
            GENERATORS[name] = adapt(name)
            added.append(name)
    return added


#: Task names contributed by the execution-verified code corpus.
CODE_TASKS: List[str] = _register_code_generators()


#: The thirty tasks of the v89 benchmark, in the registry order of the day
#: v89 was scored: nine arithmetic templates, `build_omni_corpus.TASKS`, then
#: `build_code_corpus.TASKS`. Frozen as a literal rather than read from the
#: registry, so that registering a task can never move it. `--task_set v89`
#: expands to exactly this list, and `generator_fingerprint` over it is
#: 3b99a446cd533be9bc5f8ae57d1310b4 -- the value every v87, v88, v89 and v91
#: receipt carries, which `test_v93_corpus.py` pins. A v93 receipt over this
#: set pairs with those by exact McNemar; one over any other set does not.
V89_BENCHMARK_TASKS: Tuple[str, ...] = (
    "arithmetic", "percent", "average", "algebra_one_step", "word_problem",
    "multiplication", "division", "sequence", "two_step",
    "force", "acceleration", "momentum", "kinetic_energy", "work", "power",
    "voltage", "electrical_power", "wave_speed", "molarity", "combination",
    "arithmetic_series",
    "code_loop_add", "code_loop_subtract", "code_list_sum", "code_list_extreme",
    "code_index", "code_conditional", "code_divmod", "code_nested_loop",
    "code_while_accumulate",
)

#: The tasks v93 adds (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, D7): five
#: solver-verified science tasks, three execution-verified code tasks and
#: three connectome-knowledge lookups. `--task_set new` expands to this list.
#: Their scores have no v89 baseline -- v89 cannot answer them -- so a number
#: over this set is descriptive, and the connectome three are recall (see
#: `NON_CLAIMS`).
V93_NEW_TASKS: Tuple[str, ...] = (
    "impulse", "ohms_current", "spring_energy", "permutations", "final_velocity",
    "code_range_sum", "code_list_count", "code_neg_index",
    "cns_type_count", "cns_side_count", "cns_pair_synapses",
)

#: What `--task_set` expands to. `all` is the v89 thirty followed by the
#: eleven new tasks, which is also the registry's own order today.
TASK_SETS: Dict[str, Tuple[str, ...]] = {
    "v89": V89_BENCHMARK_TASKS,
    "new": V93_NEW_TASKS,
    "all": V89_BENCHMARK_TASKS + V93_NEW_TASKS,
}


def _register_v93_generators() -> List[str]:
    """Add the v93 families to this benchmark, after the thirty above.

    Three registries, one adapter shape each, exactly as the two adapters
    above: `build_omni_corpus.V93_TASKS` (solver-verified),
    `build_code_corpus.V93_TASKS` (execution-verified) and
    `build_connectome_corpus.TASKS` (looked up in the CC-BY male-CNS tables).
    They are kept out of the builders' `TASKS` dicts so a default corpus build
    and the two adapters above stay byte-identical; this is where they enter
    the benchmark, and the trainer's probe, instead.

    Registration never shadows: a name already in `GENERATORS` is skipped, so
    the v89 thirty cannot be redefined from here. Per-task RNGs mean the
    thirty's problems do not move either (`test_v93_corpus.py` asserts both).

    The connectome tasks need `datasets/v91_malecns/malecns_types.npz` and a
    side source on disk; where those are absent the family is left out, as an
    import failure leaves a builder's family out. A `--task_set new` run on
    such a machine then fails loudly at the unknown-task check rather than
    silently scoring eight tasks as eleven.
    """

    added: List[str] = []

    def adapt(table, name):
        def generate(rng: random.Random) -> Problem:
            problem = table[name](rng)
            return Problem(name, problem.prompt, problem.answer, "novel")
        return generate

    try:
        import build_omni_corpus as omni
        omni_v93 = getattr(omni, "V93_TASKS", {})
    except Exception:  # noqa: BLE001 - optional, as for the twelve above
        omni_v93 = {}
    try:
        import build_code_corpus as code
        code_v93 = getattr(code, "V93_TASKS", {})
    except Exception:  # noqa: BLE001
        code_v93 = {}
    try:
        import build_connectome_corpus as cns
        cns_tasks = cns.TASKS if cns.data_available() else {}
    except Exception:  # noqa: BLE001
        cns_tasks = {}

    for table in (omni_v93, code_v93, cns_tasks):
        for name in table:
            if name not in GENERATORS:      # never shadow an existing task
                GENERATORS[name] = adapt(table, name)
                added.append(name)
    return added


#: Task names the v93 adapter actually registered on this machine. Equal to
#: `V93_NEW_TASKS` wherever the connectome data is present.
V93_REGISTERED_TASKS: List[str] = _register_v93_generators()

# The three v93 code tasks are execution-verified code tasks like the nine
# before them, and `CODE_TASKS` is how the live checker's tests find every
# task that `answer_check._code_trace` must run; so they are listed there too.
# `OMNI_TASKS` stays the twelve of v89 (it names what `_register_omni_generators`
# added), and the five new science tasks are found through `V93_NEW_TASKS`.
CODE_TASKS.extend(name for name in V93_REGISTERED_TASKS if name.startswith("code_"))


def expand_task_set(name: str) -> List[str]:
    """The ordered task list a `--task_set` name stands for."""

    if name not in TASK_SETS:
        raise KeyError(f"unknown task set {name!r}; choose from {', '.join(TASK_SETS)}")
    return list(TASK_SETS[name])


def task_rng(task: str, seed: int) -> random.Random:
    """A generator's own RNG, derived from its *name* and the master seed.

    Deriving from the name rather than from an index is the whole point: a
    task's stream then does not depend on how many tasks exist, what order they
    are in, or how many draws its neighbours make.
    """

    digest = hashlib.blake2b(f"{task}:{seed}".encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def generate_novel(
    count: int,
    seed: int = 65,
    tasks: Optional[Sequence[str]] = None,
    shared_rng: bool = False,
) -> List[Problem]:
    """`count` problems, round-robined over the task list.

    The interleaving order is unchanged from v65. What changed in v82 is where
    the randomness comes from.

    Until v82 one `random.Random(seed)` fed every generator in turn, so the
    stream a task saw depended on every task before it in `GENERATORS`. Adding
    `combination` to `build_omni_corpus.TASKS`, or removing a single `rng`
    draw from it -- which the commit before v82 did -- silently changed every
    problem *after* that task for the same seed. Two receipts with the same
    seed were therefore not necessarily comparable, and the pre- and
    post-commit v80 numbers are not paired.

    With `shared_rng=False` (the default from v82) each task derives its own
    RNG from its name, so tasks are independent: adding a task changes only
    that task's problems. Pass `shared_rng=True` to reproduce a pre-v82 draw
    exactly -- that is how the v80 receipt's recorded examples are still
    checkable, and `test_eval_v82.py` asserts it.
    """

    names = list(tasks or GENERATORS)
    if not names:
        return []
    if shared_rng:
        rng = random.Random(seed)
        return [GENERATORS[names[i % len(names)]](rng) for i in range(count)]
    streams = {name: task_rng(name, seed) for name in names}
    problems: List[Problem] = []
    for index in range(count):
        name = names[index % len(names)]
        problems.append(GENERATORS[name](streams[name]))
    return problems


def generator_fingerprint(
    tasks: Optional[Sequence[str]] = None, probe: int = 3, seed: int = 0
) -> str:
    """A short digest of *what the generators produce*, for receipt comparison.

    Two receipts that share a seed, a task list and this fingerprint drew the
    same problems. Two that differ in it did not, whatever their seeds say. It
    hashes each task's first few prompts and answers under a fixed probe seed,
    so a reworded template, a changed operand range, or a different number of
    rng draws all move it -- which is what the v80 comparison needed and did
    not have.

    It deliberately does not hash source code: a comment change would move that
    and would not change a single problem.
    """

    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(tasks or GENERATORS):
        rng = task_rng(name, seed)
        digest.update(name.encode("utf-8"))
        for _ in range(probe):
            problem = GENERATORS[name](rng)
            digest.update(f"|{problem.prompt}|{problem.answer!r}".encode("utf-8"))
    return digest.hexdigest()


# -- seen problems ---------------------------------------------------------


def load_seen(
    path: Path, count: int, seed: int = 65, tasks: Optional[Sequence[str]] = None
) -> List[Problem]:
    """Problems lifted verbatim from the corpus, with their stated answers.

    These are the memorisation control. Rows whose answer cannot be parsed are
    skipped rather than guessed at -- a wrong ground truth would understate the
    model and make the comparison meaningless in the flattering direction.
    """

    wanted = set(tasks or GENERATORS)
    rows: List[Problem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("task") not in wanted:
                continue
            answer = extract_answer(str(record.get("assistant", "")))
            if answer is None:
                continue
            rows.append(Problem(record["task"], str(record["user"]), answer, "seen"))
    random.Random(seed).shuffle(rows)
    return rows[:count]


# -- scoring ---------------------------------------------------------------


def extract_answer(text: str) -> Optional[float]:
    """The model's answer is the last number it produces.

    The corpus answers in several shapes -- "79", "376 - 17 = 359",
    "Calculate directly. Answer: 10.68." -- and in every one the final number is
    the answer. Taking the last rather than the first is what makes
    "376 - 17 = 359" score as 359 instead of 376.
    """

    matches = _NUMBER.findall(text.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1].rstrip(".")
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            return float(numerator) / float(denominator)
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(predicted: Optional[float], expected: float) -> bool:
    if predicted is None:
        return False
    return abs(predicted - expected) <= max(TOLERANCE, abs(expected) * 1e-6)


def wilson_interval(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because this benchmark
    routinely reports 0/30 and 30/30, where `p +- z*sqrt(p(1-p)/n)` is a
    zero-width interval and therefore a lie. Wilson gives (0.000, 0.113) for
    0/30 and (0.885, 1.000) for 30/30.

    The trainer already refuses to *select* on fewer than
    `train_mimomix_generalisation.MIN_SELECTION_PROBLEMS = 100` problems for
    this reason, citing v73's 20-problem probe reading 0.15 where a
    60-problem evaluation of the same checkpoint read 0.467. This benchmark
    runs 30 per task, so per-task numbers here carry roughly a +-18 point
    interval and must not be read as rankings. The aggregate over 21 tasks is
    the number with enough n to move.
    """

    if n <= 0:
        return (0.0, 0.0)
    p = correct / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def looks_terminated(text: str) -> bool:
    """True when the reply ends in the corpus's terminal `total <number>`."""

    return _TERMINAL.search(text.strip()) is not None


def is_truncated(text: str, tokens: Optional[int], max_new_tokens: int) -> bool:
    """The reply spent its whole budget and never reached a final answer.

    Both halves matter. A reply that stops early without "total N" simply
    failed; a reply that ends on "total 905" at exactly the cap finished. Only
    the conjunction -- budget exhausted *and* no terminal total -- means the
    harness cut the model off, and that is a fact about the benchmark rather
    than about the model, so it is reported separately instead of being scored
    as a wrong answer.
    """

    if tokens is None:
        return False
    return tokens >= max_new_tokens and not looks_terminated(text)


def abstention_score(correct: int, wrong: int, abstained: int) -> Optional[float]:
    """Correct +1, no answer 0, wrong -1, divided by n.

    Exact match is a binary grade, and a binary grade pays a model to guess:
    a wrong answer and a refusal both score zero, so guessing is free upside
    (OpenAI, "Why Language Models Hallucinate", 2025, arXiv 2509.04664). This
    is reported *beside* exact match, never instead of it -- the two answer
    different questions and this repo's headline number is still exact match.

    The range is [-1, +1]: 1.0 is every problem right, 0.0 is either total
    abstention or right-as-often-as-wrong, and negatives mean the model is
    wrong more often than right and would score better by saying nothing.
    """

    n = correct + wrong + abstained
    if n <= 0:
        return None
    return (correct - wrong) / n


def evaluate(
    checkpoint: Path,
    problems: Sequence[Problem],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    settings: Optional[Dict[str, Any]] = None,
    transcript: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score a checkpoint and return a v2 receipt.

    `max_new_tokens` defaults to 96 from v82 (it was 40 from v65 to v81); see
    `DEFAULT_MAX_NEW_TOKENS` for the measurement behind the number. Pass
    `LEGACY_MAX_NEW_TOKENS` to reproduce a pre-v82 receipt.

    `settings` is copied verbatim into the receipt's `settings` block. Callers
    should put the seed, the task list and the generator fingerprint there --
    `main` does -- so that two receipts can be compared without guessing
    whether they drew the same problems.

    Pass a list as `transcript` to collect every reply in full. The receipt's
    own `examples` block keeps twelve replies clipped to 90 characters, which
    is enough to see what a reply looks like and not enough to check whether
    its working supports its answer -- and after v86 was caught writing
    ``320 / 7 = 60`` on the way to a correct 60, that is a question worth being
    able to ask of a whole run rather than of twelve rows.
    """

    model, tokenizer, payload = load_talk_checkpoint(checkpoint)
    model.eval()

    per_task: Dict[Tuple[str, str], Dict[str, Any]] = {}
    unparsed = 0
    truncated = 0
    examples: List[Dict[str, Any]] = []
    for problem in problems:
        reply = generate_reply(model, tokenizer, problem.prompt, max_new_tokens=max_new_tokens)
        if isinstance(reply, dict):
            text = reply["reply"]
            tokens = reply.get("tokens")
        else:
            text, tokens = str(reply), None
        predicted = extract_answer(text)
        if predicted is None:
            unparsed += 1
        correct = is_correct(predicted, problem.answer)
        cut_off = is_truncated(text, tokens, max_new_tokens)
        if cut_off:
            truncated += 1

        bucket = per_task.setdefault(
            (problem.source, problem.task),
            {"n": 0, "correct": 0, "errors": [], "truncated": 0, "abstained": 0},
        )
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        bucket["truncated"] += int(cut_off)
        # An abstention is a reply that states no answer. A reply the harness
        # cut off states no answer *of the model's choosing*, so it is scored
        # 0 rather than -1: charging the model for the benchmark's own token
        # budget would make the abstention score a measure of the cap.
        bucket["abstained"] += int(predicted is None or (cut_off and not correct))
        # Exact match alone cannot tell "computed approximately" from "produced
        # noise", and the difference is the whole question for a model this
        # size. Relative error separates them: v65 answers 51.5 where the truth
        # is 51.333, which is a different failure from answering "Synonym: glad".
        if predicted is not None:
            bucket["errors"].append(
                abs(predicted - problem.answer) / max(1.0, abs(problem.answer))
            )
        if transcript is not None:
            transcript.append(
                {
                    "source": problem.source,
                    "task": problem.task,
                    "prompt": problem.prompt,
                    "reply": text,
                    "expected": problem.answer,
                    "predicted": predicted,
                    "correct": correct,
                    "tokens": tokens,
                    "truncated": cut_off,
                }
            )
        if len(examples) < 12:
            examples.append(
                {
                    "source": problem.source,
                    "task": problem.task,
                    "prompt": problem.prompt[:90],
                    "reply": text[:90],
                    "expected": problem.answer,
                    "predicted": predicted,
                    "correct": correct,
                    "tokens": tokens,
                    "truncated": cut_off,
                }
            )

    by_source: Dict[str, Dict[str, Any]] = {}
    for (source, task), counts in sorted(per_task.items()):
        entry = by_source.setdefault(
            source, {"n": 0, "correct": 0, "truncated": 0, "abstained": 0, "tasks": {}}
        )
        entry["n"] += counts["n"]
        entry["correct"] += counts["correct"]
        entry["truncated"] += counts["truncated"]
        entry["abstained"] += counts["abstained"]
        errors = sorted(counts["errors"])
        entry.setdefault("errors", []).extend(errors)
        n = counts["n"]
        wrong = n - counts["correct"] - counts["abstained"]
        low, high = wilson_interval(counts["correct"], n)
        untruncated = n - counts["truncated"]
        entry["tasks"][task] = {
            "n": n,
            "correct": counts["correct"],
            "accuracy": round(counts["correct"] / max(1, n), 4),
            "accuracy_95ci": [round(low, 4), round(high, 4)],
            "truncated": counts["truncated"],
            "abstained": counts["abstained"],
            "abstention_score": (
                None
                if (score := abstention_score(counts["correct"], wrong, counts["abstained"]))
                is None
                else round(score, 4)
            ),
            # Truncated replies are the harness's fault, so accuracy with them
            # removed from the denominator is reported too. It is an upper
            # bound in the same way `accuracy` is a lower one: neither is the
            # number, and quoting only the flattering one would be the
            # dishonesty this receipt exists to prevent.
            "accuracy_untruncated": (
                round(counts["correct"] / untruncated, 4) if untruncated > 0 else None
            ),
            "median_relative_error": (
                round(errors[len(errors) // 2], 4) if errors else None
            ),
            "within_10_percent": round(
                sum(1 for e in errors if e <= 0.10) / max(1, n), 4
            ),
        }
    for entry in by_source.values():
        n = entry["n"]
        entry["accuracy"] = round(entry["correct"] / max(1, n), 4)
        low, high = wilson_interval(entry["correct"], n)
        entry["accuracy_95ci"] = [round(low, 4), round(high, 4)]
        wrong = n - entry["correct"] - entry["abstained"]
        score = abstention_score(entry["correct"], wrong, entry["abstained"])
        entry["abstention_score"] = None if score is None else round(score, 4)
        entry["wrong"] = wrong
        untruncated = n - entry["truncated"]
        entry["accuracy_untruncated"] = (
            round(entry["correct"] / untruncated, 4) if untruncated > 0 else None
        )
        errors = sorted(entry.pop("errors", []))
        entry["median_relative_error"] = (
            round(errors[len(errors) // 2], 4) if errors else None
        )
        entry["within_10_percent"] = round(
            sum(1 for e in errors if e <= 0.10) / max(1, n), 4
        )

    seen = by_source.get("seen", {}).get("accuracy")
    novel = by_source.get("novel", {}).get("accuracy")
    receipt_settings: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "legacy_max_new_tokens": LEGACY_MAX_NEW_TOKENS,
        "tolerance": TOLERANCE,
    }
    receipt_settings.update(settings or {})

    # A checkpoint written from v85 onward records the corpus it trained on. If
    # the seen arm was drawn from a different file the gap is not a memorisation
    # measurement, so say which two files disagreed rather than publishing it.
    trained_on = (payload.get("extra") or {}).get("corpus_jsonl")
    scored_against = receipt_settings.get("corpus")
    corpus_mismatch = None
    if seen is not None and trained_on and scored_against:
        if Path(trained_on).name != Path(scored_against).name:
            corpus_mismatch = (
                f"seen rows came from {Path(scored_against).name}, but this "
                f"checkpoint trained on {Path(trained_on).name}"
            )
            print(f"memorisation gap withheld: {corpus_mismatch}", file=sys.stderr)
    receipt_settings["checkpoint_trained_on"] = trained_on

    return {
        "schema": RECEIPT_SCHEMA,
        "checkpoint": str(checkpoint),
        "dev_loss": (payload.get("extra") or {}).get("best_dev_loss"),
        "problems": len(problems),
        "unparsed_replies": unparsed,
        "truncated_replies": truncated,
        "settings": receipt_settings,
        "by_source": by_source,
        # Only a gap between a model and its OWN training rows means anything.
        # Where the checkpoint records the corpus it trained on and the seen arm
        # was drawn from a different one, the number is withheld and the reason
        # given: a wrong gap reads as a finding, and a missing one reads as a
        # missing measurement, which is what it is.
        "memorisation_gap": (
            round(seen - novel, 4)
            if seen is not None and novel is not None and not corpus_mismatch
            else None
        ),
        "memorisation_gap_withheld": corpus_mismatch,
        "examples": examples,
        "non_claims": NON_CLAIMS,
    }


#: What this benchmark does *not* establish. Rewritten in v82: the previous
#: text still said "five task types with small operands" on what had become a
#: 21-task, 630-problem run, which understated the coverage while overstating
#: nothing -- but a receipt that describes a different experiment than the one
#: it ran is not honest in either direction.
NON_CLAIMS: List[str] = [
    "Accuracy on generated problems is not general problem solving. The 21 "
    "tasks are nine arithmetic templates (two- and three-digit operands, one "
    "or two steps) and twelve single-formula science and mathematics "
    "templates. Every prompt comes from a handful of fixed phrasings, so this "
    "measures a narrow, in-distribution slice and says nothing about "
    "multi-step reasoning, unfamiliar wording, or any task not listed.",
    "A high 'seen' score with a low 'novel' score is recall, not skill; the "
    "gap is the finding, not either number alone.",
    "Answer extraction takes the last number in the reply. A reply that "
    "reasons correctly and then trails off into another number scores wrong, "
    "so this is a lower bound on the model.",
    # This line used to assert "n=30 by default". It is not the default: --novel
    # is 100 over 21 registered tasks, which is 4 or 5 per task, where the widest
    # Wilson interval is +-33 points rather than +-17. A non-claims block that
    # understates its own uncertainty by a factor of two is worse than none, so
    # the number is now computed from the run instead of asserted.
    "A per-task row's uncertainty depends on that row's n, which is "
    "novel_requested divided across the task list -- at the --novel 100 default "
    "over 21 tasks that is 4 or 5 problems per task, where the widest 95% "
    "Wilson interval spans about +-33 points, not the +-17 that n=30 gives. "
    "Every row carries its own interval in the receipt; read those rather than "
    "any rule of thumb. The trainer's MIN_SELECTION_PROBLEMS = 100 is the same "
    "argument applied to checkpoint selection.",
    "Two receipts are comparable only if their seed, task list and "
    "generator_fingerprint all match. Before v82 tasks shared one RNG, so "
    "changing any generator silently changed every later task's problems for "
    "the same seed -- v80's pre- and post-commit n=630 receipts are not "
    "paired, and no delta between them should be quoted.",
    "truncated_replies counts replies that spent the whole token budget "
    "without reaching a terminal 'total <number>'. Those are a limit of this "
    "harness, not a measured failure of the model, and 'accuracy' still counts "
    "them as wrong. Read 'accuracy' as a lower bound and "
    "'accuracy_untruncated' as an upper one.",
    "The abstention score is an alternative grading of the same replies, not "
    "a second measurement. It cannot reward calibrated uncertainty in a model "
    "that was never trained to express any -- these models emit a number or "
    "nothing.",
    # v93. The eleven new tasks enter the default registry, so a run with no
    # --task_set or --tasks now draws over 41 names. The line above about
    # comparability already covers that; this one covers the three tasks
    # whose score is not skill of any kind.
    "The cns_* tasks (cns_type_count, cns_side_count, cns_pair_synapses) "
    "measure recall over the same population of types and type pairs the "
    "corpus was generated from: benchmark and corpus draw from one "
    "`build_connectome_corpus.population()`, and there is no held-out type. "
    "A score there is how much of the male-CNS table the model retained, not "
    "generalisation to a type it never saw, and it says nothing about the "
    "10,000-odd types outside the population. Use --task_set v89 for the "
    "thirty tasks that pair with v87-v91 receipts and --task_set new for the "
    "eleven that have no baseline.",
]


def print_summary(report: Dict[str, Any]) -> None:
    settings = report.get("settings") or {}
    print(f"checkpoint  {report['checkpoint']}")
    print(f"dev loss    {report['dev_loss']}")
    print(
        f"problems    {report['problems']}   unparsed {report['unparsed_replies']}"
        f"   truncated {report.get('truncated_replies', 0)}"
    )
    print(
        f"settings    seed {settings.get('seed')}  max_new_tokens "
        f"{settings.get('max_new_tokens')}  shared_rng {settings.get('shared_rng')}"
    )
    print(f"fingerprint {settings.get('generator_fingerprint')}")
    print()
    header = (
        f"{'source':8s} {'task':18s} {'n':>5s} {'corr':>5s} {'acc':>7s} "
        f"{'95% CI':>17s} {'trunc':>6s} {'abst':>6s}"
    )
    print(header)
    print("-" * len(header))
    for source in ("seen", "novel"):
        entry = report["by_source"].get(source)
        if not entry:
            continue
        for task, counts in sorted(entry["tasks"].items()):
            low, high = counts.get("accuracy_95ci", [0.0, 0.0])
            print(
                f"{source:8s} {task:18s} {counts['n']:5d} {counts['correct']:5d} "
                f"{counts['accuracy']:7.3f} [{low:6.3f},{high:6.3f}] "
                f"{counts.get('truncated', 0):6d} {counts.get('abstained', 0):6d}"
            )
        low, high = entry.get("accuracy_95ci", [0.0, 0.0])
        print(
            f"{source:8s} {'ALL':18s} {entry['n']:5d} {entry['correct']:5d} "
            f"{entry['accuracy']:7.3f} [{low:6.3f},{high:6.3f}] "
            f"{entry.get('truncated', 0):6d} {entry.get('abstained', 0):6d}"
        )
        mre = entry.get("median_relative_error")
        untruncated = entry.get("accuracy_untruncated")
        score = entry.get("abstention_score")
        score_text = "n/a" if score is None else f"{score:+.4f}"
        untruncated_text = "n/a" if untruncated is None else f"{untruncated:.3f}"
        mre_text = "n/a" if mre is None else f"{mre:.3f}"
        print(
            f"{'':8s} {'':18s} abstention score {score_text}"
            f"   acc excl. truncated {untruncated_text}"
            f"   median rel.err {mre_text}"
            f"   within10% {entry['within_10_percent']:.3f}"
        )
        print()
    gap = report.get("memorisation_gap")
    if gap is not None:
        print(f"memorisation gap (seen - novel): {gap:+.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            "corpus to draw 'seen' problems from. It must be the corpus the "
            "CHECKPOINT WAS TRAINED ON, or the memorisation gap is meaningless. "
            "There is no default: this used to be hard-coded to "
            "datasets/v62/english_math_40k.jsonl, so every run after v62 "
            "silently measured its 'seen' arm against a corpus the model had "
            "never seen. Omit it and the seen arm is skipped and the gap "
            "reported as null, which is honest; pass the wrong one and the "
            "number is worse than missing."
        ),
    )
    parser.add_argument("--novel", type=int, default=100)
    parser.add_argument("--seen", type=int, default=100)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            f"generation cap (default {DEFAULT_MAX_NEW_TOKENS}; it was "
            f"{LEGACY_MAX_NEW_TOKENS} through v81, which truncated 11 of 22 "
            "task shapes)"
        ),
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        default=None,
        help="comma-separated subset of tasks; default is every registered task",
    )
    task_group.add_argument(
        "--task_set",
        choices=sorted(TASK_SETS),
        default=None,
        help=(
            "a named task list instead of --tasks: 'v89' is the thirty tasks "
            "every v87-v91 receipt was scored on, in their order, so the "
            "fingerprint reproduces 3b99a446cd533be9bc5f8ae57d1310b4 and "
            "--novel 630 gives 21 per task; 'new' is the eleven v93 tasks, "
            "which have no baseline; 'all' is both. With neither flag the run "
            "draws over every registered task, which since v93 is 41 names "
            "and a fingerprint that pairs with nothing published."
        ),
    )
    parser.add_argument(
        "--legacy_shared_rng",
        action="store_true",
        help=(
            "draw problems the pre-v82 way, from one RNG shared across tasks. "
            "Only for reproducing a receipt written before v82; the draw is "
            "not stable against adding or reordering a task."
        ),
    )
    parser.add_argument(
        "--combination_in_envelope",
        action="store_true",
        help=(
            "score `combination` with the narrowed generator that the corpus "
            "builder's --combination_in_envelope produces. The twelve omni "
            "tasks are ADAPTED from build_omni_corpus.TASKS, so a corpus built "
            "with that flag must be scored with this one, or the model is "
            "tested on problems its corpus never contained. Without it the "
            "global is always False in this process and the mismatch is "
            "invisible. The nine arithmetic tasks keep their own generators "
            "here and no corpus flag reaches them, which is why only this one "
            "needs mirroring."
        ),
    )
    parser.add_argument(
        "--dump_replies",
        default=None,
        help=(
            "write every reply in full to this path as JSONL. The receipt's "
            "`examples` block keeps twelve replies clipped to 90 characters, "
            "which cannot answer whether a reply's working supports its "
            "answer. `step_audit` reads this file."
        ),
    )
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    task_set = getattr(args, "task_set", None)
    if task_set:
        tasks: Optional[List[str]] = expand_task_set(task_set)
    else:
        tasks = (
            [name.strip() for name in args.tasks.split(",") if name.strip()]
            if args.tasks
            else None
        )
    unknown = [name for name in (tasks or []) if name not in GENERATORS]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")

    # Set before any problem is drawn AND before the fingerprint is computed, so
    # the digest reflects the generator actually used. The fingerprint does
    # distinguish the two shapes; what it could not do until now is ever see the
    # narrowed one, because nothing in this process could set the global.
    if getattr(args, "combination_in_envelope", False):
        try:
            import build_omni_corpus as _omni

            _omni.COMBINATION_IN_ENVELOPE = True
        except ImportError:  # the omni tasks are simply absent; nothing to narrow
            pass

    problems = list(
        generate_novel(
            args.novel, seed=args.seed, tasks=tasks, shared_rng=args.legacy_shared_rng
        )
    )
    # The seen arm is the memorisation control, and it is only a control if its
    # rows are rows the checkpoint actually trained on. Requiring the path makes
    # a wrong one a decision someone made rather than a default they inherited.
    corpus = Path(args.corpus) if args.corpus else None
    seen_skipped = None
    if not args.seen:
        seen_skipped = "not requested"
    elif corpus is None:
        seen_skipped = (
            "no --corpus given; the seen arm and the memorisation gap need the "
            "checkpoint's own training corpus"
        )
    elif not corpus.is_file():
        seen_skipped = f"corpus not found: {corpus}"
    if seen_skipped is None:
        problems.extend(load_seen(corpus, args.seen, seed=args.seed, tasks=tasks))
    else:
        print(f"seen arm skipped: {seen_skipped}", file=sys.stderr)
    settings = {
        "seed": args.seed,
        "tasks": list(tasks or GENERATORS),
        "task_set": task_set,
        "shared_rng": bool(args.legacy_shared_rng),
        "generator_fingerprint": generator_fingerprint(tasks),
        "novel_requested": args.novel,
        "seen_requested": args.seen,
        "corpus": str(corpus) if corpus else None,
        "seen_skipped": seen_skipped,
    }
    transcript: Optional[List[Dict[str, Any]]] = [] if args.dump_replies else None
    report = evaluate(
        Path(args.checkpoint), problems, args.max_new_tokens, settings=settings,
        transcript=transcript,
    )
    print_summary(report)
    if transcript is not None:
        replies = Path(args.dump_replies)
        replies.parent.mkdir(parents=True, exist_ok=True)
        with replies.open("w", encoding="utf-8") as handle:
            for record in transcript:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"replies -> {replies} ({len(transcript)} rows)")
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nreceipt -> {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
