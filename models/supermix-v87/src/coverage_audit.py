"""Which values does the benchmark ask about that the corpus never teaches?

`percent` scored 0.533 on v86 and was read as a hard task for eight versions.
It is not. Split by the percentage asked for:

    5, 10, 20, 25 percent   16/17 = 0.941
    12, 15 percent           0/13 = 0.000

`build_scratchpad_math._scratchpad_percent` draws its percentage from
``[5, 10, 20, 25, 50]`` and `eval_problem_solving._percent` draws from
``[5, 10, 12, 15, 20, 25]``. Twelve and fifteen appear in the benchmark and in
none of the corpus's forty thousand percent rows, and fifty appears in the
corpus and never in the benchmark. Nothing reported this: the two generators
live in different files, neither imports the other, and the score they produce
together looks like a middling task rather than two tasks averaged.

This module compares the two sides directly. For each task it draws prompts
from both generators, pulls every number out of them, and reports values the
benchmark asks about that the corpus never contains.

It compares *values*, not ranges, because that is what catches a categorical
hole. A range check would have to know which numbers in a prompt are operands
and which are incidental, and would still have called ``[5, 10, 20, 25, 50]``
and ``[5, 10, 12, 15, 20, 25]`` compatible on the grounds that both run 5 to 50.

What it cannot do is tell a real hole from a rare draw, so it reports a
frequency alongside every value and leaves the judgement where it belongs. A
value the benchmark asks for in a third of its problems and the corpus never
contains is a defect; one that turns up once in a thousand is a tail.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: Corpus task names that the benchmark knows under a different name.
#: `eval_problem_solving._arithmetic` emits both signs under one task; the
#: corpus splits them into two builders.
CORPUS_ALIASES = {"addition": "arithmetic", "subtraction": "arithmetic"}


def numbers_in(prompt: str) -> List[float]:
    return [float(match) for match in NUMBER.findall(prompt)]


def _profile(prompts: Iterable[str]) -> Dict[str, object]:
    counts: collections.Counter = collections.Counter()
    total = 0
    for prompt in prompts:
        total += 1
        counts.update(numbers_in(prompt))
    return {"prompts": total, "values": counts}


def compare(corpus_prompts: Sequence[str], benchmark_prompts: Sequence[str],
            threshold: float = 0.02) -> Dict[str, object]:
    """Values the benchmark asks about often and the corpus never contains.

    `threshold` is the share of benchmark prompts a missing value must appear
    in before it is called a hole rather than a tail. Two percent of thirty
    problems is under one problem, so the default errs towards reporting.
    """

    corpus = _profile(corpus_prompts)
    benchmark = _profile(benchmark_prompts)
    seen = set(corpus["values"])
    holes = []
    for value, count in benchmark["values"].most_common():
        if value in seen:
            continue
        share = count / max(1, benchmark["prompts"])
        if share >= threshold:
            holes.append({"value": value, "benchmark_prompts": count,
                          "share_of_benchmark": round(share, 4)})
    # The reverse direction is not a defect, but it is worth knowing: corpus
    # values the benchmark never asks about are training the model spends
    # capacity on and the score never sees.
    unasked = []
    asked = set(benchmark["values"])
    for value, count in corpus["values"].most_common(20):
        if value in asked:
            continue
        share = count / max(1, corpus["prompts"])
        if share >= threshold:
            unasked.append({"value": value, "corpus_prompts": count,
                            "share_of_corpus": round(share, 4)})
    return {
        "corpus_prompts": corpus["prompts"],
        "benchmark_prompts": benchmark["prompts"],
        "distinct_corpus_values": len(corpus["values"]),
        "distinct_benchmark_values": len(benchmark["values"]),
        "asked_but_never_taught": holes,
        "taught_but_never_asked": unasked,
    }


def audit(samples: int = 4000, seed: int = 87, threshold: float = 0.02,
          tasks: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Run the comparison over every task both sides define."""

    import build_code_corpus as code
    import build_omni_corpus as omni
    import build_scratchpad_math as scratch
    import eval_problem_solving as solving

    corpus_generators: Dict[str, Callable[[random.Random], str]] = {}
    for name, generator in omni.TASKS.items():
        corpus_generators[name] = lambda rng, g=generator: g(rng).prompt
    for name, generator in code.TASKS.items():
        corpus_generators[name] = lambda rng, g=generator: g(rng).prompt
    # `build_scratchpad_math.GENERATORS` is a tuple of functions with no names
    # attached; each row it produces carries its own `task`. Calling each once
    # is how the mapping is recovered, and it is also a check that every
    # generator runs at all.
    probe = random.Random(0)
    for generator in scratch.GENERATORS:
        name = str(generator(probe)["task"])
        target = CORPUS_ALIASES.get(name, name)
        corpus_generators.setdefault(
            target, lambda rng, g=generator: _scratchpad_prompt(g(rng)))

    report: Dict[str, object] = {
        "schema": "supermix-v87-coverage-audit-v1",
        "samples_per_side": samples,
        "seed": seed,
        "threshold": threshold,
        "tasks": {},
        "not_compared": [],
    }
    names = list(tasks or solving.GENERATORS)
    for name in sorted(names):
        if name not in corpus_generators:
            report["not_compared"].append(name)
            continue
        corpus_rng = random.Random(f"corpus:{name}:{seed}")
        bench_rng = solving.task_rng(name, seed)
        corpus_prompts = [corpus_generators[name](corpus_rng) for _ in range(samples)]
        bench_prompts = [solving.GENERATORS[name](bench_rng).prompt
                         for _ in range(samples)]
        report["tasks"][name] = compare(corpus_prompts, bench_prompts, threshold)
    return report


def _scratchpad_prompt(row: Dict[str, object]) -> str:
    """`build_scratchpad_math` returns a dict whose prompt field varies."""

    expression = str(row.get("expression", ""))
    if row.get("raw_prompt"):
        return expression
    return f"Solve this basic math problem: {expression}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else None)
    report = audit(args.samples, args.seed, args.threshold, tasks)

    flagged = 0
    for name, entry in report["tasks"].items():
        holes = entry["asked_but_never_taught"]
        if not holes:
            continue
        flagged += 1
        share = sum(h["share_of_benchmark"] for h in holes)
        print(f"{name}: {len(holes)} value(s) the corpus never teaches, "
              f"in up to {share:.1%} of benchmark prompts")
        for hole in holes[:8]:
            print(f"    {hole['value']:g}  in {hole['share_of_benchmark']:.1%} "
                  f"of benchmark prompts")
    if not flagged:
        print("no task asks about a value its corpus never teaches")
    if report["not_compared"]:
        print(f"\nnot compared (no corpus generator): "
              f"{', '.join(report['not_compared'])}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nreceipt -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
