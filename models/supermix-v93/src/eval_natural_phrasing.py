"""Does the answer survive a phrasing the model has never seen?

## The question this exists to answer

Every benchmark in this project generates its prompts from the same templates
the corpus was built from, so all of them measure the model on its own dialect.
That has hidden a real weakness twice. v74 scored 0.894 on its own benchmark and
answered **0 of 5** naturally-typed questions. v80 scored 0.556 on hand-typed
questions against 0.778 for the same questions rewritten into the corpus format
(`output/v85_measurements/natural_phrasing.json`) -- a 22-point gap that exists
only because of how the question was worded.

`natural_phrasings.py` widens each task from four or five textbook templates to
fifteen, including the casual register a question actually arrives in. But
widening a bank and then scoring on that bank measures nothing: a model can
memorise fifteen templates as easily as five.

So `natural_phrasings.HELD_OUT_PER_TASK` withholds the last three forms of every
task from training, and this scores on exactly those. Thirty-six phrasings across
twelve tasks that no corpus has ever contained.

## Why it is paired

Each problem is asked twice with **the same operands and the same answer**:

    trained    Given mass 42 kg and acceleration 8 m/s^2, compute the force.
    held out   force please: 42 kg at 8 m/s^2

Any difference between the two columns is the wording and nothing else, so the
gap is the quantity of interest and McNemar's test on the discordant pairs is
the right test for it. A model that has learned the task rather than the
template scores the same in both columns.

No prompt normaliser runs here. Rewriting the question into the corpus format is
what this measures the need for -- v80 needed it for 16 of 18 questions -- so
applying it would erase the measurement.

    python source/eval_natural_phrasing.py --checkpoint output/v87_corpus/v87_corpus.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_omni_corpus as omni  # noqa: E402
import natural_phrasings  # noqa: E402
from eval_problem_solving import extract_answer, wilson_interval  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

REPORT_SCHEMA = "supermix-v88-natural-phrasing-v1"


def _number(value: object) -> str:
    """Match `build_omni_corpus._number` so a held-out form renders identically."""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_pairs(per_task: int, seed: int) -> List[Dict[str, object]]:
    """One problem per row, rendered in a trained form and a held-out form.

    The problem itself comes from the ordinary generator with phrasing off, so
    operands, answer and worked response are exactly what training produced.
    Only the two prompt strings differ.
    """

    was = omni.NATURAL_PHRASINGS
    pairs: List[Dict[str, object]] = []
    try:
        for task in sorted(omni.TASKS):
            if not natural_phrasings.held_out(task):
                continue
            # The same seed twice. The generator's RNG stream does not depend on
            # which phrasing is picked, so both passes draw the same operands and
            # produce the same answer -- only the prompt differs. That identity is
            # asserted below rather than assumed.
            omni.NATURAL_PHRASINGS = False
            rng = random.Random(f"{seed}:{task}")
            trained = [omni.TASKS[task](rng) for _ in range(per_task)]

            omni.NATURAL_PHRASINGS = True
            rng = random.Random(f"{seed}:{task}")
            with natural_phrasings.held_out_only():
                withheld = [omni.TASKS[task](rng) for _ in range(per_task)]

            for a, b in zip(trained, withheld):
                if a.answer != b.answer or a.response != b.response:
                    raise AssertionError(
                        f"{task}: the two passes drew different problems, so the "
                        "comparison would not be paired"
                    )
                pairs.append({
                    "task": task,
                    "answer": float(a.answer),
                    "trained_prompt": a.prompt,
                    "held_out_prompt": b.prompt,
                })
    finally:
        omni.NATURAL_PHRASINGS = was
    return pairs


def mcnemar(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar on the discordant pairs."""

    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="held-out phrasing benchmark")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per_task", type=int, default=25)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--cap", type=int, default=96)
    ap.add_argument("--output")
    ap.add_argument("--dump_replies")
    ap.add_argument("--normalise", action="store_true",
                    help=("run prompt_normaliser over both columns first. OFF by "
                          "default because needing the rewrite is what this "
                          "measures; on, it reports what the chat server "
                          "actually does, since that normalises by default"))
    args = ap.parse_args(argv)

    pairs = build_pairs(args.per_task, args.seed)
    tasks_with_holdout = sum(1 for t in omni.TASKS if natural_phrasings.held_out(t))
    forms = sum(len(natural_phrasings.held_out(t)) for t in sorted(omni.TASKS))
    print(f"{len(pairs)} problems, each asked in a trained and a held-out form")
    print(f"held-out phrasings: {forms} across {tasks_with_holdout} tasks")

    model, tokenizer, _ = load_talk_checkpoint(args.checkpoint)
    model.eval()

    rows = []
    for n, pair in enumerate(pairs, 1):
        row = dict(pair)
        for column in ("trained", "held_out"):
            asked = pair[f"{column}_prompt"]
            if args.normalise:
                import prompt_normaliser
                rewritten = prompt_normaliser.normalise(asked)
                row[f"{column}_rewritten"] = rewritten.changed
                row[f"{column}_rule"] = rewritten.rule
                asked = rewritten.prompt
            reply = generate_reply(model, tokenizer, asked,
                                   max_new_tokens=args.cap)
            text = reply["reply"] if isinstance(reply, dict) else str(reply)
            got = extract_answer(text)
            row[f"{column}_reply"] = text
            row[f"{column}_answer"] = got
            row[f"{column}_correct"] = (
                got is not None and abs(got - pair["answer"]) < 1e-6)
        rows.append(row)
        if n % 25 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)

    trained = sum(r["trained_correct"] for r in rows)
    held = sum(r["held_out_correct"] for r in rows)
    only_trained = sum(1 for r in rows
                       if r["trained_correct"] and not r["held_out_correct"])
    only_held = sum(1 for r in rows
                    if r["held_out_correct"] and not r["trained_correct"])
    p_value = mcnemar(only_trained, only_held)

    print()
    for label, correct in (("trained phrasing", trained),
                           ("HELD-OUT phrasing", held)):
        lo, hi = wilson_interval(correct, len(rows))
        print(f"  {label:18s} {correct:4d}/{len(rows)} = "
              f"{correct / len(rows):.4f} 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  trained-only wins {only_trained}, held-out-only wins {only_held}, "
          f"McNemar exact two-sided p = {p_value:.4f}")
    print(f"\n  the phrasing gap is {(trained - held) / len(rows):+.4f}; a model "
          "that learned the task rather than the template scores zero here")

    print("\n  per task (held-out / trained):")
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        h = sum(r["held_out_correct"] for r in group)
        t = sum(r["trained_correct"] for r in group)
        print(f"    {task:20s} {h:3d}/{len(group):<3d} vs {t:3d}/{len(group)}")

    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": args.checkpoint,
        "n": len(rows),
        "held_out_per_task": natural_phrasings.HELD_OUT_PER_TASK,
        "trained": {"correct": trained, "accuracy": trained / len(rows),
                    "wilson95": wilson_interval(trained, len(rows))},
        "held_out": {"correct": held, "accuracy": held / len(rows),
                     "wilson95": wilson_interval(held, len(rows))},
        "mcnemar": {"trained_only": only_trained, "held_out_only": only_held,
                    "p_value": p_value},
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.output}")
    if args.dump_replies:
        with Path(args.dump_replies).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"replies -> {args.dump_replies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
