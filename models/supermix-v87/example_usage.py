"""Load supermix-v87 and ask it a question.

    python example_usage.py "What is 47 x 6?"
    python example_usage.py "Trace this Python and give x. x = 5
    for i in range(4): x = x + 7"

The model answers in the format it was trained on, so the question is
normalised first -- see the model card. `prompt_normaliser` prints what it
actually sent, and `answer_check` independently re-derives the answer, so a
wrong reply is reported as wrong rather than presented as fact. For a code
question `answer_check` re-derives by **running the snippet**.

`step_audit` then reads the reply's own working and reports the first step that
disagrees with exact arithmetic. That matters here: v87's `power` task writes a
confident, well-formed derivation whose middle steps are false, and the answer
alone does not show you that.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import answer_check  # noqa: E402
import prompt_normaliser  # noqa: E402
import step_audit  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

DEFAULT = "Trace this Python and give x. x = 5\nfor i in range(4): x = x + 7"


def main() -> int:
    question = " ".join(sys.argv[1:]) or DEFAULT

    model, tokenizer, _ = load_talk_checkpoint(str(HERE / "supermix_v87.pt"))
    model.eval()

    rewritten = prompt_normaliser.normalise(question)
    if rewritten.changed:
        print(f"asked as : {rewritten.prompt}  ({rewritten.rule})")

    result = generate_reply(model, tokenizer, rewritten.prompt, max_new_tokens=96)
    reply = result["reply"] if isinstance(result, dict) else str(result)
    print(f"reply    : {reply}")

    verdict = answer_check.check(rewritten.prompt, reply)
    if verdict is None:
        print("check    : NOT CHECKED (not a question this can verify)")
    elif verdict.correct:
        print(f"check    : CORRECT ({verdict.expected})")
    else:
        print(f"check    : WRONG (answered {verdict.predicted}, "
              f"expected {verdict.expected})")

    audit = step_audit.audit(reply)
    bad = audit.first_bad
    if bad is not None:
        print(f"working  : step {bad.position} is false -- '{bad.text}' "
              f"states {bad.stated:g}, arithmetic gives {bad.expected:g}")
    elif audit.written:
        print(f"working  : all {len(audit.written)} written steps check out "
              f"({audit.verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
