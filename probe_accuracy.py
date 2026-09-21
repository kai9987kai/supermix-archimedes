"""Quick accuracy probe: answer_check on a fixed sample of generated problems."""
import sys, json, random, time, torch
sys.path.insert(0, "archimedes/src")
from archimedes_core import load_archimedes, OmniCore, text_utils
from train_mimomix_talk import load_talk_checkpoint, generate_reply
import answer_check

def sample_rows(n, seed=7):
    rows = []
    for f in ["corpus/omni.jsonl", "corpus/code.jsonl", "corpus/math.jsonl"]:
        rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    return rows[:n]

def probe(model, tok, rows, use_fly=True, omni=False, tag=""):
    correct = checked = 0; per_task = {}
    t = time.time()
    for r in rows:
        kw = {}
        if hasattr(model, "fly_core"):
            kw = {}  # generate_reply uses model(...) positional; grafts default: fly on, omni off
        res = generate_reply(model, tok, r["user"], max_new_tokens=96)
        v = answer_check.check(r["user"], res["reply"])
        task = r.get("task", "?")
        if v is None: continue
        checked += 1; correct += int(v.correct)
        per_task.setdefault(task, [0, 0]); per_task[task][0] += int(v.correct); per_task[task][1] += 1
    acc = correct / max(1, checked)
    print(f"[{tag}] acc {acc:.3f} ({correct}/{checked} checked of {len(rows)}) in {time.time()-t:.0f}s")
    return acc, per_task

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = sample_rows(n)
    which = sys.argv[2:] or ["v93", "grafted"]
    results = {}
    if "v93" in which:
        m, tok, _ = load_talk_checkpoint("models/supermix-v93/supermix_v93.pt"); results["v93"] = probe(m, tok, rows, tag="v93")
    if "v87" in which:
        m, tok, _ = load_talk_checkpoint("models/supermix-v87/supermix_v87.pt"); results["v87"] = probe(m, tok, rows, tag="v87")
    for path in [w for w in which if w.endswith(".pt")]:
        m, tok, _ = load_archimedes(path); results[path] = probe(m, tok, rows, tag=path.split("/")[-1])
    if "grafted" in which:
        m, tok, _ = load_archimedes("archimedes/checkpoints/supermix_archimedes_grafted.pt"); results["grafted"] = probe(m, tok, rows, tag="grafted")
    json.dump(results, open("probe_results.json", "w"), indent=1)
    for k, (acc, pt) in results.items():
        print(k, {t: f"{c}/{n}" for t, (c, n) in sorted(pt.items())})
