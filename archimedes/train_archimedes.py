"""Stage 2: distil and fine-tune the grafted Archimedes checkpoint.

    python archimedes/train_archimedes.py --steps 400 --batch 8

Corpus
    replay   science / code / arithmetic rows regenerated with v93's own corpus
             builders (the trunk's task families; prevents forgetting)
    fly      rows written from the Fly Lab experience log: the 14 glomerular
             senses and weather of one agent, the reply is the syncytium's
             gated consensus and the action the agent executed
Teachers
    v93      self-distillation on replay rows (KL at T=2) anchors the trunk
    v87      cross-vocabulary distillation on rows v87 fully covers: its logits
             are scattered onto v93's ids through the token-string map (the
             two tokenizers segment identically, v87's vocabulary is a subset)
    fly lab  the logged consensus supervises fly_core directly (KL), and the
             logged senses supervise fly_core.sense() so the brains can read a
             described scene off the trunk (MSE)
Schedule
    grafted v87 experts are woken between 10% and 50% of training with their
    router bias annealed from dormant to the alive median; every gate starts
    at zero. Trunk lr is small, grafts get a larger one, omni encoders stay
    frozen so v48's classifier and v38's image decoder remain exact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "archimedes" / "src"))

from archimedes_core import (  # noqa: E402
    FLY_ACTIONS, OmniCore, load_archimedes, save_archimedes, text_utils, token_index_map, wake_grafted_experts,
)
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

FLY_WORDS = [
    "Fly", "fly", "lab", "agent", "agent1", "agent2", "agent3", "FORAGE", "EVADE", "SCOUT", "PIONEER", "CONSOLIDATE",
    "MAP_BEACON", "Senses", "senses", "threat", "bearing", "peer", "flow", "antennae", "Weather", "weather", "humidity",
    "wind", "rain", "Which", "way", "does", "move", "moves", "reflex", "goal", "explore", "support", "votes", "vote",
    "consensus", "up", "down", "left", "right", "action", "syncytium", "brains", "harvester", "sentinel", "cartographer",
    "forager", "navigator", "predator", "refuge", "energy", "food", "distance", "C",
]


def pct(v: float) -> str:
    """Senses as signed integer percents: half the tokens of a 2-decimal float."""
    return str(int(round(float(v) * 100)))


def fly_row(r: Dict[str, Any]) -> Tuple[str, str]:
    """One experience row as a (prompt, reply) pair that fits v93's 128-token context."""
    o = r["obs"]
    e = r["env"]
    p = r["probs"]
    a = int(r["action"])
    user = (f"Fly {r['agent']} {r['regime']}: x {pct(o[0])} y {pct(o[1])} food {pct(o[2])} threat {pct(o[3])} energy {pct(o[4])} "
            f"dist {pct(o[5])} food bearing {pct(o[6])} {pct(o[7])} threat bearing {pct(o[8])} {pct(o[9])} peer {pct(o[10])} {pct(o[11])} "
            f"antennae {pct(o[12])} {pct(o[13])} temp {e['t']:.0f} humidity {pct(e['h'])} wind {pct(e['wx'])} {pct(e['wy'])} "
            f"rain {pct(e['rain'])} predator {pct(e['pred'])}. Which way?")
    reply = (f"threat {pct(o[3])} food {pct(o[2])} energy {pct(o[4])}, votes up {pct(p[0])} down {pct(p[1])} left {pct(p[2])} right {pct(p[3])}, "
             f"the fly moves {FLY_ACTIONS[a]}, action {a}")
    return user, reply


def load_replay(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for name in ("omni", "code", "math"):
        p = root / "corpus" / f"{name}.jsonl"
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            rows.append({"user": r["user"], "assistant": r["assistant"], "kind": "replay", "task": r.get("task", name)})
    return rows


def load_fly(root: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    path = root / "fly_run" / "experience.jsonl"
    total = sum(1 for _ in open(path, encoding="utf-8"))
    rng = random.Random(seed)
    keep = set(rng.sample(range(total), min(n, total)))
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in keep:
                r = json.loads(line)
                u, a = fly_row(r)
                rows.append({"user": u, "assistant": a, "kind": "fly", "task": "fly_" + r["regime"].lower(),
                             "obs": r["obs"], "probs": r["probs"], "action": int(r["action"])})
    return rows


@torch.no_grad()
def extend_vocab(model, tok: text_utils.WordTokenizer, texts: List[str], max_new: int) -> Tuple[text_utils.WordTokenizer, int]:
    """Append tokens the corpus needs (v93 D5 rule: existing ids never move) and
    grow the tied embedding, new rows at the mean embedding plus small noise."""
    new_tok = text_utils.WordTokenizer.extend(tok, texts, max_new=max_new, min_count=2)
    added = new_tok.vocab_size - tok.vocab_size
    if added <= 0:
        return tok, 0
    old = model.embed_tokens.weight
    mean, std = old.mean(0, keepdim=True), old.std(0, keepdim=True)
    new_rows = mean + 0.1 * std * torch.randn(added, old.shape[1])
    weight = torch.nn.Parameter(torch.cat([old.detach(), new_rows], 0))
    model.embed_tokens.weight = weight
    model.embed_tokens.num_embeddings = weight.shape[0]
    model.lm_head.weight = weight  # tied
    model.lm_head.out_features = weight.shape[0]
    model.config.vocab_size = weight.shape[0]
    return new_tok, added


def encode_rows(rows, tok, seq_len: int):
    xs, ys, meta = [], [], []
    dropped = 0
    for r in rows:
        ids, plen = tok.encode_turn(r["user"], r["assistant"])
        if len(ids) > seq_len:
            dropped += 1
            continue
        labels = [-100] * plen + ids[plen:]
        pad = seq_len - len(ids)
        xs.append(ids + [text_utils.PAD] * pad)
        ys.append(labels + [-100] * pad)
        meta.append(r)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long), meta, dropped


def kd_loss(student_logits, teacher_logits, labels, T: float = 2.0):
    mask = labels != -100
    if mask.sum() == 0:
        return student_logits.new_zeros(())
    s = F.log_softmax(student_logits[mask] / T, -1)
    t = F.softmax(teacher_logits[mask] / T, -1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


class TeacherBank:
    """v93 (self) and v87 (cross-vocab) logits, computed lazily per batch."""

    def __init__(self, tok: text_utils.WordTokenizer, root: Path, want_v87: bool):
        self.v93, self.t93, _ = load_talk_checkpoint(root / "models/supermix-v93/supermix_v93.pt")
        self.v93.eval()
        self.v87 = self.t87 = None
        if want_v87:
            self.v87, self.t87, _ = load_talk_checkpoint(root / "models/supermix-v87/supermix_v87.pt")
            self.v87.eval()
            src, dst = token_index_map(self.t87, tok)
            self.map87 = torch.tensor(dst)  # v87 id -> student id
            self.unk87 = text_utils.UNK
        self.vocab = tok.vocab_size

    @torch.no_grad()
    def v93_logits(self, x):
        out = self.v93(x).logits
        if out.shape[-1] < self.vocab:
            out = F.pad(out, (0, self.vocab - out.shape[-1]), value=-1e4)
        return out

    @torch.no_grad()
    def v87_logits(self, rows, seq_len: int):
        """Teacher logits scattered onto student ids; None rows = not covered."""
        outs = []
        for r in rows:
            ids, plen = self.t87.encode_turn(r["user"], r["assistant"])
            reply_ids = ids[plen:]
            if self.unk87 in reply_ids or len(ids) > seq_len:
                outs.append(None)
                continue
            x = torch.tensor([ids + [text_utils.PAD] * (seq_len - len(ids))])
            lg = self.v87(x).logits[0]  # (T, 8635)
            full = lg.new_full((seq_len, self.vocab), -1e4)
            full[:, self.map87] = lg
            outs.append(full)
        return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(ROOT / "archimedes/checkpoints/supermix_archimedes_grafted.pt"))
    ap.add_argument("--out", default=str(ROOT / "archimedes/checkpoints/supermix_archimedes.pt"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--fly_rows", type=int, default=4000)
    ap.add_argument("--lr_trunk", type=float, default=3e-5)
    ap.add_argument("--lr_graft", type=float, default=2e-4)
    ap.add_argument("--kd93", type=float, default=0.5)
    ap.add_argument("--kd87", type=float, default=0.25)
    ap.add_argument("--fly_aux", type=float, default=1.0)
    ap.add_argument("--wake", type=str, default="0.1,0.5")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max_minutes", type=float, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    t0 = time.time()

    model, tok, payload = load_archimedes(args.inp)
    receipt = payload["archimedes"]
    print(f"[train] loaded {args.inp}: {sum(p.numel() for p in model.parameters()):,} params, vocab {tok.vocab_size}")

    replay = load_replay(ROOT)
    fly = load_fly(ROOT, args.fly_rows, args.seed)
    rows = replay + fly
    random.shuffle(rows)
    n_dev = max(64, len(rows) // 20)
    dev, train = rows[:n_dev], rows[n_dev:]
    print(f"[train] rows: replay {len(replay)} fly {len(fly)} -> train {len(train)} dev {len(dev)}")

    tok, added = extend_vocab(model, tok, [r["user"] + " " + r["assistant"] for r in fly] + [" ".join(FLY_WORDS)], max_new=160)
    print(f"[train] vocabulary extended by {added} -> {tok.vocab_size}")
    x_tr, y_tr, m_tr, d1 = encode_rows(train, tok, args.seq)
    x_dv, y_dv, m_dv, d2 = encode_rows(dev, tok, args.seq)
    unk = float((x_tr == text_utils.UNK).float().sum() / (x_tr != text_utils.PAD).float().sum())
    print(f"[train] packed train {tuple(x_tr.shape)} dev {tuple(x_dv.shape)} dropped {d1}+{d2} over-length; unk rate {unk:.4f}")
    feats_tr = OmniCore.featurize([r["user"] for r in m_tr])
    feats_dv = OmniCore.featurize([r["user"] for r in m_dv])
    is_fly_tr = torch.tensor([r["kind"] == "fly" for r in m_tr])
    obs_tr = torch.tensor([r.get("obs", [0.0] * 14) for r in m_tr])
    probs_tr = torch.tensor([r.get("probs", [0.25] * 4) for r in m_tr]).clamp_min(1e-6)
    probs_tr = probs_tr / probs_tr.sum(-1, keepdim=True)

    teachers = TeacherBank(tok, ROOT, want_v87=args.kd87 > 0)
    print("[train] teachers ready: v93" + (" + v87" if teachers.v87 is not None else ""))

    # parameter groups: omni encoders frozen; grafts fast; trunk slow
    graft_slots = {int(li): info["slots"] for li, info in receipt["grafts"]["v87_experts"]["layers"].items()}
    frozen, graft_params, trunk_params = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("omni_core.") and not (name.startswith("omni_core.to_trunk") or name.startswith("omni_core.gate") or name.startswith("omni_core.bridge_norm")):
            p.requires_grad_(False); frozen.append(name); continue
        is_graft = name.startswith(("fly_core.", "omni_core.", "cns_core."))
        if not is_graft and ".mlp.experts." in name:
            li, ei = int(name.split(".")[1]), int(name.split(".")[4])
            is_graft = ei in graft_slots.get(li, [])
        (graft_params if is_graft else trunk_params).append(p)
    print(f"[train] trainable: trunk {sum(p.numel() for p in trunk_params):,} graft {sum(p.numel() for p in graft_params):,} frozen {len(frozen)} tensors")
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": args.lr_trunk, "weight_decay": 0.01},
        {"params": graft_params, "lr": args.lr_graft, "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    warm = max(1, int(0.05 * args.steps))

    def lr_scale(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, args.steps - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    wake0, wake1 = [float(v) for v in args.wake.split(",")]

    @torch.no_grad()
    def evaluate(tag):
        model.eval()
        tot, n = 0.0, 0
        fly_hit = fly_n = 0
        sense_err = 0.0
        for i in range(0, x_dv.shape[0], args.batch):
            xb, yb = x_dv[i:i + args.batch], y_dv[i:i + args.batch]
            out = model(xb, labels=yb, omni_features=feats_dv[i:i + args.batch])
            tot += float(out.loss) * xb.shape[0]; n += xb.shape[0]
            info = model.last_graft_info.get("fly", {})
            for j, r in enumerate(m_dv[i:i + args.batch]):
                if r["kind"] != "fly":
                    continue
                fly_n += 1
                # does fly_core, fed the true senses, agree with the executed action's syncytium vote?
                pr = model.fly_core.brains_forward(torch.tensor([r["obs"]]))["probs"][0]
                fly_hit += int(pr.argmax() == torch.tensor(r["probs"]).argmax())
                if "obs" in info:
                    sense_err += float((info["obs"][j] - torch.tensor(r["obs"])).abs().mean())
        model.train()
        rep = {"dev_loss": tot / max(1, n), "fly_port_agree": fly_hit / max(1, fly_n), "sense_mae": sense_err / max(1, fly_n)}
        rep.update(model.gate_report())
        print(f"[eval {tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
        return rep

    history = [evaluate("start")]
    model.train()
    t_train = time.time()
    idx = torch.randperm(x_tr.shape[0])
    ptr = 0
    step = 0
    while step < args.steps:
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("[train] wall budget reached"); break
        if ptr + args.batch > idx.numel():
            idx = torch.randperm(x_tr.shape[0]); ptr = 0
        bi = idx[ptr:ptr + args.batch]; ptr += args.batch
        xb, yb, fb = x_tr[bi], y_tr[bi], feats_tr[bi]
        rows_b = [m_tr[i] for i in bi.tolist()]
        prog = step / max(1, args.steps)
        if wake0 <= prog:
            wake_grafted_experts(model, receipt["grafts"]["v87_experts"], (prog - wake0) / max(1e-6, wake1 - wake0))
        for g, base in zip(opt.param_groups, (args.lr_trunk, args.lr_graft)):
            g["lr"] = base * lr_scale(step)

        out = model(xb, labels=yb, omni_features=fb)
        loss_lm = out.loss
        losses = {"lm": float(loss_lm)}
        loss = loss_lm
        # v93 self-distillation on replay rows
        replay_mask = ~is_fly_tr[bi]
        if args.kd93 > 0 and replay_mask.any():
            t93 = teachers.v93_logits(xb[replay_mask])
            l = kd_loss(out.logits[replay_mask], t93, yb[replay_mask])
            loss = loss + args.kd93 * l; losses["kd93"] = float(l)
        # v87 cross-vocab distillation
        if args.kd87 > 0 and replay_mask.any():
            t87s = teachers.v87_logits([r for r, m in zip(rows_b, replay_mask.tolist()) if m], args.seq)
            sel = [k for k, t in enumerate(t87s) if t is not None]
            if sel:
                st = out.logits[replay_mask][sel]; tt = torch.stack([t87s[k] for k in sel]); lb = yb[replay_mask][sel]
                l = kd_loss(st, tt, lb); loss = loss + args.kd87 * l; losses["kd87"] = float(l)
        # fly lab: brains match the logged consensus; senses readable from text
        fly_mask = is_fly_tr[bi]
        if args.fly_aux > 0 and fly_mask.any():
            fo = model.fly_core.brains_forward(obs_tr[bi][fly_mask])
            l_port = F.kl_div(F.log_softmax(fo["consensus"], -1), probs_tr[bi][fly_mask], reduction="batchmean")
            sensed = model.last_graft_info["fly"]["obs"][fly_mask]
            l_sense = F.mse_loss(sensed, obs_tr[bi][fly_mask])
            loss = loss + args.fly_aux * (l_port + l_sense); losses["fly_port"] = float(l_port); losses["fly_sense"] = float(l_sense)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        step += 1
        if step % 10 == 0 or step == 1:
            el = time.time() - t_train
            print(f"[step {step}/{args.steps}] loss {float(loss):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in losses.items())
                  + f" lr {opt.param_groups[0]['lr']:.2e} wake {min(1.0, max(0.0, (prog - wake0) / max(1e-6, wake1 - wake0))):.2f} {el / step:.2f}s/step eta {(args.steps - step) * el / step / 60:.1f}m")
        if step % args.eval_every == 0:
            history.append({"step": step, **evaluate(str(step))})

    wake_grafted_experts(model, receipt["grafts"]["v87_experts"], 1.0)
    final = evaluate("final")
    history.append({"step": step, **final})
    model.eval()
    receipt = dict(receipt)
    receipt.update({
        "stage": "trained",
        "training": {
            "steps": step, "batch": args.batch, "seq": args.seq, "lr_trunk": args.lr_trunk, "lr_graft": args.lr_graft,
            "kd93": args.kd93, "kd87": args.kd87, "fly_aux": args.fly_aux, "wake": args.wake, "seed": args.seed,
            "rows": {"replay": len(replay), "fly": len(fly), "train": int(x_tr.shape[0]), "dev": int(x_dv.shape[0])},
            "vocab_added": added, "vocab_size": tok.vocab_size, "unk_rate": unk,
            "wall_minutes": round((time.time() - t0) / 60, 2), "history": history, "final": final,
            "omni_frozen_tensors": len(frozen),
        },
    })
    extra = dict(payload["extra"])
    extra.update({"note": "stage 2: distilled + fine-tuned", "steps": step, "best_dev_loss": final["dev_loss"]})
    save_archimedes(Path(args.out), model, tok, extra, receipt)
    json.dump(receipt, open(Path(args.out).with_suffix(".receipt.json"), "w"), indent=1, default=str)
    print(f"[train] saved {args.out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
