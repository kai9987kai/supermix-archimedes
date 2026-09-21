"""Stage 1: build the grafted Supermix Archimedes checkpoint (no training).

    python archimedes/build_archimedes.py --out archimedes/checkpoints/supermix_archimedes_grafted.pt

Every graft is function-preserving: the saved model's logits equal v93's until
stage 2 (train_archimedes.py) opens the gates. The receipt records what was
grafted from where, with file digests, so the model card can be exact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "archimedes" / "src"))

from archimedes_core import (  # noqa: E402
    FLY_ROLES_22, OmniCore, build_archimedes_from_v93, graft_fly_into_cns, graft_v87_experts,
    load_fly_snapshot, save_archimedes, sha256_file, text_utils,
)
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

SOURCES = {
    "v93": ROOT / "models/supermix-v93/supermix_v93.pt",
    "v87": ROOT / "models/supermix-v87/supermix_v87.pt",
    "v48": ROOT / "models/omni-collective-v48-frontier/omni_collective_v48_frontier.pth",
    "v38": ROOT / "models/supermix-v38-native-image-xlite-fp16/champion_model_chat_v38_native_image_xlite_single_checkpoint_fp16.pth",
    "cns_npz": ROOT / "models/supermix-v93/data/malecns_hemispheres_384.npz",
}


def read_experience(path: Path, every: int, limit: int):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % every == 0:
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def calibrate_fly_core(fly, rows, steps: int = 600) -> dict:
    """Fit role_bias and vigor so the stateless port reproduces the JS engine's
    logged descending votes and consensus. Only those 110 numbers move."""
    obs = torch.tensor([r["obs"] for r in rows])
    js_probs = torch.tensor([r["probs"] for r in rows]).clamp_min(1e-6)
    js_probs = js_probs / js_probs.sum(-1, keepdim=True)
    js_desc = torch.tensor([[b["d"] for b in r["brains"]] for r in rows])

    def score(tag):
        with torch.no_grad():
            out = fly.brains_forward(obs)
            agree = float((out["probs"].argmax(-1) == js_probs.argmax(-1)).float().mean())
            kl = float(F.kl_div(out["probs"].clamp_min(1e-6).log(), js_probs, reduction="batchmean"))
            rmse = float((out["descending"] - js_desc).pow(2).mean().sqrt())
            return {"stage": tag, "consensus_argmax_agreement": agree, "consensus_kl": kl, "descending_rmse": rmse}

    before = score("port_uncalibrated")
    # closed-form intercept first, then a short joint fit of bias + vigor
    with torch.no_grad():
        out = fly.brains_forward(obs)
        fly.role_bias.add_((js_desc - out["descending"]).mean(0))
    params = [fly.role_bias, fly.vigor]
    opt = torch.optim.Adam(params, lr=1e-2)
    for p in fly.parameters():
        p.requires_grad_(p is fly.role_bias or p is fly.vigor)
    for step in range(steps):
        out = fly.brains_forward(obs)
        loss = F.mse_loss(out["descending"], js_desc) + 2.0 * F.kl_div(F.log_softmax(out["consensus"], -1), js_probs, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
    for p in fly.parameters():
        p.requires_grad_(True)
    after = score("port_calibrated")
    return {"rows": len(rows), "before": before, "after": after,
            "vigor": {r: round(float(v), 3) for r, v in zip(FLY_ROLES_22, fly.vigor.detach())},
            "role_bias_abs_mean": float(fly.role_bias.abs().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "archimedes/checkpoints/supermix_archimedes_grafted.pt"))
    ap.add_argument("--fly_run", default=str(ROOT / "fly_run"))
    ap.add_argument("--experts_per_layer", type=int, default=12)
    ap.add_argument("--calib_rows", type=int, default=4000)
    args = ap.parse_args()

    t0 = time.time()
    fly_dir = Path(args.fly_run)
    snap_path = fly_dir / "brain_state_final.json"
    fly_receipt = json.load(open(fly_dir / "receipt.json"))
    snap = load_fly_snapshot(snap_path)

    print("[build] loading v93 trunk")
    v93 = torch.load(SOURCES["v93"], map_location="cpu", weights_only=False)
    tok = text_utils.WordTokenizer.from_dict(v93["tokenizer"])
    model = build_archimedes_from_v93(v93, snap["config"])
    trunk_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("fly_core.", "omni_core.")))

    print("[build] grafting v87 experts")
    donor, tok87, p87 = load_talk_checkpoint(SOURCES["v87"])
    v87_receipt = graft_v87_experts(model, donor, tok, tok87, per_layer=args.experts_per_layer)
    print(f"   shared tokens {v87_receipt['shared_tokens']}  lift R2 to_wide {v87_receipt['fit_r2_to_wide']:.3f} to_narrow {v87_receipt['fit_r2_to_narrow']:.3f}")
    for li, info in v87_receipt["layers"].items():
        print(f"   trunk layer {li} <- v87 layer {info['donor_layer']}: {len(info['placed'])} experts dormant (bias {info['dormant_bias']:.2f} -> {info['target_bias']:.2f} when woken), alive {info['alive_after']}")
    del donor

    print("[build] loading fly snapshot into fly_core")
    fly_load = model.fly_core.load_snapshot(snap)
    print("   ", fly_load)
    rows = read_experience(fly_dir / "experience.jsonl", every=max(1, fly_receipt["experienceRows"] // args.calib_rows), limit=args.calib_rows)
    print(f"[build] calibrating fly_core on {len(rows)} logged rows")
    calib = calibrate_fly_core(model.fly_core, rows)
    print("   before:", calib["before"])
    print("   after :", calib["after"])

    print("[build] grafting 22 fly brains into the connectome core")
    cns_receipt = graft_fly_into_cns(model, model.fly_core, SOURCES["cns_npz"], step=int(v93["extra"].get("best_step", 0)))
    print(f"   slots {cns_receipt['slots'][:5]}...  alive {cns_receipt['alive_after']}  fly-fly edges {cns_receipt['fly_fly_edges']}  fly-bio edges {cns_receipt['fly_bio_edges']}")
    for role, labels in list(cns_receipt["analogues"].items())[:22]:
        print(f"      {role:16s} <-> {labels[:3]}")

    print("[build] loading omni sources")
    omni_receipt = model.omni_core.load_sources(SOURCES["v48"], SOURCES["v38"])
    print("   ", omni_receipt)

    # function preservation check
    base, _, _ = load_talk_checkpoint(SOURCES["v93"])
    ids, _ = tok.encode_turn("What is the impulse from a force of 46 N acting for 7 s?", None)
    x = torch.tensor([ids])
    model.eval()
    with torch.no_grad():
        diff = float((base(x).logits - model(x, omni_features=OmniCore.featurize(["x"])).logits).abs().max())
    print(f"[build] function preservation: max |logit diff| vs v93 = {diff:.2e}")
    del base

    total = sum(p.numel() for p in model.parameters())
    archimedes = {
        "with_omni": True,
        "fly_config": snap["config"],
        "stage": "grafted",
        "sources": {k: {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for k, p in SOURCES.items()},
        "fly_run": {"receipt": fly_receipt, "snapshot_sha256": sha256_file(snap_path), "snapshot_tick": snap["tickCount"]},
        "grafts": {"v87_experts": v87_receipt, "fly_core": {"load": fly_load, "calibration": calib},
                   "cns_nodes": {k: v for k, v in cns_receipt.items() if k != "events"}, "omni_core": omni_receipt},
        "params": {"total": total, "trunk": trunk_params, "fly_core": sum(p.numel() for p in model.fly_core.parameters()),
                   "omni_core": sum(p.numel() for p in model.omni_core.parameters())},
        "function_preservation_max_logit_diff": diff,
        "built_seconds": round(time.time() - t0, 1),
    }
    extra = dict(v93["extra"])
    extra.update({"run_name": "supermix_archimedes", "note": "stage 1: grafted, untrained", "warm_start": "Kai9987kai/supermix-v93"})
    out = Path(args.out)
    save_archimedes(out, model, tok, extra, archimedes)
    json.dump(archimedes, open(out.with_suffix(".receipt.json"), "w"), indent=1, default=str)
    print(f"[build] saved {out} ({out.stat().st_size / 1e6:.1f} MB), params {total:,} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
