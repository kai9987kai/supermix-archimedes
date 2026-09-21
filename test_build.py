import sys, json, time, torch
sys.path.insert(0, "archimedes/src")
from archimedes_core import *
from train_mimomix_talk import load_talk_checkpoint, generate_reply

t=time.time()
v93 = torch.load("models/supermix-v93/supermix_v93.pt", map_location="cpu", weights_only=False)
snap = load_fly_snapshot(Path("fly_run/brain_state_final.json"))
model = build_archimedes_from_v93(v93, snap["config"])
tok = text_utils.WordTokenizer.from_dict(v93["tokenizer"])
print("built in", round(time.time()-t,1), "s; params:", sum(p.numel() for p in model.parameters()))
rep = model.fly_core.load_snapshot(snap); print("fly snapshot loaded:", rep)
rep = model.omni_core.load_sources(Path("models/omni-collective-v48-frontier/omni_collective_v48_frontier.pth"), Path("models/supermix-v38-native-image-xlite-fp16/champion_model_chat_v38_native_image_xlite_single_checkpoint_fp16.pth"))
print("omni loaded:", rep)

base, _, _ = load_talk_checkpoint("models/supermix-v93/supermix_v93.pt")
ids, _ = tok.encode_turn("What is 47 x 6?", None); x = torch.tensor([ids])
model.eval()
with torch.no_grad():
    a = base(x).logits; b = model(x, omni_features=OmniCore.featurize(["What is 47 x 6?"])).logits
print("function-preserving at birth: max |logit diff| =", float((a-b).abs().max()))
print("graft info keys:", list(model.last_graft_info.keys()), "fly probs:", [round(v,3) for v in model.last_graft_info["fly"]["probs"][0].tolist()])

# omni branch reproduces the originals exactly
sys.path.insert(0, "models/supermix-v38-native-image-xlite-fp16")
from model_native_image_xlite_v38 import ChampionNetUltraExpertNativeImageExtraLite
from chat_pipeline import text_to_model_input
v38 = ChampionNetUltraExpertNativeImageExtraLite(); v38.load_state_dict({k:v.float() for k,v in torch.load("models/supermix-v38-native-image-xlite-fp16/champion_model_chat_v38_native_image_xlite_single_checkpoint_fp16.pth", map_location="cpu").items()}); v38.eval()
p = "simple water cycle icon with blue water, white cloud, rain, and a yellow sun"
with torch.no_grad():
    img_orig = v38.forward_image(text_to_model_input(p, feature_mode="context_mix_v4"))
    feats = OmniCore.featurize([p])
    img_new = model.omni_core.render(feats)
    cls = model.omni_core.classify(feats)
print("v38 image preserved: max|diff| =", float((img_orig-img_new).abs().max()), "| v48 class logits:", [round(v,2) for v in cls[0].tolist()])

# FlyCore port validation against JS engine outputs
rows = []
with open("fly_run/experience.jsonl") as f:
    for i, line in enumerate(f):
        if i % 97 == 0: rows.append(json.loads(line))
        if len(rows) >= 1500: break
obs = torch.tensor([r["obs"] for r in rows]); js_probs = torch.tensor([r["probs"] for r in rows]); js_act = torch.tensor([r["action"] for r in rows])
js_desc = torch.tensor([[b["d"] for b in r["brains"]] for r in rows])
with torch.no_grad():
    out0 = model.fly_core.brains_forward(obs)
    print("PORT before bias fit: argmax match vs JS probs %.3f, vs executed action %.3f, mean KL %.4f" % (
        float((out0["probs"].argmax(-1)==js_probs.argmax(-1)).float().mean()), float((out0["probs"].argmax(-1)==js_act).float().mean()),
        float(F.kl_div(out0["probs"].log(), js_probs, reduction="batchmean"))))
    resid = js_desc - out0["descending"]
    model.fly_core.role_bias.add_(resid.mean(0))
    out1 = model.fly_core.brains_forward(obs)
    print("PORT after bias fit : argmax match vs JS probs %.3f, vs executed action %.3f, mean KL %.4f" % (
        float((out1["probs"].argmax(-1)==js_probs.argmax(-1)).float().mean()), float((out1["probs"].argmax(-1)==js_act).float().mean()),
        float(F.kl_div(out1["probs"].log(), js_probs, reduction="batchmean"))))
    per_brain = (out1["descending"]-js_desc).pow(2).mean((0,2)).sqrt()
    print("per-brain RMSE:", {r: round(float(v),2) for r,v in zip(FLY_ROLES_22, per_brain)})
    print("JS desc std   :", {r: round(float(v),2) for r,v in zip(FLY_ROLES_22, js_desc.std((0,2)))})
    print("JS action dist:", torch.bincount(js_act, minlength=4).tolist(), " JS-probs argmax dist:", torch.bincount(js_probs.argmax(-1), minlength=4).tolist())
