#!/usr/bin/env python3
"""Unified local UI for Supermix Archimedes: text, image, classifier and FlyCore."""
from __future__ import annotations

import argparse, os, sys, threading, time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "archimedes" / "src"
for p in (SRC, SRC / "champion"):
    sys.path.insert(0, str(p))

from archimedes_core import FLY_ACTIONS, FLY_ROLES_22, OmniCore, load_archimedes
import mimomix_text as text_utils

HF_REPO = "Kai9987kai/archimedes-final-model"
FLY_CHANNELS = [
    ("x position",0,1,.5),("y position",0,1,.5),("food signal",0,1,.5),
    ("threat signal",0,1,.1),("energy",0,1.5,1),("nearest-food distance",0,1,.4),
    ("food bearing sin",-1,1,0),("food bearing cos",-1,1,1),
    ("threat bearing sin",-1,1,0),("threat bearing cos",-1,1,1),
    ("peer optic-flow x",-1,1,0),("peer optic-flow y",-1,1,0),
    ("left antenna",0,1,.2),("right antenna",0,1,.2),
]


def resolve_device(name: str) -> torch.device:
    if name != "auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(explicit: Optional[str], repo: str) -> Path:
    for p in [explicit, os.getenv("ARCHIMEDES_CHECKPOINT"),
              ROOT / "archimedes/checkpoints/supermix_archimedes.pt",
              ROOT / "checkpoints/supermix_archimedes.pt"]:
        if p and Path(p).expanduser().is_file(): return Path(p).expanduser().resolve()
    files = [f for f in list_repo_files(repo) if f.lower().endswith((".pt",".pth"))]
    if not files: raise FileNotFoundError(f"No .pt/.pth checkpoint found in {repo}")
    def rank(n):
        l=n.lower(); return (20*("archimedes" in l)+10*("final" in l)-15*("grafted" in l)-20*("smoke" in l),-n.count("/"))
    return Path(hf_hub_download(repo_id=repo, filename=max(files,key=rank))).resolve()


def to_pil(t: torch.Tensor, size: int) -> Image.Image:
    t=t.detach().float().clamp(0,1).cpu()
    a=t.permute(1,2,0).mul(255).round().byte().numpy()
    im=Image.fromarray(a, "RGB")
    return im if size==64 else im.resize((size,size), Image.Resampling.NEAREST)


class Runtime:
    def __init__(self, checkpoint: Path, device: torch.device):
        self.lock=threading.RLock(); self.device=device; self.checkpoint=checkpoint
        t=time.perf_counter(); self.model,self.tok,self.payload=load_archimedes(checkpoint,map_location="cpu")
        self.model.to(device).eval(); self.load_s=time.perf_counter()-t

    def features(self, prompt): return OmniCore.featurize([prompt]).to(self.device)

    @torch.inference_mode()
    def text(self,prompt,max_tokens,cycles,use_fly,use_omni):
        prompt=(prompt or "").strip()
        if not prompt: return "",{"error":"Enter a prompt"}
        cycles=int(cycles) or None; max_tokens=int(max_tokens)
        with self.lock:
            ids,_=self.tok.encode_turn(prompt,None)
            x=torch.tensor([ids],dtype=torch.long,device=self.device)
            omni=self.features(prompt) if use_omni and self.model.omni_core is not None else None
            t=time.perf_counter()
            out=self.model(x,use_cache=True,thinking_cycles=cycles,adaptive_thinking=False,return_mtp=False,
                           past_length=0,omni_features=omni,use_fly=bool(use_fly))
            past=out.past_key_values; pos=x.shape[1]; token=out.logits[:,-1].argmax(-1,keepdim=True); emitted=[]
            for _ in range(max_tokens):
                emitted.append(token)
                if int(token[0,0])==int(text_utils.EOS): break
                out=self.model(token,past_key_values=past,use_cache=True,thinking_cycles=cycles,adaptive_thinking=False,
                               return_mtp=False,past_length=pos,omni_features=omni,use_fly=bool(use_fly))
                past=out.past_key_values; pos+=1; token=out.logits[:,-1].argmax(-1,keepdim=True)
            y=torch.cat(emitted,1) if emitted else x[:,:0]; dt=time.perf_counter()-t
            reply=self.tok.decode(y[0].cpu().tolist()).strip()
            return reply,{"tokens":int(y.shape[1]),"latency_ms":round(dt*1000,2),"tokens_per_second":round(y.shape[1]/max(dt,1e-9),3),
                          "fly_enabled":bool(use_fly),"omni_enabled":bool(omni is not None),"gates":self.model.gate_report()}

    @torch.inference_mode()
    def image(self,prompt,size):
        prompt=(prompt or "").strip()
        if not prompt:return None,{"error":"Enter an image prompt"}
        if self.model.omni_core is None:return None,{"error":"This checkpoint has no OmniCore"}
        with self.lock:
            t=time.perf_counter(); img=self.model.omni_core.render(self.features(prompt))[0]; dt=time.perf_counter()-t
            return to_pil(img,int(size)),{"native_resolution":"64x64","display_resolution":f"{int(size)}x{int(size)}","latency_ms":round(dt*1000,2)}

    @torch.inference_mode()
    def classify(self,prompt):
        prompt=(prompt or "").strip()
        if not prompt:return [],{"error":"Enter text"}
        with self.lock:
            t=time.perf_counter(); logits=self.model.omni_core.classify(self.features(prompt))[0].float(); p=torch.softmax(logits,-1); dt=time.perf_counter()-t
            order=torch.argsort(p,descending=True)
            rows=[[f"class_{int(i)}",round(float(p[i]),8),round(float(logits[i]),8)] for i in order]
            return rows,{"top_class":rows[0][0],"top_probability":rows[0][1],"latency_ms":round(dt*1000,2),
                         "note":"No semantic class-name mapping is stored, so neutral class indices are used."}

    @torch.inference_mode()
    def fly(self,*values):
        vals=[float(v) for v in values]
        with self.lock:
            t=time.perf_counter(); r=self.model.fly_core.brains_forward(torch.tensor([vals],dtype=torch.float32,device=self.device)); dt=time.perf_counter()-t
            probs=r["probs"][0].float(); cons=r["consensus"][0].float(); desc=r["descending"][0].float(); ai=int(probs.argmax())
            actions=[[FLY_ACTIONS[i],round(float(probs[i]),8),round(float(cons[i]),8)] for i in range(4)]
            brains=[]
            for j,role in enumerate(FLY_ROLES_22):
                v=desc[j]; wi=int(v.argmax()); brains.append([role,FLY_ACTIONS[wi],*[round(float(x),6) for x in v]])
            return FLY_ACTIONS[ai].upper(),actions,brains,{"selected_index":ai,"confidence":round(float(probs[ai]),8),"latency_ms":round(dt*1000,2)}

    def diagnostics(self):
        a=self.payload.get("archimedes",{})
        return {"checkpoint":str(self.checkpoint),"schema":self.payload.get("schema"),"stage":a.get("stage"),"device":str(self.device),
                "load_seconds":round(self.load_s,3),"parameters":sum(p.numel() for p in self.model.parameters()),"vocabulary":self.tok.vocab_size,
                "fly_roles":list(FLY_ROLES_22),"gates":self.model.gate_report(),"receipt_params":a.get("params"),"training_final":a.get("training",{}).get("final")}


def build_ui(rt: Runtime):
    with gr.Blocks(title="Supermix Archimedes — Unified Inference Lab") as app:
        gr.Markdown("# Supermix Archimedes — Unified Inference Lab\nExpose the final checkpoint's **text, native image, v48 classification and 22-brain FlyCore** outputs in one local app.")
        with gr.Tab("Text / Chat"):
            p=gr.Textbox(lines=5,label="Prompt",value="What is the impulse from a force of 46 N acting for 7 seconds?")
            with gr.Row(): m=gr.Slider(1,256,96,step=1,label="Maximum new tokens"); c=gr.Slider(0,8,0,step=1,label="Thinking cycles")
            with gr.Row(): f=gr.Checkbox(True,label="Use FlyCore"); o=gr.Checkbox(True,label="Use OmniCore")
            b=gr.Button("Generate",variant="primary"); out=gr.Textbox(lines=10,label="Response"); meta=gr.JSON(label="Telemetry")
            b.click(rt.text,[p,m,c,f,o],[out,meta])
        with gr.Tab("Native Image"):
            gr.Markdown("Uses the preserved **v38 Native Image XLite** decoder. True neural output is **64×64 RGB**; enlargement is display-only.")
            p=gr.Textbox(lines=3,label="Image prompt",value="simple water cycle icon with blue water, white cloud, rain, and a yellow sun")
            s=gr.Radio([64,256,512],value=512,label="Display size"); b=gr.Button("Render",variant="primary"); im=gr.Image(type="pil"); meta=gr.JSON()
            b.click(rt.image,[p,s],[im,meta])
        with gr.Tab("v48 Classification"):
            gr.Markdown("The preserved v48 head outputs 10 logits. The checkpoint has no semantic class-name mapping, so labels remain `class_0`…`class_9`.")
            p=gr.Textbox(lines=4,label="Input text",value="A simple science question about momentum"); b=gr.Button("Classify",variant="primary")
            table=gr.Dataframe(headers=["class","probability","logit"],datatype=["str","number","number"],interactive=False); meta=gr.JSON()
            b.click(rt.classify,p,[table,meta])
        with gr.Tab("22-Brain FlyCore"):
            gr.Markdown("Drive the original 14-channel Fly observation and inspect consensus plus all 22 descending-vote vectors.")
            sliders=[]
            with gr.Row():
                with gr.Column():
                    for n,lo,hi,d in FLY_CHANNELS[:7]: sliders.append(gr.Slider(lo,hi,d,step=.01,label=n))
                with gr.Column():
                    for n,lo,hi,d in FLY_CHANNELS[7:]: sliders.append(gr.Slider(lo,hi,d,step=.01,label=n))
            b=gr.Button("Run FlyCore",variant="primary"); action=gr.Textbox(label="Selected action")
            acts=gr.Dataframe(headers=["action","probability","consensus_logit"],datatype=["str","number","number"],interactive=False)
            brains=gr.Dataframe(headers=["role","local winner","up","down","left","right"],datatype=["str","str","number","number","number","number"],interactive=False); meta=gr.JSON()
            b.click(rt.fly,sliders,[action,acts,brains,meta])
        with gr.Tab("Diagnostics"):
            b=gr.Button("Refresh"); d=gr.JSON(value=rt.diagnostics()); b.click(rt.diagnostics,outputs=d)
        gr.Markdown("---\nThe image and classifier branches are experimental inherited components; this app exposes what the checkpoint already contains rather than claiming general-purpose VLM capability.")
    return app


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint"); ap.add_argument("--repo",default=HF_REPO); ap.add_argument("--device",default="auto"); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=7860); ap.add_argument("--share",action="store_true"); a=ap.parse_args()
    ckpt=resolve_checkpoint(a.checkpoint,a.repo); device=resolve_device(a.device); print(f"Loading {ckpt} on {device}...")
    rt=Runtime(ckpt,device); print(f"Loaded {sum(p.numel() for p in rt.model.parameters()):,} parameters in {rt.load_s:.2f}s")
    app=build_ui(rt); app.queue(); app.launch(server_name=a.host,server_port=a.port,share=a.share,show_error=True)

if __name__=="__main__": main()
