from huggingface_hub import snapshot_download
import sys
repos = [
    "Kai9987kai/supermix-v93",
    "Kai9987kai/omni-collective-v48-frontier",
    "Kai9987kai/supermix-v87",
    "Kai9987kai/supermix-v38-native-image-xlite-fp16",
]
for r in repos:
    name = r.split("/")[1]
    print(f"downloading {r} -> models/{name}", flush=True)
    p = snapshot_download(repo_id=r, local_dir=f"models/{name}")
    print(f"  done: {p}", flush=True)
print("ALL DONE", flush=True)
