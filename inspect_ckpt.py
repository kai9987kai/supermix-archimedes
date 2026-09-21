import torch, sys, json
from collections import Counter
path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
print("TYPE:", type(ck))
if isinstance(ck, dict):
    print("TOP KEYS:", list(ck.keys())[:40])
    for k, v in ck.items():
        if isinstance(v, dict) and k not in ("state_dict","model_state_dict","model","student_state_dict"):
            print(f"  [{k}] dict keys: {list(v.keys())[:30]}")
        elif not isinstance(v, (dict, torch.Tensor)):
            s = repr(v)
            print(f"  [{k}] = {s[:300]}")
    sd = None
    for k in ("state_dict","model_state_dict","model","student_state_dict","weights"):
        if k in ck and isinstance(ck[k], dict):
            sd = ck[k]; print(f"STATE DICT under [{k}]"); break
    if sd is None:
        if all(isinstance(v, torch.Tensor) for v in ck.values()):
            sd = ck; print("STATE DICT is top-level")
    if sd:
        total = 0; dtypes = Counter()
        print(f"N tensors: {len(sd)}")
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                total += v.numel(); dtypes[str(v.dtype)] += 1
        print(f"TOTAL PARAMS: {total:,}  dtypes: {dict(dtypes)}")
        # print prefixes summary
        prefixes = Counter(k.split(".")[0] for k in sd)
        print("PREFIXES:", dict(prefixes))
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else 80
        for i, (k, v) in enumerate(sd.items()):
            if i >= lim: print(f"... ({len(sd)-lim} more)"); break
            print(f"  {k:70s} {tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)} {v.dtype if isinstance(v, torch.Tensor) else ''}")
