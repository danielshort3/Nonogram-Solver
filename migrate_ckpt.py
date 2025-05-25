import torch, glob, os
from collections import OrderedDict

# Grab latest checkpoint file
src = sorted(
    glob.glob("models/checkpoint_*.pth"),
    key=lambda p: int(p.split("_")[-1].split(".")[0]),
    reverse=True,
)[0]
dst = src.replace(".pth", "_renamed.pth")

ckpt = torch.load(src, map_location="cpu", weights_only=False)
old_state = ckpt["model_state_dict"]
new_state = OrderedDict()

# rename keys
for k, v in old_state.items():
    k2 = k.replace("row_clue_transformer", "row_trans") \
          .replace("col_clue_transformer", "col_trans")
    new_state[k2] = v

ckpt["model_state_dict"] = new_state
torch.save(ckpt, dst)
print(f"✔ wrote {dst}")
