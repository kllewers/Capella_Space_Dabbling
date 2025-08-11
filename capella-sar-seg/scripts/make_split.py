# scripts/make_split.py
import json, os, random
random.seed(42)

img_dir = "data/tiles/images"
ids = sorted([f.replace(".npy","") for f in os.listdir(img_dir) if f.endswith(".npy")])
random.shuffle(ids)
n = len(ids)
train = ids[: int(n*0.9)]
val   = ids[int(n*0.9):]

with open("splits.json","w") as f:
    json.dump({"train": train, "val": val}, f, indent=2)

print(f"train={len(train)}, val={len(val)}")
