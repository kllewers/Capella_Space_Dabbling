#!/usr/bin/env python3
import os, random, numpy as np, torch, matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

IMG_DIR = "data/tiles/images"
MSK_DIR = "data/tiles/masks"
WEIGHTS = "models/unet_mnetv3_sar.pt"

# ---- selection knobs ----
VAR_THR = 5e-5          # minimum variance on SAR ch0 to avoid flat/black tiles
REQUIRE_POS_MASK = False  # set True only if you have real (non-zero) masks

def has_texture_and_mask(stem, require_pos=False, var_thr=VAR_THR):
    x = np.load(os.path.join(IMG_DIR, stem + ".npy"))
    msk_stem = stem.replace("img_", "msk_", 1) if stem.startswith("img_") else stem
    y = np.load(os.path.join(MSK_DIR, msk_stem + ".npy"))
    has_var = (x[..., 0].var() > var_thr)
    has_pos = (y.max() > 0) if require_pos else True
    return has_var and has_pos

def pick_interesting_tile(ids, tries=500):
    for _ in range(tries):
        cand = random.choice(ids)
        if has_texture_and_mask(cand, REQUIRE_POS_MASK, VAR_THR):
            return cand
    # fallback: just return something
    return random.choice(ids)

def main():
    ids = sorted([f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".npy")])
    if not ids:
        raise SystemExit("No .npy tiles found in data/tiles/images")

    stem = pick_interesting_tile(ids)
    msk_stem = stem.replace("img_", "msk_", 1) if stem.startswith("img_") else stem

    # load image and mask
    x = np.load(os.path.join(IMG_DIR, stem + ".npy"))          # (H,W,C) in [0,1]
    y = np.load(os.path.join(MSK_DIR, msk_stem + ".npy"))      # (H,W) {0,1}

    # load model
    c = x.shape[-1]
    model = smp.Unet(encoder_name="timm-mobilenetv3_small_075",
                     encoder_weights=None, in_channels=c, classes=1)
    sd = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # predict
    with torch.no_grad():
        inp = torch.from_numpy(x.transpose(2, 0, 1))[None].float()
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0].numpy()
        pred = (prob > 0.5).astype(np.uint8)

    # quick stats
    print(f"Picked tile: {stem}")
    print(f"SAR ch0: min={x[...,0].min():.4f} max={x[...,0].max():.4f} mean={x[...,0].mean():.4f} var={x[...,0].var():.6f}")
    print(f"Mask unique values: {np.unique(y)} | Pred positives: {pred.sum()}")

    # plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("SAR ch0"); plt.imshow(x[:, :, 0], cmap="gray", vmin=0, vmax=1); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Mask GT"); plt.imshow(y, cmap="gray", vmin=0, vmax=1); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Pred");    plt.imshow(pred, cmap="gray", vmin=0, vmax=1); plt.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
