#!/usr/bin/env python3
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

# ---- headless plotting for snapshots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============== Config ==============
IMG_DIR      = "data/tiles/images"
MSK_DIR      = "data/tiles/masks"
SPLIT_JSON   = "splits.json"
OUT_WEIGHTS  = "models/unet_mnetv3_sar.pt"

BATCH_SIZE       = 4          # if slow on CPU, try less (e.g., 2)
LR               = 1e-3
EPOCHS           = 15         # early stopping can end sooner
NUM_WORKERS      = 0          # safest on macOS; try 2 if stable
PREFETCH_FACTOR  = 2          # ignored when NUM_WORKERS = 0
PRINT_EVERY      = 10         # batches

# ---- Early Stopping settings ----
ES_MONITOR       = "iou"      # "iou" or "val_loss"
ES_MIN_DELTA     = 1e-3       # min improvement to reset patience
ES_PATIENCE      = 5          # epochs to wait before stopping

# ---- Snapshot settings ----
SNAPSHOT_DIR     = "outputs/snapshots"
SNAPSHOT_EVERY   = 1          # save every epoch
N_SNAPSHOTS      = 4          # tiles per snapshot grid

# Optional: disable Albumentations update check noise
# os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Use all cores but leave one free (macOS/CPU)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ============== Data ==============
class NPYSARDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = ids
        self.augment = augment
        self.tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.RandomGamma(gamma_limit=(80,120), p=0.2),
        ]) if augment else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        stem = self.ids[i]  # e.g. "img_000204"
        img_path = os.path.join(IMG_DIR, stem + ".npy")
        mask_stem = stem.replace("img_", "msk_", 1) if stem.startswith("img_") else stem
        msk_path = os.path.join(MSK_DIR, mask_stem + ".npy")

        # Load as writable arrays for albumentations/torch
        x = np.load(img_path).astype(np.float32, copy=True)  # (H,W,C)
        y = np.load(msk_path).astype(np.uint8,   copy=True)  # (H,W)

        if self.tf:
            aug = self.tf(image=x, mask=y)
            x, y = aug["image"], aug["mask"]

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        x_t = torch.from_numpy(x.transpose(2, 0, 1)).float()            # (C,H,W)
        y_t = torch.from_numpy((y > 0).astype(np.float32))[None, ...]   # (1,H,W)
        return x_t, y_t


def load_ids():
    with open(SPLIT_JSON) as f:
        sp = json.load(f)
    train = sp["train"]
    val   = sp["val"]

    # ---- quick subset for faster iteration (adjust/remove as needed) ----
    # train = train[:1000]
    # val   = val[:200]
    # --------------------------------------------------------------------

    return train, val


# ============== Loss / Metrics ==============
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, y):
        bce = self.bce(logits, y)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        inter = (probs * y).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * inter + smooth) / (denom + smooth)).mean()
        return bce + 0.5 * dice


def iou_f1_from_logits(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * y).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    prec = ((p * y).sum() / (p.sum() + 1e-6)).item()
    rec  = ((p * y).sum() / (y.sum() + 1e-6)).item()
    f1 = (2 * prec * rec) / (prec + rec + 1e-6)
    return iou, f1


# ============== Snapshots ==============
def save_epoch_snapshots(model, val_dl, device, epoch, out_dir=SNAPSHOT_DIR, n=N_SNAPSHOTS):
    """
    Save a grid of SAR/GT/Pred for n tiles from the validation loader.
    One PNG per epoch: outputs/snapshots/epoch_XX.png
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    imgs, gts, preds = [], [], []
    grabbed = 0

    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)

            bs = xb.size(0)
            for i in range(bs):
                if grabbed >= n:
                    break
                x_np = xb[i, 0].detach().cpu().numpy()                         # show ch0
                y_np = yb[i, 0].detach().cpu().numpy()
                p_np = (prob[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                imgs.append(x_np); gts.append(y_np); preds.append(p_np)
                grabbed += 1
            if grabbed >= n:
                break

    if grabbed == 0:
        return

    cols = 3
    rows = grabbed
    plt.figure(figsize=(cols*4, rows*4))
    for r in range(rows):
        plt.subplot(rows, cols, r*cols + 1)
        plt.imshow(imgs[r], cmap="gray", vmin=0, vmax=1); plt.title("SAR ch0"); plt.axis("off")

        plt.subplot(rows, cols, r*cols + 2)
        plt.imshow(gts[r], cmap="gray", vmin=0, vmax=1); plt.title("Mask GT"); plt.axis("off")

        plt.subplot(rows, cols, r*cols + 3)
        plt.imshow(preds[r], cmap="gray", vmin=0, vmax=1); plt.title("Pred"); plt.axis("off")

    out_path = os.path.join(out_dir, f"epoch_{epoch:02d}.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"  ↳ snapshot saved: {out_path}")


# ============== Training ==============
def main():
    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)

    train_ids, val_ids = load_ids()
    sample = np.load(os.path.join(IMG_DIR, train_ids[0] + ".npy"))
    c = int(sample.shape[-1])
    print(f"Channels={c}, train={len(train_ids)}, val={len(val_ids)}")

    # Datasets
    train_ds = NPYSARDataset(train_ids, augment=True)
    val_ds   = NPYSARDataset(val_ids, augment=False)

    # DataLoaders (single-threaded to avoid macOS hang)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=False,
        pin_memory=False,
        drop_last=True,
        timeout=0
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=False,
        pin_memory=False,
        timeout=0
    )

    # Warm-up: fetch one batch so you see progress immediately
    print("Warm-up: pulling one batch...", flush=True)
    xb, yb = next(iter(train_dl))
    print("Warm-up OK:", tuple(xb.shape), tuple(yb.shape), flush=True)

    # Model / device
    model = smp.Unet(
        encoder_name="timm-mobilenetv3_small_075",
        encoder_weights=None,
        in_channels=c,
        classes=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = BCEDiceLoss()

    # ----- Early Stopping state -----
    best_metric = -float("inf") if ES_MONITOR == "iou" else float("inf")
    patience_left = ES_PATIENCE

    def is_improved(curr_metric, best_metric):
        if ES_MONITOR == "iou":
            return (curr_metric - best_metric) > ES_MIN_DELTA
        else:  # monitor val_loss (lower is better)
            return (best_metric - curr_metric) > ES_MIN_DELTA

    best_path = OUT_WEIGHTS

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        t0 = time.time()

        for bi, (x, y) in enumerate(train_dl, start=1):
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * x.size(0)

            if bi % PRINT_EVERY == 0:
                print(f"  epoch {epoch:02d} | batch {bi:05d} | last_loss {loss.item():.4f}", flush=True)

        tr_time = time.time() - t0
        tr_loss /= len(train_dl.dataset)

        # Validation
        model.eval()
        vl_loss = 0.0; vl_iou = 0.0; vl_f1 = 0.0; n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                iou, f1 = iou_f1_from_logits(logits, y)
                bs = x.size(0)
                vl_loss += loss.item() * bs
                vl_iou  += iou * bs
                vl_f1   += f1 * bs
                n += bs
        vl_loss /= n; vl_iou /= n; vl_f1 /= n

        print(f"Epoch {epoch:02d} | time {tr_time:.1f}s | train {tr_loss:.4f} | "
              f"val {vl_loss:.4f} | IoU {vl_iou:.3f} | F1 {vl_f1:.3f}")

        # ---- Save PNG snapshot this epoch ----
        if (epoch % SNAPSHOT_EVERY) == 0:
            save_epoch_snapshots(model, val_dl, device, epoch,
                                 out_dir=SNAPSHOT_DIR, n=N_SNAPSHOTS)

        # ----- Early Stopping check (monitor IoU by default) -----
        curr_metric = vl_iou if ES_MONITOR == "iou" else vl_loss
        better = is_improved(curr_metric, best_metric)

        if better:
            best_metric = curr_metric
            patience_left = ES_PATIENCE
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ improvement! saved best to {best_path} (best {ES_MONITOR}={best_metric:.4f})")
        else:
            patience_left -= 1
            print(f"  ↳ no improvement on {ES_MONITOR}. patience left: {patience_left}")
            if patience_left <= 0:
                print(f"Early stopping triggered (best {ES_MONITOR}={best_metric:.4f}).")
                break

if __name__ == "__main__":
    main()
