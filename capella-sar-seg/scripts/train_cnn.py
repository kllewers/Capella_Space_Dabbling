import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

IMG_DIR = "data/tiles/images"
MSK_DIR = "data/tiles/masks"
SPLIT_JSON = "splits.json"
OUT_WEIGHTS = "models/unet_mnetv3_sar.pt"
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 12
NUM_WORKERS = 4

# ----- Dataset -----
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

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        stem = self.ids[i]
        x = np.load(os.path.join(IMG_DIR, stem + ".npy"))      # (H,W,C)
        y = np.load(os.path.join(MSK_DIR, stem + ".npy"))      # (H,W)
        if self.tf:
            aug = self.tf(image=x, mask=y)
            x, y = aug["image"], aug["mask"]
        # to CHW tensors
        x = torch.from_numpy(x.transpose(2,0,1)).float()
        y = torch.from_numpy((y > 0).astype(np.float32))[None, ...]  # (1,H,W)
        return x, y

def load_ids():
    with open(SPLIT_JSON) as f: sp = json.load(f)
    return sp["train"], sp["val"]

# ----- Loss & metrics -----
class BCEDiceLoss(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, y):
        bce = self.bce(logits, y)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        inter = (probs*y).sum(dim=(1,2,3))
        denom = probs.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
        dice = 1 - ((2*inter + smooth) / (denom + smooth)).mean()
        return bce + dice*0.5

def iou_f1_from_logits(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p*y).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    prec = ((p*y).sum() / (p.sum() + 1e-6)).item()
    rec  = ((p*y).sum() / (y.sum() + 1e-6)).item()
    f1 = (2*prec*rec) / (prec+rec+1e-6)
    return iou, f1

def main():
    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
    train_ids, val_ids = load_ids()
    # infer channels from a sample
    c = np.load(os.path.join(IMG_DIR, train_ids[0] + ".npy")).shape[-1]
    print(f"Channels={c}, train={len(train_ids)}, val={len(val_ids)}")

    train_ds = NPYSARDataset(train_ids, augment=True)
    val_ds   = NPYSARDataset(val_ids, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model: U-Net with MobileNetV3-small encoder
    model = smp.Unet(encoder_name="timm-mobilenetv3_small_075",
                     encoder_weights=None,
                     in_channels=c, classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = BCEDiceLoss()

    best_iou = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()*x.size(0)
        tr_loss /= len(train_dl.dataset)

        # validation
        model.eval()
        vl_loss = 0.0; vl_iou=0.0; vl_f1=0.0; n=0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                iou, f1 = iou_f1_from_logits(logits, y)
                bs = x.size(0)
                vl_loss += loss.item()*bs
                vl_iou  += iou*bs
                vl_f1   += f1*bs
                n += bs
        vl_loss/=n; vl_iou/=n; vl_f1/=n
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {vl_loss:.4f} | IoU {vl_iou:.3f} | F1 {vl_f1:.3f}")

        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(), OUT_WEIGHTS)
            print(f"  ↳ saved best to {OUT_WEIGHTS}")

if __name__ == "__main__":
    main()
