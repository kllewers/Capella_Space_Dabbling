import os, random, numpy as np, torch, matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

IMG_DIR="data/tiles/images"; MSK_DIR="data/tiles/masks"
WEIGHTS="models/unet_mnetv3_sar.pt"

# pick a random val tile
ids = sorted([f.replace(".npy","") for f in os.listdir(IMG_DIR) if f.endswith(".npy")])
idx = random.randrange(len(ids))
stem = ids[idx]
x = np.load(os.path.join(IMG_DIR, stem + ".npy"))
y = np.load(os.path.join(MSK_DIR, stem + ".npy"))

c = x.shape[-1]
model = smp.Unet(encoder_name="timm-mobilenetv3_small_075",
                 encoder_weights=None, in_channels=c, classes=1)
sd = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict(sd); model.eval()

with torch.no_grad():
    inp = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float()
    logits = model(inp)
    pred = (torch.sigmoid(logits)[0,0].numpy()>0.5).astype(np.uint8)

# show SAR (channel 0), GT, Pred
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("SAR ch0"); plt.imshow(x[:,:,0], cmap="gray"); plt.axis("off")
plt.subplot(1,3,2); plt.title("Mask GT"); plt.imshow(y, cmap="gray"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Pred"); plt.imshow(pred, cmap="gray"); plt.axis("off")
plt.tight_layout(); plt.show()
