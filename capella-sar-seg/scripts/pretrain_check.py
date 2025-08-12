import os
import numpy as np
import matplotlib.pyplot as plt
import math

IMG_DIR = "data/tiles/images"
MSK_DIR = "data/tiles/masks"

# List all image files
imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".npy")])
n_tiles = len(imgs)
print(f"Found {n_tiles} image tiles.")

# Determine grid size
cols = 4  # Number of SAR/mask pairs per row
rows = math.ceil(n_tiles / cols)

# Create big figure
fig, axes = plt.subplots(rows, cols*2, figsize=(cols*4, rows*4))
axes = axes.ravel()

for i, fname in enumerate(imgs):
    img_path = os.path.join(IMG_DIR, fname)
    msk_path = os.path.join(MSK_DIR, fname.replace("img_", "msk_", 1))

    img = np.load(img_path)
    msk = np.load(msk_path)

    # Flatten to 2D if needed
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] == 1 else img[0]

    # Contrast stretch for SAR visualization
    lo, hi = np.percentile(img, (2, 98))
    img_disp = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    # SAR
    axes[i*2].imshow(img_disp, cmap="gray")
    axes[i*2].set_title(f"SAR {fname}")
    axes[i*2].axis("off")

    # Mask
    axes[i*2 + 1].imshow(msk, cmap="gray")
    axes[i*2 + 1].set_title("Mask")
    axes[i*2 + 1].axis("off")

# Hide unused subplots
for ax in axes[n_tiles*2:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("all_tiles_overview.png", dpi=150)
plt.show()

print("Saved all tiles to all_tiles_overview.png")
