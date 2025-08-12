#!/usr/bin/env python3
"""
Sliding‑window inference over a GeoTIFF with overlap‑averaging.

Writes:
  <out_prefix>_prob.tif     (float32 probabilities in [0,1])
  <out_prefix>_mask.tif     (uint8 {0,1} by threshold)
  <out_prefix>_quicklook.png  (PNG sanity check)
"""

import os, argparse
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import segmentation_models_pytorch as smp

# headless plotting for the quicklook
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helpers ----------
def to_db(a):
    return 10.0 * np.log10(np.clip(a, 1e-10, None))

def norm_db(a_db, lo, hi):
    a = np.clip(a_db, lo, hi)
    return (a - lo) / (hi - lo + 1e-12)

def gen_starts(total, size, stride):
    """Yield window starts so that we also cover the right/bottom edge."""
    if total <= size:
        yield 0
        return
    pos = 0
    while pos + size < total:
        yield pos
        pos += stride
    # ensure last window touches the end
    yield max(0, total - size)

def edge_pad_hw(x, target_h, target_w):
    """Pad (H,W,1) to (target_h,target_w,1) by edge replication."""
    h, w = x.shape[:2]
    ph = max(0, target_h - h)
    pw = max(0, target_w - w)
    if ph == 0 and pw == 0:
        return x
    return np.pad(x, ((0,ph),(0,pw),(0,0)), mode="edge")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Run UNet model on a GeoTIFF (sliding window).")

    # Defaults set to your paths
    ap.add_argument("--tif", default="/Users/kitlewers/Capella_SAR_Fun/capella-sar-seg/data/raw/scene_shanghai/CAPELLA_C11_SP_GEO_HH_20250320045730_20250320045802_preview.tif",
                    help="Input GeoTIFF")
    ap.add_argument("--weights", default="/Users/kitlewers/Capella_SAR_Fun/capella-sar-seg/models/unet_mnetv3_sar.pt",
                    help="Model weights .pt")
    ap.add_argument("--out-prefix", default="outputs/preview_pred",
                    help="Output prefix (writes *_prob.tif, *_mask.tif, *_quicklook.png)")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--stride",    type=int, default=412)
    ap.add_argument("--db-clip",   type=float, nargs=2, default=[-25.0, 5.0],
                    help="dB clipping range used in training, e.g. -25 5")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for mask")
    return ap.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model (must match training!)
    model = smp.Unet(encoder_name="timm-mobilenetv3_small_075",
                     encoder_weights=None, in_channels=1, classes=1)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd)
    model.eval().to(device)

    with rasterio.open(args.tif) as src:
        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform

        # accumulation buffers for overlap‑averaging
        prob_sum = np.zeros((H, W), np.float32)
        weight   = np.zeros((H, W), np.float32)

        ts = args.tile_size
        st = args.stride

        # slide over the image
        for y0 in gen_starts(H, ts, st):
            for x0 in gen_starts(W, ts, st):
                # crop (allow smaller tiles at edges)
                h_win = min(ts, H - y0)
                w_win = min(ts, W - x0)
                win = Window(col_off=x0, row_off=y0, width=w_win, height=h_win)

                tile = src.read(1, window=win)  # band-1
                # to dB -> normalize -> expand channel dim
                db  = to_db(tile.astype(np.float64))
                x01 = norm_db(db, args.db_clip[0], args.db_clip[1]).astype(np.float32)[..., None]

                # pad small edge tiles to model's expected size
                x01_padded = edge_pad_hw(x01, ts, ts)

                # run model
                with torch.no_grad():
                    inp = torch.from_numpy(x01_padded.transpose(2,0,1))[None].to(device)  # (1,1,ts,ts)
                    logits = model(inp)
                    prob_full = torch.sigmoid(logits)[0,0].detach().cpu().numpy()  # (ts,ts)

                # unpad back to the original window size
                prob_crop = prob_full[:h_win, :w_win]

                # accumulate
                prob_sum[y0:y0+h_win, x0:x0+w_win] += prob_crop
                weight  [y0:y0+h_win, x0:x0+w_win] += 1.0

        # avoid div by zero
        weight[weight == 0] = 1.0
        prob = prob_sum / weight
        mask = (prob >= args.threshold).astype(np.uint8)

        # write GeoTIFFs
        prof = src.profile.copy()
        # probability GeoTIFF
        prof.update(dtype=rasterio.float32, count=1, compress="deflate", predictor=3)
        prob_path = f"{args.out_prefix}_prob.tif"
        with rasterio.open(prob_path, "w", **prof) as dst:
            dst.write(prob.astype(np.float32), 1)

        # mask GeoTIFF
        prof.update(dtype=rasterio.uint8, count=1, compress="deflate", predictor=2)
        mask_path = f"{args.out_prefix}_mask.tif"
        with rasterio.open(mask_path, "w", **prof) as dst:
            dst.write(mask.astype(np.uint8), 1)

        # quicklook PNG: show SAR (scaled) and overlay mask edges
        # Re-read a downsample if massive; for preview image size is small, so use full res.
        # Normalize SAR again for display:
        with rasterio.open(args.tif) as src2:
            sar = src2.read(1)
        sar_db = to_db(sar.astype(np.float64))
        sar_01 = norm_db(sar_db, args.db_clip[0], args.db_clip[1])

        plt.figure(figsize=(10,10))
        plt.imshow(sar_01, cmap="gray", vmin=0, vmax=1)
        # overlay mask boundary
        plt.imshow(np.ma.masked_where(mask==0, mask), alpha=0.25)
        plt.title("SAR preview with predicted mask overlay")
        plt.axis("off")
        png_path = f"{args.out_prefix}_quicklook.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()

        print("Wrote:")
        print(" ", prob_path)
        print(" ", mask_path)
        print(" ", png_path)

if __name__ == "__main__":
    main()
