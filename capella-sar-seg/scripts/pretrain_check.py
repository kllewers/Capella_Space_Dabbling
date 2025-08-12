#!/usr/bin/env python3
"""
Auto PCA (multi-band) or dB contrast (single-band) for SAR GeoTIFFs.

Usage example:
python scripts/pca_or_db.py \
  --tif "data/raw/scene_shanghai/CAPELLA_C11_SP_GEO_HH_20250320045730_20250320045802_preview.tif" \
  --out-prefix "outputs/shanghai_vis" \
  --tile 1024 \
  --components 3 \
  --png-downsample 8 \
  --nodata 0 \
  --db-clip -25 10 \
  --equalize clahe
"""
import os, argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from sklearn.decomposition import IncrementalPCA

# headless plotting for PNGs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# helpers
# -------------------------
def to_db(a, eps=1e-6):
    a = a.astype(np.float32, copy=False)
    return 10.0 * np.log10(np.clip(a, eps, None))

def minmax_scale(x, vmin, vmax):
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-12)
    return x

def percent_clip(x, p_lo=2, p_hi=98):
    lo = np.nanpercentile(x, p_lo)
    hi = np.nanpercentile(x, p_hi)
    return minmax_scale(x, lo, hi)

def clahe_gray(img01, clip_limit=0.01, tile_grid=(8,8)):
    # simple CLAHE via skimage if available; otherwise fallback to percent stretch
    try:
        from skimage import exposure
        return exposure.equalize_adapthist(img01, clip_limit=clip_limit, nbins=256, kernel_size=tile_grid)
    except Exception:
        return percent_clip(img01, 2, 98)

def equalize_method(img01, method):
    if method == "none":
        return img01
    if method == "hist":
        try:
            from skimage import exposure
            return exposure.equalize_hist(img01)
        except Exception:
            return percent_clip(img01, 2, 98)
    if method == "clahe":
        return clahe_gray(img01)
    return img01

def write_png(path, arr01):
    # arr01: HxW (gray) or HxWx3 (RGB) in [0,1]
    arr01 = np.clip(arr01, 0, 1)
    plt.imsave(path, arr01, cmap=None if arr01.ndim==3 else "gray")

# -------------------------
# core
# -------------------------
def run_single_band_db(src, args):
    """Produce a contrast‑enhanced dB visualization from a single‑band SAR."""
    H, W = src.height, src.width
    out_tif = f"{args.out_prefix}_db.tif"
    out_png = f"{args.out_prefix}_db.png"

    # allocate output memmap to avoid huge RAM usage
    out = np.memmap(out_tif + ".tmp", dtype=np.uint8, mode="w+", shape=(H, W))

    tile = args.tile
    db_lo, db_hi = (args.db_clip if args.db_clip else (-25.0, 10.0))

    # 1) convert to dB + global minmax (or provided db_clip), write 8‑bit
    for y in range(0, H, tile):
        h = min(tile, H - y)
        for x in range(0, W, tile):
            w = min(tile, W - x)
            win = Window(x, y, w, h)
            block = src.read(1, window=win)
            if args.nodata is not None:
                mask = (block == args.nodata)
            else:
                mask = np.zeros_like(block, dtype=bool)

            db = to_db(block)
            if args.db_clip is None:
                # percentile clip per block to avoid global calc cost
                db01 = percent_clip(db)
            else:
                db01 = minmax_scale(db, db_lo, db_hi)

            db01[mask] = 0.0
            out[y:y+h, x:x+w] = (db01 * 255.0 + 0.5).astype(np.uint8)

    # 2) write GeoTIFF (single band, 8‑bit)
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="DEFLATE")
    with rasterio.open(out_tif, "w", **profile) as dst:
        # stream back from memmap
        for y in range(0, H, tile):
            h = min(tile, H - y)
            for x in range(0, W, tile):
                w = min(tile, W - x)
                dst.write(out[y:y+h, x:x+w], 1, window=Window(x, y, w, h))
    try:
        os.remove(out_tif + ".tmp")
    except Exception:
        pass

    # 3) PNG quicklook (downsampled to save space)
    if args.png_downsample > 1:
        scale = args.png_downsample
        new_h = H // scale
        new_w = W // scale
        # cheap decimation
        png_img = out.reshape(H//scale, scale, W//scale, scale).mean(axis=(1,3)) / 255.0
    else:
        png_img = (np.array(out, dtype=np.float32) / 255.0)
    # optional equalization for PNG only
    png_img = equalize_method(png_img, args.equalize)
    write_png(out_png, png_img)

    print(f"Wrote dB GeoTIFF: {out_tif}")
    print(f"Wrote quicklook PNG: {out_png}")

def run_multiband_pca(src, args):
    """Incremental PCA for multi‑band rasters, outputs 3‑band RGB."""
    H, W = src.height, src.width
    B = src.count
    comps = min(args.components, B)

    # ---- Pass 1: fit PCA on subsampled pixels ----
    ipca = IncrementalPCA(n_components=comps, batch_size=args.tile*args.tile)
    for y in range(0, H, args.tile):
        h = min(args.tile, H - y)
        for x in range(0, W, args.tile):
            w = min(args.tile, W - x)
            win = Window(x, y, w, h)
            block = src.read(window=win)  # (B,h,w)
            block = block.reshape(B, -1).T  # (N,B)
            if args.sample_step > 1:
                block = block[::args.sample_step]
            ipca.partial_fit(block)

    # ---- Pass 2: transform and track global min/max per component ----
    mins = np.full(comps, np.inf, dtype=np.float32)
    maxs = np.full(comps, -np.inf, dtype=np.float32)
    for y in range(0, H, args.tile):
        h = min(args.tile, H - y)
        for x in range(0, W, args.tile):
            w = min(args.tile, W - x)
            win = Window(x, y, w, h)
            block = src.read(window=win).reshape(B, -1).T  # (N,B)
            z = ipca.transform(block)  # (N,C)
            mins = np.minimum(mins, z.min(axis=0))
            maxs = np.maximum(maxs, z.max(axis=0))

    # ---- Pass 3: write 3‑band GeoTIFF (or pad comps) ----
    out_tif = f"{args.out_prefix}_pca_rgb.tif"
    out_png = f"{args.out_prefix}_pca_rgb.png"

    profile = src.profile.copy()
    out_bands = 3
    profile.update(count=out_bands, dtype="uint8", compress="DEFLATE")
    with rasterio.open(out_tif, "w", **profile) as dst:
        for y in range(0, H, args.tile):
            h = min(args.tile, H - y)
            for x in range(0, W, args.tile):
                w = min(args.tile, W - x)
                win = Window(x, y, w, h)
                block = src.read(window=win).reshape(B, -1).T
                z = ipca.transform(block)  # (N,C)
                # scale each component to 0..1 using global mins/maxs
                z01 = (z - mins) / (maxs - mins + 1e-12)
                # pack to RGB (repeat if comps < 3)
                if comps == 1:
                    rgb = np.repeat(z01, 3, axis=1)
                elif comps == 2:
                    rgb = np.concatenate([z01, z01[:, :1]], axis=1)  # [PC1,PC2,PC1]
                else:
                    rgb = z01[:, :3]
                rgb = (rgb * 255.0 + 0.5).astype(np.uint8).reshape(h, w, 3)
                # write bands
                for b in range(3):
                    dst.write(rgb[:,:,b], b+1, window=win)

    # quicklook PNG (downsample)
    factor = max(1, args.png_downsample)
    with rasterio.open(out_tif) as qsrc:
        if factor > 1:
            data = qsrc.read(
                out_shape=(3, qsrc.height//factor, qsrc.width//factor),
                resampling=Resampling.average
            )
        else:
            data = qsrc.read()
    data01 = (data.transpose(1,2,0).astype(np.float32))/255.0
    write_png(out_png, data01)
    print(f"Wrote PCA RGB GeoTIFF: {out_tif}")
    print(f"Wrote quicklook PNG: {out_png}")

# -------------------------
# CLI / main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Auto PCA (multi-band) or dB contrast (single-band) for SAR GeoTIFFs")
    ap.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., outputs/my_scene)")
    ap.add_argument("--tile", type=int, default=1024, help="Tile size for streaming")
    ap.add_argument("--components", type=int, default=3, help="PCA components (if multiband)")
    ap.add_argument("--sample-step", type=int, default=2, help="Subsampling factor when fitting PCA")
    ap.add_argument("--png-downsample", type=int, default=8, help="Downsample factor for PNG quicklook")
    ap.add_argument("--nodata", type=float, default=None, help="NoData value, e.g., 0")
    ap.add_argument("--db-clip", type=float, nargs=2, default=None, help="Override dB display range, e.g., -25 10")
    ap.add_argument("--equalize", choices=["none","hist","clahe"], default="clahe",
                    help="Equalization method for PNG quicklook")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    with rasterio.open(args.tif) as src:
        print(f"Input: {args.tif} | size={src.width}x{src.height} | bands={src.count} | CRS={src.crs}")
        if args.nodata is not None:
            print(f"Treating value {args.nodata} as NoData")

        if src.count >= 2:
            print("Detected multi-band raster → running Incremental PCA…")
            run_multiband_pca(src, args)
        else:
            print("Detected single-band raster → making contrast‑enhanced dB visualization…")
            run_single_band_db(src, args)

if __name__ == "__main__":
    main()
