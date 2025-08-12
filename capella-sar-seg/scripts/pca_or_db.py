#!/usr/bin/env python3
"""
Auto PCA (multi-band) or dB contrast (single-band) for SAR GeoTIFFs.

Examples:
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
    """Convert linear amplitude/power to dB safely."""
    a = a.astype(np.float32, copy=False)
    return 10.0 * np.log10(np.clip(a, eps, None))

def minmax_scale(x, vmin, vmax):
    """Scale to [0,1] given fixed vmin/vmax."""
    x = np.clip(x, vmin, vmax)
    return (x - vmin) / (vmax - vmin + 1e-12)

def equalize_method(img01, method):
    """Optional equalization (PNG only)."""
    if method == "none":
        return img01
    try:
        from skimage import exposure
        if method == "hist":
            return exposure.equalize_hist(img01)
        if method == "clahe":
            return exposure.equalize_adapthist(img01, clip_limit=0.01, nbins=256, kernel_size=(8,8))
    except Exception:
        pass
    # fallback: light percentile stretch
    lo = np.nanpercentile(img01, 2)
    hi = np.nanpercentile(img01, 98)
    return np.clip((img01 - lo) / (hi - lo + 1e-12), 0, 1)

def write_png(path, arr01):
    """arr01: HxW (gray) or HxWx3 (RGB) in [0,1]."""
    arr01 = np.clip(arr01, 0, 1)
    plt.imsave(path, arr01, cmap=None if arr01.ndim==3 else "gray")

def downsample_mean_uint8(arr, scale, nodata=None, pad=False):
    """
    Block-average downsample by `scale` for uint8 image.
    If pad=False (default), crops to a multiple of scale.
    If pad=True, pads with nodata (or 0) to next multiple before averaging.
    """
    if scale <= 1:
        return arr.astype(np.uint8, copy=False)
    H, W = arr.shape
    if pad:
        from math import ceil
        Hp = int(ceil(H/scale))*scale
        Wp = int(ceil(W/scale))*scale
        if Hp != H or Wp != W:
            fill = nodata if nodata is not None else 0
            a = np.full((Hp, Wp), fill, dtype=arr.dtype)
            a[:H, :W] = arr
        else:
            a = arr
        Hc, Wc = Hp, Wp
    else:
        Hc = (H // scale) * scale
        Wc = (W // scale) * scale
        a = arr[:Hc, :Wc]

    if nodata is not None:
        af = a.astype(np.float32)
        m  = (a == nodata)
        af[m] = np.nan
        af = af.reshape(Hc//scale, scale, Wc//scale, scale)
        out = np.nanmean(af, axis=(1,3))
        out = np.nan_to_num(out, nan=0.0)
    else:
        af = a.reshape(Hc//scale, scale, Wc//scale, scale)
        out = af.mean(axis=(1,3))

    return np.clip(out, 0, 255).astype(np.uint8)

# -------------------------
# core
# -------------------------
def run_single_band_db(src, args):
    """
    Single-band SAR visualization.
    If Capella GEO metadata indicates dB-per-count (scale_factor ~ 0.001–0.01),
    use: db = DN * scale_factor.
    Otherwise: db = 10*log10(linear) (with optional linear scale).
    """
    import json

    H, W = src.height, src.width
    out_tif = f"{args.out_prefix}_db.tif"
    out_png = f"{args.out_prefix}_db.png"

    # ---- detect Capella dB-per-DN ----
    capella_db_per_dn = False
    sf_db = None
    try:
        tag = (src.tags().get("TIFFTAG_IMAGEDESCRIPTION")
               or src.tags().get("IMAGEDESCRIPTION"))
        if tag:
            meta = json.loads(tag)
            sf = meta.get("collect", {}).get("image", {}).get("scale_factor", None)
            radiom = meta.get("collect", {}).get("image", {}).get("radiometry", "")
            if isinstance(sf, (int, float)) and 0.0005 <= sf <= 0.05 and "sigma" in str(radiom).lower():
                capella_db_per_dn = True
                sf_db = float(sf)
                print(f"[Capella] Detected sigma0 in dB-per-DN. scale_factor={sf_db}")
    except Exception as e:
        print(f"[Capella] Could not parse TIFFTAG_IMAGEDESCRIPTION ({e}). Using generic path.")

    # ---- pick global db range ----
    if args.db_clip is not None:
        db_lo, db_hi = float(args.db_clip[0]), float(args.db_clip[1])
        print(f"Using provided dB clip: [{db_lo}, {db_hi}]")
    else:
        # sample to estimate percentiles
        samples = []
        step = max(1, args.sample_step)
        print("Estimating global dB range from samples…")
        for y in range(0, H, args.tile):
            h = min(args.tile, H - y)
            for x in range(0, W, args.tile):
                w = min(args.tile, W - x)
                win = rasterio.windows.Window(x, y, w, h)
                DN = src.read(1, window=win)
                if args.nodata is not None:
                    DN = DN[DN != args.nodata]
                DN = DN[::step]
                if DN.size == 0:
                    continue
                if capella_db_per_dn:
                    db = DN.astype(np.float32) * sf_db
                else:
                    # generic: assume DN already linear-ish → dB via log10
                    db = 10.0 * np.log10(np.clip(DN.astype(np.float32), 1e-6, None))
                samples.append(db)
        if not samples:
            raise RuntimeError("No valid samples found to estimate dB range.")
        sam = np.concatenate(samples)
        db_lo = float(np.percentile(sam, 2))
        db_hi = float(np.percentile(sam, 98))
        print(f"Estimated dB clip from samples: [{db_lo:.2f}, {db_hi:.2f}]")

    # ---- write 8-bit dB GeoTIFF (streaming) ----
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="DEFLATE")
    if args.nodata is not None:
        profile.update(nodata=np.uint8(0))

    with rasterio.open(out_tif, "w", **profile) as dst:
        for y in range(0, H, args.tile):
            h = min(args.tile, H - y)
            for x in range(0, W, args.tile):
                w = min(args.tile, W - x)
                win = rasterio.windows.Window(x, y, w, h)
                DN = src.read(1, window=win)
                if args.nodata is not None:
                    mask = (DN == args.nodata)
                else:
                    mask = np.zeros_like(DN, dtype=bool)

                if capella_db_per_dn:
                    db = DN.astype(np.float32) * sf_db
                else:
                    db = 10.0 * np.log10(np.clip(DN.astype(np.float32), 1e-6, None))

                db01 = np.clip((db - db_lo) / (db_hi - db_lo + 1e-12), 0, 1)
                db01[mask] = 0.0
                u8 = (db01 * 255.0 + 0.5).astype(np.uint8)
                dst.write(u8, 1, window=win)

    # ---- quicklook PNG (read back and downsample with GDAL) ----
    print("Building quicklook PNG…")
    factor = max(1, args.png_downsample)
    with rasterio.open(out_tif) as qsrc:
        if factor > 1:
            q = qsrc.read(
                out_shape=(1, qsrc.height//factor, qsrc.width//factor),
                resampling=Resampling.average
            )[0]
        else:
            q = qsrc.read(1)
    img01 = q.astype(np.float32) / 255.0
    img01 = equalize_method(img01, args.equalize)
    write_png(out_png, img01)

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
            block = src.read(window=win)          # (B,h,w)
            block = block.reshape(B, -1).T        # (N,B)
            if args.sample_step > 1:
                block = block[::args.sample_step]
            if block.size:
                ipca.partial_fit(block)

    # ---- Pass 2: transform and track global min/max per component ----
    mins = np.full(comps, np.inf, dtype=np.float32)
    maxs = np.full(comps, -np.inf, dtype=np.float32)
    for y in range(0, H, args.tile):
        h = min(args.tile, H - y)
        for x in range(0, W, args.tile):
            w = min(args.tile, W - x)
            win = Window(x, y, w, h)
            block = src.read(window=win).reshape(B, -1).T
            if block.size:
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
                if block.size == 0:
                    continue
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
    ap.add_argument("--sample-step", type=int, default=2, help="Subsampling factor for sampling/IPCA")
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
