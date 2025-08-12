#!/usr/bin/env python3
import os, glob, argparse, json
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from pyproj import Transformer, CRS
from tqdm import tqdm

# ---------- helpers ----------
def to_db(a): return 10.0 * np.log10(np.clip(a, 1e-10, None))
def norm_db(a_db, lo, hi):
    a = np.clip(a_db, lo, hi)
    return (a - lo) / (hi - lo)

def pick_tif(path_or_dir):
    if os.path.isdir(path_or_dir):
        cands = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir)
                 if f.lower().endswith(".tif") and "_preview" not in f]
        if not cands:
            raise FileNotFoundError(f"No GEO .tif found in folder: {path_or_dir}")
        return sorted(cands)[0]
    cands = sorted(glob.glob(path_or_dir))
    if not cands:
        raise FileNotFoundError(f"No tif matches: {path_or_dir}")
    return cands[0]

def bounds_epsg4326(src):
    left, bottom, right, top = src.bounds
    crs = src.crs
    if crs and crs.to_epsg() != 4326:
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = tr.transform(left,  bottom)
        east, north = tr.transform(right, top)
        return (west, south, east, north)
    return (left, bottom, right, top)

def bbox4326_from_capella_json(json_path):
    with open(json_path, "r") as f:
        meta = json.load(f)
    img = meta["collect"]["image"]
    gt = img["image_geometry"]["geotransform"]  # [x0, px, 0, y0, 0, py]
    rows = int(img["rows"]); cols = int(img["columns"])
    x0 = float(gt[0]); y0 = float(gt[3]); px = float(gt[1]); py = float(gt[5])
    x1 = x0 + px * cols
    y1 = y0 + py * rows
    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y1, y0) if y1 <= y0 else (y0, y1)
    wkt = img["image_geometry"]["coordinate_system"]["wkt"]
    crs_prod = CRS.from_wkt(wkt)
    tr = Transformer.from_crs(crs_prod, "EPSG:4326", always_xy=True)
    min_lon, min_lat = tr.transform(xmin, ymin)
    max_lon, max_lat = tr.transform(xmax, ymax)
    # tiny buffer to avoid degenerate bbox
    eps = 1e-5
    return (min_lon - eps, min_lat - eps, max_lon + eps, max_lat + eps)

def otsu_threshold(x01):
    # x01 in [0,1]
    hist, bin_edges = np.histogram(x01.ravel(), bins=256, range=(0,1))
    hist = hist.astype(np.float64); total = hist.sum() + 1e-12
    hist /= total
    cdf = np.cumsum(hist)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean = (hist * bins).sum()
    between = (mean * cdf - np.cumsum(hist * bins))**2 / (cdf*(1-cdf) + 1e-12)
    k = int(np.nanargmax(between))
    return bins[k]

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Stream 512x512 tiles from Capella GEO (no OSM).")
    ap.add_argument("--raster", default="data/raw/scene_shanghai",
                    help="Path to GEO .tif OR a folder containing it")
    ap.add_argument("--out-images", default="data/tiles/images")
    ap.add_argument("--out-masks",  default="data/tiles/masks")
    ap.add_argument("--tile-size",  type=int, default=512)
    ap.add_argument("--stride",     type=int, default=412,
                    help="Step in pixels; 512=no overlap, 412≈100px overlap")
    ap.add_argument("--db-clip",    type=float, nargs=2, default=[-30.0, 0.0])
    ap.add_argument("--aoi",        type=float, nargs=4, default=None,
                    help="AOI bbox as minLon minLat maxLon maxLat (EPSG:4326)")
    ap.add_argument("--capella-json", type=str, default=None,
                    help="Path to Capella GEO metadata JSON; used to derive ROI if --aoi not given")
    ap.add_argument("--mask-mode", choices=["threshold","zeros"], default="threshold",
                    help="threshold: Otsu on SAR dB; zeros: all-background masks")
    return ap.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(args.out_masks,  exist_ok=True)

    tif = pick_tif(args.raster)
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        transform, crs = src.transform, src.crs

        # ROI: --aoi > --capella-json > full image
        if args.aoi:
            bbox4326 = tuple(args.aoi)
        elif args.capella_json:
            bbox4326 = bbox4326_from_capella_json(args.capella_json)
        else:
            bbox4326 = bounds_epsg4326(src)
        print("ROI (WGS84):", bbox4326)

        # Build processing window
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        minx, miny = tr.transform(bbox4326[0], bbox4326[1])
        maxx, maxy = tr.transform(bbox4326[2], bbox4326[3])
        win = from_bounds(minx, miny, maxx, maxy, transform).round_offsets().round_lengths()
        x0, y0, w, h = int(win.col_off), int(win.row_off), int(win.width), int(win.height)

        # Compute last full-window starts so we never read partial tiles
        if h < args.tile_size or w < args.tile_size:
            print("AOI smaller than tile size; nothing to do.")
            return
        y_stop = y0 + ((h - args.tile_size) // args.stride) * args.stride
        x_stop = x0 + ((w - args.tile_size) // args.stride) * args.stride

        ny = (y_stop - y0) // args.stride + 1
        nx = (x_stop - x0) // args.stride + 1

        idx = 0
        pbar = tqdm(total=ny * nx, desc="Tiling")
        for y in range(y0, y_stop + 1, args.stride):
            for x in range(x0, x_stop + 1, args.stride):
                wdw = Window(col_off=x, row_off=y, width=args.tile_size, height=args.tile_size)
                tile = src.read(1, window=wdw, boundless=False)

                # super defensive guard (shouldn't trigger with clamped loops)
                if tile.size == 0 or tile.shape != (args.tile_size, args.tile_size):
                    pbar.update(1)
                    continue

                # Normalize SAR tile to [0,1]
                db  = to_db(tile.astype(np.float64))
                img = norm_db(db, args.db_clip[0], args.db_clip[1]).astype(np.float32)[..., None]

                # Make mask
                if args.mask_mode == "threshold":
                    thr = otsu_threshold(img[..., 0])
                    msk = (img[..., 0] > thr).astype(np.uint8)
                else:  # zeros
                    msk = np.zeros((args.tile_size, args.tile_size), np.uint8)

                # save (assert shapes to be safe)
                assert img.shape[:2] == msk.shape[:2] == (args.tile_size, args.tile_size)
                stem = f"{idx:06d}"
                np.save(os.path.join(args.out_images, f"img_{stem}.npy"), img)
                np.save(os.path.join(args.out_masks,  f"msk_{stem}.npy"), msk)
                idx += 1
                pbar.update(1)

        pbar.close()
        print(f"Done. Wrote {idx} tiles to {args.out_images} and {args.out_masks}")

if __name__ == "__main__":
    main()
