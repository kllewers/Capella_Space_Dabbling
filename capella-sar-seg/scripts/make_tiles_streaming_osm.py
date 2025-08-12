#!/usr/bin/env python3
import os, glob, argparse, json
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import rasterize
from shapely.geometry import box, Polygon, mapping, shape
from shapely.strtree import STRtree
from pyproj import Transformer, CRS
import osmnx as ox
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
        if not cands: raise FileNotFoundError(f"No GEO .tif found in folder: {path_or_dir}")
        return sorted(cands)[0]
    cands = sorted(glob.glob(path_or_dir))
    if not cands: raise FileNotFoundError(f"No tif matches: {path_or_dir}")
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
    eps = 1e-5
    return (min_lon - eps, min_lat - eps, max_lon + eps, max_lat + eps)

# ----- OSM -----
def osmnx_sane_defaults():
    ox.settings.timeout = 180
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.memory = True

def fetch_osm_buildings_polygon(poly4326):
    """
    poly4326: shapely Polygon in EPSG:4326 (lon,lat)
    returns GeoDataFrame of building polygons (EPSG:4326) or []
    """
    tags = {"building": True}
    try:
        gdf = ox.geometries_from_polygon(poly4326, tags=tags)
    except AttributeError:
        from osmnx.features import features_from_polygon
        gdf = features_from_polygon(poly4326, tags)

    if gdf is None or gdf.empty: return []
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    return gdf

def reproject_geoms_gdf_to_crs(gdf4326, dst_crs):
    if gdf4326 is None or len(gdf4326) == 0: return []
    try:
        gdf_proj = gdf4326.to_crs(dst_crs)
        return list(gdf_proj.geometry.values)
    except Exception:
        out = []
        tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        for g in gdf4326.geometry.values:
            gj = mapping(g)
            def rec(coords):
                if isinstance(coords[0], (float, int)):
                    x, y = coords
                    X, Y = tr.transform(x, y)
                    return (X, Y)
                return [rec(c) for c in coords]
            gj["coordinates"] = rec(gj["coordinates"])
            out.append(shape(gj))
        return out

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Stream tiles from Capella GEO with OSM building masks.")
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
    ap.add_argument("--aoi-poly",   type=float, nargs='+', default=None,
                    help="AOI polygon as lat,lon repeating (CRS:84 order). Example: lat1 lon1 lat2 lon2 ...")
    ap.add_argument("--capella-json", type=str, default=None,
                    help="Capella GEO metadata JSON; used for ROI if --aoi/--aoi-poly not given")
    ap.add_argument("--min-positive", type=int, default=0,
                    help="If >0, only save tiles whose rasterized mask has at least this many positive pixels")
    return ap.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(args.out_masks,  exist_ok=True)
    osmnx_sane_defaults()

    tif = pick_tif(args.raster)
    with rasterio.open(tif) as src:
        transform, crs = src.transform, src.crs

        # --- ROI selection priority: polygon > bbox > json > full image ---
        poly4326 = None
        if args.aoi_poly and len(args.aoi_poly) >= 6 and len(args.aoi_poly) % 2 == 0:
            # args are CRS:84 (lat,lon). Convert to lon,lat tuples.
            coords = args.aoi_poly
            latlon = list(zip(coords[0::2], coords[1::2]))  # [(lat,lon), ...]
            lonlat = [(lo, la) for la, lo in latlon]
            poly4326 = Polygon(lonlat)
            bbox4326 = poly4326.bounds
        elif args.aoi:
            bbox4326 = tuple(args.aoi)
            poly4326 = Polygon([
                (bbox4326[0], bbox4326[1]),
                (bbox4326[2], bbox4326[1]),
                (bbox4326[2], bbox4326[3]),
                (bbox4326[0], bbox4326[3])
            ])
        elif args.capella_json:
            bbox4326 = bbox4326_from_capella_json(args.capella_json)
            poly4326 = Polygon([
                (bbox4326[0], bbox4326[1]),
                (bbox4326[2], bbox4326[1]),
                (bbox4326[2], bbox4326[3]),
                (bbox4326[0], bbox4326[3])
            ])
        else:
            bbox4326 = bounds_epsg4326(src)
            poly4326 = Polygon([
                (bbox4326[0], bbox4326[1]),
                (bbox4326[2], bbox4326[1]),
                (bbox4326[2], bbox4326[3]),
                (bbox4326[0], bbox4326[3])
            ])

        print("ROI bbox (WGS84):", bbox4326)

        # Build processing window in raster CRS
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        minx, miny = tr.transform(bbox4326[0], bbox4326[1])
        maxx, maxy = tr.transform(bbox4326[2], bbox4326[3])
        win = from_bounds(minx, miny, maxx, maxy, transform).round_offsets().round_lengths()
        x0, y0, w, h = int(win.col_off), int(win.row_off), int(win.width), int(win.height)

        # Fetch OSM buildings for polygon AOI, then reproject
        bldg_gdf_4326 = fetch_osm_buildings_polygon(poly4326)
        geoms = reproject_geoms_gdf_to_crs(bldg_gdf_4326, crs) if len(bldg_gdf_4326) else []
        sindex = STRtree(geoms) if geoms else None

        # Tiling
        ny = (h - args.tile_size) // args.stride + 1 if h >= args.tile_size else 0
        nx = (w - args.tile_size) // args.stride + 1 if w >= args.tile_size else 0
        pbar = tqdm(total=ny * nx, desc="Tiling")
        idx = 0

        for y in range(y0, y0 + h - args.tile_size + 1, args.stride):
            for x in range(x0, x0 + w - args.tile_size + 1, args.stride):
                wdw = Window(col_off=x, row_off=y, width=args.tile_size, height=args.tile_size)
                tile = src.read(1, window=wdw, boundless=False)
                if tile.size == 0:
                    pbar.update(1); continue

                # Per-tile transform & polygon
                tile_transform = rasterio.windows.transform(wdw, transform)
                x_ul, y_ul = (tile_transform * (0, 0))
                x_lr, y_lr = (tile_transform * (args.tile_size, args.tile_size))
                tile_poly = box(min(x_ul, x_lr), min(y_ul, y_lr), max(x_ul, x_lr), max(y_ul, y_lr))

                # Intersect buildings with tile_poly
                cand = sindex.query(tile_poly) if sindex else []
                cand = [g for g in cand if g.intersects(tile_poly)] if cand else []

                # Rasterize mask
                if cand:
                    msk = rasterize(
                        ((mapping(g), 1) for g in cand),
                        out_shape=(args.tile_size, args.tile_size),
                        transform=tile_transform,
                        fill=0, all_touched=False, dtype=np.uint8
                    )
                else:
                    msk = np.zeros((args.tile_size, args.tile_size), np.uint8)

                if args.min_positive > 0 and msk.sum() < args.min_positive:
                    pbar.update(1); continue  # skip boring tiles

                # Normalize SAR tile
                db  = to_db(tile.astype(np.float64))
                img = norm_db(db, args.db_clip[0], args.db_clip[1]).astype(np.float32)[..., None]

                # Save
                stem = f"{idx:06d}"
                np.save(os.path.join(args.out_images, f"img_{stem}.npy"), img)
                np.save(os.path.join(args.out_masks,  f"msk_{stem}.npy"), msk)
                idx += 1
                pbar.update(1)

        pbar.close()
        print(f"Done. Wrote {idx} tiles to {args.out_images} and {args.out_masks}")

if __name__ == "__main__":
    main()
