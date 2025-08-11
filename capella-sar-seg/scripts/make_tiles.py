#!/usr/bin/env python3
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box, mapping
import osmnx as ox
from tqdm import tqdm

# Config
SCENE_TIF = "data/raw/scene1"  # folder with the GEO .tif
OUT_IMG_DIR = "data/tiles/images"
OUT_MSK_DIR = "data/tiles/masks"
TILE_SIZE = 512
STRIDE = 412
DB_CLIP = (-30.0, 0.0)

def to_db(intensity):
    return 10.0 * np.log10(np.clip(intensity, 1e-10, None))

def norm_db(arr_db, lo, hi):
    arr = np.clip(arr_db, lo, hi)
    return (arr - lo) / (hi - lo)

def load_geo_tif(folder):
    tifs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif") and "_preview" not in f]
    if not tifs:
        raise FileNotFoundError("No GEO .tif found in scene folder.")
    tif = tifs[0]
    src = rasterio.open(tif)
    arr = src.read(1)  # single-pol HH intensity
    return arr, src

def fetch_osm_buildings(bounds):
    # bounds: (minx, miny, maxx, maxy) in EPSG:4326; ensure our data is in 4326
    # Capella GEO is usually in EPSG:4326; if not, we’ll reproject bounds to 4326 first.
    # osmnx expects lat/lon bbox: north, south, east, west
    west, south, east, north = bounds[0], bounds[1], bounds[2], bounds[3]
    try:
        gdf = ox.features_from_bbox(north, south, east, west, tags={"building": True})
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf.to_crs("EPSG:4326")
        return [shape(geom) for geom in gdf.geometry.values]
    except Exception:
        return []

def rasterize_buildings(geoms, out_shape, transform):
    if not geoms:
        return np.zeros(out_shape, dtype=np.uint8)
    shapes = ((mapping(g), 1) for g in geoms)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8
    )
    return mask

def tile_arrays(img, msk, tile_size, stride):
    H, W = img.shape[:2]
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            yield y, x, img[y:y+tile_size, x:x+tile_size], msk[y:y+tile_size, x:x+tile_size]

def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    # Load raster
    intensity, src = load_geo_tif(SCENE_TIF)
    crs = src.crs
    transform = src.transform

    # If not WGS84, reproject bounds to EPSG:4326 for OSM query
    if crs and crs.to_epsg() != 4326:
        from pyproj import Transformer
        h, w = intensity.shape
        xs = [0, w]
        ys = [0, h]
        # compute bounds in raster coords -> world coords
        minx, miny = rasterio.transform.xy(transform, ys[1], xs[0], offset='ul')
        maxx, maxy = rasterio.transform.xy(transform, ys[0], xs[1], offset='lr')
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(minx, miny)
        east, north = transformer.transform(maxx, maxy)
        bounds4326 = (west, south, east, north)
    else:
        # Already EPSG:4326
        left, bottom, right, top = src.bounds
        bounds4326 = (left, bottom, right, top)

    # Pull OSM buildings
    buildings = fetch_osm_buildings(bounds4326)

    # Convert intensity -> dB -> [0,1]
    db = to_db(intensity.astype(np.float64))
    norm = norm_db(db, DB_CLIP[0], DB_CLIP[1])
    img = norm[..., None].astype(np.float32)  # shape (H, W, 1)

    # Rasterize buildings to mask on this grid
    mask = rasterize_buildings(buildings, img.shape[:2], transform)

    # Tile and save
    idx = 0
    for y, x, im_t, ms_t in tqdm(tile_arrays(img, mask, TILE_SIZE, STRIDE), total=None):
        np.save(os.path.join(OUT_IMG_DIR, f"img_{idx:06d}.npy"), im_t)
        np.save(os.path.join(OUT_MSK_DIR, f"msk_{idx:06d}.npy"), ms_t.astype(np.uint8))
        idx += 1

    print(f"Done. Wrote {idx} tiles to {OUT_IMG_DIR} and {OUT_MSK_DIR}")

if __name__ == "__main__":
    main()
