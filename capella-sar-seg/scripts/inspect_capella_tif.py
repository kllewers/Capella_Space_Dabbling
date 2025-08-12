#!/usr/bin/env python3
import os, sys, numpy as np, rasterio as rio

def pct(a, q): 
    return float(np.nanpercentile(a, q)) if a.size else np.nan

def classify(values, dtype, nodata):
    vmin, vmax = float(values.min()), float(values.max())
    has_neg = (vmin < 0)
    # Heuristics:
    # - uint8/uint16 with max <= 255/65535 and many zeros => likely preview/display-scaled
    # - float with all positive values in a "small" range (~0..10) => likely linear sigma0
    # - presence of negatives and small magnitude (~ -40..+10 dB) => likely already dB
    if np.isnan(vmin) or np.isnan(vmax):
        return "Unknown (all NaN?)"

    if has_neg:
        # dB usually in [-40, +10]ish for SAR backscatter
        if -100 <= vmin <= 0 and -20 <= vmax <= 20:
            return "Likely dB backscatter (log10 scale)"
        return "Has negatives (possibly dB or other transform)"

    # No negatives:
    if np.issubdtype(dtype, np.integer):
        # lots of 0 plus max <= 255 is a big preview tell
        if vmax <= 255:
            return "Likely display-scaled PREVIEW (8‑bit). Not physical units."
        else:
            return "Integer linear intensity (provider-specific scaling)"
    else:
        # float positive
        if vmax <= 2.0 and vmin >= 0.0:
            return "Likely linear sigma0 (radiometrically calibrated)"
        elif vmax <= 1000.0:
            return "Positive floats (could be linear amplitude/power)"
        return "Unusual positive float range"

def main():
    tif = sys.argv[1] if len(sys.argv) > 1 else None
    if not tif or not os.path.exists(tif):
        print("Usage: python scripts/inspect_capella_tif.py /path/to/file.tif")
        sys.exit(1)

    with rio.open(tif) as ds:
        print(f"Path: {tif}")
        print(f"Size: {ds.width} x {ds.height} | Bands: {ds.count} | CRS: {ds.crs}")
        print(f"DType: {ds.dtypes[0]}")
        print(f"NoData (dataset): {ds.nodata}")

        # Dataset & band-level tags (where providers sometimes stash clues)
        ds_tags = ds.tags()
        b_tags = ds.tags(1)
        if ds_tags:  print("\n[Dataset tags]")
        for k,v in ds_tags.items(): print(f"  {k} = {v}")
        if b_tags:   print("\n[Band 1 tags]")
        for k,v in b_tags.items(): print(f"  {k} = {v}")

        # Sample a modest window grid to avoid reading all into RAM
        H, W = ds.height, ds.width
        step = max(1, min(H, W) // 2000)  # thin sampling for big rasters
        arr = ds.read(1)[::step, ::step].astype("float64")

        # Apply NoData if present in band tags
        nodata = ds.nodata
        if nodata is None and "NODATA" in b_tags:
            try: nodata = float(b_tags["NODATA"])
            except: pass

        if nodata is not None:
            mask = (arr != nodata)
            arr = np.where(mask, arr, np.nan)
            valid = arr[np.isfinite(arr)]
        else:
            valid = arr[np.isfinite(arr)]

        if valid.size == 0:
            print("\nNo valid samples found (all nodata?).")
            return

        print("\n[Value stats on subsample]")
        print(f"  min/max: {np.nanmin(arr):.6g} / {np.nanmax(arr):.6g}")
        print(f"  p0.1/p1/p50/p99/p99.9: {pct(valid,0.1):.6g} / {pct(valid,1):.6g} / "
              f"{pct(valid,50):.6g} / {pct(valid,99):.6g} / {pct(valid,99.9):.6g}")

        # Heuristic classification
        kind = classify(valid, np.dtype(ds.dtypes[0]), nodata)
        print(f"\n[Heuristic interpretation] {kind}")

        # What would the dB range look like if this were linear?
        # (harmless even if already dB; you'll just see similar numbers)
        eps = 1e-12
        db = 10.0 * np.log10(np.clip(valid, eps, None))
        print("\n[If treated as linear → dB transform]")
        print(f"  dB p1/p50/p99: {pct(db,1):.3f} / {pct(db,50):.3f} / {pct(db,99):.3f} (dB)")

        # A quick recommendation
        print("\n[Recommendation]")
        if "PREVIEW" in os.path.basename(tif).upper() or "preview" in (ds_tags.get("DESCRIPTION","").lower()):
            print("  This looks like a PREVIEW (8‑bit) product. Use the full GEO product for analysis.")
        elif "dB" in (b_tags.get("DESCRIPTION","") + ds_tags.get("DESCRIPTION","")).lower():
            print("  Appears to be already in dB; you can visualize directly without log transform.")
        elif "sigma" in (b_tags.get("DESCRIPTION","") + ds_tags.get("DESCRIPTION","")).lower():
            print("  Appears to be sigma⁰ (linear). Apply 10*log10 for dB visualization if desired.")
        else:
            print("  If values are positive floats with no negatives, assume linear sigma⁰ and use 10*log10 for dB.")
        print("\nDone.")

if __name__ == "__main__":
    main()
