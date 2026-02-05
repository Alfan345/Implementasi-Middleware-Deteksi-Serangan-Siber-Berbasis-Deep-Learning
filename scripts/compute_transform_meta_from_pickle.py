#!/usr/bin/env python3
"""
Compute medians and clip quantiles (q001, q999) for features from a pickle DataFrame.
Usage:
  python scripts/compute_transform_meta_from_pickle.py --pkl path/to/lite_clean_data_collapsed.pkl --artifacts artifacts_dir

Notes:
- If artifacts/transform_meta.json exists, this script will back it up and overwrite with new medians/clip_quantiles.
- Heuristic cleaning: drop extreme outliers using median +/- iqr_multiplier * IQR before computing quantiles.
- For packet-length features we also cap values at packet_length_cap (default 65535) for plausibility.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import math
import sys

def robust_stats(series: pd.Series, iqr_multiplier=10, packet_length_cap=None):
    ser = pd.to_numeric(series, errors='coerce').dropna().astype(float)
    if ser.empty:
        return None, (None, None), 0

    # optional cap for packet length features
    if packet_length_cap is not None:
        ser = ser.where(ser <= packet_length_cap, np.nan).dropna()

    if ser.empty:
        # fallback to original (uncapped) numeric values
        ser = pd.to_numeric(series, errors='coerce').dropna().astype(float)
        if ser.empty:
            return None, (None, None), 0

    # initial median and IQR
    med = float(ser.median())
    q1 = float(ser.quantile(0.25))
    q3 = float(ser.quantile(0.75))
    iqr = q3 - q1
    # define bounds
    lower = med - iqr_multiplier * iqr
    upper = med + iqr_multiplier * iqr
    # if IQR == 0, expand bounds slightly around median
    if iqr == 0:
        lower = med - max(1.0, abs(med)*0.01)
        upper = med + max(1.0, abs(med)*0.01)

    filtered = ser[(ser >= lower) & (ser <= upper)]
    used_count = len(filtered)
    if filtered.empty:
        filtered = ser  # fallback

    # compute median and q001/q999 on filtered
    med_final = float(filtered.median())
    try:
        q001 = float(filtered.quantile(0.001))
        q999 = float(filtered.quantile(0.999))
    except Exception:
        q001, q999 = None, None

    return med_final, (q001, q999), used_count

def infer_cols_from_df(df: pd.DataFrame):
    # prefer canonical names used by preprocessor, else use df.columns
    return [c for c in df.columns]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl', required=True, help='Path to pickle file (pandas pickled DataFrame)')
    p.add_argument('--artifacts', '-a', default='artifacts', help='Artifacts dir (will write transform_meta.json here)')
    p.add_argument('--iqr_multiplier', type=float, default=10.0, help='IQR multiplier for outlier filtering')
    p.add_argument('--packet_length_cap', type=float, default=65535.0, help='Cap for packet length features (plausibility)')
    p.add_argument('--dry_run', action='store_true', help='Do not overwrite transform_meta.json; just print proposed results')
    args = p.parse_args()

    pkl_path = Path(args.pkl)
    artifacts_dir = Path(args.artifacts)
    if not pkl_path.exists():
        print("Pickle file not found:", pkl_path)
        sys.exit(1)
    if not artifacts_dir.exists():
        print("Artifacts dir not found, creating:", artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pickle (this may take a while)...")
    # Try pandas read_pickle first, fall back to joblib.load for joblib-backed pickles
    df = None
    try:
        df = pd.read_pickle(pkl_path)
    except Exception:
        try:
            obj = joblib.load(pkl_path)
            # if joblib returned a dict with 'full_df', prefer that
            if isinstance(obj, dict) and 'full_df' in obj and obj['full_df'] is not None:
                df = obj['full_df']
            elif isinstance(obj, dict) and 'X' in obj and isinstance(obj['X'], (list, np.ndarray)):
                df = pd.DataFrame(obj['X'])
            elif isinstance(obj, pd.DataFrame):
                df = obj
            else:
                # try to coerce to DataFrame
                try:
                    df = pd.DataFrame(obj)
                except Exception as e:
                    print("Loaded object from joblib is not a DataFrame and cannot be converted:", type(obj), e)
                    sys.exit(1)
        except Exception as e:
            print("Failed to load pickle with pandas or joblib:", e)
            sys.exit(1)
    if not isinstance(df, pd.DataFrame):
        print("Loaded object is not a DataFrame and cannot be converted:", type(df))
        sys.exit(1)

    # Normalize column names
    df.columns = [str(c).replace('\ufeff','').strip() for c in df.columns]

    # Try to re-use existing transform_meta to preserve 'cols' ordering if present
    existing_meta_path = artifacts_dir / 'transform_meta.json'
    cols = None
    heavy_cols = []
    if existing_meta_path.exists():
        try:
            with existing_meta_path.open('r') as f:
                meta_existing = json.load(f)
            cols = meta_existing.get('cols')
            heavy_cols = meta_existing.get('heavy_cols', [])
            print("Using existing 'cols' from", existing_meta_path)
        except Exception as e:
            print("Could not read existing transform_meta.json:", e)

    if not cols:
        cols = infer_cols_from_df(df)
        print(f"Inferred {len(cols)} cols from DataFrame")

    medians = {}
    clip_quantiles = {}
    plaus_warnings = {}
    for c in cols:
        if c not in df.columns:
            medians[c] = None
            clip_quantiles[c] = [None, None]
            continue
        # decide whether to cap packet length
        packet_cap = None
        if any(x in c for x in ["Packet Length Mean","Fwd Packet Length Mean","Bwd Packet Length Mean",
                                "Packet Length Max","Fwd Packet Length Max","Bwd Packet Length Max"]):
            packet_cap = args.packet_length_cap

        med, (q001, q999), used = robust_stats(df[c], iqr_multiplier=args.iqr_multiplier, packet_length_cap=packet_cap)
        medians[c] = med
        clip_quantiles[c] = [q001, q999]

        # simple plausibility checks to warn operator
        if med is not None:
            if any(x in c for x in ["Packet Length Mean","Fwd Packet Length Mean","Bwd Packet Length Mean"]) and med > args.packet_length_cap:
                plaus_warnings[c] = f"median {med} > packet_length_cap {args.packet_length_cap}"
            if any(x in c for x in ["Flow Packets/s","Flow Bytes/s","Fwd Packets/s","Bwd Packets/s"]) and (q999 is not None and q999 > 1e12):
                plaus_warnings[c] = f"q999 very large: {q999}"

    new_meta = {
        "cols": cols,
        "heavy_cols": heavy_cols,
        "medians": medians,
        "clip_quantiles": clip_quantiles
    }

    print("\nSample of computed medians and q999 for first 30 cols:")
    printed = 0
    for k in cols:
        print(k, "median:", medians.get(k), "q999:", clip_quantiles.get(k)[1])
        printed += 1
        if printed >= 30:
            break

    if plaus_warnings:
        print("\nPlausibility warnings (please inspect):")
        for k,v in plaus_warnings.items():
            print(" -", k, ":", v)

    if args.dry_run:
        print("\nDry run: not writing transform_meta.json. Exiting.")
        return

    # backup existing transform_meta.json if present
    if existing_meta_path.exists():
        backup = existing_meta_path.with_suffix('.json.bak')
        existing_meta_path.replace(backup)
        print(f"Backed up existing transform_meta.json to {backup}")

    out_path = artifacts_dir / 'transform_meta.json'
    with out_path.open('w') as f:
        json.dump(new_meta, f, indent=2)
    print(f"Wrote new transform_meta.json to {out_path}")

if __name__ == '__main__':
    main()