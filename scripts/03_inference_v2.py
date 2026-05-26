"""
Script 03 v2: Inference menggunakan model XGBoost + temporal features
Mendukung: CSV frame-level, CSV aggregated, atau 7 nilai FER inline.

Usage:
  # Dari frame-level CSV (direkomendasikan — bisa hitung temporal features)
  python 03_inference_v2.py --input ../project/data/fer_features/test_features.csv

  # Dari aggregated CSV (temporal stats tidak tersedia, fallback ke mean saja)
  python 03_inference_v2.py --input ../project/data/fer_features/test_features_agg.csv --agg

  # Single clip: 7 nilai FER (tanpa temporal, hanya mean)
  python 03_inference_v2.py --single 0.05 0.02 0.01 0.60 0.05 0.10 0.17
"""

import argparse, sys, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import skew
from scipy.stats import linregress

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_models.joblib"

STATES   = ["Boredom", "Confusion", "Frustration", "Engagement"]
FER_COLS = ["p_angry","p_disgust","p_fear","p_happy","p_sad","p_surprise","p_neutral"]

FEAT_COLS = (
    [f"{e}_{s}" for e in FER_COLS
     for s in ["mean","std","max","min","skew","slope"]]
    + ["n_frames"]
)


def load_models():
    if not MODEL_PATH.exists():
        sys.exit(f"[Error] Model tidak ditemukan: {MODEL_PATH}\n"
                 "Jalankan 02_calibrate_v2.py terlebih dahulu.")
    return joblib.load(MODEL_PATH)


def temporal_features(group: pd.DataFrame) -> dict:
    feats = {}
    for col in FER_COLS:
        vals = group[col].values
        feats[f"{col}_mean"]  = float(np.mean(vals))
        feats[f"{col}_std"]   = float(np.std(vals))
        feats[f"{col}_max"]   = float(np.max(vals))
        feats[f"{col}_min"]   = float(np.min(vals))
        feats[f"{col}_skew"]  = float(skew(vals)) if len(vals) > 2 else 0.0
        if len(vals) > 1:
            slope, *_ = linregress(np.arange(len(vals)), vals)
            feats[f"{col}_slope"] = float(slope)
        else:
            feats[f"{col}_slope"] = 0.0
    feats["n_frames"] = len(group)
    return feats


def build_feat_df(df: pd.DataFrame, is_frame_level: bool) -> pd.DataFrame:
    """Ubah input df menjadi feature matrix 43 kolom."""
    if is_frame_level and "video_id" in df.columns:
        rows = []
        for vid_id, grp in df.groupby("video_id"):
            row = {"video_id": vid_id}
            row.update(temporal_features(grp))
            rows.append(row)
        return pd.DataFrame(rows)
    else:
        # Aggregated atau single: isi std/max/min/skew/slope dengan nol
        out = pd.DataFrame()
        if "video_id" in df.columns:
            out["video_id"] = df["video_id"]
        for col in FER_COLS:
            out[f"{col}_mean"]  = df[col].values
            out[f"{col}_std"]   = 0.0
            out[f"{col}_max"]   = df[col].values
            out[f"{col}_min"]   = df[col].values
            out[f"{col}_skew"]  = 0.0
            out[f"{col}_slope"] = 0.0
        out["n_frames"] = 1
        return out


def run_inference(feat_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    X   = feat_df[FEAT_COLS].values.astype(np.float32)
    out = feat_df[["video_id"]].copy() if "video_id" in feat_df.columns \
          else pd.DataFrame(index=feat_df.index)

    for state in STATES:
        model  = models[state]["model"]
        thresh = models[state]["threshold"]
        prob   = model.predict_proba(X)[:, 1]
        out[f"{state}_pred"]       = (prob >= thresh).astype(int)
        out[f"{state}_confidence"] = np.round(prob, 4)

    return out


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input",  type=Path)
    g.add_argument("--single", nargs=7, type=float,
                   metavar=("ANGRY","DISGUST","FEAR","HAPPY","SAD","SURPRISE","NEUTRAL"))
    p.add_argument("--agg",    action="store_true",
                   help="Input adalah aggregated CSV (temporal features tidak tersedia)")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    models = load_models()

    if args.single:
        raw = dict(zip(FER_COLS, args.single))
        df  = pd.DataFrame([raw])
        df["video_id"] = "single_input"
        feat_df = build_feat_df(df, is_frame_level=False)
        print(f"\nInput: {raw}")
    else:
        df = pd.read_csv(args.input)
        is_frame = not args.agg and "timestamp" in df.columns
        feat_df  = build_feat_df(df, is_frame_level=is_frame)
        print(f"Input: {len(feat_df)} video(s) "
              f"({'frame-level → temporal' if is_frame else 'aggregated'})")

    out = run_inference(feat_df, models)

    print(f"\n{'='*72}")
    print(f"  PREDIKSI  ({len(out)} clips)")
    print(f"{'='*72}")
    print(f"{'video_id':<22}" +
          "".join(f"  {s[:4]}_pred conf" for s in STATES))
    print("─" * 72)
    for _, row in out.head(20).iterrows():
        vid  = str(row.get("video_id",""))[:22]
        line = f"{vid:<22}"
        for s in STATES:
            line += f"  {int(row[f'{s}_pred'])}         {row[f'{s}_confidence']:.3f}"
        print(line)
    if len(out) > 20:
        print(f"  … ({len(out)-20} baris lagi)")

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"\n[Saved] → {args.output}")


if __name__ == "__main__":
    main()
