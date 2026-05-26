"""
Script 02 v2: Improved Calibration (FER → DAiSEE affective states)

Peningkatan dari v1:
  1. Temporal features  — bukan hanya mean, tapi std/max/min/skew/slope per emosi
                          (42 fitur dari frame-level, bukan 3-4 dari aggregated)
  2. Semua 7 emosi      — tidak dibatasi subset teori; biarkan model yang pilih
  3. XGBoost classifier — tangkap interaksi non-linear antar fitur
  4. Threshold tuning   — cari threshold optimal per state di validation set (max F1)
  5. Feature importance — tampilkan fitur terpenting per state
"""

import json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import skew
from scipy.stats import linregress
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, classification_report,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
V1_DATA    = ROOT.parent / "project" / "data"
FEAT_DIR   = V1_DATA / "fer_features"
LABEL_DIR  = V1_DATA / "binary_labels"
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

COEFF_PATH = MODEL_DIR / "calibration_v2.json"
MODEL_PATH = MODEL_DIR / "xgb_models.joblib"

STATES    = ["Boredom", "Confusion", "Frustration", "Engagement"]
FER_COLS  = ["p_angry","p_disgust","p_fear","p_happy","p_sad","p_surprise","p_neutral"]


# ── Temporal feature extraction ───────────────────────────────────────────────
def temporal_features(group: pd.DataFrame) -> dict:
    """
    Dari frame-level FER per video, hitung 6 statistik × 7 emosi = 42 fitur.
    Tambah n_frames sebagai fitur ke-43.
    """
    feats = {}
    for col in FER_COLS:
        vals = group[col].values
        feats[f"{col}_mean"]  = float(np.mean(vals))
        feats[f"{col}_std"]   = float(np.std(vals))
        feats[f"{col}_max"]   = float(np.max(vals))
        feats[f"{col}_min"]   = float(np.min(vals))
        feats[f"{col}_skew"]  = float(skew(vals)) if len(vals) > 2 else 0.0
        # Slope: tren naik/turun emosi sepanjang video
        if len(vals) > 1:
            slope, *_ = linregress(np.arange(len(vals)), vals)
            feats[f"{col}_slope"] = float(slope)
        else:
            feats[f"{col}_slope"] = 0.0
    feats["n_frames"] = len(group)
    return feats


def build_temporal_dataset(split: str):
    frame_path = FEAT_DIR / f"{split}_features.csv"
    label_path = LABEL_DIR / f"{split}_binary.csv"

    frames = pd.read_csv(frame_path)
    labels = pd.read_csv(label_path)
    labels.columns = labels.columns.str.strip()

    # Compute temporal features per video
    feat_rows = []
    for vid_id, grp in frames.groupby("video_id"):
        row = {"video_id": vid_id}
        row.update(temporal_features(grp))
        feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows)
    merged  = feat_df.merge(labels, left_on="video_id", right_on="ClipID", how="inner")

    print(f"  [{split}] {len(merged)} videos, {len(feat_df.columns)-1} fitur temporal")
    return merged


# ── Threshold tuning ──────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob, step=0.01) -> tuple[float, float]:
    """Cari threshold yang maksimalkan F1 di validation set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, step):
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t  = t
    return float(best_t), float(best_f1)


# ── Train & evaluate ──────────────────────────────────────────────────────────
FEAT_COLS_TEMPORAL = (
    [f"{e}_{s}" for e in FER_COLS
     for s in ["mean","std","max","min","skew","slope"]]
    + ["n_frames"]
)

def get_X(df: pd.DataFrame) -> np.ndarray:
    return df[FEAT_COLS_TEMPORAL].values.astype(np.float32)


def train_state(
    train_df, val_df, test_df, state: str
) -> tuple[dict, object, float]:

    X_train, y_train = get_X(train_df), train_df[state].values
    X_val,   y_val   = get_X(val_df),   val_df[state].values
    X_test,  y_test  = get_X(test_df),  test_df[state].values

    # Hitung scale_pos_weight untuk imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Threshold tuning di val set
    val_prob  = model.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(y_val, val_prob)

    # Metrics
    results = {"state": state, "threshold": best_t, "metrics": {}}
    for sp_name, X_sp, y_sp in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        y_prob = model.predict_proba(X_sp)[:, 1]
        y_pred = (y_prob >= best_t).astype(int)
        results["metrics"][sp_name] = {
            "accuracy":  float(accuracy_score(y_sp, y_pred)),
            "precision": float(precision_score(y_sp, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_sp, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_sp, y_pred, zero_division=0)),
            "kappa":     float(cohen_kappa_score(y_sp, y_pred)),
            "report":    classification_report(y_sp, y_pred, zero_division=0),
        }

    # Top-10 feature importance
    imp   = model.feature_importances_
    names = FEAT_COLS_TEMPORAL
    top10 = sorted(zip(names, imp), key=lambda x: -x[1])[:10]
    results["top_features"] = {k: float(v) for k, v in top10}

    return results, model, best_t


def print_metrics(results: dict):
    state = results["state"]
    print(f"\n{'─'*60}")
    print(f"  {state}  (threshold={results['threshold']:.2f})")
    print(f"{'─'*60}")
    print(f"  Top features:")
    for k, v in list(results["top_features"].items())[:5]:
        print(f"    {k:<30} {v:.4f}")
    print(f"\n  {'Split':<8} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Kappa':>7}")
    for sp, m in results["metrics"].items():
        print(f"  {sp:<8} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['kappa']:>7.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Membangun temporal features dari frame-level data…")
    train_df = build_temporal_dataset("train")
    val_df   = build_temporal_dataset("val")
    test_df  = build_temporal_dataset("test")

    all_results = {}
    all_models  = {}

    for state in STATES:
        results, model, threshold = train_state(train_df, val_df, test_df, state)
        print_metrics(results)
        all_results[state] = {
            k: v for k, v in results.items() if k != "metrics"
        }
        all_results[state]["metrics"] = {
            sp: {mk: mv for mk, mv in m.items() if mk != "report"}
            for sp, m in results["metrics"].items()
        }
        all_models[state] = {"model": model, "threshold": threshold}

    # Simpan JSON hasil + model
    with open(COEFF_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    joblib.dump(all_models, MODEL_PATH)
    print(f"\n[Saved] Hasil   → {COEFF_PATH}")
    print(f"[Saved] Models  → {MODEL_PATH}")

    # Full classification reports
    print("\n" + "="*60)
    print("  FULL CLASSIFICATION REPORTS (test set)")
    print("="*60)
    X_test = get_X(test_df)
    for state in STATES:
        y_test  = test_df[state].values
        model   = all_models[state]["model"]
        thresh  = all_models[state]["threshold"]
        y_prob  = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_prob >= thresh).astype(int)
        print(f"\n  [{state}]  threshold={thresh:.2f}")
        print(classification_report(y_test, y_pred, zero_division=0,
              target_names=["not present", "present"]))

    # Perbandingan ringkas v1 vs v2
    v1_path = ROOT.parent / "project" / "models" / "calibration_coefficients.json"
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print("\n" + "="*60)
        print("  PERBANDINGAN v1 (LogReg, mean) vs v2 (XGBoost, temporal)")
        print("="*60)
        print(f"  {'State':<14} {'v1 F1':>8} {'v2 F1':>8} {'Delta':>8} {'v1 Kappa':>10} {'v2 Kappa':>10}")
        print("  " + "─"*58)
        for state in STATES:
            f1_v1    = v1[state]["metrics"]["test"]["f1"]
            kappa_v1 = v1[state]["metrics"]["test"]["kappa"]
            f1_v2    = all_results[state]["metrics"]["test"]["f1"]
            kappa_v2 = all_results[state]["metrics"]["test"]["kappa"]
            delta    = f1_v2 - f1_v1
            arrow    = "▲" if delta > 0 else "▼"
            print(f"  {state:<14} {f1_v1:>8.3f} {f1_v2:>8.3f} {arrow}{abs(delta):>7.3f} "
                  f"{kappa_v1:>10.3f} {kappa_v2:>10.3f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
