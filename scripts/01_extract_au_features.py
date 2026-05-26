"""
Script 01 v2: Extract AU Features dari DAiSEE Videos menggunakan py-feat
Action Units (20 AU) + temporal stats → jauh lebih akurat dari FER emotions

Laptop sample (verifikasi):
  python 01_extract_au_features.py --sample 50

Laptop full (biarkan jalan 2 hari):
  python 01_extract_au_features.py

Colab A100 full (set di notebook):
  python 01_extract_au_features.py --device cuda --batch-size 16
"""

import os, warnings, argparse, time, tempfile
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from threading import Lock

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DAISEE_ROOT = ROOT.parent / "DAiSEE_extracted" / "DAiSEE"
DATASET_DIR = DAISEE_ROOT / "DataSet"
LABELS_DIR  = DAISEE_ROOT / "Labels"
OUTPUT_DIR  = ROOT / "data" / "au_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BROKEN_VIDEOS = {"2100552061.avi"}

SPLIT_MAP = {
    "train": ("Train",      "TrainLabels.csv"),
    "test":  ("Test",       "TestLabels.csv"),
    "val":   ("Validation", "ValidationLabels.csv"),
}

AU_COLS = [
    "AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10",
    "AU11","AU12","AU14","AU15","AU17","AU20","AU23","AU24",
    "AU25","AU26","AU28","AU43",
]

FPS_TARGET  = 1
ROLLING_WIN = 3

# ── py-feat detector (lazy-load) ──────────────────────────────────────────────
_detector  = None
_det_lock  = Lock()

def get_detector(device="cuda", batch_size=1):
    global _detector
    if _detector is None:
        with _det_lock:
            if _detector is None:
                from feat import Detector
                print(f"Loading py-feat Detector (device={device})…")
                _detector = Detector(
                    au_model="xgb",
                    face_model="retinaface",
                    landmark_model="mobilefacenet",
                    emotion_model="resmasknet",
                    facepose_model="img2pose",
                    device=device,
                )
                print("Detector loaded.")
    return _detector


# ── Single video ──────────────────────────────────────────────────────────────
def extract_one(video_path: Path, device: str, batch_size: int) -> pd.DataFrame | None:
    det = get_detector(device, batch_size)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(fps / FPS_TARGET)))

    # Kumpulkan frame yang diperlukan saja (1fps) lalu simpan ke tmp
    frame_paths = []
    timestamps  = []
    tmpdir = tempfile.mkdtemp()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            p = os.path.join(tmpdir, f"f{idx:05d}.png")
            cv2.imwrite(p, frame)
            frame_paths.append(p)
            timestamps.append(idx / fps)
        idx += 1
    cap.release()

    if not frame_paths:
        _cleanup(tmpdir, frame_paths)
        return None

    try:
        # Batch detect — output_size agar semua frame same shape
        result = det.detect_image(frame_paths, output_size=(640, 480))
    except Exception as e:
        tqdm.write(f"  [warn] {video_path.name}: {e}")
        _cleanup(tmpdir, frame_paths)
        return None

    _cleanup(tmpdir, frame_paths)

    # Ambil AU columns yang berhasil terdeteksi
    au_present = [c for c in AU_COLS if c in result.columns]
    if not au_present:
        return None

    # py-feat bisa return beberapa baris per frame (multi-face).
    # Ambil satu baris per input image — face dengan FaceScore tertinggi.
    result = result.reset_index(drop=True)
    if "input" in result.columns:
        score_col = "FaceScore" if "FaceScore" in result.columns else None
        if score_col:
            result = (result.sort_values(score_col, ascending=False)
                            .groupby("input", sort=False).first()
                            .reset_index(drop=True))
        else:
            result = result.groupby("input", sort=False).first().reset_index(drop=True)

    # Align ke timestamps (ambil min dari keduanya agar aman)
    n  = min(len(result), len(timestamps))
    df = result[au_present].iloc[:n].copy().reset_index(drop=True)
    df.insert(0, "timestamp", timestamps[:n])

    # Isi NaN (frame tanpa muka) dengan forward/backward fill
    df[au_present] = df[au_present].ffill().bfill()
    if df[au_present].isna().all(axis=None):
        return None

    # Tambahkan AU yang hilang sebagai 0
    for col in AU_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Rolling mean untuk smoothing
    df[AU_COLS] = df[AU_COLS].rolling(window=ROLLING_WIN, min_periods=1).mean()

    return df[["timestamp"] + AU_COLS]


def _cleanup(tmpdir, frame_paths):
    for p in frame_paths:
        try:
            os.unlink(p)
        except Exception:
            pass
    try:
        os.rmdir(tmpdir)
    except Exception:
        pass


def find_video(split_dir: Path, clip_id: str) -> Path | None:
    m = list(split_dir.rglob(clip_id))
    return m[0] if m else None


def load_done(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    try:
        return set(pd.read_csv(output_path, usecols=["video_id"])["video_id"].unique())
    except (pd.errors.EmptyDataError, ValueError):
        tqdm.write(f"  [warn] {output_path.name} corrupt — starting fresh")
        output_path.unlink()
        return set()


# ── Per-split ─────────────────────────────────────────────────────────────────
def process_split(split: str, device: str, batch_size: int,
                  sample: int | None, dry_run: bool):
    split_folder, label_file = SPLIT_MAP[split]
    split_dir   = DATASET_DIR / split_folder
    output_path = OUTPUT_DIR / f"{split}_au_features.csv"

    labels = pd.read_csv(LABELS_DIR / label_file)
    labels.columns = labels.columns.str.strip()
    clip_ids  = labels["ClipID"].tolist()
    done      = load_done(output_path)
    remaining = [c for c in clip_ids if c not in done and c not in BROKEN_VIDEOS]

    # Sample mode: ambil N video per split untuk verifikasi laptop
    if sample is not None:
        remaining = remaining[:sample]

    tqdm.write(
        f"\n[{split}] {len(remaining)} to process "
        f"(total={len(clip_ids)}, done={len(done)}"
        + (f", sample={sample}" if sample else "") + ")"
    )
    if dry_run:
        return

    file_mode    = "a" if output_path.exists() else "w"
    write_header = not output_path.exists()
    ok = no_face = missing = 0
    t_start = time.perf_counter()

    with open(output_path, file_mode, buffering=1) as fout:
        pbar = tqdm(total=len(remaining), desc=split, unit="vid")
        for clip_id in remaining:
            vp = find_video(split_dir, clip_id)
            if vp is None:
                missing += 1
            else:
                df = extract_one(vp, device, batch_size)
                if df is None:
                    no_face += 1
                else:
                    df.insert(0, "video_id", clip_id)
                    df.to_csv(fout, index=False, header=write_header)
                    write_header = False
                    ok += 1

            pbar.update(1)
            elapsed = time.perf_counter() - t_start
            rate    = ok / max(elapsed, 1)
            remain  = (len(remaining) - ok - no_face - missing) / max(rate, 1e-6)
            pbar.set_postfix(ok=ok, no_face=no_face, miss=missing,
                             eta=f"{remain/3600:.1f}h")
        pbar.close()

    tqdm.write(f"[{split}] done — ok={ok}, no_face={no_face}, missing={missing}")


def aggregate(split: str):
    src = OUTPUT_DIR / f"{split}_au_features.csv"
    dst = OUTPUT_DIR / f"{split}_au_features_agg.csv"
    if not src.exists():
        print(f"[agg] {src.name} not found.")
        return
    df  = pd.read_csv(src)
    agg = df.groupby("video_id")[AU_COLS].mean().reset_index()
    agg.to_csv(dst, index=False)
    print(f"[agg] {split}: {len(agg)} videos → {dst.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits",     nargs="+", default=["train", "test", "val"])
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size untuk Colab A100 (default 1 untuk laptop)")
    p.add_argument("--sample",     type=int, default=None,
                   help="Proses hanya N video per split (untuk verifikasi laptop)")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--agg-only",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.agg_only:
        for s in args.splits:
            aggregate(s)
        return

    print(f"Device    : {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Splits    : {args.splits}")
    print(f"Sample    : {args.sample if args.sample else 'full'}\n")

    # Pre-load detector
    get_detector(args.device, args.batch_size)

    for split in args.splits:
        process_split(split, args.device, args.batch_size,
                      args.sample, args.dry_run)
        if not args.dry_run:
            aggregate(split)

    print("\n[Done]")


if __name__ == "__main__":
    main()
