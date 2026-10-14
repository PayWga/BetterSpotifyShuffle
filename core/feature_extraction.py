"""Extract auditable acoustic measurements and safely update ShuffleRec's track catalog.

Run from any directory:
    python scripts/feature_extraction.py --dry-run
    python scripts/feature_extraction.py --audio-dir data/temp_audio

Audio files must be supplied locally (and used with appropriate permissions).
The script deliberately does not search/download arbitrary recordings: matching a
search result to the Spotify recording reliably is a separate acquisition task.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

LOG = logging.getLogger("feature_extraction")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKLOG_PATH = DATA_DIR / "dsp_backlog.csv"
MASTER_PATH = DATA_DIR / "master_track_table.csv"
AUDIO_DIR = DATA_DIR / "temp_audio"
SUPPORTED_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
SAMPLE_RATE = 22050
ANALYSIS_SECONDS = 60
EXTRACTOR_VERSION = "librosa_acoustics_v1"
LOCAL_COLUMNS = ("local_rms_db", "local_onset_strength", "local_spectral_centroid", "local_spectral_contrast", "local_beat_regularity")
FEATURE_COLUMNS = ("tempo", "key", "mode") + LOCAL_COLUMNS


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Replace an entire CSV safely against interrupted writes (not concurrent writers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def find_local_audio(track_id: str, audio_dir: Path) -> Path | None:
    for ext in SUPPORTED_EXTENSIONS:
        candidate = audio_dir / f"{track_id}{ext}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def key_mode(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Krumhansl-Schmuckler chroma proxy; threshold is heuristic, not calibrated."""
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr).mean(axis=1)
    if not np.isfinite(chroma).all() or np.std(chroma) < 1e-6:
        return np.nan, np.nan
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    candidates = []
    for k in range(12):
        for mode, profile in ((1, major), (0, minor)):
            corr = np.corrcoef(np.roll(profile, k), chroma)[0, 1]
            if np.isfinite(corr):
                candidates.append((float(corr), k, mode))
    if not candidates:
        return np.nan, np.nan
    score, key, mode = max(candidates)
    return (float(key), float(mode)) if score >= 0.35 else (np.nan, np.nan)


def rhythm(y: np.ndarray, sr: int) -> tuple[float, float]:
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(frames) < 4 or not np.isfinite(tempo) or tempo <= 0:
        return np.nan, np.nan
    times = librosa.frames_to_time(frames, sr=sr)
    intervals = np.diff(times)
    if len(intervals) == 0 or np.mean(intervals) <= 0:
        return tempo, np.nan
    regularity = 1 / (1 + np.std(intervals) / np.mean(intervals))
    return tempo, float(np.clip(regularity, 0, 1))


def extract_audio_features(audio_path: Path, seconds: float = ANALYSIS_SECONDS) -> dict[str, float] | None:
    """Partial acoustic estimates; missing/unreliable measurements remain NaN.

    Local descriptors are NOT Spotify energy/danceability/instrumentalness.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=seconds)
        if len(y) < sr or not np.isfinite(y).all():
            raise ValueError("audio is too short, empty, or non-finite")
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        if rms < 1e-6:
            raise ValueError("audio is effectively silent")
    except Exception as exc:
        LOG.warning("Cannot analyze %s: %s", audio_path, exc)
        return None

    result = {column: np.nan for column in FEATURE_COLUMNS}
    result["local_rms_db"] = float(20 * np.log10(rms))

    analyses = {
        "rhythm": lambda: rhythm(y, sr),
        "tonality": lambda: key_mode(y, sr),
        "onset": lambda: float(np.mean(librosa.onset.onset_strength(y=y, sr=sr))),
        "centroid": lambda: float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "contrast": lambda: float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))),
    }
    for name, compute in analyses.items():
        try:
            value = compute()
            if name == "rhythm":
                result["tempo"], result["local_beat_regularity"] = value
            elif name == "tonality":
                result["key"], result["mode"] = value
            else:
                result[{"onset": "local_onset_strength", "centroid": "local_spectral_centroid", "contrast": "local_spectral_contrast"}[name]] = value
        except Exception as exc:
            LOG.warning("%s failed for %s: %s", name, audio_path.name, exc)

    for col, value in result.items():
        if not np.isfinite(value):
            result[col] = np.nan
    return result


def read_backlog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"track_id": "string"}, low_memory=False)
    if "track_id" not in df:
        raise ValueError("Backlog needs a track_id column")
    df["track_id"] = df["track_id"].str.strip()
    df = df.dropna(subset=["track_id"])
    df = df[df["track_id"] != ""].drop_duplicates(subset=["track_id"]).copy()
    if "attempts" not in df:
        df["attempts"] = 0
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    if "last_error" not in df:
        df["last_error"] = ""
    return df


def save_batch(results: list[dict], master_path: Path) -> None:
    """Fill missing catalog values only. Keep existing values and unrelated columns."""
    if not results:
        return
    master = pd.read_csv(master_path, dtype={"id": "string"}, low_memory=False)
    if "id" not in master:
        raise ValueError("Master CSV must have id")
    master["id"] = master["id"].str.strip()
    if master["id"].isna().any() or (master["id"] == "").any() or master["id"].duplicated().any():
        raise ValueError("Invalid/duplicate master IDs: repair the database first")
    master = master.set_index("id")
    for record in results:
        track_id = record["track_id"]
        if track_id not in master.index:
            master.loc[track_id, "name"] = record.get("track_name")
            master.loc[track_id, "artists"] = record.get("artist_name")
        for col in FEATURE_COLUMNS:
            val = record.get(col, np.nan)
            if pd.isna(val):
                continue
            if col not in master.columns:
                master[col] = np.nan
            provenance = f"{col}_source"
            if provenance not in master.columns:
                master[provenance] = pd.Series(pd.NA, index=master.index, dtype="string")
            if pd.isna(master.at[track_id, col]):
                master.at[track_id, col] = val
                master.at[track_id, provenance] = EXTRACTOR_VERSION
    atomic_csv(master.reset_index(), master_path)


def process(backlog_path: Path = BACKLOG_PATH, master_path: Path = MASTER_PATH,
            audio_dir: Path = AUDIO_DIR, batch_size: int = 10, max_attempts: int = 3,
            dry_run: bool = False) -> dict[str, int]:
    """Process each eligible backlog item once; commit batches then remove successes."""
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master database: {master_path}")
    if not backlog_path.exists():
        raise FileNotFoundError(f"Missing backlog: {backlog_path}")
    if batch_size <= 0 or max_attempts <= 0:
        raise ValueError("batch_size and max_attempts must be positive")
    backlog = read_backlog(backlog_path)
    eligible = backlog[backlog["attempts"] < max_attempts].copy()
    stats = {"eligible": len(eligible), "saved": 0, "failed": 0}
    LOG.info("Eligible backlog entries: %d", len(eligible))
    for start in range(0, len(eligible), batch_size):
        rows = eligible.iloc[start:start + batch_size]
        successful, failures = [], {}
        for item in rows.to_dict("records"):
            tid = str(item["track_id"])
            audio = find_local_audio(tid, audio_dir)
            if audio is None:
                failures[tid] = "audio_not_found"
                continue
            features = extract_audio_features(audio)
            if features is None:
                failures[tid] = "invalid_audio"
                continue
            successful.append({**item, **features, "audio_file": str(audio)})
        if dry_run:
            LOG.info("DRY RUN: would save %d, mark %d failures", len(successful), len(failures))
            stats["saved"] += len(successful)
            stats["failed"] += len(failures)
            continue
        if successful:
            save_batch(successful, master_path)
            successful_ids = {r["track_id"] for r in successful}
            backlog = backlog[~backlog["track_id"].isin(successful_ids)].copy()
            stats["saved"] += len(successful)
        for tid, message in failures.items():
            mask = backlog["track_id"] == tid
            backlog.loc[mask, "attempts"] += 1
            backlog.loc[mask, "last_error"] = message
            stats["failed"] += 1
        atomic_csv(backlog, backlog_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", type=Path, default=BACKLOG_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stats = process(args.backlog, args.master, args.audio_dir, args.batch_size, args.max_attempts, args.dry_run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
