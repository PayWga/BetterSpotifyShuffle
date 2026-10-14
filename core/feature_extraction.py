"""Extract auditable acoustic measurements and safely update ShuffleRec's track catalog.

Run from any directory:
    python core/feature_extraction.py --dry-run --limit 10
    python core/feature_extraction.py --limit 10
    python core/feature_extraction.py --local-only

Uses cached/local audio first, then searches YouTube using yt-dlp. Matching is
conservative but heuristic; it cannot prove recording identity. Use audio with
appropriate permissions. Dry runs never download audio or write files.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
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


def text_value(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def artist_names(value) -> list[str]:
    text = text_value(value)
    if text.startswith("["):
        try:
            names = ast.literal_eval(text)
            if isinstance(names, list):
                return [str(name).strip() for name in names if str(name).strip()]
        except (ValueError, SyntaxError):
            pass
    return [text] if text else []


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def match_score(item: dict, entry: dict) -> float | None:
    """Reject obvious mismatches before ranking a small set of search results."""
    title = normalized(text_value(item.get("track_name")))
    artists = artist_names(item.get("artist_name"))
    candidate = normalized(text_value(entry.get("title")))
    if not title or not artists or not candidate or entry.get("is_live"):
        return None
    words = set(candidate.split())
    if f" {title} " not in f" {candidate} ":
        return None
    credits = " " + " ".join([candidate] + [normalized(text_value(entry.get(k)))
                                           for k in ("uploader", "channel", "artist")]) + " "
    if not all(" " + normalized(artist) + " " in credits for artist in artists):
        return None
    versions = {"live", "cover", "remix", "acoustic", "karaoke", "instrumental",
                "slowed", "sped", "nightcore", "remaster", "remastered", "demo", "edit"}
    # Version terms belonging to the artist's name are not recording labels.
    requested = set(normalized(title + " " + " ".join(artists)).split())
    if (words & versions) - requested:
        return None
    if (set(title.split()) & versions) - words:
        return None
    score = 1.0
    expected_ms = pd.to_numeric(item.get("duration_ms"), errors="coerce")
    actual = pd.to_numeric(entry.get("duration"), errors="coerce")
    if pd.notna(expected_ms) and expected_ms > 0 and pd.notna(actual):
        expected = expected_ms / 1000
        error = abs(actual - expected)
        if error > max(12., expected * .12):
            return None
        score += 1 - error / max(12., expected * .12)
    if "topic" in normalized(text_value(entry.get("channel") or entry.get("uploader"))):
        score += .25
    return score


def audio_source(audio: Path) -> str:
    sidecar = audio.with_suffix(".source.json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))["url"]
        except (OSError, ValueError, KeyError, TypeError):
            LOG.warning("Cannot read source sidecar for %s", audio.name)
    return str(audio.resolve())


def acquire_audio(item: dict, audio_dir: Path) -> tuple[Path | None, str | None]:
    """Search five candidates, download the best match, and cache its provenance."""
    tid = str(item["track_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tid):
        raise ValueError("invalid_track_id")
    title = text_value(item.get("track_name"))
    artists = artist_names(item.get("artist_name"))
    if not title or not artists:
        raise ValueError("missing_search_metadata")
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install yt-dlp: python -m pip install 'yt-dlp[default]'") from exc
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and ffprobe must be available on PATH")
    common = {"quiet": True, "no_warnings": False, "noplaylist": True,
              "socket_timeout": 20, "retries": 2, "fragment_retries": 2,
              "extractor_retries": 2, "cachedir": False, "noprogress": True}
    runtimes = {runtime: {} for runtime in ("deno", "node") if shutil.which(runtime)}
    if runtimes:
        common["js_runtimes"] = runtimes
    try:
        with yt_dlp.YoutubeDL({**common, "extract_flat": "in_playlist"}) as search:
            info = search.extract_info(f"ytsearch5:{' '.join(artists)} {title} official audio", download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError("search_failed: check network access and yt-dlp version") from exc
    candidates = []
    for entry in (info or {}).get("entries", []) or []:
        if not entry:
            continue
        score = match_score(item, entry)
        url = entry.get("webpage_url") or entry.get("url")
        if score is not None and url:
            candidates.append((score, url))
    if not candidates:
        return None, None
    audio_dir.mkdir(parents=True, exist_ok=True)
    options = {**common, "format": "bestaudio/best", "overwrites": False,
               "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}
    errors = []
    for _, url in sorted(candidates, key=lambda pair: pair[0], reverse=True):
        try:
            # Never resume one recording's partial download as another recording.
            # Retain failed staging files for inspection rather than deleting them.
            staging = Path(tempfile.mkdtemp(prefix=f".{tid}_", dir=audio_dir))
            with yt_dlp.YoutubeDL({**options, "outtmpl": str(staging / f"{tid}.%(ext)s")}) as downloader:
                result = downloader.extract_info(url, download=True)
            audio = staging / f"{tid}.wav"
            if not audio.is_file() or audio.stat().st_size == 0:
                raise RuntimeError("download produced no converted WAV")
            source = result.get("webpage_url") or url
            sidecar = audio.with_suffix(".source.json")
            sidecar.write_text(
                json.dumps({"url": source, "title": result.get("title"),
                            "matching": "artist_title_version_duration_v1"}, ensure_ascii=False),
                encoding="utf-8")
            # Publish provenance first, so an interrupted run cannot expose an
            # acquired WAV at the cache root without its source.
            destination = audio_dir / f"{tid}.wav"
            os.replace(sidecar, destination.with_suffix(".source.json"))
            os.replace(audio, destination)
            return destination, source
        except Exception as exc:
            errors.append(type(exc).__name__)
            LOG.warning("Download candidate failed for %s (%s)", tid, type(exc).__name__)
    raise RuntimeError("download_failed: " + ", ".join(errors))


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
    if not re.fullmatch(r"[A-Za-z0-9_-]+", track_id):
        raise ValueError("invalid_track_id")
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
    df = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""], low_memory=False)
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
    df["last_error"] = df["last_error"].astype("string").fillna("")
    return df


def read_master(path: Path) -> pd.DataFrame:
    """Preserve CSV metadata text; only empty cells and numeric NaNs are missing."""
    master = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""], low_memory=False)
    if "id" not in master:
        raise ValueError("Master CSV must have id")
    master["id"] = master["id"].str.strip()
    if master["id"].isna().any() or (master["id"] == "").any() or master["id"].duplicated().any():
        raise ValueError("Invalid/duplicate master IDs: repair the database first")
    for col in FEATURE_COLUMNS:
        if col in master:
            values = master[col].mask(master[col].str.casefold() == "nan")
            # Object columns accept new numeric estimates without rewriting
            # existing feature text or triggering pandas string-dtype errors.
            master[col] = values.astype(object)
    return master.set_index("id")


def save_batch(results: list[dict], master_path: Path) -> None:
    """Fill missing catalog values only. Keep existing values and unrelated columns."""
    if not results:
        return
    master = read_master(master_path)
    for record in results:
        track_id = record["track_id"]
        if track_id not in master.index:
            operational = {"track_id", "attempts", "last_error", "audio_file", "audio_source"}
            metadata = {key: value for key, value in record.items()
                        if key not in operational and key not in FEATURE_COLUMNS and key != "id"}
            metadata.setdefault("name", record.get("track_name"))
            metadata.setdefault("artists", record.get("artist_name"))
            new_row = pd.DataFrame([metadata], index=pd.Index([track_id], name="id"))
            master = pd.concat([master, new_row])
        for col in FEATURE_COLUMNS:
            val = record.get(col, np.nan)
            if pd.isna(val):
                continue
            if col not in master.columns:
                master[col] = np.nan
            provenance = f"{col}_source"
            if provenance not in master.columns:
                master[provenance] = pd.Series(pd.NA, index=master.index, dtype="string")
            if not isinstance(master[provenance].dtype, pd.StringDtype):
                master[provenance] = master[provenance].astype("string")
            if pd.isna(master.at[track_id, col]):
                master.at[track_id, col] = val
                master.at[track_id, provenance] = EXTRACTOR_VERSION
                source_col = f"{col}_audio_source"
                if source_col not in master:
                    master[source_col] = pd.Series(pd.NA, index=master.index, dtype="string")
                if not isinstance(master[source_col].dtype, pd.StringDtype):
                    master[source_col] = master[source_col].astype("string")
                master.at[track_id, source_col] = record.get("audio_source", record.get("audio_file", pd.NA))
    atomic_csv(master.reset_index(), master_path)


def process(backlog_path: Path = BACKLOG_PATH, master_path: Path = MASTER_PATH,
            audio_dir: Path = AUDIO_DIR, batch_size: int = 10, max_attempts: int = 3,
            dry_run: bool = False, local_only: bool = False,
            limit: int | None = None, retry_failed: bool = False) -> dict[str, int]:
    """Process each eligible backlog item once; commit batches then remove successes."""
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master database: {master_path}")
    if not backlog_path.exists():
        raise FileNotFoundError(f"Missing backlog: {backlog_path}")
    if batch_size <= 0 or max_attempts <= 0:
        raise ValueError("batch_size and max_attempts must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    master = read_master(master_path)
    backlog = read_backlog(backlog_path)
    eligible = (backlog if retry_failed else backlog[backlog["attempts"] < max_attempts]).copy()
    if limit is not None:
        eligible = eligible.head(limit)
    stats = {"eligible": len(eligible), "saved": 0, "failed": 0}
    LOG.info("Eligible backlog entries: %d", len(eligible))
    for start in range(0, len(eligible), batch_size):
        rows = eligible.iloc[start:start + batch_size]
        successful, failures = [], {}
        for item in rows.to_dict("records"):
            tid = str(item["track_id"])
            if tid in master.index:
                metadata = master.loc[tid]
                for target, names in {"track_name": ("name", "track_name"),
                                      "artist_name": ("artists", "artist_name"),
                                      "duration_ms": ("duration_ms",)}.items():
                    if not text_value(item.get(target)):
                        item[target] = next((metadata[name] for name in names
                                             if name in metadata and text_value(metadata[name])), None)
            LOG.info("Processing %s", tid)
            try:
                audio = find_local_audio(tid, audio_dir)
                source = audio_source(audio) if audio else None
                if audio is None and not local_only and not dry_run:
                    audio, source = acquire_audio(item, audio_dir)
            except Exception as exc:
                # Persist a useful reason without storing URLs, tokens or full network responses.
                reason = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__
                failures[tid] = reason[:200]
                LOG.warning("Audio acquisition failed for %s: %s", tid, failures[tid])
                continue
            if audio is None:
                failures[tid] = "audio_not_found"
                continue
            features = extract_audio_features(audio)
            if features is None:
                failures[tid] = "invalid_audio"
                continue
            successful.append({**item, **features, "audio_file": str(audio), "audio_source": source})
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
    parser.add_argument("--local-only", action="store_true", help="Disable audio search/download")
    parser.add_argument("--limit", type=int, help="Maximum number of tracks to process this run")
    parser.add_argument("--retry-failed", action="store_true", help="Include tracks at the attempt limit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stats = process(args.backlog, args.master, args.audio_dir, args.batch_size, args.max_attempts,
                    args.dry_run, args.local_only, args.limit, args.retry_failed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
