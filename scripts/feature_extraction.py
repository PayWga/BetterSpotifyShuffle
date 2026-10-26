"""Extract auditable acoustic measurements and safely update ShuffleRec's track catalog.

Run from any directory:
    python scripts/feature_extraction.py --dry-run --limit 10
    python scripts/feature_extraction.py --limit 100 --workers 4
    python scripts/feature_extraction.py --analysis-seconds full

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
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stderr
from copy import copy
from functools import cache
from io import StringIO
from pathlib import Path
from time import perf_counter, sleep
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

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
CSV_REPLACE_DELAYS = (.25, .5, 1., 2., 4.)
DEFERRED_ACQUISITION_CODES = ("authentication_required", "http_429",
                              "session_reload_required", "js_challenge_failed")


class CookieConfigurationError(RuntimeError):
    """Authentication could not be loaded before processing the backlog."""


def load_youtube_cookies(cookie_file: Path | None, browser: str | None):
    """Read opt-in authentication once; never export or rewrite its source."""
    if cookie_file is None and browser is None:
        return None
    if cookie_file is not None and browser is not None:
        raise CookieConfigurationError("Choose either --cookies or --cookies-from-browser, not both.")
    if cookie_file is not None and not Path(cookie_file).is_file():
        raise CookieConfigurationError("The supplied cookie file does not exist or is not a file.")
    import yt_dlp
    from yt_dlp.cookies import SUPPORTED_BROWSERS, load_cookies
    specification = None
    if browser is not None:
        name, _, profile = browser.partition(":")
        if name not in SUPPORTED_BROWSERS:
            raise CookieConfigurationError("Unsupported cookie browser. Use BROWSER or BROWSER:PROFILE; "
                                           "supported browsers: " + ", ".join(sorted(SUPPORTED_BROWSERS)))
        specification = (name, profile) if profile else (name,)
    try:
        # yt-dlp can print raw malformed cookie lines even in quiet mode. Load
        # before starting workers and suppress those potentially secret values.
        # No cookiefile parameter on the loader means exiting cannot write back.
        with redirect_stderr(StringIO()), yt_dlp.YoutubeDL({"quiet": True}) as loader:
            jar = load_cookies(str(cookie_file) if cookie_file is not None else None,
                               specification, loader)
    except Exception:
        raise CookieConfigurationError("Could not load cookies. For browser cookies, close the browser "
                                       "and check the profile/decryption support; alternatively use "
                                       "--cookies with a Netscape-format export. No tracks were attempted.") from None
    if not len(jar):
        raise CookieConfigurationError("No cookies were loaded. Check the cookie file or browser profile.")
    return jar


def youtube_session(options: dict, cookiejar=None):
    """Give each yt-dlp context independent in-memory cookie objects."""
    import yt_dlp
    session = yt_dlp.YoutubeDL(options)
    if cookiejar is not None:
        from yt_dlp.cookies import YoutubeDLCookieJar
        session.cookiejar = YoutubeDLCookieJar()
        for cookie in cookiejar:
            session.cookiejar.set_cookie(copy(cookie))
    return session


class CsvSaveBlocked(PermissionError):
    """The prepared CSV remains available when destination access stays denied."""

    def __init__(self, destination: Path, snapshot: Path):
        self.destination = destination
        self.snapshot = snapshot
        super().__init__(
            f"Cannot replace {destination} after {len(CSV_REPLACE_DELAYS) + 1} attempts. "
            "Check programs holding the CSV open and file/folder permissions. "
            f"Prepared snapshot retained at {snapshot}. The destination was not replaced by this save.")


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
    text = re.sub(r"['’`]", "", text)
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def canonical_title(text: str) -> str:
    text = re.sub(r"(?i)(\(\s*)with\b", r"\1feat", text)
    text = normalized(text)
    text = re.sub(r"\b(?:ft|featuring)\b", "feat", text)
    text = re.sub(r"\b(?:rmx|re\s+mix|remixed)\b", "remix", text)
    text = re.sub(r"\bremastered\b", "remaster", text)
    text = re.sub(r"\b(japanese|korean|english|chinese) ver\b", r"\1 version", text)
    text = re.sub(r"\b(?:jp|jpn) (?:ver|version)\b", "japanese version", text)
    text = re.sub(r"\bstripped (?:down|back)(?: version)?\b", "stripped", text)
    return re.sub(r"\bm v\b", "music video", text)


def language_annotation(text: str) -> str | None:
    label = canonical_title(text)
    match = re.fullmatch(r"(japanese|korean|english|chinese)(?: (?:version|official|music|video|audio))*", label)
    return match.group(1) if match else None


def recording_languages(text: str) -> set[str]:
    labels = re.split(r"[()\[\]「」]|\s+[-–—]\s+", text)
    languages = {value for label in labels if (value := language_annotation(label))}
    languages.update(re.findall(r"\b(japanese|korean|english|chinese) version\b", canonical_title(text)))
    return languages


def recording_annotation(text: str) -> bool:
    return bool(language_annotation(text) or re.fullmatch(
        r"(?:live|stripped|acoustic|instrumental|karaoke|cover|demo|sped up|slowed|"
        r"(?:(?:19|20)\d{2} )?remaster(?: (?:19|20)\d{2})?)(?: version)?", canonical_title(text)))


def bilingual_title(text: str) -> str:
    """Recognize a non-Latin title followed by its explicit Latin-script alias."""
    def replace(match):
        alias = match.group(2)
        if recording_annotation(alias) or normalized(alias) in {'lyrics', 'audio', 'video'}:
            return match.group(0)
        return alias
    return re.sub(r"(?<!\w)([^\W\d_A-Za-z]+(?:\s+[^\W\d_A-Za-z]+)*)\s*[\[(]"
                  r"([A-Za-z0-9][^()\[\]]*)[\])]", replace, text)


def title_parts(text: str) -> tuple[str, str]:
    text = bilingual_title(text)
    featured = []

    def feature(match):
        featured.append(match.group(1).strip())
        return ' '

    text = re.sub(r"(?i)\(\s*(?:feat\.?|ft\.?|featuring|with)\s+([^()]*)\)", feature, text)
    text = re.sub(r"[\[(]([^()\[\]]+)[\])]",
                  lambda match: ' ' if recording_annotation(match.group(1)) else match.group(0), text)
    segments = re.split(r"\s+[-–—]\s+", text)
    while len(segments) > 1 and recording_annotation(segments[-1]):
        segments.pop()
    text = canonical_title(' - '.join(segments))
    text = re.sub(r"(?:\s+(?:19|20)\d{2})?\s+remaster(?:\s+(?:19|20)\d{2})?$", "", text)
    text = re.sub(r"\s+(?:japanese|korean|english|chinese) version$", '', text)
    parts = text.split(" feat ", 1)
    if len(parts) == 2:
        featured.append(parts[1])
    return parts[0].strip(), canonical_title(', '.join(featured))


def credit_present(artist: str, credits: str) -> bool:
    if f" {normalized(artist)} " in credits:
        return True
    # Spotify enrichment and many exports join collaborators with commas. Do
    # not accept only the first artist: every separated credit must be present.
    names = [name.strip() for name in re.split(r"[,;]", artist) if name.strip()]
    return len(names) > 1 and all(f" {normalized(name)} " in credits for name in names)


def remix_label(text: str) -> bool:
    """Identify explicit remix/alternate-mix labels for version matching."""
    return bool(re.search(r"\b(?:re\s*mix(?:ed|es)?|rmx|bootleg|mash\s*up|rework(?:ed)?|"
                          r"(?:club|dance|dj|extended|dub|festival)\s*mix)\b", normalized(text)))


def remix_recording(entry: dict) -> bool:
    for key in ("title", "track", "alt_title", "version", "tags"):
        value = entry.get(key)
        values = value if isinstance(value, (list, tuple)) else [value]
        if any(remix_label(value) for value in values if isinstance(value, str)):
            return True
    return bool(entry.get("remixed_by") or entry.get("remix_artist"))


def presentation_title(text: str) -> str:
    """Remove presentation brackets; original recording labels remain checked."""
    words = {"official", "unofficial", "audio", "video", "music", "lyric", "lyrics",
             "letra", "letras", "paroles", "text", "visualizer", "visualiser", "hd", "hq",
             "4k", "1080p", "fan", "made", "fanmade", "reupload", "reuploaded", "upload",
             "full", "only", "lyrical", "m", "v"}

    def replace(match):
        label = normalized(match.group(1))
        if (label and set(label.split()) <= words) or re.match(r"^(?:lyrics?|letras?|paroles) by\s+", label):
            return " "
        return match.group(0)

    return re.sub(r"[\[(]([^\[\]()]+)[\])]", replace, text)


def candidate_song_title(entry: dict, artists: list[str], featured: str) -> str:
    """Separate artist/title order and recognized pipe-delimited branding."""
    raw = text_value(entry.get('title'))
    segments = raw.split('|')
    branding = {normalized(text_value(entry.get(key))) for key in ('uploader', 'channel')}
    if len(segments) > 1 and all(not presentation_title(f'({segment})').strip()
                                or normalized(segment) in branding for segment in segments[1:]):
        raw = segments[0]
    parts = re.split(r"\s+[-–—]\s+", raw, maxsplit=1)
    if len(parts) == 2:
        if any(credit_present(artist, ' ' + normalized(parts[0]) + ' ') for artist in artists):
            raw = parts[1]
        else:
            remainder = ' ' + canonical_title(presentation_title(parts[1])) + ' '
            for artist in artists + ([featured] if featured else []):
                for name in [artist] + re.split(r'[,;]', artist):
                    remainder = remainder.replace(' ' + normalized(name.strip()) + ' ', ' ')
            if not (set(remainder.split()) - {'feat', 'and', 'x'}):
                raw = parts[0]
    return presentation_title(raw)


def match_result(item: dict, entry: dict, *, allow_missing_artists: bool = False) -> tuple[float | None, str]:
    """Handle formatting variations without accepting conflicting recordings."""
    title = canonical_title(text_value(item.get("track_name")))
    core_title, featured = title_parts(text_value(item.get("track_name")))
    artists = artist_names(item.get("spotify_artist_names")) or artist_names(item.get("artist_name"))
    candidate = canonical_title(text_value(entry.get("title")))
    requested_remix = remix_label(title)
    candidate_remix = remix_recording(entry)
    if candidate_remix and not requested_remix:
        return None, "remix"
    if requested_remix and not candidate_remix:
        return None, "remix_required"
    if not title or not artists or not candidate or entry.get("is_live"):
        return None, "missing_metadata_or_live_stream"
    requested_languages = recording_languages(text_value(item.get('track_name')))
    candidate_languages = recording_languages(text_value(entry.get('title')))
    if candidate_languages != requested_languages:
        return None, 'language_version'
    words = set(candidate.split())
    expected_ms = pd.to_numeric(item.get("duration_ms"), errors="coerce")
    actual = pd.to_numeric(entry.get("duration"), errors="coerce")
    expected = expected_ms / 1000 if pd.notna(expected_ms) and expected_ms > 0 else np.nan
    close_duration = (pd.notna(expected) and pd.notna(actual)
                      and abs(actual - expected) <= max(3., expected * .02))
    song_title = candidate_song_title(entry, artists, featured)
    candidate_core, _ = title_parts(song_title)
    if not core_title or f" {core_title} " not in f" {candidate_core} ":
        return None, "title"
    if requested_remix:
        # Featured-credit parsing can put a later remix suffix into the credit
        # text. Require that suffix independently so duration relaxation cannot
        # substitute a different named remix.
        labels = re.split(r"[()\[\]]|\s+[-–—]\s+", text_value(item.get("track_name")))
        for label in labels:
            if remix_label(label) and f" {canonical_title(label)} " not in f" {candidate} ":
                return None, "remix_version"
        # A matching upload title must not conceal contradictory named-remix
        # metadata. Bare base-song track names do not themselves contradict it.
        for key in ("track", "alt_title", "version", "remixed_by", "remix_artist"):
            value = entry.get(key)
            values = value if isinstance(value, (list, tuple)) else [value]
            for version in values:
                if not isinstance(version, str) or not version.strip():
                    continue
                if key in ("remixed_by", "remix_artist"):
                    if not credit_present(version, " " + title + " " + " ".join(artists) + " "):
                        return None, "remix_version"
                    continue
                if not remix_label(version):
                    continue
                version = canonical_title(version)
                if key == "version":
                    if f" {version} " not in f" {title} ":
                        return None, "remix_version"
                elif (f" {core_title} " not in f" {version} "
                      or any(remix_label(label) and f" {canonical_title(label)} " not in f" {version} "
                             for label in labels)):
                    return None, "remix_version"
    credit_fields = []
    for key in ("uploader", "channel", "artist", "artists", "creator", "creators"):
        values = entry.get(key)
        values = values if isinstance(values, (list, tuple)) else [values]
        credit_fields.extend(normalized(re.sub(r"(?i)vevo$", "", value))
                             for value in values if isinstance(value, str))
    credits = " " + " ".join([candidate] + credit_fields) + " "
    artist_missing = (not all(credit_present(artist, credits) for artist in artists)
                      or bool(featured and not credit_present(featured, credits)))
    primary_artist = re.split(r"[,;]", artists[0], maxsplit=1)[0].strip()
    primary_present = credit_present(primary_artist, credits)
    if artist_missing and not (primary_present and (allow_missing_artists or close_duration)):
        return None, "artist"
    # A wider search must not accept a different title merely containing the
    # requested phrase (e.g. "Another Example Song" for "Example Song").
    if f" {core_title} " not in f" {candidate_core} ":
        return None, "title"
    remaining = f" {candidate_core} ".replace(f" {core_title} ", " ")
    for artist in artists + ([featured] if featured else []):
        for name in [artist] + re.split(r"[,;]", artist):
            remaining = remaining.replace(f" {normalized(name.strip())} ", " ")
    # Remove complete presentation phrases only after removing the requested
    # title, so these words in a real song title retain their meaning.
    remaining = re.sub(r"\b(?:audio only|full audio)\b", "audio", remaining)
    remaining = re.sub(r"\boriginal mix\b", " ", remaining)
    remaining = re.sub(r"\b(?:lyrics?|letras?|paroles) by\s+.*$", " ", remaining)
    noise = {"official", "audio", "video", "music", "lyric", "lyrics", "visualizer",
             "visualiser", "hd", "hq", "4k", "1080p", "officialaudio", "officialvideo",
             "unofficial", "reupload", "reuploaded", "letra", "letras", "paroles", "fanmade"}
    noise.update({'lyrical', 'm', 'v'})
    requested_labels = set(title.split()) - set(core_title.split())
    if set(remaining.split()) - noise - requested_labels:
        return None, "title"
    versions = {"live", "cover", "acoustic", "karaoke", "instrumental",
                "slowed", "sped", "nightcore", "demo", "edit", "reverb", "stripped"}
    # Version terms belonging to the artist's name are not recording labels.
    requested = set(normalized(title + " " + " ".join(artists)).split())
    if (words & versions) - requested:
        return None, "version"
    if (set(title.split()) & versions) - words:
        return None, "version"
    unverified_remaster = False
    if "remaster" in title.split():
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", title))
        candidate_years = set(re.findall(r"\b(?:19|20)\d{2}\b", candidate))
        if "remaster" in words and years and candidate_years and not years.issubset(candidate_years):
            return None, "remaster_year"
        unverified_remaster = "remaster" not in words or not years.issubset(candidate_years)
        if unverified_remaster and not close_duration:
            return None, "remaster_unverified_duration"
    score = 1.0
    if pd.notna(expected_ms) and expected_ms > 0 and pd.notna(actual):
        expected = expected_ms / 1000
        error = abs(actual - expected)
        if error > max(12., expected * .12):
            return None, "duration"
        score += 1 - error / max(12., expected * .12)
    if "topic" in normalized(text_value(entry.get("channel") or entry.get("uploader"))):
        score += .25
    if artist_missing and allow_missing_artists:
        # Provisional search candidate only: never publish/download on this score.
        return score * .5, "needs_artist_metadata"
    notes = []
    if artist_missing:
        notes.append("primary_artist_duration")
        score *= .8
    if unverified_remaster:
        notes.append("unverified_remaster")
        score *= .8
    return score, "matched_" + "_".join(notes) if notes else "matched"


def match_score(item: dict, entry: dict) -> float | None:
    return match_result(item, entry)[0]


def acquisition_error_code(exc: Exception) -> str:
    message = str(exc).casefold()
    if message.startswith("resolved_recording_mismatch:"):
        return message
    if re.search(r"(?:not available|blocked|not made.*available).*\b(?:country|region)\b", message):
        return "geo_restricted"
    if "page needs to be reloaded" in message:
        return "session_reload_required"
    if "signature solving failed" in message or "challenge solving failed" in message:
        return "js_challenge_failed"
    for status in (429, 403):
        if re.search(rf"\b(?:http(?: error)?|status(?: code)?)\s*[:=]?\s*{status}\b", message):
            return f"http_{status}"
    for text, code in (("sign in", "authentication_required"), ("not available", "unavailable"),
                       ("private video", "unavailable"), ("timed out", "network_timeout")):
        if text in message:
            return code
    return type(exc).__name__


def audio_source(audio: Path) -> str:
    sidecar = audio.with_suffix(".source.json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))["url"]
        except (OSError, ValueError, KeyError, TypeError):
            LOG.warning("Cannot read source sidecar for %s", audio.name)
    return str(audio.resolve())


def acquire_audio(item: dict, audio_dir: Path, *, cookiejar=None) -> tuple[Path | None, str | None]:
    """Search conservatively, with a broader fallback query and isolated downloads."""
    tid = str(item["track_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tid):
        raise ValueError("invalid_track_id")
    title = text_value(item.get("track_name"))
    artists = artist_names(item.get("spotify_artist_names")) or artist_names(item.get("artist_name"))
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
    options = {**common, "format": "bestaudio/best", "overwrites": False,
               "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}
    core, _ = title_parts(title)
    requested_words = set(canonical_title(title).split())
    recording_labels = sorted(recording_languages(title)) + [
        label for label in ('live', 'stripped', 'acoustic', 'instrumental', 'karaoke', 'slowed', 'sped')
        if label in requested_words]
    versions = ''.join(' ' + label for label in recording_labels)
    versions += " remaster" if "remaster" in requested_words else ""
    queries = [f"ytsearch5:{' '.join(artists)} {title} official audio",
               f"ytsearch10:{' '.join(artists)} {core}{versions}"]
    primary_artist = re.split(r"[,;]", artists[0], maxsplit=1)[0].strip()
    queries.append(f"ytsearch15:{primary_artist} {core}{versions}")
    lyric_versions = ''.join(' ' + label for label in recording_labels)
    queries.append(f"ytsearch15:{primary_artist} {core}{lyric_versions} lyrics")
    errors, search_errors, attempted = [], [], set()
    rejections = Counter()
    rejection_examples, rejected_urls = [], set()

    def reject(entry, reason):
        url = text_value(entry.get("webpage_url") or entry.get("url"))
        identity = (url or text_value(entry.get("title")), reason)
        if identity in rejected_urls:
            return
        rejected_urls.add(identity)
        rejections[reason] += 1
        if len(rejection_examples) < 5:
            # Do not log signed URLs, arbitrary query parameters or userinfo.
            parsed = urlsplit(url)
            query = ""
            if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
                video = parse_qs(parsed.query).get("v", [])
                query = urlencode({"v": video[0]}) if video else ""
            safe_url = urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, query, ""))
            rejection_examples.append((reason, text_value(entry.get("title"))[:200],
                                       text_value(entry.get("uploader") or entry.get("channel"))[:100],
                                       safe_url[:300]))
    saw_results = False
    search_time = download_time = conversion_time = 0.
    for query in queries:
        tick = perf_counter()
        try:
            with youtube_session({**common, "extract_flat": "in_playlist"}, cookiejar) as search:
                info = search.extract_info(query, download=False)
        except yt_dlp.utils.DownloadError as exc:
            code = acquisition_error_code(exc)
            search_errors.append(code)
            if code in DEFERRED_ACQUISITION_CODES:
                raise RuntimeError("search_failed: " + code) from exc
            continue
        finally:
            search_time += perf_counter() - tick
        candidates = []
        for entry in (info or {}).get("entries", []) or []:
            if not entry:
                continue
            saw_results = True
            score, reason = match_result(item, entry, allow_missing_artists=True)
            url = entry.get("webpage_url") or entry.get("url")
            if score is not None and url and url not in attempted:
                candidates.append((score, url, entry, reason))
            elif score is None:
                reject(entry, reason)
        for _, url, entry, match_reason in sorted(candidates, key=lambda pair: pair[0], reverse=True):
            if url in attempted:
                continue
            attempted.add(url)
            audio_dir.mkdir(parents=True, exist_ok=True)
            # Never resume one recording's partial download as another recording.
            # Retain failed staging files for inspection rather than deleting them.
            tick = perf_counter()
            conversion_started, converted = None, 0.

            def postprocess_hook(event):
                nonlocal conversion_started, converted
                if event.get("postprocessor") not in ("ExtractAudio", "FFmpegExtractAudio"):
                    return
                if event.get("status") == "started":
                    conversion_started = perf_counter()
                elif event.get("status") == "finished" and conversion_started is not None:
                    converted += perf_counter() - conversion_started
                    conversion_started = None

            try:
                if match_reason == "needs_artist_metadata":
                    with youtube_session(options, cookiejar) as resolver:
                        resolved = resolver.extract_info(url, download=False)
                    entry = {**entry, **resolved}
                    score, reason = match_result(item, entry)
                    if score is None:
                        reject(entry, reason)
                        continue
                staging = Path(tempfile.mkdtemp(prefix=f".{tid}_", dir=audio_dir))
                with youtube_session({**options, "outtmpl": str(staging / f"{tid}.%(ext)s"),
                                      "postprocessor_hooks": [postprocess_hook]}, cookiejar) as downloader:
                    result = downloader.extract_info(url, download=True)
                recording = {**entry, **result}
                resolved_score, reason = match_result(item, recording)
                if resolved_score is None:
                    raise RuntimeError("resolved_recording_mismatch: " + reason)
                audio = staging / f"{tid}.wav"
                if not audio.is_file() or audio.stat().st_size == 0:
                    raise RuntimeError("download produced no converted WAV")
                source = result.get("webpage_url") or url
                sidecar = audio.with_suffix(".source.json")
                identity_fields = ("title", "track", "alt_title", "version", "tags", "remixed_by",
                                   "remix_artist", "artist", "artists", "creator", "creators",
                                   "uploader", "channel", "duration", "is_live")
                identity = {key: recording[key] for key in identity_fields if key in recording}
                sidecar.write_text(json.dumps({**identity, "url": source,
                    "matching": "structured_recording_identity_v6",
                    "match_evidence": reason}, ensure_ascii=False), encoding="utf-8")
                destination = audio_dir / f"{tid}.wav"
                destination_sidecar = destination.with_suffix(".source.json")
                if destination.exists() or destination_sidecar.exists():
                    previous = Path(tempfile.mkdtemp(prefix=f".{tid}_previous_", dir=audio_dir))
                    for existing in (destination, destination_sidecar):
                        if existing.exists():
                            shutil.copy2(existing, previous / existing.name)
                    LOG.info("Preserved previous audio/provenance for %s at %s", tid, previous)
                # Never pair old (possibly remix) audio with new clean provenance
                # if replacing the audio itself is denied by Windows.
                os.replace(audio, destination)
                os.replace(sidecar, destination_sidecar)
                return destination, source
            except Exception as exc:
                code = acquisition_error_code(exc)
                errors.append(code)
                LOG.warning("Download candidate failed for %s (%s)", tid, code)
                if code in DEFERRED_ACQUISITION_CODES:
                    raise RuntimeError("download_failed: " + code) from exc
            finally:
                if conversion_started is not None:
                    converted += perf_counter() - conversion_started
                conversion_time += converted
                download_time += max(0., perf_counter() - tick - converted)
                LOG.info("Acquisition %s: search %.2fs, resolve/download %.2fs, conversion %.2fs",
                         tid, search_time, download_time, conversion_time)
    LOG.warning("No acquired audio for %s; rejected candidates: %s", tid, dict(rejections))
    for reason, candidate_title, uploader, url in rejection_examples:
        LOG.warning("Rejected candidate for %s: reason=%s title=%r uploader=%r url=%s",
                    tid, reason, candidate_title, uploader, url)
    if errors:
        raise RuntimeError("download_failed: " + ", ".join(sorted(set(errors))))
    if search_errors:
        raise RuntimeError("search_failed: " + ", ".join(sorted(set(search_errors))))
    raise RuntimeError("no_matching_recording" if saw_results else "no_search_results")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Serialize once, retry transient sharing errors, retain complete failed saves.

    This does not permit concurrent writers. If the destination changes while
    preparing or retrying a save, abort rather than overwrite external updates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    original_signature = file_signature(path) if path.exists() else None
    fd, temp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".csv", dir=path.parent)
    prepared = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        prepared = True
        for attempt in range(len(CSV_REPLACE_DELAYS) + 1):
            current_signature = file_signature(path) if path.exists() else None
            if current_signature != original_signature:
                raise RuntimeError(f"CSV changed while preparing/retrying save: {path}. "
                                   f"Prepared snapshot retained at {temp}; external updates were not overwritten.")
            try:
                os.replace(temp, path)
                return
            except PermissionError as exc:
                if attempt == len(CSV_REPLACE_DELAYS):
                    raise CsvSaveBlocked(path, Path(temp)) from exc
                delay = CSV_REPLACE_DELAYS[attempt]
                LOG.warning("Access denied replacing %s; retry %d/%d in %.2fs. "
                            "The prepared CSV is retained while waiting.",
                            path.name, attempt + 1, len(CSV_REPLACE_DELAYS), delay)
                sleep(delay)
    finally:
        # A complete snapshot is useful for recovery after failed replacement;
        # incomplete serialization artifacts must not be mistaken for snapshots.
        if not prepared and os.path.exists(temp):
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


def rhythm(y: np.ndarray, sr: int, onset_envelope: np.ndarray | None = None) -> tuple[float, float]:
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_envelope)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(frames) < 4 or not np.isfinite(tempo) or tempo <= 0:
        return np.nan, np.nan
    times = librosa.frames_to_time(frames, sr=sr)
    intervals = np.diff(times)
    if len(intervals) == 0 or np.mean(intervals) <= 0:
        return tempo, np.nan
    regularity = 1 / (1 + np.std(intervals) / np.mean(intervals))
    return tempo, float(np.clip(regularity, 0, 1))


def extract_audio_features(audio_path: Path, seconds: float | None = ANALYSIS_SECONDS) -> dict[str, float | str] | None:
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

    @cache
    def log_mel():
        mel = librosa.feature.melspectrogram(S=spectrum() ** 2, sr=sr)
        return librosa.power_to_db(mel)

    @cache
    def onset_envelope(aggregation):
        # beat_track(y=...) uses median aggregation; the standalone onset
        # descriptor uses mean. Share their transform, not their aggregation.
        aggregate = np.median if aggregation == "median" else np.mean
        return librosa.onset.onset_strength(S=log_mel(), sr=sr, aggregate=aggregate)

    @cache
    def spectrum():
        return np.abs(librosa.stft(y))

    analyses = {
        "rhythm": lambda: rhythm(y, sr, onset_envelope=onset_envelope("median")),
        "tonality": lambda: key_mode(y, sr),
        "onset": lambda: float(np.mean(onset_envelope("mean"))),
        "centroid": lambda: float(np.mean(librosa.feature.spectral_centroid(S=spectrum(), sr=sr))),
        "contrast": lambda: float(np.mean(librosa.feature.spectral_contrast(S=spectrum(), sr=sr))),
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
    result["_analysis_seconds"] = len(y) / sr
    result["_extractor_version"] = EXTRACTOR_VERSION
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


def save_batch(results: list[dict], master_path: Path,
               master: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Fill missing values; append new rows together and return the saved snapshot.

    Supplying the coordinator's master avoids rereading the whole CSV at every
    checkpoint. The two-argument form remains available for standalone callers.
    """
    if not results:
        return master
    if master is None:
        master = read_master(master_path)
    new_rows = {}
    for record in results:
        track_id = record["track_id"]
        if track_id not in master.index and track_id not in new_rows:
            operational = {"track_id", "attempts", "last_error", "audio_file", "audio_source"}
            metadata = {key: value for key, value in record.items()
                        if key not in operational and key not in FEATURE_COLUMNS
                        and key != "id" and not key.startswith("_")}
            metadata.setdefault("name", record.get("track_name"))
            metadata.setdefault("artists", record.get("artist_name"))
            new_rows[track_id] = metadata
    if new_rows:
        incoming = pd.DataFrame.from_dict(new_rows, orient="index")
        incoming.index.name = "id"
        for col in incoming.columns.difference(master.columns):
            master[col] = pd.Series(pd.NA, index=master.index, dtype="string")
        incoming = incoming.reindex(columns=master.columns)
        # Match schema once on the small incoming frame. This also preserves
        # extension dtypes when a row lacks optional name/artist metadata.
        for col in master.columns:
            incoming[col] = incoming[col].astype(master[col].dtype)
        master = pd.concat([master, incoming])
    # Backlog enrichment must survive successful extraction even when the track
    # already exists in master; fill only gaps and preserve all existing text.
    metadata_fields = ('duration_ms', 'isrc', 'spotify_track_name', 'spotify_artist_names',
                       'spotify_resolved_id', 'spotify_metadata_source', 'spotify_metadata_checked_at')
    for record in results:
        for col in metadata_fields:
            value = record.get(col)
            if (value is None or pd.isna(value) or not str(value).strip()
                    or (col == 'duration_ms' and str(value).strip().casefold() == 'nan')):
                continue
            if col not in master:
                master[col] = pd.Series(pd.NA, index=master.index, dtype='string')
            previous = master.at[record['track_id'], col]
            if (pd.isna(previous) or not str(previous).strip()
                    or (col == 'duration_ms' and str(previous).strip().casefold() == 'nan')):
                master.at[record['track_id'], col] = str(value)
    # Allocate/cast columns once per checkpoint, not for every track.
    active = [col for col in FEATURE_COLUMNS if any(pd.notna(row.get(col, np.nan)) for row in results)]
    for col in active:
        if col not in master:
            master[col] = np.nan
        for suffix in ("_source", "_audio_source"):
            name = col + suffix
            if name not in master:
                master[name] = pd.Series(pd.NA, index=master.index, dtype="string")
            elif not isinstance(master[name].dtype, pd.StringDtype):
                master[name] = master[name].astype("string")
        duration = col + "_analysis_seconds"
        if duration not in master:
            master[duration] = np.nan
        else:
            master[duration] = master[duration].astype(object)
    for record in results:
        track_id = record["track_id"]
        for col in active:
            val = record.get(col, np.nan)
            if pd.isna(val):
                continue
            if pd.isna(master.at[track_id, col]):
                master.at[track_id, col] = val
                master.at[track_id, col + "_source"] = record.get("_extractor_version", EXTRACTOR_VERSION)
                master.at[track_id, col + "_audio_source"] = record.get("audio_source", record.get("audio_file", pd.NA))
                master.at[track_id, col + "_analysis_seconds"] = record.get("_analysis_seconds", np.nan)
    atomic_csv(master.reset_index(), master_path)
    return master


def prepare_track(item: dict, audio_dir: Path, local_only: bool,
                  dry_run: bool, *, cookiejar=None) -> tuple[dict | None, str | None, float]:
    """Acquisition stage: return an audio path, never a decoded signal or CSV."""
    tid = str(item["track_id"])
    started = perf_counter()
    LOG.info("Processing %s", tid)
    try:
        audio = find_local_audio(tid, audio_dir)
        if audio is not None:
            sidecar = audio.with_suffix(".source.json")
            if sidecar.is_file():
                try:
                    cached_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    if isinstance(cached_metadata, dict):
                        requested_remix = remix_label(text_value(item.get("track_name")))
                        incompatible = (match_result(item, cached_metadata)[0] is None if requested_remix
                                        else remix_recording(cached_metadata))
                        if incompatible:
                            LOG.warning("Ignoring cached audio with incompatible remix metadata for %s; "
                                        "original file retained", tid)
                            audio = None
                except (OSError, ValueError, TypeError):
                    pass  # audio_source reports malformed provenance as before.
        source = audio_source(audio) if audio else None
        if audio is None and not local_only and not dry_run:
            auth = {"cookiejar": cookiejar} if cookiejar is not None else {}
            audio, source = acquire_audio(item, audio_dir, **auth)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__
        LOG.warning("Audio acquisition failed for %s: %s", tid, reason[:200])
        return None, reason[:200], perf_counter() - started
    acquired = perf_counter() - started
    if audio is None:
        return None, "audio_not_found", acquired
    return {"item": item, "audio": audio, "source": source, "acquired": acquired}, None, acquired


def analyze_track(prepared: dict, analysis_seconds: float | None) -> tuple[dict | None, str | None, float]:
    """Analysis stage: decode only when an analysis worker is available."""
    item, audio, source = prepared["item"], prepared["audio"], prepared["source"]
    tid = str(item["track_id"])
    started = perf_counter()
    try:
        features = extract_audio_features(audio, seconds=analysis_seconds)
    except Exception as exc:
        LOG.warning("Analysis failed for %s (%s)", tid, type(exc).__name__)
        return None, "analysis_failed: " + type(exc).__name__, perf_counter() - started
    analyzed = perf_counter() - started
    LOG.info("Finished %s: acquisition %.2fs, analysis %.2fs", tid, prepared["acquired"], analyzed)
    if features is None:
        return None, "invalid_audio", analyzed
    return {**item, **features, "audio_file": str(audio), "audio_source": source}, None, analyzed


def process_track(item: dict, audio_dir: Path, local_only: bool,
                  dry_run: bool, analysis_seconds: float | None) -> tuple[dict | None, str | None, float, float]:
    """Compatibility entry point for processing a single track synchronously."""
    prepared, error, acquired = prepare_track(item, audio_dir, local_only, dry_run)
    if error:
        return None, error, acquired, 0.
    record, error, analyzed = analyze_track(prepared, analysis_seconds)
    return record, error, acquired, analyzed


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def process(backlog_path: Path = BACKLOG_PATH, master_path: Path = MASTER_PATH,
            audio_dir: Path = AUDIO_DIR, batch_size: int = 100, max_attempts: int = 3,
            dry_run: bool = False, local_only: bool = False,
            limit: int | None = None, retry_failed: bool = False,
            workers: int | None = None, analysis_seconds: float | None = ANALYSIS_SECONDS,
            acquisition_workers: int | None = None, analysis_workers: int | None = None,
            queue_size: int | None = None, *, cookies: Path | None = None,
            cookies_from_browser: str | None = None) -> dict[str, int]:
    """Two independently sized stages, bounded ready paths, one CSV coordinator."""
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master database: {master_path}")
    if not backlog_path.exists():
        raise FileNotFoundError(f"Missing backlog: {backlog_path}")
    if batch_size <= 0 or max_attempts <= 0 or (workers is not None and workers <= 0):
        raise ValueError("batch_size, max_attempts and workers must be positive")
    acquisition_workers = (workers or 8) if acquisition_workers is None else acquisition_workers
    analysis_workers = (workers or 4) if analysis_workers is None else analysis_workers
    queue_size = max(acquisition_workers, analysis_workers * 2) if queue_size is None else queue_size
    if acquisition_workers <= 0 or analysis_workers <= 0 or queue_size <= 0:
        raise ValueError("acquisition_workers, analysis_workers and queue_size must be positive")
    if analysis_seconds is not None and (not np.isfinite(analysis_seconds) or analysis_seconds <= 0):
        raise ValueError("analysis_seconds must be positive, or None for the full song")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    cookiejar = (load_youtube_cookies(cookies, cookies_from_browser)
                 if not dry_run and not local_only else None)
    acquisition_auth = {"cookiejar": cookiejar} if cookiejar is not None else {}
    started = perf_counter()
    signatures = {path: file_signature(path) for path in (master_path, backlog_path)}
    master = read_master(master_path)
    backlog = read_backlog(backlog_path)
    eligible = (backlog if retry_failed else backlog[(backlog["attempts"] < max_attempts)
                                                    | (backlog["last_error"] == "excluded_remix")]).copy()
    if retry_failed:
        eligible = eligible.sort_values("attempts", kind="stable")
    if limit is not None:
        eligible = eligible.head(limit)
    stats = {"eligible": len(eligible), "saved": 0, "failed": 0}
    LOG.info("Loaded catalogs in %.2fs; eligible %d, acquisition workers %d, analysis workers %d, queue %d, checkpoint %d",
             perf_counter() - started, len(eligible), acquisition_workers, analysis_workers, queue_size, batch_size)
    successful, failures, deferred = [], {}, {}
    acquire_time = analyze_time = checkpoint_time = 0.
    error_counts = Counter()
    checkpoint_started = False
    service_challenges = deferred_total = 0
    acquisition_blocked = False

    def checkpoint():
        nonlocal master, backlog, checkpoint_time, checkpoint_started
        if not successful and not failures and not deferred:
            return
        checkpoint_started = True
        tick = perf_counter()
        if dry_run:
            LOG.info("DRY RUN: would save %d, mark %d failures, defer %d",
                     len(successful), len(failures), len(deferred))
        else:
            for path, signature in signatures.items():
                if file_signature(path) != signature:
                    raise RuntimeError(f"Catalog changed during extraction: {path.name}; stop other writers")
            if successful:
                master = save_batch(successful, master_path, master=master)
                signatures[master_path] = file_signature(master_path)
            remaining = backlog[~backlog["track_id"].isin({row["track_id"] for row in successful})].copy()
            for tid, message in failures.items():
                mask = remaining["track_id"] == tid
                remaining.loc[mask, "attempts"] += 1
                remaining.loc[mask, "last_error"] = message
            # Service challenges are not evidence that a track cannot be found.
            # Record the reason without spending its retry allowance.
            for tid, message in deferred.items():
                remaining.loc[remaining["track_id"] == tid, "last_error"] = message
            atomic_csv(remaining, backlog_path)
            backlog = remaining
            signatures[backlog_path] = file_signature(backlog_path)
        stats["saved"] += len(successful)
        stats["failed"] += len(failures)
        elapsed = perf_counter() - tick
        checkpoint_time += elapsed
        LOG.info("Checkpoint: %d saved, %d failed (%.2fs)", stats["saved"], stats["failed"], elapsed)
        successful.clear()
        failures.clear()
        deferred.clear()
        checkpoint_started = False

    def items():
        for item in eligible.to_dict("records"):
            tid = str(item["track_id"])
            if tid in master.index:
                metadata = master.loc[tid]
                for target, names in {"track_name": ("name", "track_name"),
                                      "artist_name": ("artists", "artist_name"),
                                      "duration_ms": ("duration_ms",)}.items():
                    if not text_value(item.get(target)):
                        item[target] = next((metadata[name] for name in names
                                             if name in metadata and text_value(metadata[name])), None)
            yield item

    iterator = iter(items())
    acquisition_pool = ThreadPoolExecutor(max_workers=acquisition_workers, thread_name_prefix="acquire")
    analysis_pool = ThreadPoolExecutor(max_workers=analysis_workers, thread_name_prefix="analyze")
    acquiring, analyzing, ready = {}, {}, deque()
    exhausted = False

    try:
        while True:
            while ready and len(analyzing) < analysis_workers:
                prepared = ready.popleft()
                future = analysis_pool.submit(analyze_track, prepared, analysis_seconds)
                analyzing[future] = str(prepared["item"]["track_id"])
            # Reserve a queue slot for every acquisition in flight, so finished
            # downloads can always be collected without blocking either stage.
            while (not exhausted and not acquisition_blocked and len(acquiring) < acquisition_workers
                   and len(ready) + len(acquiring) < queue_size):
                item = next(iterator, None)
                if item is None:
                    exhausted = True
                    break
                future = acquisition_pool.submit(prepare_track, item, audio_dir, local_only, dry_run,
                                                 **acquisition_auth)
                acquiring[future] = str(item["track_id"])
            pending = set(acquiring) | set(analyzing)
            if not pending:
                if not ready:
                    break
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                if future in acquiring:
                    tid = acquiring.pop(future)
                    prepared, error, acquired = future.result()
                    acquire_time += acquired
                    if not error:
                        ready.append(prepared)
                        continue
                    service_code = next((code for code in DEFERRED_ACQUISITION_CODES if code in error), None)
                    if service_code:
                        deferred[tid] = error
                        deferred_total += 1
                        error_counts[error] += 1
                        service_challenges += int(service_code != "http_429")
                        if not acquisition_blocked and (service_code == "http_429" or service_challenges >= 3):
                            acquisition_blocked = True
                            LOG.warning("Acquisition paused for this run after service challenges. "
                                        "Finishing in-flight work; blocked and unstarted tracks keep "
                                        "their attempt allowance. Check YouTube access before retrying.")
                        if len(successful) + len(failures) + len(deferred) >= batch_size:
                            checkpoint()
                        continue
                    record = None
                else:
                    tid = analyzing.pop(future)
                    record, error, analyzed = future.result()
                    analyze_time += analyzed
                if error:
                    failures[tid] = error
                    error_counts[error] += 1
                else:
                    successful.append(record)
                if len(successful) + len(failures) + len(deferred) >= batch_size:
                    checkpoint()
        checkpoint()
    except KeyboardInterrupt:
        # Completed results already collected by the coordinator are safe to save;
        # in-flight tracks retain their backlog entries and their audio cache.
        if not checkpoint_started:
            checkpoint()
        LOG.warning("Interrupted; completed checkpoints are retained, remaining backlog entries can be retried")
        raise
    finally:
        for future in (*acquiring, *analyzing):
            future.cancel()
        acquisition_pool.shutdown(wait=True, cancel_futures=True)
        analysis_pool.shutdown(wait=True, cancel_futures=True)
    LOG.info("Timing: wall %.2fs, acquisition worker-time %.2fs, analysis worker-time %.2fs, checkpoints %.2fs",
             perf_counter() - started, acquire_time, analyze_time, checkpoint_time)
    if error_counts:
        LOG.info("Failure summary for this run: %s", dict(error_counts))
    if deferred_total:
        stats["deferred"] = (stats["eligible"] - stats["saved"] - stats["failed"]
                             if acquisition_blocked else deferred_total)
        LOG.warning("Deferred %d tracks without consuming attempts (%d challenged requests). "
                    "Deferred includes unstarted tracks when acquisition was paused.",
                    stats["deferred"], deferred_total)
    return stats


def parse_analysis_seconds(value: str) -> float | None:
    if value.casefold() == "full":
        return None
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a positive number of seconds or 'full'") from exc
    if not np.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("Use a positive number of seconds or 'full'")
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", type=Path, default=BACKLOG_PATH)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--batch-size", type=int, default=100, help="Completed tracks per CSV checkpoint (default: 100)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Disable audio search/download")
    parser.add_argument("--limit", type=int, help="Maximum number of tracks to process this run")
    parser.add_argument("--retry-failed", action="store_true", help="Include tracks at the attempt limit")
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--cookies", type=Path, metavar="FILE",
                                help="Read a Netscape-format cookie file without rewriting it")
    authentication.add_argument("--cookies-from-browser", metavar="BROWSER[:PROFILE]",
                                help="Load browser cookies once, e.g. chrome, edge, firefox or chrome:Default")
    parser.add_argument("--workers", type=int, help="Compatibility shorthand: set both pool sizes")
    parser.add_argument("--acquisition-workers", type=int, help="Concurrent searches/downloads (default: 8)")
    parser.add_argument("--analysis-workers", type=int, help="Concurrent audio analyses (default: 4)")
    parser.add_argument("--queue-size", type=int, help="Bound on ready paths plus acquisitions in flight (default: max(acquisition workers, twice analysis workers))")
    parser.add_argument("--analysis-seconds", type=parse_analysis_seconds, default=ANALYSIS_SECONDS,
                        help="Seconds of audio to analyze, or 'full' (default: 60)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        stats = process(args.backlog, args.master, args.audio_dir, args.batch_size, args.max_attempts,
                        args.dry_run, args.local_only, args.limit, args.retry_failed,
                        args.workers, args.analysis_seconds, args.acquisition_workers,
                        args.analysis_workers, args.queue_size, cookies=args.cookies,
                        cookies_from_browser=args.cookies_from_browser)
    except (CsvSaveBlocked, CookieConfigurationError) as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from None
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
