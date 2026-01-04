import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import shutil
import traceback
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

from faster_whisper import WhisperModel

# -----------------------------
# Audio conversion
# -----------------------------
def to_wav_16k_mono(input_path: str, wav_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        wav_path,
    ]

    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        # include last ~20 lines to avoid huge dumps
        tail = "\n".join((p.stderr or "").splitlines()[-20:])
        raise subprocess.CalledProcessError(p.returncode, cmd, output=None, stderr=tail)

# -----------------------------
# SRT formatting
# -----------------------------
def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@dataclass
class SubtitleBlock:
    start: float
    end: float
    lines: List[str]  # 1–2 lines

# -----------------------------
# Logging helpers
# -----------------------------
def log(msg: str, *, quiet: bool = False):
    if not quiet:
        print(msg, flush=True)

def progress_line(msg: str, *, enabled: bool, quiet: bool):
    if quiet or not enabled:
        return
    # one-line, overwritable progress indicator
    sys.stdout.write("\r" + msg[:120].ljust(120))
    sys.stdout.flush()

def progress_done(*, enabled: bool, quiet: bool):
    if quiet or not enabled:
        return
    sys.stdout.write("\n")
    sys.stdout.flush()

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def probe_duration_seconds(path: str) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return None

# -----------------------------
# Validation helpers
# -----------------------------
def die(msg: str, code: int = 1, *, quiet: bool = False) -> int:
    # Always show errors, even in --quiet
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    return code

def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def preflight_checks(input_path: Path, output_path: Path, overwrite: bool) -> Tuple[bool, str]:
    if not input_path.exists():
        return False, f"Input file not found: {input_path}"
    if input_path.is_dir():
        return False, f"Input path is a directory, expected a media file: {input_path}"
    if output_path.exists() and not overwrite:
        return False, f"Output already exists: {output_path} (use --overwrite)"
    if output_path.exists() and output_path.is_dir():
        return False, f"Output path is a directory, expected a file: {output_path}"
    return True, ""

def init_whisper_model_with_fallback(
    model_name: str,
    device: str,
    quiet: bool,
):
    """
    If device == 'cuda', try CUDA first, fall back to CPU on failure.
    Returns: (model, device_used, compute_type_used)
    """
    if device != "cuda":
        compute_type = "int8" if device == "cpu" else "float16"
        return WhisperModel(model_name, device=device, compute_type=compute_type), device, compute_type

    # Try CUDA
    try:
        compute_type = "float16"
        m = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        log("   CUDA available: using device=cuda compute_type=float16", quiet=quiet)
        return m, "cuda", compute_type
    except Exception as e:
        # Fall back to CPU
        log(f"   CUDA init failed; falling back to CPU. Reason: {e}", quiet=quiet)
        compute_type = "int8"
        m = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        log("   Using device=cpu compute_type=int8", quiet=quiet)
        return m, "cpu", compute_type


# -----------------------------
# Chunking rules
# -----------------------------
def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def wrap_text_lines(text: str, max_chars_per_line: int) -> List[str]:
    """
    Greedy word wrap into as many lines as needed. Never drops words.
    """
    text = normalize_spaces(text)
    if not text:
        return []

    words = text.split(" ")
    lines: List[str] = []
    cur: List[str] = []

    def cur_len() -> int:
        return len(" ".join(cur)) if cur else 0

    for w in words:
        if not cur:
            cur = [w]
            continue

        if cur_len() + 1 + len(w) <= max_chars_per_line:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]

    if cur:
        lines.append(" ".join(cur))

    return lines

def block_fits(text: str, max_chars_per_line: int, max_lines: int) -> bool:
    return len(wrap_text_lines(text, max_chars_per_line)) <= max_lines

def wrap_fallback_blocks(text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
    """
    Final fallback: wrap into lines then group lines into blocks of max_lines.
    Never drops words.
    """
    lines = wrap_text_lines(text, max_chars_per_line)
    blocks: List[str] = []
    for i in range(0, len(lines), max_lines):
        blocks.append(" ".join(lines[i:i + max_lines]))
    return blocks

def split_on_delims(text: str, delims: str) -> List[str]:
    """
    Split using delimiter characters as boundaries, keeping delimiter attached to
    the preceding chunk. Uses match spans; no punctuation duplication bugs.
    Only splits when delimiter is followed by whitespace.
    """
    text = normalize_spaces(text)
    if not text:
        return []

    # Match minimal up to one-or-more delimiters, require whitespace boundary.
    pattern = re.compile(rf".+?(?:[{re.escape(delims)}]+)(?=\s+)")
    parts: List[str] = []
    last_end = 0

    for m in pattern.finditer(text):
        end = m.end()
        chunk = text[last_end:end].strip()
        if chunk:
            parts.append(chunk)

        # advance past boundary whitespace
        ws = re.match(r"\s+", text[end:])
        last_end = end + (ws.end() if ws else 0)

    rem = text[last_end:].strip()
    if rem:
        parts.append(rem)

    # filter out "dangling punctuation-only" tokens (rare)
    parts = [p for p in parts if re.search(r"\w", p)]
    return parts

def split_text_into_blocks(
    text: str,
    max_chars_per_line: int,
    max_lines: int,
    allow_commas: bool = True,
    allow_medium: bool = True,
    prefer_punct_splits: bool = False,
) -> List[str]:
    """
    Prefer splitting on strong punctuation first, then medium, then weak,
    and only then fall back to word-wrapping.

    Guarantees: no words dropped; each returned block will fit into <= max_lines
    when wrapped at max_chars_per_line (or will be wrapped-fallback produced).
    """
    text = normalize_spaces(text)
    if not text:
        return []

    # Define punctuation tiers (descending strength)
    tiers: List[str] = [".?!"]
    if allow_medium:
        tiers.append(";:")
    if allow_commas:
        tiers.append(",")

    def refine_chunk(chunk: str, tier_index: int) -> List[str]:
        chunk = normalize_spaces(chunk)
        if not chunk:
            return []

        fits = block_fits(chunk, max_chars_per_line, max_lines)

        # If it fits, keep as-is unless we're explicitly preferring punctuation splits
        # at the strongest tier (tier_index == 0).
        if fits and not (prefer_punct_splits and tier_index == 0):
            return [chunk]

        # If we've exhausted tiers, fallback to wrap
        if tier_index >= len(tiers):
            return wrap_fallback_blocks(chunk, max_chars_per_line, max_lines)

        # Split on current tier; if no split happens, advance tier
        parts = split_on_delims(chunk, tiers[tier_index])
        if len(parts) <= 1:
            return refine_chunk(chunk, tier_index + 1)

        # Recurse: each part may still be too large, so refine further
        out: List[str] = []
        for p in parts:
            out.extend(refine_chunk(p, tier_index + 1))
        return out

    # Start by refining the whole text from strongest tier
    blocks = refine_chunk(text, 0)

    # Final safety pass: ensure fit (should already hold)
    safe: List[str] = []
    for b in blocks:
        if block_fits(b, max_chars_per_line, max_lines):
            safe.append(b)
        else:
            safe.extend(wrap_fallback_blocks(b, max_chars_per_line, max_lines))

    return safe

def preferred_split_index(text: str) -> int:
    """
    Return an index at which to split text into two parts, preferring punctuation boundaries.
    If no good boundary, return -1.
    """
    # Prefer these punctuation marks as split points.
    punct = [". ", "? ", "! ", "; ", ": ", ", "]
    for p in punct:
        idx = text.rfind(p)
        if idx != -1 and idx > 20:  # avoid splitting too early
            return idx + len(p)  # split after punctuation+space
    # Fall back to last space
    sp = text.rfind(" ")
    if sp > 20:
        return sp + 1
    return -1

def distribute_time(start: float, end: float, parts: List[str]) -> List[Tuple[float, float, str]]:
    """
    Split the [start,end] interval across N text parts proportionally by character count.
    """
    total = max(1, sum(len(p) for p in parts))
    dur = max(0.0, end - start)
    out: List[Tuple[float, float, str]] = []
    t = start
    for i, p in enumerate(parts):
        frac = len(p) / total
        seg_dur = dur * frac if i < len(parts) - 1 else (end - t)
        out.append((t, t + seg_dur, p))
        t += seg_dur
    return out

def enforce_timing(
    blocks: List[Tuple[float, float, str]],
    min_dur: float,
    max_dur: float
) -> List[Tuple[float, float, str]]:
    """
    - Merge too-short blocks with the next when possible.
    - Split too-long blocks if text is long (basic proportional split).
    """
    merged: List[Tuple[float, float, str]] = []
    i = 0
    while i < len(blocks):
        s, e, txt = blocks[i]
        dur = e - s

        # Merge if too short and there is a next block
        if dur < min_dur and i + 1 < len(blocks):
            ns, ne, ntxt = blocks[i + 1]
            merged.append((s, ne, normalize_spaces(txt + " " + ntxt)))
            i += 2
            continue

        # Keep otherwise
        merged.append((s, e, txt))
        i += 1

    # Split too-long blocks (only if text is large; otherwise leave timing)
    final: List[Tuple[float, float, str]] = []
    for s, e, txt in merged:
        dur = e - s
        if dur > max_dur and len(txt) > 120:
            # split into 2
            cut = preferred_split_index(txt)
            if cut == -1:
                cut = len(txt) // 2
            p1 = normalize_spaces(txt[:cut])
            p2 = normalize_spaces(txt[cut:])
            parts = [p1, p2] if p2 else [p1]
            final.extend(distribute_time(s, e, parts))
        else:
            final.append((s, e, txt))

    return final

def chunk_segments_to_subtitles(
    segments,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    target_cps: float = 17.0,
    min_duration: float = 1.0,
    max_duration: float = 6.0,
    allow_commas: bool = True,
    allow_medium: bool = True,
    prefer_punct_splits: bool = False,
) -> List[SubtitleBlock]:
    """
    Convert whisper segments to subtitle blocks with readability constraints.
    """
    raw: List[Tuple[float, float, str]] = []
    for seg in segments:
        txt = normalize_spaces(seg.text)
        if not txt:
            continue
        raw.append((float(seg.start), float(seg.end), txt))

    # First, split any segment text that is too long to wrap into <= max_lines.
    split_raw: List[Tuple[float, float, str]] = []
    for s, e, txt in raw:
        parts = split_text_into_blocks(txt, max_chars_per_line, max_lines, allow_commas, allow_medium, prefer_punct_splits)
        if len(parts) == 1:
            split_raw.append((s, e, txt))
        else:
            split_raw.extend(distribute_time(s, e, parts))

    # Enforce min/max duration heuristics
    split_raw = enforce_timing(split_raw, min_duration, max_duration)

    # Apply reading speed: if a block is too “dense,” split it.
    density_fixed: List[Tuple[float, float, str]] = []
    for s, e, txt in split_raw:
        dur = max(0.01, e - s)
        cps = len(txt) / dur
        if cps > target_cps and len(txt) > 80:
            # Split into 2 (simple)
            cut = preferred_split_index(txt)
            if cut == -1:
                cut = len(txt) // 2
            p1 = normalize_spaces(txt[:cut])
            p2 = normalize_spaces(txt[cut:])
            parts = [p1, p2] if p2 else [p1]
            density_fixed.extend(distribute_time(s, e, parts))
        else:
            density_fixed.append((s, e, txt))

    # Finally, wrap each text into 1–2 lines. If it still overflows, split again.
    subs: List[SubtitleBlock] = []
    for s, e, txt in density_fixed:
        txt = normalize_spaces(txt)
        parts = split_text_into_blocks(txt, max_chars_per_line, max_lines, allow_commas, allow_medium, prefer_punct_splits)
        timed_parts = distribute_time(s, e, parts) if len(parts) > 1 else [(s, e, txt)]
        for ps, pe, ptxt in timed_parts:
            lines = wrap_text_lines(ptxt, max_chars_per_line)
            # If overflow (rare), force a hard split
            if len(lines) > max_lines:
                lines = lines[:max_lines]
            subs.append(SubtitleBlock(start=ps, end=pe, lines=lines))

    return subs

def write_srt(subs: List[SubtitleBlock], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, sb in enumerate(subs, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(sb.start)} --> {format_srt_time(sb.end)}\n")
            f.write("\n".join(sb.lines).strip() + "\n\n")

# -----------------------------
# Main CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser(description="Local SRT generator (faster-whisper + ffmpeg)")
    ap.add_argument("input", help="Path to audio/video file")
    ap.add_argument("-o", "--output", default="output.srt", help="Output SRT path")
    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="cpu or cuda")
    ap.add_argument("--language", default=None, help="Optional language code (e.g., en). If omitted, auto-detect.")
    ap.add_argument("--max_chars", type=int, default=None, help="Max characters per subtitle line")
    ap.add_argument("--max_lines", type=int, default=None, help="Max lines per subtitle block")
    ap.add_argument("--target_cps", type=float, default=None, help="Target characters-per-second for readability")
    ap.add_argument("--min_dur", type=float, default=None, help="Minimum subtitle duration in seconds")
    ap.add_argument("--max_dur", type=float, default=None, help="Maximum subtitle duration in seconds")
    ap.add_argument("--keep_wav", action="store_true", help="Do not delete temporary WAV file")
    ap.add_argument("--no-comma-split", action="store_true", help="Do not split on commas")
    ap.add_argument("--no-medium-split", action="store_true", help="Do not split on ';' or ':'")
    ap.add_argument("--prefer-punct-splits", action="store_true",
                    help="Prefer punctuation-based splits even if text already fits")
    ap.add_argument(
        "--mode",
        choices=["shorts", "yt", "podcast"],
        default=None,
        help="Preset configuration: 'shorts' (short captions, faster pacing), 'yt' (standard video captions), or 'podcast' (longer captions, slower pacing)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    ap.add_argument("--debug", action="store_true", help="Print full traceback on errors")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    ap.add_argument("--tmpdir", default=None, help="Directory for temporary WAV (defaults to system temp)")
    args = ap.parse_args()

    PRESETS = {
        # Designed for vertical shorts: shorter lines, slightly faster pacing, and
        # more aggressive punctuation splitting so captions feel "snappier".
        "shorts": {
            "max_chars": 28,
            "max_lines": 2,
            "target_cps": 15.0,
            "min_dur": 0.7,
            "max_dur": 3.0,
            "prefer_punct_splits": True,
            # In shorts, commas tend to over-fragment; default to off.
            "allow_commas": False,
            "allow_medium": True,
        },
        # Standard YouTube video captions: comfortable line length, stable pacing.
        "yt": {
            "max_chars": 42,
            "max_lines": 2,
            "target_cps": 17.0,
            "min_dur": 1.0,
            "max_dur": 6.0,
            "prefer_punct_splits": False,
            "allow_commas": True,
            "allow_medium": True,
        },
        "podcast": {
            "max_chars": 40,
            "max_lines": 2,
            "target_cps": 16.0,
            "min_dur": 0.9,
            "max_dur": 5.0,
            "prefer_punct_splits": True,
            "allow_commas": True,
            "allow_medium": True,
        },
    }

    # Start with baseline defaults (equivalent to your previous defaults)
    resolved = {
        "max_chars": 42,
        "max_lines": 2,
        "target_cps": 17.0,
        "min_dur": 1.0,
        "max_dur": 6.0,
        "prefer_punct_splits": args.prefer_punct_splits,
        "allow_commas": not args.no_comma_split,
        "allow_medium": not args.no_medium_split,
    }

    # If a mode is selected, apply its preset as the new baseline
    if args.mode:
        preset = PRESETS[args.mode]
        # apply preset to resolved
        for k, v in preset.items():
            resolved[k] = v

    # Now, apply user overrides (only for parameters that were passed explicitly)
    # For numeric args, explicit means args.<name> is not None.
    if args.max_chars is not None:
        resolved["max_chars"] = args.max_chars
    if args.max_lines is not None:
        resolved["max_lines"] = args.max_lines
    if args.target_cps is not None:
        resolved["target_cps"] = args.target_cps
    if args.min_dur is not None:
        resolved["min_dur"] = args.min_dur
    if args.max_dur is not None:
        resolved["max_dur"] = args.max_dur

    # For boolean-ish punctuation flags:
    # These are explicit by presence; your current flags already reflect explicit choices.
    # However, when mode is set, we want the mode to control defaults unless user explicitly asked otherwise.
    # We treat the presence of the --no-* flags as explicit overrides.
    if args.no_comma_split:
        resolved["allow_commas"] = False
    if args.no_medium_split:
        resolved["allow_medium"] = False
    if args.prefer_punct_splits:
        resolved["prefer_punct_splits"] = True

    quiet = args.quiet
    show_progress = not args.no_progress

    # Preflight checks
    input_path = Path(args.input)
    output_path = Path(args.output)

    ok, reason = preflight_checks(input_path, output_path, args.overwrite)
    if not ok:
        return die(reason, code=2)

    # Ensure output directory exists
    try:
        ensure_parent_dir(output_path)
    except Exception as e:
        return die(f"Could not create output directory for: {output_path} ({e})", code=2)
    
    if shutil.which("ffmpeg") is None:
        return die("ffmpeg not found on PATH. Install it or add it to PATH.", code=2)

    tmpdir = args.tmpdir if args.tmpdir else None
    fd, tmp_wav = tempfile.mkstemp(prefix="srtgen_", suffix=".wav", dir=tmpdir)
    os.close(fd)

    started = time.time()
    try:
        log("1/5 Converting audio with ffmpeg...", quiet=quiet)
        to_wav_16k_mono(str(input_path), tmp_wav)

        log("2/5 Loading model...", quiet=quiet)
        model, device_used, compute_type_used = init_whisper_model_with_fallback(
            model_name=args.model,
            device=args.device,
            quiet=quiet,
        )

        log("3/5 Transcribing...", quiet=quiet)
        t0 = time.time()

        segments_iter, info = model.transcribe(
            tmp_wav,
            vad_filter=True,
            language=args.language
        )

        seg_list = []
        dur_total = probe_duration_seconds(str(tmp_wav))
        last_ratio = 0.0
        for idx, seg in enumerate(segments_iter, start=1):
            seg_list.append(seg)
            pct = ""
            if dur_total and dur_total > 0:
                ratio = seg.end / dur_total
                ratio = max(last_ratio, ratio)
                ratio = min(1.0, ratio)
                last_ratio = ratio
                pct = f"{ratio * 100:5.1f}%"
            progress_line(
                f"   {pct} segments: {idx} | t={format_duration(seg.end)}",
                enabled=show_progress,
                quiet=quiet
            )
        progress_done(enabled=show_progress, quiet=quiet)

        log(f"   Transcription complete: {len(seg_list)} segments in {format_duration(time.time() - t0)}", quiet=quiet)

        log("4/5 Chunking + formatting...", quiet=quiet)
        t1 = time.time()

        subs = chunk_segments_to_subtitles(
            seg_list,
            max_chars_per_line=resolved["max_chars"],
            max_lines=resolved["max_lines"],
            target_cps=resolved["target_cps"],
            min_duration=resolved["min_dur"],
            max_duration=resolved["max_dur"],
            allow_commas=resolved["allow_commas"],
            allow_medium=resolved["allow_medium"],
            prefer_punct_splits=resolved["prefer_punct_splits"],
        )

        log(f"   Chunking complete: {len(subs)} subtitle blocks in {format_duration(time.time() - t1)}", quiet=quiet)

        log("5/5 Writing SRT...", quiet=quiet)
        write_srt(subs, str(output_path))

        log(f"Done: {output_path}  (total {format_duration(time.time() - started)})", quiet=quiet)
        return 0

    except FileNotFoundError as e:
        # Common: ffmpeg not found, or input path typo (though we preflight input)
        return die(str(e), code=2)

    except subprocess.CalledProcessError as e:
        detail = f"\nffmpeg stderr (tail):\n{e.stderr}" if getattr(e, "stderr", None) else ""
        return die(f"ffmpeg failed (exit {e.returncode}). Is the input a valid audio/video file?{detail}", code=2)


    except KeyboardInterrupt:
        return die("Interrupted by user.", code=130)

    except Exception as e:
        # Unexpected failure
        if args.debug:
            traceback.print_exc()
        return die(f"Unhandled error: {e}", code=1)

    finally:
        if not args.keep_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

if __name__ == "__main__":
    raise SystemExit(main())
#END FILE