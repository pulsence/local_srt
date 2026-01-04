import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

from faster_whisper import WhisperModel


# -----------------------------
# Audio conversion
# -----------------------------
def to_wav_16k_mono(input_path: str, wav_path: str):
    subprocess.check_call(
    [
        "ffmpeg",
        "-y",
        "-loglevel", "error",   # or "quiet"
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        wav_path,
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)



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
# Chunking rules
# -----------------------------
def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wrap_text_lines(text: str, max_chars_per_line: int) -> List[str]:
    """
    Greedy word wrap into as many lines as needed.
    NEVER drops words.
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


def split_text_into_blocks(text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
    """
    Break long text into multiple block-texts that can be wrapped into <= max_lines.
    """
    text = normalize_spaces(text)
    blocks: List[str] = []

    # Quick accept if it fits when wrapped
    lines = wrap_text_lines(text, max_chars_per_line)
    if not lines:
        return []
    
    blocks: List[str] = []
    for i in range(0, len(lines), max_lines):
        block_lines = lines[i:i + max_lines]
        blocks.append(" ".join(block_lines))  # join preserves words; re-wrap is stable
    return blocks


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
    max_duration: float = 6.0
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
        parts = split_text_into_blocks(txt, max_chars_per_line, max_lines)
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
        parts = split_text_into_blocks(txt, max_chars_per_line, max_lines)
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
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--language", default=None, help="Optional language code (e.g., en). If omitted, auto-detect.")
    ap.add_argument("--max_chars", type=int, default=42, help="Max characters per subtitle line")
    ap.add_argument("--max_lines", type=int, default=2, help="Max lines per subtitle block")
    ap.add_argument("--target_cps", type=float, default=17.0, help="Target characters-per-second for readability")
    ap.add_argument("--min_dur", type=float, default=1.0, help="Minimum subtitle duration in seconds")
    ap.add_argument("--max_dur", type=float, default=6.0, help="Maximum subtitle duration in seconds")
    ap.add_argument("--keep_wav", action="store_true", help="Do not delete temporary WAV file")
    args = ap.parse_args()

    tmp_wav = "tmp_16k_mono.wav"
    try:
        to_wav_16k_mono(args.input, tmp_wav)

        compute_type = "int8" if args.device == "cpu" else "float16"
        model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

        segments, info = model.transcribe(
            tmp_wav,
            vad_filter=True,
            language=args.language
        )

        seg_list = list(segments)
        subs = chunk_segments_to_subtitles(
            seg_list,
            max_chars_per_line=args.max_chars,
            max_lines=args.max_lines,
            target_cps=args.target_cps,
            min_duration=args.min_dur,
            max_duration=args.max_dur,
        )

        write_srt(subs, args.output)
        print(f"Wrote: {args.output}")

    finally:
        if not args.keep_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                # If Windows file lock happens for some reason, don’t fail the whole run.
                pass


if __name__ == "__main__":
    main()
