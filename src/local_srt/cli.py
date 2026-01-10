#!/usr/bin/env python3
"""Command-line interface for Local SRT.

This is the main entry point for the srtgen command-line tool.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Local imports from refactored modules
from .audio import detect_silences, to_wav_16k_mono
from .batch import default_output_for, expand_inputs, preflight_one
from .config import MODE_ALIASES, PRESETS, apply_overrides, load_config_file
from .logging_utils import die, format_duration, log, progress_done, progress_line, warn
from .model_management import (
    delete_model_cli,
    diagnose,
    download_model_cli,
    list_available_models,
    list_downloaded_models,
)
from .models import TOOL_VERSION, ResolvedConfig
from .output_writers import (
    segments_to_jsonable,
    write_ass,
    write_json_bundle,
    write_srt,
    write_txt,
    write_vtt,
)
from .subtitle_generation import (
    apply_silence_alignment,
    chunk_segments_to_subtitles,
    chunk_words_to_subtitles,
    collect_words,
    hygiene_and_polish,
    words_to_subtitles,
)
from .system import ensure_parent_dir, ffmpeg_ok, probe_duration_seconds
from .whisper_wrapper import init_whisper_model


# ============================================================
# Run one file
# ============================================================

def run_one(
    *,
    input_path: Path,
    output_path: Path,
    fmt: str,
    transcript_path: Optional[Path],
    segments_path: Optional[Path],
    json_bundle_path: Optional[Path],
    args: argparse.Namespace,
    cfg: ResolvedConfig,
    quiet: bool,
    show_progress: bool,
) -> int:
    """Process a single media file and generate subtitles.

    Args:
        input_path: Path to input media file
        output_path: Path for primary output file
        fmt: Output format (srt/vtt/ass/txt/json)
        transcript_path: Optional path for plain text transcript
        segments_path: Optional path for segments JSON
        json_bundle_path: Optional path for complete JSON bundle
        args: Command-line arguments namespace
        cfg: Resolved configuration
        quiet: Suppress non-error output
        show_progress: Show transcription progress

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if not ffmpeg_ok():
        return die("ffmpeg not found on PATH. Install it or add it to PATH.", 2)

    ensure_parent_dir(output_path)
    if transcript_path:
        ensure_parent_dir(transcript_path)
    if segments_path:
        ensure_parent_dir(segments_path)
    if json_bundle_path:
        ensure_parent_dir(json_bundle_path)

    tmpdir = args.tmpdir if args.tmpdir else None
    fd, tmp_wav = tempfile.mkstemp(prefix="srtgen_", suffix=".wav", dir=tmpdir)
    os.close(fd)

    started = time.time()
    try:
        log(f"Input: {input_path}", quiet=quiet)
        log(f"Output: {output_path}", quiet=quiet)

        if args.dry_run:
            log("Dry run: skipping transcription.", quiet=quiet)
            return 0

        log("1/5 Converting audio with ffmpeg...", quiet=quiet)
        to_wav_16k_mono(str(input_path), tmp_wav)

        log("2/5 Loading model...", quiet=quiet)
        model, device_used, compute_type_used = init_whisper_model(
            model_name=args.model,
            device=args.device,
            quiet=quiet,
            strict_cuda=args.strict_cuda,
        )

        log("3/5 Transcribing...", quiet=quiet)
        t0 = time.time()

        segments_iter, info = model.transcribe(
            tmp_wav,
            vad_filter=cfg.vad_filter,
            language=args.language,
            word_timestamps=cfg.word_timestamps,
        )

        seg_list: List[Any] = []
        dur_total = probe_duration_seconds(str(tmp_wav))
        last_ratio = 0.0

        # Progress uses:
        # - segment time in media (seg.end)
        # - wall elapsed
        # - realtime factor = seg.end / elapsed
        # - ETA in wall-time = (dur_total - seg.end) / realtime_factor
        for idx, seg in enumerate(segments_iter, start=1):
            seg_list.append(seg)

            pct = ""
            eta = ""
            rt = ""

            now = time.time()
            elapsed = max(0.001, now - t0)
            media_t = float(getattr(seg, "end", 0.0))
            if media_t > 0:
                rtf = media_t / elapsed  # media seconds per wall second
                rt = f"{rtf:4.2f}x"

            if dur_total and dur_total > 0:
                ratio = media_t / dur_total
                ratio = max(last_ratio, ratio)
                ratio = min(1.0, ratio)
                last_ratio = ratio
                pct = f"{ratio * 100:5.1f}%"

                if media_t > 0:
                    rtf = media_t / elapsed
                    if rtf > 0.01:
                        remaining_media = max(0.0, dur_total - media_t)
                        eta_sec = remaining_media / rtf
                        eta = f"ETA {format_duration(eta_sec)}"

            progress_line(
                f"   {pct} segs:{idx:5d} | media_t={format_duration(media_t)} | {rt} {eta}",
                enabled=show_progress,
                quiet=quiet,
            )

        progress_done(enabled=show_progress, quiet=quiet)
        log(f"   Transcription complete: {len(seg_list)} segments in {format_duration(time.time() - t0)}", quiet=quiet)

        log("4/5 Chunking + formatting...", quiet=quiet)
        t1 = time.time()
        silences: List[Tuple[float, float]] = []
        if cfg.use_silence_split:
            silences = detect_silences(
                tmp_wav,
                min_silence_dur=cfg.silence_min_dur,
                silence_threshold_db=cfg.silence_threshold_db,
            )

        words = collect_words(seg_list) if cfg.word_timestamps else []
        if args.word_level:
            if not words:
                return die("Word-level output requested but no word timestamps are available.", 2)
            subs = words_to_subtitles(words)
        else:
            if cfg.use_silence_split and words:
                subs = chunk_words_to_subtitles(words, cfg, silences)
            else:
                if cfg.use_silence_split and not words:
                    warn("Silence splitting requested but no word timestamps returned; falling back.", quiet=quiet)
                subs = chunk_segments_to_subtitles(seg_list, cfg)

        if cfg.use_silence_split and silences:
            subs = apply_silence_alignment(subs, silences)
        subs = hygiene_and_polish(
            subs,
            min_gap=cfg.min_gap,
            pad=cfg.pad,
            silence_intervals=silences if cfg.use_silence_split else None,
        )
        log(f"   Chunking complete: {len(subs)} subtitle blocks in {format_duration(time.time() - t1)}", quiet=quiet)

        log("5/5 Writing outputs...", quiet=quiet)

        if fmt == "srt":
            write_srt(subs, output_path, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        elif fmt == "vtt":
            write_vtt(subs, output_path, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        elif fmt == "ass":
            write_ass(subs, output_path, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        elif fmt == "txt":
            write_txt(subs, output_path)
        elif fmt == "json":
            write_json_bundle(
                output_path,
                input_file=str(input_path),
                device_used=device_used,
                compute_type_used=compute_type_used,
                cfg=cfg,
                segments=seg_list,
                subs=subs,
                tool_version=TOOL_VERSION,
            )
        else:
            return die(f"Unknown format: {fmt}", 2)

        # Optional side outputs
        if transcript_path:
            # transcript from subtitle blocks
            write_txt(subs, transcript_path)

        if segments_path:
            import json

            ensure_parent_dir(segments_path)
            tmp = segments_path.with_suffix(segments_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "input_file": str(input_path),
                        "segments": segments_to_jsonable(seg_list, include_words=cfg.word_timestamps),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, segments_path)

        if json_bundle_path:
            write_json_bundle(
                json_bundle_path,
                input_file=str(input_path),
                device_used=device_used,
                compute_type_used=compute_type_used,
                cfg=cfg,
                segments=seg_list,
                subs=subs,
                tool_version=TOOL_VERSION,
            )

        log(f"Done: {output_path} (total {format_duration(time.time() - started)})", quiet=quiet)
        return 0

    finally:
        if not args.keep_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """Main entry point for the srtgen command-line tool.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    ap = argparse.ArgumentParser(description="Local SRT/VTT generator (faster-whisper + ffmpeg)")
    ap.add_argument("inputs", nargs="*", help="Media file(s), directory, or glob pattern(s)")
    ap.add_argument("--glob", default=None, help="Additional glob pattern to include (optional)")
    ap.add_argument("--outdir", default=None, help="Output directory (batch mode). If omitted, writes next to input.")
    ap.add_argument("--keep-structure", action="store_true", help="When using --outdir, preserve directory structure.")
    ap.add_argument("--root", default=None, help="Base root for --keep-structure (defaults to common parent when possible).")

    ap.add_argument("-o", "--output", default=None, help="Single-file output path (only valid when one input expands to one file).")

    ap.add_argument("--format", choices=["srt", "vtt", "ass", "txt", "json"], default="srt", help="Primary output format")
    ap.add_argument("--emit-transcript", default=None, help="Also write a transcript TXT to this path (or outdir-based if a directory).")
    ap.add_argument("--emit-segments", default=None, help="Also write segments JSON to this path (or outdir-based if a directory).")
    ap.add_argument("--emit-bundle", default=None, help="Also write a full JSON bundle (segments+subs+config) to this path (or outdir-based if a directory).")

    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto/cpu/cuda")
    ap.add_argument("--strict-cuda", action="store_true", help="If set, fail instead of falling back when CUDA init fails.")
    ap.add_argument("--language", default=None, help="Optional language code (e.g., en). If omitted, auto-detect.")
    ap.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Request word timestamps (preserved in JSON outputs).",
    )
    ap.add_argument("--word-level", action="store_true", help="Output word-level subtitle cues (requires word timestamps).")

    ap.add_argument("--mode", default=None, help="Preset modes: shorts | yt | podcast (aliases: short, youtube, pod).")
    ap.add_argument("--config", default=None, help="JSON config file. CLI args override config.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and show resolved settings, but do not transcribe.")

    ap.add_argument("--max_chars", type=int, default=None)
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--target_cps", type=float, default=None)
    ap.add_argument("--min_dur", type=float, default=None)
    ap.add_argument("--max_dur", type=float, default=None)

    ap.add_argument("--no-comma-split", action="store_true")
    ap.add_argument("--no-medium-split", action="store_true")
    ap.add_argument("--prefer-punct-splits", action="store_true")

    ap.add_argument("--min-gap", type=float, default=None, help="Minimum gap between consecutive subtitle cues (seconds).")
    ap.add_argument("--pad", type=float, default=None, help="Pad cues into silence where possible (seconds).")
    ap.add_argument("--no-silence-split", action="store_true", help="Disable silence-based splitting/alignment.")
    ap.add_argument("--silence-min-dur", type=float, default=None, help="Minimum silence duration for splits (seconds).")
    ap.add_argument("--silence-threshold", type=float, default=None, help="Silence threshold in dB (e.g., -35).")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    ap.add_argument("--keep_wav", action="store_true", help="Do not delete temporary WAV file")
    ap.add_argument("--tmpdir", default=None, help="Directory for temporary WAV (defaults to system temp)")

    ap.add_argument("--list-models", action="store_true", help="List downloaded faster-whisper models and exit.")
    ap.add_argument("--list-available-models", action="store_true", help="List available faster-whisper model names and exit.")
    ap.add_argument("--download-model", default=None, help="Download a faster-whisper model and exit.")
    ap.add_argument("--delete-model", default=None, help="Delete a downloaded model from cache and exit.")

    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true", help="Batch mode: continue processing other files on error.")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--diagnose", action="store_true")

    args = ap.parse_args()

    if args.version:
        print(TOOL_VERSION)
        return 0

    if args.diagnose:
        diagnose()
        return 0

    if args.list_models or args.list_available_models or args.download_model or args.delete_model:
        if args.inputs:
            return die("Model management options must be used without input files.", 2)
        rc = 0
        if args.list_models:
            downloaded = list_downloaded_models()
            if downloaded:
                print("Downloaded models:")
                for name, path in downloaded:
                    print(f"  - {name}: {path}")
            else:
                print("No downloaded models found.")
            print("Available models:")
            print("  " + ", ".join(list_available_models()))
        if args.list_available_models and not args.list_models:
            print("Available models:")
            print("  " + ", ".join(list_available_models()))
        if args.download_model:
            rc = download_model_cli(args.download_model)
            if rc != 0:
                return rc
        if args.delete_model:
            rc = delete_model_cli(args.delete_model)
            if rc != 0:
                return rc
        return rc

    quiet = args.quiet
    show_progress = not args.no_progress

    if args.mode:
        mode_key = args.mode.lower()
        if mode_key not in MODE_ALIASES:
            return die(
                f"Invalid --mode '{args.mode}'. "
                f"Valid modes: shorts, yt, podcast",
                code=2,
            )
        args.mode = MODE_ALIASES[mode_key]

    # Expand inputs (files, dirs, globs)
    if not args.inputs:
        return die("No input files provided.", 2)
    files = expand_inputs(args.inputs, args.glob)
    files = [p for p in files if p.exists() and p.is_file()]
    if not files:
        return die("No input files found after expansion.", 2)

    # Compute outdir/root
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        ensure_parent_dir(outdir / "dummy.txt")  # ensures outdir's parent exists
        outdir.mkdir(parents=True, exist_ok=True)

    if args.root:
        base_root = Path(args.root)
    else:
        # best-effort common parent for keep-structure
        base_root = None
        if args.keep_structure and len(files) > 1:
            try:
                base_root = Path(os.path.commonpath([str(f.resolve()) for f in files]))
                if base_root.is_file():
                    base_root = base_root.parent
            except Exception:
                base_root = None

    # Build config: defaults -> config file -> preset -> CLI overrides
    cfg = ResolvedConfig()

    try:
        cfg_file = load_config_file(args.config)
    except Exception as e:
        return die(str(e), 2)

    cfg = apply_overrides(cfg, cfg_file)

    if args.mode:
        cfg = apply_overrides(cfg, PRESETS[args.mode])

    # CLI overrides
    if args.max_chars is not None:
        cfg.max_chars = args.max_chars
    if args.max_lines is not None:
        cfg.max_lines = args.max_lines
    if args.target_cps is not None:
        cfg.target_cps = args.target_cps
    if args.min_dur is not None:
        cfg.min_dur = args.min_dur
    if args.max_dur is not None:
        cfg.max_dur = args.max_dur

    if args.no_comma_split:
        cfg.allow_commas = False
    if args.no_medium_split:
        cfg.allow_medium = False
    if args.prefer_punct_splits:
        cfg.prefer_punct_splits = True

    if args.min_gap is not None:
        cfg.min_gap = float(args.min_gap)
    if args.pad is not None:
        cfg.pad = float(args.pad)

    if args.no_silence_split:
        cfg.use_silence_split = False
    if args.silence_min_dur is not None:
        cfg.silence_min_dur = float(args.silence_min_dur)
    if args.silence_threshold is not None:
        cfg.silence_threshold_db = float(args.silence_threshold)

    if args.word_timestamps:
        cfg.word_timestamps = True
    if args.word_level or cfg.use_silence_split:
        cfg.word_timestamps = True

    # Basic dependency check
    if not ffmpeg_ok():
        return die("ffmpeg not found on PATH. Install it or add it to PATH.", 2)

    # If user provided --output, require single file expansion
    if args.output is not None and len(files) != 1:
        return die("--output may only be used when exactly one input file is provided (after expansion).", 2)

    # Show resolved config for transparency
    if args.dry_run and not quiet:
        import json
        log("Resolved config:", quiet=quiet)
        log(json.dumps(dataclasses.asdict(cfg), indent=2), quiet=quiet)

    # Run (single or batch)
    failures: List[Tuple[Path, str]] = []
    for f in files:
        # Determine output path(s)
        if args.output:
            primary_out = Path(args.output)
        else:
            primary_out = default_output_for(f, outdir, args.format, args.keep_structure, base_root)

        ok, reason = preflight_one(f, primary_out, args.overwrite)
        if not ok:
            failures.append((f, reason))
            if not args.continue_on_error:
                return die(reason, 2)
            warn(f"{f}: {reason}", quiet=quiet)
            continue

        # Side outputs
        def side_path(user_path: Optional[str], ext: str) -> Optional[Path]:
            if not user_path:
                return None
            p = Path(user_path)
            if p.exists() and p.is_dir():
                # directory: mirror naming there
                return (p / f.name).with_suffix(ext)
            if user_path.endswith(os.sep) or user_path.endswith("/"):
                # treat as directory-like
                p.mkdir(parents=True, exist_ok=True)
                return (p / f.name).with_suffix(ext)
            # explicit file path
            return p

        transcript_out = side_path(args.emit_transcript, ".txt")
        segments_out = side_path(args.emit_segments, ".segments.json")
        bundle_out = side_path(args.emit_bundle, ".bundle.json") if args.emit_bundle else None

        try:
            rc = run_one(
                input_path=f,
                output_path=primary_out,
                fmt=args.format,
                transcript_path=transcript_out,
                segments_path=segments_out,
                json_bundle_path=bundle_out,
                args=args,
                cfg=cfg,
                quiet=quiet,
                show_progress=show_progress,
            )
            if rc != 0:
                failures.append((f, f"failed with exit code {rc}"))
                if not args.continue_on_error:
                    return rc
        except KeyboardInterrupt:
            return die("Interrupted by user.", 130)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            failures.append((f, str(e)))
            if not args.continue_on_error:
                return die(f"{f}: {e}", 1)
            warn(f"{f}: {e}", quiet=quiet)

    if failures:
        if not quiet:
            log("\nSummary: failures:", quiet=quiet)
            for f, msg in failures:
                log(f"  - {f}: {msg}", quiet=quiet)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
