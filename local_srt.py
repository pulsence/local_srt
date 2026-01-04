import argparse
import subprocess
from faster_whisper import WhisperModel

def to_wav_16k_mono(input_path: str, wav_path: str):
    # ffmpeg -i input -ac 1 -ar 16000 -vn output.wav
    subprocess.check_call([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-vn",
        wav_path
    ])

def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        i = 1
        for seg in segments:
            start = format_srt_time(seg.start)
            end = format_srt_time(seg.end)
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            i += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to audio/video file")
    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--output", default="output.srt")
    args = ap.parse_args()

    wav_path = "tmp_16k_mono.wav"
    to_wav_16k_mono(args.input, wav_path)

    model = WhisperModel(args.model, device=args.device, compute_type="int8" if args.device == "cpu" else "float16")
    segments, info = model.transcribe(wav_path, vad_filter=True)

    write_srt(list(segments), args.output)
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
