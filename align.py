from faster_whisper import WhisperModel
from pydub import AudioSegment
from pathlib import Path
import re
import json


CHUNK_DIR = "source/run_20260521_160657_copy"
SRT_FILE = "source/run_20260521_160657_copy/subtitles.srt"
MODEL_SIZE = "base"
DEVICE = "cpu"

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type="int8"
)

def srt_time_to_seconds(time_str):

    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")

    return (
        int(h) * 3600 +
        int(m) * 60 +
        int(s) +
        int(ms) / 1000
    )

def parse_srt(srt_path):

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())

    subtitles = []

    for block in blocks:

        lines = block.splitlines()

        if len(lines) < 3:
            continue

        timing = lines[1]

        text = " ".join(lines[2:])

        start_str, end_str = timing.split(" --> ")

        subtitles.append({
            "text": text,
            "start": srt_time_to_seconds(start_str),
            "end": srt_time_to_seconds(end_str)
        })

    return subtitles

subs = parse_srt(SRT_FILE)

chunk_files = sorted(
    Path(CHUNK_DIR).glob("chunk_*.wav")
)

all_words = []

for idx, chunk_file in enumerate(chunk_files):

    if idx >= len(subs):
        break

    sub = subs[idx]

    sentence_text = sub["text"]

    offset = subs[idx]["start"]

    print(f"\nProcessing: {chunk_file.name}")
    print(sentence_text)

    segments, info = model.transcribe(
        str(chunk_file),
        beam_size=5,
        word_timestamps=True
    )

    for segment in segments:

        for word in segment.words:

            clean_word = word.word.strip()

            if not clean_word:
                continue

            all_words.append({
                "word": clean_word,
                "start": round(word.start + offset, 3),
                "end": round(word.end + offset, 3)
            })

with open("word_timestamps.json", "w", encoding="utf-8") as f:
    json.dump(
        all_words,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nDONE: word_timestamps.json")