import os
import re
import json
from pathlib import Path

import torch
import whisperx


torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())


class WhisperWordTimestamper:

    def __init__(
        self,
        chunk_dir: str,
        srt_file: str,
        device: str = "cpu",
        language: str = "en",
    ):
        self.chunk_dir = Path(chunk_dir)
        self.srt_file = srt_file
        self.device = device
        self.language = language

        # CHỈ load align model
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

    @staticmethod
    def _srt_time_to_seconds(time_str: str) -> float:
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def _parse_srt(self):
        with open(self.srt_file, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r"\n\s*\n", content.strip())

        subtitles = []

        for block in blocks:
            lines = block.splitlines()

            if len(lines) < 3:
                continue

            start_str, end_str = lines[1].split(" --> ")

            subtitles.append({
                "text": " ".join(lines[2:]).strip(),
                "start": self._srt_time_to_seconds(start_str),
                "end": self._srt_time_to_seconds(end_str),
            })

        return subtitles

    def _align_chunk(
        self,
        chunk_file: Path,
        text: str,
        offset: float,
        duration: float,
    ):
        audio = whisperx.load_audio(str(chunk_file))

        with torch.inference_mode():

            result = whisperx.align(
                [{
                    "text": text,
                    "start": 0.0,
                    "end": duration,
                }],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

        words = []

        for segment in result["segments"]:

            for w in segment.get("words", []):

                word = w["word"].strip()

                if not word:
                    continue

                start = round(w["start"] + offset, 3)
                end = round(w["end"] + offset, 3)

                if end <= start:
                    continue

                words.append({
                    "word": word,
                    "start": start,
                    "end": end,
                })

        return words

    def run(self, output_path="word_timestamps.json"):

        subtitles = self._parse_srt()

        chunk_files = sorted(
            self.chunk_dir.glob("chunk_*.wav"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        all_words = []

        for idx, chunk_file in enumerate(chunk_files):

            if idx >= len(subtitles):
                break

            sub = subtitles[idx]

            print(f"Aligning: {chunk_file.name}")

            words = self._align_chunk(
                chunk_file=chunk_file,
                text=sub["text"],
                offset=sub["start"],
                duration=sub["end"] - sub["start"],
            )

            all_words.extend(words)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_words, f, ensure_ascii=False, indent=2)

        print(f"\nDONE: {output_path}")

        return all_words