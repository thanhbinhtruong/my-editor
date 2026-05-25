from faster_whisper import WhisperModel
from pydub import AudioSegment
from pathlib import Path
import re
import json


class WhisperWordTimestamper:

    def __init__(
        self,
        chunk_dir: str,
        srt_file: str,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ):
        self.chunk_dir = Path(chunk_dir)
        self.srt_file = srt_file
        self.beam_size = beam_size

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    @staticmethod
    def _srt_time_to_seconds(time_str: str) -> float:
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def _parse_srt(self) -> list[dict]:
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
                "text": " ".join(lines[2:]),
                "start": self._srt_time_to_seconds(start_str),
                "end": self._srt_time_to_seconds(end_str),
            })

        return subtitles

    def _transcribe_chunk(self, chunk_file: Path, offset: float) -> list[dict]:
        words = []
        segments, _ = self.model.transcribe(
            str(chunk_file),
            beam_size=self.beam_size,
            word_timestamps=True,
        )

        for segment in segments:
            for word in segment.words:
                clean_word = word.word.strip()
                if not clean_word:
                    continue
                words.append({
                    "word": clean_word,
                    "start": round(word.start + offset, 3),
                    "end": round(word.end + offset, 3),
                })

        return words

    def run(self, output_path: str = "word_timestamps.json") -> list[dict]:
        subs = self._parse_srt()
        chunk_files = sorted(self.chunk_dir.glob("chunk_*.wav"))
        all_words = []

        for idx, chunk_file in enumerate(chunk_files):
            if idx >= len(subs):
                break

            sub = subs[idx]
            print(f"\nProcessing: {chunk_file.name}")
            print(sub["text"])

            words = self._transcribe_chunk(chunk_file, offset=sub["start"])
            all_words.extend(words)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_words, f, ensure_ascii=False, indent=2)

        print(f"\nDONE: {output_path}")
        return all_words


if __name__ == "__main__":
    timestamper = WhisperWordTimestamper(
        chunk_dir="source/run_20260521_160657_copy",
        srt_file="source/run_20260521_160657_copy/subtitles.srt",
        model_size="base",
        device="cpu",
    )
    timestamper.run()