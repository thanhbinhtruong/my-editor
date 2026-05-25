from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import subprocess
import json


class SubtitleVideoRenderer:

    def __init__(
        self,
        timestamp_json: str,
        input_video: str,
        font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        video_width: int = 1920,
        video_height: int = 1080,
        fps: int = 30,
        font_size: int = 52,
        pause_threshold: float = 0.45,
        final_output: str = "subtitle_final.mp4",
    ):
        self.timestamp_json = timestamp_json
        self.input_video = input_video
        self.font_path = font_path
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
        self.font_size = font_size
        self.pause_threshold = pause_threshold
        self.final_output = final_output

        self.words_data: list[dict] = []
        self.sentence_blocks: list[list[dict]] = []
        self.font: ImageFont.FreeTypeFont = None
        self._frames: list[np.ndarray] = []

    # ----------------------------------------------------------
    # LOAD & PREPARE
    # ----------------------------------------------------------

    def _load_words(self):
        with open(self.timestamp_json, "r", encoding="utf-8") as f:
            self.words_data = json.load(f)

    def _group_sentences(self):
        self.sentence_blocks = []
        current_block = []

        for i, word in enumerate(self.words_data):
            if i == 0:
                current_block.append(word)
                continue

            gap = word["start"] - self.words_data[i - 1]["end"]

            if gap > self.pause_threshold:
                self.sentence_blocks.append(current_block)
                current_block = []

            current_block.append(word)

        if current_block:
            self.sentence_blocks.append(current_block)

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _get_current_block_and_word(self, current_time: float) -> tuple[list | None, int]:
        for block in self.sentence_blocks:
            if block[0]["start"] <= current_time <= block[-1]["end"]:
                active_index = -1
                for idx, word in enumerate(block):
                    if word["start"] <= current_time < word["end"]:
                        active_index = idx
                        break
                return block, active_index
        return None, -1

    def _split_lines(self, draw: ImageDraw.Draw, words: list[dict]) -> list[list[dict]]:
        max_width = self.video_width * 0.75
        lines = []
        current_line = []

        for word in words:
            test_line = current_line + [word]
            text = " ".join(w["word"] for w in test_line)
            bbox = draw.textbbox((0, 0), text, font=self.font)
            width = bbox[2] - bbox[0]

            if width > max_width and current_line:
                lines.append(current_line)
                current_line = [word]
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)

        return lines

    # ----------------------------------------------------------
    # DRAW
    # ----------------------------------------------------------

    def _draw_active_word(self, draw: ImageDraw.Draw, x: int, y: int, word: str, w: int):
        padding_x, padding_y = 24, 16

        # glow
        draw.rounded_rectangle(
            (x - padding_x - 6, y - padding_y - 6,
             x + w + padding_x + 6, y + self.font_size + padding_y + 6),
            radius=28,
            fill=(180, 80, 255, 255),
        )
        # box
        draw.rounded_rectangle(
            (x - padding_x, y - padding_y,
             x + w + padding_x, y + self.font_size + padding_y),
            radius=24,
            fill=(120, 0, 255, 255),
        )

        draw.text((x, y), word, font=self.font, fill=(255, 255, 255, 255))

    def _draw_frame(self, current_time: float) -> np.ndarray:
        # nền trong suốt
        img = Image.new("RGBA", (self.video_width, self.video_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        current_block, active_index = self._get_current_block_and_word(current_time)

        if current_block is None:
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

        lines = self._split_lines(draw, current_block)
        line_spacing = 40
        total_text_height = len(lines) * self.font_size + (len(lines) - 1) * line_spacing
        start_y = self.video_height - 840 - total_text_height // 2
        global_word_index = 0
        spacing = 24

        for line in lines:
            widths = []
            total_width = 0

            for item in line:
                bbox = draw.textbbox((0, 0), item["word"], font=self.font)
                w = bbox[2] - bbox[0]
                widths.append(w)
                total_width += w + spacing
            total_width -= spacing

            x = (self.video_width - total_width) // 2

            for idx, item in enumerate(line):
                word = item["word"]
                w = widths[idx]

                if global_word_index == active_index:
                    self._draw_active_word(draw, x, start_y, word, w)
                else:
                    draw.text((x, start_y), word, font=self.font, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
                x += w + spacing
                global_word_index += 1

            start_y += self.font_size + line_spacing

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)



    def _render_and_merge(self):
        total_frames = int(self.words_data[-1]["end"] * self.fps)

        process = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.video_width}x{self.video_height}",
            "-pix_fmt", "bgra",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-i", self.input_video,
            "-filter_complex", "[1:v][0:v]overlay=0:0",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self.final_output,
        ], stdin=subprocess.PIPE)

        for frame_idx in range(total_frames):
            if frame_idx % 100 == 0:
                print(f"  Rendering frame {frame_idx}/{total_frames}...")

            frame = self._draw_frame(frame_idx / self.fps)
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()
        print("DONE:", self.final_output)
        
    # ----------------------------------------------------------
    # ENTRY POINT
    # ----------------------------------------------------------

    def run(self):
        self._load_words()
        self._group_sentences()
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self._render_and_merge()


if __name__ == "__main__":
    renderer = SubtitleVideoRenderer(
        timestamp_json="word_timestamps.json",
        input_video="source/run_20260521_160657_copy/final_video.mp4",
    )
    renderer.run()