from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
import json
import re


class SubtitleVideoRenderer:

    def __init__(
        self,
        timestamp_json,
        input_video,
        srt_file,
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        w=1920,
        h=1080,
        fps=30,
        font_size=52,
        output="out.mp4",
    ):
        self.timestamp_json = timestamp_json
        self.input_video = input_video
        self.srt_file = srt_file

        self.W = w
        self.H = h
        self.fps = fps
        self.font_size = font_size
        self.output = output

        self.font = ImageFont.truetype(font_path, font_size)

        self.words = []
        self.blocks = []

        # caches
        self.base_cache = {}   # block_id -> static image
        self.layout_cache = {} # block_id -> layout info

    # ─────────────────────────────
    # LOAD WORDS
    # ─────────────────────────────
    def load_words(self):
        with open(self.timestamp_json, "r", encoding="utf-8") as f:
            self.words = json.load(f)

    def srt_to_sec(self, t):
        h, m, s_ms = t.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def load_srt(self):
        with open(self.srt_file, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = []
        for b in re.split(r"\n\s*\n", content.strip()):
            lines = b.splitlines()
            if len(lines) < 2:
                continue
            try:
                start, end = lines[1].split(" --> ")
                blocks.append((self.srt_to_sec(start), self.srt_to_sec(end)))
            except:
                pass

        return blocks

    # ─────────────────────────────
    # GROUP WORDS
    # ─────────────────────────────
    def group_words(self):
        srt_blocks = self.load_srt()

        blocks = []
        i = 0

        for start, end in srt_blocks:
            block = []

            while i < len(self.words):
                w = self.words[i]

                if w["start"] > end + 0.3:
                    break

                if w["start"] >= start - 0.3:
                    block.append(w)

                i += 1

            if block:
                blocks.append(block)

        self.blocks = blocks
        print("Blocks:", len(blocks))

    # ─────────────────────────────
    # PRE-COMPUTE LAYOUT (IMPORTANT)
    # ─────────────────────────────
    def build_layout(self, block):
        spacing = 18
        max_width = self.W * 0.75

        lines = []
        current = []

        # split lines
        for w in block:
            test = current + [w]
            text = " ".join(x["word"] for x in test)

            bbox = self.font.getbbox(text)
            width = bbox[2] - bbox[0]

            if width > max_width and current:
                lines.append(current)
                current = [w]
            else:
                current = test

        if current:
            lines.append(current)

        # build positions
        layout = []
        y = self.H - 900  # tăng lên = cao hơn (350-450 tùy ý)

        for line in lines:
            widths = []
            total = 0

            for w in line:
                bbox = self.font.getbbox(w["word"])
                ww = bbox[2] - bbox[0]
                widths.append(ww)
                total += ww + spacing

            total -= spacing
            x = (self.W - total) // 2

            layout.append((line, widths, x, y))
            y += self.font_size + 30

        return layout

    # ─────────────────────────────
    # BUILD STATIC BASE FRAME (KEY OPTIMIZATION)
    # ─────────────────────────────
    def build_base_frame(self, block_id, block):
        img = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        layout = self.build_layout(block)

        self.layout_cache[block_id] = layout

        global_idx = 0

        for line, widths, x, y in layout:
            for i, w in enumerate(line):
                word = w["word"]
                ww = widths[i]

                draw.text(
                    (x, y),
                    word,
                    font=self.font,
                    fill=(255, 255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 255),
                )

                x += ww + 18
                global_idx += 1

        # Keep RGBA for transparency overlay
        self.base_cache[block_id] = np.array(img)

    # ─────────────────────────────
    # BUILD ALL BASE FRAMES
    # ─────────────────────────────
    def build_all_base(self):
        for i, block in enumerate(self.blocks):
            self.build_base_frame(i, block)

        print("Base frames ready:", len(self.base_cache))

    # ─────────────────────────────
    # RENDER HIGHLIGHT ONLY
    # ─────────────────────────────
    def render(self, block_id, active_index):
        base = self.base_cache[block_id].copy()
        img = Image.fromarray(base)
        draw = ImageDraw.Draw(img)

        layout = self.layout_cache[block_id]

        idx = 0

        for line, widths, x0, y in layout:
            x = x0

            for i, w in enumerate(line):
                ww = widths[i]

                if idx == active_index:
                    # Draw highlight background
                    draw.rounded_rectangle(
                        (x - 10, y - 10, x + ww + 10, y + self.font_size + 10),
                        radius=18,
                        fill=(120, 0, 255, 220),
                    )
                    # Redraw text on top of highlight
                    draw.text(
                        (x, y),
                        w["word"],
                        font=self.font,
                        fill=(255, 255, 255, 255),
                        stroke_width=2,
                        stroke_fill=(0, 0, 0, 255),
                    )

                x += ww + 18
                idx += 1

        # Keep RGBA for transparency
        return np.array(img)

    # ─────────────────────────────
    # BUILD EVENTS
    # ─────────────────────────────
    def build_events(self):
        events = []

        for bid, block in enumerate(self.blocks):
            for i, w in enumerate(block):
                events.append((w["start"], bid, i))

        events.sort(key=lambda x: x[0])
        return events

    # ─────────────────────────────
    # GET VIDEO DURATION
    # ─────────────────────────────
    def get_video_duration(self):
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", self.input_video
            ],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())

    # ─────────────────────────────
    # RUN
    # ─────────────────────────────
    def run(self):
        self.load_words()
        self.group_words()
        self.build_all_base()

        events = self.build_events()

        # Get actual video duration to prevent frame mismatch
        video_duration = self.get_video_duration()
        word_duration = self.words[-1]["end"]
        actual_duration = min(video_duration, word_duration)
        total_frames = int(actual_duration * self.fps)

        print(f"Video duration: {video_duration:.2f}s, Word duration: {word_duration:.2f}s")
        print(f"Rendering {total_frames} frames...")

        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",  # Use RGBA for transparency
                "-s",
                f"{self.W}x{self.H}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-i",
                self.input_video,
                "-filter_complex",
                "[1:v][0:v]overlay=0:0",  # overlay respects alpha channel
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                self.output,
            ],
            stdin=subprocess.PIPE,
        )

        current = 0
        block_id = 0
        active_index = -1

        empty = np.zeros((self.H, self.W, 4), dtype=np.uint8)  # RGBA = 4 channels

        for frame in range(total_frames):
            t = frame / self.fps

            while current < len(events) and events[current][0] <= t:
                _, block_id, active_index = events[current]
                current += 1

            if block_id is None or active_index < 0:
                frame_img = self.base_cache.get(block_id, empty)
            else:
                frame_img = self.render(block_id, active_index)

            assert frame_img.shape == (self.H, self.W, 4), frame_img.shape  # RGBA = 4 channels
            assert frame_img.dtype == np.uint8
            assert frame_img.nbytes == self.W * self.H * 4  # 4 bytes per pixel

            frame_img = np.ascontiguousarray(frame_img)

            try:
                ffmpeg.stdin.write(frame_img.tobytes())
            except BrokenPipeError:
                print(f"\nNote: ffmpeg closed stdin at frame {frame}/{total_frames}")
                break

            if frame % (self.fps * 5) == 0:
                print(frame, "/", total_frames)

        ffmpeg.stdin.close()
        ffmpeg.wait()

        print("DONE:", self.output)


# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    r = SubtitleRendererFastV2(
        timestamp_json="word_timestamps.json",
        input_video="input.mp4",
        srt_file="subtitles.srt",
    )
    r.run()