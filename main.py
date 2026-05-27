import subprocess
from whisper_word_timestamper import WhisperWordTimestamper
from subtitle_video_renderer import SubtitleVideoRenderer

def merge_intro(intro_video="media/intro_video.mp4", subtitles_video="outputs/subtitles_video.mp4", output="outputs/easy_english_podcast.mp4"):

    cmd = [
        "ffmpeg",
        "-y",

        "-i", intro_video,
        "-i", subtitles_video,

        "-filter_complex",
        (
            # scale cả 2 video về 1920x1080
            "[0:v]scale=1920:1080,fps=30[v0];"
            "[1:v]scale=1920:1080,fps=30[v1];"

            # concat
            "[v0][0:a][v1][1:a]"
            "concat=n=2:v=1:a=1[v][a]"
        ),

        "-map", "[v]",
        "-map", "[a]",

        "-c:v", "libx264",
        "-c:a", "aac",

        output
    ]

    subprocess.run(cmd)

    print(f"Done: {output}")

#!/usr/bin/env python3
"""
Tạo video background đen 16:9 với sóng âm từ file audio.
Có 2 nhân vật: brian.png (góc dưới trái), tracy.png (góc dưới phải).
Logo: logo.png (góc trên phải).

Cách dùng:
    python main.py

Yêu cầu:
    pip install Pillow
    ffmpeg cài sẵn trong hệ thống
"""

import os
import subprocess
import sys
import tempfile
import struct
import wave

# ─── CẤU HÌNH – chỉnh ở đây nếu muốn ───────────────────────────────────────
WIDTH             = 1920          # độ rộng video
HEIGHT            = 1080          # độ cao video
FPS               = 30            # frames per second
BAR_COLOR         = "white"       # white | red | blue | green | #RRGGBB
NUM_BARS          = 80            # số cột bars
BAR_WIDTH_RATIO   = 0.55          # độ dày cột (0–1)
WAVE_HEIGHT_RATIO = 0.12          # chiều cao tối đa bars so với height (0–1)
WAVE_WIDTH_RATIO  = 0.35          # độ rộng vùng sóng âm so với width (0–1)
BOTTOM_MARGIN     = 80            # khoảng cách từ đáy bars tới mép dưới (px)
BG_COLOR          = (10, 14, 20)  # màu nền fallback nếu không có background.png
BG_IMG            = "media/background.png"  # ảnh nền (để None nếu muốn dùng màu nền)

# ─── NHÂN VẬT ────────────────────────────────────────────────────────────────
BRIAN_IMG         = "media/brian.png"   # nhân vật góc dưới trái
TRACY_IMG         = "media/tracy.png"   # nhân vật góc dưới phải
CHAR_HEIGHT       = 600           # chiều cao nhân vật (px), tự scale width theo tỉ lệ
CHAR_MARGIN_X     = 30            # khoảng cách từ nhân vật tới mép trái/phải (px)
CHAR_MARGIN_Y     = 0             # khoảng cách từ chân nhân vật tới mép dưới (px)

# ─── LOGO ────────────────────────────────────────────────────────────────────
LOGO_IMG          = "media/logo.png"    # logo góc trên phải
LOGO_HEIGHT       = 200           # chiều cao logo (px), tự scale width theo tỉ lệ
LOGO_MARGIN_X     = 40            # khoảng cách từ logo tới mép phải (px)
LOGO_MARGIN_Y     = 30            # khoảng cách từ logo tới mép trên (px)
# ────────────────────────────────────────────────────────────────────────────


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg chưa được cài.")
        print("   Ubuntu/Debian : sudo apt install ffmpeg")
        print("   macOS         : brew install ffmpeg")
        print("   Windows       : https://ffmpeg.org/download.html")
        sys.exit(1)


def check_pillow():
    try:
        from PIL import Image, ImageDraw
        return Image, ImageDraw
    except ImportError:
        print("❌ Thiếu Pillow. Chạy: pip install Pillow")
        sys.exit(1)


def parse_color(name):
    table = {
        "white": (255, 255, 255),
        "red":   (255,  60,  60),
        "blue":  ( 60, 140, 255),
        "green": ( 60, 220, 100),
    }
    if name.lower() in table:
        return table[name.lower()]
    if name.startswith("#"):
        h = name.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def extract_audio(audio_path):
    """Convert audio → mono 16-bit PCM, trả về (samples, framerate, duration)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", tmp_wav],
            capture_output=True, check=True,
        )
        with wave.open(tmp_wav, "rb") as wf:
            n_frames  = wf.getnframes()
            framerate = wf.getframerate()
            raw       = wf.readframes(n_frames)
        samples  = list(struct.unpack(f"<{n_frames}h", raw))
        duration = n_frames / framerate
        return samples, framerate, duration
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def load_character(PIL_Image, path, target_height):
    """Load PNG nhân vật, scale theo chiều cao, giữ nguyên alpha channel."""
    if not os.path.isfile(path):
        print(f"⚠️  Không tìm thấy {path} – bỏ qua nhân vật này.")
        return None
    img = PIL_Image.open(path).convert("RGBA")
    w, h = img.size
    scale = target_height / h
    new_w = int(w * scale)
    img = img.resize((new_w, target_height), PIL_Image.LANCZOS)
    print(f"   Loaded {path}  →  {new_w}×{target_height}px")
    return img


def load_logo(PIL_Image, path, target_height):
    """Load PNG logo, scale theo chiều cao, giữ nguyên alpha channel."""
    if not os.path.isfile(path):
        print(f"⚠️  Không tìm thấy {path} – bỏ qua logo.")
        return None
    img = PIL_Image.open(path).convert("RGBA")
    w, h = img.size
    scale = target_height / h
    new_w = int(w * scale)
    img = img.resize((new_w, target_height), PIL_Image.LANCZOS)
    print(f"   Loaded {path}  →  {new_w}×{target_height}px")
    return img


def load_background(PIL_Image, path):
    """Load ảnh nền, resize về đúng WIDTH×HEIGHT."""
    if not path or not os.path.isfile(path):
        print(f"⚠️  Không tìm thấy {path} – dùng màu nền mặc định.")
        return None
    img = PIL_Image.open(path).convert("RGB")
    img = img.resize((WIDTH, HEIGHT), PIL_Image.LANCZOS)
    print(f"   Loaded {path}  →  {WIDTH}×{HEIGHT}px")
    return img


def build_base_frame(PIL_Image, bg_img, brian, tracy, brian_x, brian_y, tracy_x, tracy_y, logo, logo_x, logo_y):
    """Tạo sẵn 1 frame RGB có background + 2 nhân vật + logo.
    Chỉ gọi 1 lần trước vòng lặp — mỗi frame chỉ cần .copy() từ cái này."""
    if bg_img:
        base = bg_img.copy().convert("RGBA")
    else:
        base = PIL_Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR + (255,))
    if brian:
        base.paste(brian, (brian_x, brian_y), brian)
    if tracy:
        base.paste(tracy, (tracy_x, tracy_y), tracy)
    if logo:
        base.paste(logo, (logo_x, logo_y), logo)
    return base.convert("RGB")


def render(audio_path, output_path):
    # ── Import Pillow trước tiên ───────────────────────────────────────────────
    PIL_Image, PIL_ImageDraw = check_pillow()
    check_ffmpeg()

    print(f"📂 Audio   : {audio_path}")
    samples, framerate, duration = extract_audio(audio_path)
    print(f"   Duration : {duration:.2f}s  |  {framerate}Hz  |  {len(samples):,} samples")

    # ── Load background ────────────────────────────────────────────────────────
    print("🖼️  Load background...")
    bg_img = load_background(PIL_Image, BG_IMG)

    # ── Load nhân vật ─────────────────────────────────────────────────────────
    print("🖼️  Load nhân vật...")
    brian = load_character(PIL_Image, BRIAN_IMG, CHAR_HEIGHT)
    tracy = load_character(PIL_Image, TRACY_IMG, CHAR_HEIGHT)

    brian_x = WIDTH - (brian.size[0] if brian else 0)
    brian_y = HEIGHT - CHAR_HEIGHT + 10
    tracy_x = -250
    tracy_y = HEIGHT - CHAR_HEIGHT + 10

    # ── Load logo ─────────────────────────────────────────────────────────────
    print("🖼️  Load logo...")
    logo = load_logo(PIL_Image, LOGO_IMG, LOGO_HEIGHT)
    if logo:
        logo_x = WIDTH - logo.size[0] - LOGO_MARGIN_X
        logo_y = LOGO_MARGIN_Y
    else:
        logo_x = logo_y = 0

    # ── Pre-composite 1 lần: BG + nhân vật + logo → base_frame cố định ───────
    print("🖼️  Pre-composite background + nhân vật + logo...")
    base_frame = build_base_frame(
        PIL_Image,
        bg_img,
        brian, tracy, brian_x, brian_y, tracy_x, tracy_y,
        logo, logo_x, logo_y,
    )

    total_frames      = int(duration * FPS)
    samples_per_frame = len(samples) / total_frames
    color             = parse_color(BAR_COLOR)
    max_bar_h         = int(HEIGHT * WAVE_HEIGHT_RATIO)
    wave_width        = int(WIDTH * WAVE_WIDTH_RATIO)
    wave_x_start      = (WIDTH - wave_width) // 2
    bar_slot          = wave_width / NUM_BARS
    bar_w             = max(1, int(bar_slot * BAR_WIDTH_RATIO))
    bar_gap           = bar_slot - bar_w
    base_y            = HEIGHT - BOTTOM_MARGIN

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    # ── Tính global max RMS ───────────────────────────────────────────────────
    print("⏳ Phân tích audio...")
    window_samples = max(int(samples_per_frame * 3), 1)
    global_max = 1.0
    for idx in range(total_frames):
        s0 = int(idx * samples_per_frame)
        s1 = min(s0 + window_samples, len(samples))
        seg = samples[s0:s1]
        if seg:
            chunk = max(1, len(seg) // NUM_BARS)
            for i in range(NUM_BARS):
                s = seg[i * chunk:(i + 1) * chunk]
                if s:
                    v = (sum(x * x for x in s) / len(s)) ** 0.5
                    if v > global_max:
                        global_max = v

    SMOOTH    = 0.4
    prev_norm = [0.0] * NUM_BARS

    print(f"🎬 Render  : {total_frames} frames  ({WIDTH}x{HEIGHT} @ {FPS}fps)")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for idx in range(total_frames):
            s0  = int(idx * samples_per_frame)
            s1  = min(s0 + window_samples, len(samples))
            seg = samples[s0:s1] if s1 <= len(samples) else samples[s0:]

            chunk = max(1, len(seg) // NUM_BARS)
            rms   = []
            for i in range(NUM_BARS):
                s = seg[i * chunk:(i + 1) * chunk]
                rms.append((sum(x * x for x in s) / len(s)) ** 0.5 if s else 0)

            raw  = [(v / global_max) ** 0.7 for v in rms]
            norm = [SMOOTH * prev_norm[i] + (1 - SMOOTH) * raw[i] for i in range(NUM_BARS)]
            prev_norm = norm

            # ── Vẽ frame: copy base_frame rồi vẽ bars lên trên ───────────────
            img  = base_frame.copy()
            draw = PIL_ImageDraw.Draw(img)

            for i, amp in enumerate(norm):
                bh = max(2, int(amp * max_bar_h))
                x0 = wave_x_start + int(i * bar_slot + bar_gap / 2)
                x1 = x0 + bar_w
                y1 = base_y
                y0 = base_y - bh
                if bh >= 4:
                    draw.rounded_rectangle([x0, y0, x1, y1], radius=bar_w // 2, fill=color)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=color)

            proc.stdin.write(img.tobytes())

            if idx % max(1, total_frames // 20) == 0:
                print(f"   {idx/total_frames*100:5.1f}%  ({idx}/{total_frames})", end="\r")

        proc.stdin.close()
        proc.wait()

    except BrokenPipeError:
        print("\n❌ ffmpeg bị lỗi:\n" + proc.stderr.read().decode())
        sys.exit(1)

    if proc.returncode != 0:
        print("\n❌ ffmpeg thất bại:\n" + proc.stderr.read().decode())
        sys.exit(1)

    print(f"\n✅ Xong! Video: {output_path}")


if __name__ == "__main__":
    run_dir = Path("source/run_20260525_085330")
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = run_dir / "full_audio.wav"
    subtitle_path = run_dir / "subtitles.srt"
    intro_video = "media/intro_video.mp4"
    output_wave_video = run_dir / "outputs/lambda_video.mp4"
    output_subtitles_video = run_dir / "outputs/subtitles_video.mp4"
    output_timestamp_json = run_dir / "outputs/word_timestamps.json"
    output_final_video = run_dir / "outputs/easy_english_podcast_topic_learning_english_changed_our_lives_forever.mp4"

    # create video sóng âm từ audio
    render(audio, output_wave_video)

    # get wording timestamp
    timestamper = WhisperWordTimestamper(
        chunk_dir=run_dir,
        srt_file=subtitle_path,
        device="cpu",
        language="en",
    )
    timestamper.run(output_timestamp_json)

    # add subtitle into video sóng âm
    renderer = SubtitleVideoRenderer(
        timestamp_json=output_timestamp_json,
        input_video=output_wave_video,
        srt_file=subtitle_path,
        output=output_subtitles_video
    )
    renderer.run()

    # merge video intro and subtitle video
    merge_intro(intro_video, output_subtitles_video, output_final_video)
