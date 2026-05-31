import os
import subprocess
import sys
import tempfile
import struct
import wave
from pathlib import Path

import math
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WIDTH             = 1920
HEIGHT            = 1080
FPS               = 30
BAR_COLOR         = "white"
NUM_BARS          = 80
BAR_WIDTH_RATIO   = 0.55
WAVE_HEIGHT_RATIO = 0.12
WAVE_WIDTH_RATIO  = 0.35
BOTTOM_MARGIN     = 80
BG_IMG            = "media/background.png"

# Characters
BRIAN_IMG         = "media/brian.png"
CHAR_HEIGHT       = 600
CHAR_MARGIN_X     = 30

# Logo
LOGO_IMG          = "media/logo.png"
LOGO_HEIGHT       = 200
LOGO_MARGIN_X     = 40
LOGO_MARGIN_Y     = 30

# Lipsync - Mouth blend
MOUTH_OPEN_THRESHOLD  = 0.02
MOUTH_CLOSE_THRESHOLD = 0.015
MOUTH_SMOOTHNESS      = 0.3
MOUTH_MIN_BLEND       = 0.0
MOUTH_MAX_BLEND       = 1.0

# Lipsync - Zoom
ZOOM_ENABLED      = True
ZOOM_MIN_SCALE    = 1.0
ZOOM_MAX_SCALE    = 1.30
ZOOM_SMOOTHNESS   = 0.25
ZOOM_CENTER_Y     = 0.5  # vertical only

# Lipsync - Tilt
TILT_ENABLED          = True
TILT_OPEN_THRESHOLD   = 0.03
TILT_CLOSE_THRESHOLD  = 0.02
TILT_MAX_ANGLE        = 75
TILT_SMOOTHNESS       = 0.3
TILT_LEFT_PHASE       = 4
TILT_RIGHT_PHASE      = 4
TILT_CENTER_PHASE     = 4

# Lipsync - Mirror
MIRROR_ENABLED      = True
MIRROR_HOLD_SECONDS = 1.0
MIRROR_INTERVAL     = 2.0
# ─────────────────────────────────────────────────────────────────────────────


# ─── UTILS ───────────────────────────────────────────────────────────────────

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg chưa được cài.")
        sys.exit(1)


def parse_color(name):
    table = {"white": (255,255,255), "red": (255,60,60), "blue": (60,140,255), "green": (60,220,100)}
    if name.lower() in table:
        return table[name.lower()]
    h = name.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def extract_audio_pcm(audio_path):
    """Convert audio → mono 16-bit PCM. Returns (samples, framerate, duration)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", tmp_wav],
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
        os.remove(tmp_wav)


def load_image(path, target_height, label="image"):
    """Load RGBA image scaled to target_height, preserving aspect ratio."""
    if not os.path.isfile(path):
        print(f"⚠️  {label} không tìm thấy: {path}")
        return None
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    new_w = int(w * target_height / h)
    img = img.resize((new_w, target_height), Image.LANCZOS)
    print(f"   {label}: {path} → {new_w}×{target_height}px")
    return img


def fix_alpha(img):
    """Set any pixel with alpha>0 to fully opaque (fixes semi-transparent artifacts)."""
    arr = np.array(img)
    mask = arr[:, :, 3] > 0
    arr[mask, 3] = 255
    return Image.fromarray(arr)


# ─── LIPSYNC EFFECTS ─────────────────────────────────────────────────────────

def blend_images(img_closed, img_open, blend_factor):
    if blend_factor <= 0: return img_closed
    if blend_factor >= 1: return img_open
    a = np.array(img_closed, dtype=np.float32)
    b = np.array(img_open,   dtype=np.float32)
    return Image.fromarray((a * (1 - blend_factor) + b * blend_factor).astype(np.uint8))


def apply_zoom_vertical(img, scale):
    """Zoom vertically only (width stays fixed)."""
    if scale <= 1.0: return img
    w, h = img.size
    new_h = int(h * scale)
    zoomed = img.resize((w, new_h), Image.LANCZOS)
    crop_y = int((new_h - h) * ZOOM_CENTER_Y)
    return zoomed.crop((0, crop_y, w, crop_y + h))


def apply_tilt(img, angle):
    """
    Xoay ảnh quanh bottom-center. Trả về (rotated_img, offset_x).
    - rotated_img: RGBA, rộng hơn gốc, không bị crop góc
    - offset_x: số px dịch sang trái khi paste để chân cố định vị trí
    """
    if abs(angle) < 0.1:
        return img, 0
    w, h = img.size
    rad = math.radians(abs(angle))

    # Tính padding chính xác dựa trên hình học xoay quanh bottom-center
    max_dx = h * math.sin(rad) + w / 2 * (1 - math.cos(rad))
    max_dy = h * (1 - math.cos(rad)) + w / 2 * math.sin(rad)
    pad_x  = int(max_dx) + 10
    pad_y  = int(max_dy) + 10

    canvas_w = w + 2 * pad_x
    canvas_h = h + pad_y

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(img, (pad_x, pad_y), img)

    # Pivot = bottom-center của ảnh trong canvas
    cx = pad_x + w // 2
    cy = pad_y + h

    rotated = canvas.rotate(angle, resample=Image.BICUBIC, expand=False, center=(cx, cy))
    # Giữ toàn bộ chiều ngang, chiều cao = h, bottom cố định
    result  = rotated.crop((0, cy - h, canvas_w, cy))
    return result, -pad_x


def analyze_audio(audio_path):
    """
    Analyze audio and return per-frame blend arrays:
    - mouth_blends: 0=closed, 1=open
    - amplitudes: normalized RMS energy
    - tilt_blends: 0=straight, 1=tilted
    """
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if audio.max() > 0:
        audio = audio / max(audio.max(), abs(audio.min()))

    frame_size = int(sr / FPS)
    n_frames   = len(audio) // frame_size

    raw_amp   = []
    raw_mouth = []
    raw_tilt  = []
    mouth_open = tilt_on = False

    for i in range(n_frames):
        energy = float(np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2)))
        raw_amp.append(energy)

        mouth_open = energy > (MOUTH_CLOSE_THRESHOLD if mouth_open else MOUTH_OPEN_THRESHOLD)
        raw_mouth.append(1.0 if mouth_open else 0.0)

        tilt_on = energy > (TILT_CLOSE_THRESHOLD if tilt_on else TILT_OPEN_THRESHOLD)
        raw_tilt.append(1.0 if tilt_on else 0.0)

    def smooth(raw, alpha):
        out, prev = [], 0.0
        for v in raw:
            prev = alpha * prev + (1 - alpha) * v
            out.append(prev)
        return out

    amplitudes   = smooth(raw_amp,   ZOOM_SMOOTHNESS)
    mouth_blends = [max(MOUTH_MIN_BLEND, min(MOUTH_MAX_BLEND, v)) for v in smooth(raw_mouth, MOUTH_SMOOTHNESS)]
    tilt_blends  = [max(0.0, min(1.0, v)) for v in smooth(raw_tilt, TILT_SMOOTHNESS)]

    return mouth_blends, amplitudes, tilt_blends


# ─── LIPSYNC FRAME GENERATOR ─────────────────────────────────────────────────

class LipsyncState:
    def __init__(self, fps):
        self.tilt_state       = "idle"
        self.tilt_frame_count = 0
        self.tilt_angle       = 0.0
        self.tilt_offset_x    = 0
        self.mirror_flipped   = False
        self.mirror_flip_at   = -int(MIRROR_INTERVAL * fps)
        self.hold_frames      = int(MIRROR_HOLD_SECONDS * fps)
        self.interval_frames  = int(MIRROR_INTERVAL * fps)


def get_tracy_frame(frame_idx, mouth_blends, amplitudes, tilt_blends, max_amp,
                    mouth_closed, mouth_open, target_w, target_h, state):
    # 1. Mouth blend
    blend = mouth_blends[frame_idx] if frame_idx < len(mouth_blends) else 0.0
    img = blend_images(mouth_closed, mouth_open, blend)

    # 2. Vertical zoom
    if ZOOM_ENABLED and frame_idx < len(amplitudes):
        norm  = amplitudes[frame_idx] / max_amp if max_amp > 0 else 0
        scale = ZOOM_MIN_SCALE + (ZOOM_MAX_SCALE - ZOOM_MIN_SCALE) * norm
        img = apply_zoom_vertical(img, scale)

    # # 3. Tilt (left → right → center rocking)
    # if TILT_ENABLED and frame_idx < len(tilt_blends):
    #     should = tilt_blends[frame_idx] > 0.5
    #     s = state

    #     if s.tilt_state == "idle":
    #         if should:
    #             s.tilt_state, s.tilt_frame_count = "left", 0
    #         s.tilt_angle = 0.0
    #     elif s.tilt_state == "left":
    #         s.tilt_angle = -TILT_MAX_ANGLE * (s.tilt_frame_count / TILT_LEFT_PHASE)
    #         s.tilt_frame_count += 1
    #         if s.tilt_frame_count >= TILT_LEFT_PHASE:
    #             s.tilt_state, s.tilt_frame_count = "right", 0
    #     elif s.tilt_state == "right":
    #         progress = s.tilt_frame_count / TILT_RIGHT_PHASE
    #         s.tilt_angle = -TILT_MAX_ANGLE + 2 * TILT_MAX_ANGLE * progress
    #         s.tilt_frame_count += 1
    #         if s.tilt_frame_count >= TILT_RIGHT_PHASE:
    #             s.tilt_state, s.tilt_frame_count = "center", 0
    #     elif s.tilt_state == "center":
    #         s.tilt_angle = TILT_MAX_ANGLE * (1 - s.tilt_frame_count / TILT_CENTER_PHASE)
    #         s.tilt_frame_count += 1
    #         if s.tilt_frame_count >= TILT_CENTER_PHASE:
    #             s.tilt_state = "idle"
    #             s.tilt_angle = 0.0

    #     img, tilt_offset_x = apply_tilt(img, s.tilt_angle)
    #     state.tilt_offset_x = tilt_offset_x

    # else:
    #     state.tilt_offset_x = 0

    # 4. Mirror flip toggle
    if MIRROR_ENABLED:
        elapsed = frame_idx - state.mirror_flip_at
        if not state.mirror_flipped and elapsed >= state.interval_frames:
            state.mirror_flipped, state.mirror_flip_at = True, frame_idx
        elif state.mirror_flipped and elapsed >= state.hold_frames:
            state.mirror_flipped, state.mirror_flip_at = False, frame_idx
        if state.mirror_flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 5. Scale to final size (chỉ scale height, width tự theo aspect ratio)
    w_img, h_img = img.size
    new_w = int(w_img * target_h / h_img)
    return img.resize((new_w, target_h), Image.LANCZOS), state.tilt_offset_x


# ─── MAIN RENDER ─────────────────────────────────────────────────────────────

def render(audio_path, output_path):
    check_ffmpeg()

    print(f"📂 Audio: {audio_path}")
    samples, framerate, duration = extract_audio_pcm(str(audio_path))
    print(f"   {duration:.2f}s  |  {framerate}Hz  |  {len(samples):,} samples")

    # Load assets
    print("🖼️  Loading assets...")
    bg = Image.open(BG_IMG).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS) if os.path.isfile(BG_IMG) else None
    brian = load_image(BRIAN_IMG, CHAR_HEIGHT, "Brian")
    logo  = load_image(LOGO_IMG,  LOGO_HEIGHT, "Logo")

    mouth_closed = fix_alpha(Image.open("mouth_close.png").convert("RGBA"))
    mouth_open   = fix_alpha(Image.open("mouth_open.png").convert("RGBA"))
    if mouth_closed.size != mouth_open.size:
        mouth_open = mouth_open.resize(mouth_closed.size, Image.LANCZOS)

    # Analyze audio
    print("🎵 Analyzing audio...")
    mouth_blends, amplitudes, tilt_blends = analyze_audio(str(audio_path))
    max_amp = max(amplitudes) if amplitudes else 1.0

    # Layout positions
    brian_x = WIDTH - (brian.size[0] if brian else 0)
    brian_y = HEIGHT - CHAR_HEIGHT + 10
    logo_x  = WIDTH - (logo.size[0] if logo else 0) - LOGO_MARGIN_X
    logo_y  = LOGO_MARGIN_Y
    tracy_w = int(mouth_closed.size[0] * CHAR_HEIGHT / mouth_closed.size[1])
    tracy_x, tracy_y = 0, HEIGHT - CHAR_HEIGHT + 10

    # Pre-composite static background (BG + Brian + Logo)
    base = (bg.copy() if bg else Image.new("RGB", (WIDTH, HEIGHT), (10, 14, 20))).convert("RGBA")
    if brian: base.paste(brian, (brian_x, brian_y), brian)
    if logo:  base.paste(logo,  (logo_x,  logo_y),  logo)
    base = base.convert("RGB")

    # Waveform config
    total_frames      = int(duration * FPS)
    samples_per_frame = len(samples) / total_frames
    color             = parse_color(BAR_COLOR)
    max_bar_h         = int(HEIGHT * WAVE_HEIGHT_RATIO)
    wave_w            = int(WIDTH * WAVE_WIDTH_RATIO)
    wave_x0           = (WIDTH - wave_w) // 2
    bar_slot          = wave_w / NUM_BARS
    bar_w             = max(1, int(bar_slot * BAR_WIDTH_RATIO))
    bar_gap           = bar_slot - bar_w
    base_y            = HEIGHT - BOTTOM_MARGIN
    window            = max(int(samples_per_frame * 3), 1)

    # Pre-scan global max RMS for normalization
    print("⏳ Pre-scanning audio levels...")
    global_max = 1.0
    for idx in range(total_frames):
        s0 = int(idx * samples_per_frame)
        seg = samples[s0:min(s0 + window, len(samples))]
        if seg:
            chunk = max(1, len(seg) // NUM_BARS)
            for i in range(NUM_BARS):
                s = seg[i*chunk:(i+1)*chunk]
                if s:
                    v = (sum(x*x for x in s) / len(s)) ** 0.5
                    if v > global_max:
                        global_max = v

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(output_path),
    ]

    lipsync = LipsyncState(FPS)
    SMOOTH    = 0.4
    prev_norm = [0.0] * NUM_BARS

    print(f"🎬 Rendering {total_frames} frames ({WIDTH}x{HEIGHT} @ {FPS}fps)...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for idx in range(total_frames):
            s0  = int(idx * samples_per_frame)
            seg = samples[s0:min(s0 + window, len(samples))]
            chunk = max(1, len(seg) // NUM_BARS)

            rms  = [(sum(x*x for x in seg[i*chunk:(i+1)*chunk]) / max(1, len(seg[i*chunk:(i+1)*chunk]))) ** 0.5
                    for i in range(NUM_BARS)]
            raw  = [(v / global_max) ** 0.7 for v in rms]
            norm = [SMOOTH * prev_norm[i] + (1-SMOOTH) * raw[i] for i in range(NUM_BARS)]
            prev_norm = norm

            # Composite frame
            frame = base.copy().convert("RGBA")
            draw  = ImageDraw.Draw(frame)

            # Draw waveform bars
            for i, amp in enumerate(norm):
                bh = max(2, int(amp * max_bar_h))
                x0 = wave_x0 + int(i * bar_slot + bar_gap / 2)
                y0 = base_y - bh
                x1 = x0 + bar_w
                if bh >= 4:
                    draw.rounded_rectangle([x0, y0, x1, base_y], radius=bar_w//2, fill=color)
                else:
                    draw.rectangle([x0, y0, x1, base_y], fill=color)

            # Generate & paste Tracy với alpha mask
            tracy, tilt_off = get_tracy_frame(idx, mouth_blends, amplitudes, tilt_blends, max_amp,
                                              mouth_closed, mouth_open, tracy_w, CHAR_HEIGHT, lipsync)
            frame.paste(tracy, (tracy_x + tilt_off, tracy_y), tracy)

            proc.stdin.write(frame.convert("RGB").tobytes())

            if idx % max(1, total_frames // 20) == 0:
                print(f"   {idx/total_frames*100:5.1f}%  ({idx}/{total_frames})", end="\r")

    except BrokenPipeError:
        print("\n❌ ffmpeg error:\n" + proc.stderr.read().decode())
        sys.exit(1)
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        print("\n❌ ffmpeg failed:\n" + proc.stderr.read().decode())
        sys.exit(1)

    print(f"\n✅ Done: {output_path}")


def merge_intro(intro_video, subtitles_video, output):
    cmd = [
        "ffmpeg", "-y",
        "-i", intro_video,
        "-i", subtitles_video,
        "-filter_complex",
        "[0:v]scale=1920:1080,fps=30[v0];"
        "[1:v]scale=1920:1080,fps=30[v1];"
        "[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-c:a", "aac",
        output,
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Merged: {output}")

if __name__ == "__main__":
    run_dir = Path("source/run_20260528_061817_copy")
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
    # timestamper = WhisperWordTimestamper(
    #     chunk_dir=run_dir,
    #     srt_file=subtitle_path,
    #     device="cpu",
    #     language="en",
    # )
    # timestamper.run(output_timestamp_json)

    # add subtitle into video sóng âm
    # renderer = SubtitleVideoRenderer(
    #     timestamp_json=output_timestamp_json,
    #     input_video=output_wave_video,
    #     srt_file=subtitle_path,
    #     output=output_subtitles_video
    # )
    # renderer.run()

    # merge video intro and subtitle video
    # merge_intro(intro_video, output_subtitles_video, output_final_video)
