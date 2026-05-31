#!/usr/bin/env python3
"""
Simple Lipsync Tool - Đóng/mở miệng theo âm thanh + Zoom, Tilt, Pan theo sóng âm
Cách dùng:
    python lypsync.py audio.mp3 image_closed.png image_open.png output.mp4
"""

import sys
import numpy as np
import PIL.Image
from PIL import Image, ImageEnhance
import soundfile as sf
import subprocess
from pathlib import Path


# ─── CẤU HÌNH ZOOM ─────────────────────────────────────────────────────────────
ZOOM_ENABLED      = True          # Bật/tắt hiệu ứng zoom theo sóng âm
ZOOM_MODE         = "vertical"   # "vertical" = chỉ dọc, "both" = cả 2 chiều
ZOOM_MIN_SCALE    = 1.0           # Scale nhỏ nhất (không zoom)
ZOOM_MAX_SCALE    = 1.30          # Scale lớn nhất (zoom 30% chiều dọc)
ZOOM_SMOOTHNESS   = 0.25          # Độ mượt của zoom (càng nhỏ càng mượt)
ZOOM_CENTER_X     = 0.5           # Tâm zoom X (0.5 = giữa hình)
ZOOM_CENTER_Y     = 0.5           # Tâm zoom Y (0.5 = giữa chiều dọc)

# ─── CẤU HÌNH TILT / NGHIÊNG ─────────────────────────────────────────────────────
TILT_ENABLED      = True          # Bật/tắt hiệu ứng nghiêng trái/phải
TILT_MAX_ANGLE    = 5             # Góc nghiêng tối đa (độ, dương = nghiêng phải)
TILT_SMOOTHNESS   = 0.2           # Độ mượt của tilt (càng nhỏ càng mượt)

# ─── CẤU HÌNH PAN / DỊCH CHUYỂN TRẬI PHẢI ───────────────────────────────────────
PAN_ENABLED       = True          # Bật/tắt hiệu ứng lật/dịch trái/phải
PAN_MAX_OFFSET    = 30            # Dịch chuyển tối đa (pixel)
PAN_SMOOTHNESS    = 0.15          # Độ mượt của pan (càng nhỏ càng mượt)

# ─── CẤU HÌNH WOBBLE / LẮC ───────────────────────────────────────────────────────
WOBBLE_ENABLED    = True          # Bật/tắt hiệu ứng lắc theo nhịp bass
WOBBLE_STEREO_MODE = True         # True = dùng stereo L/R cho tilt/pan, False = dùng RMS

# ─── CẤU HÌNH LIPSYNC (ĐÓNG/MỞ MIỆNG) ─────────────────────────────────────────────
MOUTH_MODE        = "blend"       # "switch" = đóng/mở đột ngột, "blend" = mix mượt theo amplitude
MOUTH_OPEN_THRESHOLD = 0.02       # Ngưỡng để bắt đầu mở miệng
MOUTH_CLOSE_THRESHOLD = 0.015     # Ngưỡng để đóng miệng (hysteresis - thấp hơn để tránh chớp)
MOUTH_SMOOTHNESS  = 0.3           # Độ mượt của blend (càng nhỏ càng mượt)
MOUTH_MIN_BLEND   = 0.0           # Blend tối thiểu (0 = hoàn toàn đóng)
MOUTH_MAX_BLEND   = 1.0           # Blend tối đa (1 = hoàn toàn mở)


def analyze_audio(audio_path: str, threshold: float = 0.02):
    """
    Phân tích audio và trả về:
    - mouth_blends: list float (0-1, 0=đóng, 1=mở, giá trị giữa = blend)
    - amplitudes: list float (0-1, mức âm lượng cho zoom)
    - left_right_diff: list float (-1 đến 1, chênh lệch L/R cho tilt/pan)

    Args:
        audio_path: Đường dẫn file audio
        threshold: Ngưỡng âm lượng để xác định "có âm" (0.0-1.0)
    """
    # Đọc audio
    audio, sr = sf.read(audio_path)

    # Giữ nguyên stereo nếu có, chuyển mono nếu cần
    is_stereo = len(audio.shape) > 1 and audio.shape[1] >= 2

    if is_stereo:
        audio_left = audio[:, 0]
        audio_right = audio[:, 1]
        audio_mono = audio.mean(axis=1)
    else:
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio_left = audio_right = audio_mono = audio

    # Chuẩn hóa audio về [-1, 1]
    if audio_mono.max() > 0:
        max_val = max(audio_mono.max(), abs(audio_mono.min()))
        audio_mono = audio_mono / max_val
        if is_stereo:
            audio_left = audio_left / max_val
            audio_right = audio_right / max_val

    # Tính RMS energy cho mỗi frame nhỏ (50ms)
    frame_size = int(sr / fps)  # 50ms per frame
    n_frames = len(audio_mono) // frame_size

    raw_amplitudes = []
    raw_left_right = []  # Chênh lệc trái/phải cho tilt/pan
    raw_mouth_blend = []  # Blend value cho miệng (0-1)

    # Biến theo dõi trạng thái miệng (cho hysteresis)
    is_mouth_open = False

    for i in range(n_frames):
        frame_mono = audio_mono[i * frame_size:(i + 1) * frame_size]
        # RMS energy
        energy = np.sqrt(np.mean(frame_mono ** 2))
        raw_amplitudes.append(energy)

        # Xác định trạng thái miệng với hysteresis (tránh chớp nháy)
        if is_mouth_open:
            # Đang mở → cần âm lượng thấp hơn mới đóng
            is_mouth_open = energy > MOUTH_CLOSE_THRESHOLD
        else:
            # Đang đóng → cần âm lượng cao hơn mới mở
            is_mouth_open = energy > MOUTH_OPEN_THRESHOLD

        raw_mouth_blend.append(1.0 if is_mouth_open else 0.0)

        # Tính chênh lệc L/R cho stereo effect
        if is_stereo and WOBBLE_ENABLED:
            frame_left = audio_left[i * frame_size:(i + 1) * frame_size]
            frame_right = audio_right[i * frame_size:(i + 1) * frame_size]
            energy_left = np.sqrt(np.mean(frame_left ** 2)) if len(frame_left) > 0 else 0
            energy_right = np.sqrt(np.mean(frame_right ** 2)) if len(frame_right) > 0 else 0
            # Diff: dương = right mạnh hơn, âm = left mạnh hơn
            diff = (energy_right - energy_left) if (energy_left + energy_right) > 0 else 0
            raw_left_right.append(diff)
        else:
            raw_left_right.append(0)

    # Thêm frames cuối nếu còn dư
    if len(raw_mouth_blend) * frame_size < len(audio_mono):
        raw_amplitudes.append(raw_amplitudes[-1] if raw_amplitudes else 0)
        raw_left_right.append(raw_left_right[-1] if raw_left_right else 0)
        raw_mouth_blend.append(raw_mouth_blend[-1] if raw_mouth_blend else 0)

    # Làm mượt amplitudes cho zoom
    amplitudes = []
    prev_amp = 0
    for amp in raw_amplitudes:
        smoothed = ZOOM_SMOOTHNESS * prev_amp + (1 - ZOOM_SMOOTHNESS) * amp
        amplitudes.append(smoothed)
        prev_amp = smoothed

    # Làm mượt mouth blend (quan trọng - tránh chớp nháy)
    mouth_blends = []
    prev_blend = 0
    for blend in raw_mouth_blend:
        smoothed = MOUTH_SMOOTHNESS * prev_blend + (1 - MOUTH_SMOOTHNESS) * blend
        # Clamp vào [MOUTH_MIN_BLEND, MOUTH_MAX_BLEND]
        clamped = max(MOUTH_MIN_BLEND, min(MOUTH_MAX_BLEND, smoothed))
        mouth_blends.append(clamped)
        prev_blend = smoothed

    # Làm mượt left_right diff cho tilt/pan
    left_right_diff = []
    prev_diff = 0
    for diff in raw_left_right:
        smoothed = 0.2 * prev_diff + 0.8 * diff
        left_right_diff.append(smoothed)
        prev_diff = smoothed

    return mouth_blends, amplitudes, left_right_diff


def apply_zoom(img, scale, center_x=0.5, center_y=0.5):
    """
    Zoom ảnh theo scale factor với tâm zoom tại (center_x, center_y)

    Args:
        img: PIL Image
        scale: Tỷ lệ zoom (1.0 = không zoom, >1 = phóng to)
        center_x, center_y: Vị trí tâm zoom (0-1)
    """
    if scale <= 1.0 or not ZOOM_ENABLED:
        return img

    w, h = img.size

    if ZOOM_MODE == "vertical":
        # Chỉ scale chiều dọc
        new_w = w  # Giữ nguyên chiều ngang
        new_h = int(h * scale)  # Scale chiều dọc

        # Resize (chỉ stretch dọc)
        zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Tính vị trí crop để giữ tâm zoom (chính giữa chiều ngang, theo center_y chiều dọc)
        crop_x1 = 0  # Không crop ngang vì không scale ngang
        crop_y1 = int((new_h - h) * center_y)
        crop_x2 = crop_x1 + w
        crop_y2 = crop_y1 + h

        return zoomed.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    else:  # ZOOM_MODE == "both"
        # Scale cả 2 chiều
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Phóng to ảnh
        zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Tính vị trí crop để giữ tâm zoom
        crop_x1 = int((new_w - w) * center_x)
        crop_y1 = int((new_h - h) * center_y)
        crop_x2 = crop_x1 + w
        crop_y2 = crop_y1 + h

        # Crop về kích thước gốc
        return zoomed.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def apply_tilt(img, angle):
    """
    Nghiêng ảnh theo góc (độ)
    - angle > 0: nghiêng phải (clockwise)
    - angle < 0: nghiêng trái (counter-clockwise)
    """
    if not TILT_ENABLED or abs(angle) < 0.1:
        return img

    w, h = img.size

    # Tính size mới để rotate không bị crop
    angle_rad = abs(angle) * np.pi / 180
    new_w = int(w * np.cos(angle_rad) + h * np.sin(angle_rad)) + 10
    new_h = int(h * np.cos(angle_rad) + w * np.sin(angle_rad)) + 10

    # Tạo canvas lớn hơn
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    offset_x = (new_w - w) // 2
    offset_y = (new_h - h) // 2
    canvas.paste(img, (offset_x, offset_y))

    # Rotate
    rotated = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    # Crop về kích thước gốc (center)
    crop_x1 = (new_w - w) // 2
    crop_y1 = (new_h - h) // 2
    crop_x2 = crop_x1 + w
    crop_y2 = crop_y1 + h

    return rotated.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def apply_pan(img, offset_x, offset_y=0):
    """
    Dịch chuyển ảnh (pan) theo offset pixel
    - offset_x > 0: dịch phải
    - offset_x < 0: dịch trái
    """
    if not PAN_ENABLED or abs(offset_x) < 1:
        return img

    w, h = img.size

    # Tạo canvas lớn hơn với màu đen
    pad = int(abs(offset_x)) + 5
    new_w = w + 2 * pad
    new_h = h + 2 * pad
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))

    # Paste ảnh vào canvas với offset
    paste_x = pad - int(offset_x)
    paste_y = pad - int(offset_y)
    canvas.paste(img, (paste_x, paste_y))

    # Crop về kích thước gốc
    crop_x1 = pad
    crop_y1 = pad
    crop_x2 = crop_x1 + w
    crop_y2 = crop_y1 + h

    return canvas.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def blend_images(img_closed, img_open, blend_factor):
    """
    Blend 2 hình theo blend_factor (0.0 = toàn đóng, 1.0 = toàn mở)

    Args:
        img_closed: Hình miệng đóng
        img_open: Hình miệng mở
        blend_factor: 0-1, mức blend giữa 2 hình
    """
    if blend_factor <= 0:
        return img_closed
    elif blend_factor >= 1:
        return img_open

    # Chuyển thành numpy arrays để blending nhanh hơn
    arr_closed = np.array(img_closed).astype(np.float32)
    arr_open = np.array(img_open).astype(np.float32)

    # Linear blend: result = closed * (1 - alpha) + open * alpha
    arr_result = arr_closed * (1 - blend_factor) + arr_open * blend_factor

    return Image.fromarray(arr_result.astype(np.uint8))


def create_lipsync_video(
    audio_path: str,
    image_closed: str,
    image_open: str,
    output_path: str,
    fps: int = 20,
    threshold: float = 0.02
):
    """
    Tạo video lipsync từ audio và 2 hình miệng đóng/mở
    Có hiệu ứng zoom, tilt, pan theo sóng âm
    Blend mượt giữa đóng/mở miệng (không chớp nháy)
    """
    print(f"🎵 Phân tích audio: {audio_path}")
    mouth_blends, amplitudes, left_right_diff = analyze_audio(audio_path, threshold)

    # Tính % frames mở miệng (blend > 0.5)
    open_frames = sum(1 for b in mouth_blends if b > 0.5)
    print(f"   → {len(mouth_blends)} frames, {open_frames} frames mở miệng ({open_frames/len(mouth_blends)*100:.1f}%)")

    if ZOOM_ENABLED:
        max_amp = max(amplitudes) if amplitudes else 1
        print(f"   📊 Zoom: amplitude max={max_amp:.4f} (scale: {ZOOM_MIN_SCALE} → {ZOOM_MAX_SCALE})")

    if TILT_ENABLED:
        max_diff = max(abs(d) for d in left_right_diff) if left_right_diff else 1
        print(f"   ↻️  Tilt: L/R diff max={max_diff:.4f} (angle: ±{TILT_MAX_ANGLE}°)")

    if PAN_ENABLED:
        print(f"   ↔️  Pan: offset max=±{PAN_MAX_OFFSET}px")

    print(f"   👄 Lipsync: smooth blend (threshold: {MOUTH_CLOSE_THRESHOLD} → {MOUTH_OPEN_THRESHOLD})")

    # Đọc 2 hình
    img_closed = Image.open(image_closed).convert("RGB")
    img_open = Image.open(image_open).convert("RGB")

    # Đảm bảo 2 hình cùng size
    if img_closed.size != img_open.size:
        print(f"⚠️ 2 hình khác size: {img_closed.size} vs {img_open.size}")
        print(f"   Resize {img_open.size} → {img_closed.size}")
        img_open = img_open.resize(img_closed.size)

    # H.264 yêu cầu width, height chia hết cho 2
    width, height = img_closed.size
    if width % 2 != 0 or height % 2 != 0:
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1
        print(f"⚠️ Size {width}x{height} không chia hết cho 2 → resize {new_width}x{new_height}")
        img_closed = img_closed.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_open = img_open.resize((new_width, new_height), Image.Resampling.LANCZOS)

    width, height = img_closed.size

    # Tạo thư mục tạm cho frames
    temp_dir = Path("/tmp/lipsync_frames")
    temp_dir.mkdir(exist_ok=True)

    # Chuẩn hóa amplitudes và left_right diff
    max_amp = max(amplitudes) if amplitudes else 1
    if max_amp == 0:
        max_amp = 1

    max_diff = max(abs(d) for d in left_right_diff) if left_right_diff else 1
    if max_diff == 0:
        max_diff = 1

    print(f"🖼️  Tạo {len(mouth_blends)} frames...")
    for i, blend_factor in enumerate(mouth_blends):
        # ── Blend 2 hình miệng đóng/mở MỜT (không chớp nháy) ───────────────────────
        img = blend_images(img_closed, img_open, blend_factor)

        # ── Áp dụng ZOOM theo amplitude ────────────────────────────────────────
        if ZOOM_ENABLED:
            norm_amp = amplitudes[i] / max_amp
            scale = ZOOM_MIN_SCALE + (ZOOM_MAX_SCALE - ZOOM_MIN_SCALE) * norm_amp
            img = apply_zoom(img, scale, ZOOM_CENTER_X, ZOOM_CENTER_Y)

        # ── Áp dụng TILT (nghiêng) theo chênh lệch L/R ────────────────────────────
        if TILT_ENABLED:
            norm_diff = left_right_diff[i] / max_diff
            # norm_diff: -1 (left mạnh) → +1 (right mạnh)
            angle = norm_diff * TILT_MAX_ANGLE
            img = apply_tilt(img, angle)

        # ── Áp dụng PAN (dịch chuyển) theo chênh lệch L/R ───────────────────────────
        if PAN_ENABLED:
            norm_diff = left_right_diff[i] / max_diff
            # norm_diff: -1 (left) → +1 (right)
            offset_x = norm_diff * PAN_MAX_OFFSET
            img = apply_pan(img, offset_x)

        frame_path = temp_dir / f"frame_{i:05d}.png"
        img.save(frame_path)

        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(mouth_blends)}")

    # Dùng ffmpeg để tạo video
    print(f"🎬 Ghép video với ffmpeg...")
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-framerate", str(fps),
        "-i", str(temp_dir / "frame_%05d.png"),
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",  # Video dài bằng audio
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg lỗi:\n{result.stderr}")
        return False

    print(f"✅ Xong! Video saved: {output_path}")

    # Dọn dẹp
    subprocess.run(["rm", "-rf", temp_dir])
    return True


if __name__ == "__main__":
   

    audio_path = "output.wav"
    image_closed = "mouth_close.png"
    image_open = "mouth_open.png"
    output_path = "output_lipsync.mp4"
    fps = 30
    threshold =  0.02

    create_lipsync_video(audio_path, image_closed, image_open, output_path, fps, threshold)
