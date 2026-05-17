import json
import random
from pathlib import Path

import soundfile as sf
from pydub import AudioSegment
from f5_tts.api import F5TTS

# =========================
# CONFIG
# =========================

SPEAKER_VOICE = {
    "Tracy": "girl",
    "Brian": "boy",
}

VOICE_DIR = Path("emotion_voice")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_DBFS = -18.0

# =========================
# LOAD DATA
# =========================

with open("dialogue.json", "r", encoding="utf-8") as f:
    dialogs = json.load(f)

with open("emotion_voice/ref_text.json", "r", encoding="utf-8") as f:
    ref_texts = json.load(f)

ref_text_map = {}
for item in ref_texts:
    ref_text_map.setdefault(item["emotion"].lower(), []).append(item["text"])

# =========================
# INIT MODEL
# =========================

f5tts = F5TTS()

# =========================
# HELPERS
# =========================

def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def normalize_audio(audio):
    if audio.dBFS == float("-inf"):
        return audio
    return audio.apply_gain(TARGET_DBFS - audio.dBFS)

def pick_voice_file(voice_type, emotion):
    # Thử đúng emotion trước, fallback sang neutral
    for emo in [emotion, "neutral"]:
        for i in [1, 2]:
            p = VOICE_DIR / f"{voice_type}_{emo}_{i}.wav"
            if p.exists():
                return p
    # Lấy bất kỳ file nào khớp voice_type
    files = list(VOICE_DIR.glob(f"{voice_type}_*.wav"))
    if files:
        return random.choice(files)
    raise FileNotFoundError(f"No voice file found for {voice_type}")

# =========================
# GENERATE
# =========================

combined_audio = AudioSegment.silent(duration=0)
subtitles = []
current_time = 0.0

for idx, item in enumerate(dialogs):
    speaker = item["speaker"]
    text = item["text"]
    emotion = item.get("emotion", "neutral").lower()

    voice_type = SPEAKER_VOICE.get(speaker, "girl")
    output_file = OUTPUT_DIR / f"{idx:03d}_{speaker}.wav"

    print(f"[{idx+1}/{len(dialogs)}] {speaker} ({emotion})")

    try:
        ref_file = pick_voice_file(voice_type, emotion)
        ref_text = random.choice(ref_text_map.get(emotion) or ref_text_map.get("neutral") or ["Hello."])

        f5tts.infer(
            ref_file=str(ref_file),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output_file),
            seed=None,
        )

        audio_data, sr = sf.read(output_file)
        duration = len(audio_data) / sr

        subtitles.append({
            "index": len(subtitles) + 1,
            "start": current_time,
            "end": current_time + duration,
            "speaker": speaker,
            "text": text,
        })
        current_time += duration

        seg = normalize_audio(AudioSegment.from_wav(output_file))
        combined_audio += seg

    except Exception as e:
        print(f"  ERROR: {e} — skipping")

# =========================
# EXPORT
# =========================

combined_audio.export(OUTPUT_DIR / "full_podcast.wav", format="wav")
print(f"Saved: {OUTPUT_DIR / 'full_podcast.wav'}")

with open(OUTPUT_DIR / "timestamps.srt", "w", encoding="utf-8") as f:
    for sub in subtitles:
        f.write(f"{sub['index']}\n")
        f.write(f"{seconds_to_srt_time(sub['start'])} --> {seconds_to_srt_time(sub['end'])}\n")
        f.write(f"{sub['speaker']}: {sub['text']}\n\n")
print(f"Saved: {OUTPUT_DIR / 'timestamps.srt'}")

print(f"\nDONE — {len(subtitles)}/{len(dialogs)} segments OK")