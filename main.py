import json
import random
from pathlib import Path

import soundfile as sf
from pydub import AudioSegment

from f5_tts.api import F5TTS

# =========================
# CONFIG
# =========================

# Mapping speaker -> voice type
SPEAKER_VOICE = {
    "Tracy": "girl",
    "Brian": "boy",
}

# Emotion fallback
EMOTION_FALLBACK = {
    "calm": "calm",
    "curious": "curious",
    "disappointed": "disappointed",
    "excited": "excited",
    "frustrated": "frustrated",
    "happy": "happy",
    "impressed": "impressed",
    "nervous": "nervous",
    "neutral": "neutral",
    "sad": "sad",
    "sorrowful": "sorrowful",
}

# =========================
# PATHS
# =========================

VOICE_DIR = Path("emotion_voice")

DIALOG_FILE = "dialogue.json"
REF_TEXT_FILE = "emotion_voice/ref_text.json"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SRT_FILE = OUTPUT_DIR / "timestamps.srt"
FINAL_AUDIO = OUTPUT_DIR / "full_podcast.wav"

# =========================
# AUDIO SETTINGS
# =========================

TARGET_DBFS = -18.0

# =========================
# LOAD JSON
# =========================

with open(DIALOG_FILE, "r", encoding="utf-8") as f:
    dialogs = json.load(f)

with open(REF_TEXT_FILE, "r", encoding="utf-8") as f:
    ref_texts = json.load(f)

# =========================
# BUILD REF TEXT MAP
# =========================

ref_text_map = {}

for item in ref_texts:

    emotion = item["emotion"].lower()
    text = item["text"]

    if emotion not in ref_text_map:
        ref_text_map[emotion] = []

    ref_text_map[emotion].append(text)

# =========================
# INIT MODEL
# =========================

f5tts = F5TTS()

# =========================
# HELPERS
# =========================

def seconds_to_srt_time(seconds: float):

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def normalize_audio(audio: AudioSegment):

    change_in_dbfs = TARGET_DBFS - audio.dBFS

    return audio.apply_gain(change_in_dbfs)

# =========================
# TIMESTAMP STORAGE
# =========================

subtitles = []

current_time = 0.0

# =========================
# FINAL MERGED AUDIO
# =========================

combined_audio = AudioSegment.silent(duration=0)

# =========================
# GENERATE
# =========================

for idx, item in enumerate(dialogs):

    speaker = item["speaker"]
    text = item["text"]

    emotion = item.get("emotion", "neutral").lower()

    # ---------------------
    # Voice type
    # ---------------------

    voice_type = SPEAKER_VOICE.get(speaker, "girl")

    # ---------------------
    # Emotion fallback
    # ---------------------

    voice_emotion = EMOTION_FALLBACK.get(
        emotion,
        "neutral"
    )

    # ---------------------
    # Random voice file
    # ---------------------

    voice_index = random.choice([1, 2])

    ref_file = (
        VOICE_DIR /
        f"{voice_type}_{voice_emotion}_{voice_index}.mp3"
    )

    # ---------------------
    # Random ref text
    # ---------------------

    if emotion in ref_text_map:

        ref_text = random.choice(
            ref_text_map[emotion]
        )

    elif voice_emotion in ref_text_map:

        ref_text = random.choice(
            ref_text_map[voice_emotion]
        )

    else:

        ref_text = "Hello, this is a sample voice."

    # ---------------------
    # Output file
    # ---------------------

    output_file = (
        OUTPUT_DIR /
        f"{idx:03d}_{speaker}.wav"
    )

    # ---------------------
    # Log
    # ---------------------

    print("=" * 60)

    print(f"Speaker : {speaker}")
    print(f"Emotion : {emotion}")
    print(f"Voice   : {ref_file.name}")
    print(f"Output  : {output_file.name}")

    # =====================
    # GENERATE AUDIO
    # =====================

    try:

        wav, sr, spec = f5tts.infer(
            ref_file=str(ref_file),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output_file),
            seed=None,
        )

        # =====================
        # LOAD GENERATED AUDIO
        # =====================

        audio_data, sample_rate = sf.read(output_file)

        duration = len(audio_data) / sample_rate

        # =====================
        # TIMESTAMP
        # =====================

        start_time = current_time
        end_time = current_time + duration

        subtitles.append({
            "index": idx + 1,
            "start": start_time,
            "end": end_time,
            "speaker": speaker,
            "text": text
        })

        # move timeline
        current_time = end_time

        # =====================
        # MERGE AUDIO
        # =====================

        audio_segment = AudioSegment.from_wav(
            output_file
        )

        # normalize volume
        audio_segment = normalize_audio(
            audio_segment
        )

        combined_audio += audio_segment

    except Exception as e:

        print(f"ERROR: {e}")

# =========================
# EXPORT FINAL PODCAST
# =========================

print("\nExporting final podcast...")

combined_audio.export(
    FINAL_AUDIO,
    format="wav"
)

print(f"Saved: {FINAL_AUDIO}")

# =========================
# WRITE SRT
# =========================

print("\nWriting subtitle file...")

with open(SRT_FILE, "w", encoding="utf-8") as f:

    for sub in subtitles:

        start = seconds_to_srt_time(
            sub["start"]
        )

        end = seconds_to_srt_time(
            sub["end"]
        )

        f.write(f"{sub['index']}\n")

        f.write(
            f"{start} --> {end}\n"
        )

        f.write(
            f"{sub['speaker']}: "
            f"{sub['text']}\n\n"
        )

print(f"Saved: {SRT_FILE}")

# =========================
# DONE
# =========================

print("\nDONE")