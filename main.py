import json
import random
from pathlib import Path
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

# Folder paths
VOICE_DIR = Path("emotion_voice")
DIALOG_FILE = "dialogue.json"
REF_TEXT_FILE = "emotion_voice/ref_text.json"
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

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

# Example:
# {
#   "sad": ["text1", "text2"],
#   "sorrowful": [...]
# }

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

    voice_emotion = EMOTION_FALLBACK.get(emotion, "calm")

    # ---------------------
    # Random voice file
    # Example:
    # boy_calm_1.mp3
    # boy_calm_2.mp3
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
        ref_text = random.choice(ref_text_map[emotion])

    elif voice_emotion in ref_text_map:
        ref_text = random.choice(ref_text_map[voice_emotion])

    else:
        # fallback
        ref_text = "Hello, this is a sample voice."

    # ---------------------
    # Output file
    # ---------------------

    output_file = OUTPUT_DIR / f"{idx:03d}_{speaker}.wav"

    # ---------------------
    # Generate
    # ---------------------

    print("=" * 60)
    print(f"Speaker : {speaker}")
    print(f"Emotion : {emotion}")
    print(f"Voice   : {ref_file.name}")
    print(f"Output  : {output_file.name}")

    try:
        wav, sr, spec = f5tts.infer(
            ref_file=str(ref_file),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output_file),
            seed=None,
        )

    except Exception as e:
        print(f"ERROR: {e}")

print("\nDONE")