import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import numpy as np
import json
import os
import random
from pathlib import Path
from datetime import datetime

VOICE_DIR = "emotion_voice"


model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

speaker_gender_map = {
    "Tracy": "girl",
    "Brian": "boy",
}

with open("emotion_voice/ref_text.json", "r", encoding="utf-8") as f:
    ref_data = json.load(f)

emotion_texts = {
    item["emotion"]: item["text"]
    for item in ref_data
}

voice_prompts = {}

for file in os.listdir(VOICE_DIR):

    if not file.endswith(".wav"):
        continue

    filename = file.replace(".wav", "")

    parts = filename.split("_")

    speaker = parts[0]
    emotion = parts[1]
    idx = parts[2]

    key = f"{speaker}_{emotion}"

    audio_path = os.path.join(VOICE_DIR, file)

    ref_text = emotion_texts[emotion]

    prompt = model.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    if key not in voice_prompts:
        voice_prompts[key] = []

    voice_prompts[key].append({
        "file": file,
        "prompt": prompt
    })

    print(f"Loaded: {file}")


with open("segments.json", "r", encoding="utf-8") as f:
    segments = json.load(f)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("output") / f"run_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

all_wavs = []
for i, seg in enumerate(segments):

    speaker_name = seg["speaker"]
    emotion = seg["emotion"]
    text = seg["text"]

    # Tracy -> girl
    # Brian -> boy
    gender = speaker_gender_map[speaker_name]

    key = f"{gender}_{emotion}"

    # random prompt
    selected = random.choice(voice_prompts[key])

    prompt = selected["prompt"]

    print(f"[{i}] {speaker_name} -> {selected['file']}")

    wav, sr = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=prompt,
    )

    if isinstance(wav, list):
        wav = wav[0]

    chunk_path = output_dir / f"chunk_{i:03d}.wav"

    # save chunk
    sf.write(chunk_path, wav, sr)

    all_wavs.append(wav)

    # add pause
    pause = np.zeros(int(sr * 0.3))
    all_wavs.append(pause)


final_audio = np.concatenate(all_wavs)
full_audio_path = output_dir / "full_audio.wav"

sf.write(
    full_audio_path,
    final_audio,
    sr
)

print(f"Done. Saved to: {output_dir}")