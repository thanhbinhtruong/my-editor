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
    "Qwen/Qwen3-TTS-24kHz-2B-v1",
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


srt_lines = []
current_time = 0.0

def format_srt_time(seconds: float):

    ms = int((seconds % 1) * 1000)
    total_seconds = int(seconds)

    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600

    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


all_wavs = []
total_segments = len(segments)

for i, seg in enumerate(segments):

    speaker_name = seg["speaker"]
    emotion = seg["emotion"]
    text = seg["text"]

    # Tracy -> girl
    # Brian -> boy
    gender = speaker_gender_map[speaker_name]

    key = f"{gender}_{emotion}"

    # fallback to neutral if emotion not found
    if key not in voice_prompts:
        print(f"[WARNING] {key} not found, using {gender}_neutral")
        key = f"{gender}_neutral"

    selected = random.choice(voice_prompts[key])

    prompt = selected["prompt"]

    print(f"[{i+1}/{total_segments}] {speaker_name} -> {selected['file']}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

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


    duration = len(wav) / sr

    start_time = current_time
    end_time = current_time + duration

    srt_lines.append(
        f"{i + 1}\n"
        f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
        f"{text}\n"
    )

    # update current time
    current_time = end_time + 0.3

    # append audio
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


srt_path = output_dir / "subtitles.srt"

with open(srt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(srt_lines))

print(f"SRT saved: {srt_path}")

print(f"Done. Saved to: {output_dir}")