import os
import librosa
import soundfile as sf

INPUT_DIR = "Datasets/faatiha"
OUTPUT_DIR = "Datasets/faatiha/wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in sorted(os.listdir(INPUT_DIR)):

    if not filename.endswith(".mp3"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    output_filename = filename.replace(".mp3", ".wav")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Processing: {filename}")

    # Load audio as mono and resample to 16 kHz
    audio, sample_rate = librosa.load(
        input_path,
        sr=16000,
        mono=True
    )

    # Save WAV
    sf.write(
        output_path,
        audio,
        16000,
        subtype="PCM_16"
    )

    print(f"Saved: {output_path}")

print("\nAudio preparation completed!")