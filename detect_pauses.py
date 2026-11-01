import librosa

audio_path = "Datasets/faatiha/ayah_007.mp3"

# Load audio
y, sr = librosa.load(audio_path, sr=None, mono=True)

# Detect non-silent intervals
intervals = librosa.effects.split(
    y,
    top_db=30,
    frame_length=2048,
    hop_length=512
)

print("\nDetected sound intervals:\n")

for i, (start, end) in enumerate(intervals, start=1):

    start_time = start / sr
    end_time = end / sr
    duration = end_time - start_time

    print(
        f"Part {i}: "
        f"{start_time:.2f}s -> {end_time:.2f}s "
        f"({duration:.2f}s)"
    )