import sherpa_onnx
import wave
import numpy as np
from pathlib import Path


# =========================
# PATHS
# =========================

MODEL_PATH = Path("Models/quran/zipformer_p_arabic_v3.int8.onnx")
TOKENS_PATH = Path("Models/quran/tokens.txt")
AUDIO_PATH = Path("Datasets/faatiha/wav/ayah_001.wav")


# =========================
# CHECK FILES
# =========================

for path in [MODEL_PATH, TOKENS_PATH, AUDIO_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File haipo: {path}")


# =========================
# LOAD TOKENS
# =========================

token_lines = TOKENS_PATH.read_text(encoding="utf-8").splitlines()

id_to_token = {}

for line in token_lines:
    parts = line.rsplit(" ", 1)

    if len(parts) == 2:
        token = parts[0]
        token_id = int(parts[1])
        id_to_token[token_id] = token


# =========================
# READ AUDIO
# =========================

with wave.open(str(AUDIO_PATH), "rb") as f:

    sample_rate = f.getframerate()
    channels = f.getnchannels()
    sample_width = f.getsampwidth()
    num_frames = f.getnframes()

    print("Audio information:")
    print(f"Sample rate : {sample_rate}")
    print(f"Channels    : {channels}")
    print(f"Sample width: {sample_width}")
    print(f"Frames      : {num_frames}")


# =========================
# CREATE ONLINE RECOGNIZER
# =========================

print("\nLoading Quran model...")

recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
    tokens=str(TOKENS_PATH),
    model=str(MODEL_PATH),
    num_threads=2,
    provider="cpu",
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

print("Model loaded successfully.")


# =========================
# LOAD AUDIO
# =========================

with wave.open(str(AUDIO_PATH), "rb") as f:

    samples = f.readframes(f.getnframes())

    audio = np.frombuffer(
        samples,
        dtype=np.int16
    ).astype(np.float32)

    audio /= 32768.0

    sample_rate = f.getframerate()


# =========================
# CREATE STREAM
# =========================

stream = recognizer.create_stream()


# =========================
# FEED AUDIO
# =========================

stream.accept_waveform(
    sample_rate,
    audio
)


# Streaming tail padding
tail_padding = np.zeros(
    int(0.66 * sample_rate),
    dtype=np.float32
)

stream.accept_waveform(
    sample_rate,
    tail_padding
)

stream.input_finished()


# =========================
# DECODE
# =========================

print("\nRunning Quran recitation recognition...")

while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)


# =========================
# RESULT
# =========================

result = recognizer.get_result(stream)

print("\n==============================")
print("MODEL RESULT")
print("==============================")

print(result)


# =========================
# RESULT CHARACTERS
# =========================

print("\n==============================")
print("MODEL RESULT CHARACTERS")
print("==============================")

for i, char in enumerate(result):
    print(i, repr(char))


# =========================
# TOKEN LOOKUP
# =========================

print("\n==============================")
print("TOKEN LOOKUP")
print("==============================")

print("NOTE:")
print("The displayed result is rendered text.")
print("We inspect each character against tokens.txt.")

for char in result:

    matches = [
        token_id
        for token_id, token in id_to_token.items()
        if token == char
    ]

    print(
        repr(char),
        "=>",
        matches
    )