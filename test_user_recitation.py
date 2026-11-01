import sherpa_onnx
import soundfile as sf
import numpy as np
from pathlib import Path


# ============================================================
# FILES
# ============================================================

MODEL_PATH = Path("Models/quran/zipformer_p_arabic_v3.int8.onnx")
TOKENS_PATH = Path("Models/quran/tokens.txt")
AUDIO_PATH = Path("user_faatiha_ayah1.wav")


# ============================================================
# CHECK FILES
# ============================================================

for path in [MODEL_PATH, TOKENS_PATH, AUDIO_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File haipo: {path}")


# ============================================================
# LOAD QURAN MODEL
# ============================================================

print("Loading Quran model...")

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


# ============================================================
# READ USER AUDIO
# ============================================================

audio, sample_rate = sf.read(
    str(AUDIO_PATH),
    dtype="float32"
)


# ============================================================
# CONVERT TO MONO
# ============================================================

if audio.ndim > 1:
    audio = audio.mean(axis=1)


# ============================================================
# AUDIO INFORMATION
# ============================================================

duration = len(audio) / sample_rate
max_volume = np.max(np.abs(audio))
average_volume = np.mean(np.abs(audio))

print("\nAudio information:")
print(f"Sample rate : {sample_rate} Hz")
print("Channels    : 1")
print(f"Duration    : {duration:.2f} seconds")
print(f"Max volume  : {max_volume:.6f}")
print(f"Average     : {average_volume:.6f}")


# ============================================================
# BASIC AUDIO CHECK
# ============================================================

if max_volume < 0.001:

    print("\nERROR: Hakuna sauti kwenye recording.")

    raise SystemExit


elif max_volume < 0.01:

    print("\nWARNING: Sauti ni ndogo sana.")


else:

    print("\nAudio iko sawa.")


# ============================================================
# MAKE SURE SAMPLE RATE IS 16 KHZ
# ============================================================

if sample_rate != 16000:

    raise ValueError(
        f"Model inahitaji 16000 Hz, lakini audio ina {sample_rate} Hz."
    )


# ============================================================
# CREATE STREAM
# ============================================================

stream = recognizer.create_stream()


# ============================================================
# SEND AUDIO TO MODEL
# ============================================================

stream.accept_waveform(
    sample_rate,
    audio
)


# ============================================================
# TAIL PADDING
# ============================================================

tail_padding = np.zeros(
    int(0.66 * sample_rate),
    dtype=np.float32
)

stream.accept_waveform(
    sample_rate,
    tail_padding
)


# ============================================================
# FINISH AUDIO
# ============================================================

stream.input_finished()


# ============================================================
# DECODE
# ============================================================

print("\nListening to your recitation...")

while recognizer.is_ready(stream):

    recognizer.decode_stream(stream)


# ============================================================
# GET RESULT
# ============================================================

result = recognizer.get_result(stream)


# ============================================================
# SHOW RESULT
# ============================================================

print("\n" + "=" * 60)
print("YOUR RECITATION RESULT")
print("=" * 60)

if result.strip():

    print(result)

else:

    print("[MODEL HAIJATAMBUA SAUTI]")


# ============================================================
# EXPECTED AYAH
# ============================================================

print("\n" + "=" * 60)
print("EXPECTED AYAH")
print("=" * 60)

print(
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)