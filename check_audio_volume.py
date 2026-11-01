import soundfile as sf
import numpy as np

AUDIO_FILE = "microphone_test.wav"

audio, sample_rate = sf.read(AUDIO_FILE)

# Kama audio ina channels nyingi
if audio.ndim > 1:
    audio = audio.mean(axis=1)

max_volume = np.max(np.abs(audio))
average_volume = np.mean(np.abs(audio))

print("=" * 50)
print("MICROPHONE AUDIO TEST")
print("=" * 50)

print(f"Sample rate   : {sample_rate} Hz")
print(f"Duration      : {len(audio) / sample_rate:.2f} seconds")
print(f"Max volume    : {max_volume:.6f}")
print(f"Average volume: {average_volume:.6f}")

if max_volume < 0.001:
    print("\n❌ Hakuna sauti iliyorekodiwa.")
elif max_volume < 0.01:
    print("\n⚠️ Sauti ni ndogo sana.")
else:
    print("\n✅ Microphone inarekodi sauti vizuri.")