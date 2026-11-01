import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 8

print("=" * 50)
print("QURAN RECITATION TEST")
print("=" * 50)

print("\nSoma:")
print("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")

input("\nBonyeza ENTER kuanza...")

print("\n🎙️ SOMA AYAH SASA...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write(
    "user_faatiha_ayah1.wav",
    audio,
    SAMPLE_RATE
)

print("\n✅ Recording imehifadhiwa:")
print("user_faatiha_ayah1.wav")