import sounddevice as sd
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_DIR = Path("Datasets/faatiha/user")
OUTPUT_FILE = OUTPUT_DIR / "ayah_001.wav"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("QURAN RECITATION RECORDER")
print("=" * 50)

print("\nSoma Ayah hii:")
print("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")

input("\nBonyeza ENTER kuanza kurecord...")

print("\n🎙️ RECORDING...")
print("Soma Ayah sasa.")
print("Baada ya kumaliza, subiri kidogo kisha bonyeza ENTER.")

recording = sd.rec(
    int(15 * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32"
)

input()

sd.stop()

# Chukua sehemu iliyorekodiwa hadi wakati wa kubonyeza Enter
# Kwa prototype hii tunatumia sekunde 15 maximum.
sf.write(
    OUTPUT_FILE,
    recording,
    SAMPLE_RATE
)

print("\n✅ Recording imehifadhiwa:")
print(OUTPUT_FILE)