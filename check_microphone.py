import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5

print("🎙️ Rekodi sekunde 5...")
print("Sema: TEST TEST TEST")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write(
    "microphone_test.wav",
    audio,
    SAMPLE_RATE
)

print("✅ Imemaliza kurecord.")
print("File: microphone_test.wav")