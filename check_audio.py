from mutagen.mp3 import MP3
import os

AUDIO_DIR = "dataset/faatiha"

for filename in sorted(os.listdir(AUDIO_DIR)):

    if filename.endswith(".mp3"):

        filepath = os.path.join(AUDIO_DIR, filename)

        audio = MP3(filepath)

        duration = audio.info.length

        print(f"{filename} -> {duration:.2f} seconds")