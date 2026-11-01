import requests
import os

BASE_URL = "https://cdn.islamic.network/quran/audio/128/ar.alafasy"

OUTPUT_DIR = "Datasets/faatiha"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Al-Faatiha ina Aya 7
for ayah_id in range(1, 8):

    url = f"{BASE_URL}/{ayah_id}.mp3"

    filename = f"ayah_{ayah_id:03d}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    print(f"Downloading Ayah {ayah_id}...")

    response = requests.get(url)

    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)

        print(f"Saved: {filepath}")
    else:
        print(f"Failed Ayah {ayah_id}: HTTP {response.status_code}")

print("\nDone!")