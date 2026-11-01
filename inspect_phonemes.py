import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TOKENS_PATH = BASE_DIR / "Models" / "quran" / "tokens.txt"
PHONEME_PATH = BASE_DIR / "Models" / "quran" / "quran_text2phoneme.json"
DATASET_PATH = BASE_DIR / "Datasets" / "faatiha" / "faatiha_dataset.json"


# --------------------------------------------------
# 1. Load token dictionary
# --------------------------------------------------

token_lines = TOKENS_PATH.read_text(encoding="utf-8").splitlines()

token_to_id = {}

for line in token_lines:
    parts = line.rsplit(" ", 1)

    if len(parts) == 2:
        token = parts[0]
        token_id = int(parts[1])
        token_to_id[token] = token_id


# --------------------------------------------------
# 2. Load Quran phoneme mapping
# --------------------------------------------------

phoneme_data = json.loads(
    PHONEME_PATH.read_text(encoding="utf-8")
)


# --------------------------------------------------
# 3. Load Faatiha dataset
# --------------------------------------------------

dataset = json.loads(
    DATASET_PATH.read_text(encoding="utf-8")
)

ayah_text = dataset[0]["text"]


# --------------------------------------------------
# 4. Normalize Arabic text
# --------------------------------------------------

def normalize_text(text):
    return (
        text
        .replace("ٱ", "ا")
        .replace("ٰ", "")
        .replace("ٓ", "")
    )


# --------------------------------------------------
# 5. Find canonical phoneme
# --------------------------------------------------

normalized_ayah = normalize_text(ayah_text)

canonical_phoneme = None

for text, phoneme in phoneme_data.items():

    if normalize_text(text) == normalized_ayah:
        canonical_phoneme = phoneme
        break


if canonical_phoneme is None:
    print("Canonical phoneme NOT FOUND.")
    exit()


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("=" * 50)
print("FAATIHA AYAH 1")
print("=" * 50)

print()
print("QURAN TEXT:")
print(ayah_text)

print()
print("CANONICAL PHONEME:")
print(canonical_phoneme)

print()
print("PHONEME CHARACTERS:")
print(list(canonical_phoneme))

print()
print("TOKEN IDs:")
print()

for char in canonical_phoneme:

    token_id = token_to_id.get(char)

    print(repr(char), "=>", token_id)