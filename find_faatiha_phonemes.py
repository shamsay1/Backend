import json
from pathlib import Path
from difflib import SequenceMatcher


BASE_DIR = Path(__file__).resolve().parent

PHONEME_PATH = (
    BASE_DIR
    / "Models"
    / "quran"
    / "quran_text2phoneme.json"
)

DATASET_PATH = (
    BASE_DIR
    / "Datasets"
    / "faatiha"
    / "faatiha_dataset.json"
)


# ============================================================
# LOAD DATA
# ============================================================

phoneme_data = json.loads(
    PHONEME_PATH.read_text(
        encoding="utf-8"
    )
)

dataset = json.loads(
    DATASET_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_text(text):

    replacements = {
        "ٱ": "ا",
        "ٲ": "ا",
        "ٳ": "ا",
        "ٵ": "ا",
        "ٶ": "ا",
        "ٷ": "ا",

        "ٰ": "",
        "ٓ": "",
        "ٔ": "",
        "ٕ": "",

        "ۥ": "",
        "ۦ": "",

        "ۖ": "",
        "ۗ": "",
        "ۚ": "",
        "ۛ": "",
        "ۜ": "",
        "۟": "",
        "۠": "",
        "ۡ": "",
        "ۢ": "",
        "ۭ": "",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Remove spaces
    text = text.replace(
        " ",
        ""
    )

    return text


# ============================================================
# PREPARE CANONICAL ENTRIES
# ============================================================

canonical_entries = []

for text, phoneme in phoneme_data.items():

    normalized = normalize_text(
        text
    )

    canonical_entries.append({
        "original_text": text,
        "normalized_text": normalized,
        "phoneme": phoneme
    })


# ============================================================
# FIND BEST MATCH
# ============================================================

for item in dataset:

    ayah_number = item["ayah_number"]

    dataset_text = item["text"]

    normalized_dataset = normalize_text(
        dataset_text
    )

    print()
    print("=" * 70)
    print(
        f"AYAH {ayah_number}"
    )
    print("=" * 70)

    print()
    print("DATASET:")
    print(dataset_text)

    print()
    print("NORMALIZED:")
    print(normalized_dataset)

    best_match = None
    best_score = 0

    # --------------------------------------------------------
    # Compare against every canonical entry
    # --------------------------------------------------------

    for entry in canonical_entries:

        score = SequenceMatcher(
            None,
            normalized_dataset,
            entry["normalized_text"]
        ).ratio()

        if score > best_score:

            best_score = score
            best_match = entry

    # --------------------------------------------------------
    # Display best match
    # --------------------------------------------------------

    if best_match is None:

        print()
        print("✗ No candidate found.")

        continue

    print()
    print(
        f"BEST MATCH: "
        f"{best_score * 100:.2f}%"
    )

    print()
    print("CANONICAL SOURCE TEXT:")

    print(
        best_match["original_text"]
    )

    print()
    print("CANONICAL NORMALIZED:")

    print(
        best_match["normalized_text"]
    )

    print()
    print("CANONICAL PHONEME:")

    print(
        best_match["phoneme"]
    )