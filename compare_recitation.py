import json
from pathlib import Path
from rapidfuzz.distance import Levenshtein


BASE_DIR = Path(__file__).resolve().parent

TOKENS_PATH = BASE_DIR / "Models" / "quran" / "tokens.txt"
PHONEME_PATH = BASE_DIR / "Models" / "quran" / "quran_text2phoneme.json"
DATASET_PATH = BASE_DIR / "Datasets" / "faatiha" / "faatiha_dataset.json"


# ============================================================
# 1. LOAD TOKENS
# ============================================================

token_lines = TOKENS_PATH.read_text(encoding="utf-8").splitlines()

token_to_id = {}

for line in token_lines:
    parts = line.rsplit(" ", 1)

    if len(parts) == 2:
        token = parts[0]
        token_id = int(parts[1])
        token_to_id[token] = token_id


# ============================================================
# 2. LOAD QURAN PHONEME DATA
# ============================================================

phoneme_data = json.loads(
    PHONEME_PATH.read_text(encoding="utf-8")
)

dataset = json.loads(
    DATASET_PATH.read_text(encoding="utf-8")
)


# ============================================================
# 3. NORMALIZE QURAN TEXT
# ============================================================

def normalize_text(text):
    return (
        text
        .replace("ٱ", "ا")
        .replace("ٰ", "")
        .replace("ٓ", "")
    )


# ============================================================
# 4. GET AYAH
# ============================================================

ayah_text = dataset[0]["text"]

normalized_ayah = normalize_text(ayah_text)


# ============================================================
# 5. FIND CANONICAL PHONEME
# ============================================================

canonical_phoneme = None

for text, phoneme in phoneme_data.items():

    if normalize_text(text) == normalized_ayah:

        canonical_phoneme = phoneme
        break


if canonical_phoneme is None:
    raise ValueError(
        "Canonical phoneme haikupatikana."
    )


# ============================================================
# 6. MODEL RESULT
# ============================================================

model_result = "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم"


# ============================================================
# 7. REMOVE SPACES
# ============================================================

canonical = canonical_phoneme.replace(" ", "")
model = model_result.replace(" ", "")


# ============================================================
# 8. CONVERT TO TOKEN IDS
# ============================================================

canonical_tokens = []

for char in canonical:

    token_id = token_to_id.get(char)

    if token_id is not None:
        canonical_tokens.append(token_id)


model_tokens = []

for char in model:

    token_id = token_to_id.get(char)

    if token_id is not None:
        model_tokens.append(token_id)


# ============================================================
# 9. ALIGNMENT
# ============================================================

edit_distance = Levenshtein.distance(
    canonical_tokens,
    model_tokens
)

alignment = Levenshtein.editops(
    canonical_tokens,
    model_tokens
)


# ============================================================
# 10. OUTPUT
# ============================================================

print("=" * 60)
print("QURAN RECITATION ALIGNMENT")
print("=" * 60)

print()

print("CANONICAL PHONEME:")
print(canonical_phoneme)

print()

print("MODEL RESULT:")
print(model_result)

print()

print("CANONICAL TOKENS:")
print(canonical_tokens)

print()

print("MODEL TOKENS:")
print(model_tokens)

print()

print("=" * 60)
print("ALIGNMENT RESULT")
print("=" * 60)

print()

if not alignment:

    print("✓ Hakuna tofauti iliyogunduliwa.")

else:

    print(f"Total differences: {len(alignment)}")
    print()

    for operation in alignment:

        tag = operation.tag
        src_pos = operation.src_pos
        dest_pos = operation.dest_pos

        if tag == "insert":

            token = model_tokens[dest_pos]

            print(
                f"INSERT  → position {dest_pos} "
                f"token {token}"
            )

        elif tag == "delete":

            token = canonical_tokens[src_pos]

            print(
                f"DELETE  → position {src_pos} "
                f"token {token}"
            )

        elif tag == "replace":

            expected = canonical_tokens[src_pos]
            received = model_tokens[dest_pos]

            print(
                f"REPLACE → position {src_pos} "
                f"expected={expected}, "
                f"received={received}"
            )


# ============================================================
# 11. ERROR RATE
# ============================================================

error_rate = (
    edit_distance / len(canonical_tokens)
) * 100

match_rate = 100 - error_rate


print()

print("=" * 60)
print("SCORE")
print("=" * 60)

print()

print(f"Edit distance : {edit_distance}")
print(f"Canonical length : {len(canonical_tokens)}")
print(f"Error rate : {error_rate:.2f}%")
print(f"Alignment match : {match_rate:.2f}%")