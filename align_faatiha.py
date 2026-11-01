import json
from pathlib import Path
from rapidfuzz.distance import Levenshtein


BASE_DIR = Path(__file__).resolve().parent

RESULTS_PATH = (
    BASE_DIR
    / "Datasets"
    / "faatiha"
    / "faatiha_results_v2.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

results = json.loads(
    RESULTS_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# CLEAN PHONEME STRING
# ============================================================

def clean_phoneme(text):
    """
    Tunaondoa spaces tu.
    Hatuondoi ۦ wala alama nyingine,
    kwa sababu tunataka kwanza kuona
    model ilitoa nini hasa.
    """

    return text.replace(
        " ",
        ""
    )


# ============================================================
# ALIGN TWO STRINGS
# ============================================================

def align_strings(reference, hypothesis):

    operations = Levenshtein.editops(
        reference,
        hypothesis
    )

    return operations


# ============================================================
# DISPLAY ALIGNMENT
# ============================================================

print()
print("=" * 70)
print("FAATIHA PHONEME ALIGNMENT")
print("=" * 70)


for item in results:

    ayah = item[
        "ayah_number"
    ]

    canonical = clean_phoneme(
        item[
            "canonical_phoneme"
        ]
    )

    model = clean_phoneme(
        item[
            "model_result"
        ]
    )


    print()
    print("=" * 70)

    print(
        f"AYAH {ayah}"
    )

    print("=" * 70)


    print()
    print("CANONICAL:")

    print(
        canonical
    )


    print()
    print("MODEL:")

    print(
        model
    )


    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    print()

    print(
        f"Canonical length : "
        f"{len(canonical)}"
    )

    print(
        f"Model length     : "
        f"{len(model)}"
    )


    # --------------------------------------------------------
    # EDIT OPERATIONS
    # --------------------------------------------------------

    operations = align_strings(
        canonical,
        model
    )


    if not operations:

        print()
        print(
            "✓ EXACT MATCH"
        )

        continue


    print()
    print(
        f"Total differences: "
        f"{len(operations)}"
    )


    print()
    print("DIFFERENCES:")


    for operation in operations:

        tag = operation.tag

        source_position = operation.src_pos

        destination_position = (
            operation.dest_pos
        )


        if tag == "insert":

            inserted_char = model[
                destination_position
            ]

            print(
                f"INSERT  "
                f"model[{destination_position}] "
                f"= {repr(inserted_char)}"
            )


        elif tag == "delete":

            deleted_char = canonical[
                source_position
            ]

            print(
                f"DELETE  "
                f"canonical[{source_position}] "
                f"= {repr(deleted_char)}"
            )


        elif tag == "replace":

            canonical_char = canonical[
                source_position
            ]

            model_char = model[
                destination_position
            ]

            print(
                f"REPLACE "
                f"canonical[{source_position}] "
                f"{repr(canonical_char)} "
                f"-> "
                f"model[{destination_position}] "
                f"{repr(model_char)}"
            )


    # --------------------------------------------------------
    # SHOW CONTEXT
    # --------------------------------------------------------

    print()
    print("MODEL CHARACTERS:")

    for index, char in enumerate(model):

        print(
            f"{index:3} "
            f"{repr(char)}"
        )


print()
print("=" * 70)
print("ALIGNMENT COMPLETED")
print("=" * 70)