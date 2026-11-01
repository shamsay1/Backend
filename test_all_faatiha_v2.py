import json
import wave
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import sherpa_onnx


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "quran"
    / "zipformer_p_arabic_v3.int8.onnx"
)

TOKENS_PATH = (
    BASE_DIR
    / "Models"
    / "quran"
    / "tokens.txt"
)

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

AUDIO_DIR = (
    BASE_DIR
    / "Datasets"
    / "faatiha"
    / "wav"
)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    MODEL_PATH,
    TOKENS_PATH,
    PHONEME_PATH,
    DATASET_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"File haipo: {path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading tokens...")

token_lines = TOKENS_PATH.read_text(
    encoding="utf-8"
).splitlines()

token_to_id = {}

for line in token_lines:

    parts = line.rsplit(" ", 1)

    if len(parts) == 2:

        token = parts[0]

        token_id = int(parts[1])

        token_to_id[token] = token_id


print("Loading Quran phoneme data...")

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
# NORMALIZE QURAN TEXT
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

    text = text.replace(
        " ",
        ""
    )

    return text


# ============================================================
# FIND CANONICAL PHONEME
# ============================================================

def find_canonical_phoneme(
    dataset_text
):

    normalized_dataset = normalize_text(
        dataset_text
    )

    best_match = None

    best_score = 0

    for text, phoneme in phoneme_data.items():

        normalized_source = normalize_text(
            text
        )

        score = SequenceMatcher(
            None,
            normalized_dataset,
            normalized_source
        ).ratio()

        if score > best_score:

            best_score = score

            best_match = phoneme

    return best_match, best_score


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Quran model...")

recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(

    tokens=str(TOKENS_PATH),

    model=str(MODEL_PATH),

    num_threads=2,

    provider="cpu",

    sample_rate=16000,

    feature_dim=80,

    decoding_method="greedy_search",
)

print("Model loaded successfully.")


# ============================================================
# RUN MODEL
# ============================================================

def recognize_audio(audio_path):

    with wave.open(
        str(audio_path),
        "rb"
    ) as f:

        sample_rate = f.getframerate()

        frames = f.readframes(
            f.getnframes()
        )

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(np.float32)

        audio /= 32768.0


    stream = recognizer.create_stream()


    stream.accept_waveform(
        sample_rate,
        audio
    )


    # Important for streaming model
    tail_padding = np.zeros(
        int(0.66 * sample_rate),
        dtype=np.float32
    )


    stream.accept_waveform(
        sample_rate,
        tail_padding
    )


    stream.input_finished()


    while recognizer.is_ready(stream):

        recognizer.decode_stream(
            stream
        )


    result = recognizer.get_result(
        stream
    )


    return result


# ============================================================
# RESULTS
# ============================================================

results = []


print()
print("=" * 70)
print("TESTING ALL FAATIHA AYAHS")
print("=" * 70)


for item in dataset:

    ayah_number = item[
        "ayah_number"
    ]

    quran_text = item[
        "text"
    ]

    audio_filename = (
        Path(item["audio"]).stem
        + ".wav"
    )

    audio_path = (
        AUDIO_DIR
        / audio_filename
    )


    print()
    print("=" * 70)

    print(
        f"AYAH {ayah_number}"
    )

    print("=" * 70)


    print()
    print(
        f"Audio: {audio_filename}"
    )


    if not audio_path.exists():

        print(
            "Audio haipo."
        )

        continue


    # --------------------------------------------------------
    # CANONICAL
    # --------------------------------------------------------

    canonical_phoneme, match_score = (
        find_canonical_phoneme(
            quran_text
        )
    )


    if canonical_phoneme is None:

        print(
            "Canonical phoneme haikupatikana."
        )

        continue


    print()
    print("QURAN TEXT:")

    print(
        quran_text
    )


    print()
    print(
        "Canonical source match:"
    )

    print(
        f"{match_score * 100:.2f}%"
    )


    print()
    print(
        "CANONICAL PHONEME:"
    )

    print(
        canonical_phoneme
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Running Quran model..."
    )


    model_result = recognize_audio(
        audio_path
    )


    print()
    print(
        "MODEL RESULT:"
    )

    print(
        model_result
    )


    # --------------------------------------------------------
    # NORMALIZED MODEL RESULT
    # --------------------------------------------------------

    canonical_clean = (
        canonical_phoneme
        .replace(" ", "")
    )

    model_clean = (
        model_result
        .replace(" ", "")
    )


    # --------------------------------------------------------
    # SIMPLE CHARACTER ALIGNMENT
    # --------------------------------------------------------

    similarity = SequenceMatcher(
        None,
        canonical_clean,
        model_clean
    ).ratio()


    print()
    print(
        "PHONEME STRING SIMILARITY:"
    )

    print(
        f"{similarity * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "ayah_number":
            ayah_number,

        "quran_text":
            quran_text,

        "canonical_phoneme":
            canonical_phoneme,

        "model_result":
            model_result,

        "canonical_source_match":
            round(
                match_score * 100,
                2
            ),

        "phoneme_similarity":
            round(
                similarity * 100,
                2
            ),
    })


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    BASE_DIR
    / "Datasets"
    / "faatiha"
    / "faatiha_results_v2.json"
)


OUTPUT_PATH.write_text(

    json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    ),

    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FAATIHA SUMMARY")
print("=" * 70)


for result in results:

    print(
        f"Ayah {result['ayah_number']}: "
        f"{result['phoneme_similarity']:.2f}%"
    )


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_PATH
)

print()
print(
    "Test completed."
)