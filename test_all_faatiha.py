import json
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
from rapidfuzz.distance import Levenshtein


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

required_files = [
    MODEL_PATH,
    TOKENS_PATH,
    PHONEME_PATH,
    DATASET_PATH,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"File haipo: {path}"
        )


if not AUDIO_DIR.exists():

    raise FileNotFoundError(
        f"Audio folder haipo: {AUDIO_DIR}"
    )


# ============================================================
# LOAD TOKENS
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


# ============================================================
# LOAD PHONEME DATA
# ============================================================

print("Loading Quran phoneme data...")

phoneme_data = json.loads(
    PHONEME_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# LOAD DATASET
# ============================================================

dataset = json.loads(
    DATASET_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    return (
        text
        .replace("ٱ", "ا")
        .replace("ٰ", "")
        .replace("ٓ", "")
    )


# ============================================================
# FIND CANONICAL PHONEME
# ============================================================

def get_canonical_phoneme(ayah_text):

    normalized = normalize_text(
        ayah_text
    )

    for text, phoneme in phoneme_data.items():

        if normalize_text(text) == normalized:

            return phoneme

    return None


# ============================================================
# PHONEME → TOKEN IDS
# ============================================================

def phoneme_to_tokens(phoneme):

    tokens = []

    for char in phoneme.replace(" ", ""):

        token_id = token_to_id.get(char)

        if token_id is not None:

            tokens.append(token_id)

    return tokens


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
# RECOGNIZE AUDIO
# ============================================================

def recognize_audio(audio_path):

    with wave.open(
        str(audio_path),
        "rb"
    ) as f:

        sample_rate = f.getframerate()
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        frames = f.getnframes()

        if channels != 1:

            raise ValueError(
                f"Audio lazima iwe mono: {audio_path}"
            )

        if sample_rate != 16000:

            raise ValueError(
                f"Audio lazima iwe 16000 Hz: {audio_path}"
            )

        if sample_width != 2:

            raise ValueError(
                f"Audio lazima iwe 16-bit PCM: {audio_path}"
            )

        samples = f.readframes(frames)

    audio = np.frombuffer(
        samples,
        dtype=np.int16
    ).astype(np.float32)

    audio /= 32768.0

    stream = recognizer.create_stream()

    stream.accept_waveform(
        sample_rate,
        audio
    )

    # --------------------------------------------------------
    # Padding ya mwisho
    # --------------------------------------------------------

    tail_padding = np.zeros(
        int(0.66 * sample_rate),
        dtype=np.float32
    )

    stream.accept_waveform(
        sample_rate,
        tail_padding
    )

    stream.input_finished()

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    while recognizer.is_ready(stream):

        recognizer.decode_stream(stream)

    result = recognizer.get_result(stream)

    return result


# ============================================================
# START TEST
# ============================================================

print()
print("=" * 70)
print("TESTING ALL FAATIHA AYAHS")
print("=" * 70)


all_results = []


# ============================================================
# LOOP THROUGH ALL AYAHS
# ============================================================

for item in dataset:

    ayah_number = item["ayah_number"]

    ayah_text = item["text"]

    # --------------------------------------------------------
    # IMPORTANT:
    # Dataset ina .mp3 lakini prepared audio ina .wav
    # --------------------------------------------------------

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
    print(f"AYAH {ayah_number}")
    print("=" * 70)

    print(
        f"Audio: {audio_filename}"
    )

    # --------------------------------------------------------
    # Check audio
    # --------------------------------------------------------

    if not audio_path.exists():

        print(
            f"Audio haipo: {audio_path}"
        )

        continue

    # --------------------------------------------------------
    # Get canonical phoneme
    # --------------------------------------------------------

    canonical_phoneme = (
        get_canonical_phoneme(
            ayah_text
        )
    )

    if canonical_phoneme is None:

        print(
            "Canonical phoneme haikupatikana."
        )

        continue

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    print(
        "Running Quran model..."
    )

    model_result = recognize_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Convert to tokens
    # --------------------------------------------------------

    canonical_tokens = (
        phoneme_to_tokens(
            canonical_phoneme
        )
    )

    model_tokens = (
        phoneme_to_tokens(
            model_result
        )
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    edit_distance = (
        Levenshtein.distance(
            canonical_tokens,
            model_tokens
        )
    )

    alignment = (
        Levenshtein.editops(
            canonical_tokens,
            model_tokens
        )
    )

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    if len(canonical_tokens) > 0:

        error_rate = (
            edit_distance
            / len(canonical_tokens)
        ) * 100

        match_rate = (
            100
            - error_rate
        )

    else:

        error_rate = 100

        match_rate = 0

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print()
    print("QURAN TEXT:")
    print(ayah_text)

    print()
    print("CANONICAL PHONEME:")
    print(canonical_phoneme)

    print()
    print("MODEL RESULT:")
    print(model_result)

    print()
    print(
        "Canonical tokens :",
        len(canonical_tokens)
    )

    print(
        "Model tokens     :",
        len(model_tokens)
    )

    print(
        "Edit distance    :",
        edit_distance
    )

    print(
        f"Error rate       : "
        f"{error_rate:.2f}%"
    )

    print(
        f"Alignment match  : "
        f"{match_rate:.2f}%"
    )

    # --------------------------------------------------------
    # DIFFERENCES
    # --------------------------------------------------------

    if alignment:

        print()
        print("DIFFERENCES:")

        for operation in alignment:

            tag = operation.tag

            src_pos = operation.src_pos

            dest_pos = operation.dest_pos

            # ------------------------------------------------
            # INSERT
            # ------------------------------------------------

            if tag == "insert":

                token = model_tokens[
                    dest_pos
                ]

                print(
                    f"  INSERT  "
                    f"position={dest_pos} "
                    f"token={token}"
                )

            # ------------------------------------------------
            # DELETE
            # ------------------------------------------------

            elif tag == "delete":

                token = canonical_tokens[
                    src_pos
                ]

                print(
                    f"  DELETE  "
                    f"position={src_pos} "
                    f"token={token}"
                )

            # ------------------------------------------------
            # REPLACE
            # ------------------------------------------------

            elif tag == "replace":

                expected = canonical_tokens[
                    src_pos
                ]

                received = model_tokens[
                    dest_pos
                ]

                print(
                    f"  REPLACE "
                    f"position={src_pos} "
                    f"expected={expected} "
                    f"received={received}"
                )

    else:

        print()
        print(
            "✓ No token differences."
        )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    all_results.append({

        "ayah": ayah_number,

        "audio": audio_filename,

        "text": ayah_text,

        "canonical_phoneme":
            canonical_phoneme,

        "model_result":
            model_result,

        "canonical_token_count":
            len(canonical_tokens),

        "model_token_count":
            len(model_tokens),

        "edit_distance":
            edit_distance,

        "error_rate":
            round(
                error_rate,
                2
            ),

        "alignment_match":
            round(
                match_rate,
                2
            )

    })


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FAATIHA SUMMARY")
print("=" * 70)

if not all_results:

    print()
    print(
        "Hakuna Ayah iliyofanikiwa kuprocess."
    )

else:

    print()

    for result in all_results:

        print(
            f"Ayah {result['ayah']}: "
            f"{result['alignment_match']:.2f}% "
            f"match | "
            f"{result['edit_distance']} edits"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_PATH = (
    BASE_DIR
    / "Datasets"
    / "faatiha"
    / "faatiha_results.json"
)


RESULT_PATH.write_text(
    json.dumps(
        all_results,
        ensure_ascii=False,
        indent=4
    ),
    encoding="utf-8"
)


print()
print(
    f"Results saved to:"
)

print(
    RESULT_PATH
)

print()
print(
    "Test completed."
)