import asyncio
import re
from pathlib import Path

import av
import numpy as np

from faster_whisper import WhisperModel
from sqlalchemy import select

from Models.database import AsyncSessionLocal
from Models.ayah import Ayah
from Models.ayah_segment import AyahSegment
from Models.segment_word import SegmentWord


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FOLDER = Path(
    r"C:\Users\DELL\Desktop\Quran_pro\Backend\Datasets\faatiha"
)

WHISPER_MODEL = "small"

LANGUAGE = "ar"

WORDS_PER_SEGMENT = 2

SAMPLE_RATE = 16000

START_AYAH_ID = 1

END_AYAH_ID = 7


# ============================================================
# ARABIC NORMALIZATION
# ============================================================

def normalize_arabic(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    # Remove tashkeel
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text
    )

    # Remove tatweel
    text = text.replace("ـ", "")

    # Normalize Alef variants
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ٱ", "ا")

    # Normalize Ya
    text = text.replace("ى", "ي")

    # Normalize Ta Marbuta
    text = text.replace("ة", "ه")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text
    )

    # Remove multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DECODE AUDIO
# ============================================================

def decode_audio(
    audio_path: str,
    sampling_rate: int = 16000
) -> np.ndarray:

    container = av.open(
        audio_path,
        mode="r"
    )

    audio_stream = container.streams.audio[0]

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate
    )

    audio_chunks = []

    for frame in container.decode(
        audio_stream
    ):

        converted = resampler.resample(
            frame
        )

        if isinstance(
            converted,
            list
        ):
            audio_chunks.extend(
                converted
            )
        else:
            audio_chunks.append(
                converted
            )

    try:

        remaining = resampler.resample(
            None
        )

        if remaining:

            if isinstance(
                remaining,
                list
            ):
                audio_chunks.extend(
                    remaining
                )
            else:
                audio_chunks.append(
                    remaining
                )

    except Exception:
        pass

    container.close()

    if not audio_chunks:

        raise RuntimeError(
            f"Audio haikuweza kusomwa: "
            f"{audio_path}"
        )

    samples = []

    for frame in audio_chunks:

        data = frame.to_ndarray()

        data = data.reshape(-1)

        samples.append(
            data
        )

    audio = np.concatenate(
        samples
    )

    audio = audio.astype(
        np.float32
    ) / 32768.0

    return audio


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

def load_whisper():

    print()
    print("=" * 60)
    print("LOADING WHISPER MODEL")
    print("=" * 60)

    model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8"
    )

    print()
    print("Whisper model loaded.")

    return model


# ============================================================
# TRANSCRIBE ONE AUDIO
# ============================================================

def transcribe_audio(
    model,
    audio_path: Path
):

    print()
    print("-" * 60)
    print(
        f"TRANSCRIBING: {audio_path.name}"
    )
    print("-" * 60)

    audio = decode_audio(
        str(audio_path),
        SAMPLE_RATE
    )

    duration = (
        len(audio) / SAMPLE_RATE
    )

    print(
        f"Audio duration: "
        f"{duration:.2f}s"
    )

    segments, info = model.transcribe(
        audio,
        language=LANGUAGE,
        word_timestamps=True,
        beam_size=5,
        vad_filter=False
    )

    words = []

    for segment in segments:

        if not segment.words:

            continue

        for word in segment.words:

            word_text = word.word.strip()

            if not word_text:

                continue

            words.append({

                "text": word_text,

                "start": float(
                    word.start
                ),

                "end": float(
                    word.end
                )

            })

    print(
        f"Whisper detected "
        f"{len(words)} words."
    )

    for word in words:

        print(
            f"  {word['text']} "
            f"{word['start']:.3f}s -> "
            f"{word['end']:.3f}s"
        )

    return words


# ============================================================
# PREPARE QURAN WORDS
# ============================================================

def prepare_quran_words(
    text: str
):

    normalized = normalize_arabic(
        text
    )

    return normalized.split()


# ============================================================
# PREPARE WHISPER WORDS
# ============================================================

def prepare_whisper_words(
    words
):

    result = []

    for word in words:

        normalized = normalize_arabic(
            word["text"]
        )

        if not normalized:

            continue

        result.append({

            "text": normalized,

            "original": word["text"],

            "start": word["start"],

            "end": word["end"]

        })

    return result


# ============================================================
# ALIGN WORDS
# ============================================================

def align_words(
    quran_words,
    whisper_words
):

    aligned = []

    q_index = 0

    w_index = 0

    while (
        q_index < len(quran_words)
        and
        w_index < len(whisper_words)
    ):

        q_word = quran_words[
            q_index
        ]

        w_word = whisper_words[
            w_index
        ]

        # ----------------------------------------------------
        # Exact normalized match
        # ----------------------------------------------------

        if q_word == w_word["text"]:

            aligned.append({

                "text": q_word,

                "start": w_word["start"],

                "end": w_word["end"]

            })

            q_index += 1

            w_index += 1

            continue

        # ----------------------------------------------------
        # Search next Whisper word
        # ----------------------------------------------------

        found = False

        for next_index in range(
            w_index + 1,
            min(
                w_index + 3,
                len(whisper_words)
            )
        ):

            if (
                q_word
                ==
                whisper_words[
                    next_index
                ]["text"]
            ):

                w_index = next_index

                aligned.append({

                    "text": q_word,

                    "start":
                        whisper_words[
                            w_index
                        ]["start"],

                    "end":
                        whisper_words[
                            w_index
                        ]["end"]

                })

                q_index += 1

                w_index += 1

                found = True

                break

        if found:

            continue

        # ----------------------------------------------------
        # Fallback timestamp
        # ----------------------------------------------------

        print(
            "WARNING: word mismatch:"
        )

        print(
            f"  Quran:   {q_word}"
        )

        print(
            f"  Whisper: {w_word['text']}"
        )

        aligned.append({

            "text": q_word,

            "start": w_word["start"],

            "end": w_word["end"]

        })

        q_index += 1

        w_index += 1

    return aligned


# ============================================================
# CREATE SEGMENTS
# ============================================================

def create_segments(
    aligned_words
):

    segments = []

    current_words = []

    for word in aligned_words:

        current_words.append(
            word
        )

        if (
            len(current_words)
            >= WORDS_PER_SEGMENT
        ):

            segments.append(
                current_words
            )

            current_words = []

    if current_words:

        segments.append(
            current_words
        )

    return segments


# ============================================================
# GET AYAH
# ============================================================

async def get_ayah(
    ayah_id: int
):

    async with AsyncSessionLocal() as db:

        result = await db.execute(
            select(Ayah).where(
                Ayah.id == ayah_id
            )
        )

        return result.scalar_one_or_none()


# ============================================================
# DELETE OLD SEGMENTS
# ============================================================

async def delete_old_segments(
    ayah_id: int
):

    async with AsyncSessionLocal() as db:

        result = await db.execute(
            select(AyahSegment).where(
                AyahSegment.ayah_id == ayah_id
            )
        )

        segments = result.scalars().all()

        for segment in segments:

            await db.delete(
                segment
            )

        await db.commit()

        if segments:

            print(
                f"Deleted old segmentation: "
                f"{len(segments)} segments"
            )


# ============================================================
# SAVE SEGMENTATION
# ============================================================

async def save_segmentation(
    ayah_id: int,
    segments
):

    async with AsyncSessionLocal() as db:

        for segment_number, words in enumerate(
            segments,
            start=1
        ):

            if not words:

                continue

            segment_text = " ".join(
                word["text"]
                for word in words
            )

            start_time = (
                words[0]["start"]
            )

            end_time = (
                words[-1]["end"]
            )

            segment = AyahSegment(

                ayah_id=ayah_id,

                segment_number=segment_number,

                text=segment_text,

                start_time=start_time,

                end_time=end_time

            )

            db.add(
                segment
            )

            await db.flush()

            for word_number, word in enumerate(
                words,
                start=1
            ):

                segment_word = SegmentWord(

                    segment_id=segment.id,

                    word_number=word_number,

                    text=word["text"],

                    start_time=word["start"],

                    end_time=word["end"]

                )

                db.add(
                    segment_word
                )

        await db.commit()


# ============================================================
# PROCESS ONE AYAH
# ============================================================

async def process_ayah(
    model,
    ayah: Ayah,
    audio_path: Path
):

    print()
    print()
    print("=" * 60)
    print(
        f"PROCESSING AYAH {ayah.id}"
    )
    print("=" * 60)

    print()
    print("QURAN:")
    print(ayah.text)

    # --------------------------------------------------------
    # Quran words
    # --------------------------------------------------------

    quran_words = prepare_quran_words(
        ayah.text
    )

    print()
    print("QURAN WORDS:")

    print(
        quran_words
    )

    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    whisper_words = transcribe_audio(
        model,
        audio_path
    )

    if not whisper_words:

        print(
            "ERROR: Whisper detected no words."
        )

        return False

    whisper_words = prepare_whisper_words(
        whisper_words
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    aligned_words = align_words(
        quran_words,
        whisper_words
    )

    if not aligned_words:

        print(
            "ERROR: Alignment failed."
        )

        return False

    # --------------------------------------------------------
    # Display timestamps
    # --------------------------------------------------------

    print()
    print("FINAL WORD TIMESTAMPS")
    print("-" * 60)

    for index, word in enumerate(
        aligned_words,
        start=1
    ):

        print(
            f"{index}. "
            f"{word['text']} "
            f"-> "
            f"{word['start']:.3f}s - "
            f"{word['end']:.3f}s"
        )

    # --------------------------------------------------------
    # Create segments
    # --------------------------------------------------------

    segments = create_segments(
        aligned_words
    )

    print()
    print("SEGMENTS")
    print("-" * 60)

    for index, segment in enumerate(
        segments,
        start=1
    ):

        text = " ".join(
            word["text"]
            for word in segment
        )

        print(
            f"{index}. "
            f"{text} "
            f"-> "
            f"{segment[0]['start']:.3f}s - "
            f"{segment[-1]['end']:.3f}s"
        )

    # --------------------------------------------------------
    # Delete old
    # --------------------------------------------------------

    await delete_old_segments(
        ayah.id
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    await save_segmentation(
        ayah.id,
        segments
    )

    print()
    print(
        f"SUCCESS: Ayah {ayah.id}"
    )

    print(
        f"Saved segments: "
        f"{len(segments)}"
    )

    print(
        f"Saved words: "
        f"{len(aligned_words)}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

async def main():

    print()
    print("=" * 60)
    print("QURAN - FAATIHA SEGMENTATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    model = load_whisper()

    success_count = 0

    failed_count = 0

    # --------------------------------------------------------
    # Process Ayah 1 -> 7
    # --------------------------------------------------------

    for ayah_id in range(
        START_AYAH_ID,
        END_AYAH_ID + 1
    ):

        print()
        print(
            "#" * 60
        )

        print(
            f"AYAH {ayah_id} / "
            f"{END_AYAH_ID}"
        )

        print(
            "#" * 60
        )

        # ----------------------------------------------------
        # Get database Ayah
        # ----------------------------------------------------

        ayah = await get_ayah(
            ayah_id
        )

        if not ayah:

            print(
                f"ERROR: Ayah ID "
                f"{ayah_id} not found."
            )

            failed_count += 1

            continue

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        audio_path = (
            AUDIO_FOLDER
            /
            f"ayah_{ayah_id:03d}.mp3"
        )

        if not audio_path.exists():

            print()
            print(
                "ERROR: Audio not found:"
            )

            print(
                audio_path
            )

            failed_count += 1

            continue

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            success = await process_ayah(
                model,
                ayah,
                audio_path
            )

            if success:

                success_count += 1

            else:

                failed_count += 1

        except Exception as e:

            print()
            print(
                f"ERROR processing "
                f"Ayah {ayah_id}:"
            )

            print(
                repr(e)
            )

            failed_count += 1

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print()
    print(
        f"Successful: "
        f"{success_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    print()

    if failed_count == 0:

        print(
            "ALL 7 AYAT COMPLETED SUCCESSFULLY."
        )

    else:

        print(
            "Some Ayat failed. "
            "Check the errors above."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )