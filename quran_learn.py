import numpy as np
from difflib import SequenceMatcher

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from quran_recitation import (   
    recognizer,
    SAMPLE_RATE,
    IGNORE_FIRST_SECONDS,
    SPEECH_RMS_THRESHOLD,
    MIN_SPEECH_SECONDS,
    MIN_COVERAGE_FOR_END,
    END_SILENCE_SECONDS,
    FAST_END_SILENCE_SECONDS,
    COVERAGE_FOR_FAST_END,
    LONG_SILENCE_SECONDS,
    NO_SPEECH_TIMEOUT_SECONDS,
    MAX_RECORDING_SECONDS,
    EXPECTED_INDEX,
    normalize_quran_text,
    normalize_phoneme,
    get_expected_phoneme,
    calculate_phoneme_accuracy,
    finish_stream,
    _map_to_rec,
    _LETTER_VARIANTS,
)

router = APIRouter(prefix="/quran", tags=["Quran Learn"])


# ============================================================
# SETTINGS (rekebisha hapa)
# ============================================================

LEARN_PASS_THRESHOLD = 75.0   # usahihi wa jumla unaohitajika
WORD_OK_RATIO = 0.5           # neno linahesabiwa sahihi ikiwa >= 50% ya herufi zake zimelingana


# ============================================================
# WORDS
# ============================================================

def resolve_words(target_text: str, words: list | None) -> list[str]:
    """
    Tumia orodha ya maneno kutoka Flutter (SegmentWord) ikiwa inalingana na
    text. Vinginevyo gawanya text kwa nafasi. Index ya neno hapa ndiyo
    inayorudishwa kama wrong_word_index.
    """
    target_sk = normalize_quran_text(target_text)

    given = [str(w) for w in (words or []) if normalize_quran_text(str(w))]

    if given:
        joined = "".join(normalize_quran_text(w) for w in given)

        if (
            joined == target_sk
            or SequenceMatcher(None, joined, target_sk, autojunk=False).ratio() >= 0.9
        ):
            return given

    return [w for w in target_text.split() if normalize_quran_text(w)]


# ============================================================
# TEXT POSITION -> PHONEME POSITION
# ============================================================

def _align_boundaries(source_sk: str, phoneme: str, cuts: list[int]) -> list[int]:
    """
    cuts = nafasi (offsets) ndani ya source_sk (herufi za msingi za maandishi).
    Inarudisha nafasi zinazolingana ndani ya string ya phoneme (pamoja na harakat).
    """
    ph_idx: list[int] = []
    ph_letters: list[str] = []

    for i, c in enumerate(phoneme):
        if "\u0621" <= c <= "\u064A":
            ph_idx.append(i)
            ph_letters.append(c.translate(_LETTER_VARIANTS))

    ph_sk = "".join(ph_letters)

    ops = SequenceMatcher(None, source_sk, ph_sk, autojunk=False).get_opcodes()

    out: list[int] = []

    for p in cuts:
        j = _map_to_rec(ops, p)

        if j <= 0:
            pos = 0
        elif j >= len(ph_idx):
            pos = len(phoneme)
        else:
            pos = ph_idx[j]

        if out and pos < out[-1]:
            pos = out[-1]

        out.append(pos)

    return out


def build_expected(
    ayah_text: str,
    segment_text: str,
    words: list | None,
    previous_text: str = "",
):
    """
    Inarudisha (word_phonemes, word_texts) au None.

    segment_text tupu  -> Aya nzima
    segment_text ipo   -> segment (phoneme inatafutwa kwa text ya segment,
                          si kwa namba ya Aya)
    """

    is_segment = bool(segment_text and segment_text.strip())
    target_text = segment_text if is_segment else ayah_text

    ws = resolve_words(target_text, words)

    if not ws:
        return None

    skels = [normalize_quran_text(w) for w in ws]
    joined = "".join(skels)

    cuts_rel = [0]
    for s in skels:
        cuts_rel.append(cuts_rel[-1] + len(s))

    direct = None
    phoneme = None
    source_sk = ""
    base = 0

    # ---------- 1. Exact lookup ----------
    if is_segment:
        direct = EXPECTED_INDEX.get(normalize_quran_text(segment_text))
        if direct is not None:
            phoneme, source_sk, base = direct, joined, 0
    else:
        direct = get_expected_phoneme(ayah_text)
        if direct is not None:
            phoneme = direct
            source_sk = normalize_quran_text(ayah_text)
            base = 0

    # ---------- 2. Segment: kata kutoka phoneme ya Aya nzima ----------
    if phoneme is None:
        if not is_segment:
            return None

        ayah_ph = get_expected_phoneme(ayah_text)

        if ayah_ph is None:
            return None

        ayah_sk = normalize_quran_text(ayah_text)
        hint = len(normalize_quran_text(previous_text))

        pos = ayah_sk.find(joined, max(0, hint - 3))

        if pos < 0:
            pos = ayah_sk.find(joined)

        if pos < 0:
            sm = SequenceMatcher(None, ayah_sk, joined, autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]

            if not blocks:
                print(f"[learn] segment haipatikani ndani ya aya: {joined!r}")
                return None

            pos = max(0, blocks[0].a - blocks[0].b)
            print(f"[learn] fuzzy segment position = {pos}")

        phoneme, source_sk, base = ayah_ph, ayah_sk, pos

    cuts = [base + c for c in cuts_rel]
    bounds = _align_boundaries(source_sk, phoneme, cuts)

    if direct is not None:
        bounds[0] = 0
        bounds[-1] = len(phoneme)

    word_ph = [phoneme[bounds[i]:bounds[i + 1]] for i in range(len(ws))]

    if not any(word_ph):
        return None

    return word_ph, ws


# ============================================================
# EVALUATION (WORD LEVEL)
# ============================================================

def evaluate_words(expected_words: list[str], recognized: str) -> list[float]:
    """Kwa kila neno: uwiano (0..1) wa herufi zilizolingana."""

    n = len(expected_words)
    expected = "".join(expected_words)
    rec = normalize_phoneme(recognized)

    bounds = [0]
    for w in expected_words:
        bounds.append(bounds[-1] + len(w))

    if not rec:
        return [1.0 if len(w) == 0 else 0.0 for w in expected_words]

    ops = SequenceMatcher(None, expected, rec, autojunk=False).get_opcodes()

    matched = [0] * n

    for tag, i1, i2, _j1, _j2 in ops:
        if tag != "equal":
            continue

        for k in range(n):
            overlap = min(bounds[k + 1], i2) - max(bounds[k], i1)
            if overlap > 0:
                matched[k] += overlap

    ratios = []

    for k in range(n):
        length = len(expected_words[k])
        ratios.append(1.0 if length == 0 else round(matched[k] / length, 3))

    return ratios


def build_learn_result(
    mode: str,
    ayah_number: int,
    segment_number: int,
    expected_words: list[str],
    word_texts: list[str],
    recognized: str,
    reason: str,
) -> dict:

    expected = "".join(expected_words)
    recognized = normalize_phoneme(recognized)

    accuracy = calculate_phoneme_accuracy(expected, recognized)
    ratios = evaluate_words(expected_words, recognized)

    bad = [i for i, r in enumerate(ratios) if r < WORD_OK_RATIO]

    no_speech = reason == "no_speech" or not recognized

    passed = (
        not no_speech
        and accuracy >= LEARN_PASS_THRESHOLD
        and not bad
    )

    wrong_idx = None
    wrong_word = None

    if not passed and not no_speech:
        if bad:
            wrong_idx = bad[0]  # kosa la kwanza kwa mpangilio
        else:
            wrong_idx = min(range(len(ratios)), key=ratios.__getitem__)

        wrong_word = word_texts[wrong_idx]

    label = f"Segment {segment_number}" if mode == "segment" else f"Aya {ayah_number}"

    if passed:
        message = f"{label} sahihi. Usahihi wako ni {accuracy}%."
    elif no_speech:
        message = f"Sikusikia sauti yako. Jaribu tena kusoma {label}."
    elif wrong_word:
        message = f"Umekosea neno «{wrong_word}». Usahihi wako ni {accuracy}%."
    else:
        message = f"{label} haijapita. Usahihi wako ni {accuracy}%. Jaribu tena."

    return {
        "success": True,
        "mode": mode,
        "ayah_number": ayah_number,
        "segment_number": segment_number,
        "accuracy": accuracy,
        "passed": passed,
        "wrong_word": wrong_word,
        "wrong_word_index": wrong_idx,
        "word_scores": ratios,
        "word_count": len(word_texts),
        "expected_phoneme": expected,
        "recognized_phoneme": recognized,
        "sample_rate": SAMPLE_RATE,
        "reason": reason,
        "message": message,
    }


# ============================================================
# LEARN WEBSOCKET
#
# Client -> JSON:
#   {
#     "mode": "segment" | "ayah",
#     "ayah_number": 1,
#     "ayah_text": "<ayah.text NZIMA>",
#     "segment_number": 1,                 (segment tu)
#     "segment_text": "<segment.text>",    (segment tu)
#     "previous_text": "<segments zilizopita>",   (hiari)
#     "words": ["بسم", "الله"]             (SegmentWord kwa mpangilio)
#   }
# Client -> binary PCM16 16kHz mono
# Client -> text "stop"
#
# Server -> {"type":"ready"}
# Server -> {"type":"partial", ...}
# Server -> {"type":"final", passed, accuracy, wrong_word,
#            wrong_word_index, word_count, ...}
# Server -> {"type":"error","detail":"..."}
# ============================================================

@router.websocket("/ws/learn")
async def ws_learn(ws: WebSocket):

    await ws.accept()

    try:
        init = await ws.receive_json()

        mode = str(init.get("mode", "segment")).lower()
        ayah_number = int(init.get("ayah_number", 0) or 0)
        ayah_text = str(init.get("ayah_text", "") or "")
        segment_number = int(init.get("segment_number", 0) or 0)
        segment_text = str(init.get("segment_text", "") or "")
        previous_text = str(init.get("previous_text", "") or "")
        words = init.get("words") or []

        if mode not in ("segment", "ayah"):
            await ws.send_json({"type": "error", "detail": "mode si sahihi."})
            await ws.close()
            return

        if not ayah_text.strip():
            await ws.send_json({"type": "error", "detail": "ayah_text haijatumwa."})
            await ws.close()
            return

        if mode == "segment" and not segment_text.strip():
            await ws.send_json({"type": "error", "detail": "segment_text haijatumwa."})
            await ws.close()
            return

        if mode == "ayah":
            segment_text = ""

        built = build_expected(ayah_text, segment_text, words, previous_text)

        if built is None:
            what = f"Segment {segment_number}" if mode == "segment" else f"Aya {ayah_number}"
            key = normalize_quran_text(segment_text or ayah_text)
            print(f"[learn] no phoneme | {what} | key={key!r}")
            await ws.send_json({
                "type": "error",
                "detail": (
                    f"Phoneme ya {what} (Aya {ayah_number}) haikupatikana. "
                    f"Hakikisha ayah_text ni maandishi kamili ya Aya."
                ),
            })
            await ws.close()
            return

        expected_words, word_texts = built
        expected = "".join(expected_words)

        async def send_final(recognized: str, reason: str):
            await ws.send_json({
                "type": "final",
                **build_learn_result(
                    mode, ayah_number, segment_number,
                    expected_words, word_texts, recognized, reason,
                ),
            })
            await ws.close()

        stream = recognizer.create_stream()

        total_samples = 0
        speech_started = False
        speech_accum = 0.0
        last_speech_t = 0.0
        last_recognized = ""

        await ws.send_json({"type": "ready"})

        while True:

            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                return

            if message.get("text") is not None:
                if message["text"].strip().lower() == "stop":
                    recognized = finish_stream(stream)
                    await send_final(
                        recognized,
                        "client_stop" if speech_started or recognized else "no_speech",
                    )
                    return
                continue

            data = message.get("bytes")

            if not data:
                continue

            data = data[: len(data) - (len(data) % 2)]

            if not data:
                continue

            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            chunk_seconds = len(chunk) / SAMPLE_RATE
            total_samples += len(chunk)
            now_t = total_samples / SAMPLE_RATE

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if now_t > IGNORE_FIRST_SECONDS and rms > SPEECH_RMS_THRESHOLD:
                speech_accum += chunk_seconds
                last_speech_t = now_t

                if speech_accum >= MIN_SPEECH_SECONDS:
                    speech_started = True

            stream.accept_waveform(SAMPLE_RATE, chunk)

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            recognized = normalize_phoneme(recognizer.get_result(stream))
            coverage = len(recognized) / max(len(expected), 1)

            if recognized != last_recognized:
                last_recognized = recognized
                await ws.send_json({
                    "type": "partial",
                    "recognized_phoneme": recognized,
                    "progress": round(min(coverage, 1.0), 2),
                })

            if not speech_started:
                if now_t >= NO_SPEECH_TIMEOUT_SECONDS:
                    await send_final("", "no_speech")
                    return
                continue

            silence = now_t - last_speech_t

            finished = False
            reason = "completed"

            if coverage >= COVERAGE_FOR_FAST_END and silence >= FAST_END_SILENCE_SECONDS:
                finished, reason = True, "fast_end"

            elif coverage >= MIN_COVERAGE_FOR_END and silence >= END_SILENCE_SECONDS:
                finished, reason = True, "silence"

            elif silence >= LONG_SILENCE_SECONDS:
                finished, reason = True, "long_silence"

            elif now_t >= MAX_RECORDING_SECONDS:
                finished, reason = True, "max_duration"

            if finished:
                await send_final(finish_stream(stream), reason)
                return

    except WebSocketDisconnect:
        return

    except Exception as e:
        try:
            await ws.send_json({"type": "error", "detail": f"Quran AI error: {e}"})
            await ws.close()
        except Exception:
            pass


# ============================================================
# PREVIEW (kupima bila audio)
#
# POST /quran/learn/preview
# {"ayah_text": "...", "segment_text": "...", "words": ["..",".."]}
# ============================================================

class PreviewBody(BaseModel):
    ayah_text: str
    segment_text: str = ""
    previous_text: str = ""
    words: list[str] = []


@router.post("/learn/preview")
async def learn_preview(body: PreviewBody):

    built = build_expected(
        body.ayah_text, body.segment_text, body.words, body.previous_text
    )

    if built is None:
        return {"found": False}

    word_ph, word_texts = built

    return {
        "found": True,
        "words": [
            {"index": i, "text": t, "phoneme": p}
            for i, (t, p) in enumerate(zip(word_texts, word_ph))
        ],
        "expected_phoneme": "".join(word_ph),
    }
