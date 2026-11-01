from pathlib import Path
import re
import tempfile
import json
import math
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
import sherpa_onnx

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Form,
    WebSocket,
    WebSocketDisconnect,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Models" / "quran" / "zipformer_p_arabic_v3.int8.onnx"
TOKENS_PATH = BASE_DIR / "Models" / "quran" / "tokens.txt"
TEXT_TO_PHONEME_PATH = BASE_DIR / "Models" / "quran" / "quran_text2phoneme.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Quran model haipo: {MODEL_PATH}")

if not TOKENS_PATH.exists():
    raise FileNotFoundError(f"tokens.txt haipo: {TOKENS_PATH}")

if not TEXT_TO_PHONEME_PATH.exists():
    raise FileNotFoundError(
        f"quran_text2phoneme.json haipo: {TEXT_TO_PHONEME_PATH}"
    )


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
PASS_THRESHOLD = 75.0

# --- Live end-of-speech detection (kwa muda wa audio) ---
SPEECH_RMS_THRESHOLD = 0.02
IGNORE_FIRST_SECONDS = 0.6
MIN_SPEECH_SECONDS = 0.3
MIN_COVERAGE_FOR_END = 0.30
END_SILENCE_SECONDS = 1.2
FAST_END_SILENCE_SECONDS = 0.5
COVERAGE_FOR_FAST_END = 0.90
LONG_SILENCE_SECONDS = 3.0
NO_SPEECH_TIMEOUT_SECONDS = 10.0
MAX_RECORDING_SECONDS = 30.0

FUZZY_MIN_SCORE = 0.85


# ============================================================
# NORMALIZE
# ============================================================

_QURAN_MARKS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF"
    "\u200c\u200d\u200e\u200f]"
)

_LETTER_VARIANTS = str.maketrans({
    "ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا",
    "ى": "ي", "ی": "ي", "ک": "ك",
})


def normalize_quran_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ـ", "")
    text = _QURAN_MARKS.sub("", text)
    text = text.translate(_LETTER_VARIANTS)
    text = "".join(c for c in text if "\u0621" <= c <= "\u064A")
    return text


def normalize_phoneme(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = "".join(text.split())
    return text


# ============================================================
# LOAD TEXT → PHONEME DATA (+ FAST INDEX)
# ============================================================

print("Loading Quran text-to-phoneme data...")

with open(TEXT_TO_PHONEME_PATH, "r", encoding="utf-8") as f:
    quran_text2phoneme = json.load(f)

EXPECTED_INDEX: dict[str, str] = {
    normalize_quran_text(text): normalize_phoneme(phoneme)
    for text, phoneme in quran_text2phoneme.items()
}

print(f"Loaded {len(EXPECTED_INDEX)} Quran text-to-phoneme entries.")


# ============================================================
# LOAD QURAN AI MODEL
# ============================================================

print("Loading Quran AI model...")

recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
    tokens=str(TOKENS_PATH),
    model=str(MODEL_PATH),
    num_threads=2,
    provider="cpu",
    sample_rate=SAMPLE_RATE,
    feature_dim=80,
    decoding_method="greedy_search",
)

print("Quran AI model loaded successfully.")


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/quran",
    tags=["Quran AI"],
)


# ============================================================
# HELPERS
# ============================================================

def get_expected_phoneme(ayah_text: str) -> str | None:
    key = normalize_quran_text(ayah_text)

    exact = EXPECTED_INDEX.get(key)
    if exact is not None:
        return exact

    best_phoneme = None
    best_score = 0.0

    for source_key, phoneme in EXPECTED_INDEX.items():
        if abs(len(source_key) - len(key)) > max(5, len(key) * 0.3):
            continue

        score = SequenceMatcher(None, key, source_key).ratio()

        if score > best_score:
            best_score = score
            best_phoneme = phoneme

    if best_phoneme is not None and best_score >= FUZZY_MIN_SCORE:
        print(f"[fuzzy] match {best_score:.2%}")
        EXPECTED_INDEX[key] = best_phoneme
        return best_phoneme

    print("=" * 60)
    print(f"[no match] best={best_score:.2%}")
    print(f"received text : {ayah_text!r}")
    print(f"normalized key: {key!r}")
    print(f"key length    : {len(key)}")
    print("=" * 60)
    return None


def calculate_phoneme_accuracy(expected: str, recognized: str) -> float:
    expected = normalize_phoneme(expected)
    recognized = normalize_phoneme(recognized)

    if not expected or not recognized:
        return 0.0

    ratio = SequenceMatcher(None, expected, recognized).ratio()
    accuracy = max(0.0, min(100.0, ratio * 100))
    return round(accuracy, 2)


def build_result(
    ayah_number: int,
    expected: str,
    recognized: str,
    reason: str = "completed",
) -> dict:
    accuracy = calculate_phoneme_accuracy(expected, recognized)
    passed = accuracy >= PASS_THRESHOLD

    if passed:
        message = f"Umeipata Aya {ayah_number}. Usahihi wako ni {accuracy}%."
    else:
        message = (
            f"Hujapata Aya {ayah_number}. "
            f"Usahihi wako ni {accuracy}%. Jaribu tena."
        )

    return {
        "success": True,
        "ayah_number": ayah_number,
        "accuracy": accuracy,
        "passed": passed,
        "expected_phoneme": expected,
        "recognized_phoneme": recognized,
        "sample_rate": SAMPLE_RATE,
        "reason": reason,
        "message": message,
    }


def finish_stream(stream) -> str:
    """Maliza stream (tail padding) na rudisha matokeo ya mwisho."""
    tail_padding = np.zeros(int(0.66 * SAMPLE_RATE), dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, tail_padding)
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    return normalize_phoneme(recognizer.get_result(stream))


# ============================================================
# LIVE (WEBSOCKET) CHECK  /quran/ws/recite
# ============================================================

@router.websocket("/ws/recite")
async def ws_recite(ws: WebSocket):

    await ws.accept()

    try:
        init = await ws.receive_json()

        ayah_number = int(init.get("ayah_number", 0))
        ayah_text = str(init.get("ayah_text", ""))

        if ayah_number < 1 or not ayah_text.strip():
            await ws.send_json({
                "type": "error",
                "detail": "Taarifa za Aya si sahihi.",
            })
            await ws.close()
            return

        expected = get_expected_phoneme(ayah_text)

        if expected is None:
            await ws.send_json({
                "type": "error",
                "detail": (
                    f"Phoneme ya Aya {ayah_number} haikupatikana "
                    f"kwenye quran_text2phoneme.json."
                ),
            })
            await ws.close()
            return

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
                    await ws.send_json({
                        "type": "final",
                        **build_result(
                            ayah_number, expected, recognized, "client_stop"
                        ),
                    })
                    await ws.close()
                    return
                continue

            data = message.get("bytes")

            if not data:
                continue

            data = data[: len(data) - (len(data) % 2)]

            if not data:
                continue

            chunk = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

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
                    await ws.send_json({
                        "type": "final",
                        **build_result(ayah_number, expected, "", "no_speech"),
                    })
                    await ws.close()
                    return
                continue

            silence = now_t - last_speech_t

            finished = False
            reason = "completed"

            if (
                coverage >= COVERAGE_FOR_FAST_END
                and silence >= FAST_END_SILENCE_SECONDS
            ):
                finished, reason = True, "fast_end"

            elif (
                coverage >= MIN_COVERAGE_FOR_END
                and silence >= END_SILENCE_SECONDS
            ):
                finished, reason = True, "silence"

            elif silence >= LONG_SILENCE_SECONDS:
                finished, reason = True, "long_silence"

            elif now_t >= MAX_RECORDING_SECONDS:
                finished, reason = True, "max_duration"

            if finished:
                final_recognized = finish_stream(stream)

                await ws.send_json({
                    "type": "final",
                    **build_result(
                        ayah_number, expected, final_recognized, reason
                    ),
                })
                await ws.close()
                return

    except WebSocketDisconnect:
        return

    except Exception as e:
        try:
            await ws.send_json({
                "type": "error",
                "detail": f"Quran AI error: {e}",
            })
            await ws.close()
        except Exception:
            pass


# ============================================================
# MEMORIZATION (TAHFIDH) - CONTINUOUS LIVE CHECK
#
# Server -> {"type": "final",
#            "status": "wrong | missing | skipped | no_speech | all_done",
#            "ayah_index": 2, "message": "...",
#            "expected_letter": "ح", "heard_letter": "ه"}
#
#   wrong   = herufi / harakat / matamshi yamekosewa  -> Flutter: ijii.ogg
#   missing = neno / herufi nyingi zimeachwa          -> Flutter: kengele
#   skipped = Aya imerukwa                            -> Flutter: kengele
# ============================================================

# --- Ukali (rekebisha hapa) ---

# True  = harakat (fatha/damma/kasra/tanwin) zinakaguliwa (rabbi != rabba)
# False = herufi tu (kama zamani). Weka False kama kengele zinalia bila sababu.
STRICT_VOWELS = True

MEM_MISSING_LETTERS = 1        # herufi zilizokosekana kabla ya kuhesabiwa kosa
MEM_WORD_MISSING_LETTERS = 4   # >= hii = neno zima limeachwa (kengele); chini yake = kosa la herufi (ijii)
MEM_CONFIRM_LETTERS = 3        # herufi sahihi zinazofuata pengo ili kuthibitisha kosa
MEM_WINDOW_MARGIN = 25
MEM_COMPLETE_RATIO = 0.7
MEM_STUCK_SILENCE_SECONDS = 3.0
MEM_NO_SPEECH_TIMEOUT_SECONDS = 8.0
MEM_MAX_SECONDS = 900.0

MEM_SKIP_RATIO = 0.60
MEM_SKIP_MIN_LETTERS = 8
MEM_SKIP_MARGIN = 0.15

# --- Herufi zinazochanganywa (haa laini ه / haa nzito ح, n.k.) ---
# Mtu akibadilisha herufi moja kwa nyingine kwenye kundi hapa chini,
# ni kosa la HERUFI (ijii.ogg), si neno lililoachwa (kengele).
MEM_CONFUSABLE_CONFIRM_LETTERS = 1   # herufi sahihi zinazohitajika baada ya kosa (kwa herufi hizi)
MEM_SWAP_MAX_BLOCK = 6               # kipande kikubwa kuliko hiki hakihesabiwi kama kubadilisha herufi

# --- Uhakiki wa uhakika wa sauti (confidence) ---
# Modeli mara nyingi "inasahihisha" makosa na kutoa herufi sahihi hata user
# akikosea. Lakini huitoa kwa uhakika mdogo. Herufi (si harakat) iliyosikika
# kwa uhakika chini ya MEM_LOW_CONF inahesabiwa kosa la herufi -> ijii.
#   - Ikiwa ijii inalia sana bila sababu  -> PUNGUZA (mfano 0.45)
#   - Ikiwa makosa bado hayakamatwi       -> ONGEZA (mfano 0.70)
USE_CONFIDENCE_CHECK = True
MEM_LOW_CONF = 0.60
MEM_LOW_CONF_CONFIRM = 0     # herufi zinazofuata kabla ya kuthibitisha (0 = mara moja)
MEM_DEBUG = True             # chapisha modeli inasikia nini

_CONFUSABLE_GROUPS = [
    "هح",    # haa laini / haa nzito
    "حخ",    # haa nzito / khaa
    "هخ",
    "تط",    # taa / twaa
    "طد",
    "سص",    # siin / swaad
    "ثس",    # thaa / siin
    "ذز",    # dhaal / zayn
    "ذظ",    # dhaal / dhwaa
    "زظ",
    "دض",    # daal / dwaad
    "صض",
    "قك",    # qaaf / kaaf
    "قغ",
    "غخ",
    "كخ",
    "عء",    # ayn / hamza
]

_CONFUSABLE_PAIRS: set[tuple[str, str]] = set()

for _g in _CONFUSABLE_GROUPS:
    for _a in _g:
        for _b in _g:
            if _a != _b:
                _CONFUSABLE_PAIRS.add((_a, _b))

# herufi + harakat za vokali (fatha, damma, kasra, tanwin). Shadda/sukun
# zimeachwa nje kwa makusudi kwa sababu model huzitambua si thabiti.
_NON_LETTERS = re.compile("[^\u0621-\u064A]")
_NON_LETTERS_OR_VOWELS = re.compile("[^\u0621-\u064A\u064B-\u0650]")


def to_skeleton(text: str) -> str:
    """
    Herufi za msingi (na vokali fupi kama STRICT_VOWELS=True).
    Herufi zinazojirudia mfululizo zinakuwa moja, ili urefu wa
    madd/ghunna usilete kengele za uongo.
    """
    text = normalize_phoneme(text)

    pattern = _NON_LETTERS_OR_VOWELS if STRICT_VOWELS else _NON_LETTERS
    text = pattern.sub("", text)

    out: list[str] = []

    for ch in text:
        if not out or out[-1] != ch:
            out.append(ch)

    return "".join(out)


def find_letter_swap(expected_chunk: str, heard_chunk: str):
    """
    Ikiwa kipande kilichotarajiwa na kilichosikika vina herufi zinazochanganywa
    (mfano ح na ه), rudisha (expected_letter, heard_letter). Vinginevyo None.
    """
    exp = _NON_LETTERS.sub("", expected_chunk)
    heard = _NON_LETTERS.sub("", heard_chunk)

    for a in exp:
        for b in heard:
            if (a, b) in _CONFUSABLE_PAIRS:
                return a, b

    return None


def first_letter_diff(expected_chunk: str, heard_chunk: str):
    """Herufi ya kwanza inayotofautiana (kwa kuonyesha user). Au None."""
    exp = _NON_LETTERS.sub("", expected_chunk)
    heard = _NON_LETTERS.sub("", heard_chunk)

    for a, b in zip(exp, heard):
        if a != b:
            return a, b

    return None


def get_token_probs(stream, text: str):
    """
    Rudisha list ya (herufi, uhakika) inayolingana 1:1 na to_skeleton(text),
    au None kama sherpa-onnx yako haitoi probabilities.
    """
    if not USE_CONFIDENCE_CHECK or not hasattr(recognizer, "get_result_all"):
        return None

    try:
        r = recognizer.get_result_all(stream)
    except Exception:
        return None

    tokens = list(getattr(r, "tokens", None) or [])
    probs = list(getattr(r, "ys_probs", None) or [])

    if not tokens or len(tokens) != len(probs):
        return None

    pattern = _NON_LETTERS_OR_VOWELS if STRICT_VOWELS else _NON_LETTERS
    out: list[tuple[str, float]] = []

    for tok, p in zip(tokens, probs):
        try:
            p = float(p)
        except Exception:
            return None

        if p <= 0.0:
            p = math.exp(p)          # ilikuwa log-prob
        p = min(p, 1.0)

        for ch in pattern.sub("", normalize_phoneme(str(tok))):
            if out and out[-1][0] == ch:
                out[-1] = (ch, max(out[-1][1], p))
            else:
                out.append((ch, p))

    if "".join(c for c, _ in out) != to_skeleton(text):
        return None

    return out


def _map_to_rec(ops, i_target: int) -> int:
    for tag, i1, i2, j1, j2 in ops:
        if i1 <= i_target <= i2:
            if tag == "equal":
                return j1 + (i_target - i1)
            return j2 if i_target > i1 else j1

    return ops[-1][4] if ops else 0


class MemorizeTracker:

    def __init__(self, full: str, starts: list[int], ends: list[int]):
        self.full = full
        self.starts = starts
        self.ends = ends
        self.n = len(starts)

        self.k = 0
        self.base_j = 0

    @staticmethod
    def _bell(
        status: str,
        ayah_index: int,
        missing: int = 0,
        expected_letter: str | None = None,
        heard_letter: str | None = None,
    ) -> dict:
        return {
            "status": status,
            "ayah_index": ayah_index,
            "missing": missing,
            "expected_letter": expected_letter,
            "heard_letter": heard_letter,
        }

    def _ayah_at(self, absolute_pos: int) -> int:
        for idx in range(self.k, self.n):
            if absolute_pos < self.ends[idx]:
                return idx
        return self.n - 1

    def _detect_skip(self, rec: str) -> bool:
        n = min(len(rec), 30)

        if n < MEM_SKIP_MIN_LETTERS:
            return False

        head = rec[:n]
        k = self.k

        current = self.full[self.starts[k]: self.starts[k] + n]
        current_score = SequenceMatcher(None, current, head).ratio()

        for j in (k + 1, k + 2):
            if j >= self.n:
                break

            nxt = self.full[self.starts[j]: self.starts[j] + n]
            score = SequenceMatcher(None, nxt, head).ratio()

            if score >= MEM_SKIP_RATIO and score > current_score + MEM_SKIP_MARGIN:
                return True

        return False

    def _low_conf_event(self, ops, window, rec, seg_start, final):
        """Herufi iliyosikika kwa uhakika mdogo = kosa la herufi (ijii)."""
        probs = self.probs

        if not probs or len(probs) != len(self.rec_full):
            return None

        total = len(rec)

        for tag, i1, i2, j1, j2 in ops:
            if tag != "equal":
                continue

            for off in range(j2 - j1):
                j = j1 + off
                exp_ch = window[i1 + off]

                if not ("\u0621" <= exp_ch <= "\u064A"):
                    continue          # harakat hazikaguliwi hapa

                ch, p = probs[self.base_j + j]

                if p >= MEM_LOW_CONF:
                    continue

                after = total - (j + 1)

                if not final and after < MEM_LOW_CONF_CONFIRM:
                    continue

                ayah_idx = self._ayah_at(seg_start + i1 + off)

                if MEM_DEBUG:
                    print(
                        f"[memorize] low-confidence letter {exp_ch!r} "
                        f"p={p:.2f} (< {MEM_LOW_CONF})"
                    )

                return self._bell("wrong", ayah_idx, 0, exp_ch, None)

        return None

    def update(self, recognized: str, final: bool = False, probs=None):
        """
        Rudisha (done, event)
          done  : orodha ya index za aya zilizokamilika sasa hivi
          event : None | {"status", "ayah_index", "missing",
                          "expected_letter", "heard_letter"}
        """

        rec_full = to_skeleton(recognized)
        self.rec_full = rec_full
        self.probs = probs
        done: list[int] = []

        while self.k < self.n:

            k = self.k
            seg_start = self.starts[k]
            seg_end = self.ends[k]
            seg_len = seg_end - seg_start

            if seg_len <= 0:
                done.append(k)
                self.k += 1
                continue

            rec = rec_full[self.base_j:]

            if not rec:
                if final:
                    status = "no_speech" if k == 0 else "missing"
                    return done, self._bell(status, k, seg_len)
                return done, None

            win_end = min(
                self.ends[min(k + 1, self.n - 1)],
                seg_start + len(rec) + MEM_WINDOW_MARGIN,
            )

            window = self.full[seg_start:win_end]

            ops = SequenceMatcher(
                None, window, rec, autojunk=False
            ).get_opcodes()

            # ---------- 1. Ameruka aya ----------
            if self._detect_skip(rec):
                return done, self._bell("skipped", k, seg_len)

            # ---------- 2. Kosa la herufi / neno ----------
            for idx, (tag, i1, i2, j1, j2) in enumerate(ops):

                if tag not in ("delete", "replace"):
                    continue

                missing = (i2 - i1) - (j2 - j1)

                # Je, ni kubadilisha herufi (mfano ح -> ه)?
                swap = None
                if tag == "replace" and (i2 - i1) <= MEM_SWAP_MAX_BLOCK:
                    swap = find_letter_swap(window[i1:i2], rec[j1:j2])

                if swap is not None:
                    kind = "wrong"            # herufi imebadilishwa -> ijii
                elif missing >= MEM_WORD_MISSING_LETTERS:
                    kind = "missing"          # neno limeachwa -> kengele
                elif missing >= MEM_MISSING_LETTERS or tag == "replace":
                    kind = "wrong"            # herufi/harakat zimekosewa -> ijii
                else:
                    continue

                if final:
                    confirm = 0
                elif swap is not None:
                    confirm = MEM_CONFUSABLE_CONFIRM_LETTERS
                else:
                    confirm = MEM_CONFIRM_LETTERS

                after = sum(
                    b2 - b1
                    for t, a1, a2, b1, b2 in ops[idx + 1:]
                    if t == "equal"
                )

                if after < confirm:
                    continue

                ayah_idx = self._ayah_at(seg_start + i1)
                ayah_len = max(self.ends[ayah_idx] - self.starts[ayah_idx], 1)

                if kind == "wrong":
                    status = "wrong"
                elif after > 0 and missing >= max(10, MEM_SKIP_RATIO * ayah_len):
                    status = "skipped"
                else:
                    status = "missing"

                exp_letter = None
                heard_letter = None

                if status == "wrong" and tag == "replace":
                    pair = swap or first_letter_diff(window[i1:i2], rec[j1:j2])
                    if pair is not None:
                        exp_letter, heard_letter = pair

                return done, self._bell(
                    status,
                    ayah_idx,
                    max(missing, 0),
                    exp_letter,
                    heard_letter,
                )

            # ---------- 2b. Herufi iliyosikika kwa uhakika mdogo ----------
            conf_event = self._low_conf_event(
                ops, window, rec, seg_start, final
            )

            if conf_event is not None:
                return done, conf_event

            # ---------- 3. Aya imekamilika? ----------
            matched_end = 0
            matched_in_seg = 0
            pending_in_seg = 0

            for tag, i1, i2, j1, j2 in ops:
                if tag == "equal":
                    matched_end = max(matched_end, i2)

                    if i1 < seg_len:
                        matched_in_seg += min(i2, seg_len) - i1

                elif tag in ("delete", "replace") and i1 < seg_len:
                    gap = (i2 - i1) - (j2 - j1)

                    if gap > 0:
                        pending_in_seg += gap
                    elif tag == "replace":
                        pending_in_seg += 1   # herufi imebadilishwa

            if (
                matched_end >= seg_len
                and matched_in_seg >= seg_len * MEM_COMPLETE_RATIO
                and pending_in_seg < MEM_MISSING_LETTERS
            ):
                self.base_j += _map_to_rec(ops, seg_len)
                done.append(k)
                self.k += 1
                continue

            # ---------- 4. Amemaliza kusoma lakini aya haijakamilika ----------
            if final:
                return done, self._bell(
                    "missing", k, max(1, seg_len - matched_in_seg)
                )

            return done, None

        return done, {
            "status": "all_done",
            "ayah_index": self.n - 1,
            "missing": 0,
            "expected_letter": None,
            "heard_letter": None,
        }


def memorize_message(status: str, number: int) -> str:
    if status == "all_done":
        return "Hongera! Umemaliza Aya zote."

    if status == "skipped":
        return f"Umeruka Aya. Rudi usome Aya {number}."

    if status == "no_speech":
        return f"Hukusoma chochote. Soma Aya {number} kutoka moyoni."

    if status == "wrong":
        return f"Umekosea herufi au matamshi kwenye Aya {number}. Rudi usome tena."

    return f"Umeacha neno au herufi kwenye Aya {number}. Soma tena."


@router.websocket("/ws/memorize")
async def ws_memorize(ws: WebSocket):

    await ws.accept()

    try:
        init = await ws.receive_json()

        items = init.get("ayahs") or []

        if not items:
            await ws.send_json({
                "type": "error",
                "detail": "Hakuna Aya zilizotumwa.",
            })
            await ws.close()
            return

        full = ""
        starts: list[int] = []
        ends: list[int] = []
        numbers: list[int] = []

        for item in items:
            number = int(item.get("number", 0))
            text = str(item.get("text", ""))

            phoneme = get_expected_phoneme(text) if text.strip() else None

            if phoneme is None:
                await ws.send_json({
                    "type": "error",
                    "detail": (
                        f"Phoneme ya Aya {number} haikupatikana "
                        f"kwenye quran_text2phoneme.json."
                    ),
                })
                await ws.close()
                return

            starts.append(len(full))

            for ch in to_skeleton(phoneme):
                if not full or full[-1] != ch:
                    full += ch

            ends.append(len(full))
            numbers.append(number)

        print(f"[memorize] expected skeleton (strict={STRICT_VOWELS}): {full[:80]}...")

        tracker = MemorizeTracker(full, starts, ends)

        stream = recognizer.create_stream()

        total_samples = 0
        speech_started = False
        speech_accum = 0.0
        last_speech_t = 0.0
        last_event_t = 0.0
        last_recognized = ""
        probs_warned = False

        await ws.send_json({"type": "ready"})

        async def emit(done: list[int], event: dict | None, now_t: float) -> bool:
            nonlocal last_event_t

            for idx in done:
                await ws.send_json({"type": "ayah_done", "index": idx})
                last_event_t = now_t

            if event is None:
                return False

            status = event["status"]
            ayah_index = max(0, min(event.get("ayah_index", 0), len(numbers) - 1))
            number = numbers[ayah_index]

            print(
                f"[memorize] status={status} aya={number} "
                f"missing={event.get('missing', 0)} "
                f"expected={event.get('expected_letter')} "
                f"heard={event.get('heard_letter')}"
            )

            await ws.send_json({
                "type": "final",
                "status": status,
                "passed": status == "all_done",
                "ayah_index": ayah_index,
                "ayah_number": number,
                "missing_letters": event.get("missing", 0),
                "expected_letter": event.get("expected_letter"),
                "heard_letter": event.get("heard_letter"),
                "message": memorize_message(status, number),
            })

            await ws.close()
            return True

        while True:

            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                return

            if message.get("text") is not None:
                if message["text"].strip().lower() == "stop":
                    final_text = finish_stream(stream)
                    done, event = tracker.update(
                        final_text,
                        final=True,
                        probs=get_token_probs(stream, final_text),
                    )
                    if event is None:
                        event = {
                            "status": "missing",
                            "ayah_index": tracker.k,
                            "missing": 0,
                        }
                    await emit(done, event, total_samples / SAMPLE_RATE)
                    return
                continue

            data = message.get("bytes")

            if not data:
                continue

            data = data[: len(data) - (len(data) % 2)]

            if not data:
                continue

            chunk = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

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

            if recognized != last_recognized:
                last_recognized = recognized

                probs = get_token_probs(stream, recognized)

                if MEM_DEBUG:
                    print(f"[memorize] heard   : {to_skeleton(recognized)}")

                    if USE_CONFIDENCE_CHECK and probs is None and not probs_warned:
                        probs_warned = True
                        print(
                            "[memorize] confidence HAIPATIKANI: sherpa-onnx "
                            "haitoi ys_probs. Sasisha: pip install -U sherpa-onnx"
                        )

                done, event = tracker.update(recognized, probs=probs)

                if await emit(done, event, now_t):
                    return

            if now_t > IGNORE_FIRST_SECONDS:

                active = speech_started or tracker.k > 0

                limit = (
                    MEM_STUCK_SILENCE_SECONDS
                    if active
                    else MEM_NO_SPEECH_TIMEOUT_SECONDS
                )

                idle = now_t - max(last_speech_t, last_event_t)

                if idle >= limit or now_t >= MEM_MAX_SECONDS:

                    final_text = finish_stream(stream)
                    done, event = tracker.update(
                        final_text,
                        final=True,
                        probs=get_token_probs(stream, final_text),
                    )

                    if event is None:
                        event = {
                            "status": "missing",
                            "ayah_index": tracker.k,
                            "missing": 0,
                        }

                    await emit(done, event, now_t)
                    return

    except WebSocketDisconnect:
        return

    except Exception as e:
        try:
            await ws.send_json({
                "type": "error",
                "detail": f"Quran AI error: {e}",
            })
            await ws.close()
        except Exception:
            pass


# ============================================================
# OLD HTTP ENDPOINT (backup)
# ============================================================

@router.post("/check-recitation")
async def check_recitation(
    audio: UploadFile = File(...),
    ayah_number: int = Form(...),
    ayah_text: str = Form(...),
):

    if ayah_number < 1:
        raise HTTPException(400, "Namba ya Aya si sahihi.")

    if not ayah_text.strip():
        raise HTTPException(400, "Maandishi ya Aya hayajatumwa.")

    expected = get_expected_phoneme(ayah_text)

    if expected is None:
        raise HTTPException(
            400,
            f"Phoneme ya Aya {ayah_number} haikupatikana "
            f"kwenye quran_text2phoneme.json.",
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(400, "Audio file ni tupu.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = Path(tmp.name)

        samples, sample_rate = sf.read(str(temp_path), dtype="float32")

        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        if sample_rate != SAMPLE_RATE:
            raise HTTPException(
                400,
                f"Audio lazima iwe 16000 Hz. Imepokelewa {sample_rate} Hz.",
            )

        if len(samples) == 0:
            raise HTTPException(400, "Audio haina data.")

        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        recognized = finish_stream(stream)

        return build_result(ayah_number, expected, recognized)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(500, f"Quran AI error: {e}")

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()