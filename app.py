from pathlib import Path
from dataclasses import dataclass

import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "TBOGamer22/wav2vec2-quran-phonetics"

BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE = BASE_DIR / "audio" / "test.wav"

SAMPLE_RATE = 16000

REFERENCE_WORDS = [
    "al-ḥamdu",
    "lillahi",
    "rabbi",
    "l-ʿālamīna",
]

WORD_LABELS_ARABIC = [
    "الْحَمْدُ",
    "لِلَّهِ",
    "رَبِّ",
    "الْعَالَمِينَ",
]


# ============================================================
# VERSION 6 SEGMENTATION SETTINGS
# ============================================================

WORD_DURATION_RANGES = [
    (0.45, 1.10),
    (0.45, 1.15),
    (0.25, 0.75),
    (1.00, None),  # Final word: flexible for madd
]

MAX_TRAILING_AUDIO = 0.35


# ============================================================
# DATA CLASS
# ============================================================


@dataclass
class WordBoundary:
    word_index: int
    word: str
    arabic: str

    start_frame: int
    end_frame: int

    start_time: float
    end_time: float

    duration: float
    segmentation_score: float


# ============================================================
# HELPERS
# ============================================================


def word_to_token_ids(word, processor):

    vocab = processor.tokenizer.get_vocab()

    token_ids = []

    for char in word:

        if char not in vocab:
            raise ValueError(
                f"Character {char!r} from {word!r} " f"is not in the model vocabulary."
            )

        token_ids.append(vocab[char])

    return token_ids


def ctc_segment_score(
    log_probs,
    token_ids,
    start_frame,
    end_frame,
    blank_id,
):
    """
    CTC Viterbi score for a target word inside
    a particular frame interval.
    """

    if end_frame <= start_frame:
        return float("-inf")

    segment = log_probs[start_frame:end_frame]

    if segment.shape[0] < len(token_ids):
        return float("-inf")

    expanded = [blank_id]

    for token in token_ids:
        expanded.append(token)
        expanded.append(blank_id)

    states = len(expanded)

    dp = torch.full(
        (states,),
        float("-inf"),
        dtype=segment.dtype,
        device=segment.device,
    )

    dp[0] = segment[0, blank_id]

    if states > 1:
        dp[1] = segment[0, expanded[1]]

    for t in range(1, segment.shape[0]):

        new_dp = torch.full_like(
            dp,
            float("-inf"),
        )

        for state in range(states):

            best_previous = dp[state]

            if state > 0:

                best_previous = torch.maximum(
                    best_previous,
                    dp[state - 1],
                )

            if state > 1:

                current_token = expanded[state]
                previous_token = expanded[state - 2]

                if current_token != blank_id and current_token != previous_token:

                    best_previous = torch.maximum(
                        best_previous,
                        dp[state - 2],
                    )

            new_dp[state] = best_previous + segment[t, expanded[state]]

        dp = new_dp

    return torch.maximum(
        dp[-1],
        dp[-2],
    ).item()


def frame_to_seconds(
    frame,
    frame_duration,
):

    return frame * frame_duration


def decode_ctc(
    logits,
    processor,
    blank_id,
):
    """
    Greedy CTC decoding.

    Returns:
        decoded phonetic string
        average confidence
    """

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    best_ids = torch.argmax(
        probabilities,
        dim=-1,
    )

    best_confidences = torch.max(
        probabilities,
        dim=-1,
    ).values

    # CTC collapse
    collapsed = []

    previous_id = None

    for token_id in best_ids.tolist():

        # Remove repeated tokens
        if token_id == previous_id:
            continue

        previous_id = token_id

        # Remove blank
        if token_id == blank_id:
            continue

        collapsed.append(token_id)

    decoded = processor.tokenizer.decode(
        collapsed,
        group_tokens=False,
    )

    # Confidence of the frame-level predictions.
    #
    # This is NOT a word correctness probability.
    # We only report it as an acoustic confidence indicator.
    confidence = best_confidences.mean().item() * 100

    return decoded, confidence


def normalize_phonetics(text):
    """
    Normalize spacing for comparison.
    """

    return text.replace(
        " ",
        "",
    )


def levenshtein_distance(
    a,
    b,
):
    """
    Character-level Levenshtein distance.
    """

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):

        for j in range(1, cols):

            if a[i - 1] == b[j - 1]:

                cost = 0

            else:

                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def similarity_score(
    reference,
    detected,
):
    """
    Simple normalized edit similarity.

    This is NOT a Tajweed score.
    """

    reference = normalize_phonetics(reference)

    detected = normalize_phonetics(detected)

    if len(reference) == 0:
        return 0.0

    distance = levenshtein_distance(
        reference,
        detected,
    )

    similarity = 1 - distance / len(reference)

    return (
        max(
            0.0,
            similarity,
        )
        * 100
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 90)
print("LOADING MODEL")
print("=" * 90)

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

model.eval()

blank_id = model.config.pad_token_id

print(f"CTC blank ID: {blank_id}")


# ============================================================
# LOAD AUDIO
# ============================================================

print()
print("Loading audio...")

audio, sample_rate = sf.read(
    AUDIO_FILE,
    dtype="float32",
)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if sample_rate != SAMPLE_RATE:

    raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, " f"got {sample_rate} Hz.")

audio_duration = len(audio) / sample_rate

print(f"Audio:       {AUDIO_FILE.name}")

print(f"Sample rate: {sample_rate} Hz")

print(f"Duration:    " f"{audio_duration:.2f} seconds")


# ============================================================
# RUN MODEL ON WHOLE AYAH
# ============================================================

print()
print("Running model on whole ayah...")

whole_inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt",
)

with torch.no_grad():

    whole_outputs = model(
        input_values=whole_inputs.input_values,
    )

whole_logits = whole_outputs.logits

whole_log_probs = torch.log_softmax(
    whole_logits,
    dim=-1,
)[0]

num_frames = whole_log_probs.shape[0]

input_to_logits_ratio = getattr(
    model.config,
    "inputs_to_logits_ratio",
    320,
)

frame_duration = input_to_logits_ratio / SAMPLE_RATE

print(f"Logits shape: " f"{tuple(whole_logits.shape)}")

print(f"Number of CTC frames: " f"{num_frames}")

print(f"Frame duration: " f"{frame_duration * 1000:.2f} ms")


# ============================================================
# REFERENCE TOKEN IDS
# ============================================================

print()
print("=" * 90)
print("REFERENCE WORDS")
print("=" * 90)

word_token_ids = []

for i, word in enumerate(REFERENCE_WORDS):

    token_ids = word_to_token_ids(
        word,
        processor,
    )

    word_token_ids.append(token_ids)

    print()
    print(f"{i + 1}. " f"{WORD_LABELS_ARABIC[i]}")

    print(f"   Phonetics: {word}")

    print(f"   Token IDs: {token_ids}")


# ============================================================
# DURATION RANGES IN FRAMES
# ============================================================

duration_ranges_frames = []

for minimum_seconds, maximum_seconds in WORD_DURATION_RANGES:

    minimum_frames = max(
        1,
        round(minimum_seconds / frame_duration),
    )

    if maximum_seconds is None:

        maximum_frames = None

    else:

        maximum_frames = min(
            num_frames,
            round(maximum_seconds / frame_duration),
        )

    duration_ranges_frames.append(
        (
            minimum_frames,
            maximum_frames,
        )
    )


# ============================================================
# VERSION 6 SEGMENTATION
# ============================================================

print()
print("=" * 90)
print("FINDING WORD BOUNDARIES")
print("=" * 90)

max_trailing_frames = round(MAX_TRAILING_AUDIO / frame_duration)

minimum_final_end_frame = max(
    1,
    num_frames - max_trailing_frames,
)

num_words = len(REFERENCE_WORDS)

dp = [[float("-inf")] * (num_frames + 1) for _ in range(num_words)]

back = [[None] * (num_frames + 1) for _ in range(num_words)]


# ============================================================
# FIRST WORD
# ============================================================

minimum_frames, maximum_frames = duration_ranges_frames[0]

for end_frame in range(
    minimum_frames,
    maximum_frames + 1,
):

    score = ctc_segment_score(
        whole_log_probs,
        word_token_ids[0],
        0,
        end_frame,
        blank_id,
    )

    dp[0][end_frame] = score


# ============================================================
# REMAINING WORDS
# ============================================================

for word_index in range(
    1,
    num_words,
):

    minimum_frames, maximum_frames = duration_ranges_frames[word_index]

    for end_frame in range(
        1,
        num_frames + 1,
    ):

        # Final word
        if word_index == num_words - 1:

            if end_frame < minimum_final_end_frame:
                continue

            start_min = 0

            start_max = end_frame - minimum_frames

        # Normal word
        else:

            start_min = max(
                0,
                end_frame - maximum_frames,
            )

            start_max = end_frame - minimum_frames

        if start_max < start_min:
            continue

        best_score = float("-inf")
        best_start = None

        for start_frame in range(
            start_min,
            start_max + 1,
        ):

            previous_score = dp[word_index - 1][start_frame]

            if previous_score == float("-inf"):
                continue

            current_score = ctc_segment_score(
                whole_log_probs,
                word_token_ids[word_index],
                start_frame,
                end_frame,
                blank_id,
            )

            total_score = previous_score + current_score

            if total_score > best_score:

                best_score = total_score
                best_start = start_frame

        dp[word_index][end_frame] = best_score

        back[word_index][end_frame] = best_start


# ============================================================
# BEST FINAL END
# ============================================================

best_final_score = float("-inf")
best_final_frame = None

for end_frame in range(
    minimum_final_end_frame,
    num_frames + 1,
):

    score = dp[-1][end_frame]

    if score > best_final_score:

        best_final_score = score
        best_final_frame = end_frame


if best_final_frame is None:

    raise RuntimeError("No valid word segmentation found.")


# ============================================================
# BACKTRACK
# ============================================================

boundaries = []

current_end = best_final_frame

for word_index in reversed(range(num_words)):

    if word_index == 0:

        start_frame = 0

    else:

        start_frame = back[word_index][current_end]

        if start_frame is None:

            raise RuntimeError("Could not reconstruct " "word boundaries.")

    boundaries.append(
        (
            word_index,
            start_frame,
            current_end,
        )
    )

    current_end = start_frame


boundaries.reverse()


# ============================================================
# BUILD BOUNDARY OBJECTS
# ============================================================

results = []

for (
    word_index,
    start_frame,
    end_frame,
) in boundaries:

    start_time = frame_to_seconds(
        start_frame,
        frame_duration,
    )

    end_time = frame_to_seconds(
        end_frame,
        frame_duration,
    )

    duration = end_time - start_time

    score = ctc_segment_score(
        whole_log_probs,
        word_token_ids[word_index],
        start_frame,
        end_frame,
        blank_id,
    )

    results.append(
        WordBoundary(
            word_index=word_index,
            word=REFERENCE_WORDS[word_index],
            arabic=WORD_LABELS_ARABIC[word_index],
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            segmentation_score=score,
        )
    )


# ============================================================
# PRINT BOUNDARIES
# ============================================================

print()
print("=" * 90)
print("DETECTED WORD BOUNDARIES")
print("=" * 90)

for result in results:

    print()
    print(f"{result.word_index + 1}. " f"{result.arabic}")

    print(f"   " f"{result.start_time:.2f}s " f"-> " f"{result.end_time:.2f}s")

    print(f"   Duration: " f"{result.duration:.2f}s")


# ============================================================
# INDEPENDENT WORD RECOGNITION
# ============================================================

print()
print("=" * 90)
print("INDEPENDENT WORD PHONETIC RECOGNITION")
print("=" * 90)

print()
print(
    "Each detected word segment is now analyzed " "independently by the phonetic model."
)

word_results = []


for result in results:

    # --------------------------------------------------------
    # Extract audio samples
    # --------------------------------------------------------

    start_sample = round(result.start_time * SAMPLE_RATE)

    end_sample = round(result.end_time * SAMPLE_RATE)

    word_audio = audio[start_sample:end_sample]

    # --------------------------------------------------------
    # Run the model on THIS WORD ONLY
    # --------------------------------------------------------

    word_inputs = processor(
        word_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    with torch.no_grad():

        word_outputs = model(
            input_values=(word_inputs.input_values),
        )

    word_logits = (word_outputs.logits)[0]

    # --------------------------------------------------------
    # Unconstrained recognition
    # --------------------------------------------------------

    detected_phonetics, confidence = decode_ctc(
        word_logits,
        processor,
        blank_id,
    )

    reference = result.word

    similarity = similarity_score(
        reference,
        detected_phonetics,
    )

    word_results.append(
        (
            result,
            detected_phonetics,
            confidence,
            similarity,
        )
    )


# ============================================================
# PRINT WORD ANALYSIS
# ============================================================

for (
    result,
    detected,
    confidence,
    similarity,
) in word_results:

    print()
    print("-" * 90)

    print(f"WORD {result.word_index + 1}: " f"{result.arabic}")

    print(f"Time: " f"{result.start_time:.2f}s " f"-> " f"{result.end_time:.2f}s")

    print(f"Duration: " f"{result.duration:.2f}s")

    print()

    print(f"Expected phonetics:")

    print(f"  {result.word}")

    print()

    print(f"Detected phonetics:")

    print(f"  {detected}")

    print()

    print(f"Model frame confidence: " f"{confidence:.2f}%")

    print(f"Phonetic similarity: " f"{similarity:.2f}%")


# ============================================================
# OVERALL SUMMARY
# ============================================================

print()
print("=" * 90)
print("VERSION 7 SUMMARY")
print("=" * 90)

print()

print(f"Ayah duration: " f"{audio_duration:.2f}s")

print()

for (
    result,
    detected,
    confidence,
    similarity,
) in word_results:

    print(
        f"{result.arabic:<15} "
        f"Expected: {result.word:<15} "
        f"Detected: {detected:<20} "
        f"Similarity: {similarity:6.2f}%"
    )

print()

print("IMPORTANT:")

print("The similarity and confidence values are " "phonetic recognition diagnostics.")

print("They are NOT Tajweed correctness scores.")

print()
print("Done.")
