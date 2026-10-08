from pathlib import Path
from dataclasses import dataclass

import librosa
import torch
from transformers import AutoProcessor, AutoModelForCTC

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "TBOGamer22/wav2vec2-quran-phonetics"

BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE = BASE_DIR / "audio" / "test.wav"

SAMPLE_RATE = 16000

SURAH_ID = 1
AYAH_ID = 2

# Quran words
REFERENCE_WORDS = [
    "al-ḥamdu",
    "lillahi",
    "rabbi",
    "l-ʿālamīna",
]

# Arabic words corresponding to REFERENCE_WORDS
WORD_LABELS_ARABIC = [
    "الْحَمْدُ",
    "لِلَّهِ",
    "رَبِّ",
    "الْعَالَمِينَ",
]

REFERENCE = "|".join(REFERENCE_WORDS)

# Forced alignment is performed without word separators.
ALIGNMENT_REFERENCE = "".join(REFERENCE_WORDS)

WEAK_THRESHOLD = 0.50
STRONG_THRESHOLD = 0.75


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class TokenSegment:
    token: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def frame_to_time(
    frame: int,
    frame_duration: float,
) -> float:
    return frame * frame_duration


def similarity_score(
    reference: str,
    predicted: str,
) -> tuple[int, float]:
    """
    Levenshtein edit distance.

    Returns:
        distance
        similarity percentage
    """

    n = len(reference)
    m = len(predicted)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if reference[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    distance = dp[n][m]

    denominator = max(
        n,
        m,
        1,
    )

    similarity = 1 - distance / denominator

    return distance, similarity


def edit_operations(
    reference: str,
    predicted: str,
):
    """
    Returns:

        MATCH
        SUBSTITUTE
        DELETE
        INSERT
    """

    n = len(reference)
    m = len(predicted)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if reference[i - 1] == predicted[j - 1]:

                dp[i][j] = dp[i - 1][j - 1]

            else:

                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                )

    operations = []

    i = n
    j = m

    while i > 0 or j > 0:

        # ----------------------------------------------------
        # MATCH
        # ----------------------------------------------------

        if (
            i > 0
            and j > 0
            and reference[i - 1] == predicted[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):

            operations.append(
                {
                    "operation": "MATCH",
                    "reference_index": i - 1,
                    "predicted_index": j - 1,
                    "reference_token": reference[i - 1],
                    "predicted_token": predicted[j - 1],
                }
            )

            i -= 1
            j -= 1

            continue

        # ----------------------------------------------------
        # SUBSTITUTE
        # ----------------------------------------------------

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:

            operations.append(
                {
                    "operation": "SUBSTITUTE",
                    "reference_index": i - 1,
                    "predicted_index": j - 1,
                    "reference_token": reference[i - 1],
                    "predicted_token": predicted[j - 1],
                }
            )

            i -= 1
            j -= 1

            continue

        # ----------------------------------------------------
        # DELETE
        # ----------------------------------------------------

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:

            operations.append(
                {
                    "operation": "DELETE",
                    "reference_index": i - 1,
                    "predicted_index": None,
                    "reference_token": reference[i - 1],
                    "predicted_token": None,
                }
            )

            i -= 1

            continue

        # ----------------------------------------------------
        # INSERT
        # ----------------------------------------------------

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:

            operations.append(
                {
                    "operation": "INSERT",
                    "reference_index": None,
                    "predicted_index": j - 1,
                    "reference_token": None,
                    "predicted_token": predicted[j - 1],
                }
            )

            j -= 1

            continue

        raise RuntimeError("Could not reconstruct edit operations.")

    operations.reverse()

    return operations


def greedy_ctc_decode(
    logits: torch.Tensor,
    blank_id: int,
    id_to_token: dict[int, str],
    frame_duration: float,
):
    """
    Standard greedy CTC decoding.

    Returns:
        predicted string
        token segments
    """

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    best_ids = torch.argmax(
        probabilities,
        dim=-1,
    )

    segments = []

    previous_raw_id = None

    active_id = None
    active_start = None
    active_frames = []

    for frame, token_id_tensor in enumerate(best_ids):

        token_id = int(token_id_tensor.item())

        # ----------------------------------------------------
        # BLANK
        # ----------------------------------------------------

        if token_id == blank_id:

            if active_id is not None:

                end_frame = frame

                confidence = torch.tensor(active_frames).mean().item()

                token = id_to_token.get(
                    active_id,
                    f"<UNK:{active_id}>",
                )

                segments.append(
                    TokenSegment(
                        token=token,
                        start_frame=active_start,
                        end_frame=end_frame,
                        start_time=frame_to_time(
                            active_start,
                            frame_duration,
                        ),
                        end_time=frame_to_time(
                            end_frame,
                            frame_duration,
                        ),
                        confidence=confidence,
                    )
                )

                active_id = None
                active_start = None
                active_frames = []

            previous_raw_id = token_id

            continue

        # ----------------------------------------------------
        # SAME TOKEN AS PREVIOUS RAW FRAME
        # ----------------------------------------------------

        if token_id == previous_raw_id:

            if active_id == token_id:

                active_frames.append(
                    probabilities[
                        frame,
                        token_id,
                    ].item()
                )

            previous_raw_id = token_id

            continue

        # ----------------------------------------------------
        # NEW TOKEN
        # ----------------------------------------------------

        if active_id is not None:

            end_frame = frame

            confidence = torch.tensor(active_frames).mean().item()

            token = id_to_token.get(
                active_id,
                f"<UNK:{active_id}>",
            )

            segments.append(
                TokenSegment(
                    token=token,
                    start_frame=active_start,
                    end_frame=end_frame,
                    start_time=frame_to_time(
                        active_start,
                        frame_duration,
                    ),
                    end_time=frame_to_time(
                        end_frame,
                        frame_duration,
                    ),
                    confidence=confidence,
                )
            )

        active_id = token_id
        active_start = frame

        active_frames = [
            probabilities[
                frame,
                token_id,
            ].item()
        ]

        previous_raw_id = token_id

    # --------------------------------------------------------
    # CLOSE FINAL TOKEN
    # --------------------------------------------------------

    if active_id is not None:

        end_frame = len(best_ids)

        confidence = torch.tensor(active_frames).mean().item()

        token = id_to_token.get(
            active_id,
            f"<UNK:{active_id}>",
        )

        segments.append(
            TokenSegment(
                token=token,
                start_frame=active_start,
                end_frame=end_frame,
                start_time=frame_to_time(
                    active_start,
                    frame_duration,
                ),
                end_time=frame_to_time(
                    end_frame,
                    frame_duration,
                ),
                confidence=confidence,
            )
        )

    predicted = "".join(segment.token for segment in segments)

    return predicted, segments


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Quran phoneme model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForCTC.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded.")

tokenizer = processor.tokenizer

vocab = tokenizer.get_vocab()

id_to_token = {token_id: token for token, token_id in vocab.items()}

blank_id = model.config.pad_token_id

print(f"\nCTC blank ID: {blank_id}")

print("\nReference:")
print(REFERENCE)


# ============================================================
# VALIDATE REFERENCE
# ============================================================

for character in ALIGNMENT_REFERENCE:

    if character not in vocab:

        raise ValueError(
            f"\nCharacter {character!r} "
            "is not in the model vocabulary."
            f"\nVocabulary: {vocab}"
        )


alignment_target_ids = [vocab[character] for character in ALIGNMENT_REFERENCE]

print("\nAlignment reference:")
print(ALIGNMENT_REFERENCE)

print("\nAlignment token IDs:")
print(alignment_target_ids)


# ============================================================
# LOAD AUDIO
# ============================================================

if not AUDIO_FILE.exists():

    raise FileNotFoundError(f"\nAudio file not found:\n" f"{AUDIO_FILE}")


print("\nLoading audio...")

audio, sample_rate = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True,
)

duration = len(audio) / SAMPLE_RATE

print(f"Audio:       {AUDIO_FILE.name}")

print(f"Sample rate: {sample_rate} Hz")

print(f"Duration:    {duration:.2f} seconds")


# ============================================================
# RUN MODEL
# ============================================================

print("\nRunning model...")

inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt",
    padding=True,
)

with torch.inference_mode():

    logits = model(
        input_values=inputs.input_values,
        attention_mask=inputs.attention_mask,
    ).logits[0]


print(f"Logits shape: {tuple(logits.shape)}")

num_frames = logits.shape[0]

print(f"Number of CTC frames: {num_frames}")


# ============================================================
# FRAME TIMING
# ============================================================

input_to_logits_ratio = getattr(
    model.config,
    "inputs_to_logits_ratio",
    320,
)

frame_duration = input_to_logits_ratio / SAMPLE_RATE

print(f"Frame duration: " f"{frame_duration * 1000:.2f} ms")


# ============================================================
# PART 1
# UNCONSTRAINED CTC RECOGNITION
# ============================================================

print("\n")
print("=" * 90)
print("1. UNCONSTRAINED CTC RECOGNITION")
print("=" * 90)

print("\nThis is the model's actual greedy prediction.")

predicted_text, predicted_segments = greedy_ctc_decode(
    logits=logits,
    blank_id=blank_id,
    id_to_token=id_to_token,
    frame_duration=frame_duration,
)

print("\nExpected:")
print(REFERENCE)

print("\nModel prediction:")
print(predicted_text)


# ============================================================
# RECOGNIZED TOKEN SEGMENTS
# ============================================================

print("\n")
print("Recognized phoneme segments:")
print("-" * 90)

for segment in predicted_segments:

    if segment.token == "|":
        token_type = "WORD"
    else:
        token_type = ""

    print(
        f"{segment.token:>4}   "
        f"{segment.start_time:6.2f}s -> "
        f"{segment.end_time:6.2f}s   "
        f"{segment.confidence * 100:6.2f}%   "
        f"{token_type}"
    )


# ============================================================
# REMOVE WORD SEPARATORS
# ============================================================

predicted_phonemes = predicted_text.replace("|", "")

reference_phonemes = ALIGNMENT_REFERENCE


# ============================================================
# PHONETIC SEQUENCE SIMILARITY
# ============================================================

distance, similarity = similarity_score(
    reference_phonemes,
    predicted_phonemes,
)

print("\n")
print("=" * 90)
print("PHONETIC SEQUENCE COMPARISON")
print("=" * 90)

print("\nReference phonemes:")
print(reference_phonemes)

print("\nModel phonemes:")
print(predicted_phonemes)

print(f"\nEdit distance: {distance}")

print(f"Phonetic sequence similarity: " f"{similarity * 100:.2f}%")

print("\nThis is a model-recognition similarity," "\nnot a Tajweed score.")


# ============================================================
# WORD SEPARATOR CHECK
# ============================================================

reference_separator_count = REFERENCE.count("|")
predicted_separator_count = predicted_text.count("|")

print("\n")
print("=" * 90)
print("WORD SEPARATOR CHECK")
print("=" * 90)

print(f"\nExpected word separators: " f"{reference_separator_count}")

print(f"Model word separators:    " f"{predicted_separator_count}")


# ============================================================
# EDIT OPERATIONS
# ============================================================

operations = edit_operations(
    reference_phonemes,
    predicted_phonemes,
)

problem_operations = [
    operation for operation in operations if operation["operation"] != "MATCH"
]


print("\n")
print("=" * 90)
print("2. RECOGNITION DIFFERENCES")
print("=" * 90)

if not problem_operations:

    print("\nNo phoneme differences found.")

else:

    for index, operation in enumerate(
        problem_operations,
        start=1,
    ):

        op = operation["operation"]

        expected = operation["reference_token"]

        detected = operation["predicted_token"]

        if op == "SUBSTITUTE":

            print(
                f"{index:3}. SUBSTITUTE   "
                f"expected={expected!r}   "
                f"detected={detected!r}"
            )

        elif op == "DELETE":

            print(
                f"{index:3}. DELETE       "
                f"expected={expected!r}   "
                f"detected=<missing>"
            )

        elif op == "INSERT":

            print(
                f"{index:3}. INSERT       "
                f"expected=<none>   "
                f"detected={detected!r}"
            )


# ============================================================
# PART 2
# FORCED ALIGNMENT
# ============================================================

print("\n")
print("=" * 90)
print("3. FORCED ALIGNMENT")
print("=" * 90)

print(
    "\nThe expected phoneme sequence is forced onto"
    "\nthe audio only to obtain timestamps."
)

log_probs = torch.log_softmax(
    logits,
    dim=-1,
)


# ============================================================
# BUILD CTC TARGET
# ============================================================

extended = [blank_id]

for token_id in alignment_target_ids:

    extended.append(token_id)
    extended.append(blank_id)

extended = torch.tensor(
    extended,
    dtype=torch.long,
)

num_states = len(extended)

NEG_INF = -float("inf")


# ============================================================
# VITERBI INITIALIZATION
# ============================================================

previous = torch.full(
    (num_states,),
    NEG_INF,
)

previous[0] = log_probs[
    0,
    blank_id,
]

if num_states > 1:

    previous[1] = log_probs[
        0,
        extended[1],
    ]


backpointer = torch.full(
    (
        num_frames,
        num_states,
    ),
    -1,
    dtype=torch.int16,
)

backpointer[0, 0] = 0

if num_states > 1:

    backpointer[0, 1] = 0


# ============================================================
# VITERBI
# ============================================================

for t in range(
    1,
    num_frames,
):

    current = torch.full(
        (num_states,),
        NEG_INF,
    )

    for s in range(num_states):

        token_id = extended[s]

        # ----------------------------------------------------
        # Stay
        # ----------------------------------------------------

        best_score = previous[s]
        best_previous_state = s

        # ----------------------------------------------------
        # Move one state
        # ----------------------------------------------------

        if s > 0:

            score = previous[s - 1]

            if score > best_score:

                best_score = score
                best_previous_state = s - 1

        # ----------------------------------------------------
        # Skip one state
        # ----------------------------------------------------

        if s > 1:

            if token_id != extended[s - 2]:

                score = previous[s - 2]

                if score > best_score:

                    best_score = score
                    best_previous_state = s - 2

        current[s] = (
            best_score
            + log_probs[
                t,
                token_id,
            ]
        )

        backpointer[
            t,
            s,
        ] = best_previous_state

    previous = current


# ============================================================
# BEST FINAL STATE
# ============================================================

final_candidates = [
    num_states - 1,
    num_states - 2,
]

best_final_state = max(
    final_candidates,
    key=lambda state: previous[state].item(),
)

best_score = previous[best_final_state].item()


# ============================================================
# BACKTRACK
# ============================================================

states = []

state = best_final_state

for t in range(
    num_frames - 1,
    -1,
    -1,
):

    states.append(state)

    if t > 0:

        state = int(
            backpointer[
                t,
                state,
            ]
        )

states.reverse()


# ============================================================
# EXTRACT ALIGNED SEGMENTS
# ============================================================

segments = []

current_state = None
start_frame = None

for frame, state in enumerate(states):

    if extended[state] == blank_id:

        if current_state is not None:

            segments.append(
                (
                    current_state,
                    start_frame,
                    frame,
                )
            )

            current_state = None
            start_frame = None

        continue

    if state != current_state:

        if current_state is not None:

            segments.append(
                (
                    current_state,
                    start_frame,
                    frame,
                )
            )

        current_state = state
        start_frame = frame


if current_state is not None:

    segments.append(
        (
            current_state,
            start_frame,
            num_frames,
        )
    )


# ============================================================
# CREATE ALIGNMENT OBJECTS
# ============================================================

forced_alignment = []

for state, start_frame, end_frame in segments:

    token_id = int(extended[state])

    token = id_to_token.get(
        token_id,
        f"<UNK:{token_id}>",
    )

    token_log_probs = log_probs[
        start_frame:end_frame,
        token_id,
    ]

    confidence = torch.exp(token_log_probs.mean()).item()

    forced_alignment.append(
        TokenSegment(
            token=token,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=frame_to_time(
                start_frame,
                frame_duration,
            ),
            end_time=frame_to_time(
                end_frame,
                frame_duration,
            ),
            confidence=confidence,
        )
    )


# ============================================================
# PRINT FORCED ALIGNMENT
# ============================================================

print("\n")
print("Forced phoneme timestamps:")
print("-" * 90)

for segment in forced_alignment:

    if segment.confidence >= STRONG_THRESHOLD:

        level = "STRONG"

    elif segment.confidence >= WEAK_THRESHOLD:

        level = "MODERATE"

    else:

        level = "WEAK"

    print(
        f"{segment.token:>4}   "
        f"{segment.start_time:6.2f}s -> "
        f"{segment.end_time:6.2f}s   "
        f"{segment.confidence * 100:6.2f}%   "
        f"{level}"
    )


# ============================================================
# PART 3
# WORD-LEVEL FORCED TIMING
# ============================================================

print("\n")
print("=" * 90)
print("4. WORD-LEVEL TIMING")
print("=" * 90)


word_results = []

alignment_position = 0

for word_index, word in enumerate(REFERENCE_WORDS):

    word_length = len(word)

    word_segments = forced_alignment[
        alignment_position : alignment_position + word_length
    ]

    alignment_position += word_length

    if not word_segments:
        continue

    word_start = word_segments[0].start_time

    word_end = word_segments[-1].end_time

    valid_confidences = [
        segment.confidence for segment in word_segments if segment.confidence > 0
    ]

    if valid_confidences:

        word_evidence = torch.exp(
            torch.tensor(
                [
                    torch.log(torch.tensor(confidence))
                    for confidence in valid_confidences
                ]
            ).mean()
        ).item()

    else:

        word_evidence = 0.0

    word_results.append(
        {
            "arabic_index": word_index,
            "word": word,
            "start": word_start,
            "end": word_end,
            "confidence": word_evidence,
            "segments": word_segments,
        }
    )


for index, word in enumerate(word_results):

    arabic_word = WORD_LABELS_ARABIC[index] if index < len(WORD_LABELS_ARABIC) else ""

    print("\n" + "-" * 90)

    print(f"Word {index + 1}: " f"{arabic_word}")

    print(f"Phonetics: " f"{word['word']}")

    print(f"Time: " f"{word['start']:.2f}s -> " f"{word['end']:.2f}s")

    print(f"Duration: " f"{word['end'] - word['start']:.2f}s")

    print(f"Forced-alignment evidence: " f"{word['confidence'] * 100:.2f}%")


# ============================================================
# PART 4
# MAP RECOGNITION ERRORS TO TIME
# ============================================================

print("\n")
print("=" * 90)
print("5. POSSIBLE DIFFERENCE LOCATIONS")
print("=" * 90)

print(
    "\nThese timestamps combine:"
    "\n- unconstrained recognition"
    "\n- reference forced alignment"
    "\n"
    "\nThey are candidate locations for further analysis,"
    "\nnot confirmed Tajweed errors."
)


for index, operation in enumerate(
    problem_operations,
    start=1,
):

    op = operation["operation"]

    reference_index = operation["reference_index"]

    predicted_index = operation["predicted_index"]

    expected = operation["reference_token"]

    detected = operation["predicted_token"]

    # --------------------------------------------------------
    # Reference timestamp
    # --------------------------------------------------------

    if reference_index is not None:

        if 0 <= reference_index < len(forced_alignment):

            reference_segment = forced_alignment[reference_index]

            time_text = (
                f"{reference_segment.start_time:.2f}s"
                f" -> "
                f"{reference_segment.end_time:.2f}s"
            )

        else:

            time_text = "unknown"

    else:

        time_text = "unknown"

    # --------------------------------------------------------
    # INSERT timestamp
    # --------------------------------------------------------

    if (
        op == "INSERT"
        and predicted_index is not None
        and 0 <= predicted_index < len(predicted_segments)
    ):

        predicted_segment = predicted_segments[predicted_index]

        time_text = (
            f"{predicted_segment.start_time:.2f}s"
            f" -> "
            f"{predicted_segment.end_time:.2f}s"
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if op == "SUBSTITUTE":

        print(f"\n{index}. " f"{time_text}")

        print(f"   Expected: {expected!r}")

        print(f"   Model heard: {detected!r}")

        print(f"   Type: POSSIBLE SUBSTITUTION")

    elif op == "DELETE":

        print(f"\n{index}. " f"{time_text}")

        print(f"   Expected: {expected!r}")

        print(f"   Model heard: nothing")

        print(f"   Type: POSSIBLE OMISSION")

    elif op == "INSERT":

        print(f"\n{index}. " f"{time_text}")

        print(f"   Expected: nothing")

        print(f"   Model heard: {detected!r}")

        print(f"   Type: POSSIBLE EXTRA PHONEME")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print("\nExpected phoneme sequence:")

print(f"  {reference_phonemes}")

print("\nModel's actual phoneme sequence:")

print(f"  {predicted_phonemes}")

print(f"\nRecognition edit distance:" f" {distance}")

print(f"\nRecognition phonetic similarity:" f" {similarity * 100:.2f}%")

print(f"\nNumber of recognition differences:" f" {len(problem_operations)}")

print(f"\nForced-alignment CTC score:" f" {best_score:.2f}")

print("\nIMPORTANT:")

print(
    "The forced alignment does NOT mean the model"
    "\nrecognized every expected phoneme."
)

print("\nThe recognition result above is the model's" "\nunconstrained prediction.")

print(
    "\nThe forced alignment is only used to put"
    "\nthe expected phonemes on the timeline."
)

print(
    "\nThis is still a phonetic-analysis prototype,"
    "\nnot yet a Tajweed grading system."
)

print("\nDone.")
