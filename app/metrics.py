import re
from typing import List, Dict, Tuple, Union

# Each segment comes from Whisper output
# Expected keys: "start" (float), "end" (float), "text" (str)
Segment = Dict[str, Union[float, str]]


def _assign_speakers(
    segments: List[Segment],
    gap_threshold: float = 0.8
) -> List[Tuple[str, float, float, str]]:
    """
    Assign speakers A/B using a simple silence-gap heuristic.

    If the silence between two segments exceeds `gap_threshold`,
    we assume the speaker switched.

    This is lightweight and intentionally avoids heavy diarization models.
    """
    turns = []
    speaker = "A"
    last_end = None

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg["text"]).strip()

        # Switch speaker if a long pause is detected
        if last_end is not None and (start - last_end) > gap_threshold:
            speaker = "B" if speaker == "A" else "A"

        turns.append((speaker, start, end, text))
        last_end = end

    return turns


def _talk_time_ratio(
    turns: List[Tuple[str, float, float, str]]
) -> Dict[str, float]:
    """
    Compute weighted talk-time ratio per speaker.

    Weighting:
    - 60% duration-based
    - 40% word-count-based

    This avoids bias from either long silence or short rapid speech.
    """
    duration = {"A": 0.0, "B": 0.0}
    words = {"A": 0, "B": 0}

    for speaker, start, end, text in turns:
        duration[speaker] += (end - start)
        words[speaker] += len(text.split())

    total_time = duration["A"] + duration["B"] or 1e-6
    total_words = words["A"] + words["B"] or 1e-6

    return {
        "A": 0.6 * (duration["A"] / total_time) +
             0.4 * (words["A"] / total_words),
        "B": 0.6 * (duration["B"] / total_time) +
             0.4 * (words["B"] / total_words),
    }


def _count_questions(
    turns: List[Tuple[str, float, float, str]]
) -> int:
    """
    Count questions using both '?' and interrogative keywords.

    Regex is intentionally simple and fast for real-time APIs.
    """
    question_pattern = re.compile(
        r"(\?|"
        r"\b(what|why|how|when|where|who|which|can|could|would|"
        r"should|do|did|does|is|are|am)\b)",
        re.IGNORECASE,
    )

    return sum(
        1 for _, _, _, text in turns
        if question_pattern.search(text)
    )


def _longest_monologue(
    turns: List[Tuple[str, float, float, str]]
) -> float:
    """
    Find the longest uninterrupted speaking duration by a single speaker.
    """
    longest = 0.0
    current_speaker = None
    start_time = end_time = 0.0

    for speaker, start, end, _ in turns:
        if speaker != current_speaker:
            # Close previous monologue
            if current_speaker is not None:
                longest = max(longest, end_time - start_time)

            current_speaker = speaker
            start_time = start
            end_time = end
        else:
            end_time = end

    # Final check
    if current_speaker is not None:
        longest = max(longest, end_time - start_time)

    return longest


def _actionable_insight(
    talk_ratio: Dict[str, float],
    num_questions: int,
    sentiment: str
) -> str:
    """
    Generate human-readable coaching insight from metrics.
    """
    if talk_ratio["A"] > 0.7 or talk_ratio["B"] > 0.7:
        dominant = "A" if talk_ratio["A"] > talk_ratio["B"] else "B"
        return (
            f"Speaker {dominant} is dominating "
            f"(~{talk_ratio[dominant] * 100:.1f}%). "
            "Encourage the other speaker to participate more."
        )

    if num_questions < 3:
        return (
            "Few questions were asked. Recommend using more open-ended "
            "questions to improve engagement."
        )

    if sentiment.upper() == "NEGATIVE":
        return (
            "Overall negative sentiment detected. Address objections "
            "proactively and clarify next steps."
        )

    return (
        "Balanced interaction with healthy engagement. "
        "Consider summarizing agreed points and next steps."
    )


def analyze_metrics(
    segments: List[Segment],
    sentiment: str = "NEUTRAL"
) -> Dict[str, Union[float, int, Dict[str, float], str]]:
    """
    Public API function.

    Takes Whisper segments + overall sentiment
    and returns interpretable call-quality metrics.
    """
    turns = _assign_speakers(segments)

    talk_ratio = _talk_time_ratio(turns)
    num_questions = _count_questions(turns)
    longest_mono = _longest_monologue(turns)

    insight = _actionable_insight(
        talk_ratio,
        num_questions,
        sentiment
    )

    return {
        "talk_time_ratio": talk_ratio,
        "num_questions": num_questions,
        "longest_monologue_s": round(longest_mono, 2),
        "insight": insight,
    }
