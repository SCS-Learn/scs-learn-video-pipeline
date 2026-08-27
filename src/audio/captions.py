"""Write an .srt from the classified transcript.

    python -m src.audio.captions --lecture-dir data/15210-lecture12

Captions are part of the PRIVACY SURFACE, not a separate deliverable. The audio
pass silences every non-instructor word; a caption track built from the same
transcript without a speaker filter then publishes, in text, exactly what was
just removed from the audio. That is not a theoretical hole -- 15-210 lecture
12 shipped with five muted student utterances legible in captions.srt,
including "There's a lot of K's and K''s in slide" and "We could do some kind
of rebalancing".

So student speech is REDACTED here, by speaker label, using the same
instructor label audio.py mutes against. A redacted cue keeps its timing and
carries a neutral marker rather than vanishing, because a caption track with
silent gaps reads as broken while a marked one reads as deliberate -- and it
matches what the picture is doing at that moment, where cards.py has replaced
the student's question with a card.
"""

import json
import pysrt
import anthropic
import os
from dotenv import load_dotenv

from src.audio.transcription import get_instructor_label
from src.paths import LecturePaths, lecture_parser

# What a redacted cue says. Deliberately not the question text: cards.py already
# renders that ON SCREEN where it has been reviewed, and a caption is not
# reviewed.
REDACTION = "[student audio removed]"


def polish_captions(segments):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    indexed_input = [
        {"index": i, "text": seg["text"].strip()} for i, seg in enumerate(segments)
    ]

    prompt = f"""You are correcting automatic speech recognition (ASR) errors in a
    lecture transcript.

    Below is a list of transcript segments. Some may contain ASR mishears —
    words that sound similar to the correct term but are wrong given the
    course's subject matter (e.g. "binary stitch tree" should be "binary search tree"
    in an algorithms course).

    For each segment, return the corrected text. If a segment has no errors,
    return it unchanged. Do not paraphrase or alter meaning — only fix clear
    ASR mishears of course-specific terms.

    Segments:
    {json.dumps(indexed_input, indent=2)}

    Respond with ONLY a JSON array, no other text, no markdown fences:
    [
    {{"index": 0, "corrected_text": "..."}},
    {{"index": 1, "corrected_text": "..."}}
    ]

    Every index must appear exactly once."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=40000,
        messages=[{"role": "user", "content": prompt}],
    )

    first_block = response.content[0]
    raw_text = first_block.text.strip()

    try:
        corrections = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Claude did not return valid JSON:\n{raw_text}")

    result_by_index = {c["index"]: c["corrected_text"] for c in corrections}

    if len(result_by_index) != len(segments):
        raise ValueError(
            f"Expected {len(segments)} corrections, got {len(result_by_index)} "
            f"— check for missing/duplicate indices."
        )

    for i, seg in enumerate(segments):
        seg["text"] = result_by_index[i]

    return segments


def seconds_to_subriptime(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return pysrt.SubRipTime(
        hours=hours, minutes=minutes, seconds=secs, milliseconds=millis
    )


def is_instructor_only(seg, instructor_label):
    """True when every attributed word in this segment is the instructor's.

    Word level, not segment level, and for the same reason audio.py mutes at
    word level: diarization hands back segments whose speaker field is one
    person while individual words inside belong to another. A segment-level test
    passes those through whole. Unattributed words (speaker is None) do not
    count against it -- they are unknown, not known-student -- but a segment
    whose OWN speaker is not the instructor is redacted regardless.
    """
    if seg.get("speaker") not in (instructor_label, None):
        return False
    for word in seg.get("words", []):
        spk = word.get("speaker")
        if spk is not None and spk != instructor_label:
            return False
    return True


def generate_captions(segments, output_path="captions.srt",
                      instructor_label=None, redact=True):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    subs = pysrt.SubRipFile()

    index = 0
    n_redacted = 0
    for seg in segments:
        text = seg["text"]
        if not text.strip():
            continue
        index += 1

        if redact and instructor_label is not None \
                and not is_instructor_only(seg, instructor_label):
            n_redacted += 1
            subs.append(pysrt.SubRipItem(
                index=index,
                start=seconds_to_subriptime(seg["start"]),
                end=seconds_to_subriptime(seg["end"]),
                text=REDACTION,
            ))
            continue

        subs.append(pysrt.SubRipItem(
            index=index,
            start=seconds_to_subriptime(seg["start"]),
            end=seconds_to_subriptime(seg["end"]),
            text=text,
        ))

    subs.save(output_path, encoding="utf-8")
    if redact:
        print(f"[captions] redacted {n_redacted} non-instructor cue(s)")
    return subs


def main():
    parser = lecture_parser("Write an .srt from the classified transcript.")
    parser.add_argument("--include-student-speech", action="store_true",
                        help="Write student speech into the .srt verbatim. "
                             "OFF by default: the audio pass mutes those words, "
                             "and captioning them republishes exactly what was "
                             "removed. For internal review only -- never for a "
                             "published track.")
    parser.add_argument("--polish", action="store_true",
                        help="Also run the Claude ASR-mishear correction pass "
                             "(off by default: it rewrites caption text, so "
                             "review the diff before publishing)")
    args = parser.parse_args()
    p = LecturePaths(args.lecture_dir)

    with open(p.resolve_transcript_classified()) as f:
        segments = json.load(f)

    if args.polish:
        print("[captions] polishing ASR mishears via Claude ...")
        segments = polish_captions(segments)

    instructor_label = get_instructor_label(segments)
    if args.include_student_speech:
        print("[captions] WARNING: --include-student-speech -- this .srt will "
              "carry the words the audio pass muted. Do not publish it.")
    else:
        print(f"[captions] instructor={instructor_label}; every other speaker "
              f"is redacted to {REDACTION!r}")

    subs = generate_captions(segments, output_path=p.captions,
                             instructor_label=instructor_label,
                             redact=not args.include_student_speech)
    print(f"[captions] wrote {p.captions} ({len(subs)} cues)")


if __name__ == "__main__":
    main()
