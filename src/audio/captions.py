import json
import pysrt
import anthropic
import os
from dotenv import load_dotenv

from src.paths import LecturePaths, lecture_parser


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


def generate_captions(segments, output_path="captions.srt"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    subs = pysrt.SubRipFile()

    index = 1
    for seg in segments:
        text = seg["text"]
        if not text.strip():
            continue

        item = pysrt.SubRipItem(
            index=index,
            start=seconds_to_subriptime(seg["start"]),
            end=seconds_to_subriptime(seg["end"]),
            text=text,
        )
        subs.append(item)
        index += 1

    subs.save(output_path, encoding="utf-8")
    return subs


def main():
    parser = lecture_parser("Write an .srt from the classified transcript.")
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

    subs = generate_captions(segments, output_path=p.captions)
    print(f"[captions] wrote {p.captions} ({len(subs)} cues)")


if __name__ == "__main__":
    main()
