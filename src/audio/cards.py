import subprocess
from PIL import Image, ImageDraw, ImageFont
import textwrap
import json
from src.audio.transcription import get_instructor_label
from src.audio.audio import merge_speaker_spans


def render_card(text, width=1920, height=1080, out_path="card.png"):
    img = Image.new("RGB", (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=48)

    wrapped = textwrap.fill(text, width=40)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) / 2
    y = (height - text_h) / 2

    draw.multiline_text(
        (x, y), wrapped, font=font, fill=(255, 255, 255), align="center"
    )
    img.save(out_path)
    return out_path


def get_span_text(segments, start, end, instructor_label):
    words = []
    for seg in segments:
        for word in seg.get("words", []):
            if word.get("speaker") == instructor_label:
                continue
            if start <= word["start"] < end:
                words.append(word["word"])
    return " ".join(words)


def overlay_question_cards(
    screen_path,
    segments,
    question_intervals,
    instructor_label,
    out_path,
    width=1920,
    height=1080,
):
    if not question_intervals:
        subprocess.run(
            ["ffmpeg", "-y", "-i", screen_path, "-c", "copy", out_path], check=True
        )
        return out_path

    card_paths = []
    for i, iv in enumerate(question_intervals):
        text = get_span_text(segments, iv["start"], iv["end"], instructor_label)
        card_path = f"data/15210-lecture12/cards/card_{i}.png"
        render_card(text, width=width, height=height, out_path=card_path)
        card_paths.append(card_path)

    cmd = ["ffmpeg", "-y", "-i", screen_path]
    for card_path in card_paths:
        cmd += ["-loop", "1", "-i", card_path]

    filter_parts = []
    current_label = "0:v"
    for i, iv in enumerate(question_intervals):
        input_index = i + 1
        out_label = f"tmp{i}" if i < len(question_intervals) - 1 else "outv"
        filter_parts.append(
            f"[{current_label}][{input_index}:v]overlay="
            f"enable='between(t,{iv['start']},{iv['end']})'[{out_label}]"
        )
        current_label = out_label

    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        out_path,
    ]

    subprocess.run(cmd, check=True)
    return out_path


if __name__ == "__main__":
    with open("data/transcription/transcript_classified.json") as f:
        segments = json.load(f)

    instructor_label = get_instructor_label()
    intervals = merge_speaker_spans(segments, instructor_label)
    question_intervals = [iv for iv in intervals if iv["is_student_question"]]

    overlay_question_cards(
        screen_path="data/15210-lecture12/screen.mp4",
        segments=segments,
        question_intervals=question_intervals,
        instructor_label=instructor_label,
        out_path="data/15210-lecture12/screen_with_cards.mp4",
    )
