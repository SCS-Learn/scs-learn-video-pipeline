import subprocess
from PIL import Image, ImageDraw, ImageFont
import textwrap
import json
from src.audio.transcription import get_instructor_label
from src.audio.audio import merge_speaker_spans
import os

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "assets",
    "student-question-card-template.png",
)
FONT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "fonts", "OpenSans-Regular.ttf"
)

BLACK = (0, 0, 0)


def render_card(
    question_text,
    template_path=TEMPLATE_PATH,
    font_path=FONT_PATH,
    out_path="card.png",
):
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    WIDTH = 1920
    HEIGHT = 1080
    MAX_FONT_SIZE = 64
    MIN_FONT_SIZE = 32
    WRAP_WIDTH = 42

    font_size = MAX_FONT_SIZE
    while font_size >= MIN_FONT_SIZE:
        font = ImageFont.truetype(font_path, font_size)
        wrapped = textwrap.fill(question_text, width=WRAP_WIDTH)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=18)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= WIDTH - 200 and text_h <= HEIGHT - 400:
            break
        font_size -= 4

    x = (WIDTH - text_w) / 2
    y = 300 + (HEIGHT - 400 - text_h) / 2

    draw.multiline_text(
        (x, y), wrapped, font=font, fill=BLACK, spacing=18, align="center"
    )

    img.save(out_path)
    return out_path


def get_span_text(segments, start, end):
    texts = []
    for seg in segments:
        if seg["start"] < end and seg["end"] > start:
            texts.append(seg["text"].strip())
    return " ".join(texts)


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
        text = get_span_text(segments, iv["start"], iv["end"])
        card_path = f"data/15210-lecture12/cards/card_{i}.png"
        render_card(text, out_path=card_path)
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
        "h264_nvenc",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-shortest",
        out_path,
    ]

    subprocess.run(cmd, check=True)
    return out_path


def main():
    with open("data/transcription/transcript_classified.json") as f:
        segments = json.load(f)

    instructor_label = get_instructor_label(segments)
    intervals = merge_speaker_spans(segments, instructor_label)
    question_intervals = [iv for iv in intervals if iv["is_student_question"]]

    for i, iv in enumerate(question_intervals):
        print(f"{i}: {iv['start']:.2f} - {iv['end']:.2f}")

    overlay_question_cards(
        screen_path="data/15210-lecture12/screen.mp4",
        segments=segments,
        question_intervals=question_intervals,
        instructor_label=instructor_label,
        out_path="data/15210-lecture12/screen_with_cards.mp4",
    )


if __name__ == "__main__":
    main()
