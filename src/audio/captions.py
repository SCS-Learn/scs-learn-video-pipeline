import json
import pysrt


def seconds_to_subriptime(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return pysrt.SubRipTime(
        hours=hours, minutes=minutes, seconds=secs, milliseconds=millis
    )


def generate_captions(segments, output_path="captions.srt"):
    subs = pysrt.SubRipFile()

    index = 1
    for seg in segments:
        text = " ".join(word["word"] for word in seg.get("words", []))
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


if __name__ == "__main__":
    with open("data/transcription/transcript.json") as f:
        segments = json.load(f)

    generate_captions(segments, output_path="data/transcription/captions.srt")
