import subprocess
import numpy as np
import soundfile as sf
import json
from src.audio.transcription import get_instructor_label


def merge_speaker_spans(segments, instructor_label, gap_tolerance=0.3):
    intervals = []
    current_start = None
    current_end = None
    current_is_question = None

    for seg in segments:
        for word in seg.get("words", []):
            speaker = word.get("speaker")
            if speaker is None:
                continue

            if speaker != instructor_label:
                if current_start is None:
                    current_start = word["start"]
                    current_is_question = seg.get("is_student_question", False)
                elif word["start"] - current_end > gap_tolerance:
                    intervals.append(
                        {
                            "start": current_start,
                            "end": current_end,
                            "is_student_question": current_is_question,
                        }
                    )
                    current_start = word["start"]
                    current_is_question = seg.get("is_student_question", False)
                current_end = word["end"]
            else:
                if current_start is not None:
                    intervals.append(
                        {
                            "start": current_start,
                            "end": current_end,
                            "is_student_question": current_is_question,
                        }
                    )
                    current_start = None
                    current_is_question = None

    if current_start is not None:
        intervals.append(
            {
                "start": current_start,
                "end": current_end,
                "is_student_question": current_is_question,
            }
        )

    return intervals


def mute_student_audio(wav_path, intervals, out_wav_path, fade_samples=480):
    audio, sr = sf.read(wav_path)

    for iv in intervals:
        start_sample = int(iv["start"] * sr)
        end_sample = int(iv["end"] * sr)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if end_sample - start_sample <= 2 * fade_samples:
            audio[start_sample:end_sample] = 0
            continue

        fade_out = np.linspace(1, 0, fade_samples)
        fade_in = np.linspace(0, 1, fade_samples)

        if audio.ndim == 1:
            audio[start_sample : start_sample + fade_samples] *= fade_out
        else:
            audio[start_sample : start_sample + fade_samples] *= fade_out[:, None]

        audio[start_sample + fade_samples : end_sample - fade_samples] = 0

        if audio.ndim == 1:
            audio[end_sample - fade_samples : end_sample] *= fade_in
        else:
            audio[end_sample - fade_samples : end_sample] *= fade_in[:, None]

    sf.write(out_wav_path, audio, sr)
    return out_wav_path


def put_audio_into_video(video_path, wav_path, out_video_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            wav_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            out_video_path,
        ],
        check=True,
    )
    return out_video_path


if __name__ == "__main__":
    with open("data/transcription/transcript_classified.json") as f:
        segments = json.load(f)

    instructor_label = get_instructor_label()
    intervals = merge_speaker_spans(segments, instructor_label)

    mute_student_audio(
        wav_path="data/15210-lecture12/camera.wav",
        intervals=intervals,
        out_wav_path="data/15210-lecture12/camera_muted.wav",
    )

    put_audio_into_video(
        video_path="data/15210-lecture12/camera.mp4",
        wav_path="data/15210-lecture12/camera_muted.wav",
        out_video_path="data/15210-lecture12/camera_muted.mp4",
    )
