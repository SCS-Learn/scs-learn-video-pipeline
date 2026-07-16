import subprocess
import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline
import os
from dotenv import load_dotenv
import json


def convert_mp4_to_wav():
    command = [
        "ffmpeg",
        "-y",
        "-i",
        "data/15210-lecture12/camera.mp4",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "data/15210-lecture12/camera.wav",
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")


def generate_transcript():
    load_dotenv()

    device = "cuda"
    audio_file = "data/15210-lecture12/camera.wav"
    batch_size = 16
    compute_type = "float16"

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    gc.collect()
    torch.cuda.empty_cache()
    del model

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    diarize_model = DiarizationPipeline(token=os.getenv("HF_TOKEN"), device=device)
    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    with open("data/transcription/transcript.json", "w") as f:
        json.dump(result["segments"], f, indent=2)


def get_instructor_label():
    with open("data/transcription/transcript.json") as f:
        segments = json.load(f)

    total_speak_time = {}

    for seg in segments:
        speaker = seg.get("speaker")
        if speaker is None:
            continue
        duration = seg["end"] - seg["start"]
        total_speak_time[speaker] = total_speak_time.get(speaker, 0) + duration

    instructor_speaker = max(total_speak_time, key=total_speak_time.get)
    return instructor_speaker
