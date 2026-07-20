import subprocess
import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline
import os
from dotenv import load_dotenv
import json
import anthropic


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

def identify_student_questions(segments, instructor_label):
    load_dotenv()

    client = anthropic.Anthropic(api_key = os.getenv("ANTHROPIC_API_KEY"))

    non_instructor_segments = [
        seg for seg in segments if seg.get("speaker") != instructor_label
    ]
    
    if not non_instructor_segments:
        for seg in segments:
            seg["is_student_question"] = False
        return segments

    indexed_input = [
        {"index": i, "text": seg["text"].strip()}
        for i, seg in enumerate(non_instructor_segments)
    ]

    prompt = f"""You are analyzing a lecture transcript. Below is a list of
    segments spoken by someone other than the instructor (labeled "{instructor_label}").

    For each segment, decide whether it is a STUDENT QUESTION — an actual
    question being asked to the instructor — as opposed to a comment, aside,
    background noise, or non-question remark.

    Segments:
    {json.dumps(indexed_input, indent=2)}

    Respond with ONLY a JSON array, no other text, no markdown fences, in this
    exact format:
    [
    {{"index": 0, "is_student_question": true}},
    {{"index": 1, "is_student_question": false}}
    ]

    Every index from the input must appear exactly once in your response."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    try:
        classifications = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Claude did not return valid JSON:\n{raw_text}")

    result_by_index = {c["index"]: c["is_student_question"] for c in classifications}

    if len(result_by_index) != len(non_instructor_segments):
        raise ValueError(
            f"Expected {len(non_instructor_segments)} classifications, "
            f"got {len(result_by_index)} — check for missing/duplicate indices."
        )
    
    for i, seg in enumerate(non_instructor_segments):
        seg["is_student_question"] = result_by_index[i]
    
    for seg in segments:
        if seg.get("speaker") == instructor_label:
            seg["is_student_question"] = False
    
    return segments

def main():
    with open("data/transcription/transcript.json") as f:
        segments = json.load(f)
    
    instructor_label = get_instructor_label()

    segments = identify_student_questions(segments, instructor_label)
    with open("data/transcription/transcript_classified.json", "w") as f:
        json.dump(segments, f, indent=2)

if __name__ == "__main__":
    main()