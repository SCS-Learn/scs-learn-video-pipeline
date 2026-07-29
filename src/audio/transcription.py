import subprocess
import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline
import os
from dotenv import load_dotenv
import json
import anthropic

from src.paths import LecturePaths, lecture_parser


def convert_mp4_to_wav(video_path, wav_path):
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        wav_path,
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")


def generate_transcript(audio_file, out_json, device=None, batch_size=16):
    load_dotenv()

    # CPU has no float16 path in ctranslate2; fall back so a local smoke test
    # does not need a GPU.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcription] device={device} compute_type={compute_type}")

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    gc.collect()
    if device == "cuda":
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
    if device == "cuda":
        torch.cuda.empty_cache()
    del model_a

    # whisperx defaults to the gated pyannote/speaker-diarization-community-1;
    # pin to speaker-diarization-3.1 (accept its gate + the PLDA files it pulls
    # from community-1 on the HF model pages) to avoid a 403.
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        token=os.getenv("HF_TOKEN"),
        device=device,
    )
    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result["segments"], f, indent=2)
    return out_json


def get_instructor_label(segments):
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
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

    For each segment:
    1. Decide whether it is a STUDENT QUESTION — an actual question being asked
    to the instructor — as opposed to a comment, aside, background noise, or
    non-question remark.
    2. If it IS a question, also provide a cleaned-up version of the text: remove
    filler words ("um", "like"), remove any names mentioned, and phrase it as
    a clear, concise question suitable for display on a card. If it is NOT a
    question, return the original text unchanged.

    Segments:
    {json.dumps(indexed_input, indent=2)}

    Respond with ONLY a JSON array, no other text, no markdown fences:
    [
    {{"index": 0, "is_student_question": true, "text": "A clean version of the question..."}},
    {{"index": 1, "is_student_question": false, "text": "original text unchanged"}}
    ]

    Every index must appear exactly once."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    first_block = response.content[0]
    raw_text = first_block.text.strip()

    try:
        classifications = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Claude did not return valid JSON:\n{raw_text}")

    result_by_index = {c["index"]: c for c in classifications}
    if len(result_by_index) != len(non_instructor_segments):
        raise ValueError(
            f"Expected {len(non_instructor_segments)} classifications, "
            f"got {len(result_by_index)}."
        )

    for i, seg in enumerate(non_instructor_segments):
        seg["is_student_question"] = result_by_index[i]["is_student_question"]
        seg["text"] = result_by_index[i]["text"]

    for seg in segments:
        if seg.get("speaker") == instructor_label:
            seg["is_student_question"] = False

    return segments


def main():
    parser = lecture_parser("Transcribe + diarize + classify student questions.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    p = LecturePaths(args.lecture_dir)

    convert_mp4_to_wav(p.camera, p.camera_wav)
    generate_transcript(p.camera_wav, p.transcript,
                        device=args.device, batch_size=args.batch_size)

    with open(p.transcript) as f:
        segments = json.load(f)

    instructor_label = get_instructor_label(segments)
    print(f"[transcription] instructor speaker label: {instructor_label}")

    classified_segments = identify_student_questions(segments, instructor_label)
    with open(p.transcript_classified, "w") as f:
        json.dump(classified_segments, f, indent=2)
    n_q = sum(1 for s in classified_segments if s.get("is_student_question"))
    print(f"[transcription] wrote {p.transcript_classified} "
          f"({len(classified_segments)} segments, {n_q} student questions)")


if __name__ == "__main__":
    main()
