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
    2. If it IS a question, rewrite it as a SHORT question suitable for a slide.
    Remove filler words ("um", "like"), remove any names, and state only the
    core thing being asked.

    HARD LIMIT: at most 160 characters, ideally one sentence under 100.
    Students ramble; do not transcribe them verbatim. If a question runs on for
    several clauses, identify what is actually being asked and write just that.
    A 300-character card renders as an unreadable wall of text.

    If it is NOT a question, return the original text unchanged.

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


def identify_repeated_questions(segments, instructor_label, window=10.0,
                                max_student_words=12, batch=40):
    """Recover questions the instructor repeats back to the room.

    identify_student_questions only ever inspects non-instructor segments, and
    forces every instructor segment to False. That is structurally unable to
    catch the commonest shape in a real lecture: a student asks from the floor,
    the room mic barely picks them up, and the instructor repeats the question
    so everyone can hear it. Diarization then puts the *question text* on the
    instructor and leaves the student with a fragment.

    Measured on lecture 12 at 598.3s the student's whole segment transcribes as
    "Yeah?", while "If you want an old version, where are they stored?" is
    attributed to the instructor 2 seconds later. Only 4 of 1,237 segments were
    flagged across 91 minutes, and this is a large part of why.

    The card still goes on the STUDENT's span -- that is when the audio is
    muted, so the card fills the silence -- but its text comes from the
    instructor's restatement, which is cleaner audio and better phrased than
    anything the room mic captured.
    """
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    cands = []
    for i, seg in enumerate(segments):
        if seg.get("speaker") == instructor_label or seg.get("is_student_question"):
            continue
        # A student whose question was captured cleanly does not need this pass;
        # the giveaway is a very short fragment ("Yeah?", "Sorry?", inaudible).
        if len(seg["text"].split()) > max_student_words:
            continue
        after = []
        for nxt in segments[i + 1:]:
            if nxt["start"] - seg["end"] > window:
                break
            if nxt.get("speaker") == instructor_label:
                after.append(nxt["text"].strip())
            if sum(len(a) for a in after) > 600:
                break
        if after:
            cands.append({"index": i, "student": seg["text"].strip(),
                          "instructor_follows": " ".join(after)[:600]})

    if not cands:
        print("[transcription] repeat-back pass: no candidates")
        return segments

    found = 0
    for k in range(0, len(cands), batch):
        chunk = cands[k:k + batch]
        prompt = f"""In a lecture, students often ask questions the room mic barely
    captures, and the instructor repeats the question back before answering.

    Each item below is a short non-instructor segment plus what the instructor
    said immediately after. Decide whether the instructor is REPEATING BACK a
    student question. Restating a question sounds like "so the question is...",
    "you're asking whether...", or simply asking the question aloud before
    answering it. Continuing to lecture, or asking the room a rhetorical or
    checking question ("make sense?", "any questions?"), is NOT a repeat-back.

    Be conservative: only say true when the instructor is clearly voicing a
    question that came from a student.

    If true, write that question as a SHORT question for a slide: at most 160
    characters, no names, no filler.

    Items:
    {json.dumps(chunk, indent=2)}

    Respond with ONLY a JSON array, no markdown:
    [{{"index": <the item's index>, "is_repeat_back": true, "text": "..."}}]
    Every index must appear exactly once."""

        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=4000,
                                      messages=[{"role": "user", "content": prompt}])
        try:
            out = json.loads(resp.content[0].text.strip())
        except json.JSONDecodeError:
            print("[transcription] repeat-back pass: unparseable reply, skipping batch")
            continue
        for r in out:
            if not r.get("is_repeat_back"):
                continue
            seg = segments[r["index"]]
            seg["is_student_question"] = True
            seg["text"] = r["text"]
            seg["question_source"] = "instructor_repeat_back"
            found += 1

    print(f"[transcription] repeat-back pass: {len(cands)} candidates, "
          f"{found} additional student questions recovered")
    return segments


def main():
    parser = lecture_parser("Transcribe + diarize + classify student questions.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reclassify", action="store_true",
                        help="Re-run only the question classification over the "
                             "existing transcript. No ASR, no diarization, so "
                             "no GPU -- use this after changing the prompts "
                             "rather than paying for transcription again.")
    parser.add_argument("--no-repeat-back", action="store_true",
                        help="Skip recovering questions the instructor repeats")
    args = parser.parse_args()
    p = LecturePaths(args.lecture_dir)

    if args.reclassify:
        src = (p.resolve_transcript_classified()
               if os.path.exists(p.transcript_classified)
               else p.resolve_transcript())
        print(f"[transcription] reclassifying {src} (no ASR, no GPU)")
        with open(src) as f:
            segments = json.load(f)
    else:
        # resolve_camera(), not p.camera: sync may have cut pre-lecture black off
        # the front, and every timestamp produced here is relative to that cut.
        convert_mp4_to_wav(p.resolve_camera(), p.camera_wav)
        generate_transcript(p.camera_wav, p.transcript,
                            device=args.device, batch_size=args.batch_size)
        with open(p.transcript) as f:
            segments = json.load(f)

    instructor_label = get_instructor_label(segments)
    print(f"[transcription] instructor speaker label: {instructor_label}")

    if args.reclassify:
        # Start from a clean slate so a rerun cannot inherit a stale verdict.
        for s in segments:
            s.pop("is_student_question", None)
            s.pop("question_source", None)

    classified_segments = identify_student_questions(segments, instructor_label)
    if not args.no_repeat_back:
        classified_segments = identify_repeated_questions(
            classified_segments, instructor_label)
    with open(p.transcript_classified, "w") as f:
        json.dump(classified_segments, f, indent=2)
    n_q = sum(1 for s in classified_segments if s.get("is_student_question"))
    print(f"[transcription] wrote {p.transcript_classified} "
          f"({len(classified_segments)} segments, {n_q} student questions)")


if __name__ == "__main__":
    main()
