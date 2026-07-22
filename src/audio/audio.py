import subprocess
import numpy as np
import soundfile as sf

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
                    intervals.append({
                        "start": current_start,
                        "end": current_end,
                        "is_student_question": current_is_question,
                    })
                    current_start = word["start"]
                    current_is_question = seg.get("is_student_question", False)
                current_end = word["end"]
            else:
                if current_start is not None:
                    intervals.append({
                        "start": current_start,
                        "end": current_end,
                        "is_student_question": current_is_question,
                    })
                    current_start = None
                    current_is_question = None

    if current_start is not None:
        intervals.append({
            "start": current_start,
            "end": current_end,
            "is_student_question": current_is_question,
        })

    return intervals

