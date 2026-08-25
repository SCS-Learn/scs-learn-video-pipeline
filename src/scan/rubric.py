"""What "a good lecture" means, expressed as a table you can argue with.

Every threshold the scanner uses lives in METRICS below -- nothing is buried in
the measurement code. That split is the whole point: `src/scan/*_metrics.py`
answer "what is this lecture like", and this file alone answers "is that good".
Retuning the grader is editing a table, not editing logic.

    python -m src.scan --explain-rubric        # print the whole table

Two scores come out, and the difference between them is the useful part:

    score       what the lecture is like as recorded
    potential   what it would be like after the remediation the pipeline can
                actually apply (loudness normalisation, denoise, letterboxing)

A lecture at 52/78 is quiet and hissy and entirely worth publishing. A lecture
at 52/54 is a monotone talk over a static slide, and no amount of encoder
settings fixes that. Sorting a semester by `score` finds work; sorting by
`potential` finds the lectures that are genuinely not worth the GPU hours.

Each metric declares:

    dimension  which of the five categories it rolls up into
    weight     relative weight WITHIN that dimension (dimensions are weighted
               against each other by DIMENSIONS, below)
    scale      how a raw number becomes a 0..1 sub-score:
                 ("ramp", bad, good)          linear, either direction
                 ("band", hard_lo, lo, hi, hard_hi)   inverted-U, 1.0 in [lo,hi]
                 ("bool",)                    True -> 1.0
    fixable    whether the pipeline can improve it after the fact. Drives
               `potential` and, more importantly, drives whether the advice is
               "re-record this" or "run loudnorm".
    tier       which scan tier measures it, so a partial scan reports honest
               coverage instead of scoring absent metrics as zero.
    why        the one line that justifies the thresholds. If you disagree with
               a number, this is the sentence to disagree with.

Where the numbers come from is recorded per metric. Several are calibrated
against the two lectures this repo has processed end to end (15-210 lecture 12,
a large hall with almost no audible questions; 17-635 lecture 13, a small class
with constant interaction and mixed-up diarization). Two lectures is not a
calibration set, which is exactly why the scanner also reports each metric's
percentile WITHIN the scanned cohort -- see `cohort` in report.py. On a real
semester, trust the percentile before you trust the absolute band.
"""

# How the five dimensions weigh against one another. Sums to 1.0.
#
# audio is the heaviest because it is the failure a viewer will not sit
# through, and because a lecture recording that is unintelligible is not
# recoverable by anything downstream of the microphone. delivery is second
# because "is this person worth watching for 90 minutes" is the question the
# scanner exists to answer. burden is last and small: it is a cost signal, not
# a quality one, and it is here so that two otherwise equal lectures sort by
# how much anonymization work they imply.
DIMENSIONS = {
    "audio":     {"weight": 0.28, "label": "Audio quality",
                  "blurb": "Can you hear and understand the lecturer?"},
    "delivery":  {"weight": 0.22, "label": "Delivery & engagement",
                  "blurb": "Is the lecturer worth listening to for 90 minutes?"},
    "visual":    {"weight": 0.17, "label": "Visual quality",
                  "blurb": "Is there a usable slide stream and a visible instructor?"},
    "structure": {"weight": 0.13, "label": "Content & structure",
                  "blurb": "Is it a taught lecture rather than admin and dead air?"},
    "burden":    {"weight": 0.20, "label": "Student exposure & burden (inverted)",
                  "blurb": "How few students are in it, and how much "
                           "anonymization work does it imply?"},
}

# Scan tiers, cheapest first. Each is a superset of the work of the one before.
# --tier picks how far to go; every tier's results are cached per lecture, so
# deepening a scan re-runs only the new tiers.
TIERS = ["probe", "signal", "vision", "speech"]

# Costs are measured on this corpus: 79-90 minute lectures, M5 laptop.
TIER_BLURB = {
    "probe":  "ffprobe plus metadata.json and chapters.json. ~1s.",
    "signal": "One audio decode (loudness, room tone, prosody) and one "
              "keyframe pass over each stream (slide changes, black lead, "
              "sharpness, exposure). No ML. ~40s.",
    "vision": "insightface over ~200 sampled camera frames: is the instructor "
              "in shot, how big, and how many students appear. ~15s on CoreML.",
    "speech": "Everything derived from transcript_classified.json, if the "
              "lecture has one. ~0.3s. It does NOT run ASR: transcription is a "
              "GPU stage and scanning a semester is not a reason to spend it, "
              "so a lecture with no transcript simply reports lower coverage.",
}


def ramp(bad, good):
    return ("ramp", bad, good)


def band(hard_lo, lo, hi, hard_hi):
    return ("band", hard_lo, lo, hi, hard_hi)


BOOL = ("bool",)


# --------------------------------------------------------------------------
# The table.
# --------------------------------------------------------------------------
METRICS = {

    # --- audio ------------------------------------------------------------
    "loudness_lufs": {
        "dimension": "audio", "weight": 1.0, "tier": "signal", "fixable": True,
        "label": "Integrated loudness", "unit": "LUFS",
        "scale": band(-34.0, -26.0, -14.0, -8.0),
        "why": "Only a gate against the extremes. Anything inside the band is "
               "one loudnorm pass from correct, so being quiet is not a "
               "quality problem -- it is a to-do. Below -34 the signal is so "
               "low that normalising it just raises the noise with it.",
    },
    "loudness_range_lu": {
        "dimension": "audio", "weight": 1.0, "tier": "signal", "fixable": True,
        "label": "Loudness range", "unit": "LU",
        "scale": band(1.0, 3.0, 13.0, 22.0),
        "why": "Under 3 LU is a hard-limited room mic with the life squeezed "
               "out of it; over 22 LU means the quiet passages will be "
               "inaudible on a laptop speaker even after normalisation, "
               "because the loud ones set the ceiling.",
    },
    "true_peak_dbtp": {
        "dimension": "audio", "weight": 1.0, "tier": "signal", "fixable": False,
        "label": "True peak", "unit": "dBTP",
        "scale": ramp(0.5, -3.0),
        "why": "Above about -0.5 dBTP the encoder will produce inter-sample "
               "clipping on playback. Already-clipped samples cannot be "
               "un-clipped, which is why this is not marked fixable.",
    },
    "clipped_pct": {
        "dimension": "audio", "weight": 1.5, "tier": "signal", "fixable": False,
        "label": "Clipped samples", "unit": "%",
        "scale": ramp(0.5, 0.0),
        "why": "Distortion that was baked in at record time. Half a percent of "
               "samples at full scale is audible crunch on every loud syllable.",
    },
    "snr_db": {
        "dimension": "audio", "weight": 3.0, "tier": "signal", "fixable": True,
        "label": "Speech-to-noise ratio", "unit": "dB",
        "scale": ramp(6.0, 26.0),
        "why": "The single best predictor of whether a lecture is pleasant to "
               "listen to. Measured as median speech-frame RMS against the "
               "10th-percentile frame RMS (the room tone). Below ~10 dB you "
               "get HVAC hiss under every word; above ~25 dB it is a clean "
               "lapel mic. Marked fixable because a denoise pass genuinely "
               "helps -- but only so far, so it stays heavily weighted in the "
               "raw score too.",
    },
    "noise_floor_dbfs": {
        "dimension": "audio", "weight": 1.5, "tier": "signal", "fixable": True,
        "label": "Noise floor", "unit": "dBFS",
        "scale": ramp(-32.0, -58.0),
        "why": "Absolute room tone, independent of how loud the speaker is. "
               "Catches the case where SNR looks fine only because someone was "
               "shouting over a fan.",
    },
    "level_stability_db": {
        "dimension": "audio", "weight": 1.5, "tier": "signal", "fixable": True,
        "label": "Level stability", "unit": "dB (sd of per-minute level)",
        "scale": ramp(6.0, 1.0),
        "why": "Standard deviation of the median speech level taken minute by "
               "minute. Catches the lecturer who walks away from a fixed mic, "
               "a radio pack drifting, or a recording spliced from two sources "
               "at different gains -- none of which show up in an integrated "
               "loudness figure, because that averages exactly the variation "
               "being looked for. The two reference lectures sit at 1.7 and "
               "0.9 dB, so anything past ~4 is a real fault.",
    },
    "dropped_word_pct": {
        "dimension": "audio", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Words lost to dropouts", "unit": "% of words",
        "scale": ramp(6.0, 0.5),
        "why": "Words the transcript places at a time where the audio is "
               "sitting at the room-tone floor. Needs the word timings, which "
               "is why it is a speech-tier metric: measured on level alone, "
               "'a short silence between speech' is indistinguishable from the "
               "gap between two ordinary words, and an earlier attempt at this "
               "metric duly reported 476 dropouts an hour on a clean lecture. "
               "Against the transcript both references land at 0.6%.",
    },
    "asr_confidence": {
        "dimension": "audio", "weight": 3.0, "tier": "speech", "fixable": False,
        "label": "ASR confidence", "unit": "mean avg_logprob",
        "scale": ramp(-0.95, -0.22),
        "why": "The best end-to-end intelligibility proxy available, and it is "
               "free because transcription already writes avg_logprob per "
               "segment. It folds mic quality, room acoustics, accent clarity "
               "and mumbling into one number -- precisely the combination a "
               "human means by 'hard to follow'. On the two reference lectures "
               "this sits around -0.30.",
    },
    "asr_poor_segment_pct": {
        "dimension": "audio", "weight": 2.0, "tier": "speech", "fixable": False,
        "label": "Poorly-transcribed segments", "unit": "%",
        "scale": ramp(30.0, 3.0),
        "why": "Share of segments below -0.6 avg_logprob. The mean can look "
               "healthy while a fifth of the lecture is guesswork; this catches "
               "the lecture that is fine at the lectern and mud at the "
               "whiteboard. Also predicts bad captions, which are a privacy "
               "surface here, not a nicety.",
    },

    # --- delivery ---------------------------------------------------------
    "pitch_variety_st": {
        "dimension": "delivery", "weight": 3.0, "tier": "signal", "fixable": False,
        "label": "Pitch variety", "unit": "semitones (sd)",
        "scale": ramp(0.8, 3.4),
        "why": "The monotone detector, and the highest-weighted delivery "
               "metric. Standard deviation of voiced-frame F0 in semitones, so "
               "it is comparable across low and high voices. Speakers rated "
               "engaging land near 3-5 st; a flat read is under 1.5. This is "
               "measured, not guessed, and it is the closest thing to an "
               "objective handle on 'engaging lecturer' that exists.",
    },
    "energy_variety_db": {
        "dimension": "delivery", "weight": 2.0, "tier": "signal", "fixable": False,
        "label": "Dynamic variety", "unit": "dB (sd)",
        "scale": ramp(1.5, 7.5),
        "why": "Standard deviation of speech-frame RMS. Pitch variety with no "
               "energy variety is sing-song; the two together are emphasis. "
               "Kept separate from loudness_range_lu, which is a programme "
               "measure over the whole file including silence.",
    },
    "speech_rate_wpm": {
        "dimension": "delivery", "weight": 2.0, "tier": "speech", "fixable": False,
        "label": "Speech rate", "unit": "words/min",
        "scale": band(85.0, 120.0, 180.0, 235.0),
        "why": "Measured over speaking time only, not wall clock, so pauses do "
               "not drag it down. Comprehension work puts the sweet spot for "
               "technical content around 130-160 wpm. Under 100 is a slog. The "
               "upper bound started at 215 and was moved out to 235 after "
               "15-210 lecture 12 measured 211 -- a genuinely fast lecturer "
               "the project has already published, so 211 has to score as "
               "'fast', not as 'unusable'.",
    },
    "rate_variability": {
        "dimension": "delivery", "weight": 1.0, "tier": "speech", "fixable": False,
        "label": "Pacing variation", "unit": "CV of per-minute wpm",
        "scale": band(0.02, 0.10, 0.32, 0.60),
        "why": "A lecturer who slows down for the hard part and speeds through "
               "the recap is doing something a metronome is not. Too high, "
               "though, and it is not pacing -- it is a stop-start delivery "
               "with long stalls.",
    },
    "dead_air_pct": {
        "dimension": "delivery", "weight": 2.0, "tier": "speech", "fixable": True,
        "label": "Dead air", "unit": "% of runtime",
        "scale": ramp(20.0, 2.0),
        "why": "Runtime in a gap of 5s or more between transcribed speech. "
               "Measured against the transcript rather than the waveform "
               "deliberately: a lecture hall between sentences is not quiet, "
               "and an RMS gate set loose enough to call it silence finds one "
               "gap in 79 minutes where the transcript finds nine. Some dead "
               "air is real teaching -- waiting after a question, writing on "
               "the board -- hence the generous threshold and the fixable flag: "
               "20% of a 90-minute lecture is 18 minutes of nothing, and it can "
               "be cut. Both reference lectures sit near 1.6%.",
    },
    "longest_dead_air_s": {
        "dimension": "delivery", "weight": 1.0, "tier": "speech", "fixable": True,
        "label": "Longest silence", "unit": "s",
        "scale": ramp(180.0, 20.0),
        "why": "Total dead air can be acceptable while one three-minute gap "
               "makes a viewer assume the video is broken and close the tab. "
               "The references peak at 12.6s and 17.6s.",
    },
    "filler_per_100w": {
        "dimension": "delivery", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Filler and verbal tics", "unit": "per 100 words",
        "scale": ramp(9.0, 1.0),
        "why": "Unambiguous fillers (um, uh, you know, I mean, basically) plus "
               "tag questions -- the '..., right?' at the end of a statement, "
               "which is the tic lecturers actually have. Tags matter here "
               "because Whisper strips most um and uh before anything can "
               "count them, so a filler metric built on those alone measures "
               "the ASR rather than the speaker; the references land at 1.3 "
               "and 2.3 per 100 words once tags are included, against 0.5 and "
               "0.8 without. Ambiguous words -- so, right, like, well -- are "
               "excluded entirely: they are ordinary connectives in a "
               "technical lecture. Not fixable; this pipeline does not edit "
               "disfluencies out of a recording.",
    },
    "interaction_per_hour": {
        "dimension": "delivery", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Class interaction", "unit": "student turns/hour",
        "scale": ramp(0.0, 7.0),
        "why": "A room that asks questions is a room that is following, and "
               "the exchange is usually the most valuable part of the "
               "recording. Saturates at 7/hour so a seminar does not outrank a "
               "good large-hall lecture purely on format. Note the tension with "
               "student_speech_pct under burden: interaction is good for the "
               "viewer and expensive for us, and the rubric says both out loud "
               "rather than pretending it is one-sided.",
    },
    "signpost_per_1000w": {
        "dimension": "delivery", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Signposting", "unit": "per 1000 words",
        "scale": ramp(2.0, 14.0),
        "why": "Discourse markers that tell a listener where they are: 'first', "
               "'next', 'recall that', 'the key idea', 'to summarise', 'so far "
               "we have'. Structure a listener can hear. Cheap to count, and it "
               "separates a prepared lecture from a rambled one.",
    },
    "class_question_per_hour": {
        "dimension": "delivery", "weight": 1.0, "tier": "speech", "fixable": False,
        "label": "Questions to the class", "unit": "per hour",
        "scale": ramp(2.0, 40.0),
        "why": "Instructor sentences ending in a question mark, with tag "
               "questions removed -- counting those, both references read ~160 "
               "an hour and the metric was measuring the word 'right'. Real "
               "ones ('What is the trade-off?', 'Can anyone confirm?') come to "
               "83 and 53 an hour. Read as a floor rather than a ranking: "
               "checking for understanding is a teaching behaviour, and it "
               "shows up in the transcript whether or not anyone answers.",
    },

    # --- visual -----------------------------------------------------------
    "slide_change_per_hour": {
        "dimension": "visual", "weight": 2.0, "tier": "signal", "fixable": False,
        "label": "Slide changes", "unit": "per hour",
        "scale": band(2.0, 14.0, 100.0, 260.0),
        "why": "From freezedetect over the screen stream, the same signal "
               "src/video/scenes.py already uses to decide its cuts. Under ~14 "
               "an hour there is effectively no slide track and the right-rail "
               "layout has nothing to show. Over ~260 the 'screen' is video "
               "playback or someone scrolling code, which the Scene A window "
               "handles badly.",
    },
    "longest_static_slide_s": {
        "dimension": "visual", "weight": 1.5, "tier": "signal", "fixable": True,
        "label": "Longest unchanged slide", "unit": "s",
        "scale": ramp(1200.0, 150.0),
        "why": "Twenty minutes on one slide means the lecture happened on the "
               "whiteboard. Marked fixable because scenes.py cuts to full-frame "
               "camera exactly there -- but only if the camera is any good, so "
               "read it next to instructor_in_frame_pct.",
    },
    "screen_black_pct": {
        "dimension": "visual", "weight": 2.0, "tier": "signal", "fixable": True,
        "label": "Black screen", "unit": "% of runtime",
        "scale": ramp(20.0, 1.0),
        "why": "Measured with blackdetect at -v info, because at -v error "
               "ffmpeg reports a wholly black file as having no black at all. "
               "The lead-in is trimmed by sync.py; black in the MIDDLE is a "
               "projector that was off, and nothing downstream removes it.",
    },
    "screen_aspect": {
        "dimension": "visual", "weight": 1.0, "tier": "probe", "fixable": True,
        "label": "Screen aspect fit", "unit": "score",
        "scale": ramp(0.0, 1.0),
        "why": "1.0 for 16:9 content, ~0.6 for a 4:3 deck pillarboxed into a "
               "16:9 frame. Not a failure -- layout._slide_filter letterboxes "
               "it correctly -- but a 4:3 deck uses about three quarters of the "
               "1380x776 slide window, so the type ends up smaller.",
    },
    "screen_height": {
        "dimension": "visual", "weight": 1.0, "tier": "probe", "fixable": False,
        "label": "Screen resolution", "unit": "px tall",
        "scale": ramp(480.0, 1080.0),
        "why": "The slide window is 1380x776. A 720p capture is upscaled into "
               "it and code on slides goes soft; below 480p it is unreadable "
               "and no encoder setting recovers it.",
    },
    "camera_sharpness": {
        "dimension": "visual", "weight": 2.0, "tier": "signal", "fixable": False,
        "label": "Camera sharpness", "unit": "Laplacian variance @480x270",
        "scale": ramp(120.0, 800.0),
        "why": "Variance of the Laplacian over sampled camera frames -- the "
               "standard cheap focus measure -- always computed at 480x270 so "
               "the number is comparable across source resolutions. The two "
               "reference cameras measure 1870 and 1464, so the band is set "
               "well below them: they are in focus, and the metric exists to "
               "catch the one that is not. Read it as a floor rather than a "
               "ranking, because the measure also rises with scene detail -- a "
               "busy lecture hall out-scores a plain wall at equal focus.",
    },
    "camera_exposure": {
        "dimension": "visual", "weight": 1.5, "tier": "signal", "fixable": True,
        "label": "Camera exposure", "unit": "score",
        "scale": ramp(0.0, 1.0),
        "why": "Penalises crushed blacks, blown highlights and low contrast on "
               "sampled frames. The classic lecture-hall failure is the lights "
               "down for the projector and the lecturer in silhouette. Partly "
               "fixable with a curve, which is why it is not weighted higher.",
    },
    "instructor_in_frame_pct": {
        "dimension": "visual", "weight": 2.5, "tier": "vision", "fixable": False,
        "label": "Instructor in frame", "unit": "% of sampled frames",
        "scale": ramp(35.0, 92.0),
        "why": "Scene B is a full-bleed camera and the Scene A rail is a "
               "person-shaped hole. A camera pointed at an empty lectern for "
               "half the lecture breaks both. Highest-weighted visual metric "
               "because there is no remedy short of re-shooting.",
    },
    "instructor_face_px": {
        "dimension": "visual", "weight": 2.0, "tier": "vision", "fixable": False,
        "label": "Instructor apparent size", "unit": "px face height @1080p",
        "scale": ramp(12.0, 55.0),
        "why": "The brand rail is 432px wide against a 1920px source, so the "
               "instructor is scaled to roughly 22%. A face under ~12px in the "
               "source is a smudge in the rail. assets/brand/PIPELINE.md "
               "already flags that this layout renders him ~1.7x smaller than "
               "the pre-brand corner PiP, so the margin here is thin.",
    },

    # --- structure --------------------------------------------------------
    "has_opening": {
        "dimension": "structure", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Opens with an agenda", "scale": BOOL,
        "why": "Any of 'today we', 'last time', 'the plan for', 'we will cover' "
               "in the first three minutes. A lecture that tells you where it "
               "is going survives being watched out of sequence, which is "
               "exactly how open courseware gets watched.",
    },
    "has_closing": {
        "dimension": "structure", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Closes with a summary", "scale": BOOL,
        "why": "'to summarise', 'next time', 'the key takeaway', 'what we "
               "covered' in the last five minutes. Also a useful tell that the "
               "recording did not simply stop mid-thought.",
    },
    "content_density": {
        "dimension": "structure", "weight": 1.5, "tier": "speech", "fixable": False,
        "label": "Content density", "unit": "content words/min",
        "scale": ramp(30.0, 95.0),
        "why": "Words per minute of wall clock after stripping stopwords and "
               "fillers. Unlike speech_rate_wpm this DOES include the silence, "
               "so it measures how much teaching actually happened per minute "
               "of video the viewer has to sit through.",
    },
    "lexical_diversity": {
        "dimension": "structure", "weight": 1.0, "tier": "speech", "fixable": False,
        "label": "Lexical diversity", "unit": "type/token (sampled)",
        "scale": band(0.15, 0.28, 0.60, 0.85),
        "why": "Measured on fixed 500-token windows so it does not simply fall "
               "with length. Low is repetitive; suspiciously high on a lecture "
               "transcript usually means ASR noise rather than a rich "
               "vocabulary, hence the upper bound.",
    },
    "admin_talk_pct": {
        "dimension": "structure", "weight": 2.0, "tier": "speech", "fixable": True,
        "label": "Administrative talk", "unit": "% of words",
        "scale": ramp(28.0, 3.0),
        "why": "Words inside spans about homework deadlines, exam logistics, "
               "office hours, the grading scheme and the course platform. This "
               "is the metric that most separates a lecture worth publishing "
               "openly from one that is 40% course admin nobody outside the "
               "section can use. Fixable in the sense that it can be cut, and "
               "the scanner reports the spans so it can be.",
    },
    "topic_focus": {
        "dimension": "structure", "weight": 1.0, "tier": "speech", "fixable": False,
        "label": "Topic focus", "unit": "0-1",
        "scale": ramp(0.05, 0.35),
        "why": "Vocabulary overlap between the thirds of the lecture. High "
               "means one subject developed throughout; low means a grab-bag. "
               "Neither is wrong, but a focused lecture is far more useful as a "
               "standalone open-courseware unit.",
    },
    "chapter_count": {
        "dimension": "structure", "weight": 1.0, "tier": "probe", "fixable": True,
        "label": "Panopto chapters", "unit": "count",
        "scale": ramp(0.0, 8.0),
        "why": "Straight out of chapters.json. Free structure metadata, and it "
               "means someone already thought about the lecture's sections. "
               "Fixable: chapters can be generated from the transcript.",
    },
    "duration_min": {
        "dimension": "structure", "weight": 1.5, "tier": "probe", "fixable": False,
        "label": "Duration", "unit": "min",
        "scale": band(12.0, 38.0, 95.0, 150.0),
        "why": "Under ~12 minutes it is a fragment or a failed recording, not a "
               "lecture. Over ~150 the recorder was left running after the room "
               "emptied -- which also inflates dead_air_pct, so the two "
               "corroborate each other.",
    },

    # --- burden (inverted cost: 1.0 means cheap and safe) -----------------
    "student_face_pct": {
        "dimension": "burden", "weight": 2.5, "tier": "vision", "fixable": False,
        "label": "Frames showing any student", "unit": "% of sampled frames",
        "scale": ramp(40.0, 0.0),
        "why": "The most direct measure of the thing we most want to avoid: "
               "how much of the lecture has a non-instructor face on screen at "
               "all. Distinct from multi_face_pct, which needs two faces "
               "together -- a camera that cuts to a single student asking a "
               "question exposes that student just as completely. Scores a "
               "clean 1.0 only at zero, because zero students on camera is the "
               "goal, not merely few. Both reference lectures are near it: "
               "17-635 lecture 13 has no student ever in frame, and 15-210 "
               "lecture 12 has one in about 5% of sampled frames.",
    },
    "student_face_clusters": {
        "dimension": "burden", "weight": 2.5, "tier": "vision", "fixable": False,
        "label": "Distinct student faces", "unit": "clusters",
        "scale": ramp(10.0, 0.0),
        "why": "Each distinct non-instructor face cluster is somebody whose "
               "face has to be pixelated correctly for the whole runtime, and "
               "one more chance for the fail-closed detector to be handed an "
               "ambiguous crop. Cost and risk at once, and the count is how "
               "many separate people are at stake -- so the band bottoms out "
               "at 10 rather than 16, and only zero scores full marks.",
    },
    "multi_face_pct": {
        "dimension": "burden", "weight": 2.0, "tier": "vision", "fixable": False,
        "label": "Frames with several faces", "unit": "%",
        "scale": ramp(45.0, 2.0),
        "why": "How often the camera sees the room rather than the lecturer. A "
               "camera that pans across the audience is the worst case for this "
               "pipeline: every frame is anonymization work and any miss is a "
               "published student.",
    },
    "student_speech_pct": {
        "dimension": "burden", "weight": 2.0, "tier": "speech", "fixable": False,
        "label": "Student speech", "unit": "% of speech time",
        "scale": ramp(14.0, 0.5),
        "why": "Every second of it gets muted and, where it is a question, "
               "replaced with a rendered card -- and cards.py is a 20-minute "
               "local-only stage. Read alongside interaction_per_hour, which "
               "rewards the same thing from the viewer's side.",
    },
    "named_mentions_per_hour": {
        "dimension": "burden", "weight": 0.75, "tier": "speech", "fixable": False,
        "label": "Distinct personal names", "unit": "per hour",
        "scale": ramp(25.0, 1.0),
        "why": "A PII tripwire, not a classifier -- weighted low for that "
               "reason, and the candidate list is reported so a human can look. "
               "An instructor who calls on students by name publishes those "
               "names in the captions even after the audio is muted, which is "
               "the whole point of catching it. Counts DISTINCT capitalised "
               "mid-sentence tokens occurring at most four times, excluding a "
               "technical stoplist: students get named once or twice, "
               "LangChain gets named forty times. That filter leaves exactly "
               "four candidates on 15-210 lecture 12, all real names; on "
               "17-635 lecture 13 it leaves nine, of which six are real and "
               "three (Indian, Lankchain, Tampa) are Whisper capitalising "
               "nouns. Expect roughly that precision.",
    },
    "sync_risk": {
        "dimension": "burden", "weight": 2.0, "tier": "signal", "fixable": False,
        "label": "Sync headroom", "unit": "0-1",
        "scale": ramp(0.0, 1.0),
        "why": "1.0 when the camera/screen duration delta comfortably covers "
               "the screen's black lead; falls towards 0 as the residual black "
               "approaches what the duration alignment removes. CLAUDE.md "
               "records lecture 12 clearing this by 89.7s -- a lecture that "
               "does NOT clear it publishes minutes of black, and verify.py "
               "passes it, because black encodes perfectly well.",
    },
    "est_pipeline_hours": {
        "dimension": "burden", "weight": 1.5, "tier": "probe", "fixable": False,
        "label": "Estimated processing time", "unit": "hours",
        "scale": ramp(8.0, 1.5),
        "why": "Rough wall-clock for the full local pipeline, driven by "
               "duration and resolution. Purely a scheduling signal, and the "
               "lightest weight in the lightest dimension.",
    },
}


# --------------------------------------------------------------------------
# Gates. A failed gate is not a low score -- it is "do not publish this",
# whatever the score says. They are deliberately few and deliberately blunt.
# --------------------------------------------------------------------------
GATES = [
    {"id": "media_readable", "label": "Camera and screen both decode",
     "why": "Nothing downstream can run. Usually a truncated download."},
    {"id": "has_audio", "label": "Camera has an audio stream",
     "why": "The audio pass, the transcript and every timing in the pipeline "
            "come off this track."},
    {"id": "not_silent", "label": "Speech is actually present",
     "why": "A recording of an empty room, or a mic that was never switched on."},
    {"id": "duration_sane", "label": "Runtime between 8 min and 4 h",
     "why": "Outside that it is a fragment or a recorder left running "
            "overnight, and either way it is not a lecture."},
    {"id": "intelligible", "label": "Speech is transcribable",
     "why": "Mean avg_logprob above -1.1. Below that the ASR is guessing, the "
            "captions would be fiction, and the speaker-level muting that "
            "protects students is built on those same word timings."},
    {"id": "instructor_visible", "label": "Instructor is in shot",
     "why": "Present in at least 25% of sampled frames. Both brand scenes are "
            "built around a person being there."},
    {"id": "sync_recoverable", "label": "Camera and screen can be aligned",
     "why": "Duration delta under 25% of runtime. Beyond that they are not two "
            "views of the same event and sync.py will produce nonsense."},
]

# score -> (grade, verdict, blurb). Checked top down.
GRADES = [
    (85.0, "A", "publish",
     "Strong on every axis. Queue it."),
    (72.0, "B", "publish",
     "Good. Any weak metric is fixable in post."),
    (58.0, "C", "review",
     "Usable but flawed. Read the remediation list before committing GPU time."),
    (44.0, "D", "marginal",
     "Publishable only if the course needs the coverage. Expect complaints."),
    (0.0,  "F", "skip",
     "Not worth the pipeline hours unless it is re-recorded."),
]


def score_metric(metric_id, value):
    """Raw measurement -> 0..1 sub-score, or None when not measured."""
    if value is None:
        return None
    spec = METRICS[metric_id]
    kind = spec["scale"][0]
    if kind == "bool":
        return 1.0 if value else 0.0
    if kind == "ramp":
        _, bad, good = spec["scale"]
        if good == bad:
            return 1.0
        return max(0.0, min(1.0, (float(value) - bad) / (good - bad)))
    if kind == "band":
        _, hard_lo, lo, hi, hard_hi = spec["scale"]
        v = float(value)
        if v <= hard_lo or v >= hard_hi:
            return 0.0
        if v < lo:
            return (v - hard_lo) / (lo - hard_lo) if lo > hard_lo else 1.0
        if v > hi:
            return (hard_hi - v) / (hard_hi - hi) if hard_hi > hi else 1.0
        return 1.0
    raise ValueError(f"unknown scale {kind!r} on {metric_id}")


# What a metric becomes if the pipeline does the remediation it is capable of.
# Used only for `potential`; deliberately conservative -- these are outcomes
# already observed from the corresponding ffmpeg pass, not best cases.
REMEDIATED = {
    "loudness_lufs":         -18.0,   # loudnorm targets this directly
    "loudness_range_lu":       9.0,   # loudnorm pulls the range into band
    "snr_db":                 "+6",   # afftdn buys about 6 dB before artefacts
    "noise_floor_dbfs":       "-8",   # same pass, measured on the floor itself
    "dead_air_pct":            4.0,   # gaps >5s are cuttable
    "longest_dead_air_s":     25.0,
    "screen_black_pct":        1.0,   # sync.py trims the lead
    "longest_static_slide_s": 200.0,  # scenes.py cuts away from a dead slide
    "camera_exposure":       "+0.25",
    "screen_aspect":           1.0,   # letterboxed correctly by _slide_filter
    "admin_talk_pct":          5.0,   # the spans are reported, so they are cuttable
    "chapter_count":           8.0,   # generatable from the transcript
}


def remediated_value(metric_id, value):
    """Best plausible post-remediation value, for the `potential` score."""
    if value is None or metric_id not in REMEDIATED:
        return value
    target = REMEDIATED[metric_id]
    if isinstance(target, str):                      # relative adjustment
        return float(value) + float(target)
    # Absolute target: only ever an improvement, never a downgrade.
    now = score_metric(metric_id, value)
    then = score_metric(metric_id, target)
    return target if (then is not None and now is not None and then > now) else value


def grade_for(score, gates_failed):
    if gates_failed:
        return "F", "skip", "Failed a hard gate: " + "; ".join(gates_failed) + "."
    for threshold, letter, verdict, blurb in GRADES:
        if score >= threshold:
            return letter, verdict, blurb
    return "F", "skip", GRADES[-1][3]


def dimension_ids(dimension):
    return [k for k, v in METRICS.items() if v["dimension"] == dimension]


def _wrap(text, width):
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


def explain(width=78):
    """The whole rubric as text. `python -m src.scan --explain-rubric`."""
    out = []
    add = out.append
    add("=" * width)
    add("LECTURE SCANNER RUBRIC")
    add("=" * width)
    add("")
    add("Two scores per lecture:")
    add("  score      the lecture as recorded")
    add("  potential  after the remediation this pipeline can actually apply")
    add("The gap between them is the difference between 'needs work' and")
    add("'not worth the GPU hours'.")
    add("")
    add("TIERS (--tier, cheapest first; results cached per lecture)")
    for t in TIERS:
        for i, line in enumerate(_wrap(TIER_BLURB[t], width - 11)):
            add(f"  {t:<8} {line}" if i == 0 else f"  {'':<8} {line}")
    add("")
    add("HARD GATES -- any failure means 'skip', whatever the score")
    for g in GATES:
        add(f"  [{g['id']}] {g['label']}")
        for line in _wrap(g["why"], width - 6):
            add(f"      {line}")
    add("")
    for dim, meta in DIMENSIONS.items():
        add("-" * width)
        add(f"{meta['label'].upper()}   weight {meta['weight']:.0%}")
        add(meta["blurb"])
        add("-" * width)
        ids = dimension_ids(dim)
        total_w = sum(METRICS[i]["weight"] for i in ids)
        for mid in ids:
            s = METRICS[mid]
            share = s["weight"] / total_w
            kind = s["scale"][0]
            if kind == "ramp":
                _, bad, good = s["scale"]
                rng = f"{bad:g} (score 0) -> {good:g} (score 1)"
            elif kind == "band":
                _, hl, lo, hi, hh = s["scale"]
                rng = f"0 at/below {hl:g}, best {lo:g}..{hi:g}, 0 at/above {hh:g}"
            else:
                rng = "true / false"
            add(f"  {s['label']}  [{mid}]")
            add(f"      {share:>5.1%} of dimension | {s.get('unit', 'flag')} | {rng}")
            add(f"      tier={s['tier']}  "
                f"{'fixable in post' if s['fixable'] else 'inherent to the recording'}")
            for line in _wrap(s["why"], width - 6):
                add(f"      {line}")
            add("")
    add("-" * width)
    add("GRADES")
    for threshold, letter, verdict, blurb in GRADES:
        add(f"  {letter}  >= {threshold:>4.0f}  {verdict:<9} {blurb}")
    return "\n".join(out)
