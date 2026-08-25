# Lecture scanner

Grades a semester of downloaded lectures so you can decide which are worth
putting through the pipeline, and in what order.

```bash
# what it measures and why every threshold is what it is
python -m src.scan --explain-rubric

# sweep a semester cheaply first (~40s a lecture)
python -m src.scan --courses-dir data/fall2026 --tier signal --jobs 6

# go deeper on the survivors and write reports
python -m src.scan --courses-dir data/fall2026 --tier speech \
    --out reports/fall2026 --format md,csv,html

# one lecture, every metric
python -m src.scan --lecture-dir data/15210-lecture12 --explain
```

Everything runs on a laptop. Nothing here needs a GPU or PSC.

---

## Two numbers per lecture

```
score       the lecture as recorded
potential   the same lecture after remediation the pipeline can actually apply
```

The gap is the useful part. **52 → 78** is quiet, hissy, and worth publishing:
one `loudnorm` pass away from fine. **52 → 54** is a monotone talk over a slide
nobody advanced, and no encoder setting fixes that. Sort by `score` to find
work; sort by `potential` to find lectures that are not worth the hours.

## Five dimensions

| Dimension | Weight | Question |
|---|---:|---|
| Audio quality | 28% | Can you hear and understand the lecturer? |
| Delivery & engagement | 22% | Is this worth 90 minutes of someone's attention? |
| Visual quality | 17% | Is there a usable slide track and a visible instructor? |
| Content & structure | 13% | Is it teaching, or is it admin and dead air? |
| Student exposure & burden | 20% | How few students are in it? |

44 metrics across those five. `--explain-rubric` prints every one with its
band, its tier, whether it is fixable in post, and the sentence justifying its
thresholds. **All of it lives in `rubric.py` as a data table** — retuning the
grader is editing that table, never the measurement code.

### On "engaging"

The honest handle is `pitch_variety_st`: the standard deviation of voiced-frame
F0, in semitones so a bass and a tenor compare. Monotone delivery measures
under ~1.5; animated delivery measures over ~3. The two reference lectures come
in at 4.11 and 2.95. It is weighted highest in its dimension because it is the
one aspect of delivery a machine can measure without pretending to judge
charisma.

Supporting it: dynamic variety, speech rate and its variation, dead air, filler
and tag-question rate, class interaction, and signposting.

### On "minimal students"

Four metrics, 20% of the total score:

- `student_face_pct` — frames with any non-instructor face. Scores 1.0 only at
  **zero**, because zero is the goal, not "few".
- `student_face_clusters` — how many distinct people would need pixelating.
- `multi_face_pct` — how often the camera is looking at the room.
- `student_speech_pct` — how much audio gets muted and carded.

Note the deliberate tension: `interaction_per_hour` **rewards** a room that asks
questions, because that is good for the viewer, while `student_speech_pct`
**penalises** it, because it is expensive and risky for us. The rubric states
both rather than pretending the trade-off does not exist.

## Tiers

Cumulative, cheapest first, cached per lecture in `scan.json`. Deepening a scan
re-runs only what is new. Measured on 79–90 minute lectures, M5 laptop:

| Tier | Cost | What it adds |
|---|---|---|
| `probe` | ~1 s | ffprobe + `metadata.json` / `chapters.json` |
| `signal` | ~40 s | loudness, SNR, prosody, slide changes, black, exposure |
| `vision` | ~15 s | instructor presence and size, student faces (CoreML) |
| `speech` | ~0.3 s | everything from `transcript_classified.json` |

`speech` reads a transcript if the lecture has one and reports reduced coverage
if not. It never runs ASR itself — transcription is a GPU stage and scanning a
semester is not a reason to spend it.

A tier that did not run is **not** scored as zero. Dimensions renormalise over
what was measured and report `coverage`. Below 55% coverage a lecture gets a
provisional score and **no grade** — a probe-only pass over this corpus scored
93.8 from three metrics, and a number that confident that early is worse than
no number.

## Gates

Separate from the score, because "how good is it" and "publish it at all" are
different questions and averaging them answers neither. Any failed gate means
`skip` whatever the score says: unreadable media, no audio, silence, absurd
runtime, untranscribable speech, no instructor in shot, unalignable streams.

A gate whose measurement is absent is skipped, not failed. Unmeasured is not
bad.

## Calibration, and how much to trust it

Thresholds were set against the two lectures this repo has taken end to end
(15-210 lecture 12, a large hall; 17-635 lecture 13, a small interactive
class). **Two lectures is not a calibration set.** That is exactly why the
scanner also reports each metric's percentile *within the scanned cohort* —
over a real semester, trust the percentile before the absolute band.

Several numbers were moved during development because the measurement was
wrong, and those corrections are recorded in each metric's `why`:

- Dead air measured on the waveform found 1 gap in 79 minutes where the
  transcript found 9 — a lecture hall between sentences is not quiet. Dead air
  is now transcript-derived.
- A level-based dropout detector reported **476 dropouts an hour** on a clean
  recording. Replaced by per-minute level stability plus, where a transcript
  exists, words whose audio sits at the room-tone floor.
- Counting every question-final sentence gave ~160 questions an hour on both
  references, because most were "…, right?". Tag questions are now counted as
  the filler they are.

## Validation

The screen path reproduces two independently documented values on 15-210
lecture 12: a black lead of **630.2 s** against CLAUDE.md's 629.0, and a sync
margin of **88.4 s** against its 89.7 — both inside the 2.4 s keyframe grid.
The vision path independently reproduces face_anon's documented quirk of that
instructor splitting into exactly three clusters.

## Files

| File | Role |
|---|---|
| `rubric.py` | **every threshold and weight.** The thing to argue with |
| `media.py` | ffmpeg/ffprobe plumbing; keyframe grid, one-pass audio decode |
| `audio_metrics.py` | loudness, SNR, room tone, level stability, prosody |
| `video_metrics.py` | slide changes, black lead, aspect, sharpness, exposure |
| `face_metrics.py` | instructor presence, student exposure (face_anon's detector) |
| `speech_metrics.py` | transcript: pacing, structure, interaction, PII flag |
| `lexicon.py` | filler / signpost / admin / name-stoplist word lists |
| `score.py` | measurements → dimensions → grade, gates, remediation |
| `report.py` | markdown, CSV, HTML, per-lecture detail, cohort percentiles |
| `discover.py` | find lectures, pick the raw camera/screen streams |
| `scanner.py` | run the tiers for one lecture, cache the measurements |

## Things it does not do

- It does not transcribe. No transcript means the speech tier is unmeasured.
- It does not decide who the instructor is for anonymization purposes.
  `face_anon --preview` does that, and getting it backwards blurs the lecturer
  and publishes the students.
- `named_mentions_per_hour` is a **review flag**, not a classifier. On 17-635
  lecture 13 it returns nine candidates of which six are real names and three
  are Whisper capitalising ordinary nouns. Read the list; do not act on the
  count alone.
- The instructor ranking measures *delivery*, not teaching. A lecturer handed a
  bad room and a mic that was not switched on ranks low for reasons that are
  not theirs.
