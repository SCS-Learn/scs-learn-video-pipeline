# scs-learn-video-pipeline

Anonymizes CMU lecture recordings so they can be published openly. Two privacy
passes: student **audio** is muted and their questions replaced with rendered
cards, and student **faces** are pixelated while the instructor is left clear.

PSC grant `cis260220p` ("Privacy-Preserving AI Pipeline for Open Publication of
University Lecture Recordings").

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # audio stages
pip install -r requirements-video.txt    # face anonymization (separate: see below)
cp .env.sample .env                      # then fill it in

python -m src.pipeline --lecture-dir data/15210-lecture12 --dry-run
```

Every stage takes `--lecture-dir`, so nothing needs editing to process a
different lecture. `src/pipeline.py` runs them in the right order.

### Required setup: sign in to PSC

The GPU stages run on PSC Bridges-2, and **you must authenticate interactively
before an agent can do anything there.** No credential is stored anywhere.

```bash
./scripts/psc.sh install-config   # writes the ~/.ssh/config stanzas, once
ssh psc                           # log in, then `exit`  -- REAL terminal needed
ssh psc-dtn                       # again, for file transfers
./scripts/psc.sh status           # confirm
```

Both sessions persist ~8h via SSH multiplexing; later commands reuse them with
no password. **These two `ssh` commands must be run by a human in a real
terminal** — they need a TTY for the password prompt, so an agent cannot do
them. Everything afterwards an agent can drive.

Your PSC password is a **Kerberos** password, separate from CMU SSO — reset at
<https://apr.psc.edu>, or `kpasswd` (never `passwd`) on-system. Each person
needs their own PSC account under the grant; credentials are non-transferable.

---

## Two machines, and what runs where

Getting this wrong wastes hours.

| Where | What runs there |
|---|---|
| **Your laptop** | `ingestion.py` (needs a real browser), `panopto_download.py`, `cards.py`, all editing, CPU testing |
| **PSC login node** | `ls`, `squeue`, `sacct`, `sinfo`, `module avail`, `pip list`. **Nothing else.** |
| **PSC DTN** (`data.bridges2.psc.edu`) | all file transfers |
| **PSC compute** (`sbatch`/`interact`) | `transcription.py`, `face_anon.py` |

### Login node policy — do not violate

> "All production computing must be done on Bridges-2's compute nodes, NOT on
> Bridges-2's login nodes." · "You cannot use Bridges-2's login nodes for your
> work." · "File transfers can no longer be initiated from the Bridges-2 login
> nodes." — Bridges-2 User Guide

Over a login session **never** import `torch`/`whisperx`/`insightface`, run
`ffmpeg`, run a broad `find` over `/jet` or `/ocean`, transcode, or infer. Use
`sbatch` or `interact`. `scripts/psc.sh sync` refuses to transfer over a login
node.

---

## Pipeline stages

| # | Stage | Runs on | Produces |
|---|---|---|---|
| 1 | `src/ingestion.py` | laptop | `manifest.json` (manual browser SSO+Duo) |
| 2 | `src/panopto_download.py` | laptop → DTN | `camera.mp4`, `screen.mp4` |
| 3 | `src/sync.py` | CPU | `screen_sync.mp4` |
| 4 | `src/audio/transcription.py` | **GPU** | `transcript_classified.json` |
| 5 | `src/audio/audio.py` | CPU | `camera_muted.mp4` |
| 6 | `src/video/face_anon.py` | **GPU, not V100** | `camera_muted_anon.mp4` |
| 7 | `src/audio/cards.py` | **laptop only** | `screen_with_cards.mp4` |
| 8 | `src/audio/captions.py` | CPU | `captions.srt` |
| 9 | `src/assembly/assembly.py` | CPU | final `<key>.mp4` |

Order constraints that aren't obvious:
- **face_anon runs after audio** (it consumes `camera_muted.mp4`, carrying the
  muted audio through) and **before assembly** (which composites the anonymized
  camera in).
- cards needs `sync` to have produced `screen_sync.mp4`.

Only stages 4 and 6 need a GPU. Everything else is ffmpeg/CPU.

**`cards.py` takes 20+ minutes and is not to be run on PSC.** `src/pipeline.py`
enforces this — it refuses to run `cards` when it detects it is on PSC. Run the
pipeline on your laptop, or `--skip cards` there and do it locally.

Submit GPU stages with:
```bash
./scripts/psc.sh sbatch scripts/psc_face_anon.sbatch
```

---

## Environments

| Env | Where | Contents |
|---|---|---|
| `.venv` | laptop | `requirements.txt` |
| `scs-learn` | PSC | audio: whisperx, torch, pyannote, anthropic, pysrt, soundfile, pillow |
| `scs-video` | PSC | video: insightface, **onnxruntime-gpu**, opencv, sklearn |

Separate on purpose: `insightface` pulls plain `onnxruntime`, which conflicts
with `onnxruntime-gpu`. Locally that split is `requirements.txt` vs
`requirements-video.txt`.

`ffmpeg` is **not on PSC's default PATH** — `module load ffmpeg` (4.3.1, with
libx264/libx265).

---

## Gotchas that will cost you hours

**`face_anon.py` cannot run on a V100.** Bridges-2's GPU partition is
heterogeneous (`v100-16`, `v100-32`, `l40s-48`, `h100-80`). V100 is compute
capability sm_70, and the cuDNN in `onnxruntime-gpu` 1.28 has no sm_70 kernels.
It dies after ~3 seconds:

```
err 209 no kernel image is available for execution on the device
CUDNN_FE failure 11 ; Conv node. Name:'Conv_0'
```

Use `--gpus=l40s-48:8` or `--gpus=h100-80:8`. **This, not slowness, is why the
stage never ran on PSC.** `transcription.py` is fine on V100 because PyTorch
still ships sm_70 kernels — same node, opposite outcome.

**There is no NVENC path on PSC.** PSC's `ffmpeg/4.3.1` is built without nvenc
(hwaccels: vdpau/vaapi/vulkan only), and most Bridges-2 GPUs lack the silicon
anyway (V100/A100/H100 have none; only L40S does). All H.264 encoding is
`libx264` on CPU. Don't "fix" a slow encode by reaching for `h264_nvenc`.

**Don't reintroduce ffmpeg `overlay` in `cards.py`.** The card template is a
fully opaque 1920x1080 PNG and the screen is exactly 1920x1080, so a card
*replaces* the frame — nothing to composite. The old approach opened one
full-length looped image input per question, costing O(questions × duration).

**HuggingFace gating.** `transcription.py` pins
`pyannote/speaker-diarization-3.1`. Accept its conditions **and** those of
`pyannote/speaker-diarization-community-1`, which 3.1 pulls PLDA files from.
Otherwise: 403.

**Transcripts are per-lecture.** They live in the lecture dir, not a shared
`data/transcription/`. The old shared layout meant a second lecture silently
overwrote the first one's transcript and every downstream stage built the wrong
video. `src/paths.py` still reads the legacy location if the new one is absent.

**`/jet/home` is 25GB and typically near full.** Keep working data under
`/ocean/projects/cis260220p/$USER/`.

**GPU SUs are scarce.** ~495 GPU-hours for the whole grant; an 8-GPU node burns
8/hour. Regular (CPU) is 5,000. Check with `./scripts/psc.sh quota`. Keep a
`--time` cap on every job.

---

## Agent operating rules

1. **Never run compute on a PSC login node.** If unsure a command is "light",
   assume it is not.
2. **Never run `cards.py` on PSC.** 20+ minutes; laptop only.
3. **Never use `--gpus=v100-*` for `face_anon.py`.** It will crash.
4. **Keep `--time` on job scripts.** A runaway 8-GPU job is a visible fraction
   of a 495 GPU-hour grant.
5. **Dry-run first.** `src.pipeline --dry-run`, `cards.py --dry-run`,
   `face_anon.py --benchmark`.
6. **Verify the instructor before a full anonymization run:** `face_anon.py
   --preview` writes `face_clusters.png`. Backwards means blurring the
   instructor and exposing students.
7. **Anonymization is fail-closed; keep it that way.** A face whose identity
   can't be established is blurred, never skipped. Don't loosen
   `--sim-threshold` to make output look nicer — at 0.3 a measurably different
   person (0.348) passed as the instructor and went unblurred.
8. **`assembly.py` refuses to publish an unanonymized video.** If
   `camera_muted_anon.mp4` is missing it errors out rather than quietly using
   the un-anonymized camera. Don't reach for `--allow-unanonymized` casually.
9. **Never commit secrets.** `.env` and `.env.bak*` are gitignored.
10. **Ask before `git push`** or anything else leaving this machine.

---

## Known limitations

- **`face_anon.py` has never completed a full lecture.** The only PSC attempt
  hit the V100 crash above; local testing covered a 300-frame window.
- **Question detection may be conservative.** On lecture 12, only 4 of 1,237
  segments were flagged `is_student_question` across 91 minutes. Spot-check
  before treating card coverage as complete — this is a privacy guarantee.
- **`captions.py --polish` is opt-in.** The ASR-mishear correction rewrites
  caption text, so review the diff before publishing.
- **`ingestion.py` needs a human.** CMU's SSO form was too brittle to script,
  so it opens a browser and waits for you to complete SSO + Duo.
