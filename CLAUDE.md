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
ssh psc                           # log in, then `exit`   -- REAL terminal needed
ssh -N psc-dtn                    # for transfers; leave it running (see below)
./scripts/psc.sh status           # confirm
```

The DTN needs `-N` (request no remote command). It **refuses interactive
shells** — a plain `ssh psc-dtn` ends in `Login denied: Only file transfers are
allowed on this account`, which looks like a failure but is expected.
Authentication still succeeds and the master socket is still created, so
scp/sftp/rsync work either way.

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
| **PSC compute** (`sbatch`/`interact`) | `transcription.py`, `face_anon.py`, `track_instructor.py` |

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
| 3 | `src/sync.py` | CPU | `screen_sync.mp4` (+ `camera_sync.mp4`) |
| 4 | `src/audio/transcription.py` | **GPU** | `transcript_classified.json` |
| 5 | `src/audio/audio.py` | CPU | `camera_muted.mp4` |
| 6 | `src/video/face_anon.py` | **GPU, not V100** | `camera_muted_anon.mp4` |
| 7 | `src/video/track_instructor.py` | **GPU, not V100** | `camera_muted_anon_tracked.mp4` |
| 8 | `src/audio/cards.py` | **laptop only** | `screen_with_cards.mp4` |
| 9 | `src/audio/captions.py` | CPU | `captions.srt` |
| 10 | `src/assembly/assembly.py` | CPU | final `<key>.mp4` |

Order constraints that aren't obvious:
- **face_anon runs after audio** (it consumes `camera_muted.mp4`, carrying the
  muted audio through) and **before assembly** (which composites the anonymized
  camera in).
- **track_instructor runs after face_anon.** It crops in on the instructor, and
  zooming an *un*-anonymized camera makes any student in frame more
  identifiable, not less. The stage warns loudly if its input has no `anon` in
  the name, but the ordering is what actually protects you.
- cards needs `sync` to have produced `screen_sync.mp4`.

Only stages 4, 6 and 7 need a GPU. Everything else is ffmpeg/CPU.

`track_instructor` is optional in effect — `assembly` falls back to the
uncropped camera when its output is absent, and logs which one it used either
way. Skip it with `--skip track_instructor`. It is on the stage list rather
than left as a manual command because `assembly` *prefers* its output whenever
the file exists: off the list, two runs over the same lecture could produce
different final videos depending on whether someone remembered to run it.

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
| `scs-video` | PSC | video: insightface, **onnxruntime-gpu**, onnx, opencv, sklearn |

Separate on purpose: `insightface` pulls plain `onnxruntime`, which conflicts
with `onnxruntime-gpu`. Locally that split is `requirements.txt` vs
`requirements-video.txt`. Never install both files into one venv — the CPU
build wins and inference silently drops ~25x.

Both files list **direct dependencies only**, not a `pip freeze`. For an exact
reproduction, `pip freeze > requirements.lock.txt` from a working env and
install from that.

`ffmpeg` is **not on PSC's default PATH**. `module load ffmpeg` gives 4.3.1 with
libx264/libx265, but that module is a **singularity wrapper**, not a binary —
`/opt/packages/ffmpeg/4.3.1/ffmpeg` shell-execs `singularity exec -B /ocean -B
/bil ...`, and the `/bil` bind mount fails on the GPU nodes. Anything that pipes
into ffmpeg dies there.

Both conda envs already carry a native `ffmpeg` (7.1.1, gpl build, libx264 and
libx265) at `$CONDA_PREFIX/bin/ffmpeg`. On compute nodes prefer that and do not
load the module — `source activate` already puts it first on PATH, and loading
the module afterwards is precisely what shadows it.

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

`track_instructor.py` inherits this exactly: it imports `build_app` from
`face_anon`, so it is the same detector, the same onnxruntime, the same crash.

**There is no NVENC path on PSC.** PSC's `ffmpeg/4.3.1` is built without nvenc
(hwaccels: vdpau/vaapi/vulkan only), and most Bridges-2 GPUs lack the silicon
anyway (V100/A100/H100 have none; only L40S does). All H.264 encoding is
`libx264` on CPU. Don't "fix" a slow encode by reaching for `h264_nvenc`.

**Never hardcode `camera.mp4` after sync — use `paths.resolve_camera()`.**
`sync.py` removes the black lead in front of the lecture, and when that trim is
needed it writes `camera_sync.mp4` and every later stage must read *that*.
Transcript, card and caption timings are all relative to whichever file was
actually cut to; mixing the two shifts every student question by the length of
the trim, which un-mutes students. The trim comes off the screen **and** the
camera together — cutting only the screen slides it out of sync by exactly the
amount removed.

On lecture 12 this changes nothing: the screen is black for its first 629.0s,
the duration alignment already removes 718.7s, so the residual is zero and
`camera_sync.mp4` is never written. The margin was 89.7 seconds. A lecture whose
screen starts only 3 minutes early but is dark for 10 would have published
seven minutes of black, and `verify.py` would have passed it — a black,
well-encoded segment satisfies every duration, size and bytes-per-second check.

**Measure black with ffmpeg at `-v info`, not `-v error`.** `blackdetect` logs
its findings at info level, so `-v error` reports a completely black file as
having no black at all. Scan far enough, too: `-t 400` on lecture 12 reports the
black ending at 400s because that is where the scan stopped, not where the black
did. `sync.py` warns when a run is still going at the edge of its window.

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

## Troubleshooting: symptom → cause

Look the error up here before investigating from scratch. Every row was hit for
real during development.

| Symptom | Cause | Fix |
|---|---|---|
| `err 209 no kernel image is available for execution on the device`, `CUDNN_FE failure 11`, dies ~3s into `Conv_0` | GPU is a V100 (sm_70); `onnxruntime-gpu` 1.28's cuDNN has no sm_70 kernels | Resubmit with `--gpus=l40s-48:8` or `h100-80:8`. Never v100 for face_anon. |
| Output video far shorter/smaller than its source (e.g. 1 MB against a 169 MB input) | Truncated or aborted encode. The stage still exits 0, so nothing notices | `python -m src.verify --lecture-dir <dir>`. Re-run the stage; do **not** let assembly consume it. |
| HTTP 403 fetching pyannote models | Gated model. 3.1 additionally pulls PLDA files from `community-1` | Accept conditions on **both** `pyannote/speaker-diarization-3.1` and `pyannote/speaker-diarization-community-1` |
| `refusing to assemble: ...camera_muted_anon.mp4 does not exist` | face_anon hasn't run, so faces are not anonymized | Run face_anon first. `--allow-unanonymized` only if that is genuinely intended. |
| `refusing to run ['cards'] on PSC` | cards is local-only: 20+ min, not for PSC | Run the pipeline on a laptop, or `--skip cards` on PSC and do it locally |
| `h264_nvenc` "listed by ffmpeg but fails to open an encode session" | GPU has no NVENC block (V100/A100/H100 have none) | Use `--encoder libx264`. There is no GPU encode path on PSC. |
| `ffmpeg: command not found` on PSC | Not on the default PATH | `module load ffmpeg` — but on GPU nodes prefer the conda env's ffmpeg, below |
| All chunk workers die with `BrokenPipeError: [Errno 32] Broken pipe`, *after* detection succeeded; log shows `container creation failed ... stat /bil: permission denied` | PSC's ffmpeg module is a singularity wrapper that bind-mounts `/bil`; the mount fails on GPU nodes, so ffmpeg never starts and the encode pipe breaks | Use `$CONDA_PREFIX/bin/ffmpeg` (7.1.1, gpl, libx264) instead of `module load ffmpeg`. `psc_face_anon.sbatch` now does this. |
| ssh: three instant `Permission denied` with **no** `password:` prompt | No TTY, so ssh could not ask. `ssh -M -N -f` also implies `-n` | Run `ssh psc` in a real terminal. Never `-f` with password auth. |
| ssh: password prompt appears but is rejected | PSC uses a **Kerberos** password, not CMU Andrew/SSO | Reset at <https://apr.psc.edu>, or `kpasswd` on-system (never `passwd`) |
| `psc.sh sync`: "no session to the DTN" | Transfers may not go through a login node | `ssh -N psc-dtn` once, in a real terminal; leave it running |
| DTN: `Login denied: Only file transfers are allowed on this account` | The DTN refuses interactive shells by design | Not a failure. Use `ssh -N`, and scp/sftp/rsync over it. |
| Inference silently runs on CPU and is ~25x slower | `onnxruntime` (CPU build) shadowing `onnxruntime-gpu` | Use the `scs-video` env. `pip uninstall onnxruntime && pip install onnxruntime-gpu` |
| `libcublasLt.so.12: cannot open shared object file`, then silent CPU fallback | onnxruntime-gpu's CUDA libs ship as `nvidia-*` pip packages under `site-packages/nvidia/*/lib`, which the loader does not search | Add those dirs to `LD_LIBRARY_PATH` (see `scripts/psc_face_anon.sbatch`) or `module load cuda/12.6.1` |
| `Disk quota exceeded` mid-job | `/jet/home` is 25 GB and typically ~90% full | Work under `/ocean/projects/cis260220p/$USER/`; set `HF_HOME` there too |
| Downstream stages use the wrong lecture's transcript | Legacy shared `data/transcription/` rather than per-lecture | Move the transcript into the lecture dir; `src/paths.py` warns when it falls back |
| Published video opens with minutes of black before the lecture starts | The screen capture's black lead outlasted the duration-alignment trim | `sync.py` now cuts the residual off both streams. Check its `trimming a further Ns of black` line; raise `--black-scan-seconds` if it warns the scan window ended mid-run. |
| Student questions are audible, or the instructor is muted, and everything is off by a constant | A stage read `camera.mp4` while the transcript was built against `camera_sync.mp4` (or vice versa) | Use `paths.resolve_camera()`. Delete a stale `camera_sync.mp4` if sync says it needed no trim. |
| Two runs over the same lecture produce different final videos; the PiP is zoomed in one and wide in the other | `assembly` prefers `camera_muted_anon_tracked.mp4` whenever it exists, so the result depended on whether `track_instructor` had been run | It is a pipeline stage now, so it runs by default. `assembly` logs `pip source=` on every run. Use `--skip track_instructor` (or `assembly --no-tracked`) to choose the wide shot deliberately. |
| Instructor is blurred and students are clear | Wrong cluster chosen as instructor | `face_anon --preview` → inspect `face_clusters.png` → `--instructor-cluster <id>` |
| cards takes tens of minutes and thrashes memory | Old ffmpeg `overlay` chain, O(questions × duration) | Should be cut-render-concat. Check `cards.py` has not been reverted. |

## Agent operating rules

1. **Never run compute on a PSC login node.** If unsure a command is "light",
   assume it is not.
2. **Never run `cards.py` on PSC.** 20+ minutes; laptop only.
3. **Never use `--gpus=v100-*` for `face_anon.py` or `track_instructor.py`.**
   Both use the same insightface detector; both will crash.
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

- **`face_anon.py` has never completed a full lecture *on PSC*.** Three attempts,
  three different failures, none of them the code:
  1. V100 — `err 209 no kernel image`, died in ~3s (job 42801726).
  2. H100 — `libcublasLt.so.12` missing, so onnxruntime fell back to
     `CPUExecutionProvider` *silently* and benchmarked at 0.77 fps
     detection-only, i.e. 81 min of inference even at chunks=8 (job 42801022).
     Fixed by the `LD_LIBRARY_PATH` block in the sbatch.
  3. H100 with CUDA genuinely active — all 29,832 detections completed in
     ~2 min across 8 workers, then every worker died with `BrokenPipeError`
     because the ffmpeg singularity container could not start (job 42802510,
     failed at 8:05). Fixed by using the conda ffmpeg.

  It has run end to end locally on lecture 12: `camera_muted_anon.mp4` is
  4772.9s against a 4773.7s source, and `track_instructor` produced a
  full-length crop from it. The detection path is now also proven on an H100 at
  ~250 detections/s aggregate. What remains unproven on PSC is the encode.
- **Question detection may be conservative.** On lecture 12, only 4 of 1,237
  segments were flagged `is_student_question` across 91 minutes. Spot-check
  before treating card coverage as complete — this is a privacy guarantee.
- **`captions.py --polish` is opt-in.** The ASR-mishear correction rewrites
  caption text, so review the diff before publishing.
- **`ingestion.py` needs a human.** CMU's SSO form was too brittle to script,
  so it opens a browser and waits for you to complete SSO + Duo.
