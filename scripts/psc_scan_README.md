# Scanning a semester on Bridges-2

Operator instructions for `scripts/psc_scan.sbatch` — download ~45 lectures
from a harvested manifest onto `/ocean` and grade them with `src/scan`, so the
media never comes down a home connection.

**The media stays on `/ocean` and is never transferred to the laptop.** The
only things worth pulling back are `scan.json` (a few KB per lecture) and the
`scan-report.*` files. A lecture is hundreds of megabytes; the whole point of
running this on PSC is that they stay there.

---

## 0. One-time: the two SSH sessions

Both must be typed by a human in a real terminal — they need a TTY for the
password prompt (a Kerberos password, not CMU SSO).

```bash
./scripts/psc.sh install-config     # writes the ~/.ssh/config stanzas, once
ssh psc                             # log in, then `exit`
ssh -N psc-dtn                      # transfers; leave it running
./scripts/psc.sh status             # confirm the session is alive
```

`ssh -N psc-dtn` sits there with no prompt — that is correct. The DTN refuses
interactive shells (`Login denied: Only file transfers are allowed on this
account`), which looks like a failure and is not.

---

## 1. Push the code

```bash
./scripts/psc.sh sync
```

## 2. Push the manifest — separately

`psc.sh sync` excludes `manifest*.json` on purpose, so the manifest does **not**
travel with the code. Copy it over the DTN yourself:

```bash
ssh -N psc-dtn                      # if not already up
PSC_USER=<your psc username>
ssh psc-dtn 'true' 2>/dev/null      # (the DTN refuses shells; use scp/rsync)

scp manifest.15210.json \
    psc-dtn:/ocean/projects/cis260220p/$PSC_USER/manifests/
```

Create the directory first if it does not exist:

```bash
./scripts/psc.sh run 'mkdir -p /ocean/projects/cis260220p/$USER/manifests'
```

(`psc.sh run` is for light management only — `mkdir`, `ls`, `squeue`. Never
ffmpeg, never a broad `find`, never anything that computes.)

---

## 3. Submit

Start small. A three-lecture run proves the manifest, the node's outbound
network and the ffmpeg before committing six hours to forty-five lectures.

```bash
./scripts/psc.sh sbatch scripts/psc_scan.sbatch \
  '--export=ALL,MANIFEST=/ocean/projects/cis260220p/$USER/manifests/manifest.15210.json,CORPUS_DIR=/ocean/projects/cis260220p/$USER/corpus/15-210,TIER=signal,LIMIT=3'
```

**Keep the single quotes.** `psc.sh sbatch` passes its arguments through to the
remote shell unexpanded, so `$USER` resolves to your *PSC* username there.
Without the quotes your laptop expands it first and the job writes to
`/ocean/projects/cis260220p/<your-mac-username>/`, which you do not own.

Then the full sweep — drop `LIMIT`:

```bash
./scripts/psc.sh sbatch scripts/psc_scan.sbatch \
  '--export=ALL,MANIFEST=/ocean/projects/cis260220p/$USER/manifests/manifest.15210.json,CORPUS_DIR=/ocean/projects/cis260220p/$USER/corpus/15-210,TIER=signal'
```

The second run re-uses everything the first fetched: already-downloaded
lectures are skipped, and the scan tiers are cached in each lecture's
`scan.json`.

### Environment variables

| Var | Default | Notes |
|---|---|---|
| `MANIFEST` | — | **Required**, absolute. Not synced with the code (see step 2). |
| `CORPUS_DIR` | — | **Required**, absolute, must be under `/ocean`. The job refuses anything else — `/jet/home` is 25 GB and ~90% full. |
| `TIER` | `signal` | `probe` \| `signal` \| `vision` \| `speech` \| `judge`. Cumulative and cached. |
| `JOBS` | `6` | Lectures scanned in parallel. ffmpeg is already threaded; more is often slower. |
| `DL_JOBS` | `4` | Concurrent downloads. Network-bound. |
| `LIMIT` | all | Max lectures to fetch. Use it on a first run. |
| `REPORT_DIR` | `$CORPUS_DIR/_reports` | Where `scan-report.{md,csv,html}` and `scan-results.json` land. |

### Going deeper on the survivors

Read the ranking, then re-run at a deeper tier. Nothing is re-downloaded and
nothing already measured is re-measured:

```bash
./scripts/psc.sh sbatch scripts/psc_scan.sbatch \
  '--export=ALL,MANIFEST=...,CORPUS_DIR=...,TIER=speech'
```

`speech` reads an existing `transcript_classified.json` per lecture, so it only
says anything for lectures that have been through transcription.

---

## 4. Watch it

```bash
./scripts/psc.sh sq                          # your queue
./scripts/psc.sh run 'tail -40 /ocean/projects/cis260220p/$USER/scs-learn-video-pipeline/logs/scan-<jobid>.out'
./scripts/psc.sh run 'sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS'
```

The log path follows `PSC_REMOTE_REPO` from your `.env` — the job writes to
`logs/scan-%j.out` inside the repo checkout. The first block of the `.out` file
echoes the resolved node, partition, python, ffmpeg path, manifest, tier and
disk free, so it is self-explaining when something looks wrong later.

If the job sits `PENDING`, `RM-shared` is simply busy — unlike the GPU
partition there is no `--gpus` count to get wrong here.

---

## 5. Pull back the results only

The reports (a few hundred KB at most):

```bash
PSC_USER=<your psc username>
mkdir -p reports/15-210
rsync -avz \
  psc-dtn:/ocean/projects/cis260220p/$PSC_USER/corpus/15-210/_reports/ \
  reports/15-210/
```

And, if you want the per-lecture caches too — `--include`/`--exclude` ordering
matters, the `--exclude '*'` at the end is what stops the mp4s coming with it:

```bash
rsync -avz \
  --include '*/' --include 'scan.json' --include 'metadata.json' --exclude '*' \
  psc-dtn:/ocean/projects/cis260220p/$PSC_USER/corpus/15-210/ \
  reports/15-210-cache/
```

Sanity-check what you are about to move before you move it:

```bash
rsync -avzn --stats \
  --include '*/' --include 'scan.json' --exclude '*' \
  psc-dtn:/ocean/projects/cis260220p/$PSC_USER/corpus/15-210/ /tmp/check/
```

`-n` is a dry run. If that prints hundreds of megabytes, an `--include` is
wrong and you are about to pull video.

To read the ranking without transferring anything at all:

```bash
./scripts/psc.sh run 'cat /ocean/projects/cis260220p/$USER/corpus/15-210/_reports/scan-report.md'
```

---

## Housekeeping

Check the allocation now and then — RM-shared bills core-hours against the
5,000 SU Regular grant, which this job spends slowly, but nothing else does:

```bash
./scripts/psc.sh quota
./scripts/psc.sh run 'du -sh /ocean/projects/cis260220p/$USER/corpus/*'
```

When the scan has told you which lectures are worth publishing, the rest of the
corpus can go:

```bash
./scripts/psc.sh run 'rm -rf /ocean/projects/cis260220p/$USER/corpus/15-210/<key>'
```

Close the shared session when you are done: `./scripts/psc.sh logout`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: no manifest at ...` | `psc.sh sync` excludes `manifest*.json`, so it never arrived | scp it over the DTN — step 2 |
| `ERROR: CORPUS_DIR must live under /ocean` | Pointed at `/jet/home` or the repo | `/jet/home` is 25 GB and near full. Use `/ocean/projects/cis260220p/$USER/...` |
| `ERROR: ffmpeg is not usable on this node`, mentioning `stat for /bil: permission denied` | PSC's ffmpeg module is a singularity wrapper whose `/bil` bind mount fails on compute nodes | Use the `scs-learn` conda env's own ffmpeg 7.1.1 at `$CONDA_PREFIX/bin/ffmpeg`. Do **not** `module load ffmpeg` |
| `[fetch] ... FAILED rc=...` on a few streams | A dead Panopto session, or a CloudFront hiccup | Re-submit. Everything already on disk is skipped; only the missing streams retry |
| Job exits 1 at the fetch step | *Every* lecture failed — usually a stale manifest whose URLs have expired | Re-harvest the manifest with `src/ingestion.py` (needs a browser, so: on your laptop) |
| `Disk quota exceeded` | Something wrote to `/jet/home` | The job pins `HF_HOME`, `XDG_CACHE_HOME`, `TORCH_HOME`, `INSIGHTFACE_HOME`, `MPLCONFIGDIR` and `TMPDIR` under `/ocean`. Check nothing new bypasses that |
| `psc.sh sync`: "no session to the DTN" | Transfers may not go through a login node | `ssh -N psc-dtn` once, in a real terminal, and leave it running |
| Vision tier records a per-lecture error about insightface | `scs-learn` has no insightface; `scs-video` does, but its `onnxruntime-gpu` cannot share an env with this one | Run the vision tier locally, or in `scs-video`. Do not `pip install insightface` into `scs-learn` — the CPU onnxruntime it pulls would shadow `onnxruntime-gpu` for `face_anon` |
