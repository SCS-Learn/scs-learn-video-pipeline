"""Run the anonymization pipeline end to end on one lecture.

    python -m src.pipeline --lecture-dir data/15210-lecture12
    python -m src.pipeline --lecture-dir data/... --from transcription --to cards
    python -m src.pipeline --lecture-dir data/... --dry-run

Stage order matters in two non-obvious ways:

  * face_anon must run AFTER audio (it consumes camera_muted.mp4, so the muted
    audio is carried through the remux) and BEFORE assembly (which composites
    the anonymized camera into the final video).
  * cards needs sync to have produced screen_sync.mp4 first.

Where each stage belongs is encoded in STAGES. `cards` is marked local_only:
it takes 20+ minutes and is not to be run on PSC. The runner refuses rather
than quietly doing the wrong thing; --force overrides.
"""

import argparse
import os
import socket
import subprocess
import sys
import time

from src.paths import LecturePaths

# name, module, where it runs, note
STAGES = [
    ("sync",          "src.sync",              "cpu",        "trim screen to camera"),
    ("transcription", "src.audio.transcription", "gpu",      "whisperx + diarization + question classification"),
    ("audio",         "src.audio.audio",       "cpu",        "mute non-instructor audio"),
    ("face_anon",     "src.video.face_anon",   "gpu_no_v100", "pixelate non-instructor faces"),
    ("cards",         "src.audio.cards",       "local_only", "burn question cards (20+ min; NOT on PSC)"),
    ("captions",      "src.audio.captions",    "cpu",        "write .srt"),
    ("assembly",      "src.assembly.assembly", "cpu",        "final picture-in-picture"),
]
STAGE_NAMES = [s[0] for s in STAGES]


def on_psc():
    """True when running on a PSC machine (login or compute)."""
    host = socket.getfqdn()
    return "bridges2" in host or "psc.edu" in host or os.path.isdir("/jet/home")


def run_stage(name, module, lecture_dir, extra=(), dry_run=False):
    cmd = [sys.executable, "-m", module, "--lecture-dir", lecture_dir, *extra]
    printable = " ".join(cmd)
    if dry_run:
        print(f"  [dry-run] {printable}")
        return 0.0
    print(f"\n{'=' * 70}\n[pipeline] {name}: {printable}\n{'=' * 70}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"[pipeline] {name} finished in {dt / 60:.1f} min", flush=True)
    return dt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lecture-dir", required=True)
    parser.add_argument("--from", dest="start", default=None, choices=STAGE_NAMES)
    parser.add_argument("--to", dest="end", default=None, choices=STAGE_NAMES)
    parser.add_argument("--only", default=None, choices=STAGE_NAMES)
    parser.add_argument("--skip", action="append", default=[], choices=STAGE_NAMES)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them")
    parser.add_argument("--force", action="store_true",
                        help="Run local_only stages even on PSC")
    args = parser.parse_args()

    if not os.path.isdir(args.lecture_dir):
        raise SystemExit(f"no such lecture dir: {args.lecture_dir}")
    p = LecturePaths(args.lecture_dir)

    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    else:
        i0 = STAGE_NAMES.index(args.start) if args.start else 0
        i1 = STAGE_NAMES.index(args.end) if args.end else len(STAGES) - 1
        if i1 < i0:
            raise SystemExit(f"--from {args.start} comes after --to {args.end}")
        selected = STAGES[i0:i1 + 1]
    selected = [s for s in selected if s[0] not in args.skip]

    here_is_psc = on_psc()
    print(f"[pipeline] lecture={p.dir} key={p.key}")
    print(f"[pipeline] running on {'PSC' if here_is_psc else 'this machine'} "
          f"({socket.getfqdn()})")
    print("[pipeline] stages:")
    for name, _, where, note in selected:
        print(f"    {name:14s} [{where:12s}] {note}")

    blocked = [s[0] for s in selected if s[2] == "local_only" and here_is_psc]
    if blocked and not args.force:
        raise SystemExit(
            f"\nrefusing to run {blocked} on PSC.\n"
            f"These stages are marked local_only -- cards takes 20+ minutes and\n"
            f"is not to be run on PSC. Either run this pipeline on your laptop,\n"
            f"or skip them here and do them locally:\n"
            f"    python -m src.pipeline --lecture-dir {p.dir} "
            f"{' '.join('--skip ' + b for b in blocked)}\n"
            f"Use --force only if you genuinely mean to override this.")

    total = 0.0
    for name, module, where, _ in selected:
        extra = []
        if name == "face_anon" and where == "gpu_no_v100":
            # Guard rail rather than a silent crash: onnxruntime-gpu has no
            # sm_70 kernels, so this stage dies ~3s into a V100 job.
            gpu = os.environ.get("SLURM_JOB_GPUS", "") + os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if here_is_psc and "v100" in os.environ.get("SLURM_JOB_PARTITION", "").lower():
                print("[pipeline] WARNING: face_anon cannot run on V100 (sm_70 "
                      "unsupported by onnxruntime-gpu). Use l40s-48 or h100-80.")
        total += run_stage(name, module, args.lecture_dir, extra, args.dry_run)

    if not args.dry_run:
        print(f"\n[pipeline] all stages done in {total / 60:.1f} min")
        if os.path.exists(p.final):
            size = os.path.getsize(p.final) / 1e6
            print(f"[pipeline] final video: {p.final} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
