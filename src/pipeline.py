"""Run the anonymization pipeline end to end on one lecture.

    python -m src.pipeline --lecture-dir data/15210-lecture12
    python -m src.pipeline --lecture-dir data/... --from transcription --to cards
    python -m src.pipeline --lecture-dir data/... --dry-run

Stage order matters in two non-obvious ways:

  * face_anon must run AFTER audio (it consumes camera_muted.mp4, so the muted
    audio is carried through the remux) and BEFORE assembly (which composites
    the anonymized camera into the final video).
  * track_instructor must run AFTER face_anon: it crops in on the instructor,
    and zooming an un-anonymized camera makes any student in frame MORE
    identifiable, not less.
  * cards needs sync to have produced screen_sync.mp4 first.
  * layout must run AFTER cards and scenes. It draws the slide window from
    screen_with_cards.mp4, so running it first drops every question card -- the
    privacy substitute for a student's muted question -- out of the picture.
    scenes, in turn, must read screen_sync.mp4 rather than the carded file: a
    card is a static full-frame image and would otherwise read as a frozen
    slide and cut away from itself.
  * assembly no longer composites anything. layout decides the picture from
    assets/brand/plates; assembly stream-copies it and adds the sound. It
    refuses to run if the layout render is missing rather than falling back to
    the pre-brand corner picture-in-picture, which would publish a
    different-looking video under the same filename.

track_instructor is on the stage list but does not run by default -- nothing in
the current picture consumes its output. See STAGES for why. Run it on its own
with --only track_instructor if you need the pre-brand corner
picture-in-picture.

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
from src.verify import VerificationError, verify_stage

# name, module, where it runs, note, and whether it runs by default.
#
# track_instructor is OFF by default. It used to be on because assembly
# preferred its output whenever the file existed, so leaving it out made two
# runs over the same lecture produce different videos. That is no longer true:
# the picture now comes from src/assembly/layout.py, which does its own motion
# tracking on the CPU and reads the anonymized camera directly -- it never
# opens the tracked crop. Running it anyway would spend GPU hours out of a
# ~495-hour grant producing a file only `assembly --legacy-pip` reads. It stays
# selectable with `--only track_instructor` for exactly that case.
STAGES = [
    ("sync",          "src.sync",              "cpu",        "trim screen to camera", True),
    ("transcription", "src.audio.transcription", "gpu",      "whisperx + diarization + question classification", True),
    ("audio",         "src.audio.audio",       "cpu",        "mute non-instructor audio", True),
    ("face_anon",     "src.video.face_anon",   "gpu_no_v100", "pixelate non-instructor faces", True),
    ("track_instructor", "src.video.track_instructor", "gpu_no_v100", "crop the camera to follow the instructor (only --legacy-pip uses it)", False),
    ("cards",         "src.audio.cards",       "local_only", "burn question cards (20+ min; NOT on PSC)", True),
    ("scenes",        "src.video.scenes",      "cpu",        "decide Scene A vs Scene B per interval", True),
    ("layout",        "src.assembly.layout",   "cpu",        "composite the SCS brand scenes (the picture)", True),
    ("captions",      "src.audio.captions",    "cpu",        "write .srt", True),
    ("assembly",      "src.assembly.assembly", "cpu",        "finish: card sting, faststart, camera-only cut", True),
]
STAGE_NAMES = [s[0] for s in STAGES]
DEFAULT_OFF = [s[0] for s in STAGES if not s[4]]


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
    parser.add_argument("--with", dest="with_stage", action="append", default=[],
                        choices=DEFAULT_OFF,
                        help=f"Also run a stage that is off by default "
                             f"({', '.join(DEFAULT_OFF)})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them")
    parser.add_argument("--force", action="store_true",
                        help="Run local_only stages even on PSC")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the output check after each stage. A stage "
                             "can exit 0 having written a truncated file, so "
                             "this is on by default.")
    args = parser.parse_args()

    if not os.path.isdir(args.lecture_dir):
        raise SystemExit(f"no such lecture dir: {args.lecture_dir}")
    p = LecturePaths(args.lecture_dir)

    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    elif args.with_stage:
        i0 = STAGE_NAMES.index(args.start) if args.start else 0
        i1 = STAGE_NAMES.index(args.end) if args.end else len(STAGES) - 1
        selected = [s for s in STAGES[i0:i1 + 1]
                    if s[4] or s[0] in args.with_stage]
    else:
        i0 = STAGE_NAMES.index(args.start) if args.start else 0
        i1 = STAGE_NAMES.index(args.end) if args.end else len(STAGES) - 1
        if i1 < i0:
            raise SystemExit(f"--from {args.start} comes after --to {args.end}")
        selected = [s for s in STAGES[i0:i1 + 1] if s[4]]
    selected = [s for s in selected if s[0] not in args.skip]

    here_is_psc = on_psc()
    print(f"[pipeline] lecture={p.dir} key={p.key}")
    print(f"[pipeline] running on {'PSC' if here_is_psc else 'this machine'} "
          f"({socket.getfqdn()})")
    print("[pipeline] stages:")
    for name, _, where, note, _on in selected:
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
    for name, module, where, _note, _on in selected:
        extra = []
        if where == "gpu_no_v100":
            # Guard rail rather than a silent crash: onnxruntime-gpu has no
            # sm_70 kernels, so these stages die ~3s into a V100 job. Applies to
            # track_instructor too -- it imports face_anon's detector.
            if here_is_psc and "v100" in os.environ.get("SLURM_JOB_PARTITION", "").lower():
                print(f"[pipeline] WARNING: {name} cannot run on V100 (sm_70 "
                      f"unsupported by onnxruntime-gpu). Use l40s-48 or h100-80.")
        total += run_stage(name, module, args.lecture_dir, extra, args.dry_run)
        if not args.dry_run and not args.no_verify:
            try:
                verify_stage(name, p)
            except VerificationError as e:
                raise SystemExit(
                    f"\n{e}\n\n[pipeline] stopping after '{name}'. Later "
                    f"stages would consume this output and produce a broken "
                    f"result, which is worse than failing here.")

    if not args.dry_run:
        print(f"\n[pipeline] all stages done in {total / 60:.1f} min")
        if os.path.exists(p.final):
            size = os.path.getsize(p.final) / 1e6
            print(f"[pipeline] final video: {p.final} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
