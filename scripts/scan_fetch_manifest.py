"""Fetch the lectures in a Panopto manifest into a corpus directory.

The scanner (`python -m src.scan --courses-dir ...`) grades a whole semester,
but it can only grade what is on disk, and a 45-lecture manifest is tens of
gigabytes that nobody wants to pull down a home hotspot. This is the fetch half
of `scripts/psc_scan.sbatch`: it runs on a Bridges-2 compute node, writes into
/ocean, and hands the scanner a corpus.

    python scripts/scan_fetch_manifest.py \\
        --manifest manifest.15210.json \\
        --out-dir /ocean/projects/cis260220p/$USER/corpus/15-210 \\
        --jobs 4 --limit 3

It reuses `src.panopto_download` rather than restating it -- `download_stream`,
`write_lecture_sidecars`, `probe_resolution` and `file_size_mb` are already the
tested versions of exactly this, including the direct-mp4-vs-HLS split. What is
NOT reused is `ingest_lecture`, and deliberately: that function re-downloads
every stream unconditionally, has no notion of an already-complete lecture, and
lets an exception out to kill the run. All three are wrong for a batch of
forty-five over a six-hour walltime, so the per-lecture loop is re-stated here
around the same primitives.

No authentication is involved. The CloudFront media URLs in a manifest are open
once harvested in the browser (see src/panopto_download.py's docstring), which
is why this can run unattended on a compute node with no credential on it.

Three properties matter more than speed:

  * Skip-existing is the default. A lecture whose camera.mp4 and screen.mp4 are
    both present and non-trivial is left alone. Re-running the scan at a deeper
    tier must not re-fetch 40 GB.
  * One lecture failing must not take the other forty-four down. Failures are
    collected and printed at the end; the exit code is non-zero only if every
    attempted lecture failed, because a batch that lost one bad recording still
    produced a corpus worth scanning.
  * Partial writes are never left looking complete. Each stream lands on a
    .part file and is renamed only once it is on disk at a plausible size --
    a truncated mp4 that survives into the corpus is a lecture the scanner
    grades and mis-grades.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.panopto_download import (  # noqa: E402
    download_stream,
    file_size_mb,
    probe_resolution,
    write_lecture_sidecars,
)

# Below this a file is a stub, not a lecture: an HTML error page saved as
# .mp4, a curl that died in its first second, an aborted HLS remux. Panopto
# camera streams on this corpus run to hundreds of MB, so 2 MB is far under
# any real recording while still catching every failure mode we have seen.
MIN_STREAM_BYTES = 2 * 1024 * 1024

_print_lock = threading.Lock()
_total_bytes = 0
_bytes_lock = threading.Lock()


def log(msg):
    """Serialised print. Concurrent downloads interleave otherwise."""
    with _print_lock:
        print(msg, flush=True)


def _add_bytes(n):
    global _total_bytes
    with _bytes_lock:
        _total_bytes += n
        return _total_bytes


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}TB"


def stream_is_complete(path):
    """Present, and big enough to be a real recording rather than a stub."""
    try:
        return os.path.getsize(path) >= MIN_STREAM_BYTES
    except OSError:
        return False


def lecture_is_complete(lecture_dir, lecture):
    """True when every stream the manifest lists is already downloaded.

    Judged against the manifest's own stream list rather than a hardcoded
    {camera, screen}: a session with only one stream is complete with one file,
    and waiting for a screen.mp4 that was never published would re-fetch that
    lecture on every single run.
    """
    streams = lecture.get("streams") or []
    if not streams:
        return False
    return all(
        stream_is_complete(os.path.join(lecture_dir, s["type"] + ".mp4"))
        for s in streams
    )


def fetch_stream(stream, lecture_dir):
    """Download one stream. Returns (ok, bytes_written, note)."""
    stream_type = stream["type"]
    final = os.path.join(lecture_dir, stream_type + ".mp4")

    if stream_is_complete(final):
        size = os.path.getsize(final)
        return True, 0, f"{stream_type:6} skip (have {human(size)})"

    # Download to .part so an interrupted job -- walltime, node failure, a
    # cancelled scancel -- can never leave a half file that looks finished to
    # the next run's skip check.
    part = final + ".part"
    if os.path.exists(part):
        os.remove(part)

    started = time.time()
    rc = download_stream(stream["url"], stream["isHls"], part)
    elapsed = time.time() - started

    size = os.path.getsize(part) if os.path.exists(part) else 0
    if rc != 0 or size < MIN_STREAM_BYTES:
        if os.path.exists(part):
            os.remove(part)
        return False, 0, (f"{stream_type:6} FAILED rc={rc} got {human(size)} "
                          f"in {elapsed:.0f}s")

    os.replace(part, final)
    res = probe_resolution(final) or "unknown"
    rate = size / elapsed / 1024 / 1024 if elapsed > 0 else 0
    return True, size, (f"{stream_type:6} ok {file_size_mb(final)}MB {res} "
                        f"{elapsed:.0f}s ({rate:.1f}MB/s)")


def fetch_lecture(lecture, out_dir, skip_existing, index, total, t_start):
    """Download one lecture's streams and sidecars. Never raises."""
    key = lecture.get("key") or lecture.get("id") or "unknown"
    lecture_dir = os.path.join(out_dir, key)
    try:
        os.makedirs(lecture_dir, exist_ok=True)
        # Sidecars are rewritten every time: they are two small json files, and
        # metadata.json carries the Panopto start time the brand code decodes
        # the term from. Cheap to keep current, expensive to have stale.
        write_lecture_sidecars(lecture, lecture_dir)

        if skip_existing and lecture_is_complete(lecture_dir, lecture):
            log(f"[{index:>3}/{total}] {key}  complete, skipping")
            return key, True, []

        log(f"[{index:>3}/{total}] {key}  {lecture.get('name', '')}")
        failures = []
        for stream in lecture.get("streams", []):
            ok, nbytes, note = fetch_stream(stream, lecture_dir)
            if nbytes:
                running = _add_bytes(nbytes)
                note += f"   [total {human(running)} / {time.time() - t_start:.0f}s]"
            log(f"    {key} {note}")
            if not ok:
                failures.append(f"{key}:{stream['type']}")
        if not lecture.get("streams"):
            failures.append(f"{key}: manifest lists no streams")
        return key, not failures, failures
    except Exception as exc:                                    # noqa: BLE001
        # One bad lecture -- a malformed entry, an unwritable directory, a
        # network stack that gave up -- must cost that lecture and nothing
        # else. Forty-four good recordings are still worth scanning.
        log(f"    {key} ERROR {exc.__class__.__name__}: {exc}")
        return key, False, [f"{key}: {exc.__class__.__name__}: {exc}"]


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="scan_fetch_manifest.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="Harvested manifest json ({'lectures': [...]})")
    p.add_argument("--out-dir", required=True,
                   help="Corpus root. On PSC this must be under /ocean")
    p.add_argument("--limit", type=int, default=0,
                   help="Only fetch the first N lectures (0 = all). Use a small "
                        "value on a first run to prove the manifest cheaply")
    p.add_argument("--skip-existing", dest="skip_existing",
                   action="store_true", default=True,
                   help="Leave already-downloaded lectures alone (default)")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false",
                   help="Re-download everything. This is tens of GB; mean it")
    p.add_argument("--jobs", type=int, default=4,
                   help="Lectures fetched concurrently (default 4). These are "
                        "network-bound, so a few in flight helps and many do "
                        "not -- and HLS streams spend an ffmpeg each")
    args = p.parse_args(argv)

    with open(args.manifest) as f:
        manifest = json.load(f)
    lectures = manifest.get("lectures") or []
    if not lectures:
        print(f"no lectures in {args.manifest}", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        lectures = lectures[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    total = len(lectures)
    print(f"[fetch] {total} lecture(s) -> {args.out_dir}  "
          f"jobs={args.jobs} skip_existing={args.skip_existing}", flush=True)

    t_start = time.time()
    failures, attempted = [], 0
    ok_keys = []

    if args.jobs <= 1:
        for i, lec in enumerate(lectures, 1):
            key, ok, fails = fetch_lecture(lec, args.out_dir,
                                           args.skip_existing, i, total, t_start)
            attempted += 1
            (ok_keys.append(key) if ok else failures.extend(fails))
    else:
        # Threads, not processes: every one of these is blocked on a socket
        # inside curl or ffmpeg, so the GIL is irrelevant and a thread pool
        # keeps the shared byte counter and the print lock trivial.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(fetch_lecture, lec, args.out_dir,
                            args.skip_existing, i, total, t_start): lec
                for i, lec in enumerate(lectures, 1)
            }
            for fut in as_completed(futs):
                key, ok, fails = fut.result()
                attempted += 1
                (ok_keys.append(key) if ok else failures.extend(fails))

    elapsed = time.time() - t_start
    print()
    print("=" * 62)
    print(f"[fetch] {len(ok_keys)}/{total} lecture(s) complete, "
          f"{human(_total_bytes)} fetched in {elapsed:.0f}s "
          f"({human(_total_bytes / elapsed) if elapsed else '0B'}/s)")
    if failures:
        print(f"[fetch] {len(failures)} failure(s):")
        for f in failures:
            print(f"          {f}")
        print("[fetch] Re-run this command to retry only the missing streams; "
              "everything already on disk is skipped.")
    print("=" * 62)

    # Non-zero ONLY if nothing at all landed. A partial corpus is still worth
    # scanning, and failing the job here would take the scan step down with it
    # under `set -e` for the sake of one dead Panopto session.
    if attempted and not ok_keys:
        sys.stdout.flush()      # so the summary above lands before this does
        print("[fetch] every lecture failed -- check the manifest URLs and "
              "this node's outbound network", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
