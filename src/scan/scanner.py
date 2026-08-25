"""Scan one lecture: run the tiers, cache the measurements, grade the result.

Tiers run cheapest-first and each one's raw measurements are cached in
`scan.json` inside the lecture directory. Deepening a scan therefore re-runs
only the tiers that are new, and a crash three hours into a semester loses the
lecture in flight rather than the afternoon. That matters more than it sounds:
the whole point of the cheap tiers is to run them over everything first, and
nobody does that twice if it costs an hour each time.

    from src.scan.scanner import scan_lecture
    result = scan_lecture("data/fall2026/15-210_abc", tier="signal")

Tier costs, measured on this corpus (79-90 minute lectures, M5 laptop):

    probe    ~1s      ffprobe and the metadata files
    signal   ~40s     one audio decode (21s) + two keyframe decodes (~8s each)
    vision   ~15s     insightface over ~200 sampled frames via CoreML
    speech   ~0.3s    reads transcript_classified.json if the lecture has one

The ordering is not arbitrary. Everything a later tier needs from an earlier
one is passed forward in memory -- the speech tier asks the signal tier's
frame-level analysis whether there was audio where a word is supposed to be --
so no file is ever decoded twice in a run.
"""

import datetime
import json
import os
import traceback

from src.scan import (audio_metrics, discover, face_metrics, media, rubric,
                      score, speech_metrics, video_metrics)

CACHE_NAME = "scan.json"

# Per-worker detector, built once per process and reused across lectures.
# insightface costs a few seconds to construct, which is real money when a pool
# worker handles thirty lectures.
_APP = None


def scan_one(args):
    """Process-pool entry point.

    Lives here rather than in __main__.py because the pool starts workers with
    `spawn`, and a spawned child re-imports the target by qualified name -- a
    function defined in a module that was run as `python -m src.scan` is not
    importable under that name in the child, which fails as an opaque
    BrokenProcessPool rather than anything that names the real problem.
    """
    global _APP
    lecture_dir, tier, force, vision_frames, verbose = args
    needs_vision = (rubric.TIERS.index(tier)
                    >= rubric.TIERS.index("vision"))
    if needs_vision and _APP is None:
        try:
            from src.video.face_anon import build_app
            _APP = build_app(det_size=640, need_recognition=True, quiet=True)
        except Exception:                                       # noqa: BLE001
            _APP = None         # scan_lecture records the failure per lecture
    return scan_lecture(lecture_dir, tier=tier, force=force, app=_APP,
                        vision_frames=vision_frames, verbose=verbose)


def _tier_index(tier):
    return rubric.TIERS.index(tier)


def _load_cache(lecture_dir):
    path = os.path.join(lecture_dir, CACHE_NAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _save_cache(lecture_dir, payload):
    path = os.path.join(lecture_dir, CACHE_NAME)
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except OSError:
        pass            # a read-only corpus is not a reason to fail the scan


def _est_pipeline_hours(cam, scr):
    """Rough wall-clock for a full local pipeline pass on this lecture.

    Anchored on the one lecture that has actually been through it end to end:
    15-210 lecture 12, 79.5 minutes, roughly two and a half hours locally with
    cards.py and the layout render dominating. Scaled by runtime and by pixel
    count, since both encodes are resolution-bound.
    """
    if not cam or not cam.get("duration"):
        return None
    hours = cam["duration"] / 4773.7 * 2.5
    px = (cam.get("width", 1280) * cam.get("height", 720)) / (1280 * 720)
    if scr:
        px = max(px, (scr.get("width", 1920) * scr.get("height", 1080))
                 / (1920 * 1080))
    return float(hours * max(px, 0.5))


def scan_lecture(lecture_dir, tier="speech", force=False, app=None,
                 vision_frames=face_metrics.SAMPLE_FRAMES, verbose=False):
    """Measure and grade one lecture. Never raises -- errors land in the result."""
    want = _tier_index(tier)
    warnings, errors = [], []
    cache = {} if force else _load_cache(lecture_dir)
    metrics = dict(cache.get("metrics") or {})
    done = set(cache.get("tiers_run") or [])
    ran = set(done)

    identity = discover.lecture_identity(lecture_dir)
    identity["dir"] = lecture_dir
    identity["scanned_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    streams = discover.resolve_streams(lecture_dir)
    warnings += streams.get("notes") or []

    cam_path, scr_path = streams.get("camera"), streams.get("screen")
    probe_info = {"camera": media.probe(cam_path) if cam_path else None,
                  "screen": media.probe(scr_path) if scr_path else None}
    cam, scr = probe_info["camera"], probe_info["screen"]

    def _note(msg):
        if verbose:
            print(f"[scan] {identity.get('key')}: {msg}", flush=True)

    # --- probe ----------------------------------------------------------
    if "probe" not in done:
        try:
            if cam:
                metrics["duration_s"] = cam.get("duration")
                metrics["duration_min"] = (cam.get("duration") or 0.0) / 60.0
                metrics["camera_height"] = cam.get("height")
            if scr:
                metrics["screen_height"] = scr.get("height")
                if not scr.get("height"):
                    pass
            if identity.get("chapter_count") is not None:
                metrics["chapter_count"] = identity["chapter_count"]
            metrics["est_pipeline_hours"] = _est_pipeline_hours(cam, scr)
            if cam is None:
                errors.append("camera stream missing or unreadable")
            if scr is None:
                warnings.append("no screen stream found; slide metrics skipped")
            ran.add("probe")
        except Exception as e:                                  # noqa: BLE001
            errors.append(f"probe: {e}")

    # --- signal ---------------------------------------------------------
    levels = None
    if want >= _tier_index("signal") and cam:
        if "signal" not in done or force:
            try:
                _note("audio pass")
                pcm, loudness = media.decode_audio(cam_path)
                am, levels = audio_metrics.measure(
                    pcm, loudness, cam.get("duration") or 0.0)
                metrics.update(am)
                del pcm
                _note("camera frames")
                metrics.update(video_metrics.measure_camera(
                    cam_path, cam.get("duration") or 0.0))
                if scr:
                    _note("screen frames")
                    metrics.update(video_metrics.measure_screen(
                        scr_path, scr.get("duration") or 0.0))
                    risk, margin = video_metrics.sync_risk(
                        cam.get("duration"), scr.get("duration"),
                        metrics.get("black_lead_s"))
                    metrics["sync_risk"] = risk
                    metrics["sync_margin_s"] = margin
                    if margin is not None and margin < 0:
                        warnings.append(
                            f"screen black lead exceeds the duration alignment by "
                            f"{-margin:.0f}s -- this lecture would publish with "
                            f"black at the front and verify.py would pass it")
                ran.add("signal")
            except Exception as e:                              # noqa: BLE001
                errors.append(f"signal: {e}")
                if verbose:
                    traceback.print_exc()

    # --- vision ---------------------------------------------------------
    if want >= _tier_index("vision") and cam:
        if "vision" not in done or force:
            try:
                _note("face detection")
                metrics.update(face_metrics.measure(
                    cam_path, cam.get("duration") or 0.0, app=app,
                    sample_frames=vision_frames))
                ran.add("vision")
            except Exception as e:                              # noqa: BLE001
                errors.append(f"vision: {e}")
                if verbose:
                    traceback.print_exc()

    # --- speech ---------------------------------------------------------
    if want >= _tier_index("speech"):
        if "speech" not in done or force:
            tpath = discover.transcript_path(lecture_dir)
            if not tpath:
                warnings.append(
                    "no transcript_classified.json; every speech-tier metric is "
                    "unmeasured (run transcription, or accept reduced coverage)")
            else:
                try:
                    _note("transcript")
                    segs = speech_metrics.load_transcript(tpath)
                    if segs:
                        metrics.update(speech_metrics.measure(
                            segs, (cam or {}).get("duration") or 0.0, levels))
                        ran.add("speech")
                    else:
                        errors.append("transcript present but unreadable/empty")
                except Exception as e:                          # noqa: BLE001
                    errors.append(f"speech: {e}")
                    if verbose:
                        traceback.print_exc()

    # `levels` holds the whole frame-level array; it must not reach the cache.
    result = score.evaluate(metrics, probe_info, identity=identity,
                            tiers_run=sorted(ran, key=_tier_index),
                            warnings=warnings, errors=errors)
    _save_cache(lecture_dir, {
        "schema": 1,
        "key": result.get("key"),
        "scanned_at": identity["scanned_at"],
        "tiers_run": result["tiers_run"],
        "metrics": metrics,
    })
    return result
