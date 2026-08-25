"""Corpus discovery: find every lecture under a root, and its two streams.

The pipeline stages all take one --lecture-dir and trust it. The scanner
grades a whole semester, so something has to turn "the folder I downloaded
everything into" into a list of lectures, and then work out which file in each
one is the camera and which is the screen. That is all this module does; every
measurement lives elsewhere.

Two directory layouts are in the wild and both have to work:

    flat    data/fall2026/15-210_<guid>/camera.mp4   (panopto_download.py)
    nested  data/fall2026/15-210/lecture01/camera.mp4

so the walk is recursive and a directory is judged on what it *directly*
contains. A directory that holds another lecture is a course folder, never a
lecture itself.

The one thing worth being careful about is which camera file gets graded. A
processed lecture directory holds a dozen mp4s -- camera_muted.mp4,
camera_muted_anon.mp4, screen_with_cards.mp4, <key>-layout.mp4, <key>.mp4 --
and every one of them is our own output. Grading those would measure the
pipeline, not the recording, and the whole point of a scan is to predict how
much work a *source* needs. `resolve_streams` only ever returns raw inputs.

    from src.scan.discover import (find_lectures, resolve_streams,
                                   lecture_identity, transcript_path)

    for d in find_lectures("data/fall2026"):
        s = resolve_streams(d)
        ident = lecture_identity(d)
        print(ident["course"], ident["title"], s["camera"], s["screen"])
        for note in s["notes"]:
            print("   guessed:", note)

Nothing here raises on a corrupt or unreadable file. A semester scan meets at
least one half-downloaded lecture and one truncated metadata.json, and the
right answer is a partial record with the gap visible, not a traceback that
takes the other forty lectures down with it.
"""

import json
import os

from src.paths import LEGACY_TRANSCRIPT_DIR
from src.scan.media import probe

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".m4v")

# Files that mark a directory as a lecture on their own, without having to
# look at extensions. panopto_download writes all three.
MARKER_FILES = ("camera.mp4", "screen.mp4", "metadata.json")

# Directories never descended into. `cards` holds per-question card PNGs and
# `transition-samples` holds a few seconds of rendered mp4 per transition
# style -- both sit *inside* a lecture directory and both would otherwise be
# discovered as lectures in their own right, which additionally demotes their
# real parent to a course folder. Pruning is what keeps that from happening,
# so these are skipped rather than filtered out afterwards.
SKIP_DIRS = ("__pycache__", "cards", "transition-samples")

# Screen captures are high-resolution and remarkably cheap: a slide deck is
# mostly unchanging flat colour, so x264 spends almost nothing on it. Measured
# on this corpus at 1920x1080: ~284 kb/s for the screen against ~659 kb/s for
# the camera. Bits per pixel separates the two even when the resolutions
# differ, which is why the fallback ranks on that rather than on bitrate.
SCREEN_BPP_HINT = "lower bits-per-pixel"


# --------------------------------------------------------------------------
# Finding lectures
# --------------------------------------------------------------------------

def _is_video(name):
    return name.lower().endswith(VIDEO_EXTS)


def _skip_dir(name):
    # Hidden directories cover .git, .venv, .DS_Store's neighbours and macOS
    # bundle cruft in one rule.
    return name.startswith(".") or name in SKIP_DIRS


def _looks_like_lecture(filenames):
    names = set(filenames)
    if any(m in names for m in MARKER_FILES):
        return True
    # Fallback for hand-assembled or renamed corpora: any video at all. Loose
    # on purpose -- a directory wrongly promoted here is caught by the
    # course-folder rule below, or shows up with empty identity fields, which
    # is a visible failure rather than a silently missing lecture.
    return any(_is_video(n) for n in names)


def find_lectures(root, recursive=True):
    """Every lecture directory under `root`, sorted.

    A directory qualifies if it DIRECTLY contains camera.mp4, screen.mp4,
    metadata.json, or any video file. `recursive=False` looks only at `root`
    and its immediate children, for when the caller already knows it is
    pointing at one semester's folder.
    """
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        return []

    candidates = []
    # onerror is left at the default (swallow): an unreadable subdirectory
    # should cost us that subtree, not the scan.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        depth = 0 if dirpath == root else \
            os.path.relpath(dirpath, root).count(os.sep) + 1
        if not recursive and depth >= 1:
            dirnames[:] = []
        if _looks_like_lecture(filenames):
            candidates.append(os.path.normpath(dirpath))

    # A directory containing a lecture is the course folder (or the corpus
    # root), even when it happens to hold a stray video of its own -- a
    # promotional trailer beside the lectures, say. Demote it rather than
    # emitting both it and the lectures inside it.
    inside = set()
    for cand in candidates:
        prefix = cand + os.sep
        if any(other.startswith(prefix) for other in candidates):
            inside.add(cand)
    return sorted(c for c in candidates if c not in inside)


# --------------------------------------------------------------------------
# Which file is the camera, which is the screen
# --------------------------------------------------------------------------

def _derived_prefixes(lecture_dir, meta):
    """Filename stems the pipeline writes its finished videos under.

    assembly names its output off the directory basename (paths.key), so
    <key>.mp4 and <key>-anything.mp4 are ours. The Panopto key from
    metadata.json is included too, because a directory renamed after download
    leaves outputs under the old stem.
    """
    stems = [os.path.basename(lecture_dir.rstrip(os.sep))]
    if isinstance(meta, dict) and isinstance(meta.get("key"), str):
        stems.append(meta["key"])
    return [s for s in stems if s]


def _is_derived(name, stems):
    """Is this mp4 something the pipeline made, rather than downloaded?

    This is the load-bearing rule in the module. The scanner exists to predict
    how much work a raw recording needs; handing it camera_muted_anon.mp4
    would have it measure faces we already pixelated and silence we already
    inserted, and report a pristine lecture. Every check below excludes a real
    pipeline output seen in data/: camera_sync, camera_muted,
    camera_muted_anon, camera_muted_anon_tracked, screen_sync,
    screen_with_cards, <key>-layout*, <key>-with-intro, <key>-camera, <key>.
    """
    base = os.path.basename(name)
    lower = base.lower()
    if lower in ("camera.mp4", "screen.mp4"):
        return False
    # Anything sharing the camera/screen stem with a suffix is a later stage's
    # rewrite of it -- sync trims it, audio mutes it, face_anon blurs it.
    stem = os.path.splitext(lower)[0]
    if stem.startswith("camera_") or stem.startswith("screen_"):
        return True
    for s in stems:
        s = s.lower()
        if stem == s or stem.startswith(s + "-"):
            return True
    # Belt and braces for corpora whose directory was renamed, so neither stem
    # above matches any more.
    return any(tag in stem for tag in
               ("_anon", "-layout", "-camera", "-with-intro", "_muted"))


def _bits_per_pixel(info):
    px = (info.get("width") or 0) * (info.get("height") or 0)
    if px <= 0 or not info.get("bit_rate"):
        return 0.0
    return info["bit_rate"] / float(px)


def _pixels(info):
    return (info.get("width") or 0) * (info.get("height") or 0)


def _classify(infos, notes):
    """Split probed videos into (camera, screen) by what they look like.

    The camera is the stream carrying audio: Panopto records the room mic on
    it, and the screen capture is silent. That single fact settles almost
    every real case, so it is tried first and on its own.

    When it does not settle it -- both silent, both with audio, or bitrates
    missing -- fall back to the shape of the encode. A slide capture is the
    higher resolution of the two and by far the cheaper per pixel, because
    flat unchanging colour compresses to nothing.
    """
    with_audio = [i for i in infos if i.get("has_audio")]
    silent = [i for i in infos if not i.get("has_audio")]

    if len(with_audio) == 1 and len(silent) >= 1:
        cam = with_audio[0]
        # More than one silent file: the screen is the cheapest per pixel.
        rest = sorted(silent, key=lambda i: (_bits_per_pixel(i),
                                             -_pixels(i)))
        notes.append(f"camera is {os.path.basename(cam['path'])} (the only "
                     f"stream with an audio track); screen is "
                     f"{os.path.basename(rest[0]['path'])}")
        return cam, rest[0]

    # No audio anywhere, or audio on everything: rank on encode shape alone.
    ranked = sorted(infos, key=lambda i: (_bits_per_pixel(i), -_pixels(i)))
    screen, camera = ranked[0], ranked[-1]
    notes.append(
        f"no audio track separated the streams, so classified on encode "
        f"shape ({SCREEN_BPP_HINT} is the screen): screen is "
        f"{os.path.basename(screen['path'])} at "
        f"{screen.get('width')}x{screen.get('height')} "
        f"{(screen.get('bit_rate') or 0) / 1000:.0f} kb/s, camera is "
        f"{os.path.basename(camera['path'])} at "
        f"{camera.get('width')}x{camera.get('height')} "
        f"{(camera.get('bit_rate') or 0) / 1000:.0f} kb/s")
    return camera, screen


def resolve_streams(lecture_dir):
    """{'camera', 'screen', 'notes'} for one lecture directory.

    camera/screen are paths, or None when there is nothing to point at.
    notes carries one line per guess, so a report can show which lectures were
    resolved by convention and which were reasoned about.
    """
    notes = []
    result = {"camera": None, "screen": None, "notes": notes}
    try:
        entries = sorted(os.listdir(lecture_dir))
    except OSError as e:
        notes.append(f"cannot list {lecture_dir}: {e}")
        return result

    names = set(entries)
    camera = os.path.join(lecture_dir, "camera.mp4")
    screen = os.path.join(lecture_dir, "screen.mp4")

    # The convention, and the only case with nothing to explain. Note that
    # camera_sync.mp4 is deliberately NOT preferred here even though
    # paths.resolve_camera() prefers it: that is the file the pipeline cut,
    # and the scanner is grading what arrived from Panopto.
    if "camera.mp4" in names and "screen.mp4" in names:
        result["camera"] = camera
        result["screen"] = screen
        return result

    meta = _load_json(os.path.join(lecture_dir, "metadata.json"))
    stems = _derived_prefixes(lecture_dir, meta)
    sources = [n for n in entries
               if _is_video(n) and not _is_derived(n, stems)
               and os.path.isfile(os.path.join(lecture_dir, n))]

    # One conventional name present, the other missing: keep the one we know
    # and look for its partner among whatever else is raw.
    if "camera.mp4" in names or "screen.mp4" in names:
        known = "camera.mp4" if "camera.mp4" in names else "screen.mp4"
        other = "screen" if known == "camera.mp4" else "camera"
        result["camera" if known == "camera.mp4" else "screen"] = \
            os.path.join(lecture_dir, known)
        rest = [n for n in sources if n != known]
        if len(rest) == 1:
            result[other] = os.path.join(lecture_dir, rest[0])
            notes.append(f"{known} found; took {rest[0]} as the {other} "
                         f"(only other raw video present)")
        elif rest:
            notes.append(f"{known} found but {len(rest)} candidates for the "
                         f"{other} ({', '.join(rest)}); left it unset")
        else:
            notes.append(f"{known} found, no {other} stream in this lecture")
        return result

    if not sources:
        notes.append("no raw video found (every video here looks like a "
                     "pipeline output)")
        return result

    if len(sources) == 1:
        # A single-stream lecture is normal -- a talk with no slide capture --
        # and it is the camera, because that is the one with a speaker in it.
        result["camera"] = os.path.join(lecture_dir, sources[0])
        notes.append(f"only one raw video ({sources[0]}); treated as the "
                     f"camera, no screen capture in this lecture")
        return result

    infos = []
    for name in sources:
        info = probe(os.path.join(lecture_dir, name))
        if info and info.get("has_video"):
            infos.append(info)
        else:
            notes.append(f"ignored {name}: unreadable or has no video stream")
    if not infos:
        notes.append("no probeable video in this lecture")
        return result
    if len(infos) == 1:
        result["camera"] = infos[0]["path"]
        notes.append(f"only {os.path.basename(infos[0]['path'])} probed "
                     f"cleanly; treated as the camera")
        return result

    if len(infos) > 2:
        listed = ", ".join(os.path.basename(i["path"]) for i in infos)
        notes.append(f"{len(infos)} raw videos present ({listed}); expected "
                     f"two, so the classification below picked from all of "
                     f"them")
    cam, scr = _classify(infos, notes)
    result["camera"] = cam["path"]
    result["screen"] = scr["path"]
    return result


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def _load_json(path):
    """Parsed JSON, or None. Never raises -- see the module docstring."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _chapter_count(lecture_dir):
    """Number of Panopto chapters, or None when there is no chapters.json.

    None and 0 are different facts and a report should be able to tell them
    apart: "we never downloaded the chapter list" is a gap in the corpus,
    while "this lecture has no chapters" is a property of the lecture and a
    reason it may need manual sectioning.
    """
    path = os.path.join(lecture_dir, "chapters.json")
    if not os.path.exists(path):
        return None
    data = _load_json(path)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        # Some exports wrap the list in an envelope; take the first list-
        # valued field rather than guessing at its name.
        for value in data.values():
            if isinstance(value, list):
                return len(value)
    return None


def lecture_identity(lecture_dir):
    """What this lecture calls itself, from metadata.json where possible.

    Every field may be None: a directory that was downloaded halfway has
    media and no metadata, and the scan should still rank it.
    """
    meta = _load_json(os.path.join(lecture_dir, "metadata.json"))
    if not isinstance(meta, dict):
        meta = {}

    duration = meta.get("durationSec")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None

    start = meta.get("start")
    if not isinstance(start, int):
        start = None

    return {
        # The directory basename, matching paths.LecturePaths.key, so the
        # scanner names a lecture the same way every other stage does.
        # metadata.json's own "key" is preserved below but deliberately not
        # used here: it is `<course>_<guid>`, which is unique and unreadable,
        # and a ranked table of forty of them is unusable.
        "key": os.path.basename(os.path.normpath(lecture_dir)),
        "panopto_key": meta.get("key"),
        "course": meta.get("course"),
        "title": meta.get("name"),
        "owner": meta.get("owner"),
        "duration_s": duration,
        "chapter_count": _chapter_count(lecture_dir),
        # Passed through as the raw integer on purpose. It is seconds since
        # 1601-01-01 (Windows FILETIME), which reads as the year 2395 if
        # treated as Unix time and 426 if treated as .NET ticks -- both wrong
        # silently. brand.term_from_panopto is this repo's only decoder and
        # the term printed on every published video comes off it, so there
        # must not be a second one.
        "panopto_start": start,
    }


def transcript_path(lecture_dir):
    """The classified transcript for this lecture, or None.

    Falls back to the legacy shared data/transcription/ for lectures
    processed before transcripts moved into the lecture directory (see
    src/paths.py -- the shared layout meant a second lecture overwrote the
    first one's transcript).
    """
    local = os.path.join(lecture_dir, "transcript_classified.json")
    if os.path.exists(local):
        return local
    legacy = os.path.join(LEGACY_TRANSCRIPT_DIR,
                          "transcript_classified.json")
    if os.path.exists(legacy):
        return legacy
    return None
