"""Per-lecture path resolution, shared by every stage.

Each stage takes --lecture-dir and derives its inputs and outputs from it, so the
pipeline runs on any lecture without editing source.

Transcripts live *inside* the lecture directory. They used to be written to a
single shared data/transcription/, which meant processing a second lecture
silently overwrote the first one's transcript and every downstream stage then
built the wrong video. For lectures processed before that change, resolution
falls back to the old shared location (read-only) so existing work still runs.

    from src.paths import LecturePaths, add_lecture_args
    p = LecturePaths(args.lecture_dir)
    p.camera            -> data/15210-lecture12/camera.mp4
    p.transcript_classified
"""

import argparse
import os

LEGACY_TRANSCRIPT_DIR = os.path.join("data", "transcription")


class LecturePaths:
    """Filenames for one lecture. Nothing here touches the filesystem except
    the two `resolve_*` helpers, which check for legacy locations."""

    def __init__(self, lecture_dir):
        self.dir = os.path.normpath(lecture_dir)

    def _p(self, name):
        return os.path.join(self.dir, name)

    # --- raw inputs, from panopto_download -------------------------------
    @property
    def camera(self):
        return self._p("camera.mp4")

    @property
    def screen(self):
        return self._p("screen.mp4")

    @property
    def metadata(self):
        return self._p("metadata.json")

    # --- stage 3: sync ---------------------------------------------------
    @property
    def screen_sync(self):
        return self._p("screen_sync.mp4")

    # --- stage 4: transcription -----------------------------------------
    @property
    def camera_wav(self):
        return self._p("camera.wav")

    @property
    def transcript(self):
        return self._p("transcript.json")

    @property
    def transcript_classified(self):
        return self._p("transcript_classified.json")

    # --- stage 5: audio muting ------------------------------------------
    @property
    def camera_muted_wav(self):
        return self._p("camera_muted.wav")

    @property
    def camera_muted(self):
        return self._p("camera_muted.mp4")

    # --- stage 6: cards --------------------------------------------------
    @property
    def cards_dir(self):
        return self._p("cards")

    @property
    def screen_with_cards(self):
        return self._p("screen_with_cards.mp4")

    # --- stage 7: captions -----------------------------------------------
    @property
    def captions(self):
        return self._p("captions.srt")

    # --- stage 8: face anonymization -------------------------------------
    @property
    def camera_anon(self):
        """Anonymized camera. Named off camera_muted so face_anon's own
        <input>_anon.mp4 convention lines up."""
        return self._p("camera_muted_anon.mp4")

    @property
    def face_clusters_preview(self):
        return self._p("face_clusters.png")

    # --- stage 9: assembly -----------------------------------------------
    @property
    def key(self):
        return os.path.basename(self.dir.rstrip(os.sep)) or "lecture"

    @property
    def final(self):
        return self._p(f"{self.key}.mp4")

    # --- legacy fallbacks -------------------------------------------------
    def _resolve_legacy(self, preferred, legacy_name):
        if os.path.exists(preferred):
            return preferred
        legacy = os.path.join(LEGACY_TRANSCRIPT_DIR, legacy_name)
        if os.path.exists(legacy):
            print(f"[paths] using legacy {legacy} (per-lecture path "
                  f"{preferred} not found). Move it into the lecture dir to "
                  f"avoid collisions between lectures.")
            return legacy
        return preferred          # let the caller raise a clear FileNotFoundError

    def resolve_transcript(self):
        return self._resolve_legacy(self.transcript, "transcript.json")

    def resolve_transcript_classified(self):
        return self._resolve_legacy(self.transcript_classified,
                                    "transcript_classified.json")

    def resolve_camera_for_assembly(self):
        """Prefer the anonymized camera; fall back to merely muted.

        Assembly used to hardcode camera_muted.mp4, which meant a completed
        face-anonymization pass was silently discarded and the published video
        still showed every student's face.
        """
        if os.path.exists(self.camera_anon):
            return self.camera_anon
        return self.camera_muted


def add_lecture_args(parser, default="data/15210-lecture12"):
    parser.add_argument("--lecture-dir", default=default,
                        help="Directory holding one lecture's media "
                             f"(default: {default})")
    return parser


def lecture_parser(description, default="data/15210-lecture12"):
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_lecture_args(p, default)
    return p
