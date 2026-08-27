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

    @property
    def camera_sync(self):
        """Camera with the pre-lecture dead air removed.

        Only written when the screen's black lead outlasts the duration-based
        alignment; otherwise the camera needs no trim and this does not exist.
        Both streams are cut by the same amount so they stay on one clock.
        """
        return self._p("camera_sync.mp4")

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

    @property
    def cards_manifest(self):
        """Where the cards actually landed, written by cards.py.

        Card spans are not the raw question intervals -- overlapping cards get
        merged during planning -- so assembly reads timings from here rather
        than recomputing them from the transcript and drifting.
        """
        return self._p("cards.json")

    @property
    def card_sound(self):
        """Sting played over every question card.

        Per theme: the file lives inside each theme directory rather than
        beside them, so the lookup follows the theme rather than assuming one
        shared asset.

        The last resort is ANY theme that has the file, and it is not
        decoration. assembly's mix_card_sound returns the video unchanged when
        this path does not exist -- no error, no warning -- so a theme
        directory that has been emptied means every question card plays over
        the silence the audio pass left, and nothing says so. That is exactly
        what happened when assets/themes/professional was cleared while
        CARD_THEME still defaulted to it.
        """
        assets = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "assets", "themes")
        theme = os.environ.get("CARD_THEME", "professional")
        preferred = os.path.join(assets, theme, "question-card-sound.mp3")
        for cand in (preferred,
                     os.path.join(assets, "professional", "question-card-sound.mp3"),
                     os.path.join(assets, "question-card-sound.mp3")):
            if os.path.exists(cand):
                return cand
        for other in sorted(os.listdir(assets)) if os.path.isdir(assets) else []:
            cand = os.path.join(assets, other, "question-card-sound.mp3")
            if os.path.exists(cand):
                print(f"[paths] no question-card-sound.mp3 under theme "
                      f"{theme!r}; using {other!r}'s copy. Set CARD_THEME, or "
                      f"put the sting back, if that is not what you want.")
                return cand
        return preferred

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

    # --- stage 8b: instructor tracking (optional) ------------------------
    @property
    def camera_tracked(self):
        """Zoomed crop of the anonymized camera that follows the instructor."""
        return self._p("camera_muted_anon_tracked.mp4")

    @property
    def instructor_track(self):
        """Per-frame instructor boxes, cached by the layout renderer.

        Detection is the expensive half of following him and the framing
        derived from it is nearly free, so re-deciding how the camera moves
        should not mean re-detecting where he is. Same bargain as
        track_instructor's --save-path.
        """
        return self._p("instructor_track.json")

    # --- stage 8c: scene decisions ---------------------------------------
    @property
    def scenes(self):
        """Cut list: which layout the frame is in over each interval.

        Written by src/video/scenes.py, consumed by assembly. Kept as a file
        rather than recomputed inside assembly so the cuts can be reviewed, and
        hand-edited, before committing to a full-length encode.
        """
        return self._p("scenes.json")

    # --- stage 8d: the brand layout render --------------------------------
    @property
    def layout(self):
        """The lecture composited into the two brand scenes.

        This is the picture the published video is made of. assembly.py takes
        it as its video and adds the finishing -- the card sting, the
        camera-only cut -- rather than re-compositing, so the plate geometry
        lives in exactly one place.
        """
        return self._p(f"{self.key}-layout.mp4")

    # --- stage 9: assembly -----------------------------------------------
    @property
    def key(self):
        return os.path.basename(self.dir.rstrip(os.sep)) or "lecture"

    @property
    def final(self):
        return self._p(f"{self.key}.mp4")

    @property
    def final_camera_only(self):
        """Camera-only deliverable: everything the pipeline does to the camera
        (student audio muted, faces anonymized) with no screen composited in."""
        return self._p(f"{self.key}-camera.mp4")

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

    def resolve_camera(self):
        """The camera every stage after sync should start from.

        camera.mp4 is the raw download; camera_sync.mp4 is that same file with
        the pre-lecture black removed. Stages must not hardcode camera.mp4 --
        transcript, card and caption timings are all relative to whichever of
        these the pipeline actually cut to, and mixing the two shifts every
        student question by the length of the trim.
        """
        if os.path.exists(self.camera_sync):
            return self.camera_sync
        return self.camera

    def resolve_pip_camera(self):
        """Best available picture-in-picture source, most-processed first.

        The tracked crop is preferred when present: at 480px wide the instructor
        is only a few dozen pixels tall in the raw wide shot.
        """
        for cand in (self.camera_tracked, self.camera_anon, self.camera_muted):
            if os.path.exists(cand):
                return cand
        return self.camera_muted

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
