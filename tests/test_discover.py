"""src/scan/discover.py -- finding lectures and picking the right two files.

The load-bearing rule is which camera file gets graded. A processed lecture
directory holds a dozen mp4s and every one of them is our own output; grading
camera_muted_anon.mp4 would measure faces we already pixelated and silence we
already inserted, and report a pristine lecture. The scanner exists to predict
how much work a *source* needs, so `resolve_streams` must only ever return raw
inputs -- and a semester-wide regression there would invalidate every number in
the report without anything looking wrong.

Every tree here is built under tempfile. The video files are stub bytes: the
paths exercised below are the ones that never shell out to ffprobe, which is
what keeps these tests fast and hermetic.
"""

import json
import os
import shutil
import tempfile
import unittest

from src.scan import discover


class TempTree(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="scan-discover-test-")
        self.addCleanup(shutil.rmtree, self.root, True)

    def touch(self, *parts, data=b"\0" * 32):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def rel(self, paths):
        return sorted(os.path.relpath(p, self.root) for p in paths)


class TestFindLectures(TempTree):

    def test_a_directory_with_camera_and_screen_is_a_lecture(self):
        """The panopto_download.py layout, and the only case with nothing to
        work out."""
        self.touch("lec1", "camera.mp4")
        self.touch("lec1", "screen.mp4")
        self.assertEqual(self.rel(discover.find_lectures(self.root)), ["lec1"])

    def test_metadata_json_alone_marks_a_lecture(self):
        """A download that got the metadata and died before the media is still
        a lecture the report has to account for."""
        self.touch("lec1", "metadata.json", data=b"{}")
        self.assertEqual(self.rel(discover.find_lectures(self.root)), ["lec1"])

    def test_any_video_file_marks_a_hand_assembled_lecture(self):
        """Both directory layouts are in the wild; renamed corpora have to work."""
        self.touch("lec1", "recording.mkv")
        self.assertEqual(self.rel(discover.find_lectures(self.root)), ["lec1"])

    def test_a_parent_holding_lecture_subdirs_is_not_itself_returned(self):
        """A directory containing a lecture is a course folder. Emitting both
        it and the lectures inside it would double-count a whole semester."""
        self.touch("15-210", "lecture01", "camera.mp4")
        self.touch("15-210", "lecture02", "camera.mp4")
        found = self.rel(discover.find_lectures(self.root))
        self.assertEqual(found, [os.path.join("15-210", "lecture01"),
                                 os.path.join("15-210", "lecture02")])
        self.assertNotIn("15-210", found)

    def test_a_course_folder_with_a_stray_video_is_still_demoted(self):
        """A promotional trailer beside the lectures must not promote the
        course folder into a lecture of its own."""
        self.touch("15-210", "trailer.mp4")
        self.touch("15-210", "lecture01", "camera.mp4")
        self.assertEqual(self.rel(discover.find_lectures(self.root)),
                         [os.path.join("15-210", "lecture01")])

    def test_hidden_directories_are_skipped(self):
        """One rule covers .git, .venv and macOS bundle cruft."""
        self.touch(".hidden", "camera.mp4")
        self.touch(".git", "objects", "camera.mp4")
        self.touch("lec1", "camera.mp4")
        self.assertEqual(self.rel(discover.find_lectures(self.root)), ["lec1"])

    def test_pruned_subdirectories_are_not_discovered_as_lectures(self):
        """`cards` holds per-question PNGs and `transition-samples` holds a few
        seconds of rendered mp4 each. Both sit INSIDE a lecture, so discovering
        them would additionally demote their real parent to a course folder."""
        self.touch("lec1", "camera.mp4")
        self.touch("lec1", "cards", "q1.mp4")
        self.touch("lec1", "transition-samples", "fade.mp4")
        self.touch("lec1", "__pycache__", "x.mp4")
        self.assertEqual(self.rel(discover.find_lectures(self.root)), ["lec1"])

    def test_results_are_sorted_and_normalised(self):
        """The report is read top to bottom; an unstable order makes two scans
        of the same semester impossible to diff."""
        for name in ("c", "a", "b"):
            self.touch(name, "camera.mp4")
        found = discover.find_lectures(self.root)
        self.assertEqual(found, sorted(found))
        self.assertTrue(all(p == os.path.normpath(p) for p in found))

    def test_a_missing_or_non_directory_root_returns_an_empty_list(self):
        """--root pointed at a typo must not raise mid-scan."""
        self.assertEqual(discover.find_lectures(
            os.path.join(self.root, "nope")), [])
        self.assertEqual(discover.find_lectures(
            self.touch("a-file.txt", data=b"x")), [])

    def test_a_directory_with_no_video_and_no_markers_is_not_a_lecture(self):
        """Otherwise every intermediate folder in a corpus becomes a row."""
        self.touch("notes", "readme.txt", data=b"hi")
        self.assertEqual(discover.find_lectures(self.root), [])

    def test_non_recursive_stops_at_the_immediate_children(self):
        """For when the caller already knows it is pointing at one semester."""
        self.touch("lec1", "camera.mp4")
        self.touch("15-210", "lecture01", "camera.mp4")
        found = self.rel(discover.find_lectures(self.root, recursive=False))
        self.assertIn("lec1", found)
        self.assertNotIn(os.path.join("15-210", "lecture01"), found)


class TestResolveStreams(TempTree):

    DERIVED = ("camera_muted.mp4", "camera_muted_anon.mp4",
               "camera_muted_anon_tracked.mp4", "camera_sync.mp4",
               "screen_sync.mp4", "screen_with_cards.mp4")

    def test_raw_camera_and_screen_win_over_every_derived_output(self):
        """THE load-bearing rule. camera_sync.mp4 is deliberately NOT preferred
        here even though paths.resolve_camera() prefers it: that is the file
        the pipeline cut, and the scanner is grading what arrived from Panopto.
        Grading our own output would invalidate every number in the report."""
        for name in ("camera.mp4", "screen.mp4") + self.DERIVED:
            self.touch("lec", name)
        for name in ("lec.mp4", "lec-layout.mp4", "lec-camera.mp4",
                     "lec-with-intro.mp4"):
            self.touch("lec", name)
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertEqual(os.path.basename(streams["camera"]), "camera.mp4")
        self.assertEqual(os.path.basename(streams["screen"]), "screen.mp4")
        self.assertEqual(streams["notes"], [])

    def test_every_known_pipeline_output_is_recognised_as_derived(self):
        """Each name below is a real output seen in data/."""
        stems = ["15-210_guid"]
        for name in self.DERIVED + ("15-210_guid.mp4", "15-210_guid-layout.mp4",
                                    "15-210_guid-camera.mp4",
                                    "15-210_guid-with-intro.mp4"):
            with self.subTest(name=name):
                self.assertTrue(discover._is_derived(name, stems))

    def test_a_renamed_directory_still_recognises_its_old_outputs(self):
        """Belt and braces: a corpus renamed after download leaves outputs
        under the old stem, and the tag check has to catch them anyway."""
        for name in ("oldkey-layout.mp4", "something_anon.mp4",
                     "thing_muted.mp4", "x-with-intro.mp4"):
            with self.subTest(name=name):
                self.assertTrue(discover._is_derived(name, ["unrelated"]))

    def test_the_raw_names_are_never_derived(self):
        """If these were ever classified as ours, every lecture would resolve
        to no source at all."""
        self.assertFalse(discover._is_derived("camera.mp4", ["camera"]))
        self.assertFalse(discover._is_derived("screen.mp4", ["screen"]))
        self.assertFalse(discover._is_derived("lecture13.mp4", ["other"]))

    def test_one_conventional_name_present_and_no_raw_partner(self):
        """screen_with_cards.mp4 is ours, so there is no screen here -- and
        saying so beats silently grading the carded render."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "screen_with_cards.mp4")
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertEqual(os.path.basename(streams["camera"]), "camera.mp4")
        self.assertIsNone(streams["screen"])
        self.assertTrue(any("no screen" in n for n in streams["notes"]))

    def test_one_conventional_name_and_exactly_one_raw_partner(self):
        """A renamed capture beside a conventional camera is a safe guess, and
        the guess is recorded in notes rather than made silently."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "projector.mp4")
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertEqual(os.path.basename(streams["camera"]), "camera.mp4")
        self.assertEqual(os.path.basename(streams["screen"]), "projector.mp4")
        self.assertTrue(streams["notes"])

    def test_several_raw_partners_leaves_the_other_stream_unset(self):
        """An unset stream with a note beats a coin flip between three files."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "a.mp4")
        self.touch("lec", "b.mp4")
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertIsNone(streams["screen"])
        self.assertTrue(any("candidates" in n for n in streams["notes"]))

    def test_a_single_raw_video_is_treated_as_the_camera(self):
        """A talk with no slide capture is normal, and the camera is the one
        with a speaker in it."""
        self.touch("lec", "recording.mp4")
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertEqual(os.path.basename(streams["camera"]), "recording.mp4")
        self.assertIsNone(streams["screen"])

    def test_a_directory_of_nothing_but_our_own_output_resolves_to_nothing(self):
        """Better an empty result with a note than a scan of the pipeline."""
        for name in self.DERIVED:
            self.touch("lec", name)
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertIsNone(streams["camera"])
        self.assertIsNone(streams["screen"])
        self.assertTrue(any("pipeline output" in n for n in streams["notes"]))

    def test_an_unlistable_directory_returns_a_note_not_an_exception(self):
        """A semester scan meets at least one unreadable directory."""
        streams = discover.resolve_streams(os.path.join(self.root, "absent"))
        self.assertIsNone(streams["camera"])
        self.assertTrue(any("cannot list" in n for n in streams["notes"]))

    def test_a_corrupt_metadata_json_does_not_break_stream_resolution(self):
        """The key that names our derived outputs comes out of metadata.json;
        a truncated one must fall back to the directory basename."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "metadata.json", data=b"{not json")
        self.touch("lec", "lec-layout.mp4")
        streams = discover.resolve_streams(os.path.join(self.root, "lec"))
        self.assertEqual(os.path.basename(streams["camera"]), "camera.mp4")


class TestClassify(unittest.TestCase):

    @staticmethod
    def _info(path, w, h, bit_rate, has_audio):
        return {"path": path, "width": w, "height": h,
                "bit_rate": bit_rate, "has_audio": has_audio,
                "has_video": True}

    def test_the_stream_carrying_audio_is_the_camera(self):
        """Panopto records the room mic on the camera and the screen capture is
        silent; that single fact settles almost every real case."""
        cam = self._info("a.mp4", 1920, 1080, 659_000, True)
        scr = self._info("b.mp4", 1920, 1080, 284_000, False)
        notes = []
        self.assertEqual(discover._classify([scr, cam], notes)[0]["path"],
                         "a.mp4")
        self.assertTrue(any("only stream with an audio track" in n
                            for n in notes))

    def test_with_no_audio_anywhere_the_cheaper_per_pixel_stream_is_the_screen(self):
        """A slide deck is mostly unchanging flat colour, so x264 spends almost
        nothing on it -- ~284 kb/s against ~659 on this corpus."""
        cam = self._info("cam.mp4", 1280, 720, 659_000, False)
        scr = self._info("scr.mp4", 1920, 1080, 284_000, False)
        notes = []
        camera, screen = discover._classify([cam, scr], notes)
        self.assertEqual(camera["path"], "cam.mp4")
        self.assertEqual(screen["path"], "scr.mp4")
        self.assertTrue(any("encode shape" in n for n in notes))

    def test_bits_per_pixel_beats_raw_bitrate_when_resolutions_differ(self):
        """Ranking on bitrate alone calls the higher-resolution screen the
        camera; bits per pixel separates them either way."""
        cam = self._info("cam.mp4", 640, 360, 400_000, False)     # 1.74 bpp
        scr = self._info("scr.mp4", 1920, 1080, 500_000, False)   # 0.24 bpp
        camera, screen = discover._classify([cam, scr], [])
        self.assertEqual(camera["path"], "cam.mp4")
        self.assertEqual(screen["path"], "scr.mp4")


class TestIdentity(TempTree):

    def test_identity_comes_out_of_metadata_json(self):
        """The report names a lecture by its title; the key is the directory
        basename so it matches paths.LecturePaths.key everywhere else."""
        self.touch("15-210_guid", "camera.mp4")
        with open(os.path.join(self.root, "15-210_guid", "metadata.json"),
                  "w") as fh:
            json.dump({"key": "15-210_abc", "course": "15-210",
                       "name": "Lecture 12", "owner": "Prof",
                       "durationSec": "4773.7", "start": 13300000000}, fh)
        ident = discover.lecture_identity(
            os.path.join(self.root, "15-210_guid"))
        self.assertEqual(ident["key"], "15-210_guid")
        self.assertEqual(ident["panopto_key"], "15-210_abc")
        self.assertEqual(ident["course"], "15-210")
        self.assertEqual(ident["title"], "Lecture 12")
        self.assertEqual(ident["owner"], "Prof")
        self.assertAlmostEqual(ident["duration_s"], 4773.7)

    def test_panopto_start_is_passed_through_as_the_raw_integer(self):
        """It is seconds since 1601-01-01. brand.term_from_panopto is this
        repo's only decoder and the term printed on every published video comes
        off it, so there must not be a second one."""
        self.touch("lec", "camera.mp4")
        with open(os.path.join(self.root, "lec", "metadata.json"), "w") as fh:
            json.dump({"start": 13300000000}, fh)
        ident = discover.lecture_identity(os.path.join(self.root, "lec"))
        self.assertEqual(ident["panopto_start"], 13300000000)
        self.assertIsInstance(ident["panopto_start"], int)

    def test_a_corrupt_metadata_json_yields_partial_info_not_an_exception(self):
        """A semester scan meets at least one truncated metadata.json, and the
        right answer is a partial record with the gap visible."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "metadata.json", data=b'{"course": "15-2')
        ident = discover.lecture_identity(os.path.join(self.root, "lec"))
        self.assertEqual(ident["key"], "lec")
        for field in ("course", "title", "owner", "duration_s",
                      "panopto_key", "panopto_start"):
            with self.subTest(field=field):
                self.assertIsNone(ident[field])

    def test_a_metadata_json_that_is_not_an_object_is_ignored(self):
        """Some half-written files are a bare list or a string."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "metadata.json", data=b'["a", "b"]')
        ident = discover.lecture_identity(os.path.join(self.root, "lec"))
        self.assertEqual(ident["key"], "lec")
        self.assertIsNone(ident["course"])

    def test_an_unparseable_duration_becomes_none_not_a_crash(self):
        """durationSec has been seen as a string, and as the word 'unknown'."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "metadata.json",
                   data=b'{"durationSec": "unknown", "start": "later"}')
        ident = discover.lecture_identity(os.path.join(self.root, "lec"))
        self.assertIsNone(ident["duration_s"])
        self.assertIsNone(ident["panopto_start"])

    def test_no_chapters_file_and_an_empty_chapter_list_are_different_facts(self):
        """'we never downloaded the chapter list' is a gap in the corpus;
        'this lecture has no chapters' is a property of the lecture."""
        self.touch("none", "camera.mp4")
        self.assertIsNone(discover.lecture_identity(
            os.path.join(self.root, "none"))["chapter_count"])
        self.touch("empty", "camera.mp4")
        self.touch("empty", "chapters.json", data=b"[]")
        self.assertEqual(discover.lecture_identity(
            os.path.join(self.root, "empty"))["chapter_count"], 0)

    def test_chapters_wrapped_in_an_envelope_are_still_counted(self):
        """Some exports wrap the list rather than emitting it bare."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "chapters.json",
                   data=b'{"Entries": [{"t": 0}, {"t": 60}]}')
        self.assertEqual(discover.lecture_identity(
            os.path.join(self.root, "lec"))["chapter_count"], 2)

    def test_a_corrupt_chapters_json_reports_unmeasured(self):
        """chapter_count feeds a rubric ramp; a wrong 0 there is a metric that
        says somebody skipped sectioning a lecture they did section."""
        self.touch("lec", "camera.mp4")
        self.touch("lec", "chapters.json", data=b"[[[")
        self.assertIsNone(discover.lecture_identity(
            os.path.join(self.root, "lec"))["chapter_count"])


class TestTranscriptPath(TempTree):

    def setUp(self):
        super().setUp()
        # Point the legacy fallback at an empty temp dir so the test does not
        # depend on whether this checkout has a data/transcription/ lying about.
        self.legacy = tempfile.mkdtemp(prefix="scan-legacy-test-")
        self.addCleanup(shutil.rmtree, self.legacy, True)
        original = discover.LEGACY_TRANSCRIPT_DIR
        discover.LEGACY_TRANSCRIPT_DIR = self.legacy
        self.addCleanup(setattr, discover, "LEGACY_TRANSCRIPT_DIR", original)

    def test_the_per_lecture_transcript_is_preferred(self):
        """REGRESSION: the old shared layout meant a second lecture silently
        overwrote the first one's transcript and every downstream stage built
        the wrong video."""
        local = self.touch("lec", "transcript_classified.json", data=b"[]")
        with open(os.path.join(self.legacy,
                               "transcript_classified.json"), "wb") as fh:
            fh.write(b"[]")
        self.assertEqual(
            discover.transcript_path(os.path.join(self.root, "lec")), local)

    def test_the_legacy_shared_location_is_still_read(self):
        """For lectures processed before transcripts moved into the dir."""
        self.touch("lec", "camera.mp4")
        legacy = os.path.join(self.legacy, "transcript_classified.json")
        with open(legacy, "wb") as fh:
            fh.write(b"[]")
        self.assertEqual(
            discover.transcript_path(os.path.join(self.root, "lec")), legacy)

    def test_no_transcript_anywhere_returns_none(self):
        """Which becomes reduced coverage, not a failed scan: transcription is
        a GPU stage and scanning a semester is not a reason to spend it."""
        self.touch("lec", "camera.mp4")
        self.assertIsNone(
            discover.transcript_path(os.path.join(self.root, "lec")))


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
