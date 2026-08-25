"""src/scan/video_metrics.py -- focus, the black lead, and the sync margin.

`sync_risk` is the one that matters most and needs no media at all: it is how
the scanner catches, before anyone spends an encode, the lecture whose screen
black lead outlasts the duration alignment. CLAUDE.md records that lecture 12
cleared it by 89.7 seconds and that verify.py would have passed a video that
did not, because black encodes perfectly well.

The frame-level tests build a small H.264 file with ffmpeg into a temp dir and
skip themselves if ffmpeg is not installed. Nothing here touches data/.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from tests.support import (VIDEO_H, VIDEO_W, black_frame, have_ffmpeg,
                           pillarboxed_frame, slide_frame, write_gray_video)

from src.scan import video_metrics as V


class TestLaplacianVar(unittest.TestCase):

    def test_a_uniform_frame_has_no_focus_energy(self):
        """A flat grey field has zero Laplacian variance by construction; a
        non-zero answer means the kernel or the border trim is wrong."""
        self.assertAlmostEqual(
            V._laplacian_var(np.full((100, 100), 128, np.uint8)), 0.0)

    def test_a_sharp_edge_scores_far_higher_than_a_uniform_frame(self):
        """The whole point of the metric: an in-focus frame has edges in it."""
        sharp = np.zeros((100, 100), np.uint8)
        sharp[:, 50:] = 255
        self.assertGreater(V._laplacian_var(sharp),
                           V._laplacian_var(np.full((100, 100), 128, np.uint8)))
        self.assertGreater(V._laplacian_var(sharp), 100.0)

    def test_a_blurred_edge_scores_below_a_hard_one(self):
        """Read as a focus measure, so defocus has to lower it."""
        hard = np.zeros((100, 100), np.float32)
        hard[:, 50:] = 255.0
        soft = hard.copy()
        for i, value in enumerate(np.linspace(0, 255, 21)):
            soft[:, 40 + i] = value
        self.assertGreater(V._laplacian_var(hard), V._laplacian_var(soft))

    def test_random_noise_scores_higher_than_a_gradient(self):
        """The measure also rises with scene detail, which the rubric says out
        loud -- it is a floor, not a ranking."""
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 256, (100, 100)).astype(np.uint8)
        gradient = np.tile(np.linspace(0, 255, 100), (100, 1)).astype(np.uint8)
        self.assertGreater(V._laplacian_var(noise), V._laplacian_var(gradient))


class TestSyncRisk(unittest.TestCase):

    def test_a_comfortable_margin_scores_a_clean_one(self):
        """The duration alignment covers the black lead with room to spare."""
        risk, margin = V.sync_risk(1000.0, 2000.0, 10.0)
        self.assertEqual(risk, 1.0)
        self.assertAlmostEqual(margin, 990.0)

    def test_a_negative_margin_scores_zero(self):
        """The black outlasts what the alignment removes, so the published
        video opens on black and nothing downstream notices."""
        risk, margin = V.sync_risk(1000.0, 1010.0, 500.0)
        self.assertEqual(risk, 0.0)
        self.assertAlmostEqual(margin, -490.0)

    def test_the_lecture_twelve_case_lands_where_claude_md_says_it_does(self):
        """camera 4773.7, screen 5492.4, black lead 630.2 -> 88.5s of margin,
        which CLAUDE.md flags as uncomfortably close and this scores ~0.74."""
        risk, margin = V.sync_risk(4773.7, 5492.4, 630.2)
        self.assertAlmostEqual(margin, 88.5, places=1)
        self.assertAlmostEqual(risk, 0.74, places=2)

    def test_exactly_the_comfort_margin_scores_one(self):
        """SYNC_COMFORT_S is the point at which the metric stops worrying."""
        risk, _ = V.sync_risk(1000.0, 1000.0 + V.SYNC_COMFORT_S, 0.0)
        self.assertEqual(risk, 1.0)

    def test_zero_margin_scores_zero_without_going_negative(self):
        """Just-covered is not comfortable, but the score must clamp at 0."""
        risk, margin = V.sync_risk(1000.0, 1100.0, 100.0)
        self.assertEqual(risk, 0.0)
        self.assertAlmostEqual(margin, 0.0)

    def test_a_missing_black_lead_is_treated_as_none(self):
        """measure_screen leaves black_lead_s out when it sampled too few
        frames; that must not become a TypeError mid-semester."""
        risk, margin = V.sync_risk(1000.0, 2000.0, None)
        self.assertEqual(risk, 1.0)
        self.assertAlmostEqual(margin, 1000.0)

    def test_a_missing_duration_reports_unmeasured_rather_than_guessing(self):
        """No screen stream, or a stream that would not probe."""
        self.assertEqual(V.sync_risk(None, 5000.0, 10.0), (None, None))
        self.assertEqual(V.sync_risk(4773.7, None, 10.0), (None, None))
        self.assertEqual(V.sync_risk(0.0, 0.0, 0.0), (None, None))

    def test_risk_falls_monotonically_as_the_black_lead_grows(self):
        """A worse lecture must never score better."""
        scores = [V.sync_risk(1000.0, 1300.0, lead)[0]
                  for lead in (0.0, 100.0, 200.0, 250.0, 300.0, 400.0)]
        self.assertEqual(scores, sorted(scores, reverse=True))


@unittest.skipUnless(have_ffmpeg(), "ffmpeg/ffprobe not installed")
class TestMeasureScreen(unittest.TestCase):
    """Frame-level slide metrics, against tiny videos generated here.

    Every file is a handful of frames written into a temp directory. No
    lecture media, no data/, nothing that outlives the test.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="scan-video-test-")

        # 6 s at 10 fps. Black for the first 3 s except for one lit frame at
        # 1.2 s, then six distinct slides.
        frames = [slide_frame(9) if i == 12 else black_frame()
                  for i in range(30)]
        frames += [slide_frame(i // 5) for i in range(30)]
        cls.lead_path = write_gray_video(
            os.path.join(cls.tmp, "lead.mp4"), frames, fps=10)

        # No black at all, one slide change halfway.
        steady = [slide_frame(0)] * 20 + [slide_frame(5)] * 20
        cls.steady_path = write_gray_video(
            os.path.join(cls.tmp, "steady.mp4"), steady, fps=10)

        # A 4:3 deck pillarboxed into the 16:9 frame.
        pillar = [pillarboxed_frame(i // 5) for i in range(40)]
        cls.pillar_path = write_gray_video(
            os.path.join(cls.tmp, "pillar.mp4"), pillar, fps=10)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_black_lead_survives_a_single_lit_frame_inside_the_dark_run(self):
        """REGRESSION: lecture 12's screen is dark for 629s but flickers once
        at 132s, and taking the first non-black frame put the lead at 132 -- a
        five-minute underestimate of the number that decides whether the
        published video opens on black. The 90%-purity rule finds all 3s here,
        where 'first non-black frame' would report 1.2s."""
        m = V.measure_screen(self.lead_path, 6.0)
        self.assertAlmostEqual(m["black_lead_s"], 3.0, delta=0.15)

    def test_black_lead_does_not_walk_past_the_end_of_the_dark_run(self):
        """REGRESSION: purity alone tolerates 10% non-black by construction, so
        it kept walking into the lecture and once reported 697s of lead in a
        file holding 628s of black. Anchoring on an actually-black frame stops
        it at the dark run's end -- here 3.0s of a 6.0s file, and never more
        than the purity rule can account for."""
        m = V.measure_screen(self.lead_path, 6.0)
        step = m["screen_sample_step_s"]
        self.assertLessEqual(m["black_lead_s"], 3.0 + step)
        self.assertLess(m["black_lead_s"], 6.0)
        black_s = m["screen_black_pct"] / 100.0 * 6.0
        self.assertLessEqual(m["black_lead_s"] * V.BLACK_LEAD_PURITY,
                             black_s + step)

    def test_black_percentage_counts_dark_flat_frames_only(self):
        """A dark slide with white text is not a black frame, and only the
        per-frame standard deviation tells the two apart."""
        m = V.measure_screen(self.lead_path, 6.0)
        self.assertAlmostEqual(m["screen_black_pct"], 29 / 60 * 100.0, delta=2.0)
        clean = V.measure_screen(self.steady_path, 4.0)
        self.assertAlmostEqual(clean["screen_black_pct"], 0.0, delta=1.0)
        self.assertAlmostEqual(clean["black_lead_s"], 0.0)

    def test_slide_changes_are_counted_and_scaled_to_an_hourly_rate(self):
        """One change in the file must be one change in the count, not one per
        frame of compression noise."""
        m = V.measure_screen(self.steady_path, 4.0)
        self.assertEqual(m["slide_change_count"], 1)
        self.assertAlmostEqual(m["slide_change_per_hour"],
                               1 / (4.0 / 3600.0), places=3)

    def test_an_unchanging_deck_registers_no_slide_changes(self):
        """The threshold has to sit clear of encoder noise, or every lecture
        reads as a video playback the Scene A window handles badly."""
        path = write_gray_video(os.path.join(self.tmp, "static.mp4"),
                                [slide_frame(3)] * 30, fps=10)
        m = V.measure_screen(path, 3.0)
        self.assertEqual(m["slide_change_count"], 0)
        self.assertAlmostEqual(m["longest_static_slide_s"], 3.0, delta=0.2)

    def test_longest_static_slide_ignores_the_leading_black(self):
        """Twenty minutes of black before the lecture is a sync problem, already
        counted, not a slide nobody advanced."""
        m = V.measure_screen(self.lead_path, 6.0)
        self.assertLess(m["longest_static_slide_s"], 3.0)

    def test_a_sixteen_by_nine_deck_scores_a_clean_aspect_fit(self):
        """The slide window is 1380x776; a native 16:9 capture fills it."""
        m = V.measure_screen(self.steady_path, 4.0)
        self.assertAlmostEqual(m["screen_content_aspect"], 16 / 9, delta=0.05)
        self.assertAlmostEqual(m["screen_aspect"], 1.0, delta=0.03)

    def test_a_pillarboxed_four_by_three_deck_is_detected_as_such(self):
        """REGRESSION: 17-635 lecture 13's 4:3 deck measured as a clean 16:9
        until the content box became a frequency count over frames rather than
        a maximum, because one full-width flash defeats a union."""
        m = V.measure_screen(self.pillar_path, 4.0)
        self.assertLess(m["screen_content_aspect"], 16 / 9)
        self.assertAlmostEqual(m["screen_content_aspect"], 4 / 3, delta=0.15)
        self.assertLess(m["screen_aspect"], 0.9)

    def test_a_file_with_too_few_frames_measures_nothing(self):
        """Fewer than two samples cannot produce a difference, and inventing
        one would be worse than reporting no measurement."""
        path = write_gray_video(os.path.join(self.tmp, "one.mp4"),
                                [slide_frame(1)], fps=10)
        self.assertEqual(V.measure_screen(path, 0.1), {})

    def test_sampling_geometry_is_reported_for_the_reader(self):
        """Every rate here is derived from the step, so it has to be visible."""
        m = V.measure_screen(self.steady_path, 4.0)
        self.assertEqual(m["screen_frames_sampled"], 40)
        self.assertAlmostEqual(m["screen_sample_step_s"], 0.1, places=6)


@unittest.skipUnless(have_ffmpeg(), "ffmpeg/ffprobe not installed")
class TestMeasureCamera(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="scan-camera-test-")
        rng = np.random.default_rng(11)
        detailed = [rng.integers(0, 256, (VIDEO_H, VIDEO_W)).astype(np.uint8)
                    for _ in range(20)]
        cls.sharp_path = write_gray_video(
            os.path.join(cls.tmp, "sharp.mp4"), detailed, fps=10)
        cls.flat_path = write_gray_video(
            os.path.join(cls.tmp, "flat.mp4"),
            [np.full((VIDEO_H, VIDEO_W), 128, np.uint8)] * 20, fps=10)
        cls.dark_path = write_gray_video(
            os.path.join(cls.tmp, "dark.mp4"),
            [np.full((VIDEO_H, VIDEO_W), 2, np.uint8)] * 20, fps=10)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_detailed_camera_is_sharper_than_a_flat_one(self):
        """camera_sharpness is a floor against the lecture shot out of focus."""
        detailed = V.measure_camera(self.sharp_path, 2.0)
        flat = V.measure_camera(self.flat_path, 2.0)
        self.assertGreater(detailed["camera_sharpness"],
                           flat["camera_sharpness"])

    def test_a_crushed_dark_camera_scores_badly_on_exposure(self):
        """The classic lecture-hall failure is the lights down for the
        projector and the lecturer in silhouette: crushed blacks, no contrast."""
        dark = V.measure_camera(self.dark_path, 2.0)
        self.assertLess(dark["camera_exposure"], 0.2)
        self.assertGreater(dark["camera_crushed_pct"], 90.0)
        self.assertEqual(dark["camera_blown_pct"], 0.0)

    def test_exposure_stays_inside_zero_to_one(self):
        """It is fed straight into a ramp scaled 0..1."""
        for path in (self.sharp_path, self.flat_path, self.dark_path):
            with self.subTest(path=os.path.basename(path)):
                exposure = V.measure_camera(path, 2.0)["camera_exposure"]
                self.assertGreaterEqual(exposure, 0.0)
                self.assertLessEqual(exposure, 1.0)

    def test_a_still_camera_reports_no_motion(self):
        """camera_motion drives nothing in the rubric yet but is reported, so
        it must not read as movement on a locked-off tripod."""
        self.assertAlmostEqual(
            V.measure_camera(self.flat_path, 2.0)["camera_motion"], 0.0,
            delta=0.5)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
