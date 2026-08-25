"""src/scan/audio_metrics.py -- levels, the VAD, and the prosody pair.

Every signal here is synthesised, so the right answer is known in closed form:
a full-scale sine is -3.01 dBFS RMS, a 150 Hz tone has an F0 of 150 Hz, and a
tone that never changes pitch has a pitch variety of zero. Those three are what
`pitch_variety_st` -- the highest-weighted delivery metric, and the one the
rubric calls the closest thing to an objective handle on 'engaging lecturer' --
is actually built on.
"""

import unittest

import numpy as np

from tests.support import bursty_speech, quiet_noise, tone

from src.scan import audio_metrics as A


class TestFrameRmsDb(unittest.TestCase):

    def test_full_scale_sine_measures_minus_three_dbfs(self):
        """A sine of amplitude 1.0 has RMS 1/sqrt(2); anything else means the
        window, the hop or the cumulative-sum trick is wrong."""
        db = A._frame_rms_db(tone(200.0, 1.0, amp=1.0), A.LEVEL_WIN, A.LEVEL_HOP)
        self.assertGreater(db.size, 0)
        self.assertTrue(np.allclose(db, -3.0103, atol=0.01))

    def test_half_amplitude_sine_is_six_db_quieter(self):
        """Level must be logarithmic in amplitude, not linear."""
        loud = A._frame_rms_db(tone(200.0, 0.5, amp=1.0),
                               A.LEVEL_WIN, A.LEVEL_HOP)
        quiet = A._frame_rms_db(tone(200.0, 0.5, amp=0.5),
                                A.LEVEL_WIN, A.LEVEL_HOP)
        self.assertAlmostEqual(float(np.median(loud) - np.median(quiet)),
                               6.0206, places=2)

    def test_digital_silence_clamps_to_a_floor_rather_than_minus_inf(self):
        """log10(0) is -inf, and one -inf poisons every percentile, median and
        standard deviation computed downstream of it."""
        db = A._frame_rms_db(np.zeros(A.SR, np.float32),
                             A.LEVEL_WIN, A.LEVEL_HOP)
        self.assertTrue(np.isfinite(db).all())
        self.assertTrue(np.allclose(db, -120.0))

    def test_a_signal_shorter_than_one_window_yields_no_frames(self):
        """Must return an empty array, not raise on a negative arange."""
        db = A._frame_rms_db(np.zeros(10, np.float32),
                             A.LEVEL_WIN, A.LEVEL_HOP)
        self.assertEqual(db.size, 0)

    def test_frame_count_follows_the_window_and_hop(self):
        """The frame grid is what word timings are indexed against in
        speech_metrics; an off-by-one here shifts every dropped-word lookup."""
        x = np.zeros(A.SR, np.float32)
        db = A._frame_rms_db(x, A.LEVEL_WIN, A.LEVEL_HOP)
        expected = len(range(0, x.size - A.LEVEL_WIN + 1, A.LEVEL_HOP))
        self.assertEqual(db.size, expected)


class TestRuns(unittest.TestCase):

    def test_runs_of_a_mixed_mask(self):
        """The ordinary case: two separate runs, half-open [start, end)."""
        mask = np.array([False, True, True, False, True, False])
        self.assertEqual(list(A._runs(mask)), [(1, 3), (4, 5)])

    def test_all_true_is_one_run_covering_everything(self):
        """np.diff sees no transition at all, so both ends have to be inferred."""
        mask = np.ones(7, dtype=bool)
        self.assertEqual(list(A._runs(mask)), [(0, 7)])

    def test_all_false_yields_nothing(self):
        """No run must not become a phantom zero-length one."""
        self.assertEqual(list(A._runs(np.zeros(7, dtype=bool))), [])

    def test_empty_mask_yields_nothing(self):
        """mask[0] on an empty array would raise IndexError."""
        self.assertEqual(list(A._runs(np.zeros(0, dtype=bool))), [])

    def test_runs_touching_both_ends_are_not_dropped(self):
        """A run that starts at index 0 has no rising edge and one that ends at
        the last sample has no falling edge; both are the ones that matter."""
        mask = np.array([True, True, False, False, True, True])
        self.assertEqual(list(A._runs(mask)), [(0, 2), (4, 6)])

    def test_single_true_element(self):
        """A one-element mask has no diff at all."""
        self.assertEqual(list(A._runs(np.array([True]))), [(0, 1)])
        self.assertEqual(list(A._runs(np.array([False]))), [])

    def test_run_lengths_sum_to_the_number_of_true_samples(self):
        """A property that fails on any edge-handling mistake at once."""
        rng = np.random.default_rng(7)
        for i in range(20):
            mask = rng.random(50) > 0.5
            with self.subTest(case=i):
                total = sum(e - s for s, e in A._runs(mask))
                self.assertEqual(total, int(mask.sum()))


class TestEstimateF0(unittest.TestCase):

    def _f0_of(self, freq, seconds=5.0):
        sig = tone(freq, seconds, amp=0.5)
        n_level = 1 + (sig.size - A.LEVEL_WIN) // A.LEVEL_HOP
        return A._estimate_f0(sig, np.ones(n_level, dtype=bool))

    def test_recovers_a_mid_band_pitch(self):
        """150 Hz is an ordinary speaking F0; a few percent is the tolerance
        parabolic interpolation over a 16 kHz lag grid can give."""
        f0 = self._f0_of(150.0)
        self.assertGreater(f0.size, 100)
        self.assertAlmostEqual(float(np.median(f0)) / 150.0, 1.0, delta=0.03)

    def test_recovers_a_pitch_near_the_bottom_of_the_band(self):
        """The 40 ms window exists so 70 Hz still gets two periods in it."""
        f0 = self._f0_of(90.0)
        self.assertGreater(f0.size, 100)
        self.assertAlmostEqual(float(np.median(f0)) / 90.0, 1.0, delta=0.04)

    def test_recovers_a_pitch_near_the_top_of_the_band(self):
        """A high voice must not fall off the lag_lo end of the search."""
        f0 = self._f0_of(300.0)
        self.assertGreater(f0.size, 100)
        self.assertAlmostEqual(float(np.median(f0)) / 300.0, 1.0, delta=0.03)

    def test_everything_returned_is_inside_the_search_band(self):
        """Octave errors are the failure mode; anything outside 70-350 Hz must
        already have been filtered out before the spread is taken."""
        f0 = self._f0_of(150.0)
        self.assertTrue(np.all(f0 >= A.F0_MIN_HZ))
        self.assertTrue(np.all(f0 <= A.F0_MAX_HZ))

    def test_no_voiced_frames_returns_empty_rather_than_raising(self):
        """A lecture with no speech must degrade to 'unmeasured'."""
        sig = tone(150.0, 2.0)
        n_level = 1 + (sig.size - A.LEVEL_WIN) // A.LEVEL_HOP
        self.assertEqual(A._estimate_f0(sig, np.zeros(n_level, bool)).size, 0)

    def test_a_signal_shorter_than_the_f0_window_returns_empty(self):
        """Guards the n_frames <= 0 path."""
        self.assertEqual(A._estimate_f0(np.zeros(100, np.float32),
                                        np.ones(1, bool)).size, 0)

    def test_flat_frames_do_not_emit_a_divide_by_zero_warning(self):
        """np.divide's `where=` was added precisely because a scan of a
        semester printed one RuntimeWarning per lecture."""
        sig = quiet_noise(3.0, amp=1e-6, seed=3)
        n_level = 1 + (sig.size - A.LEVEL_WIN) // A.LEVEL_HOP
        with np.errstate(all="raise"):
            A._estimate_f0(sig, np.ones(n_level, bool))


class TestMeasure(unittest.TestCase):

    def test_measure_returns_a_two_tuple_of_metrics_and_levels(self):
        """scanner.py unpacks this; `levels` is what speech_metrics needs to
        ask whether there was audio where a word is supposed to be."""
        pcm = bursty_speech([150.0], cycles=4)
        out = A.measure(pcm, {"loudness_lufs": -20.0}, pcm.size / A.SR)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        metrics, levels = out
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(levels, dict)
        self.assertEqual(sorted(levels), ["db", "floor_db", "frame_s", "vad"])

    def test_loudness_from_ffmpeg_is_passed_straight_through(self):
        """The standardised figures are ebur128's; nothing here recomputes them."""
        pcm = bursty_speech([150.0], cycles=4)
        loud = {"loudness_lufs": -21.5, "loudness_range_lu": 7.0,
                "true_peak_dbtp": -1.2}
        metrics, _ = A.measure(pcm, loud, pcm.size / A.SR)
        for key, value in loud.items():
            self.assertEqual(metrics[key], value)

    def test_a_constant_tone_has_near_zero_pitch_variety(self):
        """The monotone detector: one unchanging pitch must read as monotone,
        not pick up spread from lag quantisation or octave errors."""
        pcm = bursty_speech([150.0])
        metrics, _ = A.measure(pcm, {"loudness_lufs": -20.0}, pcm.size / A.SR)
        self.assertIn("pitch_variety_st", metrics)
        self.assertLess(metrics["pitch_variety_st"], 0.25)
        self.assertAlmostEqual(metrics["median_f0_hz"], 150.0, delta=5.0)

    def test_a_tone_stepping_an_octave_has_clearly_larger_pitch_variety(self):
        """The other half of the same claim: real pitch movement has to move
        the number, or the metric measures nothing."""
        mono = bursty_speech([150.0])
        varied = bursty_speech([120.0, 240.0])
        flat, _ = A.measure(mono, {"loudness_lufs": -20.0}, mono.size / A.SR)
        moving, _ = A.measure(varied, {"loudness_lufs": -20.0},
                              varied.size / A.SR)
        self.assertGreater(moving["pitch_variety_st"],
                           flat["pitch_variety_st"] + 3.0)
        # An octave split evenly between two pitches is ~6 semitones of spread.
        self.assertGreater(moving["pitch_variety_st"], 4.0)

    def test_the_noise_floor_is_the_room_tone_not_the_quietest_sample(self):
        """astats reports -86 dBFS on lecture 12, which is the quietest sample
        in the file and not the level anybody hears between words."""
        pcm = bursty_speech([150.0])
        metrics, _ = A.measure(pcm, {"loudness_lufs": -20.0}, pcm.size / A.SR)
        # The synthesised room tone is gaussian at 1e-3 -> about -60 dBFS.
        self.assertAlmostEqual(metrics["noise_floor_dbfs"], -60.0, delta=4.0)

    def test_speech_and_snr_are_measured_against_that_floor(self):
        """SNR is median speech-frame RMS over the 10th-percentile frame RMS."""
        pcm = bursty_speech([150.0], quiet_s=1.0, loud_s=4.0)
        metrics, levels = A.measure(pcm, {"loudness_lufs": -20.0},
                                    pcm.size / A.SR)
        self.assertGreater(metrics["snr_db"], 30.0)
        # Four seconds of tone in every five seconds of signal.
        self.assertAlmostEqual(metrics["speech_pct"], 80.0, delta=5.0)
        self.assertAlmostEqual(metrics["speech_seconds"],
                               pcm.size / A.SR * 0.8, delta=3.0)
        self.assertEqual(levels["vad"].shape, levels["db"].shape)

    def test_clipping_is_counted_on_full_scale_samples(self):
        """Distortion baked in at record time, and not fixable afterwards."""
        pcm = tone(200.0, 2.0, amp=1.0)
        metrics, _ = A.measure(pcm, {}, 2.0)
        self.assertGreater(metrics["clipped_pct"], 0.0)
        clean, _ = A.measure(tone(200.0, 2.0, amp=0.2), {}, 2.0)
        self.assertEqual(clean["clipped_pct"], 0.0)

    def test_an_empty_signal_measures_nothing_and_does_not_raise(self):
        """A camera whose audio decode produced nothing is still a row."""
        metrics, levels = A.measure(np.zeros(0, np.float32),
                                    {"loudness_lufs": -70.0}, 0.0)
        self.assertEqual(metrics, {"loudness_lufs": -70.0})
        self.assertEqual(levels, {})

    def test_pure_digital_silence_reports_itself_rather_than_inventing_a_floor(self):
        """A mic that was never switched on must hand the not_silent gate a
        speech_pct of 0, not a plausible-looking room tone."""
        metrics, levels = A.measure(np.zeros(A.SR * 5, np.float32),
                                    {"loudness_lufs": -70.0}, 5.0)
        self.assertEqual(metrics["speech_pct"], 0.0)
        self.assertEqual(metrics["snr_db"], 0.0)
        self.assertEqual(metrics["noise_floor_dbfs"], -120.0)
        self.assertEqual(levels, {})
        self.assertNotIn("pitch_variety_st", metrics)

    def test_level_stability_needs_at_least_three_measurable_minutes(self):
        """Standard deviation of three points is already thin; two is nothing.
        A short clip must leave the metric unmeasured rather than report 0."""
        short = bursty_speech([150.0], cycles=4)          # ~20 s
        metrics, _ = A.measure(short, {"loudness_lufs": -20.0},
                               short.size / A.SR)
        self.assertNotIn("level_stability_db", metrics)

    def test_level_stability_is_near_zero_for_a_steady_speaker(self):
        """It is the standard deviation of the per-minute median speech level;
        a speaker who never moves must not look like one who wandered off."""
        pcm = bursty_speech([150.0], cycles=60, quiet_s=1.0, loud_s=4.0)
        metrics, _ = A.measure(pcm, {"loudness_lufs": -20.0}, pcm.size / A.SR)
        self.assertIn("level_stability_db", metrics)
        self.assertLess(metrics["level_stability_db"], 0.5)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
