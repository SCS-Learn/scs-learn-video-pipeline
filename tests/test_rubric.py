"""src/scan/rubric.py -- the scoring curves and the integrity of the table.

Everything the scanner calls "good" is decided here, so a silent change to a
band or a mistyped `dimension` key would move every grade in a semester with
nothing to notice it. These tests read the table rather than hardcoding its
numbers, so retuning a band stays a one-line edit and only a *structural*
mistake fails.
"""

import contextlib
import unittest

from tests.support import _ROOT  # noqa: F401  (puts the repo root on sys.path)

from src.scan import rubric


@contextlib.contextmanager
def temp_metric(metric_id, spec):
    """Register a synthetic metric for the duration of one test."""
    rubric.METRICS[metric_id] = spec
    try:
        yield metric_id
    finally:
        rubric.METRICS.pop(metric_id, None)


RAMP_UP = {"dimension": "audio", "weight": 1.0, "tier": "signal",
           "fixable": False, "label": "Synthetic rising", "unit": "u",
           "scale": rubric.ramp(10.0, 30.0), "why": "test fixture"}
RAMP_DOWN = {"dimension": "audio", "weight": 1.0, "tier": "signal",
             "fixable": False, "label": "Synthetic falling", "unit": "u",
             "scale": rubric.ramp(30.0, 10.0), "why": "test fixture"}
BAND = {"dimension": "audio", "weight": 1.0, "tier": "signal",
        "fixable": False, "label": "Synthetic band", "unit": "u",
        "scale": rubric.band(0.0, 10.0, 20.0, 30.0), "why": "test fixture"}
FLAG = {"dimension": "structure", "weight": 1.0, "tier": "speech",
        "fixable": False, "label": "Synthetic flag",
        "scale": rubric.BOOL, "why": "test fixture"}


class TestScoreMetricRamp(unittest.TestCase):

    def test_rising_ramp_endpoints_and_midpoint(self):
        """A higher-is-better ramp must hit 0 at `bad`, 1 at `good`, 0.5 halfway."""
        with temp_metric("_synth_up", RAMP_UP) as mid:
            self.assertAlmostEqual(rubric.score_metric(mid, 10.0), 0.0)
            self.assertAlmostEqual(rubric.score_metric(mid, 30.0), 1.0)
            self.assertAlmostEqual(rubric.score_metric(mid, 20.0), 0.5)

    def test_rising_ramp_clamps_outside_its_ends(self):
        """Past either end a ramp must clamp, never run above 1 or below 0."""
        with temp_metric("_synth_up", RAMP_UP) as mid:
            self.assertEqual(rubric.score_metric(mid, -1000.0), 0.0)
            self.assertEqual(rubric.score_metric(mid, 1000.0), 1.0)

    def test_falling_ramp_scores_the_other_way_round(self):
        """A lower-is-better ramp (bad > good) must not be scored as if rising."""
        with temp_metric("_synth_down", RAMP_DOWN) as mid:
            self.assertAlmostEqual(rubric.score_metric(mid, 30.0), 0.0)
            self.assertAlmostEqual(rubric.score_metric(mid, 10.0), 1.0)
            self.assertAlmostEqual(rubric.score_metric(mid, 20.0), 0.5)
            self.assertEqual(rubric.score_metric(mid, 1000.0), 0.0)
            self.assertEqual(rubric.score_metric(mid, -1000.0), 1.0)

    def test_degenerate_ramp_does_not_divide_by_zero(self):
        """bad == good must return 1.0 rather than raising ZeroDivisionError."""
        spec = dict(RAMP_UP, scale=rubric.ramp(5.0, 5.0))
        with temp_metric("_synth_flat", spec) as mid:
            self.assertEqual(rubric.score_metric(mid, 5.0), 1.0)
            self.assertEqual(rubric.score_metric(mid, 500.0), 1.0)


class TestScoreMetricBand(unittest.TestCase):

    def test_band_interior_is_full_marks(self):
        """Anything between lo and hi scores 1.0 -- the flat top of the inverted U."""
        with temp_metric("_synth_band", BAND) as mid:
            self.assertAlmostEqual(rubric.score_metric(mid, 15.0), 1.0)

    def test_band_inner_edges_score_one(self):
        """lo and hi themselves are inside the plateau, not on the slope."""
        with temp_metric("_synth_band", BAND) as mid:
            self.assertAlmostEqual(rubric.score_metric(mid, 10.0), 1.0)
            self.assertAlmostEqual(rubric.score_metric(mid, 20.0), 1.0)

    def test_band_shoulders_ramp_linearly_on_both_sides(self):
        """Both shoulders must slope, so a band is not silently a one-sided ramp."""
        with temp_metric("_synth_band", BAND) as mid:
            self.assertAlmostEqual(rubric.score_metric(mid, 5.0), 0.5)
            self.assertAlmostEqual(rubric.score_metric(mid, 25.0), 0.5)
            self.assertAlmostEqual(rubric.score_metric(mid, 2.5), 0.25)
            self.assertAlmostEqual(rubric.score_metric(mid, 27.5), 0.25)

    def test_band_hard_edges_and_beyond_are_zero(self):
        """The hard edges are inclusive zeros: at/outside them the metric fails."""
        with temp_metric("_synth_band", BAND) as mid:
            self.assertEqual(rubric.score_metric(mid, 0.0), 0.0)
            self.assertEqual(rubric.score_metric(mid, 30.0), 0.0)
            self.assertEqual(rubric.score_metric(mid, -50.0), 0.0)
            self.assertEqual(rubric.score_metric(mid, 500.0), 0.0)


class TestScoreMetricOther(unittest.TestCase):

    def test_bool_metric_maps_true_and_false(self):
        """A BOOL scale is 1.0/0.0, not a truthiness leak into the arithmetic."""
        with temp_metric("_synth_flag", FLAG) as mid:
            self.assertEqual(rubric.score_metric(mid, True), 1.0)
            self.assertEqual(rubric.score_metric(mid, False), 0.0)

    def test_none_returns_none_for_every_scale_kind(self):
        """Unmeasured must stay None -- scoring it as 0 is the error this repo
        cannot afford, because 'we did not look' would read as 'it was bad'."""
        for name, spec in (("_synth_up", RAMP_UP), ("_synth_band", BAND),
                           ("_synth_flag", FLAG)):
            with temp_metric(name, spec) as mid:
                self.assertIsNone(rubric.score_metric(mid, None))

    def test_unknown_scale_kind_raises(self):
        """A typo in a scale tuple must fail loudly, not score silently."""
        spec = dict(RAMP_UP, scale=("wobble", 1.0, 2.0))
        with temp_metric("_synth_bad", spec) as mid:
            with self.assertRaises(ValueError):
                rubric.score_metric(mid, 1.0)


class TestEveryRealMetricScores(unittest.TestCase):
    """The same properties, checked against every metric actually in the table.

    These read the thresholds out of METRICS rather than restating them, so
    recalibrating a band does not break them -- only a broken *shape* does.
    """

    def test_every_ramp_metric_hits_both_ends_and_its_midpoint(self):
        """Guards against a ramp written backwards or with a stray extra element."""
        for mid, spec in rubric.METRICS.items():
            if spec["scale"][0] != "ramp":
                continue
            _, bad, good = spec["scale"]
            if bad == good:
                continue
            with self.subTest(metric=mid):
                self.assertAlmostEqual(rubric.score_metric(mid, bad), 0.0)
                self.assertAlmostEqual(rubric.score_metric(mid, good), 1.0)
                self.assertAlmostEqual(
                    rubric.score_metric(mid, (bad + good) / 2.0), 0.5)
                beyond = good + (good - bad)
                self.assertAlmostEqual(rubric.score_metric(mid, beyond), 1.0)
                under = bad - (good - bad)
                self.assertAlmostEqual(rubric.score_metric(mid, under), 0.0)

    def test_every_band_metric_peaks_inside_and_zeroes_outside(self):
        """Guards against a band whose four numbers stopped being ordered."""
        for mid, spec in rubric.METRICS.items():
            if spec["scale"][0] != "band":
                continue
            _, hard_lo, lo, hi, hard_hi = spec["scale"]
            with self.subTest(metric=mid):
                self.assertLess(hard_lo, lo)
                self.assertLessEqual(lo, hi)
                self.assertLess(hi, hard_hi)
                self.assertAlmostEqual(rubric.score_metric(mid, lo), 1.0)
                self.assertAlmostEqual(rubric.score_metric(mid, hi), 1.0)
                self.assertAlmostEqual(
                    rubric.score_metric(mid, (lo + hi) / 2.0), 1.0)
                self.assertEqual(rubric.score_metric(mid, hard_lo), 0.0)
                self.assertEqual(rubric.score_metric(mid, hard_hi), 0.0)
                self.assertAlmostEqual(
                    rubric.score_metric(mid, (hard_lo + lo) / 2.0), 0.5)
                self.assertAlmostEqual(
                    rubric.score_metric(mid, (hi + hard_hi) / 2.0), 0.5)

    def test_both_ramp_directions_are_represented(self):
        """Half the table is better-when-lower; if that stopped being true the
        direction-normalisation the cohort percentiles rely on is untested."""
        rising = falling = 0
        for spec in rubric.METRICS.values():
            if spec["scale"][0] != "ramp":
                continue
            _, bad, good = spec["scale"]
            rising += good > bad
            falling += good < bad
        self.assertGreater(rising, 0)
        self.assertGreater(falling, 0)


class TestTableIntegrity(unittest.TestCase):

    def test_every_metric_declares_a_known_dimension(self):
        """A typo'd dimension silently drops the metric out of every score."""
        for mid, spec in rubric.METRICS.items():
            with self.subTest(metric=mid):
                self.assertIn(spec["dimension"], rubric.DIMENSIONS)

    def test_every_metric_declares_a_known_tier(self):
        """score.evaluate sorts missing tiers with TIERS.index; an unknown tier
        raises ValueError while explaining why a scan was incomplete."""
        for mid, spec in rubric.METRICS.items():
            with self.subTest(metric=mid):
                self.assertIn(spec["tier"], rubric.TIERS)

    def test_dimension_weights_sum_to_one(self):
        """The weighted total assumes this; a drift makes every score wrong."""
        total = sum(d["weight"] for d in rubric.DIMENSIONS.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_every_dimension_has_at_least_one_metric(self):
        """A dimension with no metrics contributes weight but never a score."""
        for dim in rubric.DIMENSIONS:
            with self.subTest(dimension=dim):
                self.assertTrue(rubric.dimension_ids(dim))

    def test_every_metric_has_label_why_weight_fixable_and_scale(self):
        """report.py and score.py read all five on every metric, unguarded."""
        for mid, spec in rubric.METRICS.items():
            with self.subTest(metric=mid):
                for field in ("label", "why", "weight", "fixable", "scale"):
                    self.assertIn(field, spec)
                self.assertIsInstance(spec["label"], str)
                self.assertTrue(spec["label"].strip())
                self.assertIsInstance(spec["why"], str)
                self.assertTrue(spec["why"].strip())
                self.assertIsInstance(spec["weight"], (int, float))
                self.assertGreater(spec["weight"], 0.0)
                self.assertIsInstance(spec["fixable"], bool)
                self.assertIsInstance(spec["scale"], tuple)
                self.assertIn(spec["scale"][0], ("ramp", "band", "bool"))

    def test_every_remediated_key_is_a_real_metric(self):
        """A stale key here is a remediation nobody is ever offered."""
        for mid in rubric.REMEDIATED:
            with self.subTest(metric=mid):
                self.assertIn(mid, rubric.METRICS)

    def test_every_remediated_metric_is_marked_fixable(self):
        """score._remediation skips anything not fixable, so a fixable=False
        entry in REMEDIATED would sit in the table doing nothing."""
        for mid in rubric.REMEDIATED:
            with self.subTest(metric=mid):
                self.assertTrue(rubric.METRICS[mid]["fixable"])

    def test_remediated_targets_never_make_a_metric_worse(self):
        """`potential` claims to be the lecture after remediation; a target on
        the wrong side of a band edge would make it a downgrade."""
        for mid, target in rubric.REMEDIATED.items():
            if isinstance(target, str):
                continue
            with self.subTest(metric=mid):
                self.assertIsNotNone(rubric.score_metric(mid, target))

    def test_gate_ids_are_unique_and_carry_a_label_and_why(self):
        """score._check_gates looks a gate up by id with next() and no default."""
        ids = [g["id"] for g in rubric.GATES]
        self.assertEqual(len(ids), len(set(ids)))
        for gate in rubric.GATES:
            with self.subTest(gate=gate["id"]):
                self.assertTrue(gate["label"].strip())
                self.assertTrue(gate["why"].strip())

    def test_grades_are_ordered_high_to_low_and_end_at_zero(self):
        """grade_for walks GRADES top down and falls off the end at 0.0."""
        thresholds = [t for t, _, _, _ in rubric.GRADES]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))
        self.assertEqual(thresholds[-1], 0.0)

    def test_tier_blurb_covers_every_tier(self):
        """--explain-rubric indexes TIER_BLURB by tier with no default."""
        self.assertEqual(sorted(rubric.TIER_BLURB), sorted(rubric.TIERS))


class TestRemediatedValue(unittest.TestCase):

    def test_absolute_target_is_only_ever_an_improvement(self):
        """An already-good lecture must not be dragged down to the target."""
        # loudness_lufs targets -18; a lecture already at -18 must stay there,
        # and one at the band centre must not be pulled off it.
        self.assertEqual(rubric.remediated_value("loudness_lufs", -18.0), -18.0)
        already = rubric.remediated_value("loudness_lufs", -20.0)
        self.assertGreaterEqual(rubric.score_metric("loudness_lufs", already),
                                rubric.score_metric("loudness_lufs", -20.0))

    def test_relative_target_is_applied_as_an_offset(self):
        """'+6' on snr_db means six more dB, not a jump to 6 dB."""
        self.assertAlmostEqual(rubric.remediated_value("snr_db", 9.0), 15.0)
        self.assertAlmostEqual(
            rubric.remediated_value("noise_floor_dbfs", -40.0), -48.0)

    def test_unmeasured_and_unremediable_pass_straight_through(self):
        """None must not become a number, and a metric with no remedy must not
        acquire one."""
        self.assertIsNone(rubric.remediated_value("snr_db", None))
        self.assertEqual(rubric.remediated_value("clipped_pct", 0.4), 0.4)


class TestGradeFor(unittest.TestCase):

    def test_a_failed_gate_forces_f_and_skip_at_any_score(self):
        """A high-scoring lecture with no instructor in shot is still a skip."""
        letter, verdict, blurb = rubric.grade_for(99.0, ["Instructor is in shot"])
        self.assertEqual(letter, "F")
        self.assertEqual(verdict, "skip")
        self.assertIn("Instructor is in shot", blurb)

    def test_bands_are_applied_top_down(self):
        """85 is an A and 84.9 is not; the boundary is inclusive at the top."""
        self.assertEqual(rubric.grade_for(85.0, [])[0], "A")
        self.assertEqual(rubric.grade_for(84.9, [])[0], "B")
        self.assertEqual(rubric.grade_for(0.0, [])[0], "F")


class TestExplain(unittest.TestCase):

    def test_explain_renders_every_metric_and_gate(self):
        """--explain-rubric is the documented way to argue with a threshold, so
        a metric missing from it is a threshold nobody can find."""
        text = rubric.explain()
        for mid in rubric.METRICS:
            self.assertIn(mid, text)
        for gate in rubric.GATES:
            self.assertIn(gate["id"], text)

    def test_wrap_never_exceeds_its_width_on_ordinary_prose(self):
        """explain() lays the whole rubric out inside 78 columns."""
        lines = rubric._wrap(" ".join(["word"] * 60), 30)
        self.assertTrue(all(len(line) <= 30 for line in lines))
        self.assertEqual(" ".join(lines), " ".join(["word"] * 60))


class TestOverrides(unittest.TestCase):

    def test_missing_override_file_is_a_no_op(self):
        """Deleting the calibration file must restore the absolute rubric
        rather than erroring the scan out."""
        self.assertEqual(rubric.load_overrides("/nonexistent/overrides.json"), 0)

    def test_unreadable_override_file_is_ignored_not_raised(self):
        """A truncated recalibration must not take a semester scan down."""
        import io
        import os
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            fh.write("{not json")
            path = fh.name
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(rubric.load_overrides(path, quiet=True), 0)
            self.assertIn("ignoring unreadable overrides", out.getvalue())
        finally:
            os.unlink(path)

    def test_an_override_replaces_only_the_bands_it_names(self):
        """An override says what a typical lecture in one semester is; deleting
        the file has to restore the absolute rubric, so it must never add,
        remove or reweight a metric -- only swap a scale."""
        import json
        import os
        import tempfile
        before = dict(rubric.METRICS["snr_db"])
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump({"cohort_size": 40,
                       "scales": {"snr_db": ["ramp", 3.0, 19.0],
                                  "not_a_metric": ["ramp", 0.0, 1.0]}}, fh)
            path = fh.name
        try:
            with contextlib.redirect_stdout(None):
                self.assertEqual(rubric.load_overrides(path), 1)
            self.assertEqual(rubric.METRICS["snr_db"]["scale"],
                             ("ramp", 3.0, 19.0))
            self.assertEqual(rubric.METRICS["snr_db"]["weight"],
                             before["weight"])
            self.assertNotIn("not_a_metric", rubric.METRICS)
        finally:
            rubric.METRICS["snr_db"] = before
            os.unlink(path)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
