"""src/scan/score.py -- the coverage floor, the gates, and the two scores.

Three of these guard bugs that actually shipped and are named in the module's
own docstring:

  * a probe-only pass scored 93.8 from three metrics and looked like an A;
  * a lecture whose download had failed came back a provisional 100.0 with no
    camera at all, because the media gates treated "absent" as "not measured";
  * a tier that did not run read as a zero and ranked the whole semester last.

The rest are invariants the reports depend on: `potential` never below
`score`, an unmeasured dimension reporting None rather than 0.
"""

import unittest

from tests import support
from tests.support import PROBE_METRICS, full_metrics, probe_info

from src.scan import rubric, score


class TestCoverageFloor(unittest.TestCase):

    def test_probe_only_scan_is_not_graded(self):
        """REGRESSION: a probe-only pass scored 93.8 from three metrics that
        happened to be fine. A thin scan must report '?' / 'incomplete'."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        self.assertEqual(r["grade"], "?")
        self.assertEqual(r["verdict"], "incomplete")
        self.assertNotEqual(r["grade"], "A")

    def test_probe_only_still_reports_a_provisional_score(self):
        """The number is kept -- it is the *grade* that is withheld -- so the
        report can show a provisional figure without anybody shipping on it."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        self.assertIsNotNone(r["score"])
        self.assertLess(r["coverage"], score.MIN_GRADE_COVERAGE)

    def test_probe_only_warns_and_names_the_tiers_still_to_run(self):
        """'incomplete' has to be actionable, so it says what would finish it."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        self.assertTrue(r["warnings"])
        self.assertIn("provisional", r["verdict_blurb"])
        self.assertTrue(any(t in r["verdict_blurb"] for t in rubric.TIERS))

    def test_a_full_scan_grades_normally(self):
        """The floor must not block a complete scan: full coverage, real grade."""
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "full"}, tiers_run=rubric.TIERS)
        self.assertAlmostEqual(r["coverage"], 1.0, places=6)
        self.assertEqual(r["grade"], "A")
        self.assertEqual(r["verdict"], "publish")
        self.assertEqual(r["gates_failed"], [])

    def test_nothing_measured_is_unscanned_not_zero(self):
        """No measurement at all must be '?' / 'unscanned' with score None, so
        a failed scan never ranks as a lecturer who did badly."""
        r = score.evaluate({}, probe_info(), identity={"key": "empty"})
        self.assertIsNone(r["score"])
        self.assertEqual(r["grade"], "?")
        self.assertEqual(r["verdict"], "unscanned")


class TestGates(unittest.TestCase):

    def _gate(self, result, gate_id):
        return next(g for g in result["gate_detail"] if g["id"] == gate_id)

    def test_missing_camera_fails_media_readable_and_has_audio(self):
        """REGRESSION: a lecture whose download failed scored 100.0 because an
        absent camera was treated as 'not measured'. The probe tier always
        looks, so absent IS the finding and both media gates must FAIL."""
        r = score.evaluate({}, probe_info(camera=False, screen=False),
                           identity={"key": "no-camera"}, tiers_run=["probe"])
        self.assertIs(self._gate(r, "media_readable")["passed"], False)
        self.assertIs(self._gate(r, "has_audio")["passed"], False)
        self.assertEqual(len(r["gates_failed"]), 2)

    def test_camera_without_an_audio_stream_fails_has_audio_only(self):
        """Every timing in the pipeline comes off that track; a silent-container
        camera decodes fine and must still be refused."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(audio=False),
                           identity={"key": "mute"}, tiers_run=["probe"])
        self.assertIs(self._gate(r, "media_readable")["passed"], True)
        self.assertIs(self._gate(r, "has_audio")["passed"], False)

    def test_unrun_vision_and_speech_gates_are_skipped_not_failed(self):
        """A cheap first pass must not condemn a semester for want of looking:
        instructor_visible / intelligible / not_silent report None."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        for gate_id in ("instructor_visible", "intelligible", "not_silent"):
            with self.subTest(gate=gate_id):
                self.assertIsNone(self._gate(r, gate_id)["passed"])
                self.assertEqual(self._gate(r, gate_id)["reason"],
                                 "not measured")
        self.assertEqual(r["gates_failed"], [])

    def test_sync_gate_is_skipped_when_one_stream_is_missing(self):
        """With no screen there is nothing to align, which is not a failure."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(screen=False),
                           identity={"key": "no-screen"}, tiers_run=["probe"])
        self.assertIsNone(self._gate(r, "sync_recoverable")["passed"])

    def test_sync_gate_fails_when_the_streams_are_too_far_apart(self):
        """Beyond a quarter of the runtime they are not two views of one event
        and sync.py produces nonsense."""
        r = score.evaluate(dict(PROBE_METRICS),
                           probe_info(cam_duration=3600.0, scr_duration=7200.0),
                           identity={"key": "skew"}, tiers_run=["probe"])
        self.assertIs(self._gate(r, "sync_recoverable")["passed"], False)

    def test_duration_gate_rejects_a_fragment_and_an_overnight_recorder(self):
        """Outside 8 minutes to 4 hours it is not a lecture either way."""
        short = score.evaluate({"duration_s": 120.0},
                               probe_info(cam_duration=120.0),
                               identity={"key": "frag"}, tiers_run=["probe"])
        long_ = score.evaluate({"duration_s": 20 * 3600.0},
                               probe_info(cam_duration=20 * 3600.0),
                               identity={"key": "left-on"}, tiers_run=["probe"])
        self.assertIs(self._gate(short, "duration_sane")["passed"], False)
        self.assertIs(self._gate(long_, "duration_sane")["passed"], False)

    def test_silence_and_unintelligibility_gates_fail_on_bad_measurements(self):
        """A mic that was never switched on, and ASR that is guessing."""
        m = full_metrics("good")
        m["speech_pct"] = 0.0
        m["asr_confidence"] = -1.5
        r = score.evaluate(m, probe_info(), identity={"key": "silent"},
                           tiers_run=rubric.TIERS)
        self.assertIs(self._gate(r, "not_silent")["passed"], False)
        self.assertIs(self._gate(r, "intelligible")["passed"], False)
        self.assertEqual(r["grade"], "F")
        self.assertEqual(r["verdict"], "skip")

    def test_instructor_visibility_gate_fails_on_an_empty_lectern(self):
        """Both brand scenes are built around a person being there."""
        m = full_metrics("good")
        m["instructor_in_frame_pct"] = 5.0
        r = score.evaluate(m, probe_info(), identity={"key": "empty-lectern"},
                           tiers_run=rubric.TIERS)
        self.assertIs(self._gate(r, "instructor_visible")["passed"], False)
        self.assertIn("Instructor is in shot", r["gates_failed"])

    def test_every_rubric_gate_appears_in_the_detail(self):
        """A gate defined in the rubric but never evaluated is a check that
        silently does not exist."""
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "full"}, tiers_run=rubric.TIERS)
        self.assertEqual(sorted(g["id"] for g in r["gate_detail"]),
                         sorted(g["id"] for g in rubric.GATES))


class TestDimensions(unittest.TestCase):

    def test_unmeasured_dimension_scores_none_not_zero(self):
        """REGRESSION: a tier that did not run must not read as a zero, or a
        fast first pass ranks every lecture bottom and looks like a result."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        self.assertIsNone(r["dimensions"]["audio"]["score"])
        self.assertIsNone(r["dimensions"]["delivery"]["score"])
        self.assertEqual(r["dimensions"]["audio"]["coverage"], 0.0)

    def test_partially_measured_dimension_reports_partial_coverage(self):
        """Coverage is the honest half of a renormalised score."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        visual = r["dimensions"]["visual"]
        self.assertIsNotNone(visual["score"])
        self.assertGreater(visual["coverage"], 0.0)
        self.assertLess(visual["coverage"], 1.0)

    def test_dimensions_measured_counts_only_scored_dimensions(self):
        """report.py leans on this to say how much of the rubric was seen."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        self.assertEqual(r["dimensions_measured"], 3)
        full = score.evaluate(full_metrics("good"), probe_info(),
                              identity={"key": "full"}, tiers_run=rubric.TIERS)
        self.assertEqual(full["dimensions_measured"], len(rubric.DIMENSIONS))

    def test_every_dimension_carries_its_rubric_weight(self):
        """The weighted total renormalises over these; a missing one silently
        rescales the whole score."""
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "full"}, tiers_run=rubric.TIERS)
        for dim, meta in rubric.DIMENSIONS.items():
            with self.subTest(dimension=dim):
                self.assertAlmostEqual(r["dimensions"][dim]["weight"],
                                       meta["weight"])


class TestPotential(unittest.TestCase):

    def test_potential_is_never_below_score(self):
        """Remediation can only help; a band whose target sits the wrong side
        of an edge must read 'no improvement', not a negative one."""
        cases = [dict(PROBE_METRICS), full_metrics("good"),
                 full_metrics("mediocre"), {}]
        for i, metrics in enumerate(cases):
            with self.subTest(case=i):
                r = score.evaluate(metrics, probe_info(),
                                   identity={"key": f"c{i}"},
                                   tiers_run=rubric.TIERS)
                if r["score"] is None:
                    self.assertIsNone(r["potential"])
                else:
                    self.assertGreaterEqual(r["potential"] + 1e-9, r["score"])

    def test_potential_beats_score_on_a_remediable_lecture(self):
        """A quiet, hissy lecture is a to-do, and the gap is how the report
        distinguishes 52/78 from 52/54."""
        r = score.evaluate(full_metrics("mediocre"), probe_info(),
                           identity={"key": "fixable"}, tiers_run=rubric.TIERS)
        self.assertGreater(r["potential"], r["score"] + 1.0)

    def test_an_already_perfect_lecture_has_no_headroom(self):
        """Nothing to gain must show as nothing to gain, not a rounding lift."""
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "perfect"}, tiers_run=rubric.TIERS)
        self.assertAlmostEqual(r["potential"], r["score"], places=6)
        self.assertEqual(r["remediation"], [])


class TestRemediation(unittest.TestCase):

    def test_remediation_items_are_sorted_by_gain_and_carry_a_note(self):
        """'audio is poor' is not an instruction and 'run loudnorm' is."""
        r = score.evaluate(full_metrics("mediocre"), probe_info(),
                           identity={"key": "fixable"}, tiers_run=rubric.TIERS)
        self.assertTrue(r["remediation"])
        gains = [item["gain"] for item in r["remediation"]]
        self.assertEqual(gains, sorted(gains, reverse=True))
        for item in r["remediation"]:
            with self.subTest(metric=item["metric"]):
                self.assertGreater(item["gain"], 0.0)
                self.assertTrue(item["note"])
                self.assertIn(item["metric"], rubric.REMEDIATED)

    def test_unmeasured_metrics_are_not_offered_as_remediations(self):
        """Suggesting a denoise pass for audio nobody decoded is noise."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        for item in r["remediation"]:
            with self.subTest(metric=item["metric"]):
                self.assertIsNotNone(PROBE_METRICS.get(item["metric"]))

    def test_itemised_gains_sum_to_the_headline_gap_at_full_coverage(self):
        """The report prints '+N points' from the dimension arithmetic and then
        itemises the same N; on a complete scan the two must agree."""
        r = score.evaluate(full_metrics("mediocre"), probe_info(),
                           identity={"key": "fixable"}, tiers_run=rubric.TIERS)
        self.assertAlmostEqual(r["coverage"], 1.0, places=6)
        itemised = sum(item["gain"] for item in r["remediation"])
        self.assertAlmostEqual(itemised, r["potential"] - r["score"], places=6)

    def test_itemised_gains_track_the_headline_gap_on_a_partial_scan(self):
        """Guards the fix for the itemisation drift on partial scans.

        `gain` used to be derived from the weights by hand -- divide by the
        dimension's TOTAL metric weight, multiply by the raw DIMENSIONS weight
        -- while score/potential renormalise over the weights actually
        measured. The two agree only at full coverage, so a probe+signal scan
        printed a +14.7 headline over items summing to 7.5. Gains are now
        measured by re-scoring, so the itemisation tracks the headline.

        The match is close but not exact, and deliberately not asserted as
        exact: remediations are not additive. Each gain is measured with that
        one metric fixed, while `potential` fixes all of them at once, and two
        fixable metrics inside one dimension interact through the same
        normalisation. A 10% band is tight enough to catch the old drift --
        which was 49% -- without pretending the arithmetic is linear.
        """
        metrics = {mid: support.mediocre_value(mid)
                   for mid, spec in rubric.METRICS.items()
                   if spec["tier"] in ("probe", "signal")}
        metrics["duration_s"] = 4773.7
        metrics["speech_pct"] = 55.0
        r = score.evaluate(metrics, probe_info(), identity={"key": "signal"},
                           tiers_run=["probe", "signal"])
        self.assertLess(r["coverage"], 1.0)
        itemised = sum(item["gain"] for item in r["remediation"])
        headline = r["potential"] - r["score"]
        self.assertGreater(headline, 0.0)
        self.assertGreater(itemised, headline * 0.9)
        self.assertLess(itemised, headline * 1.2)


class TestResultShape(unittest.TestCase):

    def test_result_carries_every_field_report_py_reads(self):
        """report.py tolerates missing keys but scanner.py caches this dict, so
        a dropped field is a column that quietly empties for a whole batch."""
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "k", "dir": "d", "course": "c",
                                     "title": "t", "owner": "o",
                                     "scanned_at": "2026-01-01T00:00:00"},
                           tiers_run=["probe"], warnings=["w"], errors=["e"])
        for field in ("schema", "key", "dir", "course", "title", "owner",
                      "duration_s", "scanned_at", "tiers_run", "metrics",
                      "subscores", "dimensions", "score", "potential",
                      "grade", "verdict", "verdict_blurb", "coverage",
                      "gates_failed", "gate_detail", "remediation",
                      "dimensions_measured", "warnings", "errors"):
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_subscores_cover_exactly_the_measured_metrics(self):
        """cohort_percentiles ranks on subscores, so an absent metric leaking
        in as a subscore would give it a rank it did not earn."""
        r = score.evaluate(dict(PROBE_METRICS), probe_info(),
                           identity={"key": "probe-only"}, tiers_run=["probe"])
        for mid in r["subscores"]:
            with self.subTest(metric=mid):
                self.assertIsNotNone(PROBE_METRICS.get(mid))
        self.assertTrue(set(r["subscores"]).issubset(set(rubric.METRICS)))

    def test_duration_falls_back_to_identity_when_the_probe_has_none(self):
        """A half-downloaded lecture still has metadata.json's durationSec."""
        r = score.evaluate({}, probe_info(camera=False, screen=False),
                           identity={"key": "meta-only", "duration_s": 3000.0})
        self.assertEqual(r["duration_s"], 3000.0)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
