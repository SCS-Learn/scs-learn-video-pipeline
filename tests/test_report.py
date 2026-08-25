"""src/scan/report.py -- four renderings, and the invariants they promise.

Two rules run through the module and both are tested here:

  * nothing decides what is good -- every label, unit and threshold is read out
    of the rubric at render time;
  * a metric that was not measured is not a zero, because rendering an unrun
    vision tier as 0.0 would turn "we did not look" into "there were no
    students in the room", the exact direction of error this repo cannot
    afford.

The renderers are also the part most likely to meet a malformed result: a
lecture that errored out mid-scan is still a row, so every one of them is
exercised against an empty list and against a result whose fields are missing
or None.
"""

import csv
import io
import unittest

from tests.support import PROBE_METRICS, full_metrics, probe_info

from src.scan import report, rubric, score


def graded(key, **identity):
    """A complete, well-formed result -- the ordinary case."""
    ident = {"key": key, "course": "15-210", "owner": "Prof",
             "title": f"Lecture {key}", "scanned_at": "2026-08-21T09:00:00"}
    ident.update(identity)
    return score.evaluate(full_metrics("mediocre"), probe_info(),
                          identity=ident, tiers_run=rubric.TIERS)


def partial(key):
    """A probe-only result: high provisional score, no grade."""
    return score.evaluate(dict(PROBE_METRICS), probe_info(),
                          identity={"key": key, "course": "17-635",
                                    "owner": "Other",
                                    "scanned_at": "2026-08-21T09:00:00"},
                          tiers_run=["probe"])


def broken(key):
    """A lecture whose download failed: gates blown, nothing measured."""
    return score.evaluate({}, probe_info(camera=False, screen=False),
                          identity={"key": key}, tiers_run=["probe"],
                          errors=["camera stream missing or unreadable"])


# A result dict with every optional field missing or explicitly None. report.py
# promises to tolerate exactly this, because scanner.py can hand it one.
EMPTY_RESULT = {"key": None, "dir": None, "course": None, "title": None,
                "owner": None, "duration_s": None, "scanned_at": None,
                "tiers_run": None, "metrics": None, "subscores": None,
                "dimensions": None, "score": None, "potential": None,
                "grade": None, "verdict": None, "coverage": None,
                "gates_failed": None, "gate_detail": None,
                "remediation": None, "warnings": None, "errors": None}


class TestCohortPercentiles(unittest.TestCase):

    def test_below_the_minimum_cohort_no_percentiles_are_reported(self):
        """A rank within three samples is noise wearing the costume of
        information: with two lectures every metric reads 25th or 75th and the
        loser looks systematically bad because somebody had to be second."""
        results = report.cohort_percentiles(
            [graded("a"), graded("b"), graded("c")])
        self.assertEqual(len(results), 3)
        for r in results:
            with self.subTest(key=r["key"]):
                self.assertIsNone(r["percentiles"])

    def test_a_single_lecture_gets_no_percentiles(self):
        """--explain on one lecture must not claim a cohort rank."""
        self.assertIsNone(report.cohort_percentiles([graded("a")])[0]
                          ["percentiles"])

    def test_at_the_minimum_cohort_percentiles_appear(self):
        """MIN_COHORT is the point at which a rank starts meaning something."""
        results = report.cohort_percentiles(
            [graded(f"k{i}") for i in range(report.MIN_COHORT)])
        for r in results:
            with self.subTest(key=r["key"]):
                self.assertIsInstance(r["percentiles"], dict)
                self.assertTrue(r["percentiles"])

    def test_ties_take_the_mid_rank(self):
        """A metric every lecture aces must read 50, not have everybody
        simultaneously beating everybody."""
        results = report.cohort_percentiles(
            [graded(f"k{i}") for i in range(5)])
        for mid, pct in results[0]["percentiles"].items():
            with self.subTest(metric=mid):
                self.assertAlmostEqual(pct, 50.0)

    def test_percentiles_rank_on_the_subscore_not_the_raw_value(self):
        """Half the table is better-when-lower and several metrics are bands
        where both ends are bad, so a rank over raw numbers would put the
        quietest room and the loudest one at opposite ends of one axis."""
        results = []
        # noise_floor_dbfs is better when lower: -58 is excellent, -32 is bad.
        for i, floor in enumerate([-58.0, -50.0, -40.0, -32.0]):
            m = full_metrics("good")
            m["noise_floor_dbfs"] = floor
            results.append(score.evaluate(
                m, probe_info(), identity={"key": f"k{i}"},
                tiers_run=rubric.TIERS))
        report.cohort_percentiles(results)
        quietest = results[0]["percentiles"]["noise_floor_dbfs"]
        noisiest = results[3]["percentiles"]["noise_floor_dbfs"]
        self.assertGreater(quietest, noisiest)
        self.assertAlmostEqual(quietest, 87.5)
        self.assertAlmostEqual(noisiest, 12.5)

    def test_a_metric_measured_on_too_few_lectures_gets_no_rank(self):
        """Same treatment as a small cohort: no rank rather than a flattering
        one, so a lecture is never the '100th percentile' of itself."""
        results = [graded(f"k{i}") for i in range(3)] + [partial("thin")]
        report.cohort_percentiles(results)
        # The vision-tier metrics are measured on three of the four, which is
        # under MIN_COHORT, so nobody gets a rank for them -- not even the
        # three lectures that do have the subscore.
        self.assertIsNotNone(results[0]["subscores"].get("student_face_pct"))
        self.assertNotIn("student_face_pct", results[0]["percentiles"])
        self.assertNotIn("student_face_pct", results[3]["percentiles"])
        # duration_min is measured on all four, so it is ranked.
        self.assertIn("duration_min", results[0]["percentiles"])

    def test_an_empty_cohort_is_returned_unchanged(self):
        self.assertEqual(report.cohort_percentiles([]), [])


class TestRanking(unittest.TestCase):

    def test_a_gate_failed_lecture_sinks_below_a_clean_lower_scoring_one(self):
        """REGRESSION: the grade already says skip, and a high-scoring skip at
        the top of the table reads as a recommendation."""
        high_but_failed = {"key": "failed-high", "score": 99.0,
                           "gates_failed": ["Camera and screen both decode"]}
        clean_but_low = {"key": "clean-low", "score": 10.0,
                         "gates_failed": []}
        order = [r["key"] for r in
                 report._ranked([high_but_failed, clean_but_low])]
        self.assertEqual(order, ["clean-low", "failed-high"])

    def test_clean_lectures_sort_by_score_descending(self):
        rows = [{"key": "a", "score": 40.0}, {"key": "b", "score": 80.0},
                {"key": "c", "score": 60.0}]
        self.assertEqual([r["key"] for r in report._ranked(rows)],
                         ["b", "c", "a"])

    def test_equal_scores_break_the_tie_on_key_for_a_stable_order(self):
        """Two scans of the same semester have to diff."""
        rows = [{"key": "z", "score": 50.0}, {"key": "a", "score": 50.0}]
        self.assertEqual([r["key"] for r in report._ranked(rows)], ["a", "z"])

    def test_an_ungraded_lecture_is_not_ranked_as_a_zero_scorer_in_the_cells(self):
        """It still occupies a row -- it is work somebody has to do something
        about -- but printing a 0.0 would rank a failed scan against lectures
        that were actually measured."""
        self.assertEqual(report._cells({"score": None}), ("--", "--", "--"))

    def test_the_ranked_table_in_html_puts_failures_last(self):
        """The property has to survive the render, not just the sort."""
        results = [broken("failed-high"), graded("clean")]
        results[0]["score"] = 99.0
        html_out = report.render_html(results)
        self.assertLess(html_out.index("clean"), html_out.index("failed-high"))


class TestWeakestDimension(unittest.TestCase):

    def test_an_unmeasured_dimension_is_never_the_weakest(self):
        """An unrun tier is not a weakness of the lecture, and reporting it as
        one sends people to re-record a lecture whose problem is that nobody
        ran the vision pass."""
        weak = report.weakest_dimension(partial("thin"))
        self.assertIsNotNone(weak)
        dim, _ = weak
        self.assertGreater(partial("thin")["dimensions"][dim]["coverage"], 0.0)

    def test_nothing_measured_has_no_weakest_dimension(self):
        self.assertIsNone(report.weakest_dimension(broken("gone")))
        self.assertIsNone(report.weakest_dimension(EMPTY_RESULT))

    def test_it_picks_the_lowest_scoring_measured_dimension(self):
        m = full_metrics("good")
        for mid, spec in rubric.METRICS.items():
            if spec["dimension"] == "audio":
                m[mid] = spec["scale"][1] if spec["scale"][0] != "bool" else False
        r = score.evaluate(m, probe_info(), identity={"key": "bad-audio"},
                           tiers_run=rubric.TIERS)
        self.assertEqual(report.weakest_dimension(r)[0], "audio")


class TestRenderMarkdown(unittest.TestCase):

    def test_an_empty_scan_renders_an_explanation_not_a_traceback(self):
        out = report.render_markdown([])
        self.assertIn("No lectures scanned", out)
        self.assertTrue(out.endswith("\n"))

    def test_a_result_with_missing_and_none_fields_renders(self):
        """A lecture that errored out mid-scan is still a row."""
        out = report.render_markdown([EMPTY_RESULT, {}])
        self.assertIn("Lecture scan", out)
        self.assertIn(report.UNKNOWN, out)

    def test_the_table_carries_one_row_per_lecture(self):
        out = report.render_markdown([graded("a"), graded("b"), broken("c")])
        for key in ("a", "b", "c"):
            with self.subTest(key=key):
                self.assertIn(key, out)

    def test_limit_truncates_the_ranking_only(self):
        """The per-course and needs-attention sections always cover everything:
        truncating those would hide exactly the lecture somebody needs to see,
        which is the one at the bottom."""
        results = [graded(f"k{i}") for i in range(5)] + [broken("failing")]
        out = report.render_markdown(results, limit=2)
        self.assertIn("Top 2 of 6", out)
        self.assertIn("failing", out)

    def test_pipes_in_a_title_are_escaped_so_the_table_survives(self):
        """A Panopto title with a pipe in it would otherwise split a column."""
        r = graded("a", title="Trees | Graphs")
        out = report.render_markdown([r])
        self.assertIn("Trees \\| Graphs", out)

    def test_a_failed_gate_is_explained_with_the_rubric_s_own_reason(self):
        """Nothing in the report decides what is good; the 'why' is read out of
        rubric.GATES at render time."""
        out = report.render_markdown([broken("gone")])
        self.assertIn("Gate failed", out)
        why = next(g["why"] for g in rubric.GATES if g["id"] == "media_readable")
        self.assertIn(why[:40], out)

    def test_the_cheapest_wins_section_lists_remediations_with_their_notes(self):
        """'run loudnorm' is an instruction and 'audio is poor' is not."""
        out = report.render_markdown([graded("fixable")])
        self.assertIn("Cheapest wins", out)
        self.assertIn("loudnorm", out)

    def test_a_scan_with_nothing_to_fix_says_so(self):
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "perfect"}, tiers_run=rubric.TIERS)
        self.assertIn("Nothing measurably improvable", report.render_markdown([r]))

    def test_verdict_counts_include_the_ones_the_scorer_added(self):
        """score.py emits `unscanned` and `incomplete`, which are not in
        GRADES; a summary that silently drops the lectures that failed to scan
        is the summary that gets believed."""
        out = report.render_markdown([graded("a"), partial("b"), broken("c")])
        self.assertIn("incomplete", out)
        self.assertIn("unscanned", out)


class TestRenderCsv(unittest.TestCase):

    def test_an_empty_scan_still_emits_the_header_row(self):
        """Two scans of different semesters have to concatenate."""
        rows = list(csv.reader(io.StringIO(report.render_csv([]))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "key")

    def test_a_result_with_missing_and_none_fields_renders(self):
        out = report.render_csv([EMPTY_RESULT, {}])
        self.assertEqual(len(list(csv.reader(io.StringIO(out)))), 3)

    def test_column_order_is_the_rubric_s_order(self):
        """So that two scans of different tiers still diff."""
        header = next(csv.reader(io.StringIO(report.render_csv([]))))
        self.assertEqual(header[-len(rubric.METRICS):], list(rubric.METRICS))

    def test_an_unmeasured_metric_is_an_empty_cell_not_a_zero(self):
        """A spreadsheet reads a blank as missing and a 0 as a measurement, and
        here the difference decides whether somebody re-records a lecture."""
        rows = list(csv.DictReader(io.StringIO(report.render_csv(
            [partial("thin")]))))
        self.assertEqual(rows[0]["student_face_pct"], "")
        self.assertNotEqual(rows[0]["duration_min"], "")

    def test_an_ungraded_lecture_leaves_score_blank(self):
        rows = list(csv.DictReader(io.StringIO(report.render_csv(
            [broken("gone")]))))
        self.assertEqual(rows[0]["score"], "")
        self.assertEqual(rows[0]["potential"], "")

    def test_booleans_are_written_as_one_and_zero(self):
        """A spreadsheet cannot filter on the word True reliably."""
        rows = list(csv.DictReader(io.StringIO(report.render_csv(
            [graded("a")]))))
        self.assertIn(rows[0]["has_opening"], ("0", "1"))

    def test_rows_are_newline_terminated_for_diffability(self):
        """The return value is written straight to a file next to the batch."""
        out = report.render_csv([graded("a")])
        self.assertNotIn("\r", out)


class TestRenderHtml(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.out = report.render_html(
            [graded("a"), graded("b"), partial("thin"), broken("gone")])

    def test_an_empty_scan_renders_a_complete_document(self):
        out = report.render_html([])
        self.assertTrue(out.startswith("<!doctype html>"))
        self.assertIn("Nothing scanned", out)
        self.assertIn("</html>", out)

    def test_a_result_with_missing_and_none_fields_renders(self):
        out = report.render_html([EMPTY_RESULT, {}])
        self.assertIn("</html>", out)
        self.assertIn(report.UNKNOWN, out)

    def test_all_three_theme_blocks_are_present(self):
        """Light, the prefers-color-scheme dark override, and the explicit
        data-theme override for anyone printing or embedding it."""
        self.assertIn(":root {", self.out)
        self.assertIn("@media (prefers-color-scheme: dark)", self.out)
        self.assertIn(':root:not([data-theme="light"])', self.out)
        self.assertIn(':root[data-theme="dark"]', self.out)

    def test_the_page_fetches_nothing_from_a_network(self):
        """It is opened months later off a file:// path or out of a tarball, so
        anything it needed from a network would eventually render it blank."""
        self.assertNotIn("http://", self.out)
        self.assertNotIn("https://", self.out)
        self.assertNotIn("//cdn", self.out)
        self.assertNotIn("@import", self.out)

    def test_the_page_carries_no_script(self):
        """Stated in the docstring and in the note under the ranked table, and
        it is what makes the file safe to archive and re-open."""
        self.assertNotIn("<script", self.out.lower())
        self.assertNotIn("onclick", self.out.lower())

    def test_titles_and_keys_are_html_escaped(self):
        """Panopto titles are arbitrary text and land straight in the page."""
        out = report.render_html([graded("a", title="Trees <b>&</b> Graphs")])
        self.assertIn("&lt;b&gt;", out)
        self.assertNotIn("<b>&</b>", out)

    def test_an_incomplete_scan_gets_the_neutral_chip_not_the_failing_one(self):
        """An incomplete scan looking like a failing lecture is how a good
        lecture gets dropped."""
        self.assertIn('chip-x', report.render_html([partial("thin")]))
        self.assertNotIn('chip-?', report.render_html([partial("thin")]))

    def test_an_unmeasured_dimension_is_marked_unmeasured(self):
        """Never a full-width bar at zero, so the eye does not read absent as
        bad."""
        self.assertIn("dim unmeasured", report.render_html([partial("thin")]))
        self.assertIn("not measured", report.render_html([partial("thin")]))

    def test_dimension_labels_come_from_the_rubric(self):
        """A retuned rubric changes this page without a line changing here."""
        for meta in rubric.DIMENSIONS.values():
            with self.subTest(label=meta["label"]):
                self.assertIn(report._esc(meta["label"]), self.out)

    def test_every_lecture_appears_in_the_cards(self):
        for key in ("a", "b", "thin", "gone"):
            with self.subTest(key=key):
                self.assertIn(key, self.out)


class TestRenderLecture(unittest.TestCase):

    def test_no_result_at_all_says_so(self):
        self.assertEqual(report.render_lecture(None), "No result to explain.\n")
        self.assertEqual(report.render_lecture({}), "No result to explain.\n")

    def test_a_result_with_missing_and_none_fields_renders(self):
        out = report.render_lecture(EMPTY_RESULT)
        self.assertIn("GATES", out)
        self.assertIn("DIMENSIONS", out)
        self.assertIn("METRICS", out)

    def test_every_rubric_metric_is_listed_measured_or_not(self):
        """--explain is what prints when somebody disagrees with a grade, so a
        metric missing from it is a number nobody can argue with."""
        out = report.render_lecture(graded("a"))
        for spec in rubric.METRICS.values():
            with self.subTest(label=spec["label"]):
                self.assertIn(spec["label"][:28], out)

    def test_an_unmeasured_metric_prints_not_measured_rather_than_zero(self):
        out = report.render_lecture(partial("thin"))
        self.assertIn("not measured", out)

    def test_the_missing_cohort_is_called_out_rather_than_left_blank(self):
        """An empty percentiles dict means this lecture measured nothing
        rankable; None means the scan was too small to rank anything at all.
        Only the second is a caveat."""
        no_cohort = report.render_lecture(graded("a"))
        self.assertIn("no cohort ranks", no_cohort)
        with_cohort = report.cohort_percentiles(
            [graded(f"k{i}") for i in range(report.MIN_COHORT)])
        self.assertNotIn("no cohort ranks",
                         report.render_lecture(with_cohort[0]))

    def test_percentiles_are_printed_when_the_cohort_supports_them(self):
        results = report.cohort_percentiles(
            [graded(f"k{i}") for i in range(report.MIN_COHORT)])
        self.assertIn(" p", report.render_lecture(results[0]))

    def test_a_failed_gate_is_shown_with_the_rubric_s_reason(self):
        out = report.render_lecture(broken("gone"))
        self.assertIn("FAILED", out)
        why = next(g["why"] for g in rubric.GATES if g["id"] == "media_readable")
        self.assertIn(why.split(".")[0], out)

    def test_a_lecture_with_nothing_to_fix_says_so(self):
        r = score.evaluate(full_metrics("good"), probe_info(),
                           identity={"key": "perfect"}, tiers_run=rubric.TIERS)
        self.assertIn("nothing the pipeline can fix", report.render_lecture(r))

    def test_output_stays_within_the_requested_width(self):
        """It is printed to a terminal and pasted into tickets; a wrapped line
        makes the metric table unreadable."""
        out = report.render_lecture(graded("a", title="x" * 200), width=78)
        overlong = [line for line in out.splitlines()
                    if len(line) > 78 and not line.startswith("x")]
        self.assertEqual(overlong, [])


class TestFormatters(unittest.TestCase):

    def test_runtime_formats_as_hours_and_minutes(self):
        self.assertEqual(report._hms(4773.7), "1 h 19 m")
        self.assertEqual(report._hms(600), "10 m")
        self.assertEqual(report._hms(None), "--")
        self.assertEqual(report._hms(0), "--")

    def test_scores_always_carry_one_decimal(self):
        """A column where 46.0 prints as 46 and 45.9 prints as 45.9 stops being
        a column, and these are the numbers people compare down the page."""
        self.assertEqual(report._pts(46.0), "46.0")
        self.assertEqual(report._pts(45.94), "45.9")
        self.assertEqual(report._pts(None), "--")

    def test_raw_values_carry_their_unit_from_the_rubric(self):
        self.assertEqual(report._fmt_raw("snr_db", None), "not measured")
        self.assertIn(rubric.METRICS["snr_db"]["unit"],
                      report._fmt_raw("snr_db", 18.0))

    def test_bool_metrics_render_as_yes_and_no(self):
        self.assertEqual(report._fmt_raw("has_opening", True), "yes")
        self.assertEqual(report._fmt_raw("has_opening", False), "no")

    def test_an_unknown_metric_id_does_not_raise(self):
        """CSV and card rendering both look metrics up by id from a cached
        result, which may predate a rubric edit."""
        self.assertEqual(report._fmt_raw("no_such_metric", 3.0), "3")

    def test_a_missing_key_falls_back_to_the_directory_then_to_unknown(self):
        self.assertEqual(report._key({"key": "k", "dir": "d"}), "k")
        self.assertEqual(report._key({"dir": "d"}), "d")
        self.assertEqual(report._key({}), report.UNKNOWN)

    def test_a_long_title_is_truncated_but_the_key_survives(self):
        """The key is what lets someone match a row back to a directory."""
        name = report._name({"key": "k", "title": "T" * 200})
        self.assertIn("`k`", name)
        self.assertLess(len(name), 200)

    def test_the_gain_is_never_negative(self):
        """A rounding artefact in that direction would sort to the top of
        'cheapest wins'."""
        self.assertEqual(report._gain({"score": 50.0, "potential": 49.0}), 0.0)

    def test_means_run_over_graded_lectures_only(self):
        """Folding an ungraded lecture in as a zero would report a scan that
        did not finish as a lecturer who did badly."""
        results = [{"score": 80.0}, {"score": None}, {"score": 60.0}]
        self.assertAlmostEqual(report._mean_score(results), 70.0)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
