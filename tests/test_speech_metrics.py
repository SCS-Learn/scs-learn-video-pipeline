"""src/scan/speech_metrics.py -- pacing, structure, and who was speaking.

The transcript is the privacy surface: audio.py mutes against these speaker
labels, cards.py renders from these spans and captions.py redacts against them.
The module's own docstring records that lecture 12 had five segments whose
`speaker` field said instructor while the words inside did not, so the
word-level split is tested directly rather than trusted.

Every transcript here is hand-built, so each expected number is arithmetic
anyone can check in the docstring of the test that asserts it.
"""

import json
import os
import tempfile
import unittest

from tests.support import words

from src.scan import speech_metrics as S

INST = "SPEAKER_00"
STUDENT = "SPEAKER_01"


def reference_transcript():
    """Three segments over a 200 s lecture.

        0-60      instructor, 120 words, ends in a tag question
        60-80     student, 20 words
        80-100    nothing at all          -> 20 s of dead air
        100-160   instructor, 120 words, ends in a real question
        160-200   nothing at all          -> 40 s of tail dead air
    """
    return [
        {"start": 0.0, "end": 60.0, "speaker": INST, "avg_logprob": -0.3,
         "text": "So we get a linear bound, right?",
         "words": words(120, 0.0, 60.0, INST)},
        {"start": 60.0, "end": 80.0, "speaker": STUDENT, "avg_logprob": -0.9,
         "text": "That is why I raised my hand.",
         "words": words(20, 60.0, 80.0, STUDENT)},
        {"start": 100.0, "end": 160.0, "speaker": INST, "avg_logprob": -0.2,
         "text": "What is the trade-off?",
         "words": words(120, 100.0, 160.0, INST)},
    ]


class TestPhraseCount(unittest.TestCase):

    def test_a_multi_word_phrase_counts_once(self):
        """'you know' is one filler, not two."""
        self.assertEqual(S._phrase_count("well you know it works",
                                         ["you know"]), 1)

    def test_overlapping_phrases_do_not_double_count(self):
        """Longest-first with consumption is the whole design: 'you know' and
        'know' both match the same four letters and only one may be counted."""
        self.assertEqual(
            S._phrase_count("you know what i mean",
                            ["you know", "know", "i mean", "mean"]), 2)

    def test_repeated_phrases_are_all_counted(self):
        """Consuming a match must not consume the ones after it."""
        self.assertEqual(
            S._phrase_count("um and um and um", ["um"]), 3)

    def test_matching_is_on_word_boundaries(self):
        """'um' must not fire inside 'number', or the filler metric measures
        the vocabulary of the subject rather than the speaker."""
        self.assertEqual(S._phrase_count("a large number of umbrellas",
                                         ["um"]), 0)

    def test_no_phrases_present_is_zero_not_an_error(self):
        self.assertEqual(S._phrase_count("a clean sentence", ["um", "uh"]), 0)
        self.assertEqual(S._phrase_count("", ["um"]), 0)


class TestTagQuestions(unittest.TestCase):

    def test_a_trailing_right_is_a_tag_not_a_question_to_the_class(self):
        """REGRESSION: counting every question-final segment read ~160 questions
        an hour on both references, because three quarters of them were the
        word 'right'."""
        self.assertTrue(S._TAG_RX.search("So we get a linear bound, right?"))

    def test_a_real_question_is_not_a_tag(self):
        """'What is the trade-off?' is checking for understanding."""
        self.assertIsNone(S._TAG_RX.search("What is the trade-off?"))
        self.assertIsNone(S._TAG_RX.search("Can anyone confirm this bound?"))

    def test_several_tag_forms_are_recognised(self):
        """The lexicon lists them; the regex has to accept each one."""
        for text in ("This is linear, okay?", "We push it down, yeah?",
                     "That halves the work, make sense?",
                     "The invariant holds, correct?"):
            with self.subTest(text=text):
                self.assertIsNotNone(S._TAG_RX.search(text))

    def test_a_tag_word_used_mid_sentence_is_not_a_tag(self):
        """'right' is an ordinary word in a technical lecture -- 'the right
        subtree' -- and the anchor is what keeps it out of the count."""
        self.assertIsNone(S._TAG_RX.search("Which is the right subtree?"))

    def test_a_statement_ending_in_a_tag_word_without_a_question_mark_is_not(self):
        """The metric counts tag *questions*, not the word."""
        self.assertIsNone(S._TAG_RX.search("So we get a linear bound, right."))


class TestInstructorLabel(unittest.TestCase):

    def test_picks_the_speaker_with_the_most_talking_time(self):
        """Same rule as transcription.py; getting it backwards mutes the
        instructor and publishes the students."""
        self.assertEqual(S.instructor_label(reference_transcript()), INST)

    def test_talking_time_wins_over_segment_count(self):
        """A student who interjects ten times briefly is not the instructor."""
        segs = [{"start": 0.0, "end": 600.0, "speaker": INST}]
        segs += [{"start": 600.0 + i, "end": 601.0 + i, "speaker": STUDENT}
                 for i in range(20)]
        self.assertEqual(S.instructor_label(segs), INST)

    def test_no_speaker_labels_at_all_returns_none(self):
        """A transcript with no diarization must degrade, not raise."""
        self.assertIsNone(S.instructor_label(
            [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]))

    def test_empty_transcript_returns_none(self):
        self.assertIsNone(S.instructor_label([]))


class TestMeasure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = S.measure(reference_transcript(), 200.0)

    def test_speech_rate_is_words_over_speaking_time_not_wall_clock(self):
        """240 instructor words over 120 s of instructor speech = 120 wpm.
        Measured over speaking time so pauses do not drag it down."""
        self.assertAlmostEqual(self.m["speech_rate_wpm"], 120.0, places=6)

    def test_dead_air_counts_gaps_of_five_seconds_or_more(self):
        """20 s between segments plus a 40 s tail = 60 s of a 200 s lecture."""
        self.assertAlmostEqual(self.m["dead_air_pct"], 30.0, places=6)

    def test_dead_air_includes_the_tail_after_the_last_word(self):
        """A recorder left running past the end is dead air the viewer sits
        through, and it is invisible to a gaps-between-segments count."""
        self.assertAlmostEqual(self.m["longest_dead_air_s"], 40.0, places=6)
        no_tail = S.measure(reference_transcript(), 160.0)
        self.assertAlmostEqual(no_tail["dead_air_pct"], 20.0 / 160.0 * 100.0,
                               places=6)
        self.assertAlmostEqual(no_tail["longest_dead_air_s"], 20.0, places=6)

    def test_a_gap_under_the_threshold_is_not_dead_air(self):
        """Some silence is teaching; DEAD_AIR_MIN_S is where it stops being."""
        segs = [{"start": 0.0, "end": 60.0, "speaker": INST, "text": "a",
                 "words": words(120, 0.0, 60.0, INST)},
                {"start": 63.0, "end": 120.0, "speaker": INST, "text": "b",
                 "words": words(120, 63.0, 120.0, INST)}]
        self.assertAlmostEqual(S.measure(segs, 120.0)["dead_air_pct"], 0.0)

    def test_student_speech_is_a_share_of_speech_time_not_of_runtime(self):
        """20 s of student against 140 s of speech = 14.3%. Every second of it
        gets muted and, where it is a question, replaced with a card."""
        self.assertAlmostEqual(self.m["student_speech_pct"],
                               20.0 / 140.0 * 100.0, places=6)

    def test_a_run_of_student_segments_is_one_turn(self):
        """One long question is one turn, not five, or interaction_per_hour
        measures how the diarizer chose to segment rather than the room."""
        segs = reference_transcript()
        segs.insert(2, {"start": 80.0, "end": 90.0, "speaker": STUDENT,
                        "text": "and also", "words": words(5, 80.0, 90.0,
                                                           STUDENT)})
        m = S.measure(segs, 200.0)
        self.assertEqual(m["student_turns"], 1)

    def test_interaction_is_reported_per_hour(self):
        """One turn in a 200 s lecture is 18 an hour."""
        self.assertAlmostEqual(self.m["interaction_per_hour"], 18.0, places=6)

    def test_class_questions_exclude_tag_questions(self):
        """Two question-final instructor segments, one of them a tag: only the
        real one counts, so 1 in 200 s = 18 an hour."""
        self.assertEqual(self.m["tag_question_count"], 1)
        self.assertAlmostEqual(self.m["class_question_per_hour"], 18.0,
                               places=6)
        self.assertAlmostEqual(self.m["tag_question_per_hour"], 18.0, places=6)

    def test_asr_confidence_is_the_mean_and_the_poor_share_is_a_percentage(self):
        """Segments at -0.3, -0.9 and -0.2; one of three below -0.6."""
        self.assertAlmostEqual(self.m["asr_confidence"],
                               (-0.3 - 0.9 - 0.2) / 3.0, places=6)
        self.assertAlmostEqual(self.m["asr_poor_segment_pct"],
                               100.0 / 3.0, places=6)

    def test_speaker_shape_is_reported_alongside_the_estimates(self):
        """Diarization mixes speakers on small classes, so the reader needs to
        see the shape of it to know when to distrust the burden metrics."""
        self.assertEqual(self.m["instructor_label"], INST)
        self.assertEqual(self.m["speaker_count"], 2)
        self.assertEqual(self.m["segment_count"], 3)
        self.assertEqual(self.m["word_count"], 260)
        self.assertAlmostEqual(self.m["word_timing_coverage"], 100.0)

    def test_an_empty_transcript_measures_nothing(self):
        """A lecture with no transcript reports lower coverage, not zeros."""
        self.assertEqual(S.measure([], 200.0), {})


class TestWordLevelSpeakerLabels(unittest.TestCase):
    """Word labels must beat the segment label.

    captions.py learned this the hard way: five of lecture 12's fifteen
    redacted cues were segments whose speaker field said instructor while words
    inside did not. A metric that trusts the segment republishes exactly what
    the audio pass removed.
    """

    @staticmethod
    def _mislabelled():
        # Both segments are labelled instructor, but half the words are not.
        return [
            {"start": 0.0, "end": 100.0, "speaker": INST, "text": "a b",
             "words": words(10, 0.0, 100.0, INST)},
            {"start": 100.0, "end": 120.0, "speaker": INST, "text": "c d",
             "words": words(10, 100.0, 120.0, "SPEAKER_99")},
        ]

    def test_instructor_word_share_uses_the_word_labels(self):
        """Ten of twenty words are the instructor's, so the share is 50% --
        trusting the segment field would report 100%."""
        m = S.measure(self._mislabelled(), 200.0)
        self.assertEqual(m["word_count"], 20)
        self.assertAlmostEqual(m["instructor_word_share"], 50.0, places=6)

    def test_words_with_no_speaker_field_fall_back_to_their_segment(self):
        """Not every transcript carries per-word labels; the fallback has to be
        the segment, not 'unknown'."""
        segs = [{"start": 0.0, "end": 60.0, "speaker": INST, "text": "x",
                 "words": [{"word": f"w{i}", "start": i, "end": i + 1}
                           for i in range(30)]}]
        m = S.measure(segs, 60.0)
        self.assertAlmostEqual(m["instructor_word_share"], 100.0)

    def test_a_segment_with_no_words_falls_back_to_its_text(self):
        """A transcript with no word timings must still produce counts, and say
        so through word_timing_coverage rather than degrading silently."""
        segs = [{"start": 0.0, "end": 60.0, "speaker": INST,
                 "text": "one two three four five"}]
        m = S.measure(segs, 60.0)
        self.assertEqual(m["word_count"], 5)
        self.assertAlmostEqual(m["word_timing_coverage"], 100.0)

    def test_student_speech_is_counted_per_word_not_per_segment(self):
        """Guards the fix for burden metrics reading the segment speaker.

        The module docstring has always said every speaker-conditioned metric
        counts words, not segments, and `student_speech_pct`, `student_turns`
        and `interaction_per_hour` did not: they read `s['speaker']`, the very
        field captions.py caught lying -- five of lecture 12's fifteen
        redacted cues were instructor-labelled segments with non-instructor
        words inside.

        The failure was one-directional and reassuring, which is the dangerous
        kind: a lecture needing muting and question cards reported zero
        anonymization burden while instructor_word_share correctly said 50%.
        Student exposure is the heaviest signal in this rubric, so it must
        fail towards reporting exposure, never towards hiding it.
        """
        m = S.measure(self._mislabelled(), 200.0)
        self.assertAlmostEqual(m["instructor_word_share"], 50.0, places=6)
        self.assertGreater(m["student_speech_pct"], 0.0)
        self.assertGreater(m["student_turns"], 0)


class TestTextDerivedMetrics(unittest.TestCase):

    @staticmethod
    def _long_lecture(extra_text="", opening="", closing=""):
        """A 3600 s lecture with enough instructor words to clear the n>=100
        floor that guards the text-derived metrics."""
        body = " ".join(f"traversal{i % 40} vertex edge" for i in range(60))
        segs = [
            {"start": 0.0, "end": 60.0, "speaker": INST,
             "text": f"{opening} {body}".strip(),
             "words": words(240, 0.0, 60.0, INST, stem="alpha")},
            {"start": 60.0, "end": 3000.0, "speaker": INST,
             "text": f"{body} {extra_text}".strip(),
             "words": words(600, 60.0, 3000.0, INST, stem="beta")},
            {"start": 3400.0, "end": 3600.0, "speaker": INST,
             "text": f"{closing} {body}".strip(),
             "words": words(200, 3400.0, 3600.0, INST, stem="gamma")},
        ]
        return segs

    def test_opening_cues_are_only_looked_for_at_the_start(self):
        """A lecture that says where it is going survives being watched out of
        sequence, which is how open courseware gets watched."""
        with_open = S.measure(self._long_lecture(opening="today we will cover"),
                              3600.0)
        without = S.measure(self._long_lecture(), 3600.0)
        self.assertTrue(with_open["has_opening"])
        self.assertFalse(without["has_opening"])

    def test_a_closing_cue_outside_the_closing_window_does_not_count(self):
        """'to summarise' in the first minute is a signpost, not a close."""
        late = S.measure(self._long_lecture(closing="to summarise"), 3600.0)
        early = S.measure(self._long_lecture(opening="to summarise"), 3600.0)
        self.assertTrue(late["has_closing"])
        self.assertFalse(early["has_closing"])

    def test_named_mentions_count_distinct_people_not_total_mentions(self):
        """One student named six times is one person's privacy at stake, not
        six -- and a recurring capitalised term is a technical noun, not a
        name, which is what NAME_MAX_OCCURRENCES filters out."""
        segs = [{"start": 0.0, "end": 3600.0, "speaker": INST,
                 "words": words(200, 0.0, 3600.0, INST),
                 "text": ("Today we ask Priya. " * 3
                          + "Today we apply Boruvka. " * 10)}]
        m = S.measure(segs, 3600.0)
        self.assertIn("Priya", m["named_candidates"])
        self.assertNotIn("Boruvka", m["named_candidates"])
        self.assertAlmostEqual(m["named_mentions_per_hour"],
                               float(len(m["named_candidates"])), places=6)

    def test_a_sentence_initial_capital_is_not_a_name(self):
        """Every sentence has one, so counting them makes the tripwire useless."""
        segs = [{"start": 0.0, "end": 3600.0, "speaker": INST,
                 "words": words(200, 0.0, 3600.0, INST),
                 "text": "Trees are useful. Graphs are harder. Sorting is fine."}]
        self.assertEqual(S.measure(segs, 3600.0)["named_candidates"], [])

    def test_flagged_questions_are_counted_off_the_transcript_flag(self):
        """cards.py renders one card per flagged span, so the count is the
        privacy work this lecture implies."""
        segs = reference_transcript()
        segs[1]["is_student_question"] = True
        self.assertEqual(S.measure(segs, 200.0)["flagged_questions"], 1)


class TestLoadTranscript(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="scan-transcript-test-")

    def _write(self, name, payload, raw=None):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as fh:
            fh.write(raw if raw is not None else json.dumps(payload))
        return path

    def test_segments_come_back_sorted_by_start(self):
        """Dead air and turn counting both walk the list in order."""
        path = self._write("t.json", [
            {"start": 10.0, "end": 20.0, "text": "b"},
            {"start": 0.0, "end": 5.0, "text": "a"}])
        segs = S.load_transcript(path)
        self.assertEqual([s["text"] for s in segs], ["a", "b"])

    def test_segments_without_timings_are_dropped(self):
        """Every timing downstream indexes on start/end; a segment missing one
        would raise halfway through a semester scan."""
        path = self._write("t.json", [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"text": "no timings"}])
        self.assertEqual(len(S.load_transcript(path)), 1)

    def test_corrupt_or_missing_or_empty_transcripts_return_none(self):
        """A semester scan meets at least one truncated file, and the right
        answer is a partial record, not a traceback."""
        self.assertIsNone(S.load_transcript(None))
        self.assertIsNone(S.load_transcript(
            os.path.join(self.tmp, "absent.json")))
        self.assertIsNone(S.load_transcript(
            self._write("bad.json", None, raw="[{not json")))
        self.assertIsNone(S.load_transcript(self._write("empty.json", [])))
        self.assertIsNone(S.load_transcript(
            self._write("dict.json", {"segments": []})))


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
