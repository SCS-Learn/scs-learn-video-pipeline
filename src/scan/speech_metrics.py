"""Transcript measurement: pacing, structure, interaction, and what it costs.

Reads `transcript_classified.json` -- the same file src/audio/audio.py mutes
against and src/audio/cards.py renders from -- so the scanner's view of who
spoke when is exactly the pipeline's view. Nothing is re-derived.

Two things about that file drive the design here:

**Speaker labels are per WORD, not per segment.** captions.py already learned
this the hard way: five of lecture 12's fifteen redacted cues were segments
whose speaker field said instructor while words inside did not. Every
speaker-conditioned metric below therefore counts words, not segments.

**Diarization mixes speakers on small classes.** On 17-635 lecture 13 one
SPEAKER_02 label covers both the instructor's asides and genuine student
speech. So `student_speech_pct` and `interaction_per_hour` are honest about
being estimates, and the scanner reports the diarization's own shape
(`speaker_count`, `instructor_word_share`) alongside them so a reader can see
when to distrust them.

The instructor is identified the same way transcription.py does it -- the
speaker with the most talking time. That function is reimplemented here in six
lines rather than imported, because src/audio/transcription.py imports whisperx
at module scope and the scanner must stay runnable without the ML stack.
"""

import json
import os
import re

import numpy as np

from src.scan.lexicon import (ADMIN_TERMS, CLOSING_CUES, FILLERS, NOT_NAMES,
                              OPENING_CUES, SIGNPOSTS, STOPWORDS, TAG_QUESTIONS)

DEAD_AIR_MIN_S = 5.0
POOR_SEGMENT_LOGPROB = -0.6
OPENING_WINDOW_S = 180.0
CLOSING_WINDOW_S = 300.0
TTR_WINDOW = 500
TOPIC_TOP_N = 60

# A capitalised token occurring more often than this across one lecture is a
# recurring technical term, not somebody who got called on. Students get named
# once or twice; LangChain gets named forty times. On 15-210 lecture 12 this
# leaves exactly four candidates, all of them real names.
NAME_MAX_OCCURRENCES = 4

# "This word never appears lower-case, so it is not an ordinary word" is a
# statistical argument, and it needs a sample. On three sentences about trees
# and graphs, "Trees" and "Graphs" never appear lower-case either. Below this
# many words of transcript there is no evidence to argue from and the
# sentence-initial rule in _name_candidates is switched off entirely. A
# lecture runs 8,000-13,000 words; anything under a thousand is a fragment, a
# fixture, or a transcription that failed.
NAME_RULE2_MIN_WORDS = 1000

_WORD_RX = re.compile(r"[a-z0-9']+")
_CAPS_RX = re.compile(r"\b([A-Z][a-z]{2,})\b")
# The same token shape, already lower-case. A word that occurs lower-case
# somewhere in the lecture is an ordinary word, whatever else it opens.
_LOWER_RX = re.compile(r"\b[a-z]{3,}\b")
# What may sit in front of a word and still leave it sentence-initial: the
# quotes and brackets Whisper puts round reported speech, and a lead dash.
_SENTENCE_LEAD = " \t\"'([-—‘“"
_TAG_RX = re.compile(
    r"(?:^|[,;]\s*)(?:" + "|".join(re.escape(t) for t in TAG_QUESTIONS) + r")\s*\?$",
    re.IGNORECASE)


def load_transcript(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            segs = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(segs, list) or not segs:
        return None
    return sorted((s for s in segs if "start" in s and "end" in s),
                  key=lambda s: s["start"])


def instructor_label(segments):
    """Speaker with the most talking time. Same rule as transcription.py."""
    totals = {}
    for s in segments:
        spk = s.get("speaker")
        if spk is None:
            continue
        totals[spk] = totals.get(spk, 0.0) + (s["end"] - s["start"])
    return max(totals, key=totals.get) if totals else None


def _tokens(text):
    return _WORD_RX.findall((text or "").lower())


def _phrase_count(haystack, phrases):
    """Occurrences of any phrase, longest first so overlaps count once."""
    total = 0
    scratch = haystack
    for p in sorted(phrases, key=len, reverse=True):
        pattern = r"\b" + re.escape(p) + r"\b"
        found = len(re.findall(pattern, scratch))
        if found:
            total += found
            scratch = re.sub(pattern, " ", scratch)
    return total


def _name_candidates(segments):
    """Distinct capitalised tokens that look like somebody's name.

    A review flag, not a classifier -- weighted 0.75, and the candidate list
    is printed so a human decides. So it is tuned for recall: a name it misses
    is a name nobody looks at, while a false positive costs one glance.

    The rule used to be "capitalised, and not the first word of the sentence",
    which threw away the highest-risk pattern it has. Vocative address --
    "Bradley, go ahead", "Fanny?", "Danny, why is it a value?" -- is the
    commonest way an instructor names a student, and it is ALWAYS
    sentence-initial. On the two reference transcripts every one of those was
    discarded. The skip was also implemented as "drop the first regex match",
    which is not the same thing: _CAPS_RX needs three letters, so in "So
    Bradley showed us" the first match IS Bradley, mid-sentence, and it went
    too.

    Simply dropping the skip surfaces ~64 extra tokens a lecture, nearly all
    ordinary sentence openers (Amazing, Anything, Cool) that no technical
    stoplist covers. Two rules replace it:

    1. Capitalised mid-sentence. What fired before, except that position is
       now decided by what precedes the match in the sentence rather than by
       match order. That alone recovers "Swarna" on 17-635 lecture 13 ("Or you
       may have questions or complaints regarding Swarna, right?" -- "Or" is
       two letters, so Swarna was the first MATCH and went) and "Chonk" on
       15-210 lecture 12.
    2. Capitalised at the start of a sentence, and the word never appears
       lower-case anywhere else in the lecture. That second half is the whole
       filter, and it needs no word list: an ordinary word usually turns up
       lower-case somewhere in ten thousand words of speech, and a person's
       name essentially never does. It is a statistical argument, so it
       tightens as a lecture gets longer and has no force at all on a short
       one -- hence NAME_RULE2_MIN_WORDS, below which rule 2 is off. It is
       also the rule that leaks: a word this particular lecture happened
       never to use lower-case passes it, which is most of the false
       positives counted below.

    Rule 2 deliberately does NOT require a following comma or question mark.
    That form is the classic vocative -- "Danny, why is it a value?" -- but on
    the reference transcripts it catches nothing rule 1 did not already have,
    while the two names that were genuinely lost are "Fanny doesn't know
    either." and "Bradley also showed us how it works.", where the name is the
    subject of an ordinary sentence.

    NAME_MAX_OCCURRENCES still drops recurring technical terms. It is applied
    to whichever rule admitted the token, so a token that qualified under
    rule 1 before is counted exactly as it was and cannot now be pushed over
    the cap by its sentence-initial uses.

    Rule 2 additionally ignores a sentence of one or two words closed with a
    full stop, which is a slide title read aloud rather than anything said to
    a person; a two-word sentence closed with a question mark is kept, since
    that is what "Fanny?" looks like.

    PRECISION, MEASURED, on the two reference transcripts: 12 of 40
    candidates are plausibly people, 30%, against 8 of 13 (62%) before. Recall
    is what improved and what was wanted: 8 real names to 12, including every
    one the reviewer's cases named. The 28 false positives are ordinary
    English words that this particular lecture happened never to use
    lower-case -- Especially, Helps, Reading, Whereas, Determinism, Oops --
    and they are a LEXICON problem, not a structural one: nothing inside a
    single transcript distinguishes a once-spoken "Bradley" from a
    once-spoken "Mirror". The stoplist in lexicon.py is where that gets fixed,
    the same way NOT_NAMES already absorbs Whisper's product nouns. Until it
    does, read named_mentions_per_hour as an upper bound and the candidate
    list as the actual output.
    """
    lower_seen = set()
    mid, initial = {}, {}
    n_words = 0
    for s in segments:
        text = s.get("text") or ""
        n_words += len(text.split())
        lower_seen.update(_LOWER_RX.findall(text))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for mo in _CAPS_RX.finditer(sentence):
                tok = mo.group(1)
                lead = sentence[:mo.start()].strip(_SENTENCE_LEAD)
                if lead:
                    mid[tok] = mid.get(tok, 0) + 1
                    continue
                # A one- or two-word sentence closed with a full stop is a
                # slide title read off the screen -- "Cost." "Movie ticket."
                # "Determinism." -- and 17-635 lecture 13 alone has five.
                # Closed with a question mark it is the opposite thing, and
                # the highest-risk one there is: "Fanny?"
                body = sentence.strip()
                if len(body.split()) <= 2 and not body.endswith(("?", "!")):
                    continue
                initial[tok] = initial.get(tok, 0) + 1

    out = []
    for tok in set(mid) | set(initial):
        low = tok.lower()
        if low in NOT_NAMES or low in STOPWORDS:
            continue
        if tok in mid:
            seen = mid[tok]                     # rule 1
        elif low in lower_seen or n_words < NAME_RULE2_MIN_WORDS:
            continue
        else:
            seen = initial[tok]                 # rule 2
        if seen <= NAME_MAX_OCCURRENCES:
            out.append(tok)
    return sorted(out)


def _instructor_sentences(segments, inst):
    """(sentence_text, word_count) for every sentence the instructor speaks.

    The unit an administrative span actually occupies is a sentence -- "the
    homework is due on Gradescope by the deadline" is one span whether it
    names one platform or four -- so admin_talk_pct needs the sentences, not
    just the words.

    Built from the word entries rather than the segment text because those are
    what the denominator counts, and because they carry their own punctuation
    ("change", "it.") so the sentence boundaries survive. Word counts here sum
    to exactly n_inst: same words, same per-word speaker rule, same fallback
    to the segment text when a transcript has no word timings.

    A sentence is closed at a segment boundary even without terminal
    punctuation. Whisper segments average 11 words and 1.03 sentences on the
    two reference transcripts, so that is nearly always where a sentence ends
    anyway, and it bounds how much one term can be credited with.
    """
    out = []
    for s in segments:
        ws = [w for w in (s.get("words") or []) if w.get("word")]
        if ws:
            buf = []
            for w in ws:
                if w.get("speaker", s.get("speaker")) != inst:
                    continue
                buf.append(w["word"])
                if w["word"].rstrip("\"')]”’").endswith((".", "!", "?")):
                    out.append((" ".join(buf), len(buf)))
                    buf = []
            if buf:
                out.append((" ".join(buf), len(buf)))
        elif s.get("speaker") == inst:
            for sent in re.split(r"(?<=[.!?])\s+", s.get("text") or ""):
                n = len(_tokens(sent))
                if n:
                    out.append((sent, n))
    return out


def measure(segments, duration_s, levels=None):
    """Every speech-tier metric. `levels` is audio_metrics' frame analysis."""
    m = {}
    if not segments:
        return m

    inst = instructor_label(segments)
    m["instructor_label"] = inst
    m["segment_count"] = len(segments)
    speakers = {s.get("speaker") for s in segments if s.get("speaker")}
    m["speaker_count"] = len(speakers)

    # --- word-level partition ------------------------------------------
    # Per word, because the speaker field on a segment disagrees with the
    # words inside it often enough to matter.
    inst_words, other_words, all_words = [], [], []
    for s in segments:
        ws = s.get("words") or []
        if ws:
            for w in ws:
                if not w.get("word"):
                    continue
                spk = w.get("speaker", s.get("speaker"))
                (inst_words if spk == inst else other_words).append(w)
                all_words.append(w)
        else:
            # No word timings: fall back to the segment, and say so via
            # word_timing_coverage below rather than silently degrading.
            for tok in _tokens(s.get("text")):
                fake = {"word": tok, "start": s["start"], "end": s["end"]}
                (inst_words if s.get("speaker") == inst
                 else other_words).append(fake)
                all_words.append(fake)

    n_inst, n_all = len(inst_words), len(all_words)
    if n_all == 0:
        return m
    m["word_count"] = n_all
    m["instructor_word_share"] = float(n_inst / n_all * 100.0)
    timed = sum(1 for w in all_words if "start" in w and "end" in w)
    m["word_timing_coverage"] = float(timed / n_all * 100.0)

    hours = duration_s / 3600.0 if duration_s > 0 else 0.0
    minutes = duration_s / 60.0 if duration_s > 0 else 0.0

    # --- ASR confidence -------------------------------------------------
    logprobs = [s["avg_logprob"] for s in segments
                if isinstance(s.get("avg_logprob"), (int, float))]
    if logprobs:
        m["asr_confidence"] = float(np.mean(logprobs))
        m["asr_poor_segment_pct"] = float(
            np.mean(np.asarray(logprobs) < POOR_SEGMENT_LOGPROB) * 100.0)

    # --- pacing ----------------------------------------------------------
    # The numerator (n_inst) is word-level, so the denominator has to be too,
    # or the rate is a ratio between two different populations: a segment
    # labelled instructor that contains student words excludes them from the
    # count while keeping their seconds, deflating wpm, and the mirror case
    # inflates it.
    #
    # Segment duration is apportioned by the instructor's share of the words
    # in that segment rather than summing word durations directly. Word
    # timings exclude the gaps between words, so summing them measures
    # articulation rate -- how fast the syllables come -- where what is wanted
    # is speaking rate, pauses inside a sentence included.
    inst_speaking_s = 0.0
    for s in segments:
        ws = s.get("words") or []
        span = s["end"] - s["start"]
        if not ws:
            if s.get("speaker") == inst:
                inst_speaking_s += span
            continue
        named = [w for w in ws if w.get("word")]
        if not named:
            continue
        share = sum(1 for w in named
                    if w.get("speaker", s.get("speaker")) == inst) / len(named)
        inst_speaking_s += span * share
    if inst_speaking_s > 30 and n_inst:
        m["speech_rate_wpm"] = float(n_inst / (inst_speaking_s / 60.0))
        m["instructor_speaking_pct"] = float(inst_speaking_s / duration_s * 100.0) \
            if duration_s > 0 else None

        # Words per wall-clock minute, bucketed, then the spread of the
        # minutes in which the instructor actually spoke.
        starts = np.array([w.get("start", 0.0) for w in inst_words])
        if duration_s > 0:
            bins = np.arange(0, duration_s + 60, 60)
            per_min, _ = np.histogram(starts, bins=bins)
            active = per_min[per_min >= 20]
            if active.size >= 5 and active.mean() > 0:
                m["rate_variability"] = float(active.std() / active.mean())

    # --- dead air, from the transcript rather than the waveform ----------
    prev_end, dead, longest = 0.0, 0.0, 0.0
    for s in segments:
        gap = s["start"] - prev_end
        if gap >= DEAD_AIR_MIN_S:
            dead += gap
            longest = max(longest, gap)
        prev_end = max(prev_end, s["end"])
    tail = (duration_s - prev_end) if duration_s > 0 else 0.0
    if tail >= DEAD_AIR_MIN_S:
        dead += tail
        longest = max(longest, tail)
    if duration_s > 0:
        m["dead_air_pct"] = float(dead / duration_s * 100.0)
        m["longest_dead_air_s"] = float(longest)

    # --- words the microphone did not actually capture -------------------
    if levels and levels.get("db") is not None:
        db, floor, frame_s = levels["db"], levels["floor_db"], levels["frame_s"]
        from src.scan.audio_metrics import AT_FLOOR_MARGIN_DB
        checked = dropped = 0
        for w in all_words:
            if "start" not in w or "end" not in w or w["end"] <= w["start"]:
                continue
            i0, i1 = int(w["start"] / frame_s), int(w["end"] / frame_s)
            if i1 <= i0 or i1 > db.size:
                continue
            checked += 1
            if float(np.median(db[i0:i1])) < floor + AT_FLOOR_MARGIN_DB:
                dropped += 1
        if checked > 100:
            m["dropped_word_pct"] = float(dropped / checked * 100.0)

    # --- text-derived, instructor only -----------------------------------
    inst_text = " ".join(
        (w.get("word") or "") for w in inst_words).lower()
    inst_text = re.sub(r"\s+", " ", inst_text)
    inst_tokens = _tokens(inst_text)

    # Tag questions, counted once here and used twice: they are the bulk of
    # what looked like questions to the class, and they are the filler this
    # lecturer actually uses.
    inst_segments = [s for s in segments if s.get("speaker") == inst]
    tag_count = sum(1 for s in inst_segments
                    if _TAG_RX.search((s.get("text") or "").strip()))
    m["tag_question_count"] = tag_count

    if n_inst >= 100:
        m["filler_per_100w"] = float(
            (_phrase_count(inst_text, FILLERS) + tag_count) / n_inst * 100.0)
        m["signpost_per_1000w"] = float(
            _phrase_count(inst_text, SIGNPOSTS) / n_inst * 1000.0)

        # Administrative talk is measured over the sentences it occupies, not
        # by crediting each term with a notional span.
        #
        # The old rule gave every hit a flat 18 words -- the measured mean
        # sentence length -- and summed them, so terms that co-occur inside
        # one sentence each claimed their own 18. "the homework assignment is
        # due on gradescope by the deadline" is ten words and five hits: 90
        # words attributed, 9.0x the truth. A 9,000-word lecture with 50 such
        # sentences reported 50.0% against a truth of 8.9%, taking the
        # ramp(28, 3) sub-score from 0.76 to 0.00 -- 1.5 points of the total.
        # Both reference lectures have only four hits each, so nothing on this
        # corpus showed it.
        #
        # Counting the words of any sentence that mentions administration
        # cannot exceed 100% by construction, counts co-occurring terms once,
        # and is the same span a human would cut. `admin_mentions` still
        # reports raw hits, which is what it has always meant.
        admin_words = 0
        for sentence, n_words in _instructor_sentences(segments, inst):
            if _phrase_count(sentence.lower(), ADMIN_TERMS):
                admin_words += n_words
        m["admin_talk_pct"] = float(min(100.0, admin_words / n_inst * 100.0))
        m["admin_mentions"] = int(_phrase_count(inst_text, ADMIN_TERMS))

        content = [t for t in inst_tokens if t not in STOPWORDS and len(t) > 2]
        m["content_density"] = float(len(content) / minutes) if minutes else None

        windows = [content[i:i + TTR_WINDOW]
                   for i in range(0, len(content), TTR_WINDOW)]
        windows = [w for w in windows if len(w) == TTR_WINDOW]
        if windows:
            m["lexical_diversity"] = float(
                np.mean([len(set(w)) / len(w) for w in windows]))

        # Topic focus: how much vocabulary the thirds of the lecture share.
        if len(content) >= 600:
            thirds = np.array_split(np.asarray(content, dtype=object), 3)
            tops = []
            for part in thirds:
                counts = {}
                for t in part:
                    counts[t] = counts.get(t, 0) + 1
                tops.append({t for t, _ in sorted(counts.items(),
                                                  key=lambda kv: -kv[1])[:TOPIC_TOP_N]})
            pairs = [(0, 1), (0, 2), (1, 2)]
            jac = [len(tops[i] & tops[j]) / max(len(tops[i] | tops[j]), 1)
                   for i, j in pairs]
            m["topic_focus"] = float(np.mean(jac))

    # --- structure cues ---------------------------------------------------
    def _text_between(lo, hi):
        return " ".join(
            (s.get("text") or "") for s in segments
            if s.get("speaker") == inst and lo <= s["start"] < hi).lower()

    if duration_s > 0:
        m["has_opening"] = bool(_phrase_count(
            _text_between(0.0, OPENING_WINDOW_S), OPENING_CUES))
        m["has_closing"] = bool(_phrase_count(
            _text_between(max(0.0, duration_s - CLOSING_WINDOW_S), duration_s),
            CLOSING_CUES))

    # Questions the instructor puts to the room -- genuine ones only. Counting
    # every question-final segment gave 163 an hour on both references,
    # because three quarters of them were "..., right?".
    inst_q = sum(1 for s in inst_segments
                 if (s.get("text") or "").strip().endswith("?")
                 and not _TAG_RX.search((s.get("text") or "").strip()))
    if hours > 0:
        m["class_question_per_hour"] = float(inst_q / hours)
        m["tag_question_per_hour"] = float(tag_count / hours)

    # --- interaction and the burden it creates ---------------------------
    # A "turn" is a run of consecutive non-instructor segments, so one long
    # question is one turn rather than five.
    # Per WORD, not per segment. This module's docstring has always said so,
    # and this loop did not: it read s["speaker"], which is precisely the
    # field captions.py found lying -- five of lecture 12's fifteen redacted
    # cues were segments labelled instructor with non-instructor words inside.
    #
    # The consequence was one-directional and reassuring, which is the worst
    # kind. A transcript whose student speech sits inside instructor-labelled
    # segments reported student_speech_pct = 0.0 while instructor_word_share
    # correctly said 50%: a lecture needing muting and question cards looked
    # like it needed no anonymization at all.
    #
    # A "turn" is still a run of consecutive student words, so one long
    # question counts once rather than once per word.
    turns, in_turn = 0, False
    student_s = 0.0
    total_speech_s = 0.0
    for w in all_words:
        start, end = w.get("start"), w.get("end")
        if start is None or end is None or end <= start:
            continue
        dur = end - start
        total_speech_s += dur
        spk = w.get("speaker")
        # An unlabelled word is not evidence of a student. Diarization leaves
        # gaps, and counting those as student speech would inflate the burden
        # of every lecture with an imperfect transcript.
        if spk is not None and spk != inst:
            student_s += dur
            if not in_turn:
                turns += 1
                in_turn = True
        else:
            in_turn = False
    if hours > 0:
        m["interaction_per_hour"] = float(turns / hours)
    m["student_turns"] = turns
    if total_speech_s > 0:
        m["student_speech_pct"] = float(student_s / total_speech_s * 100.0)
    m["speech_seconds_words"] = float(total_speech_s)
    m["flagged_questions"] = int(
        sum(1 for s in segments if s.get("is_student_question")))

    # --- a rough PII tripwire --------------------------------------------
    rare = _name_candidates(segments)
    if hours > 0:
        # Distinct people, not total mentions: one student named six times is
        # one person's privacy at stake, not six.
        m["named_mentions_per_hour"] = float(len(rare) / hours)
    m["named_candidates"] = rare[:40]
    return m
