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

_WORD_RX = re.compile(r"[a-z0-9']+")
_CAPS_RX = re.compile(r"\b([A-Z][a-z]{2,})\b")
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
    inst_speaking_s = sum(
        s["end"] - s["start"] for s in segments if s.get("speaker") == inst)
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

        admin_hits = _phrase_count(inst_text, ADMIN_TERMS)
        # Each hit stands in for the sentence around it; ~18 words is the
        # measured mean sentence length across both reference transcripts.
        m["admin_talk_pct"] = float(min(100.0, admin_hits * 18.0 / n_inst * 100.0))
        m["admin_mentions"] = int(admin_hits)

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
    turns, in_turn = 0, False
    student_s = 0.0
    for s in segments:
        is_student = s.get("speaker") is not None and s.get("speaker") != inst
        if is_student:
            student_s += s["end"] - s["start"]
            if not in_turn:
                turns += 1
                in_turn = True
        else:
            in_turn = False
    total_speech_s = sum(s["end"] - s["start"] for s in segments)
    if hours > 0:
        m["interaction_per_hour"] = float(turns / hours)
    m["student_turns"] = turns
    if total_speech_s > 0:
        m["student_speech_pct"] = float(student_s / total_speech_s * 100.0)
    m["flagged_questions"] = int(
        sum(1 for s in segments if s.get("is_student_question")))

    # --- a rough PII tripwire --------------------------------------------
    # Capitalised words in mid-sentence, minus a technical stoplist. Rough on
    # purpose: it flags a lecture for a human to look at, it decides nothing.
    freq = {}
    for s in segments:
        text = s.get("text") or ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            # Skip the first capitalised word: that is just the sentence
            # starting, and every sentence has one.
            for mo in list(_CAPS_RX.finditer(sentence))[1:]:
                tok = mo.group(1)
                if tok.lower() in NOT_NAMES or tok.lower() in STOPWORDS:
                    continue
                freq[tok] = freq.get(tok, 0) + 1
    rare = sorted(t for t, c in freq.items() if c <= NAME_MAX_OCCURRENCES)
    if hours > 0:
        # Distinct people, not total mentions: one student named six times is
        # one person's privacy at stake, not six.
        m["named_mentions_per_hour"] = float(len(rare) / hours)
    m["named_candidates"] = rare[:40]
    return m
