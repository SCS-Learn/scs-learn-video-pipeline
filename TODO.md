# Open items

Ordered by how much damage they do if ignored. Everything here was found by
measurement during the theme/demo work on 15210-lecture12; the numbers are real
and reproducible, not estimates.

## Privacy / correctness

**Diarization finds only 1.19% non-instructor speech, and everything keys off it.**
On lecture 12, 1,227 of 1,237 segments are attributed to `SPEAKER_00`; the other
10 total 56.9s across 79 minutes. Audio muting silences *only* non-instructor
spans, so any student speech attributed to the instructor is never muted, and
never carded. This is the root cause of "question detection may be
conservative" — the classifier was never the bottleneck. Worth measuring
against a lecture with more audience audio before trusting the coverage number.

**Mute spans leak ~300ms at each boundary.** Measured at the 598.3-604.0s
student span: 599-602s is true digital silence (-91.0 dB) but t=598 peaks at
-15.9 dB and t=603 at -28.8 dB. That is speech onset/offset, quiet but present.
Pad the merged spans by ~0.3s in `audio.py`.

**A follow-up question inside another one loses its text.**
`render_full_lecture.plan()` merges overlapping spans and keeps the FIRST
question's text, so the follow-up is muted with no card and the first card
simply runs longer. `cards.py:plan_timeline` already solved this correctly —
it clips the later span forward instead of absorbing it. The right fix is to
delete the duplicate planner and call `plan_timeline`. Did not bite on lecture
12 (spans are 40s+ apart) and fails silently: right duration, card on screen,
wrong text.

**PSC face_anon identity is unreviewed.** Job 43019503 merged 10 clusters as
"instructor", covering 91.6% of sampled faces. Run `--preview` and inspect
`face_clusters.png` before treating that output as publishable. Note the local
run pixelates the instructor for the first ~4s (backlit against the projection
screen, similarity 0.356 vs a 0.45 threshold) — that is fail-closed working,
and not fixable without dropping below the 0.348 a different person scored.

**`manifest.json` is in public git history.** Two blobs (1.5 MB, 1.3 MB) added
in `bf1ba9c`, deleted in `edcbb29`, reachable from `origin/kaveh` and
`origin/testing` on a public repo. Contains 50 CloudFront URLs to
un-anonymized recordings of 25 lectures, with no signature parameters. No
credentials. Confirm with CMU/Panopto whether those delivery URLs are
access-controlled; a git history rewrite alone does not remove them from
GitHub's object store.

## Polish

**Cards less than ~2s apart flash a frame of lecture between them.** Two cards
separated by 1s produce fade-out, one frame of layout, fade-in. Coalesce them —
open question is what the merged card should say (both questions, or just the
follow-up).

**The card sting is 5.18s; the shortest card is 3.3s.** `render_full_lecture`
scales the fade and delay by duration, but `assembly.mix_card_sound` does not —
it has the real spans from `cards.json` and should tie a fade-out to each.

## Portability (make it PSC-native)

**The rail layout is not a pipeline stage.** It lives in
`scripts/render_full_lecture.py`, is absent from `STAGES` and `verify.py`, and
duplicates compositing that belongs in `assembly.py`. Folding it in — replacing
the corner-PiP path — is what lets someone without a Mac render a lecture.

**`cards.py` may no longer be needed.** Cards are separate full-frame segments
now, so the 20+ minute screen re-encode that makes it `local_only` disappears.
Decide whether the stage survives at all.

**No sbatch for the composite, or for transcription.** `scripts/` has only
`psc_face_anon.sbatch`. The composite is CPU work: RM nodes have 128 cores
against a laptop's 10, and the Regular allocation is barely touched
(4,982 of 5,000 SU) while GPU is scarce (495 hours).

**Transfers still need a human.** `ssh -N psc-dtn` in a real terminal; an agent
cannot do it. `psc.sh sync` correctly refuses to move files over a login node.
