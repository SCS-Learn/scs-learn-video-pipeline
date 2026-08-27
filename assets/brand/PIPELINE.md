# How the pipeline uses these assets

`README.md` and `plates/README.md` in this directory are Phillip's, unedited.
This file is the pipeline's side of the same contract: what reads what, what is
checked, and what we found when we ran it against real footage.

## What reads what

| Asset | Read by | For |
|---|---|---|
| `plates/scene-a-overlay.png` | `src/assembly/brand.py` → `layout.py` | the Scene A overlay |
| `plates/scene-b-overlay.png` | same | the Scene B watermark |
| `courses.json` | `brand.course_title` | the course title under the code |
| `../fonts/static/OpenSans-*.ttf` | `brand.font` | the rail type |
| `wordmarks/`, `assets/`, `*.svg`, `*.pdf` | nothing yet | source and reference |

Nothing else in the repo composites a frame. `assembly.py` stream-copies what
`layout.py` produced.

## The contract, and how it is enforced

`brand.SLIDE_WINDOW` and `brand.SPEAKER_WINDOW` are the two rectangles from
`plates/README.md`. `brand.verify_plate` runs on every load and fails the render
if a plate does not leave them transparent — so "redesign it in Figma, export
1920x1080 with alpha, drop it in" is checked rather than hoped for. Try a
redesign without touching code:

```bash
python scripts/render_scene_proof.py --lecture-dir data/15210-lecture12 \
    --plate-dir /path/to/redesign
python -m src.assembly.layout --lecture-dir data/15210-lecture12 \
    --plate-dir /path/to/redesign --start 1740 --duration 30 --out try.mp4
```

`render_scene_proof.py` reproduces `plates/proof-scene-a.png` from *our*
compositor rather than from the design tool, over grids whose 1px yellow border
sits exactly on each window edge. If a window moved, a side goes missing.

## Per-lecture type

Drawn at render time from `TEXT_SPEC` in `brand.py`, which is the baseline /
font / size / colour table from `plates/README.md` verbatim. Sources, in
precedence order:

| Field | Where it comes from |
|---|---|
| Course code | Panopto `course` |
| Course title | `--course-title`, else the lecture's `metadata.json` `course_title`, else `courses.json` |
| Lecture title | Panopto `name` |
| Term | `--term`, else `metadata.json` `term`, else derived from Panopto `start` |

Panopto's `SessionStartTime` is **seconds since 1601-01-01** — the Windows
FILETIME epoch, not Unix. Read as Unix it lands in 2395; read as .NET ticks it
lands in year 426. `brand.term_from_panopto` is the only place that decodes it.

"Check the longest lecture title before you batch render" is automated:
`brand.fit_lines` wraps to two lines, breaks after a colon when there is one
(`Lecture 12:` / `Binary Search Trees`, matching the proof), otherwise balances
the two lines, and shrinks the type a point at a time when a title will not go —
logging every time it does. A title that does not fit even at 70% is truncated
with an ellipsis and a loud warning rather than running off the frame.

## What we found running it against real lectures

Four things for Phillip, per "if something in the layout fights the pipeline,
tell me and we will change the layout". The first two came off 15-210 lecture
12, the second two off 17-635 lecture 13 -- a different course, a different
room, a different capture appliance.

**1. The Scene B watermark is invisible.** The mark is 60% white at the top
right, and whatever is behind it keeps being white: the projection screen in
15-210's hall, a bare white wall in 17-635's room. Measured over lecture 12's Scene B runs it changes the pixels
under it by 0.003 on a 0–1 scale, where 0.06 is roughly the threshold of
visibility. It is composited correctly and cannot be seen. `Plate.legibility`
measures this on frames the render loop already has, and every render prints the
number, so this cannot ship silently. A darker mark, a soft scrim behind it, or
a different corner would all fix it; which one is a design call, so the pipeline
reports rather than corrects.

**2. The professor is smaller than he used to be.** The 432×576 window is 3:4,
so a full-height crop of the 1280×720 camera is 540px wide and scales *down* by
0.8. The pre-brand rail was 470×1080, a 313px crop scaled *up* by 1.5 — he was
about 1.7× larger on screen. The new version is sharper and the window
comfortably clears his gesture box (346px at the 90th percentile on lecture 12),
so this is a legitimate framing choice rather than a bug, but it is a visible
change. `--pip-crop-h 0.8 --pip-crop-y 0.15` zooms back in at the cost of the
top of the room; the defaults do not, because tuning them against one lecture is
how you overfit to one lecture.

**3. The rail overruns on a course with long lecture titles.** 15-210's titles
are short ("Binary Search Trees") and never showed this. 17-635's are not: 7 of
its 19 titles do not fit two 432px lines at the spec'd 32px, and the shrink-to-
fit lands them between 22px and 32px.

| Title | Rendered at |
|---|---|
| Lecture 09: Scalability | 32px |
| Lecture 11: Introduction to Architectural Design Process | 31px |
| Lecture 07: Modifiability cont. & Performance Fundamentals | 29px |
| Lecture 13: Integration of LLMs in Software Applications cont. | 28px |
| Lecture 03: Architectural Requirements (Architectural Drivers) | 24px |
| Lecture 08: Performance Fundamentals cont. & Availability Fundamental | 22px |

Every one is legible, but across a published course the rail type visibly
changes size video to video, which reads as inconsistent rather than as fitted.
A third line for the lecture title would absorb most of it; so would shorter
titles in Panopto. Both are Phillip's call, so the renderer fits and logs rather
than deciding.

**4. Not every screen capture is 16:9.** 17-635 lecture 13's is a 4:3 deck
pillarboxed into 1920x1080 -- 1440x1080 of slide with a 240px black bar each
side, stable across the whole lecture. Two consequences, both handled in
`layout._slide_filter`:

* The capture's own bars are cropped off before the slide is scaled, so the
  pillars in the finished frame are the brand field colour rather than pure
  black sitting inside the plate's rounded window.
* The crop is **verified against the question-card frames** before it is used.
  Cards are authored full-frame 1920x1080 and inserted into the same stream, so
  a box derived from a pillarboxed deck would shave 240px off both sides of
  every card -- through the CMU lockup on one side and the tartan wedge on the
  other. Evenly spaced samples never catch that: on 17-635 lecture 13 the cards
  are 9 seconds out of 4,925. When the box does not hold at a card, cropping is
  declined entirely and the reason is logged, because one crop cannot suit
  both. The cleaner long-term fix is for `cards.py` to render card art into the
  same pillarbox geometry as the deck it is replacing, which would make the
  whole screen track uniform and let the crop apply everywhere.
* **A source that is not the window's aspect is letterboxed, not cropped.**
  The plates README says "crop, do not stretch", and `increase` + `crop` is
  right for a 16:9 source. Applied to 4:3 content it would scale to 1380x1035
  and crop to 776, cutting 25% off the top and bottom -- which on these slides
  is the title and the footer bar. Slides are the one thing in this layout that
  must not lose content, so anything off-aspect is fitted inside the window and
  padded. `--slide-crop W:H:X:Y` overrides the detection; `--slide-crop none`
  disables it.

## Still open

- The unitmarks here are raster placement proxies. **Swap in the official CMU
  SCS vector unitmark before mastering**, per Phillip's note. `brand.check_footer_unitmark`
  warns if a rebuilt plate lost its linked mark, but it cannot tell a proxy from
  the real thing.
- `optional-lower-third-overlay.svg` is not wired up. It is SVG only, and this
  machine has no `rsvg-convert`, `cairosvg` or `inkscape` — which is also why
  Phillip's rsvg warning has not bitten us: we use the shipped PNGs and never
  rebuild them. A 1920×1080 RGBA PNG export of the lower third is all the
  pipeline would need to composite it for the 4–6 seconds after the first cut
  to Scene B.
