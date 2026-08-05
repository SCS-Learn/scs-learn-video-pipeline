# Card themes

Two demo looks for the same lecture. Drop resources into `fun/` or
`professional/` and the pipeline renders the same content in that style.

    assets/themes/
      fun/            <- playful demo
      professional/   <- the one you would show a dean

Nothing here is wired into the pipeline yet. `cards.py` currently hardcodes
`assets/student-question-card-template.png`; the theme lookup comes once the
resources are in and we know what we are actually working with.

---

## The three cards

| file | when it appears | what it replaces |
|---|---|---|
| `intro-card.png` | once, at the head of the lecture | full frame |
| `question-card.png` | whenever a student asks something | full frame |
| `pip-frame.png` | continuously, around the camera inset | overlay only |

Drop in whichever you have — a theme does not need all three to be useful, and
a missing file should fall back to the other theme rather than crash.

---

## Constraints these have to satisfy

These are not style preferences; they come from how the renderer works.

**`intro-card.png` and `question-card.png` must be exactly 1920x1080 and fully
opaque.** `screen.mp4` is exactly 1920x1080, so a card *replaces* the frame
rather than compositing over it — there is nothing to blend with. This is
load-bearing: CLAUDE.md specifically warns against reintroducing an ffmpeg
`overlay` chain here, because the old version opened one full-length looped
image input per question and cost O(questions x duration). Cut-render-concat
depends on the card being a complete frame.

**`question-card.png` must keep y=360..1010 clear.** Question text is drawn into
that band, centred. The rest of the template is already spoken for:

    y    0- 44   top border
    y   95-207   CMU branding
    y  267-326   heading ("Student Question")
    y  360-1010  <- TEXT GOES HERE, keep it empty
    y 1035-1079  bottom border

Text is dark (`#000000`), so that band needs to stay light. Font size scales
down to fit, and questions are condensed to ~160 characters — past that it
renders as a wall of small text. Do not put artwork in the text band; a busy
background is what made an earlier card illegible.

**`pip-frame.png` is the one file that should have transparency.** It sits
around the camera inset, so it needs RGBA with a genuinely transparent middle.
Give it the aspect ratio of the inset (16:9) and leave the centre clear —
anything opaque there covers the instructor.

**Fonts.** Drop a `.ttf` beside the cards if the theme needs its own. The
default is `assets/fonts/OpenSans-Regular.ttf`. It must be a real `.ttf`;
PIL's `truetype()` loads that, not `.otf` or a webfont.

---

## What I need from you

Which of the two a given resource belongs to is usually obvious, but tell me if
it is not. Worth knowing before I place anything:

- Is the intro card static, or does it need the lecture title / date / course
  number rendered into it? If it takes text, it needs a safe band like the
  question card does.
- Does `professional/` need to follow CMU brand guidelines specifically, or just
  read as restrained?

One thing to flag now: this is an **anonymization** pipeline. If an intro card
carries a student's name, or a photo with students in it, that undoes what
`face_anon` and the audio muting are for. Instructor and course details are
fine; anything student-identifying is not.
