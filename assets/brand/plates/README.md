# Scene plates

Static overlay images. The pipeline draws the live video first, then composites
the plate on top. The plate is transparent everywhere a live source shows through.

## Files

- `scene-a-overlay.png` - 1920x1080 RGBA. Opaque field, footer, SCS unitmark, and
  Continue Learning CTA. Transparent windows for the slide and speaker feeds.
- `scene-b-overlay.png` - 1920x1080 RGBA. Fully transparent except the 60% unitmark
  watermark at top right.
- `proof-scene-a.png` - registration test with synthetic sources. Not for production.
- `*.svg` - sources for the plates.

## The contract

Live-source windows in Scene A, which must not move:

| Source  | Position   | Size      |
| ------- | ---------- | --------- |
| Slide   | x40 y100   | 1380x776  |
| Speaker | x1448 y100 | 432x576   |

Anyone can redesign the plate in Figma, Canva, Illustrator, or Photoshop. As long as
those two rectangles stay transparent and stay at these coordinates, the pipeline
needs no code change. Export 1920x1080 PNG with alpha.

## Per-lecture text is NOT in the plate

Course code, course title, lecture title, and term change per video, so they are drawn
at render time from Panopto metadata. Baselines, left-aligned at x1448:

| Element        | Baseline | Font              | Size | Color   |
| -------------- | -------- | ----------------- | ---- | ------- |
| Course code    | y718     | Open Sans Bold    | 19   | #C41230 |
| Course title 1 | y752     | Open Sans Regular | 17   | #A5A5AA |
| Course title 2 | y777     | Open Sans Regular | 17   | #A5A5AA |
| Divider rule   | y800     | 1px line to x1880 | -    | #333336 |
| Lecture line 1 | y839     | Open Sans SemiBold| 32   | #FFFFFF |
| Lecture line 2 | y879     | Open Sans SemiBold| 32   | #FFFFFF |
| Term           | y916     | Open Sans Regular | 18   | #8E8E93 |

Check the longest expected lecture title before batch rendering. The rail is 432 px wide.

## Compositing

    ffmpeg -i slide.mp4 -i camera.mp4 -i scene-a-overlay.png -filter_complex "\
      color=c=0x0F0F10:s=1920x1080:d=1[bg];\
      [0:v]scale=1380:776:force_original_aspect_ratio=increase,crop=1380:776[slide];\
      [1:v]scale=432:576:force_original_aspect_ratio=increase,crop=432:576[cam];\
      [bg][slide]overlay=40:100[a];[a][cam]overlay=1448:100[b];\
      [b][2:v]overlay=0:0[out]" -map "[out]" -map 0:a -c:v libx264 -crf 18 out.mp4

Crop, do not stretch: `force_original_aspect_ratio=increase` plus `crop` preserves the
source aspect ratio inside each window.

## Known issue

`rsvg-convert` silently drops linked `<image>` elements, so the unitmark does not render
if you rebuild the PNGs from the SVGs with it. The current PNGs have the unitmark
composited separately. Use a renderer that honors linked images, or repeat that step.

The unitmarks here are raster placement proxies. Replace with the official CMU SCS
vector unitmark before final mastering.
