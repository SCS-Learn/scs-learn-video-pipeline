# CMU SCS Open Courseware video system

This package turns the approved Variation A concept into editable 1920 x 1080 production assets.

## Files

- `scene-a-slide-composite.svg` - slide-led layout with named slide and speaker masks, course metadata, footer, unitmark placement, CTA, and hidden safe-area guides.
- `scene-b-full-camera.svg` - clean full-bleed landscape camera layout with a 60% white unitmark watermark.
- `optional-lower-third-overlay.svg` - transparent lower third for brief use when the professor first appears full-screen.
- `cmu-scs-ocw-production-handoff.pdf` - dimensions, switching logic, brand guardrails, CTA strategy, and implementation checklist.
- `assets/` - transparent unitmark placement proxies.

## Recommended production behavior

- Use Scene A when the slide is the primary evidence.
- Cut to Scene B for walking, board work, demonstrations, room-scale gestures, and audience questions.
- Keep Scene B full bleed with only the watermark. Do not retain the footer or rail.
- Use the optional lower third for 4 to 6 seconds on the first full-camera appearance.
- Keep audio continuous. Default to a hard cut on a phrase boundary; use a 4 to 6 frame dissolve only to soften a visible framing or exposure jump.
- Keep the subscribe message in Scene A as a secondary reminder and use the strongest subscription ask on the end card.

## Brand and typography note

The supplied unitmark PNGs are placement proxies derived from the provided source. Before final mastering, replace them with the official Carnegie Mellon School of Computer Science vector unitmark from University Communications and Marketing. Do not re-typeset or rebuild the official mark.

The editable metadata uses Open Sans with an Arial fallback. Carnegie Red is `#C41230`; the field color is `#0F0F10`.

## Importing

The SVGs use named groups that remain editable in Figma, Illustrator, Inkscape, and most motion-design applications. If a motion application does not preserve linked images, relink the files in `assets/` or replace them with the official vector unitmark.
