"""A very small PDF writer: headings, body text, and monospace tables.

    from scripts.textpdf import Doc
    d = Doc(title="Spring 2026 lecture scan")
    d.heading("Course ranking")
    d.para("Ranked by mean score over ...")
    d.table(["  # COURSE   MEAN", "  1 10-707    93.8"])
    d.save("out.pdf")

Why this exists rather than reportlab or wkhtmltopdf
----------------------------------------------------
Neither is installed, and neither is worth adding to a repo whose install is
already split across two conflicting requirements files -- this pipeline has
been bitten once by a dependency that shadowed another (plain `onnxruntime`
over `onnxruntime-gpu`), and a report generator is not a good reason to risk
it again. macOS `cupsfilter` is present and does convert text to PDF, but only
as a fixed-pitch dump with no headings, which is not something to attach to an
email to faculty.

So: the PDF is written by hand. That is far less work than it sounds, because
of one thing -- the **base-14 fonts**. Helvetica, Helvetica-Bold and Courier
are guaranteed present in every PDF reader and do not have to be embedded, so
there is no font subsetting, no CMap, and no binary anything. The whole file
is ASCII, the text stays selectable and searchable, and a four-page report
comes out around 20 kB rather than the several megabytes a page-image PDF
costs.

The one thing that cannot be hand-waved is TEXT WIDTH. Wrapping a paragraph
needs to know how wide a string is in the chosen font, so the AFM advance
widths for Helvetica and Helvetica-Bold are in the table below. They are the
published metrics for those faces, in 1/1000 em, for ASCII 32..126. Courier is
monospace at a flat 600.
"""

import os

PT = 1.0
PAGE_W, PAGE_H = 612.0, 792.0          # US Letter, in points
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 54.0, 60.0, 54.0

# AFM advance widths, 1/1000 em, for ASCII 32..126.
_HELV = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]
_HELV_B = [
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
]
FONTS = {                               # name -> (pdf base font, widths)
    "body": ("Helvetica", _HELV),
    "bold": ("Helvetica-Bold", _HELV_B),
    "mono": ("Courier", None),          # None -> flat 600
}


def width(text, font="body", size=10.0):
    """Advance width of `text` in points."""
    _, table = FONTS[font]
    total = 0
    for ch in text:
        o = ord(ch)
        if table is None:
            total += 600
        elif 32 <= o <= 126:
            total += table[o - 32]
        else:
            total += table[0]           # anything exotic renders as a space
    return total * size / 1000.0


def wrap(text, max_w, font="body", size=10.0):
    """Greedy word wrap to `max_w` points. A paragraph, so greedy is right."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if width(trial, font, size) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _esc(s):
    """PDF string escaping, and drop anything outside the base encoding.

    A smart quote or an en dash in a lecture title is not worth an encoding
    table: they are replaced with their ASCII equivalents so the text stays
    readable rather than turning into a mojibake run.
    """
    s = (s.replace("—", "--").replace("–", "-")
          .replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("…", "...").replace(" ", " "))
    s = "".join(c if 32 <= ord(c) <= 126 else "?" for c in s)
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class Doc:
    """A flowing document. Text is placed top-down; pages break themselves."""

    def __init__(self, title="", subtitle="", leading=1.32):
        self.title = title
        self.subtitle = subtitle
        self.leading = leading
        self.pages = []
        self._ops = []
        self.y = PAGE_H - MARGIN_TOP
        if title:
            self.text(title, "bold", 20, gap_after=4)
            if subtitle:
                self.text(subtitle, "body", 10.5, gray=0.42, gap_after=10)
            self.rule()

    # --- primitives -------------------------------------------------------
    @property
    def content_w(self):
        return PAGE_W - 2 * MARGIN_X

    def _newpage(self):
        self.pages.append("\n".join(self._ops))
        self._ops = []
        self.y = PAGE_H - MARGIN_TOP

    def _need(self, h):
        if self.y - h < MARGIN_BOT:
            self._newpage()

    def text(self, line, font="body", size=10.0, gray=0.0, x=None,
             gap_after=0.0):
        h = size * self.leading
        self._need(h + gap_after)
        base, _ = FONTS[font]
        self.y -= h
        self._ops.append(
            f"BT /{base} {size:.2f} Tf {gray:.3f} g "
            f"1 0 0 1 {(x if x is not None else MARGIN_X):.2f} {self.y:.2f} Tm "
            f"({_esc(line)}) Tj ET")
        self.y -= gap_after

    def rule(self, gray=0.75, gap_before=4.0, gap_after=8.0):
        self._need(gap_before + gap_after + 1)
        self.y -= gap_before
        self._ops.append(
            f"{gray:.3f} G 0.6 w {MARGIN_X:.2f} {self.y:.2f} m "
            f"{PAGE_W - MARGIN_X:.2f} {self.y:.2f} l S")
        self.y -= gap_after

    # --- flowing blocks ---------------------------------------------------
    def heading(self, s, size=13.5, gap_before=14.0, gap_after=5.0):
        # Kept with what follows: a heading alone at the foot of a page is the
        # one break a flowing layout must not make.
        self._need(gap_before + size * self.leading + 3 * 10 * self.leading)
        self.y -= gap_before
        self.text(s, "bold", size, gap_after=gap_after)

    def para(self, s, size=10.0, gray=0.13, gap_after=7.0, font="body"):
        for line in wrap(s, self.content_w, font, size):
            self.text(line, font, size, gray=gray)
        self.y -= gap_after

    def bullets(self, items, size=10.0, gap_after=7.0, indent=13.0):
        for it in items:
            lines = wrap(it, self.content_w - indent, "body", size)
            self.text("-", "body", size, gray=0.45)
            self.y += size * self.leading          # same line as the dash
            for n, line in enumerate(lines):
                if n:
                    self.y -= size * self.leading
                self.text(line, "body", size, gray=0.13,
                          x=MARGIN_X + indent)
                self.y += size * self.leading
            self.y -= size * self.leading
        self.y -= gap_after

    def table(self, lines, size=8.6, gap_after=8.0, head=0):
        """Pre-formatted fixed-width rows. `head` rows repeat after a break."""
        header = lines[:head]
        for i, line in enumerate(lines):
            if (self.y - size * self.leading < MARGIN_BOT and i >= head):
                self._newpage()
                for h in header:
                    self.text(h, "mono", size, gray=0.0)
            self.text(line, "mono", size,
                      gray=0.0 if i < head else 0.13)
        self.y -= gap_after

    # --- output -----------------------------------------------------------
    def save(self, path):
        self.pages.append("\n".join(self._ops))
        pages = [p for p in self.pages if p.strip()] or [""]
        n = len(pages)

        objs = []                       # 1-indexed; objs[i] is object i+1
        objs.append(f"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{3 + i} 0 R" for i in range(n))
        objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>")
        font_base = 3 + 2 * n
        fonts = ("<< /F1 %d 0 R /F2 %d 0 R /F3 %d 0 R >>"
                 % (font_base, font_base + 1, font_base + 2))
        for i in range(n):
            objs.append(
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
                f"/Resources << /Font {fonts} >> "
                f"/Contents {3 + n + i} 0 R >>")
        for p in pages:
            data = p.encode("latin-1", "replace")
            objs.append(f"<< /Length {len(data)} >>\nstream\n{p}\nendstream")
        for base in ("Helvetica", "Helvetica-Bold", "Courier"):
            objs.append(f"<< /Type /Font /Subtype /Type1 /BaseFont /{base} "
                        f"/Encoding /WinAnsiEncoding >>")

        # The page resource dict names fonts /F1../F3, but the content stream
        # writes /Helvetica etc. Map them by giving each font its own name.
        fixed = []
        for o in objs:
            fixed.append(o.replace("/F1 ", "/Helvetica ")
                          .replace("/F2 ", "/Helvetica-Bold ")
                          .replace("/F3 ", "/Courier "))
        objs = fixed

        out = ["%PDF-1.4\n"]
        offsets = []
        pos = len(out[0])
        for i, body in enumerate(objs, start=1):
            chunk = f"{i} 0 obj\n{body}\nendobj\n"
            offsets.append(pos)
            out.append(chunk)
            pos += len(chunk.encode("latin-1", "replace"))
        xref = pos
        out.append(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n")
        for off in offsets:
            out.append(f"{off:010d} 00000 n \n")
        out.append(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R "
                   f"/Info << /Title ({_esc(self.title)}) "
                   f"/Producer (scs-learn-video-pipeline) >> >>\n"
                   f"startxref\n{xref}\n%%EOF\n")
        with open(path, "wb") as f:
            f.write("".join(out).encode("latin-1", "replace"))
        return path
