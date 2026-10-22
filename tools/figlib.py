#!/usr/bin/env python3
"""Tiny SVG toolkit for the site's figures: automata, trees, tapes and box diagrams.

Every figure is one <figure class="cs-fig">…</figure> block on a single line
(Markdown ends an HTML block at a blank line). Colours follow the page theme:
strokes and text use currentColor; accents use the CSS variables defined in
quartz/styles/custom.scss (--accent, --accent2, --hi, --muted, --paper).

    from figlib import Automaton
    A = Automaton(420, 160)
    A.state("s", 60, 80, start="left")
    A.state("t", 200, 80, accepting=True)
    A.edge("s", "t", "1")
    A.loop("s", "0", "above")
    html = A.figure("Caption with <b>bold</b> text.", alt="…")
"""
import html as _html
import math

SANS = "'Source Sans 3','Source Sans Pro',system-ui,sans-serif"
SERIF_IT = "'STIX Two Math','Latin Modern Math','Cambria Math','Times New Roman',serif"
MONO = "'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace"

DEFS = ('<defs>'
        '<marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>'
        '</defs>')


def esc(s):
    return _html.escape(str(s), quote=True)


def fmt(v):
    """Compact number formatting for SVG attributes."""
    if isinstance(v, float):
        s = f"{v:.1f}"
        return s[:-2] if s.endswith(".0") else s
    return str(v)


class SVG:
    def __init__(self, width, height, font_size=14):
        self.w, self.h = width, height
        self.items = []
        self.font_size = font_size
        self.uid = None

    # --------------------------------------------------------------- primitives
    def add(self, s):
        self.items.append(s)

    def line(self, x1, y1, x2, y2, color="currentColor", width=1.6, arrow=False, dash=None, opacity=None):
        extra = ""
        if arrow:
            extra += ' marker-end="url(#ARR)"'
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            extra += f' opacity="{opacity}"'
        self.add(f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" stroke="{color}" '
                 f'stroke-width="{width}" stroke-linecap="round"{extra}/>')

    def path(self, d, color="currentColor", width=1.6, arrow=False, fill="none", dash=None, opacity=None):
        extra = ""
        if arrow:
            extra += ' marker-end="url(#ARR)"'
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            extra += f' opacity="{opacity}"'
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')

    def circle(self, cx, cy, r, stroke="currentColor", fill="var(--paper)", width=1.6, fill_opacity=None):
        fo = f' fill-opacity="{fill_opacity}"' if fill_opacity is not None else ""
        self.add(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="{fill}"{fo} stroke="{stroke}" stroke-width="{width}"/>')

    def rect(self, x, y, w, h, rx=0, stroke="currentColor", fill="none", width=1.4, fill_opacity=None, dash=None):
        fo = f' fill-opacity="{fill_opacity}"' if fill_opacity is not None else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="{fmt(rx)}" fill="{fill}"{fo} '
                 f'stroke="{stroke}" stroke-width="{width}"{da}/>')

    def text(self, x, y, s, size=None, anchor="middle", color="currentColor", font=SANS, italic=False,
             weight=None, halo=False, baseline="central", raw=False):
        size = size or self.font_size
        style = f"font-size:{size}px;font-family:{font};"
        if italic:
            style += "font-style:italic;"
        if weight:
            style += f"font-weight:{weight};"
        if halo:
            style += "paint-order:stroke;stroke:var(--paper);stroke-width:5px;stroke-linejoin:round;"
        body = s if raw else esc(s)
        self.add(f'<text x="{fmt(x)}" y="{fmt(y)}" text-anchor="{anchor}" dominant-baseline="{baseline}" '
                 f'fill="{color}" style="{style}">{body}</text>')

    def svg(self, alt=""):
        title = f"<title>{esc(alt)}</title>" if alt else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" style="font-family:{SANS};font-size:{self.font_size}px">'
                f'{title}{DEFS}{"".join(self.items)}</svg>')

    def figure(self, caption, alt=""):
        out = f'<figure class="cs-fig">{self.svg(alt)}<figcaption>{caption}</figcaption></figure>'
        assert "\n\n" not in out
        return out.replace("\n", " ")


# ------------------------------------------------------------------- label helpers
def symbol_label(s):
    """Edge labels: explicit symbols in monospace, ε and ∅ in serif italic; commas separate."""
    return s


class Automaton(SVG):
    """Finite automata (and recursive-automaton modules, TM state diagrams)."""

    def __init__(self, width, height, r=19, font_size=14):
        super().__init__(width, height, font_size)
        self.r = r
        self.states = {}
        self.edges = []  # drawn last so states sit on top? no: edges first, then states

    def state(self, name, x, y, label=None, accepting=False, start=None, r=None, color="currentColor",
              shape="circle", w=None, italic=True, font=None, hi=False):
        """start: None or 'left'/'right'/'above'/'below' (direction the start arrow comes from).
        shape: 'circle' or 'rounded' (rounded rectangle for long names, width w)."""
        self.states[name] = dict(x=x, y=y, label=name if label is None else label, accepting=accepting,
                                 start=start, r=r or self.r, color=("var(--hi)" if hi else color),
                                 shape=shape, w=w, italic=italic, font=font)

    def _boundary(self, st, dx, dy):
        """Point on the state's outline in direction (dx, dy) from its centre."""
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        if st["shape"] == "circle":
            r = st["r"] + (3 if st["accepting"] else 0)
            return st["x"] + ux * r, st["y"] + uy * r
        # rounded rectangle: intersect with the box
        hw = (st["w"] or 2 * st["r"]) / 2 + (3 if st["accepting"] else 0)
        hh = st["r"] + (3 if st["accepting"] else 0)
        tx = hw / abs(ux) if ux else float("inf")
        ty = hh / abs(uy) if uy else float("inf")
        t = min(tx, ty)
        return st["x"] + ux * t, st["y"] + uy * t

    def edge(self, a, b, label="", bend=0, label_pos=0.5, label_side=1, label_dist=11, color="currentColor",
             width=1.6, hi=False, dash=None, label_color=None, mono=True, label_dx=0, label_dy=0):
        """Straight (bend=0) or curved edge from a to b. Positive bend curves to the left of a->b."""
        self.edges.append(("edge", a, b, label, bend, label_pos, label_side, label_dist,
                           "var(--hi)" if hi else color, width, dash, label_color, mono, label_dx, label_dy))

    def loop(self, a, label="", direction="above", size=1.0, hi=False, color="currentColor", mono=True, spread=28):
        """Self-loop; direction is 'above', 'below', 'left', 'right' or an angle in degrees (0 = right, 90 = up)."""
        self.edges.append(("loop", a, label, direction, size, "var(--hi)" if hi else color, mono, spread))

    def _draw_edge(self, e):
        (_, a, b, label, bend, lpos, lside, ldist, color, width, dash, lcolor, mono, ldx, ldy) = e
        A, B = self.states[a], self.states[b]
        x1, y1, x2, y2 = A["x"], A["y"], B["x"], B["y"]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L  # left normal (screen coords: y down, so "left" of travel)
        # flip so that positive bend curves upward for left-to-right edges
        nx, ny = -nx, -ny
        cx, cy = mx + nx * bend * 2, my + ny * bend * 2  # quadratic control point
        if bend == 0:
            sx, sy = self._boundary(A, dx, dy)
            ex, ey = self._boundary(B, -dx, -dy)
            ex2, ey2 = ex - (dx / L) * 2.5, ey - (dy / L) * 2.5
            self.path(f"M{fmt(sx)},{fmt(sy)} L{fmt(ex2)},{fmt(ey2)}", color=color, width=width, arrow=True, dash=dash)
            px = sx + (ex - sx) * lpos
            py = sy + (ey - sy) * lpos
            lx, ly = px + nx * ldist * lside, py + ny * ldist * lside
        else:
            sx, sy = self._boundary(A, cx - x1, cy - y1)
            ex, ey = self._boundary(B, cx - x2, cy - y2)
            # pull the end back a little so the arrowhead tip touches the outline
            tlen = math.hypot(cx - ex, cy - ey) or 1
            ex2, ey2 = ex + (cx - ex) / tlen * 2.5, ey + (cy - ey) / tlen * 2.5
            self.path(f"M{fmt(sx)},{fmt(sy)} Q{fmt(cx)},{fmt(cy)} {fmt(ex2)},{fmt(ey2)}", color=color, width=width,
                      arrow=True, dash=dash)
            t = lpos
            px = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t ** 2 * ex
            py = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t ** 2 * ey
            sgn = 1 if bend > 0 else -1
            lx, ly = px + nx * ldist * sgn * lside, py + ny * ldist * sgn * lside
        if label:
            self._label(lx + ldx, ly + ldy, label, lcolor or ("var(--hi)" if color == "var(--hi)" else "currentColor"), mono)

    def _label(self, x, y, label, color, mono):
        font = MONO if mono else SANS
        size = self.font_size - 1 if mono else self.font_size
        # ε, ∅ and math-ish labels in serif italic
        if label in ("ε", "∅") or not mono:
            self.text(x, y, label, size=self.font_size, color=color, font=SERIF_IT if label in ("ε", "∅") else SANS,
                      italic=label in ("ε", "∅"), halo=True)
        else:
            self.text(x, y, label, size=size, color=color, font=font, halo=True)

    def _draw_loop(self, e):
        (_, a, label, direction, size, color, mono, spread) = e
        A = self.states[a]
        ang = {"above": 90, "below": -90, "left": 180, "right": 0}.get(direction, direction)
        th = math.radians(ang)
        r = A["r"] + (3 if A["accepting"] else 0)
        if A["shape"] != "circle":
            r = (A["w"] or 2 * A["r"]) / 2 if direction in ("left", "right") else A["r"]
        sp = math.radians(spread)
        # screen coordinates: y grows downward
        def pt(t, rad):
            return A["x"] + math.cos(t) * rad, A["y"] - math.sin(t) * rad
        sx, sy = pt(th + sp, r)
        ex, ey = pt(th - sp, r)
        h = 46 * size
        c1x, c1y = pt(th + sp * 1.15, r + h)
        c2x, c2y = pt(th - sp * 1.15, r + h)
        # arrow end slightly outside the outline
        ex2, ey2 = pt(th - sp, r + 2.5)
        self.path(f"M{fmt(sx)},{fmt(sy)} C{fmt(c1x)},{fmt(c1y)} {fmt(c2x)},{fmt(c2y)} {fmt(ex2)},{fmt(ey2)}",
                  color=color, width=1.6, arrow=True)
        lx, ly = pt(th, r + h * 0.62 + 14)
        if label:
            self._label(lx, ly, label, "var(--hi)" if color == "var(--hi)" else "currentColor", mono)

    def _draw_state(self, name):
        st = self.states[name]
        x, y, r = st["x"], st["y"], st["r"]
        col = st["color"]
        if st["start"]:
            d = st["start"]
            ang = {"left": 180, "right": 0, "above": 90, "below": -90}.get(d, d)
            th = math.radians(ang)
            bx, by = self._boundary(st, math.cos(th), -math.sin(th))
            ox, oy = bx + math.cos(th) * 26, by - math.sin(th) * 26
            self.line(ox, oy, bx + math.cos(th) * 2.5, by - math.sin(th) * 2.5, color=col, arrow=True)
        fill = "var(--paper)"
        if st["shape"] == "circle":
            if st["accepting"]:
                self.circle(x, y, r + 4, stroke=col, fill=fill)
                self.circle(x, y, r, stroke=col, fill="var(--accent)", fill_opacity=0.10)
            else:
                self.circle(x, y, r, stroke=col, fill=fill)
        else:
            w = st["w"] or 2 * r
            if st["accepting"]:
                self.rect(x - w / 2 - 4, y - r - 4, w + 8, 2 * r + 8, rx=r + 4, stroke=col, fill=fill, width=1.6)
                self.rect(x - w / 2, y - r, w, 2 * r, rx=r, stroke=col, fill="var(--accent)", fill_opacity=0.10, width=1.6)
            else:
                self.rect(x - w / 2, y - r, w, 2 * r, rx=r, stroke=col, fill=fill, width=1.6)
        lab = st["label"]
        if lab:
            font = st["font"] or (SERIF_IT if st["italic"] else SANS)
            self.text(x, y + 1, lab, size=self.font_size + (1 if st["italic"] else 0), color=col if col != "currentColor" else "currentColor",
                      font=font, italic=st["italic"])

    def svg(self, alt=""):
        # edges under states
        body_before = self.items
        self.items = []
        for e in self.edges:
            if e[0] == "edge":
                self._draw_edge(e)
            else:
                self._draw_loop(e)
        edges = self.items
        self.items = []
        for n in self.states:
            self._draw_state(n)
        states = self.items
        self.items = body_before + edges + states
        out = super().svg(alt)
        self.items = body_before
        return out


# ------------------------------------------------------------------- trees
class Tree(SVG):
    """Layered ordered trees (regular-expression trees, parse trees, computation trees).

    node spec: (label, [children]) ; label strings starting with '$' are drawn in
    serif italic (nonterminals), others in monospace (terminals / operators)."""

    def __init__(self, width, height, font_size=15):
        super().__init__(width, height, font_size)

    def draw(self, root, x0, y0, level_h=34, leaf_w=26, node_style=None, circle_internal=False, hi=()):
        """Draw `root` with its top-left leaf slot at (x0, y0). circle_internal draws internal nodes in small
        circles (good for expression trees); `hi` is a set of node ids (id(node)) to draw in var(--hi)."""
        # compute leaf counts for spacing
        def leaves(n):
            lab, kids = n
            return 1 if not kids else sum(leaves(k) for k in kids)

        positions = []

        def place(n, left, depth):
            lab, kids = n
            if not kids:
                x = left + leaf_w / 2
                positions.append((n, x, y0 + depth * level_h))
                return x, left + leaf_w
            xs = []
            cur = left
            for k in kids:
                kx, cur = place(k, cur, depth + 1)
                xs.append(kx)
            x = (xs[0] + xs[-1]) / 2
            positions.append((n, x, y0 + depth * level_h))
            return x, cur

        place(root, x0, 0)
        pos = {id(n): (x, y) for n, x, y in positions}

        def edges(n):
            lab, kids = n
            px, py = pos[id(n)]
            for k in kids:
                kx, ky = pos[id(k)]
                gap_top = 13 if (circle_internal and n[1]) else 9
                gap_bot = 13 if (circle_internal and k[1]) else 9
                L = math.hypot(kx - px, ky - py) or 1
                ux, uy = (kx - px) / L, (ky - py) / L
                self.line(px + ux * gap_top, py + uy * gap_top, kx - ux * gap_bot, ky - uy * gap_bot,
                          width=1.2, color="currentColor", opacity=0.55)
                edges(k)

        edges(root)
        for n, x, y in positions:
            lab = n[0]
            col = "var(--hi)" if id(n) in hi else "currentColor"
            if circle_internal and n[1]:
                self.circle(x, y, 12, stroke=col, fill="var(--paper)", width=1.2)
            if lab.startswith("$"):
                self.text(x, y, lab[1:], size=self.font_size + 1, font=SERIF_IT, italic=True, halo=not circle_internal,
                          color=col)
            elif lab in ("ε", "∅"):
                self.text(x, y, lab, size=self.font_size + 1, font=SERIF_IT, italic=True, halo=True, color=col)
            else:
                self.text(x, y, lab, size=self.font_size, font=MONO, halo=not (circle_internal and n[1]), color=col)
        return pos


# ------------------------------------------------------------------- tapes
class Tape(SVG):
    def cells(self, x, y, symbols, cell=26, head=None, head_label=None, highlight=(), dots=True):
        for i, s in enumerate(symbols):
            hl = i in highlight
            self.rect(x + i * cell, y, cell, cell, stroke="currentColor", fill="var(--accent)" if hl else "none",
                      fill_opacity=0.12 if hl else None, width=1.2)
            if s:
                font = SERIF_IT if s in ("□",) else MONO
                self.text(x + i * cell + cell / 2, y + cell / 2 + 1, s, size=self.font_size, font=font)
        if dots:
            self.text(x + len(symbols) * cell + 14, y + cell / 2, "⋯", size=self.font_size)
        if head is not None:
            hx = x + head * cell + cell / 2
            self.line(hx, y + cell + 30, hx, y + cell + 4, arrow=True, color="var(--hi)", width=1.8)
            if head_label:
                self.text(hx, y + cell + 42, head_label, size=self.font_size - 1, color="var(--hi)", font=SERIF_IT, italic=True)
