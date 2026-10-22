#!/usr/bin/env python3
"""Matplotlib -> theme-aware inline SVG figure for the site.

Draw with the five sentinel colours below; save() swaps them for currentColor
and the CSS variables in quartz/styles/custom.scss, so plots follow light/dark
mode. Output: tools/figs/NAME.html, one <figure class="cs-fig"> line, which
tools/assemble.py inlines after a <!-- fig:NAME --> marker.

    from mplfig import plt, INK, ACCENT, ACCENT2, HI, MUTED, save
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y, color=ACCENT)
    save(fig, "sigmoid", "Caption with <b>bold</b>.", alt="...")
"""
import html as _html
import io
import os
import re

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402

INK, ACCENT, ACCENT2, HI, MUTED = "#010101", "#020202", "#030303", "#040404", "#050505"
_MAP = {INK: "currentColor", ACCENT: "var(--accent)", ACCENT2: "var(--accent2)", HI: "var(--hi)", MUTED: "var(--muted)"}
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Source Sans 3", "Source Sans Pro", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=[ACCENT, HI, ACCENT2, MUTED, INK]),
    "grid.color": MUTED, "grid.alpha": 0.35, "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name, caption="", alt="", width=None):
    """Write tools/figs/NAME.html; width (px) caps the displayed size."""
    plt.rcParams["svg.hashsalt"] = name
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<style[^>]*>.*?</style>", "", s, flags=re.S)  # global * rule would leak
    s = s.replace("<svg ", '<svg stroke-linejoin="round" ', 1)
    for k, v in _MAP.items():
        s = s.replace(k, v).replace(k.upper(), v)
    m = re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', s)
    w = float(m.group(1)) * 4 / 3 if m else 480
    s = re.sub(r' width="[\d.]+pt" height="[\d.]+pt"', "", s, count=1)
    cap = int(width or w)
    s = s.replace("<svg ", f'<svg role="img" aria-label="{_html.escape(alt or name)}" style="max-width:{cap}px;width:100%;height:auto" ', 1)
    # prefix ids so several plots can share a page
    ids = set(re.findall(r'id="([^"]+)"', s))
    for i in sorted(ids, key=len, reverse=True):
        s = s.replace(f'id="{i}"', f'id="{name}-{i}"').replace(f"#{i})", f"#{name}-{i})").replace(f'"#{i}"', f'"#{name}-{i}"')
    s = re.sub(r">\s+<", "><", s).replace("\n", " ")
    out = f'<figure class="cs-fig">{s}'
    if caption:
        out += f"<figcaption>{caption}</figcaption>"
    out += "</figure>"
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(out + "\n")
    return out
