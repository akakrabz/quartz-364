#!/usr/bin/env python3
"""Inline the generated figures into the content pages, in place.

A page marks a figure with a line

    <!-- fig:NAME -->

and this script makes sure the very next line is the current contents of
tools/figs/NAME.html (one <figure class="cs-fig">…</figure> line). Run it after
regenerating figures (python3 tools/figs_chNN.py). Idempotent: running it twice
changes nothing.

usage: python3 tools/assemble.py [content-dir]           all pages (default dir: ../content)
       python3 tools/assemble.py page.md [page.md ...]   only these pages
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[1:]
ONLY = [os.path.abspath(a) for a in ARGS if a.endswith(".md")]
CONTENT = next((a for a in ARGS if not a.endswith(".md")), os.path.join(HERE, "..", "content"))
FIGS = os.path.join(HERE, "figs")
MARK = re.compile(r"^<!--\s*fig:([a-z0-9-]+)\s*-->\s*$")


def main():
    used, missing, changed = set(), [], 0
    targets = ONLY or [os.path.join(root, fn) for root, _, files in os.walk(CONTENT) for fn in sorted(files)
                       if fn.endswith(".md")]
    for path in targets:
        if True:
            lines = open(path, encoding="utf-8").read().split("\n")
            out, i = [], 0
            while i < len(lines):
                ln = lines[i]
                out.append(ln)
                m = MARK.match(ln)
                if m:
                    name = m.group(1)
                    fp = os.path.join(FIGS, name + ".html")
                    # drop a previously inlined copy of this (or any) figure
                    if i + 1 < len(lines) and lines[i + 1].startswith('<figure class="cs-fig"'):
                        i += 1
                    if not os.path.exists(fp):
                        missing.append((os.path.relpath(path, CONTENT), name))
                    else:
                        fig = open(fp, encoding="utf-8").read().strip()
                        if "\n" in fig:
                            sys.exit(f"figure {name} spans several lines; regenerate it")
                        out.append(fig)
                        used.add(name)
                i += 1
            new = "\n".join(out)
            if new != "\n".join(lines):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
                changed += 1
    print(f"assemble: {len(used)} figures inlined, {changed} page(s) updated")
    if missing:
        print("MISSING FIGURES:", missing)
        sys.exit(1)
    if os.path.isdir(FIGS) and not ONLY:
        unused = sorted({f[:-5] for f in os.listdir(FIGS) if f.endswith(".html")} - used)
        if unused:
            print("unused figures:", unused)


if __name__ == "__main__":
    main()
