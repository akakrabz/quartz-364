#!/usr/bin/env python3
"""Check every docs.pytorch.org link in content/ against the docs sources.

usage: python3 tools/check_doclinks.py PYTORCH_DOCS_SOURCE VISION_DOCS_SOURCE [content-dir]
  e.g. sparse clones of pytorch/pytorch (release/2.14) and pytorch/vision (release/0.29), docs/source.
Tutorial links are listed, not checked.
"""
import os
import re
import sys

PT, TV = sys.argv[1], sys.argv[2]
ROOT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content")


def heading_slug(s):
    s = re.sub(r"[`*]", "", s).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s_]+", "-", s).strip("-")


def scan(src):
    """-> generated names, {page: (anchors)}"""
    gen, pages = set(), {}
    for root, _, files in os.walk(src):
        for fn in files:
            if not fn.endswith((".md", ".rst")):
                continue
            p = os.path.join(root, fn)
            page = os.path.relpath(p, src).rsplit(".", 1)[0]
            text = open(p, encoding="utf-8", errors="replace").read()
            anchors = set()
            mod = ""
            lines = text.splitlines()
            in_auto, toctree = False, False
            for i, line in enumerate(lines):
                m = re.match(r"\s*\.\. (?:current|auto)?module::\s*(\S+)", line)
                if m:
                    mod = m.group(1)
                    anchors.add("module-" + mod)
                if re.match(r"\s*\.\. autosummary::", line):
                    in_auto, toctree = True, False
                    continue
                if in_auto:
                    if re.match(r"\s*:toctree:", line):
                        toctree = True
                        continue
                    if re.match(r"\s*:\w+:", line) or not line.strip():
                        continue
                    if re.match(r"\s+[\w.~]+\s*$", line):
                        name = line.strip().lstrip("~")
                        full = name if name.startswith(("torch", "torchvision")) else f"{mod}.{name}" if mod else name
                        if toctree:
                            gen.add(full)
                        anchors.add(full)
                        continue
                    in_auto = False
                m = re.match(r"\s*\.\. (?:auto|py:)?(?:class|function|method|attribute|data|exception|autofunction|autoclass)::\s*([\w.]+)", line)
                if m:
                    n = m.group(1)
                    full = n if n.startswith(("torch", "torchvision")) or not mod else f"{mod}.{n}"
                    anchors.add(full)
                    anchors.add(n)
                # markdown and rst headings
                m = re.match(r"#{1,6}\s+(.*)", line)
                if m:
                    anchors.add(heading_slug(m.group(1)))
                if i + 1 < len(lines) and re.match(r"^[=\-~^\"']{3,}\s*$", lines[i + 1]) and line.strip():
                    anchors.add(heading_slug(line))
                m = re.match(r"\s*\(([\w\-.]+)\)=\s*$", line)   # myst label
                if m:
                    anchors.add(m.group(1))
                m = re.match(r"\s*\.\. _([\w\-.]+):", line)    # rst label
                if m:
                    anchors.add(m.group(1).lower())
            pages[page] = anchors
    return gen, pages


pt_gen, pt_pages = scan(PT)
tv_gen, tv_pages = scan(TV)

urls = {}
for root, _, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith(".md"):
            p = os.path.join(root, fn)
            for u in re.findall(r"https://docs\.pytorch\.org/[^\s)\]\"'>]+", open(p, encoding="utf-8").read()):
                urls.setdefault(u.rstrip(".,"), set()).add(os.path.relpath(p, ROOT))

bad, unchecked, ok = [], [], 0
for u in sorted(urls):
    path, _, anchor = u.partition("#")
    m = re.match(r"https://docs\.pytorch\.org/(docs|vision)/stable/(.+)\.html$", path)
    if not m or "/auto_examples/" in path:   # sphinx-gallery pages: not in sources
        unchecked.append(u)
        continue
    site, page = m.groups()
    gen, pages = (pt_gen, pt_pages) if site == "docs" else (tv_gen, tv_pages)
    g = re.match(r"(?:models/)?generated/(.+)$", page)
    if g:
        name = g.group(1)
        if name not in gen:
            bad.append((u, f"no generated page for {name}"))
            continue
        if anchor and not anchor.startswith(name + "."):
            bad.append((u, f"anchor {anchor} not a member of {name}"))
            continue
        ok += 1
        continue
    if page not in pages:
        bad.append((u, f"no page {page}"))
        continue
    if anchor and anchor not in pages[page] and anchor.lower() not in pages[page]:
        bad.append((u, f"anchor #{anchor} not found on {page}"))
        continue
    ok += 1

for u, why in bad:
    print(f"BAD  {u}\n     {why}  (in {', '.join(sorted(urls[u]))})")
for u in unchecked:
    print(f"SKIP {u}")
print(f"{len(urls)} unique URLs: {ok} ok, {len(bad)} bad, {len(unchecked)} not checked")
