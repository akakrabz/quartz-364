#!/usr/bin/env python3
"""Static checks on the assembled content, mimicking Quartz's rules.

1. Wikilinks [[target#anchor|alias]] and markdown links resolve to a page
   (by full path or unique basename, case-insensitive, like Quartz "shortest").
2. #anchors match a heading slug on the target page (github-slugger rules).
3. Every $...$ / $$...$$ expression compiles under KaTeX (strict, throwOnError),
   using the local katex.min.js.
4. Frontmatter parses; required keys present.
5. Raw-HTML blocks contain no blank lines (which would end the HTML block).
6. Callouts use only the types the site styles (custom.scss) or Quartz knows.
7. Table rows contain no bare "|" inside math (use \\lvert, \\rvert, \\mid).
8. Every <!-- fig:NAME --> marker is followed by its inlined <figure> line.
"""
import os, re, sys, json, subprocess, unicodedata, yaml

# usage: python3 tools/check.py [content-dir] [path/to/katex.min.js]
#   default content dir: ../content relative to this script; KaTeX path: node_modules/katex/dist/katex.min.js
#   (present after `npm ci`), or set the KATEX env var. Without KaTeX the math check is skipped.
#   --only a,b   report only errors on pages whose slug starts with one of these prefixes
#                (e.g. --only 1-models-of-computation/04,concepts/nfa); links are still resolved site-wide
ONLY = []
if "--only" in sys.argv:
    k = sys.argv.index("--only")
    ONLY = [x.strip().lower().removesuffix(".md").removeprefix("content/") for x in sys.argv[k + 1].split(",") if x.strip()]
    del sys.argv[k:k + 2]
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content")
KATEX = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("KATEX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "node_modules", "katex", "dist", "katex.min.js"))

# ---------------------------------------------------------------- slugs (github-slugger)
def gh_slug(s: str) -> str:
    s = s.lower()
    out = []
    for ch in s:
        if ch == " ":
            out.append("-")
        elif ch == "-" or ch == "_":
            out.append(ch)
        elif unicodedata.category(ch)[0] in ("L", "N") or unicodedata.category(ch) in ("Mn", "Mc"):
            out.append(ch)
        # else: dropped
    return "".join(out)

def strip_inline_md(s: str) -> str:
    # heading text as rendered: remove emphasis markers, code ticks, links -> text
    s = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]+)?\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]+)?\]\]", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"[*_`]", "", s)
    return s.strip()

# ---------------------------------------------------------------- collect pages
pages = {}   # slug (path without .md, lowercase) -> dict(text, headings)
for root, _, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith(".md"):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, ROOT)[:-3].replace(os.sep, "/")
            text = open(p, encoding="utf-8").read()
            pages[rel.lower()] = {"path": rel, "text": text}

errors, warnings = [], []
ALIASES = {}

# callout types: Quartz/Obsidian built-ins plus the site's own (quartz/styles/custom.scss)
CALLOUTS = {
    "note", "abstract", "summary", "tldr", "info", "todo", "tip", "hint", "important", "success", "check", "done",
    "question", "help", "faq", "warning", "caution", "attention", "failure", "fail", "missing", "danger", "error",
    "bug", "example", "quote", "cite",
    "definition", "key", "template", "recipe", "trap", "exam", "intuition", "proof", "derivation",
    "easy", "medium", "hard", "solution",
}

def strip_code(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text

for slug, pg in pages.items():
    text = pg["text"]
    # frontmatter
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        errors.append(f"{slug}: no frontmatter"); pg["fm"] = {}
    else:
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception as e:
            errors.append(f"{slug}: bad YAML frontmatter: {e}"); fm = {}
        pg["fm"] = fm
        if "title" not in fm: errors.append(f"{slug}: frontmatter missing title")
        if "description" not in fm: warnings.append(f"{slug}: frontmatter missing description")
        # Quartz writes a redirect page per alias at <folder>/<alias-slug>: an alias equal
        # to an existing page's slug overwrites that page with a redirect (to itself)
        d = os.path.dirname(slug)
        for a in fm.get("aliases") or []:
            a_slug = re.sub(r"[^\w\-/]+", "-", str(a).strip().lower().replace(" ", "-")).strip("-")
            target = f"{d}/{a_slug}" if d else a_slug
            if target in pages:
                errors.append(f"{slug}: alias {a!r} collides with page {target} (its redirect would replace the page)")
            alias_owner = ALIASES.setdefault(target, slug)
            if alias_owner != slug:
                errors.append(f"{slug}: alias {a!r} also used by {alias_owner} in the same folder")
    body = text[m.end():] if m else text
    pg["body"] = body
    # headings -> slugs (with github-slugger duplicate suffixes)
    seen = {}
    hs = []
    for line in re.sub(r"```.*?```", "", body, flags=re.S).splitlines():   # keep inline code: Quartz slugs it
        hm = re.match(r"^(#{1,6})\s+(.*)$", line)
        if hm:
            s = gh_slug(strip_inline_md(hm.group(2)))
            if s in seen:
                seen[s] += 1; s = f"{s}-{seen[s]}"
            else:
                seen[s] = 0
            hs.append(s)
    pg["headings"] = set(hs)
    # raw html blocks: no blank line inside <figure ...>...</figure> or <div ...>...</div>
    for tag in ("figure", "div"):
        for bm in re.finditer(rf"<{tag}[^>]*>.*?</{tag}>", body, re.S):
            if "\n\n" in bm.group(0):
                errors.append(f"{slug}: blank line inside <{tag}> block (would break the HTML block)")
    # callout types
    for cm in re.finditer(r"^(?:>\s*)+\[!([A-Za-z-]+)\]", body, re.M):
        if cm.group(1).lower() not in CALLOUTS:
            errors.append(f"{slug}: unknown callout type [!{cm.group(1)}]")
    # table rows: no bare | inside math
    for ln in strip_code(body).splitlines():
        t = ln.lstrip("> ").strip()
        if t.startswith("|"):
            for mm in re.finditer(r"\$(?!\$)(.+?)(?<!\\)\$", t.replace("\\|", "")):
                if "|" in mm.group(1):
                    errors.append(f"{slug}: '|' inside math in a table row (use \\lvert/\\rvert/\\mid): {mm.group(0)[:60]}")
    # figure markers
    blines = body.split("\n")
    for k, ln in enumerate(blines):
        fm_ = re.match(r"^<!--\s*fig:([a-z0-9-]+)\s*-->\s*$", ln)
        if fm_ and not (k + 1 < len(blines) and blines[k + 1].startswith('<figure class="cs-fig"')):
            errors.append(f"{slug}: figure marker fig:{fm_.group(1)} not followed by its figure (run tools/assemble.py)")

# ---------------------------------------------------------------- resolve links
ASSETS = {fn.lower() for _, _, fs in os.walk(ROOT) for fn in fs if not fn.endswith(".md")}
basenames = {}
for slug in pages:
    basenames.setdefault(slug.split("/")[-1], []).append(slug)

def resolve(target: str, from_slug: str):
    t = target.strip().lower()
    if t.endswith("/"):        # folder link -> folder index
        t = t + "index"
    t = re.sub(r"\.md$", "", t)
    if t in pages: return t
    # Quartz "shortest": multi-segment suffix match or unique basename
    if "/" in t:
        cands = [s for s in pages if s == t or s.endswith("/" + t)]
    else:
        cands = basenames.get(t, [])
    if len(cands) == 1: return cands[0]
    if len(cands) > 1: return ("AMBIGUOUS", cands)
    return None

WIKI = re.compile(r"(!?)\[\[([^\]\|#\\]*)\\?(#[^\]\|\\]+)?\\?(\|[^\]]*)?\]\]")
MDLINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")
nlinks = 0
for slug, pg in pages.items():
    body = strip_code(pg["body"])
    for m in WIKI.finditer(body):
        embed, target, anchor, alias = m.groups()
        nlinks += 1
        if target == "" and anchor:   # same-page anchor
            a = anchor[1:].lower()
            if a not in pg["headings"]: errors.append(f"{slug}: anchor {anchor} not found on same page")
            continue
        if re.search(r"\.(png|jpe?g|gif|svg|webp|pdf)$", target.strip(), re.I):
            if target.strip().split("/")[-1].lower() not in ASSETS:
                errors.append(f"{slug}: embedded file not found: [[{target}]]")
            continue
        r = resolve(target, slug)
        if r is None:
            errors.append(f"{slug}: wikilink target not found: [[{target}]]")
        elif isinstance(r, tuple):
            errors.append(f"{slug}: ambiguous wikilink [[{target}]] -> {r[1]}")
        elif anchor:
            a = anchor[1:].lower()
            if a not in pages[r]["headings"]:
                errors.append(f"{slug}: anchor {anchor} not on {r}; headings: {sorted(pages[r]['headings'])[:12]}...")
    for m in MDLINK.finditer(body):
        href = m.group(1)
        if href.startswith(("http://", "https://", "mailto:", "#")): continue
        if href.startswith("/static/"): continue
        r = resolve(href.lstrip("./"), slug)
        if r is None: errors.append(f"{slug}: markdown link target not found: {href}")

# ---------------------------------------------------------------- math validation with KaTeX
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")
exprs = []   # (slug, kind, tex)
for slug, pg in pages.items():
    body = strip_code(pg["body"])
    # inside callouts, strip the leading "> " so display blocks join
    body = re.sub(r"^>\s?", "", body, flags=re.M)
    for m in DISPLAY.finditer(body):
        exprs.append((slug, "display", m.group(1)))
    body2 = DISPLAY.sub(" ", body)
    for m in INLINE.finditer(body2):
        exprs.append((slug, "inline", m.group(1)))

node_script = r"""
const katex = require(process.argv[1]);
const exprs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = [];
for (const [slug, kind, tex] of exprs) {
  try {
    katex.renderToString(tex, { displayMode: kind === 'display', throwOnError: true, strict: 'warn', trust: false });
  } catch (e) {
    out.push([slug, kind, tex.slice(0, 120), String(e.message).slice(0, 200)]);
  }
}
console.log(JSON.stringify(out));
"""
if not os.path.exists(KATEX):
    warnings.append(f"KaTeX not found at {KATEX}; math validation skipped (run npm ci, or pass the path)")
    res = None
else:
    res = subprocess.run(["node", "-e", node_script, os.path.abspath(KATEX)], input=json.dumps(exprs), capture_output=True, text=True)
if res is None:
    pass
elif res.returncode != 0:
    errors.append("katex runner failed: " + res.stderr[:500])
else:
    for slug, kind, tex, msg in json.loads(res.stdout):
        errors.append(f"{slug}: KaTeX {kind} error in `{tex}` -> {msg}")
    stderr_warn = [l for l in res.stderr.splitlines() if "LaTeX-incompatible" in l or "strict" in l]
    for l in stderr_warn[:10]:
        warnings.append("katex strict: " + l[:200])

# ---------------------------------------------------------------- report
if ONLY:
    keep = lambda msg: any(msg.lower().startswith(p) for p in ONLY)
    errors = [e for e in errors if keep(e)]
    warnings = [w for w in warnings if keep(w) or w.startswith("KaTeX")]
print(f"pages: {len(pages)}, links checked: {nlinks}, math expressions: {len(exprs)}")
for w in warnings: print("WARN ", w)
for e in errors: print("ERROR", e)
print("OK" if not errors else f"{len(errors)} error(s)")
sys.exit(1 if errors else 0)
