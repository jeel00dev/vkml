#!/usr/bin/env python3
"""Every file, symbol and constant the documentation cites must actually exist.

WHY THIS EXISTS. Writing the reduction page I stated that `pairwise_sum` lives
in `src/backend/cpu/iterate.h`. It lives in `src/backend/cpu/reduce.h`. The
source comment I was reading from was correct; I mis-attributed it while
summarising, and nothing would have caught that -- the page rendered, the links
resolved, the examples ran, and the sentence was simply false.

Prose is where documentation rots fastest, because unlike a link there is
nothing structural to break. This walks every backticked path, `path:line`
reference and named C++ constant in web/content/ and checks it against the tree:

  path.h              -> the file exists
  path.cpp:123        -> the file exists AND has at least that many lines
  `kSomeConstant`     -> the identifier appears somewhere in src/ or include/

The constant check is deliberately loose. It cannot tell a real citation from a
coincidence, and it is not trying to: it catches the case that actually happens,
which is a name that has been renamed or removed entirely.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

from content import CLASSES, PROSE  # noqa: E402

# A backticked token that looks like a repository path, optionally with :line.
#
# Two forms are accepted and the distinction matters. A token containing a
# separator (`src/api/ops.cpp`) is unambiguously repo-relative. A bare filename
# is only treated as a repository reference when it carries a SOURCE extension,
# because documentation legitimately names files that are not in the tree --
# `vkml.json` is a member inside a checkpoint archive, and the first version of
# this gate reported it as missing. A gate that cries wolf gets switched off.
PATH_RE = re.compile(
    r"`("
    r"(?:[\w.-]+/)+[\w.-]+\.(?:h|cpp|comp|glsl|py|json|md|txt)"   # has a separator
    r"|[\w.-]+\.(?:h|cpp|comp|glsl|py)"                            # bare, source only
    r")(?::(\d+))?`")
# A backticked identifier in the project's constant style.
# Matches the identifier at the START of the backticked span, so
# `kPairwiseBlock = 32` is checked as kPairwiseBlock rather than skipped.
CONST_RE = re.compile(r"`(k[A-Z]\w+)\b")

SEARCH_DIRS = ["src", "include", "shaders", "python", "scripts", "docs", "tests", "web"]


def prose_fields(entry: dict):
    for key in ("summary", "detail", "note", "warning", "tip", "returns"):
        if entry.get(key):
            yield key, entry[key]
    for name, ptype, desc in entry.get("params", []):
        yield f"param {name}", desc


def resolve(rel: str) -> Path | None:
    """A cited path may be repo-relative or a bare basename."""
    direct = ROOT / rel
    if direct.exists():
        return direct
    if "/" not in rel:
        for d in SEARCH_DIRS:
            hits = list((ROOT / d).rglob(rel))
            if hits:
                return hits[0]
    return None


def main() -> int:
    sources = ""
    for d in ("src", "include", "shaders"):
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in {".h", ".cpp", ".comp", ".glsl"}:
                sources += p.read_text(errors="ignore")

    problems, checked_paths, checked_consts = [], 0, 0

    for op, entry in sorted(PROSE.items()):
        for where, text in prose_fields(entry):
            for rel, line in PATH_RE.findall(text):
                checked_paths += 1
                target = resolve(rel)
                if target is None:
                    problems.append(f"{op} ({where}): no such file `{rel}`")
                elif line:
                    n = target.read_text(errors="ignore").count("\n") + 1
                    if int(line) > n:
                        problems.append(
                            f"{op} ({where}): `{rel}:{line}` but the file has {n} lines")

            for const in CONST_RE.findall(text):
                checked_consts += 1
                if const not in sources:
                    problems.append(f"{op} ({where}): `{const}` appears nowhere in the sources")

    print(f"  {checked_paths} path references, {checked_consts} constant references")
    problems += unlisted_native_members()
    problems += stale_docs_references()
    problems += unrendered_markup()
    problems += headings_without_ids()
    problems += oversized_site_files()
    problems += oversized_sidebar()
    problems += capability_reasons_match_the_code()
    if problems:
        print(f"  {len(problems)} problems:")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  all resolve")
    return 0


# Paths named in prose that deliberately do not exist. Each needs a reason, so
# that adding one is a decision rather than a way to silence the check.
DOC_PATH_EXCEPTIONS = {
    # SPLIT_K_DESIGN.md quotes the name used in the task brief precisely in order
    # to say the tree does not use it. The sentence is about the mismatch.
    "docs/PERFORMANCE_MODEL.md",
    # ARCHITECTURE.md 2.3 surveys stable-diffusion.cpp, which is NOT one of the
    # six projects vendored under third_party/reference, so its paths cannot be
    # resolved here. The name happens to look like a vkml path, which is the
    # only reason it needs saying.
    "src/core/ggml_graph_cut.cpp",
}

# Everything worth indexing when resolving a documented path, INCLUDING the
# vendored reference trees. The design documents cite CUTLASS, llama.cpp,
# tinygrad and Tensile constantly -- those citations are the evidence for the
# decisions, so a checker that cannot see them reports the project's most
# carefully sourced claims as broken.
INDEX_SKIP = {".git", ".venv", "build", "node_modules", "_site", "actions-runner",
              "__pycache__"}


def _file_index() -> dict:
    index: dict[str, list] = {}
    for p in ROOT.rglob("*"):
        if p.is_file() and not INDEX_SKIP & set(p.parts):
            index.setdefault(p.name, []).append(p)
    return index


def _candidates(rel: str, index: dict) -> list:
    """Every file a documented path could be naming, best match first.

    ALL of them, not the best one, and that is the whole difficulty. A bare
    `gemm.h` matches a dozen files in the vendored CUTLASS; every candidate ties
    on any suffix score, so choosing one is arbitrary, and choosing wrong turns a
    correct citation into a false alarm. `gemm.h:346` is exactly
    `semaphore.wait(threadblock_tile_offset.k())` in cutlass/gemm/kernel/gemm.h,
    but cutlass/gemm/gemm.h sorts earlier and has 153 lines.

    Two earlier attempts at this check reported 58 and then 12 stale references
    in the documents. The true number was zero: every one was this resolution
    being too eager. So the question asked is the weaker, honest one -- does the
    tree contain SOME file of this name that satisfies the citation -- which
    still catches what actually happens, a file renamed or deleted outright.
    """
    direct = ROOT / rel
    if direct.is_file():
        return [direct]
    # `.../` is how the docs elide a long vendored path; the tail is what matters.
    tail = rel.rsplit(".../", 1)[-1]
    cands = index.get(Path(tail).name, [])
    return sorted(cands, key=lambda c: (not str(c).endswith(tail), len(str(c))))


def stale_docs_references():
    """Every `path` and `path:line` in the hand-written docs must resolve.

    check_docs_examples and this file's own path check cover the GENERATED site.
    The design documents -- the ADRs, ARCHITECTURE, THEORY, MEASUREMENT-AUDIT --
    were never checked by anything, and they are the older, longer and more
    heavily cross-referenced half. They were found clean; nothing was keeping
    them that way.

    Documents that declare themselves superseded are skipped: they describe a
    design that was rejected, so naming files that were never written is what
    they are for.
    """
    out = []
    index = _file_index()
    docs = sorted(ROOT.glob("docs/**/*.md")) + [ROOT / n for n in
                                                ("README.md", "CONTRIBUTING.md", "CLAUDE.md")]
    for doc in docs:
        if not doc.is_file():
            continue
        text = doc.read_text(errors="ignore")
        if "**SUPERSEDED" in text[:2000]:
            continue
        for rel, line in PATH_RE.findall(text):
            if rel in DOC_PATH_EXCEPTIONS:
                continue
            cands = _candidates(rel, index)
            where = doc.relative_to(ROOT)
            if not cands:
                out.append(f"{where}: `{rel}` names no file in the tree")
            elif line and not any(
                    int(line) <= c.read_text(errors="ignore").count("\n") + 1
                    for c in cands):
                out.append(f"{where}: `{rel}:{line}` but no file of that name is "
                           f"that long (closest: {cands[0].relative_to(ROOT)}, "
                           f"{cands[0].read_text(errors='ignore').count(chr(10)) + 1} lines)")
    return out


def unlisted_native_members():
    """Every public member of a nanobind class must appear in a group on its page.

    The page for a class defined in C++ is built by introspecting the BUILT
    module, so a newly bound member appears on it the moment it is bound --
    but only in the catch-all "Other members" bucket, silently and with no
    prose. That is how the Python `Tensor` came to have six members documented
    nowhere at all (`T`, `astype`, `max`, `min`, `numpy`, `view`) while the
    page still looked complete.

    Checked against `dir()` on the real class rather than against a list kept
    here, because a list kept here would need the same discipline it is meant
    to enforce.
    """
    import importlib
    out = []
    for name, spec in sorted(CLASSES.items()):
        if spec.get("lang") != "native":
            continue
        try:
            cls = getattr(importlib.import_module("vkml"), spec.get("symbol", name))
        except (ImportError, AttributeError):
            out.append(f"{name}: cannot import to verify its members")
            continue
        grouped = {m for _, members in spec.get("groups", []) for m in members}
        for attr in sorted(dir(cls)):
            if attr.startswith("_") and attr not in spec.get("extra_members", ()):
                continue
            if attr not in grouped:
                out.append(f"{name}: `{attr}` is bound but listed in no group "
                           f"-- add it to a group in web/content/classes.py")
    return out



def unrendered_markup():
    """No authored markup may survive into the built HTML.

    Two ways it did, both invisible in ordinary use, which is why both lasted.

    In the BODY, single-asterisk emphasis was never implemented. Authors wrote
    it anyway -- `*GEMV*`, `*Split-K*`, `*Nesterov*` and five more -- so five API
    pages printed literal asterisks mid-sentence, including in the paragraph
    explaining how a matmul picks its kernel.

    In the HEAD, `<meta name="description">` reused the same authored summary
    the body renders, and nothing renders a meta tag. Seven pages described
    themselves to search engines and link previews with the markdown still in
    them. Nobody reads their own meta tags, so nobody saw it.

    Deliberately checks the OUTPUT rather than the sources: the question is what
    a reader receives, and it stays true whatever the renderer gains next.
    """
    import re as _re
    site = ROOT / "web" / "_site"
    if not site.exists():
        return ["web/_site does not exist; run python web/build.py first"]

    out = []
    # A `*word*` span in text, but not `**bold**`, not `*args`, and not inside a
    # tag. Text is taken between tags so a match cannot be an attribute value.
    body_em = _re.compile(r">([^<]*?)<")
    em_span = _re.compile(r"(?<!\*)\*(?!\s|\*)([^*<>]{2,40}?)(?<!\s)\*(?!\*)")
    code_span = _re.compile(r"`([^`\n]{1,60})`")
    meta_md = _re.compile(r'<meta[^>]*name="description"[^>]*content="([^"]*)"')

    for page in sorted(site.glob("*.html")):
        text = page.read_text(errors="ignore")
        for m in meta_md.finditer(text):
            if "`" in m.group(1) or "**" in m.group(1):
                out.append(f"{page.name}: meta description still contains markdown: "
                           f"{m.group(1)[:60]!r}")
        for chunk in body_em.findall(text):
            hit = em_span.search(chunk)
            if hit:
                out.append(f"{page.name}: unrendered emphasis in the body: "
                           f"*{hit.group(1)[:40]}*")
                break
            # Backticks too. The first version of this check looked only for
            # emphasis, so a generated page rendered a heading as
            # "A lazy `detach()`" with the backticks intact and the gate passed.
            # Same defect class, different delimiter -- authored markup reaching
            # the reader unrendered.
            code = code_span.search(chunk)
            if code:
                out.append(f"{page.name}: unrendered code span in the body: "
                           f"`{code.group(1)[:40]}`")
                break
    return out


def headings_without_ids():
    """Every CONTENT heading must be linkable.

    Guide pages are authored as raw HTML, so whether a section could be linked
    depended on whether the author typed an id. It split down the middle:
    guide_perf.py wrote them and worked, guide.py did not, so "Get started" had
    seven sections and "Concepts" six that no deep link, no contents rail and no
    scroll-spy could reach -- on the two pages a newcomer opens first.

    Both pages rendered a contents rail with nothing in it, which is a worse
    signal than having no rail at all, and it stayed invisible until the rail
    was restored at laptop widths.

    The sidebar's own `<h2>Documentation</h2>` is chrome, not content, and is
    the one heading legitimately without an id -- so it is matched by name
    rather than by counting, which would have hidden a real regression behind an
    expected total.
    """
    site = ROOT / "web" / "_site"
    if not site.exists():
        return ["web/_site does not exist; run python web/build.py first"]
    out = []
    for page in sorted(site.glob("*.html")):
        text = page.read_text(errors="ignore")
        # Scope to <main class="content">. The sidebar carries its own headings
        # -- "Documentation", "Guide", "Classes", "API reference" -- which are
        # chrome and correctly have no id; a whole-document scan reports those
        # as defects, which is the same mistake that made the research probe
        # measure a nav heading as if it were the page's hierarchy.
        main = re.search(r'<main class="content">(.*?)</main>', text, re.S)
        if not main:
            out.append(f"{page.name}: no <main class=\"content\"> to check")
            continue
        # A heading inside a link card is the card's title, and the card itself
        # navigates -- it needs no anchor of its own. The API index renders each
        # operator group that way.
        region = re.sub(r'<a class="card".*?</a>', "", main.group(1), flags=re.S)
        for m in re.finditer(r"<(h2|h3)([^>]*)>(.*?)</\1>", region, re.S):
            if "id=" in m.group(2):
                continue
            label = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            out.append(f"{page.name}: <{m.group(1)}> {label[:40]!r} has no id, "
                       f"so it cannot be linked or listed")
    return out


# What a single file in the built site may weigh. Not a round number: the
# largest legitimate asset is the 1200x630 social card at 65 kB, and the budget
# sits just above it so anything bigger is a decision rather than an accident.
SITE_FILE_BUDGET_KB = 80


def oversized_site_files():
    """No binary asset in the built site may exceed the budget.

    The master logo -- 1536x1024, 2.4 MB -- was being served as the favicon, in
    the topbar at 25.6px and in the hero at 120px. Against 54.7 kB of HTML, CSS
    and JavaScript it was 97.8% of every page load, and nothing said so: the
    page looked right, so the weight was invisible.

    Derived sizes now come from scripts/make_assets.py and the landing page
    transfers 104 kB. This keeps it that way.
    """
    site = ROOT / "web" / "_site"
    if not site.exists():
        return ["web/_site does not exist; run python web/build.py first"]
    # BINARY assets only. Markup and stylesheets are text, compress about 5:1
    # in transit, and grow for a reason -- api-element-wise.html is 94 kB
    # because it documents twenty-one operators in full, which is the page doing
    # its job. Holding those to an image budget would report the reference
    # getting more complete as a regression.
    binary = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
              ".woff", ".woff2", ".ttf", ".otf", ".mp4", ".zip"}
    out = []
    for f in sorted(site.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in binary:
            continue
        kb = f.stat().st_size / 1000
        if kb > SITE_FILE_BUDGET_KB:
            out.append(f"{f.relative_to(site)} is {kb:.0f} kB, over the "
                       f"{SITE_FILE_BUDGET_KB} kB budget -- derive a smaller "
                       f"asset with scripts/make_assets.py, or raise the budget "
                       f"deliberately")
    return out


# What the section tree may show at once. Not a style preference: a nav that
# does not fit on a screen cannot be scanned, and the reader falls back to
# search or to guessing.
MAX_VISIBLE_NAV_LINKS = 34


def oversized_sidebar():
    """The section tree must stay scannable.

    It grew to 144 links and 2.6 screens at 1440x900 because `sidenav()` walked
    PAGES, CLASSES and GROUPS -- the containers the site is generated from -- so
    every operator and every class appeared on every page. Grouping alone did
    not fix it: with the classes grouped but all expanded it got WORSE, 52
    visible links and 3 screens, because sixteen layer classes are as long as
    the operator list they replaced.

    Counted from the markup rather than rendered, so this runs without a
    browser: a link is hidden if it sits inside a `grp closed` block.
    """
    site = ROOT / "web" / "_site"
    if not site.exists():
        return ["web/_site does not exist; run python web/build.py first"]
    page = site / "get-started.html"
    if not page.is_file():
        return ["get-started.html missing; cannot check the section tree"]
    m = re.search(r'<aside class="sidenav".*?</aside>', page.read_text(), re.S)
    if not m:
        return ["no .sidenav in get-started.html"]
    nav = m.group(0)
    # Drop collapsed groups, then count what is left.
    visible = re.sub(r'<div class="grp closed">.*?</div>', "", nav, flags=re.S)
    n = len(re.findall(r"<a\b", visible))
    if n > MAX_VISIBLE_NAV_LINKS:
        return [f"the section tree shows {n} links at once, over the "
                f"{MAX_VISIBLE_NAV_LINKS} budget -- group them, or collapse a "
                f"section that is not the one being read"]
    return []


def capability_reasons_match_the_code():
    """Every extracted gap needs a declared reason, and every reason a real gap.

    BOTH DIRECTIONS, and the second is the one that matters over time.

    Forward: an operator that loses Vulkan support, or an OpKind added without a
    gradient rule, must not appear on the limitations page as a bare row with no
    explanation -- an unexplained gap is the thing that makes a project look
    unaware of its own surface.

    Backward: a reason declared for a gap that no longer exists is worse. It is
    the site telling a reader that something is unsupported after somebody
    implemented it, in a table that looks machine-generated and therefore
    trustworthy. Nothing else in this repository would catch that: the tests
    would pass, the page would build, and the sentence would be false.

    This is the same shape as the C++/Python parity allow-list and the tolerance
    policy -- facts extracted, exceptions declared, and a check that fails when
    they disagree.
    """
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import research as R
        from content.capabilities import BACKEND_REASONS, GRADIENT_REASONS
        import build as B
    except Exception as e:                      # noqa: BLE001
        return [f"cannot load the capability data: {type(e).__name__}: {e}"]

    out = []
    kinds = R.op_kinds()
    ruled = set(R.autograd_rules())
    no_rule = {k for k in kinds if k not in ruled}

    for k in sorted(no_rule - set(GRADIENT_REASONS)):
        out.append(f"OpKind `{k}` has no gradient rule and no declared reason; "
                   f"add one to web/content/capabilities.py")
    for k in sorted(set(GRADIENT_REASONS) - no_rule):
        out.append(f"web/content/capabilities.py explains why `{k}` has no gradient "
                   f"rule, but it HAS one now -- the limitations page is telling "
                   f"readers something untrue; remove the entry")

    cpu_only = {n for n, f in B.FACTS.items() if not f.on_vulkan}
    for n in sorted(cpu_only - set(BACKEND_REASONS)):
        out.append(f"`{n}` does not run on Vulkan and has no declared reason")
    for n in sorted(set(BACKEND_REASONS) - cpu_only):
        out.append(f"web/content/capabilities.py explains why `{n}` is CPU-only, "
                   f"but it runs on Vulkan now; remove the entry")
    return out

if __name__ == "__main__":
    sys.exit(main())
