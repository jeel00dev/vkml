#!/usr/bin/env python3
"""Build the vkML documentation site.

TWO SOURCES, DELIBERATELY.

Signatures, argument names, defaults and the inventory are read from the
IMPORTED MODULE at build time. Nothing about the API surface is written down
twice, so a page cannot describe a signature the library no longer has -- the
failure mode of every hand-maintained API doc.

Prose, parameter descriptions and examples come from `web/content/`, keyed by the
same names. They cannot be derived: most public functions carry a docstring that
is only their signature repeated, so generating from docstrings alone would
produce a reference with no explanation in it.

Where prose is missing the page SAYS SO, in place, and the run prints coverage.

    python web/build.py            # -> web/_site
    python web/build.py --serve    # build, then serve on :8000
"""

from __future__ import annotations

import argparse
import html
import inspect
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
OUT = WEB / "_site"

# Absolute, because og:image and og:url are fetched by a crawler that has no
# page context to resolve a relative path against -- a relative og:image is
# the usual reason a link preview renders blank.
SITE_URL = "https://jeel00dev.github.io/vkml"

sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(WEB))

import vkml as V  # noqa: E402
import diagrams as DIA  # noqa: E402
import research as R  # noqa: E402
from content import CLASSES, PAGES, PROSE  # noqa: E402

GROUPS: list[tuple[str, list[str]]] = [
    (
        "Creation",
        ["tensor", "zeros", "ones", "full", "arange", "rand", "from_numpy", "asarray",
         "zeros_like", "ones_like", "full_like"],
    ),
    (
        "Element-wise",
        [
            "abs",
            "neg",
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "square",
            "sign",
            "sin",
            "cos",
            "tanh",
            "sigmoid",
            "erf",
            "erfc",
            "relu",
            "gelu",
            "silu",
            "clamp",
            "clamp_min",
            "clamp_max",
        ],
    ),
    ("Arithmetic", ["add", "sub", "mul", "div", "pow", "maximum", "minimum"]),
    (
        "Comparison",
        [
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "where",
        ],
    ),
    ("Reduction", ["sum", "mean", "prod", "amax", "amin", "argmax", "argmin"]),
    (
        "Shape & indexing",
        [
            "cat",
            "tril",
            "triu",
            "masked_fill",
            "index_select",
            "scatter_add",
            "im2col",
            "col2im",
            "detach",
        ],
    ),
    (
        "Linear algebra & NN",
        [
            "matmul",
            "softmax",
            "log_softmax",
            "layer_norm",
            "rms_norm",
            "batch_norm",
            "dropout",
            "conv2d",
            "max_pool2d",
            "avg_pool2d",
        ],
    ),
    (
        "Losses",
        [
            "cross_entropy",
            "mse_loss",
            "huber_loss",
            "kl_div",
            "binary_cross_entropy_with_logits",
        ],
    ),
    ("Autograd & execution", ["backward", "realize", "set_eager", "is_eager"]),
    ("Serialization", ["save", "load", "save_module", "load_module"]),
    (
        "Devices & introspection",
        [
            "init_vulkan",
            "available_devices",
            "best_device",
            "vulkan_available",
            "vulkan_device_count",
            "vulkan_device_names",
            "vulkan_device_reports",
            "vulkan_capabilities",
            "vulkan_stats",
            "vulkan_pipeline_stats",
            "vulkan_timestamps_supported",
            "vulkan_unavailable_reason",
            "vulkan_last_profile",
            "vulkan_submit_ms",
            "vulkan_set_profiling",
            "vulkan_set_subgroup_override",
            "set_log_level",
        ],
    ),
    (
        # Its own group rather than folded into introspection: these answer
        # "why did it do that", which is a different question from "what is
        # this device" and is not Vulkan-specific.
        "Explaining what the engine chose",
        [
            "configuration",
            "record_decisions",
            "decisions",
            "decisions_published",
            "stop_recording_decisions",
        ],
    ),
]

ADR_MENTIONS = R.adr_mentions()
BENCH_MENTIONS = R.benchmark_mentions()
FACTS = R.gather(sorted({n for _, names in GROUPS for n in names}))
CPU_ONLY = {"prod"}
ALL_NAMES = {n for _, names in GROUPS for n in names}


def group_slug(name: str) -> str:
    return "api-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Which page each operator lives on. Cross-references are rewritten through this,
# so splitting the reference into per-category pages -- the way PyTorch does --
# does not leave every inline `matmul` pointing at a page it is no longer on.
PAGE_OF: dict[str, str] = {n: group_slug(g) for g, names in GROUPS for n in names}


# ---------------------------------------------------------- syntax colours --

KEYWORDS = {
    "import",
    "from",
    "as",
    "def",
    "class",
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "in",
    "not",
    "and",
    "or",
    "is",
    "None",
    "True",
    "False",
    "with",
    "try",
    "except",
    "raise",
    "assert",
    "lambda",
    "yield",
    "pass",
    "break",
    "continue",
    "global",
    "nonlocal",
    "del",
    "await",
    "async",
}
BUILTINS = {
    "print",
    "len",
    "range",
    "float",
    "int",
    "str",
    "bool",
    "list",
    "dict",
    "tuple",
    "set",
    "type",
    "isinstance",
    "enumerate",
    "zip",
    "sorted",
    "next",
    "iter",
    "open",
}

# One pass, ordered so that a keyword inside a string or comment is never seen as
# a keyword -- strings and comments are matched first and consume their content.
PY_TOKEN = re.compile(
    r"(?P<c>#[^\n]*)"
    r"|(?P<s>\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*')"
    r"|(?P<n>\b\d+\.?\d*(?:[eE][-+]?\d+)?\b)"
    r"|(?P<f>\b[A-Za-z_]\w*(?=\s*\())"
    r"|(?P<w>\b[A-Za-z_]\w*\b)",
    re.S,
)


def highlight(src: str) -> str:
    """Colour Python source. Escapes as it goes, so callers must NOT pre-escape."""
    out, last = [], 0
    for m in PY_TOKEN.finditer(src):
        out.append(html.escape(src[last : m.start()]))
        text = html.escape(m.group(0))
        if m.lastgroup == "w":
            word = m.group(0)
            if word in KEYWORDS:
                out.append(f'<span class="tok-k">{text}</span>')
            elif word in BUILTINS:
                out.append(f'<span class="tok-b">{text}</span>')
            else:
                out.append(text)
        else:
            out.append(f'<span class="tok-{m.lastgroup}">{text}</span>')
        last = m.end()
    out.append(html.escape(src[last:]))
    return "".join(out)


REPL_LINE = re.compile(r"^(>>> |\.\.\. )(.*)$")


RAW_PRE = re.compile(r"<pre><code>(.*?)</code></pre>", re.S)


def highlight_raw_blocks(html_text: str) -> str:
    """Colour <pre><code> written by hand in web/content/.

    WHY. Content files may contain literal HTML, and a hand-written
    <pre><code> never reaches code_block(), so it never reaches highlight().
    That is how index.html shipped with ZERO token spans while the rest of the
    site had 1291 -- the highlighter was fine; the content walked around it.

    Fixing it here rather than in the content keeps ONE renderer: any block
    added by hand in future is coloured for free instead of depending on the
    author remembering. The source inside is already escaped, so it is
    unescaped before tokenising and re-escaped by highlight() as it goes.
    """
    def paint(m: re.Match) -> str:
        src = html.unescape(m.group(1))
        return f"<pre><code>{highlight(src)}</code></pre>"

    return RAW_PRE.sub(paint, html_text)


def code_block(src: str, repl: bool = False) -> str:
    src = src.strip("\n")
    copy = '<button class="copy" type="button">copy</button>'
    if not repl:
        return f"<pre>{copy}<code>{highlight(src)}</code></pre>"
    rows = []
    for line in src.split("\n"):
        m = REPL_LINE.match(line)
        if m:
            rows.append(
                f'<span class="p">{html.escape(m.group(1))}</span>'
                f"{highlight(m.group(2))}"
            )
        else:
            rows.append(f'<span class="o">{html.escape(line)}</span>')
    return f'<pre class="repl">{copy}<code>' + "\n".join(rows) + "</code></pre>"


# ------------------------------------------------------------------ markup --


def with_diagrams(body: str, where: str) -> str:
    """Replace {{diagram:name}} with the SVG that `diagrams.py` generates.

    A page asks for a diagram by name and gets whatever the tree says today.
    An unknown name raises rather than rendering the placeholder as visible
    text -- a typo should stop the build, not ship as `{{diagram:layer_stak}}`
    in the middle of a paragraph.
    """
    def sub(m: re.Match) -> str:
        name = m.group(1)
        fn = getattr(DIA, name, None)
        if fn is None:
            raise SystemExit(f"{where}: no diagram named {name!r} in web/diagrams.py")
        return fn()

    return re.sub(r"\{\{diagram:(\w+)\}\}", sub, body)


def with_heading_ids(body: str) -> str:
    """Give every h2/h3 an id, leaving hand-written ones exactly as they are.

    Guide pages are authored as raw HTML, so whether a heading was linkable came
    down to whether whoever typed it remembered. It split: guide_perf.py writes
    `<h2 id="machine">` and works, while guide.py writes `<h2>` and does not --
    so "Get started" had seven sections and "Concepts" six that no link, no
    contents rail and no scroll-spy could reach. Those are the first two pages a
    newcomer opens, and the pages most likely to be linked from an issue.

    Restoring the contents rail at laptop widths is what made it visible: both
    pages rendered the rail with nothing in it, which reads worse than not
    having one.

    Existing ids are never rewritten -- any link already shared keeps working --
    and slugs are deduplicated within the page so two sections of the same name
    cannot make one of them unreachable.
    """
    seen: set[str] = set(re.findall(r'<h[1-6][^>]*\bid="([^"]+)"', body))

    def fix(m: re.Match) -> str:
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        if "id=" in attrs:
            return m.group(0)
        text = re.sub(r"<[^>]+>", "", inner)
        slug = re.sub(r"[^a-z0-9]+", "-", html.unescape(text).lower()).strip("-")
        if not slug:
            return m.group(0)
        base, n = slug, 2
        while slug in seen:
            slug, n = f"{base}-{n}", n + 1
        seen.add(slug)
        return f'<{tag}{attrs} id="{slug}">{inner}</{tag}>'

    return re.sub(r"<(h[23])([^>]*)>(.*?)</\1>", fix, body, flags=re.S)


def plain(text: str) -> str:
    """Authored markdown reduced to the plain sentence inside it.

    The page description reuses the same authored summary the body renders, and
    a `<meta name="description">` is not rendered -- so seven pages were
    advertising themselves to search engines and link previews as
    `Adam with **decoupled** weight decay, matching \\`torch.optim.AdamW\\`.`
    with the markup intact. Nobody sees that on the page, which is why it
    survived.
    """
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links -> their text
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*(?!\s)([^*<>]+?)(?<!\s)\*", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def inline(text: str, xref: bool = True) -> str:
    """The small markdown subset the content files use.

    `name` becomes a LINK when it names another documented entry, so a reference
    to `matmul` inside a paragraph navigates the way PyTorch's do, without the
    author writing a link by hand every time.
    """
    t = html.escape(text)

    def code(m: re.Match) -> str:
        body = m.group(1)
        if xref and body in ALL_NAMES:
            return f'<a href="{PAGE_OF[body]}.html#{body}"><code>{body}</code></a>'
        # Classes too. This linked the 102 operators and not the 27 classes, so
        # 65 mentions of `Tensor`, `Module`, `Adam`, `DataLoader` and the rest
        # rendered as dead chips -- and 26 of the 27 class pages had no prose
        # anywhere on the site linking to them. They were reachable only from
        # the sidebar and search, which is not how a reader meets a name: they
        # meet it mid-sentence, on a page about something else.
        #
        # Matched on the exact key, which is what keeps `Tensor` (C++) and
        # `vkml.Tensor` (Python) apart -- two real pages documenting two real
        # surfaces, and guessing between them would send Python readers to the
        # wrong one.
        if xref and body in CLASSES:
            return f'<a href="{class_slug(body)}.html"><code>{body}</code></a>'
        return f"<code>{body}</code>"

    t = re.sub(r"`([^`]+)`", code, t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    # Single-asterisk emphasis, AFTER bold so `**x**` is already gone, and after
    # code so a `*` inside backticks is inside a tag by now.
    #
    # Authors were already writing it -- `*GEMV*`, `*Split-K*`, `*Nesterov*` and
    # five others -- and it rendered as literal asterisks on five API pages,
    # including mid-sentence in the kernel-selection explanation.
    #
    # The character class excludes `<` and `>`, which is what keeps this from
    # crossing a tag boundary and wrapping a generated <code> span. Requiring a
    # non-space next to each delimiter is what leaves `a * b` and a trailing
    # `Node*` alone.
    t = re.sub(r"\*(?!\s)([^*<>]+?)(?<!\s)\*", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def paragraphs(text: str) -> str:
    """Blank-line separated blocks. A block whose lines all start with '- ' is a list."""
    out = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if all(ln.lstrip().startswith("- ") for ln in lines):
            items = "".join(f"<li>{inline(ln.lstrip()[2:])}</li>" for ln in lines)
            out.append(f"<ul>{items}</ul>")
        else:
            out.append(f"<p>{inline(block)}</p>")
    return "".join(out)


def signature_of(name: str) -> str:
    fn = getattr(V, name)
    first = (fn.__doc__ or "").strip().split("\n", 1)[0]
    if first.startswith(f"{name}("):
        sig = first
    else:
        try:
            sig = f"{name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            sig = f"{name}(...)"
    sig = sig.replace("vkml._vkml_core.", "")
    return re.sub(r"collections\.abc\.Sequence\[int\]", "Sequence[int]", sig)


def render_signature(name: str, sig: str) -> str:
    body = html.escape(sig)
    body = body.replace(f"{name}(", f'<span class="name">{name}</span>(', 1)
    body = re.sub(r"-&gt; (.+)$", r'<span class="ret">→ \1</span>', body)
    body = re.sub(r"(\b[a-z_][a-z0-9_]*)(?=:)", r'<span class="param">\1</span>', body)
    return f'<div class="sig">{body}</div>'


def docstring_prose(name: str) -> str:
    doc = (getattr(V, name).__doc__ or "").strip()
    lines = doc.split("\n")
    if lines and lines[0].startswith(f"{name}("):
        lines = lines[1:]
    return "\n".join(lines).strip()


def admon(kind: str, text: str) -> str:
    icon = {"warning": "⚠", "note": "ⓘ", "tip": "★"}.get(kind, "ⓘ")
    cls = {"warning": "warn", "tip": "tip"}.get(kind, "note")
    return (f'<div class="admon {cls}"><span class="label">{icon} {kind.title()}</span>'
            f'<div class="body">{paragraphs(text)}</div></div>')


def impl_table(f) -> str:
    """Where this operator actually lives. Every row is extracted, not written.

    A row with no answer says so rather than being dropped: "no gradient rule"
    is information, and an absent row would read as an oversight.
    """
    rows = []
    if f.decls:
        d = f.decls[0]
        rows.append(("Declared in", f'<a href="{d.url}"><code>{d.path}:{d.line}</code></a>'))
    if f.op_kind:
        rows.append(("Graph node", f"<code>OpKind::{f.op_kind}</code>"))

    if f.cpu_kernel:
        path, line = f.cpu_kernel
        rows.append(("CPU kernel",
                     f'<a href="{R.REPO_URL}/{path}#L{line}"><code>{path}:{line}</code></a>'))
    else:
        rows.append(("CPU kernel", '<span class="none">composed from other operators</span>'))

    if f.shader:
        sh = f.shader
        extra = (f" · {len(sh['spec_constants'])} specialisation constants"
                 if sh["spec_constants"] else "")
        rows.append(("Vulkan shader",
                     f'<a href="{sh["url"]}"><code>{sh["path"]}</code></a> '
                     f'<span class="none">({sh["lines"]} lines{extra})</span>'))
    elif f.on_vulkan:
        rows.append(("Vulkan shader",
                     '<span class="none">composed, or dispatched through a shared kernel</span>'))
    else:
        rows.append(("Vulkan shader",
                     '<span class="none">not implemented — raises NotImplementedError</span>'))

    if f.has_grad:
        rows.append(("Gradient rule",
                     f'<a href="{R.REPO_URL}/src/autograd/autograd.cpp#L{f.grad_line}">'
                     f"<code>autograd.cpp:{f.grad_line}</code></a>"))
    else:
        rows.append(("Gradient rule",
                     '<span class="none">none — backward through it raises</span>'))

    # Decisions, measurements and history: the three destinations the table
    # named nowhere, so a reader who wanted to know WHY an operator behaves as
    # it does had to already know which ADR to open.
    adrs = ADR_MENTIONS.get(f.name, [])
    if adrs:
        rows.append(("Decisions", " ".join(
            f'<a href="{R.REPO_URL}/{path}">{html.escape(title.split(" — ")[0])}</a>'
            for path, title in adrs)))
    marks = BENCH_MENTIONS.get(f.name, [])
    if marks:
        rows.append(("Benchmarked", " · ".join(
            f"<code>{html.escape(m)}</code>" for m in marks[:4])))
    if f.cpu_kernel:
        rows.append(("History",
                     f'<a href="{R.history_url(f.cpu_kernel[0])}">'
                     f"commits touching the CPU kernel</a>"))

    if f.tests:
        files = sorted({t[0] for t in f.tests})
        rows.append((f"Tests (≥{len(f.tests)})",
                     " ".join(f'<a href="{R.REPO_URL}/tests/python/{fn}"><code>{fn}</code></a>'
                              for fn in files)))
    else:
        rows.append(("Tests", '<span class="none">none found by name</span>'))

    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<h4>Implementation</h4><table class="impl"><tbody>{body}</tbody></table>'


def header_doc(f) -> str:
    """The `///` block from the C++ header, verbatim.

    This prose was written next to the code by whoever made the decision it
    explains. It IS the developer documentation -- it simply was not reachable
    from a browser before.
    """
    d = next((d for d in f.decls if d.doc), None)
    if not d:
        return ""
    return (f"<h4>From the header</h4>"
            f'<div class="decl">{highlight(d.signature)}'
            f'<span class="where">— <a href="{d.url}">{d.path}:{d.line}</a></span></div>'
            + paragraphs(d.doc))


# ------------------------------------------------------------------- pages --


def api_entry(name: str) -> tuple[str, bool, str]:
    """Render one entry -> (html, documented, one-line summary for search)."""
    p = PROSE.get(name, {})
    documented = bool(p.get("summary"))

    out = [
        f'<h3 id="{name}"><code>{name}</code>'
        f'<a class="anchor" href="#{name}" aria-label="Permalink">¶</a></h3>',
        render_signature(name, signature_of(name)),
    ]

    # From the backend's own supports(), not from a list kept here by hand.
    facts_for_chip = FACTS.get(name)
    on_gpu = facts_for_chip.on_vulkan if facts_for_chip else (name not in CPU_ONLY)
    out.append(
        '<div class="support"><span class="chip yes">CPU</span>'
        f'<span class="chip {"yes" if on_gpu else "no"}">Vulkan</span>'
        + ("" if on_gpu else '<span class="chip">CPU-only by decision</span>')
        + "</div>"
    )

    summary = p.get("summary") or docstring_prose(name)
    if summary:
        out.append(paragraphs(summary))
    if not documented:
        out.append(
            '<div class="todo">No written description yet. The signature above is '
            "generated from the installed module and is current; the prose is not "
            "written. See <code>web/content/</code>.</div>"
        )

    if p.get("detail"):
        out.append(paragraphs(p["detail"]))

    if p.get("params"):
        out.append('<h4>Parameters</h4><dl class="params">')
        for pname, ptype, pdesc in p["params"]:
            opt = ' <span class="opt">optional</span>' if "=" in ptype else ""
            out.append(
                f"<dt>{html.escape(pname)} "
                f'<span class="type">({html.escape(ptype)})</span>{opt}</dt>'
                f"<dd>{inline(pdesc)}</dd>"
            )
        out.append("</dl>")

    if p.get("returns"):
        out.append(f"<h4>Returns</h4><p>{inline(p['returns'])}</p>")
    for kind in ("note", "warning", "tip"):
        if p.get(kind):
            out.append(admon(kind, p[kind]))
    if p.get("example"):
        out.append("<h4>Example</h4>" + code_block(p["example"], repl=True))
    # Everything below is EXTRACTED from the repository rather than written, so
    # a page cannot claim a shader or a test file that has moved.
    facts = FACTS.get(name)
    if facts:
        hdr = header_doc(facts)
        if hdr:
            out.append(hdr)
        # The GPU kernel's own reasoning. For the element-wise family this is
        # the most valuable prose in the repository -- why tanh clamps, why gelu
        # goes through erfc, why relu is spelled the way it is -- written next
        # to the code by whoever hit the bug it prevents.
        if facts.glsl:
            g = facts.glsl
            out.append(f"<h4>In the Vulkan kernel</h4>"
                       f'<div class="decl">{highlight(g["returns"] + " " + name)}'
                       f'_op({highlight(g["args"])})'
                       f'<span class="where">— <a href="{g["url"]}">'
                       f'{g["path"]}:{g["line"]}</a></span></div>'
                       + paragraphs(g["doc"]))
        if facts.cpu_doc:
            c = facts.cpu_doc
            out.append(f"<h4>In the CPU kernel</h4>" + paragraphs(c["doc"])
                       + f'<p class="lede"><a href="{c["url"]}">'
                       f'{c["path"]}:{c["line"]}</a></p>')
        out.append(impl_table(facts))

    if p.get("see"):
        links = ", ".join(f'<a href="#{s}"><code>{s}</code></a>' for s in p["see"])
        out.append(f"<p><strong>See also</strong> {links}</p>")

    first_line = (summary or "").strip().split("\n")[0][:110]
    return "\n".join(out), documented, first_line


def class_page(name: str, spec: dict) -> tuple[str, list[tuple[str, str, int]]]:
    """Render one class. Members come from the extractor; meaning from CLASSES.

    A member with no written entry still appears, with its signature and a link
    to its line. The mapping adds explanation; it does not decide what is
    visible, because a reference that shows only the documented half of a class
    misrepresents the class.
    """
    lang = spec.get("lang", "cpp")
    # `symbol` lets a page be keyed differently from the class it documents,
    # which is what allows the C++ `Tensor` and the Python `Tensor` -- one type,
    # two surfaces, and one place where they disagree -- to be separate pages.
    symbol = spec.get("symbol", name)
    if lang == "native":
        doc = R.nanobind_class(symbol, tuple(spec.get("extra_members", ())))
    else:
        doc = (R.all_classes() if lang == "cpp"
               else R.python_classes()).get(symbol)
    if doc is None:
        return f"<h1>{html.escape(name)}</h1>" + admon(
            "warning", f"`{name}` was not found by the extractor."), []

    body = [f"<h1><code>{html.escape(name)}</code></h1>",
            f'<p class="lede">{inline(spec["summary"])}</p>']

    decl = f"{doc.kind} {doc.name}" + (f" : {doc.bases}" if doc.bases else "")
    body.append(f'<div class="decl">{highlight(decl)}'
                f'<span class="where">— <a href="{doc.url}">{doc.path}:{doc.line}</a>'
                f"</span></div>")

    if spec.get("detail"):
        body.append(paragraphs(spec["detail"]))
    for kind in ("note", "warning", "tip"):
        if spec.get(kind):
            body.append(admon(kind, spec[kind]))

    by_name: dict[str, list] = {}
    for m in doc.public():
        by_name.setdefault(m.name, []).append(m)

    grouped = spec.get("groups") or [("Members", sorted(by_name))]
    listed = {n for _, names in grouped for n in names}
    ungrouped = sorted(set(by_name) - listed)
    if ungrouped:
        grouped = list(grouped) + [("Other members", ungrouped)]

    toc = []
    for gname, names in grouped:
        gid = "m-" + re.sub(r"[^a-z0-9]+", "-", gname.lower()).strip("-")
        body.append(f'<h2 id="{gid}">{html.escape(gname)}'
                    f'<a class="anchor" href="#{gid}">¶</a></h2>')
        toc.append((gid, gname, 2))
        for mname in names:
            overloads = by_name.get(mname)
            if not overloads:
                continue
            mid = f"m-{mname}"
            body.append(f'<h3 id="{mid}"><code>{html.escape(mname)}</code>'
                        f'<a class="anchor" href="#{mid}">¶</a></h3>')
            toc.append((mid, mname, 3))
            for m in overloads:
                body.append(f'<div class="decl">{highlight(m.signature)}'
                            f'<span class="where">— <a href="'
                            f'{R.src_link(doc.file, m.line)}">{doc.path}:{m.line}</a>'
                            f"</span></div>")
            written = spec.get("members", {}).get(mname)
            if written:
                body.append(paragraphs(written))
            else:
                own = next((m.doc for m in overloads if m.doc), "")
                if own:
                    body.append(paragraphs(own))
    if spec.get("see"):
        links = ", ".join(f'<a href="{PAGE_OF[t]}.html#{t}"><code>{t}</code></a>'
                          for t in spec["see"] if t in PAGE_OF)
        if links:
            body.append(f"<p><strong>See also</strong> {links}</p>")
    return "\n".join(body), toc


def class_slug(name: str) -> str:
    return "class-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------- navigation --

# THE READER'S STRUCTURE, declared here rather than fallen out of the build's.
#
# `sidenav()` used to walk PAGES, CLASSES and GROUPS -- the containers the site
# is generated FROM -- so the order was build order and the grouping was
# whatever list a page happened to live in. The visible result: fourteen items
# under "Guide", of which seven were architecture pages and one was a reference
# page; twenty-seven classes in one flat column mixing layers, optimisers, data
# and serialisation; and 42-52 visible links running 2.6 screens at 1440x900.
#
# Four sections because a reader arrives with one of four questions: how do I
# start, how does it work, what is it called, and can I trust it. Architecture
# is promoted out of "Guide" because those seven pages are the ones a systems
# programmer comes for.
NAV_SECTIONS: list[tuple[str, list[str]]] = [
    ("Learn", ["get-started", "concepts", "compatibility"]),
    ("Capabilities", ["capabilities", "limitations"]),
    ("Architecture", ["arch-overview", "arch-tensor", "arch-graph", "arch-autograd", "arch-cpu",
                      "arch-vulkan", "arch-shaders", "arch-numerics"]),
    ("Project", ["performance", "testing", "contributing", "reference-env"]),
]

# Classes by KIND. A reader looking for "an optimiser" should see four names,
# not scan twenty-seven. The two Tensors are labelled by surface: they document
# genuinely different things -- Python `.size` is the element count, C++
# `size(axis)` is one extent -- and the navigation is where that choice is made.
CLASS_SECTIONS: list[tuple[str, list[str]]] = [
    ("Core", ["vkml.Tensor", "Tensor", "Module"]),
    ("Layers", ["Linear", "Conv2d", "Embedding", "Sequential", "Flatten",
                "BatchNorm2d", "LayerNorm", "Dropout",
                "MultiheadAttention", "TransformerEncoderLayer",
                "ReLU", "GELU", "Sigmoid", "Tanh", "MaxPool2d", "AvgPool2d"]),
    ("Optimisers", ["Optimizer", "SGD", "Adam", "AdamW", "RMSProp"]),
    ("Data and checkpoints", ["DataLoader", "ArrayDataset", "Checkpoint"]),
]

CLASS_LABEL = {"vkml.Tensor": "Tensor <span class=\"nav-note\">Python</span>",
               "Tensor": "Tensor <span class=\"nav-note\">C++</span>"}


def reading_order() -> list[str]:
    """The sequence a reader is walked through, DERIVED from NAV_SECTIONS.

    Not a second declaration. The sidebar already states the order a reader is
    meant to meet these pages in, so prev/next reuses it rather than introducing
    a parallel list that can disagree with the navigation -- which is how a site
    ends up telling you one thing in the tree and another at the foot of a page.

    Reference pages are deliberately absent. `api-reduction` has no natural
    successor: nobody reads the operator reference front to back, and offering
    "Next: Shape and indexing" would invent a journey nobody is on.
    """
    return [slug for _, slugs in NAV_SECTIONS for slug in slugs]


def sidenav(active: str) -> str:
    """The section tree, built from the reader's structure rather than the build's."""
    title_of = {slug: title for slug, title, _ in PAGES}
    s = ['<aside class="sidenav" id="sidenav"><h2>Documentation</h2>']

    for heading, slugs in NAV_SECTIONS:
        s.append(f"<h3>{html.escape(heading)}</h3><ol>")
        for slug in slugs:
            if slug not in title_of:
                continue
            cls = ' class="active"' if slug == active else ""
            s.append(f'<li><a{cls} href="{slug}.html">'
                     f"{html.escape(title_of[slug])}</a></li>")
        s.append("</ol>")

    # The API reference lists CATEGORIES, not the 103 operators. Every operator
    # is one click further on its category page, and search now indexes all 278
    # names, so the sidebar does not have to carry the whole surface -- carrying
    # it was what made this three screens long.
    s.append("<h3>API reference</h3><ol>")
    ov = ' class="active"' if active == "api" else ""
    s.append(f'<li><a{ov} href="api.html">Overview</a></li>')
    for gname, names in GROUPS:
        slug = group_slug(gname)
        cls = ' class="active"' if slug == active else ""
        s.append(f'<li><a{cls} href="{slug}.html">{html.escape(gname)}'
                 f'<span class="nav-count">{len(names)}</span></a></li>')
    s.append("</ol>")

    if CLASSES:
        s.append("<h3>Classes</h3>")
        listed = {n for _, names in CLASS_SECTIONS for n in names}
        for heading, names in CLASS_SECTIONS + [
                ("Other", sorted(set(CLASSES) - listed))]:
            names = [n for n in names if n in CLASSES]
            if not names:
                continue
            # Collapsed unless the reader is inside it. Grouping the classes
            # was right; showing all twenty-seven at once was not -- Layers
            # alone is sixteen entries and 528px, and the four sections came to
            # a third of the whole sidebar. Same rule the operator categories
            # already used: open the one you are in, fold the rest.
            here = any(class_slug(n) == active for n in names)
            s.append(f'<div class="grp{"" if here else " closed"}">'
                     f'<button type="button" aria-expanded="{str(here).lower()}">'
                     f'<span class="chev">▾</span>{html.escape(heading)}'
                     f'<span class="nav-count">{len(names)}</span></button><ol>')
            for cname in names:
                cslug = class_slug(cname)
                ccls = ' class="active"' if cslug == active else ""
                label = CLASS_LABEL.get(cname, f"<code>{html.escape(cname)}</code>")
                s.append(f'<li><a{ccls} href="{cslug}.html">{label}</a></li>')
            s.append("</ol></div>")

    s.append("</aside>")
    return "\n".join(s)


def toc_for(entries: list[tuple[str, str, int]]) -> str:
    """Right-hand column. `entries` is (id, label, level)."""
    if not entries:
        return '<aside class="toc"></aside>'
    items = "".join(
        f'<li><a class="{"lvl3" if lvl == 3 else ""}" href="#{i}">{html.escape(t)}</a></li>'
        for i, t, lvl in entries
    )
    return (
        '<aside class="toc"><h4>☰ On this page</h4><ul>' + items + "</ul>"
        '<div class="side-links">'
        '<a href="https://github.com/jeel00dev/vkml">Source ↗</a>'
        '<a href="https://github.com/jeel00dev/vkml/issues">Issues ↗</a>'
        '<a href="https://github.com/jeel00dev/vkml/tree/main/docs">Design docs ↗</a>'
        "</div></aside>"
    )


def pagenav(prev: tuple[str, str] | None, nxt: tuple[str, str] | None) -> str:
    """Previous / next, so the docs read straight through as well as by search."""
    if not prev and not nxt:
        return ""
    bits = ['<nav class="pagenav">']
    if prev:
        bits.append(f'<a class="prev" href="{prev[1]}"><span class="dir">‹ Previous</span>'
                    f'<span class="ttl">{html.escape(prev[0])}</span></a>')
    if nxt:
        bits.append(f'<a class="next" href="{nxt[1]}"><span class="dir">Next ›</span>'
                    f'<span class="ttl">{html.escape(nxt[0])}</span></a>')
    return "".join(bits) + "</nav>"


def crumbs(*parts: tuple[str, str | None]) -> str:
    bits = ['<nav class="crumbs"><a href="index.html">⌂</a>']
    for label, href in parts:
        bits.append('<span class="sep">›</span>')
        bits.append(
            f'<a href="{href}">{html.escape(label)}</a>'
            if href
            else f'<span class="here">{html.escape(label)}</span>'
        )
    return "".join(bits) + "</nav>"


SHELL = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="icon" href="assets/favicon-32.png" sizes="32x32">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<meta property="og:type" content="website">
<meta property="og:site_name" content="vkML">
<meta property="og:title" content="{ogtitle}">
<meta property="og:description" content="{desc}">
<meta property="og:image" content="{site}/assets/og-card.png">
<meta property="og:url" content="{site}/{page}.html">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{ogtitle}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{site}/assets/og-card.png">
<link rel="stylesheet" href="theme/vkml.css">
</head>
<body class="{bodycls}">
<header class="topbar">
  <button class="icon-btn" id="nav-toggle" aria-label="Toggle navigation">☰</button>
  <a class="brand" href="index.html">
    <img src="assets/logo-64.png" alt="" width="38" height="26"><span>vk<span class="ml">ML</span></span>
  </a>
  <span class="ver">{version}</span>
  <nav>
    <a href="get-started.html">Get started</a>
    <a href="concepts.html">Concepts</a>
    <a class="{apicls}" href="api.html">API reference</a>
  </nav>
  <span class="spacer"></span>
  <button class="icon-btn" id="search-open" aria-label="Search"
          aria-expanded="false" aria-controls="search">⌕</button>
  <div class="search" id="search" role="search">
    <span class="mag" aria-hidden="true">⌕</span>
    <input id="q" type="search" placeholder="Search the docs …" autocomplete="off"
           spellcheck="false" aria-label="Search">
    <kbd aria-hidden="true">/</kbd>
    <button class="icon-btn search-close" id="search-close"
            aria-label="Close search">✕</button>
    <div class="results" id="results" role="listbox" aria-label="Search results"></div>
  </div>
  <button class="icon-btn" id="theme-toggle" aria-label="Toggle theme" title="Light / dark">◐</button>
  <a class="icon-btn" href="https://github.com/jeel00dev/vkml" title="GitHub" aria-label="GitHub">⌂</a>
</header>
<div class="layout">
{sidenav}
<main class="content">
{crumbs}
{body}
<footer class="page">
  vkML — Vulkan-first machine learning in C++20. Apache-2.0.
  Signatures on this page are generated from the installed module.
</footer>
</main>
{toc}
</div>
<button id="totop" type="button">↑ Back to top</button>
<script>window.VKML_INDEX = {index};</script>
<script src="theme/site.js"></script>
</body>
</html>
"""


def write(
    path: Path,
    *,
    title: str,
    desc: str,
    body: str,
    nav: str,
    toc: str,
    crumb: str,
    page: str,
    index_json: str,
) -> None:
    path.write_text(
        SHELL.format(
            title=html.escape(title),
            # Without the site suffix. The <title> ends "-- vkML" and og:site_name
            # already carries the project, so reusing it here rendered link
            # previews as "vkML -- machine learning on Vulkan -- vkML".
            ogtitle=html.escape(re.sub(r"\s*—\s*vkML$", "", title)),
            desc=html.escape(plain(desc)),
            body=body,
            sidenav=nav,
            toc=toc,
            crumbs=crumb,
            version=html.escape(V.__version__),
            apicls="active" if page == "api" else "",
            # The landing page is not a documentation page: it drops the
            # section tree and the measure clamp, and centres on the viewport.
            bodycls="landing" if page == "index" else "",
            page=page,
            site=SITE_URL,
            index=index_json,
        )
    )


SITE_JS = """/* Behaviour for the vkML docs. No framework: the whole site is four pages and
   the index is already in the document. */
(function () {
  var root = document.documentElement;
  var saved = localStorage.getItem('vkml-theme');
  if (saved) root.setAttribute('data-theme', saved);
  var tt = document.getElementById('theme-toggle');
  if (tt) tt.onclick = function () {
    var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    localStorage.setItem('vkml-theme', next);
  };

  var nt = document.getElementById('nav-toggle'), nav = document.getElementById('sidenav');
  if (nt && nav) nt.onclick = function () { nav.classList.toggle('open'); };

  // Collapsible groups.
  document.querySelectorAll('.grp > button').forEach(function (b) {
    b.onclick = function () { b.parentElement.classList.toggle('closed'); };
  });

  // Copy buttons.
  document.querySelectorAll('pre .copy').forEach(function (b) {
    b.onclick = function () {
      var code = b.parentElement.querySelector('code');
      // Strip REPL prompts and output so what lands on the clipboard is runnable.
      var text = b.parentElement.classList.contains('repl')
        ? Array.from(code.querySelectorAll('.p')).map(function (p) {
            var line = '', n = p.nextSibling;
            while (n && !(n.nodeType === 1 && n.classList && n.classList.contains('p'))) {
              if (n.nodeType === 3 && n.textContent.indexOf('\\n') !== -1) {
                line += n.textContent.split('\\n')[0]; break;
              }
              line += n.textContent; n = n.nextSibling;
            }
            return line;
          }).join('\\n')
        : code.innerText;
      navigator.clipboard.writeText(text).then(function () {
        b.textContent = 'copied'; setTimeout(function () { b.textContent = 'copy'; }, 1200);
      });
    };
  });

  // Back to top.
  var top = document.getElementById('totop');
  if (top) {
    top.onclick = function () { window.scrollTo({ top: 0 }); };
    addEventListener('scroll', function () {
      top.classList.toggle('show', scrollY > 600);
    }, { passive: true });
  }

  // Scroll-spy across both columns.
  var links = {};
  document.querySelectorAll('.toc a[href^="#"], .sidenav a[href^="#"]').forEach(function (a) {
    var id = a.getAttribute('href').slice(1);
    (links[id] = links[id] || []).push(a);
  });
  var heads = document.querySelectorAll('h2[id], h3[id]');
  if (heads.length && Object.keys(links).length) {
    var seen = new Set();
    new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) seen.add(e.target.id); else seen.delete(e.target.id);
      });
      var first = null;
      heads.forEach(function (h) { if (!first && seen.has(h.id)) first = h.id; });
      if (!first) return;
      document.querySelectorAll('.toc a.active, .sidenav a.active[href^="#"]')
        .forEach(function (a) { a.classList.remove('active'); });
      (links[first] || []).forEach(function (a) {
        a.classList.add('active');
        if (a.closest('.toc')) a.scrollIntoView({ block: 'nearest' });
      });
    }, { rootMargin: '-15% 0px -70% 0px' }).observe
      && heads.forEach(function (h) {
        new IntersectionObserver(function (es) {
          es.forEach(function (e) {
            if (e.isIntersecting) seen.add(e.target.id); else seen.delete(e.target.id);
            var first = null;
            heads.forEach(function (x) { if (!first && seen.has(x.id)) first = x.id; });
            if (!first) return;
            document.querySelectorAll('.toc a.active, .sidenav a.active[href^="#"]')
              .forEach(function (a) { a.classList.remove('active'); });
            (links[first] || []).forEach(function (a) { a.classList.add('active'); });
          });
        }, { rootMargin: '-15% 0px -70% 0px' }).observe(h);
      });
  }

  // Search. Substring over name and summary, name matches first.
  var q = document.getElementById('q'), box = document.getElementById('results');
  var idx = window.VKML_INDEX || [], sel = -1, shown = [];
  function render(list) {
    shown = list; sel = -1;
    if (!list.length) { box.innerHTML = '<div class="empty">No matches.</div>'; }
    else {
      box.innerHTML = list.slice(0, 40).map(function (e) {
        return '<a href="' + e.u + '"><span class="r-grp">' + e.g + '</span>' +
               '<span class="r-name">' + e.n + '</span>' +
               '<span class="r-desc">' + (e.d || '') + '</span></a>';
      }).join('');
    }
    box.classList.add('open');
  }
  if (q) {
    q.addEventListener('input', function () {
      var v = q.value.trim().toLowerCase();
      if (!v) { box.classList.remove('open'); return; }
      /* Ranked, not just filtered. The index carries qualified member names
         like `Tensor.numpy`, so a plain substring test put `from_numpy` above
         `Tensor.numpy` for the query "numpy" -- both merely contain it. What a
         reader means is almost always the thing whose OWN name begins with what
         they typed, and for a member that is the part after the dot.

         Ties break on the shorter name, which prefers the more specific entry:
         `Adam` over `AdamW` for "adam". */
      function score(e) {
        var n = e.n.toLowerCase();
        var tail = n.slice(n.lastIndexOf('.') + 1);
        if (n === v || tail === v) return 0;
        if (n.indexOf(v) === 0) return 1;
        if (tail.indexOf(v) === 0) return 2;
        if (n.indexOf(v) !== -1) return 3;
        if ((e.d || '').toLowerCase().indexOf(v) !== -1) return 4;
        return 9;
      }
      var hits = [];
      idx.forEach(function (e) {
        var s = score(e);
        if (s < 9) hits.push([s, e.n.length, e]);
      });
      hits.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
      render(hits.map(function (h) { return h[2]; }));
    });
    q.addEventListener('keydown', function (e) {
      var items = box.querySelectorAll('a');
      if (e.key === 'Escape') { closeSearch(); }
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!items.length) return;
        if (sel >= 0) items[sel].classList.remove('sel');
        sel = (sel + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
        items[sel].classList.add('sel'); items[sel].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter' && sel >= 0) { items[sel].click(); }
    });
    addEventListener('keydown', function (e) {
      if (e.key === '/' && document.activeElement !== q) {
        e.preventDefault(); openSearch();
      }
    });
    addEventListener('click', function (e) {
      if (!e.target.closest('.search') && !e.target.closest('#search-open')) {
        box.classList.remove('open');
      }
    });

    /* On a phone the field is a sheet rather than a header input, so opening it
       is an explicit act and focus has to be put in and given back. Without the
       return, closing the sheet drops focus to the top of the document and a
       keyboard or switch user loses their place in the header. */
    var opener = document.getElementById('search-open');
    var closer = document.getElementById('search-close');
    var sheet = document.getElementById('search');
    function openSearch() {
      sheet.classList.add('open');
      if (opener) opener.setAttribute('aria-expanded', 'true');
      q.focus();
    }
    function closeSearch() {
      /* Read this BEFORE hiding the sheet. Removing the class sets
         display:none, which blurs whatever was focused inside it, so asking
         afterwards always answers "not in the sheet" and focus is silently
         dropped to the document -- measured: activeElement came back empty. */
      var wasInside = sheet.contains(document.activeElement);
      sheet.classList.remove('open');
      box.classList.remove('open');
      if (opener) {
        opener.setAttribute('aria-expanded', 'false');
        /* Only take focus back if it was in the sheet: if the reader followed a
           result we are on a new page and must not steal it. */
        if (wasInside) opener.focus();
      }
    }
    if (opener) opener.onclick = openSearch;
    if (closer) closer.onclick = closeSearch;

  }
})();
"""


# ------------------------------------------------------------ capabilities --

def capability_pages() -> list[tuple[str, str, str]]:
    """Three pages built from what the build extracts, joined to declared reasons.

    Nothing here is a list a person maintains. The operator inventory, backend
    coverage, gradient rules, dtypes and the torch equivalences all come from
    the tree; `content/capabilities.py` supplies only the WHY, which no
    extractor can know. `check_docs_references.py` fails if the two disagree in
    either direction.
    """
    from content.capabilities import (BACKEND_REASONS, FEATURE_NOTES,
                                      GRADIENT_REASONS, REASON_LABELS)

    kinds = R.op_kinds()
    ruled = set(R.autograd_rules())
    no_rule = [k for k in kinds if k not in ruled]
    torch_eq = R.torch_equivalents()
    on_gpu = [n for n, f in FACTS.items() if f.on_vulkan]
    cpu_only = sorted(n for n, f in FACTS.items() if not f.on_vulkan)
    shaders = R.shader_index()

    def badge(kind: str) -> str:
        label, _ = REASON_LABELS[kind]
        return f'<span class="rz rz-{kind}">{html.escape(label)}</span>'

    # ---------------------------------------------------------- capabilities --
    rows = "".join(
        f"<tr><td><code>{html.escape(g)}</code></td><td>{len(names)}</td>"
        f'<td>{sum(1 for n in names if FACTS.get(n) and FACTS[n].on_vulkan)}</td></tr>'
        for g, names in GROUPS)
    cap = f"""
<p class="lede">What vkML does today, counted from the tree at build time. Every
number on this page is generated; none of it is typed.</p>

<div class="lstats" style="max-width:none">
  <div class="lstat"><b>{len(FACTS)}</b><span>operators</span></div>
  <div class="lstat"><b>{len(on_gpu)}</b><span>run on Vulkan</span></div>
  <div class="lstat"><b>{len(ruled)}</b><span>gradient rules</span></div>
  <div class="lstat"><b>{len(shaders)}</b><span>compute shaders</span></div>
  <div class="lstat"><b>{len(CLASSES)}</b><span>documented classes</span></div>
</div>

<h2 id="operators">Operators by category</h2>
<div class="table-scroll"><table>
<thead><tr><th>Category</th><th>Operators</th><th>On Vulkan</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2 id="dtypes">Data types</h2>
<p>Five, on both backends: <code>float32</code>, <code>float16</code>,
<code>int32</code>, <code>int64</code> and <code>bool</code>. f32&rarr;f16
narrowing is done in software rather than left to <code>OpFConvert</code>, whose
rounding mode SPIR-V leaves implementation-defined &mdash; so a narrowed value is
the same bits on every driver.</p>

<h2 id="devices">Devices</h2>
<p>Any device exposing Vulkan 1.3 compute, plus the CPU backend. The CPU one is
the correctness oracle the GPU is checked against, not a fallback: it is always
built, always available, and every Vulkan result is compared against it.</p>

<h2 id="autograd">Autograd</h2>
<p>{len(ruled)} of {len(kinds)} graph operations carry a gradient rule. The
remaining {len(no_rule)} are listed on
<a href="limitations.html#gradients">Current limitations</a> with the reason for
each &mdash; most of them are leaves or comparisons, where a gradient is not a
thing that exists.</p>

<h2 id="serialization">Serialization</h2>
<p>Checkpoints are a zip of NumPy arrays plus a JSON manifest, loaded with
<code>allow_pickle=False</code>, written to a temporary file and renamed so an
interrupted save cannot leave a half-written checkpoint in place. A PyTorch
<code>state_dict</code> loads without translation.</p>
"""

    # ---------------------------------------------------------- compatibility --
    eq_rows = "".join(
        f'<tr><td><code>{html.escape(", ".join(t))}</code></td>'
        f'<td><a href="{PAGE_OF[v]}.html#{v}"><code>{html.escape(v)}</code></a></td></tr>'
        if v in PAGE_OF else
        f'<tr><td><code>{html.escape(", ".join(t))}</code></td>'
        f"<td><code>{html.escape(v)}</code></td></tr>"
        for v, t in sorted(torch_eq.items()))
    compat = f"""
<p class="lede">vkML follows PyTorch's names and shapes deliberately, so code
ports without translation. The table below is not a claim &mdash; it is extracted
from the test suite, where every pair is an assertion that the two agree within a
declared tolerance.</p>

<h2 id="verified">Verified equivalences</h2>
<p>{len(torch_eq)} operators are tested directly against their torch
counterpart. If a row is here, a test compares them on every run.</p>
<div class="table-scroll"><table>
<thead><tr><th>PyTorch</th><th>vkML</th></tr></thead>
<tbody>{eq_rows}</tbody></table></div>

<h2 id="conventions">Conventions that carry over</h2>
<ul>
<li><strong>Layout is row-major</strong>, as NumPy, PyTorch and DLPack.
<code>shape()[0]</code> is the outermost axis.</li>
<li><strong>A <code>state_dict</code> loads directly.</strong> <code>Linear</code>
stores its weight as <code>(out_features, in_features)</code> and transposes in
<code>forward</code>, exactly as PyTorch does, so the tensors line up.</li>
<li><strong>Reductions take <code>dim</code> and <code>keepdim</code></strong>
with torch's meanings.</li>
</ul>

<h2 id="divergences">Where it deliberately differs</h2>
<ul>
<li><strong><code>.size</code> is the element count</strong>, following NumPy
rather than PyTorch, and it is a property: <code>x.size()</code> raises. Use
<code>x.shape</code>.</li>
<li><strong>Shape arguments are sequences</strong>: <code>x.reshape([3, 2])</code>,
not <code>x.reshape(3, 2)</code>.</li>
<li><strong><code>max</code> and <code>min</code> return values, not
<code>(values, indices)</code></strong>, and are spelled <code>amax</code> and
<code>amin</code> to say so.</li>
<li><strong><code>to()</code> converts dtype, not device.</strong> Placement
happens at creation: <code>V.tensor(x, device=dev)</code>.</li>
<li><strong>Seeds do not match.</strong> <code>manual_seed</code> mirrors torch in
spirit, not in stream &mdash; the two draw from different generators, so equal
seeds give different weights.</li>
</ul>
"""

    # ------------------------------------------------------------ limitations --
    grad_rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{badge(GRADIENT_REASONS[k][0])}</td>"
        f"<td>{inline(GRADIENT_REASONS[k][1])}</td></tr>"
        for k in no_rule if k in GRADIENT_REASONS)
    backend_rows = "".join(
        f"<tr><td><code>{html.escape(n)}</code></td><td>{badge(BACKEND_REASONS[n][0])}</td>"
        f"<td>{inline(BACKEND_REASONS[n][1])}</td></tr>"
        for n in cpu_only if n in BACKEND_REASONS)
    notes = "".join(
        f'<div class="limit"><h3>{inline(f["title"])} {badge(f["reason"])}</h3>'
        f'{paragraphs(f["text"])}'
        + (f'<p class="src"><a href="{R.REPO_URL}/{f["see"][0]}">{html.escape(f["see"][0])}</a></p>'
           if f.get("see") else "")
        + "</div>"
        for f in FEATURE_NOTES)
    legend = "".join(
        f'<dt>{badge(k)}</dt><dd>{html.escape(v[1])}</dd>'
        for k, v in REASON_LABELS.items())

    n_real = sum(1 for k in no_rule
                 if GRADIENT_REASONS.get(k, ("", ""))[0] != "by-design")
    limits = f"""
<p class="lede">What vkML does not do, and why. Each entry says whether the
limit is a decision, a consequence of a guarantee the project keeps, or work
that has not happened yet &mdash; those are different things, and a reader
deciding whether to adopt this needs to tell them apart.</p>

<dl class="legend">{legend}</dl>

<h2 id="gradients">Operations without a gradient rule</h2>
<p>{len(ruled)} of {len(kinds)} graph operations carry one. Of the
{len(no_rule)} that do not, {len(no_rule) - n_real} are cases where a gradient is
not a thing that exists &mdash; a leaf has no input, a comparison returns
<code>Bool</code> &mdash; and <strong>{n_real}</strong> are genuine gaps.</p>
<div class="table-scroll"><table>
<thead><tr><th>Operation</th><th>Why</th><th></th></tr></thead>
<tbody>{grad_rows}</tbody></table></div>

<h2 id="backends">Operators that do not run on Vulkan</h2>
<p>{len(on_gpu)} of {len(FACTS)} run on the GPU.</p>
<div class="table-scroll"><table>
<thead><tr><th>Operator</th><th>Why</th><th></th></tr></thead>
<tbody>{backend_rows}</tbody></table></div>

<h2 id="features">Features</h2>
{notes}
"""
    return [
        ("capabilities", "What vkML supports", cap),
        ("compatibility", "Coming from PyTorch", compat),
        ("limitations", "Current limitations", limits),
    ]


def build() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(WEB / "theme", OUT / "theme")
    (OUT / "theme" / "site.js").write_text(SITE_JS)
    (OUT / "assets").mkdir()
    for f in sorted((ROOT / "assets" / "derived").glob("*.png")):
        shutil.copy(f, OUT / "assets" / f.name)
    (OUT / ".nojekyll").write_text("")

    public = {
        n
        for n in dir(V)
        if not n.startswith("_")
        and callable(getattr(V, n))
        and not inspect.isclass(getattr(V, n))
    }
    unknown = sorted(ALL_NAMES - public)
    if unknown:
        print(f"  ERROR: listed but absent from the module: {', '.join(unknown)}")
        return 1
    missing = sorted(public - ALL_NAMES)

    documented, index, per_group = 0, [], []
    for gname, names in GROUPS:
        frags, toc, ndoc = [], [], 0
        for n in names:
            frag, ok, summary = api_entry(n)
            frags.append(frag)
            toc.append((n, n, 3))
            ndoc += ok
            index.append(
                {"n": n, "u": f"{group_slug(gname)}.html#{n}", "g": gname, "d": plain(summary)}
            )
        documented += ndoc
        per_group.append((gname, names, frags, toc, ndoc))

    for slug, title, _ in PAGES:
        index.append({"n": title, "u": f"{slug}.html", "g": "Guide", "d": ""})

    # CLASSES, and everything else the reader might type. This block used to sit
    # further down, next to the loop that writes the class pages -- which is
    # AFTER index_json was serialised, so 27 entries were appended to a list
    # nobody read again. The effect was that searching "Adam" returned nothing,
    # "DataLoader" returned nothing, and "Linear" returned two accidental
    # substring hits in other operators' descriptions with `relu` at the top.
    # Those are the first things a PyTorch user types.
    for cname, spec in CLASSES.items():
        index.append({"n": cname, "u": f"{class_slug(cname)}.html", "g": "Classes",
                      "d": plain(spec.get("summary", ""))})
        # Members too: a reader looking for `.numpy()` or `zero_grad` is looking
        # for a method, and knowing which class it hangs off is the thing they
        # do not know yet.
        lang = spec.get("lang", "cpp")
        symbol = spec.get("symbol", cname)
        doc = (R.nanobind_class(symbol, tuple(spec.get("extra_members", ())))
               if lang == "native" else
               (R.all_classes() if lang == "cpp" else R.python_classes()).get(symbol))
        if doc is None:
            continue
        # One entry per NAME, not per overload. `doc.public()` yields a Member
        # for each C++ overload, so `Tensor::reshape` -- span and
        # initializer_list -- produced three identical "Tensor.reshape" rows in
        # the results, which reads as a broken index.
        seen_members: set[str] = set()
        for m in doc.public():
            if m.name.startswith("_") or m.name in seen_members:
                continue
            seen_members.add(m.name)
            index.append({"n": f"{cname}.{m.name}",
                          "u": f"{class_slug(cname)}.html#m-{m.name}",
                          "g": "Members", "d": plain(spec.get("members", {}).get(m.name, ""))})

    # Environment switches, read from the C++ that consumes them. Searching
    # VKML_GEMM_KERNEL returned nothing despite a whole reference page for it.
    for name, meta in sorted(R.env_switches().items()):
        index.append({"n": name, "u": f"reference-env.html#{name.lower().replace('_', '-')}",
                      "g": "Environment", "d": plain(meta.get("doc", "")) if isinstance(meta, dict) else ""})

    index_json = json.dumps(index, separators=(",", ":"))

    # Overview: what is in the reference and how complete each part is.
    cards = "".join(
        f'<a class="card" href="{group_slug(g)}.html"><h3>{html.escape(g)}</h3>'
        f"<p>{len(names)} function{'s' if len(names) != 1 else ''} · "
        f"{nd}/{len(names)} documented</p></a>"
        for g, names, _, _, nd in per_group
    )
    overview = (
        "<h1>API reference</h1>"
        '<p class="lede">Every public function, with the signature read from the '
        "installed module at build time.</p>"
        + (
            admon(
                "note",
                f"{len(missing)} public function(s) are not yet grouped and "
                f"do not appear: `{', '.join(missing)}`.",
            )
            if missing
            else ""
        )
        + f'<div class="cards">{cards}</div>'
    )
    write(
        OUT / "api.html",
        title="API reference — vkML",
        desc="Every public vkML function.",
        body=overview,
        nav=sidenav("api"),
        toc=toc_for([(group_slug(g), g, 2) for g, _, _, _, _ in per_group]),
        crumb=crumbs(("API reference", None)),
        page="api",
        index_json=index_json,
    )

    for gname, names, frags, toc, nd in per_group:
        slug = group_slug(gname)
        body = (
            f"<h1>{html.escape(gname)}</h1>"
            f'<p class="lede">{len(names)} function'
            f"{'s' if len(names) != 1 else ''} — {nd} documented.</p>"
            + "\n".join(frags)
        )
        write(
            OUT / f"{slug}.html",
            title=f"{gname} — vkML",
            desc=f"vkML {gname} operators.",
            body=body,
            nav=sidenav(slug),
            toc=toc_for(toc),
            crumb=crumbs(("API reference", "api.html"), (gname, None)),
            page=slug,
            index_json=index_json,
        )

    for cname, spec in CLASSES.items():
        cbody, ctoc = class_page(cname, spec)
        slug = class_slug(cname)
        write(OUT / f"{slug}.html", title=f"{cname} — vkML",
              desc=spec.get("summary", cname), body=cbody, nav=sidenav(slug),
              toc=toc_for(ctoc), crumb=crumbs(("Classes", None), (cname, None)),
              page=slug, index_json=index_json)

    all_pages = list(PAGES) + capability_pages()
    titles = {s: t for s, t, _ in all_pages}
    order = [s for s in reading_order() if s in titles]

    for slug, title, body in all_pages:
        body = with_diagrams(body, slug)
        body = highlight_raw_blocks(body)
        body = with_heading_ids(body)
        # Prev/next, so the guide reads straight through as well as by search.
        # `pagenav()` and its eight CSS rules existed and were called from
        # nowhere: every page was a dead end for sequential reading, which is
        # what the link graph reported as twenty dead ends.
        if slug in order:
            i = order.index(slug)
            prev = ((titles[order[i - 1]], f"{order[i - 1]}.html") if i > 0 else None)
            nxt = ((titles[order[i + 1]], f"{order[i + 1]}.html")
                   if i + 1 < len(order) else None)
            body += pagenav(prev, nxt)
        heads = [
            (m.group(1), re.sub(r"<[^>]+>", "", m.group(2)), 2)
            for m in re.finditer(r'<h2 id="([^"]+)"[^>]*>(.*?)</h2>', body, re.S)
        ]
        crumb = "" if slug == "index" else crumbs((title, None))
        write(
            OUT / f"{slug}.html",
            title=f"{title} — vkML",
            desc=title,
            body=body,
            # The landing page carries no section tree and no contents rail:
            # a sidebar is a wayfinding aid for a reader who already knows what
            # they are looking for, and a first-time visitor is not that reader.
            # None of the twelve reference sites measured shows its docs tree
            # on the landing page.
            nav="" if slug == "index" else sidenav(slug),
            toc="" if slug == "index" else toc_for(heads),
            crumb=crumb,
            page=slug,
            index_json=index_json,
        )

    total = len(ALL_NAMES)
    print(f"  built {OUT.relative_to(ROOT)}")
    print(
        f"  {total} operators, {documented} with written prose ({100 * documented // total}%)"
    )
    if missing:
        print(f"  {len(missing)} ungrouped: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    rc = build()
    if rc == 0 and a.serve:
        import http.server
        import os

        os.chdir(OUT)
        print("  http://localhost:8000")
        http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=1235)
    sys.exit(rc)
