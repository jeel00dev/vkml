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

sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(WEB))

import vkml as V  # noqa: E402
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
]

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


def sidenav(active: str) -> str:
    s = [
        '<aside class="sidenav" id="sidenav"><h2>Documentation</h2>',
        "<h3>Guide</h3><ol>",
    ]
    for slug, title, _ in PAGES:
        cls = ' class="active"' if slug == active else ""
        s.append(f'<li><a{cls} href="{slug}.html">{html.escape(title)}</a></li>')
    s.append("</ol>")
    if CLASSES:
        s.append("<h3>Classes</h3><ol>")
        for cname in CLASSES:
            cslug = class_slug(cname)
            ccls = ' class="active"' if cslug == active else ""
            s.append(f'<li><a{ccls} href="{cslug}.html"><code>{cname}</code></a></li>')
        s.append("</ol>")
    s.append("<h3>API reference</h3><ol>")
    s.append(
        f"<li><a{' class="active"' if active == 'api' else ''} "
        f'href="api.html">Overview</a></li></ol>'
    )
    for gi, (gname, names) in enumerate(GROUPS, 1):
        slug = group_slug(gname)
        here = slug == active
        # Only the section being read is expanded. All of them open at once is
        # a wall of 99 links; none open is a reference you cannot scan.
        s.append(
            f'<div class="grp{"" if here else " closed"}">'
            f'<button type="button"><span class="chev">▾</span>'
            f"{html.escape(gname)}</button><ol>"
        )
        for ni, n in enumerate(names, 1):
            href = f"#{n}" if here else f"{slug}.html#{n}"
            s.append(
                f'<li><a href="{href}"><span class="num">{gi}.{ni}</span>'
                f"<code>{n}</code></a></li>"
            )
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
<link rel="icon" href="assets/vkml_logo.png">
<link rel="stylesheet" href="theme/vkml.css">
</head>
<body>
<header class="topbar">
  <button class="icon-btn" id="nav-toggle" aria-label="Toggle navigation">☰</button>
  <a class="brand" href="index.html">
    <img src="assets/vkml_logo.png" alt=""><span>vk<span class="ml">ML</span></span>
  </a>
  <span class="ver">{version}</span>
  <nav>
    <a href="get-started.html">Get started</a>
    <a href="concepts.html">Concepts</a>
    <a class="{apicls}" href="api.html">API reference</a>
  </nav>
  <span class="spacer"></span>
  <div class="search">
    <span class="mag">⌕</span>
    <input id="q" type="search" placeholder="Search the docs …" autocomplete="off"
           spellcheck="false" aria-label="Search">
    <kbd>/</kbd>
    <div class="results" id="results"></div>
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
            desc=html.escape(plain(desc)),
            body=body,
            sidenav=nav,
            toc=toc,
            crumbs=crumb,
            version=html.escape(V.__version__),
            apicls="active" if page == "api" else "",
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
      var starts = [], has = [], desc = [];
      idx.forEach(function (e) {
        var n = e.n.toLowerCase();
        if (n.indexOf(v) === 0) starts.push(e);
        else if (n.indexOf(v) !== -1) has.push(e);
        else if ((e.d || '').toLowerCase().indexOf(v) !== -1) desc.push(e);
      });
      render(starts.concat(has, desc));
    });
    q.addEventListener('keydown', function (e) {
      var items = box.querySelectorAll('a');
      if (e.key === 'Escape') { box.classList.remove('open'); q.blur(); }
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!items.length) return;
        if (sel >= 0) items[sel].classList.remove('sel');
        sel = (sel + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
        items[sel].classList.add('sel'); items[sel].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter' && sel >= 0) { items[sel].click(); }
    });
    addEventListener('keydown', function (e) {
      if (e.key === '/' && document.activeElement !== q) { e.preventDefault(); q.focus(); }
    });
    addEventListener('click', function (e) {
      if (!e.target.closest('.search')) box.classList.remove('open');
    });
  }
})();
"""


def build() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(WEB / "theme", OUT / "theme")
    (OUT / "theme" / "site.js").write_text(SITE_JS)
    (OUT / "assets").mkdir()
    shutil.copy(ROOT / "assets" / "vkml_logo.png", OUT / "assets" / "vkml_logo.png")
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
                {"n": n, "u": f"{group_slug(gname)}.html#{n}", "g": gname, "d": summary}
            )
        documented += ndoc
        per_group.append((gname, names, frags, toc, ndoc))

    for slug, title, _ in PAGES:
        index.append({"n": title, "u": f"{slug}.html", "g": "Guide", "d": ""})
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
        index.append({"n": cname, "u": f"{slug}.html", "g": "Classes",
                      "d": spec.get("summary", "")})
        write(OUT / f"{slug}.html", title=f"{cname} — vkML",
              desc=spec.get("summary", cname), body=cbody, nav=sidenav(slug),
              toc=toc_for(ctoc), crumb=crumbs(("Classes", None), (cname, None)),
              page=slug, index_json=index_json)

    for slug, title, body in PAGES:
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
            nav=sidenav(slug),
            toc=toc_for(heads),
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
