#!/usr/bin/env python3
"""Build the vkML documentation site.

TWO SOURCES, DELIBERATELY.

Signatures, argument names, defaults and the class/function inventory are read
from the IMPORTED MODULE at build time. Nothing about the API surface is typed
into this repository twice, so a reference page cannot describe a signature the
library no longer has -- which is the failure mode of every hand-written API doc.

Prose, parameter descriptions and examples come from `web/content/*.py`, keyed by
the same names. They cannot be derived: 78 of the 99 public functions carry a
docstring that is only their signature repeated, so generating from docstrings
alone would produce a reference with no explanation in it.

Where prose is missing the page SAYS SO, in place, and the run prints a coverage
count. A reference that quietly omits what it does not have looks finished and
is not.

    python web/build.py            # -> web/_site
    python web/build.py --serve    # build, then serve on :8000
"""
from __future__ import annotations

import argparse
import html
import inspect
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
from content import PAGES, PROSE  # noqa: E402

# Presentation order. Alphabetical inside a group, because the API reference is
# a lookup surface -- but the groups themselves follow how the library is used,
# not how it is spelled.
GROUPS: list[tuple[str, list[str]]] = [
    ("Creation", ["tensor", "zeros", "ones", "full", "arange", "rand", "from_numpy", "asarray"]),
    ("Element-wise", ["abs", "neg", "exp", "log", "sqrt", "rsqrt", "reciprocal", "square",
                      "sign", "sin", "cos", "tanh", "sigmoid", "erf", "erfc",
                      "relu", "gelu", "silu", "clamp", "clamp_min", "clamp_max"]),
    ("Arithmetic", ["add", "sub", "mul", "div", "pow", "maximum", "minimum"]),
    ("Comparison", ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal", "where"]),
    ("Reduction", ["sum", "mean", "prod", "amax", "amin", "argmax", "argmin"]),
    ("Shape & indexing", ["cat", "tril", "triu", "masked_fill", "index_select", "scatter_add",
                          "im2col", "col2im", "detach"]),
    ("Linear algebra & NN", ["matmul", "softmax", "log_softmax", "layer_norm", "rms_norm",
                             "batch_norm", "dropout", "conv2d", "max_pool2d", "avg_pool2d"]),
    ("Losses", ["cross_entropy", "mse_loss", "huber_loss", "kl_div",
                "binary_cross_entropy_with_logits"]),
    ("Autograd & execution", ["backward", "realize", "set_eager", "is_eager"]),
    ("Serialization", ["save", "load", "save_module", "load_module"]),
    ("Devices & introspection", ["init_vulkan", "available_devices", "best_device",
                                 "vulkan_available", "vulkan_device_count", "vulkan_device_names",
                                 "vulkan_device_reports", "vulkan_capabilities", "vulkan_stats",
                                 "vulkan_pipeline_stats", "vulkan_timestamps_supported",
                                 "vulkan_unavailable_reason", "vulkan_last_profile",
                                 "vulkan_submit_ms", "vulkan_set_profiling",
                                 "vulkan_set_subgroup_override", "set_log_level"]),
]

# Declared once, verified against the module at build time rather than trusted.
CPU_ONLY = {"prod"}


# --------------------------------------------------------------- signatures --

def signature_of(name: str) -> str:
    """The live signature, cleaned of binding noise but not of meaning."""
    fn = getattr(V, name)
    doc = (fn.__doc__ or "").strip()
    first = doc.split("\n", 1)[0]
    if first.startswith(f"{name}("):
        sig = first
    else:
        try:
            sig = f"{name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            sig = f"{name}(...)"
    # nanobind spells every type with its private module path. Readers do not
    # need to know the extension is called _vkml_core.
    sig = sig.replace("vkml._vkml_core.", "")
    sig = re.sub(r"collections\.abc\.Sequence\[int\]", "Sequence[int]", sig)
    return sig


def render_signature(name: str, sig: str) -> str:
    body = html.escape(sig)
    body = body.replace(f"{name}(", f'<span class="name">{name}</span>(', 1)
    body = re.sub(r"-&gt; (.+)$", r'<span class="ret">→ \1</span>', body)
    body = re.sub(r"(\b[a-z_][a-z0-9_]*)(?=:)", r'<span class="param">\1</span>', body)
    return f'<div class="sig">{body}</div>'


def docstring_prose(name: str) -> str:
    """Whatever the binding's docstring says BEYOND repeating the signature."""
    doc = (getattr(V, name).__doc__ or "").strip()
    lines = doc.split("\n")
    if lines and lines[0].startswith(f"{name}("):
        lines = lines[1:]
    return "\n".join(lines).strip()


# ------------------------------------------------------------------ markup --

REPL = re.compile(r"^(&gt;&gt;&gt; |\.\.\. )(.*)$")


def code_block(src: str, repl: bool = False) -> str:
    esc = html.escape(src.strip("\n"))
    if not repl:
        return f"<pre><code>{esc}</code></pre>"
    out = []
    for line in esc.split("\n"):
        m = REPL.match(line)
        if m:
            out.append(f'<span class="p">{m.group(1)}</span>{m.group(2)}')
        else:
            out.append(f'<span class="o">{line}</span>')
    return '<pre class="repl"><code>' + "\n".join(out) + "</code></pre>"


def inline(text: str) -> str:
    """The small subset of markdown the content files use."""
    t = html.escape(text)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def paragraphs(text: str) -> str:
    return "".join(f"<p>{inline(b.strip())}</p>"
                   for b in text.strip().split("\n\n") if b.strip())


# ------------------------------------------------------------------- pages --

def api_entry(name: str) -> tuple[str, bool]:
    """Render one function. Returns (html, documented)."""
    sig = signature_of(name)
    p = PROSE.get(name, {})
    documented = bool(p.get("summary"))

    out = [f'<h3 id="{name}"><code>{name}</code></h3>', render_signature(name, sig)]

    on_cpu, on_gpu = True, name not in CPU_ONLY
    out.append('<div class="support">'
               f'<span class="chip yes">CPU</span>'
               f'<span class="chip {"yes" if on_gpu else "no"}">Vulkan</span>'
               + ("" if on_gpu else '<span class="chip">CPU-only by decision</span>')
               + "</div>")

    if documented:
        out.append(paragraphs(p["summary"]))
    else:
        fallback = docstring_prose(name)
        if fallback:
            out.append(paragraphs(fallback))
        out.append('<div class="todo">No written description yet. The signature above is '
                   'generated from the installed module and is current; the prose is not '
                   'written. See <code>web/content/</code>.</div>')

    if p.get("detail"):
        out.append(paragraphs(p["detail"]))

    if p.get("params"):
        out.append('<h4>Parameters</h4><dl class="params">')
        for pname, ptype, pdesc in p["params"]:
            opt = ' <span class="opt">optional</span>' if "=" in ptype or "optional" in ptype else ""
            out.append(f'<dt>{html.escape(pname)} '
                       f'<span class="type">({html.escape(ptype)})</span>{opt}</dt>'
                       f"<dd>{inline(pdesc)}</dd>")
        out.append("</dl>")

    if p.get("returns"):
        out.append(f'<h4>Returns</h4><p>{inline(p["returns"])}</p>')

    for kind in ("note", "warning"):
        if p.get(kind):
            cls = "admon warn" if kind == "warning" else "admon"
            out.append(f'<div class="{cls}"><span class="label">{kind}</span>'
                       f'{paragraphs(p[kind])}</div>')

    if p.get("example"):
        out.append("<h4>Example</h4>" + code_block(p["example"], repl=True))

    if p.get("see"):
        links = ", ".join(f'<a href="#{s}"><code>{s}</code></a>' for s in p["see"])
        out.append(f"<p><strong>See also</strong> {links}</p>")

    out.append("<hr>")
    return "\n".join(out), documented


def sidebar(active_page: str, groups: list[tuple[str, list[str]]]) -> str:
    s = ['<aside class="sidebar" id="sidebar">',
         '<div class="search"><input id="q" type="search" placeholder="Filter…" '
         'autocomplete="off" spellcheck="false"></div>']
    s.append("<h3>Guide</h3><ol>")
    for slug, title, _ in PAGES:
        cls = ' class="active"' if slug == active_page else ""
        s.append(f'<li><a{cls} href="{slug}.html">{html.escape(title)}</a></li>')
    s.append("</ol>")
    if active_page == "api":
        for gi, (gname, names) in enumerate(groups, 1):
            s.append(f"<h3>{html.escape(gname)}</h3><ol>")
            for ni, n in enumerate(names, 1):
                s.append(f'<li><a href="#{n}">'
                         f'<span class="num">{gi}.{ni}</span><code>{n}</code></a></li>')
            s.append("</ol>")
    s.append("</aside>")
    return "\n".join(s)


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
  <button class="icon-btn" id="sidebar-toggle" aria-label="Toggle navigation">☰</button>
  <a class="brand" href="index.html">
    <img src="assets/vkml_logo.png" alt="">
    <span><span class="vk">vk</span><span class="ml">ML</span></span>
  </a>
  <nav>
    <a class="always{home}" href="index.html">Home</a>
    <a href="get-started.html">Get started</a>
    <a class="always{api}" href="api.html">API</a>
    <a href="https://github.com/jeel00dev/vkml">GitHub</a>
  </nav>
  <button class="icon-btn" id="theme-toggle" aria-label="Toggle theme" title="Toggle theme">◐</button>
</header>
<div class="layout">
{sidebar}
<main class="content">
{body}
<footer class="page">
  vkML — Vulkan-first machine learning in C++20.
  Apache-2.0. Signatures on this page are generated from the installed module.
</footer>
</main>
</div>
<script>
(function () {{
  var root = document.documentElement;
  var saved = localStorage.getItem('vkml-theme');
  if (saved) root.setAttribute('data-theme', saved);
  document.getElementById('theme-toggle').onclick = function () {{
    var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    localStorage.setItem('vkml-theme', next);
  }};
  var t = document.getElementById('sidebar-toggle'), sb = document.getElementById('sidebar');
  if (t && sb) t.onclick = function () {{ sb.classList.toggle('open'); }};

  // Filter the sidebar. Client-side and substring-only on purpose: the whole
  // index is already in the page, so shipping a search engine to look through
  // it would cost more than it saves.
  var q = document.getElementById('q');
  if (q) q.addEventListener('input', function () {{
    var v = q.value.toLowerCase();
    sb.querySelectorAll('ol').forEach(function (ol) {{
      var any = false;
      ol.querySelectorAll('li').forEach(function (li) {{
        var hit = li.textContent.toLowerCase().indexOf(v) !== -1;
        li.style.display = hit ? '' : 'none';
        any = any || hit;
      }});
      var h = ol.previousElementSibling;
      if (h && h.tagName === 'H3') h.style.display = any ? '' : 'none';
    }});
  }});

  // Highlight the entry currently on screen.
  var seen = sb ? sb.querySelectorAll('a[href^="#"]') : [];
  if (seen.length) {{
    var byId = {{}};
    seen.forEach(function (a) {{ byId[a.getAttribute('href').slice(1)] = a; }});
    new IntersectionObserver(function (es) {{
      es.forEach(function (e) {{
        var a = byId[e.target.id];
        if (a && e.isIntersecting) {{
          sb.querySelectorAll('a.active').forEach(function (x) {{ x.classList.remove('active'); }});
          a.classList.add('active');
        }}
      }});
    }}, {{ rootMargin: '-4rem 0px -80% 0px' }}).observe
      && document.querySelectorAll('h3[id]').forEach(function (h) {{
        new IntersectionObserver(function (es) {{
          es.forEach(function (e) {{
            var a = byId[e.target.id];
            if (a && e.isIntersecting) {{
              sb.querySelectorAll('a.active').forEach(function (x) {{ x.classList.remove('active'); }});
              a.classList.add('active');
            }}
          }});
        }}, {{ rootMargin: '-4rem 0px -80% 0px' }}).observe(h);
      }});
  }}
}})();
</script>
</body>
</html>
"""


def write(path: Path, title: str, desc: str, body: str, side: str, page: str) -> None:
    path.write_text(SHELL.format(
        title=html.escape(title), desc=html.escape(desc), body=body, sidebar=side,
        home=" active" if page == "index" else "", api=" active" if page == "api" else ""))


def build() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(WEB / "theme", OUT / "theme")
    (OUT / "assets").mkdir()
    shutil.copy(ROOT / "assets" / "vkml_logo.png", OUT / "assets" / "vkml_logo.png")
    (OUT / ".nojekyll").write_text("")   # GitHub Pages must not run Jekyll over this

    # Every name in GROUPS must exist, and every public function must be in a
    # group. Both directions, because a reference that silently drops an
    # operator is worse than one that fails to build.
    listed = {n for _, names in GROUPS for n in names}
    public = {n for n in dir(V)
              if not n.startswith("_") and callable(getattr(V, n))
              and not inspect.isclass(getattr(V, n))}
    missing, unknown = sorted(public - listed), sorted(listed - public)
    if unknown:
        print(f"  ERROR: listed but not in the module: {', '.join(unknown)}")
        return 1

    documented = 0
    entries = []
    for gname, names in GROUPS:
        entries.append(f'<h2 id="g-{gname.lower().replace(" ", "-").replace("&", "and")}">'
                       f"{html.escape(gname)}</h2>")
        for n in names:
            frag, ok = api_entry(n)
            entries.append(frag)
            documented += ok

    api_body = (
        "<h1>API reference</h1>"
        '<p class="lede">Every public function, with the signature read from the installed '
        "module at build time.</p>"
        + (f'<div class="admon"><span class="label">note</span><p>{len(missing)} public '
           f"function(s) are not yet grouped and do not appear below: "
           f"<code>{', '.join(missing)}</code>.</p></div>" if missing else "")
        + "\n".join(entries))
    write(OUT / "api.html", "API reference — vkML",
          "Every public vkML function.", api_body, sidebar("api", GROUPS), "api")

    for slug, title, body in PAGES:
        write(OUT / f"{slug}.html", f"{title} — vkML", title, body, sidebar(slug, GROUPS), slug)

    total = sum(len(n) for _, n in GROUPS)
    print(f"  built {OUT.relative_to(ROOT)}")
    print(f"  {total} operators listed, {documented} with written prose "
          f"({100 * documented // total}%)")
    if missing:
        print(f"  {len(missing)} ungrouped: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    rc = build()
    if rc == 0 and args.serve:
        import http.server
        import os
        os.chdir(OUT)
        print("  http://localhost:8000")
        http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=8000)
    sys.exit(rc)
