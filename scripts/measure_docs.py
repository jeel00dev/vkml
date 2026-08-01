#!/usr/bin/env python3
"""Measure the design decisions of well-regarded documentation sites.

Run this before arguing about typography. Every number in docs/ui/README.md
came from it, and it is the instrument that caught two of my own claims being
wrong -- see the note on scoping below.

Opinions about typography are cheap; the numbers are not. This loads each site
at a fixed 1440x900 laptop viewport and reads COMPUTED styles, so what comes
back is what the browser actually did, including whatever the cascade and any
runtime theming settled on.

Measured, and why each one:
  measure_px / measure_ch  the reading column. The single strongest predictor
                           of whether long prose is comfortable.
  body size / line-height  the base of the whole type scale.
  h1..h3 size + weight     the hierarchy, and how much contrast it carries.
  inline code              whether it is a chip with a border or barely marked.
  code block               size and background separation from the page.
  sidebar width + item     nav density: how many entries fit before scrolling.
  colors                   page background, text, and the accent actually used.
"""
import json
import sys

SITES = [
    ("rust-book",   "https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html"),
    ("react",       "https://react.dev/learn/thinking-in-react"),
    ("tailwind",    "https://tailwindcss.com/docs/flex-basis"),
    ("astro",       "https://docs.astro.build/en/guides/routing/"),
    ("stripe",      "https://docs.stripe.com/api/charges/object"),
    ("pytorch",     "https://pytorch.org/docs/stable/generated/torch.matmul.html"),
    ("nextjs",      "https://nextjs.org/docs/app/getting-started/layouts-and-pages"),
    ("shadcn",      "https://ui.shadcn.com/docs/components/button"),
    ("kubernetes",  "https://kubernetes.io/docs/concepts/workloads/pods/"),
    ("jax",         "https://docs.jax.dev/en/latest/quickstart.html"),
    ("hf",          "https://huggingface.co/docs/transformers/en/index"),
    ("godot",       "https://docs.godotengine.org/en/stable/tutorials/2d/index.html"),
]

JS = r"""() => {
  const cs = el => el ? getComputedStyle(el) : null;
  const num = v => parseFloat(v) || 0;

  // The reading column: the widest paragraph with real prose in it. Chosen by
  // content rather than by selector, because every site names its main region
  // differently and a wrong guess silently measures a footer.
  const ps = [...document.querySelectorAll('p')]
    .filter(p => p.textContent.trim().length > 120 && p.offsetWidth > 200);
  ps.sort((a, b) => b.textContent.length - a.textContent.length);
  const p = ps[0];
  const pcs = cs(p);

  // Approximate characters per line from the '0' advance width of the same font.
  let ch = 0;
  if (p) {
    const s = document.createElement('span');
    s.textContent = '0'.repeat(100);
    s.style.cssText = 'position:absolute;visibility:hidden;white-space:pre';
    s.style.font = pcs.font;
    document.body.appendChild(s);
    const w = s.offsetWidth / 100;
    s.remove();
    if (w) ch = Math.round(p.clientWidth / w);
  }

  // Scope every other measurement to the region that CONTAINS the reading
  // column. Querying the document picks up sidebar and footer headings, which
  // silently reports a docs site's nav as its content hierarchy -- it made vkml
  // look like it had a 1.08x h2 when the real figure is 1.60x.
  let root = p ? p.parentElement : document.body;
  while (root && root !== document.body &&
         !/^(MAIN|ARTICLE|SECTION)$/.test(root.tagName) &&
         root.clientWidth < 1.6 * (p ? p.clientWidth : 1)) root = root.parentElement;
  root = root || document.body;
  const pick = sel => root.querySelector(sel);
  const h1 = pick('h1'), h2 = pick('h2'), h3 = pick('h3');
  const code = [...root.querySelectorAll('code')]
    .find(c => !c.closest('pre') && c.offsetWidth);
  const pre = pick('pre');

  // The sidebar: the tallest nav-ish element on the left third of the page.
  const navs = [...document.querySelectorAll('nav, aside, [class*="sidebar"], [class*="Sidebar"]')]
    .filter(n => n.offsetWidth > 120 && n.offsetWidth < 460
                 && n.getBoundingClientRect().left < innerWidth / 3
                 && n.offsetHeight > 300);
  navs.sort((a, b) => b.offsetHeight - a.offsetHeight);
  const nav = navs[0];
  const navLink = nav ? nav.querySelector('a') : null;

  const out = {
    measure_px: p ? Math.round(p.clientWidth) : null,
    measure_ch: ch || null,
    body_family: pcs ? pcs.fontFamily.split(',')[0].replace(/["']/g, '') : null,
    body_size: pcs ? num(pcs.fontSize) : null,
    body_lh: pcs ? +(num(pcs.lineHeight) / num(pcs.fontSize)).toFixed(2) : null,
    body_color: pcs ? pcs.color : null,
    page_bg: getComputedStyle(document.body).backgroundColor,
  };
  for (const [k, el] of [['h1', h1], ['h2', h2], ['h3', h3]]) {
    const c = cs(el);
    out[k + '_size'] = c ? num(c.fontSize) : null;
    out[k + '_weight'] = c ? c.fontWeight : null;
    out[k + '_lh'] = c ? +(num(c.lineHeight) / num(c.fontSize)).toFixed(2) : null;
  }
  const cc = cs(code);
  out.code_inline_size = cc ? num(cc.fontSize) : null;
  out.code_inline_bg = cc ? cc.backgroundColor : null;
  out.code_inline_border = cc ? cc.borderTopWidth : null;
  out.code_inline_family = cc ? cc.fontFamily.split(',')[0].replace(/["']/g, '') : null;
  const pc = cs(pre);
  out.code_block_size = pc ? num(pc.fontSize) : null;
  out.code_block_bg = pc ? pc.backgroundColor : null;
  out.code_block_radius = pc ? pc.borderTopLeftRadius : null;
  const nc = cs(nav), nlc = cs(navLink);
  out.sidebar_w = nav ? Math.round(nav.offsetWidth) : null;
  out.sidebar_item_size = nlc ? num(nlc.fontSize) : null;
  out.sidebar_item_lh = nlc ? num(nlc.lineHeight) : null;
  return out;
}"""


def main():
    from playwright.sync_api import sync_playwright
    rows = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        ctx = b.new_context(viewport={"width": 1440, "height": 900})
        for name, url in SITES:
            page = ctx.new_page()
            try:
                page.goto(url, timeout=45000, wait_until="domcontentloaded")
                page.wait_for_timeout(2500)
                rows[name] = page.evaluate(JS)
                print(f"  {name:11} ok", flush=True)
            except Exception as e:
                rows[name] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
                print(f"  {name:11} FAILED {type(e).__name__}", flush=True)
            page.close()
        b.close()
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
