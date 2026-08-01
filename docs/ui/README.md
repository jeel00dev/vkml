# The documentation site: measurements, evidence and instruments

The site is judged by how it reads, and that cannot be assessed from the
generator's source. This directory holds the instruments and the evidence, so a
UI claim is checkable the way a numerical claim is.

Two tools:

```sh
python scripts/shoot_docs.py OUT --theme both   # render at 4 viewports, 2 themes
python scripts/measure_docs.py                  # computed styles of 12 reference sites
```

`shoot_docs.py` renders with the same engine most readers use, at 390, 768, 1280
and 1920 — the widths where layouts actually break rather than round numbers.
`measure_docs.py` loads reference documentation sites and reads **computed**
styles, so what comes back is what the browser did rather than what a stylesheet
appears to say.

---

## The reference measurements

Taken at 1440x900. `ch` is characters per line, derived from the `0` advance
width of the page's own body font.

| site | measure | ch | body | h2/body | inline code |
|---|---:|---:|---:|---:|---:|
| stripe | 503 | 65 | 14 | — | 0.85× |
| shadcn/ui | 640 | 64 | 15 | 1.25× | 0.85× |
| tailwind | 672 | 67 | 14 | — | 0.86× |
| kubernetes | 707 | 79 | 16 | 2.00× | 0.85× |
| next.js | 715 | 65 | 16 | 1.50× | 0.90× |
| astro | 720 | 79 | 16 | 2.19× | 0.81× |
| rust book | 750 | 83 | 16 | 1.50× | 0.88× |
| jax | 790 | 89 | 16 | 2.12× | — |
| godot | 796 | 80 | 16 | 1.50× | — |
| huggingface | 804 | 101 | 16.8 | 1.19× | 0.83× |
| react | 896 | 75 | 17 | 1.65× | 0.90× |
| pytorch | 896 | 101 | 16 | — | 0.88× |
| **peer median** | | **79** | **16** | **1.50×** | **0.86×** |
| **vkml, before** | 800 | **93** | 15.5 | 1.60× | 0.875em |
| **vkml, now** | 656 | **74** | 16 | 1.60× | 0.875em |

Also measured, at 1280: a right-hand contents rail is present on seven of the
twelve (tailwind 288px, astro 280, shadcn 288, jax 272, huggingface 270,
next.js 224, kubernetes 209) and absent on five (rust book, react, stripe,
pytorch, godot). The five that omit it are single-narrative reading pages; none
of them hides it as late as vkml did, at 1408px.

### Scope the probe, or it measures the navigation

The first run of `measure_docs.py` reported vkml's `h2` at 1.08× body and its
inline code at 0.79× — a nearly flat hierarchy and the smallest code on any site
measured. Both were false. It was querying the whole document, and on a
documentation page the first `h2` is usually in the sidebar. Scoped to the
region containing the reading column, and checked against the stylesheet, `h2`
is 1.60× and inline code is `.875em`: both already better than the median.

Two things nearly got "fixed" that were not broken. The same mistake later
produced two false positives in the heading-id gate, which reported the
sidebar's own headings and then the API index's card titles. Anything that
inspects a rendered docs page has to say which part of it it means.

---

## Evidence

`evidence/` holds before/after pairs for the changes where a screenshot is the
proof. One pair per thing demonstrated, not one per page, at CSS-pixel scale and
quantised — the whole set is under 1 MB, which is what makes it affordable to
keep in the repository forever.

| pair | shows |
|---|---|
| `measure-and-entry-hierarchy` | the reading column at 93 → 74 characters; the operator name promoted from a code chip to the entry's heading; the signature's accent bar reduced; inline code without borders; the contents rail restored |
| `measure-and-entry-hierarchy-light` | the same in the light theme, where removing code borders was the risk — `#f5f6f8` on white is a subtle tint to carry the distinction alone |
| `phone-api-entry` | the 390px header, where search collapses to about 130px and shows a truncated placeholder |
| `contents-rail-and-heading-ids` | "Get started" going from an empty contents rail to six linkable sections |
| `landing` | the landing page rendering inside the documentation shell |

To regenerate after a change:

```sh
python web/build.py && python scripts/shoot_docs.py /tmp/after --theme both
```

---

## Gates

Three checks in `scripts/check_docs_references.py` exist because of mistakes
found here, each red-verified:

- **unrendered markup** — `*emphasis*` was never implemented while authors used
  it, so five API pages printed literal asterisks; and `<meta description>`
  reused the authored summary, so seven pages advertised themselves with the
  markdown intact. Nobody reads their own meta tags, which is why that lasted.
- **headings without ids** — guide pages are raw HTML, so linkability depended
  on whether the author typed an `id`. Thirteen sections across the two pages a
  newcomer opens first could not be linked or listed.
- **path references in the hand-written docs** — 241 citations that nothing
  checked, including the vendored CUTLASS and llama.cpp paths that justify most
  of the design decisions.
