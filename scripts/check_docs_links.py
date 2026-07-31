import re, pathlib
site = pathlib.Path("web/_site")
pages = {p.name for p in site.glob("*.html")}
anchors = {p.name: set(re.findall(r'id="([^"]+)"', p.read_text())) for p in site.glob("*.html")}
bad = []
for p in site.glob("*.html"):
    for href in re.findall(r'href="([^"]+)"', p.read_text()):
        if href.startswith(("http", "#", "mailto")):
            continue
        f, _, frag = href.partition("#")
        if f and f not in pages and not (site / f).exists():
            bad.append(f"{p.name}: missing page {f}")
        elif frag and f in anchors and frag not in anchors[f]:
            bad.append(f"{p.name}: {f}#{frag} missing")
print(f"  {len(pages)} pages, {sum(len(a) for a in anchors.values())} anchors")
print(f"  broken links: {len(bad)}")
for b in bad[:6]:
    print("   ", b)
