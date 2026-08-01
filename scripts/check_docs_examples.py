"""Run every >>> example in the content and compare against what the page claims."""
import io, re, sys, contextlib, traceback
sys.path.insert(0, "web"); sys.path.insert(0, "python")
import numpy as np, vkml
from content import PAGES, PROSE

def run_example(src):
    """Execute a REPL transcript; return (claimed, actual) output pairs."""
    ns = {"np": np, "vkml": vkml}
    lines, out, i = src.strip().split("\n"), [], 0
    while i < len(lines):
        if not lines[i].startswith(">>> "):
            i += 1; continue
        stmt, i = lines[i][4:], i + 1
        skip = "doctest: +SKIP" in stmt
        stmt = stmt.split("#")[0].rstrip() if skip else stmt
        while i < len(lines) and lines[i].startswith("... "):
            stmt += "\n" + lines[i][4:]; i += 1
        claimed = []
        while i < len(lines) and not lines[i].startswith(">>> "):
            claimed.append(lines[i]); i += 1
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    val = eval(compile(stmt, "<ex>", "eval"), ns)
                    if val is not None:
                        print(repr(val))
                except SyntaxError:
                    exec(compile(stmt, "<ex>", "exec"), ns)
        except Exception as e:
            out.append((stmt, "\n".join(claimed), f"{type(e).__name__}: {e}")); continue
        actual = buf.getvalue().rstrip("\n")
        if skip:
            continue
        if "\n".join(claimed).strip() or actual:
            out.append((stmt, "\n".join(claimed).strip(), actual.strip()))
    return out

def guide_examples():
    """REPL transcripts embedded in the guide pages' HTML.

    These were invisible to this gate until now, and drifted exactly as the
    unchecked always do: get-started printed `[1024, 1024]` for a `.shape` that
    returns a tuple -- the same mistake already found and fixed in PROSE, which
    survived here only because nothing looked.

    The HTML wraps each line in spans, so the markup is stripped back to the
    transcript before running it.
    """
    for slug, _title, html_body in PAGES:
        for m in re.finditer(r'<pre class="repl"><code>(.*?)</code></pre>', html_body, re.S):
            block = re.sub(r"<[^>]+>", "", m.group(1))
            block = (block.replace("&gt;", ">").replace("&lt;", "<")
                          .replace("&amp;", "&").replace("&quot;", '"'))
            yield f"guide:{slug}", block


bad = 0
for name, block in guide_examples():
    for stmt, claimed, actual in run_example(block):
        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        # A line marked as machine-specific is reported, not failed: the device
        # index and driver version legitimately differ per machine.
        if "differs per machine" in claimed:
            continue
        if norm(claimed) != norm(actual):
            bad += 1
            print(f"  MISMATCH {name}: {stmt}")
            print(f"    claimed: {claimed}")
            print(f"    actual : {actual}")

for name, entry in sorted(PROSE.items()):
    ex = entry.get("example")
    if not ex:
        continue
    for stmt, claimed, actual in run_example(ex):
        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        if norm(claimed) != norm(actual):
            bad += 1
            print(f"  MISMATCH {name}: {stmt}")
            print(f"    claimed: {claimed}")
            print(f"    actual : {actual}")
print(f"\n  {bad} mismatches")
sys.exit(1 if bad else 0)
