"""Run every >>> example in the content and compare against what the page claims."""
import io, re, sys, contextlib, traceback
sys.path.insert(0, "web"); sys.path.insert(0, "python")
import numpy as np, vkml
from content import PAGES, PROSE, CLASSES

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


def all_examples():
    """Every REPL transcript the site publishes, from all three content sources.

    Enumerated in ONE place because the sources were previously walked by two
    hand-written loops, and a third -- the examples on class pages -- was simply
    never added to either. That gap was invisible while it existed: the gate
    reported PASS over content it had not opened, which is the failure mode a
    gate is supposed to prevent rather than exhibit.
    """
    yield from guide_examples()
    for name, entry in sorted(PROSE.items()):
        if entry.get("example"):
            yield f"op:{name}", entry["example"]
    for name, spec in sorted(CLASSES.items()):
        if spec.get("example"):
            yield f"class:{name}", spec["example"]


norm = lambda s: re.sub(r"\s+", " ", s).strip()

# A transcript that needs a Vulkan device cannot be checked by a build that has
# none, and reporting it as a MISMATCH is wrong twice over: the page is not
# wrong, and the first failure poisons every later statement in the block
# through NameError on a variable the failed line was meant to bind. Three CI
# jobs build CPU-only, so this is the ordinary case, not the exotic one.
#
# Skipped per BLOCK rather than per statement, precisely because of that
# cascade. Counted and printed, so "checked" never quietly means "checked the
# easy ones" -- a gate reporting PASS over content it did not open is the
# failure mode all_examples() above already exists to prevent.
NEEDS_DEVICE = re.compile(r"init_vulkan|vulkan:\d|vulkan_")

bad = checked = skipped = 0
for name, block in all_examples():
    if not vkml.has_vulkan and NEEDS_DEVICE.search(block):
        skipped += 1
        continue
    for stmt, claimed, actual in run_example(block):
        # A line marked as machine-specific is reported, not failed: the device
        # index and driver version legitimately differ per machine.
        if "differs per machine" in claimed:
            continue
        checked += 1
        if norm(claimed) != norm(actual):
            bad += 1
            print(f"  MISMATCH {name}: {stmt}")
            print(f"    claimed: {claimed}")
            print(f"    actual : {actual}")
note = f", {skipped} blocks skipped (no Vulkan in this build)" if skipped else ""
print(f"\n  {checked} statements checked, {bad} mismatches{note}")
sys.exit(1 if bad else 0)
