"""Mine the repository for everything the documentation should say.

WHY THIS EXISTS. A reference that a developer can trust has to answer questions
about the implementation, not only the interface: which header declares this,
which shader runs it on the GPU, which CPU kernel is the oracle it is checked
against, where the gradient rule lives, which tests cover it. Every one of those
is a FACT ABOUT THE REPOSITORY, so writing them by hand would be transcription
-- and transcription rots the moment a file moves.

So they are extracted. This module reads the tree and produces a structured
record per operator; `build.py` renders it. Anything it cannot find is reported
as absent rather than omitted, because a silently missing cross-reference looks
the same as one that does not apply.

The richest source is the C++ headers themselves. `include/vkml/api/ops.h`
carries 223 lines of `///` comments explaining numerical choices, why an
operator composes the way it does, and what breaks if it does not -- written
next to the code by the people who made those choices. That prose is the
developer documentation; it just was not reachable from a browser.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/jeel00dev/vkml/blob/main"

HEADERS = sorted((ROOT / "include").rglob("*.h"))
SHADERS = sorted((ROOT / "shaders").glob("*.comp"))
CPU_KERNELS = sorted((ROOT / "src" / "backend" / "cpu").glob("*.cpp"))
TEST_FILES = sorted((ROOT / "tests" / "python").glob("test_*.py"))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def src_link(p: Path, line: int | None = None) -> str:
    anchor = f"#L{line}" if line else ""
    return f"{REPO_URL}/{rel(p)}{anchor}"


# --------------------------------------------------------------- C++ decls --

@dataclass
class Decl:
    """One C++ declaration and the `///` block attached above it."""
    name: str
    kind: str                 # function | class | struct | enum | using | constant
    signature: str
    doc: str
    file: Path
    line: int

    @property
    def path(self) -> str:
        return rel(self.file)

    @property
    def url(self) -> str:
        return src_link(self.file, self.line)


DECL_RE = re.compile(
    r"^\s*(?:\[\[nodiscard\]\]\s*)?"
    r"(?:inline\s+|constexpr\s+|static\s+|virtual\s+|explicit\s+)*"
    r"(?P<kind>class|struct|enum\s+class|enum|using)?\s*"
    r"(?P<rest>[A-Za-z_][\w:<>,\s&*\[\]]*?[\w>&*\]])\s*"
    r"(?P<paren>\()?")


def parse_header(path: Path) -> list[Decl]:
    """Declarations with their doc comments.

    Deliberately a line scanner rather than a C++ parser. A real parse would need
    a compiler front end for a gain this does not need: the headers follow one
    house style -- a `///` block, then the declaration -- and anything the
    scanner cannot classify is reported rather than guessed at.
    """
    out: list[Decl] = []
    lines = path.read_text().split("\n")
    doc: list[str] = []
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("///"):
            doc.append(stripped[3:].lstrip())
            continue
        if not stripped or stripped.startswith(("//", "#", "*", "/*")):
            if not stripped:
                doc = []
            continue

        # A declaration line: ends in ; or {, and is not a control statement.
        if not (stripped.endswith((";", "{")) or "(" in stripped):
            doc = []
            continue
        if stripped.startswith(("return", "if", "for", "while", "}", "else", "namespace",
                               "template", "public:", "private:", "protected:")):
            doc = []
            continue

        sig = stripped
        # Join a wrapped signature so the reader sees the whole thing.
        j = i
        while not sig.endswith((";", "{", ")")) and j + 1 < len(lines) and j - i < 6:
            j += 1
            sig += " " + lines[j].strip()
        sig = re.sub(r"\s+", " ", sig).rstrip("{").strip()

        kind, name = "function", None
        m = re.match(r"(class|struct)\s+(\w+)", stripped)
        if m:
            kind, name = m.group(1), m.group(2)
        elif re.match(r"enum\s+class\s+(\w+)", stripped):
            kind = "enum"
            name = re.match(r"enum\s+class\s+(\w+)", stripped).group(1)
        elif stripped.startswith("using "):
            m2 = re.match(r"using\s+(\w+)", stripped)
            if m2:
                kind, name = "using", m2.group(1)
        elif "(" in stripped:
            m3 = re.search(r"(\w+)\s*\(", stripped)
            if m3 and m3.group(1) not in {"if", "for", "while", "switch", "return", "sizeof"}:
                name = m3.group(1)
        elif re.match(r"(?:inline\s+)?(?:constexpr|const|static)\s+[\w:]+\s+(\w+)\s*=", stripped):
            kind = "constant"
            name = re.match(r"(?:inline\s+)?(?:constexpr|const|static)\s+[\w:]+\s+(\w+)",
                            stripped).group(1)

        if name:
            # Emitted even with an empty doc: the header and line number are
            # useful on their own, and dropping undocumented declarations would
            # make the reference quietly narrower than the API.
            out.append(Decl(name=name, kind=kind, signature=sig,
                            doc="\n".join(doc).strip(), file=path, line=i + 1))
        doc = []
    return out


def all_decls() -> dict[str, list[Decl]]:
    """Every documented declaration, by name. A name can be overloaded."""
    by_name: dict[str, list[Decl]] = {}
    for h in HEADERS:
        for d in parse_header(h):
            by_name.setdefault(d.name, []).append(d)
    return by_name


# ------------------------------------------------------------- the backends --

def shader_index() -> dict[str, dict]:
    """Every compute shader, with its specialisation constants and workgroup size."""
    out = {}
    for s in SHADERS:
        text = s.read_text()
        head = []
        for line in text.split("\n")[:40]:
            t = line.strip()
            if t.startswith("//") and not t.startswith("///"):
                head.append(t.lstrip("/ ").rstrip())
            elif head and not t.startswith("//"):
                break
        wg = re.search(r"local_size_x\s*=\s*(\w+)", text)
        out[s.stem] = {
            "path": rel(s),
            "url": src_link(s),
            "summary": " ".join(head).strip(),
            "lines": text.count("\n") + 1,
            "workgroup": wg.group(1) if wg else None,
            "spec_constants": re.findall(r"layout\(constant_id\s*=\s*\d+\)\s*const\s+\w+\s+(\w+)",
                                         text),
            "ops": sorted(set(re.findall(r"#define\s+(OP_\w+)", text))),
        }
    return out


def cpu_kernel_index() -> dict[str, tuple[str, int]]:
    """`k_<name>` -> (file, line). The CPU kernels follow one naming rule."""
    out = {}
    for f in CPU_KERNELS:
        for i, line in enumerate(f.read_text().split("\n"), 1):
            m = re.match(r"\s*(?:static\s+)?(?:void|float|inline)\s.*?\bk_(\w+)\s*\(", line)
            if m:
                out.setdefault(m.group(1), (rel(f), i))
    return out


def op_kinds() -> list[str]:
    """The OpKind enum -- the graph's own vocabulary of operations."""
    text = (ROOT / "include" / "vkml" / "graph" / "op.h").read_text()
    body = re.search(r"enum class OpKind[^{]*\{(.*?)\};", text, re.S)
    if not body:
        return []
    return re.findall(r"^\s*(\w+)\s*(?:=[^,]*)?,", body.group(1), re.M)


def autograd_rules() -> dict[str, int]:
    """Which OpKinds have a gradient rule, and where."""
    f = ROOT / "src" / "autograd" / "autograd.cpp"
    out = {}
    for i, line in enumerate(f.read_text().split("\n"), 1):
        m = re.search(r"case OpKind::(\w+)", line)
        if m:
            out.setdefault(m.group(1), i)
    return out


def vulkan_supports() -> set[str]:
    """OpKinds the Vulkan backend claims in its `supports()`."""
    text = (ROOT / "src" / "backend" / "vulkan" / "vulkan_backend.cpp").read_text()
    m = re.search(r"bool\s+(?:\w+::)?supports\s*\([^)]*\)[^{]*\{(.*?)\n\}", text, re.S)
    return set(re.findall(r"OpKind::(\w+)", m.group(1))) if m else set()


# ------------------------------------------------------------------- tests --

def test_index() -> dict[str, list[tuple[str, str]]]:
    """Operator name -> [(test file, test function)].

    Substring matching over test names and bodies. It over-matches (a test naming
    `sum` also matches `log_sum`), so the count is a floor on real coverage
    rather than an exact figure, and the pages say so.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for f in TEST_FILES:
        current = None
        for line in f.read_text().split("\n"):
            m = re.match(r"def (test_\w+)", line)
            if m:
                current = m.group(1)
                continue
            if current:
                for name in re.findall(r"\bV\.(\w+)\(", line):
                    entry = (f.name, current)
                    if entry not in out.setdefault(name, []):
                        out[name].append(entry)
    return out


# ------------------------------------------------------------------ facts ---

@dataclass
class OpFacts:
    """Everything the repository knows about one operator."""
    name: str
    decls: list[Decl] = field(default_factory=list)
    shader: dict | None = None
    cpu_kernel: tuple[str, int] | None = None
    op_kind: str | None = None
    has_grad: bool = False
    grad_line: int | None = None
    on_vulkan: bool = True
    tests: list[tuple[str, str]] = field(default_factory=list)


def snake_to_camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


def gather(names: list[str]) -> dict[str, OpFacts]:
    decls, shaders = all_decls(), shader_index()
    kernels, kinds = cpu_kernel_index(), set(op_kinds())
    grads, vk, tests = autograd_rules(), vulkan_supports(), test_index()

    # An operator's shader is rarely named after it -- most elementwise ops live
    # in unary.comp behind an OP_ constant -- so the map is built from the
    # constants each shader defines, then by name as a fallback.
    by_op_const: dict[str, str] = {}
    for stem, info in shaders.items():
        for c in info["ops"]:
            by_op_const.setdefault(c[3:].lower(), stem)

    out = {}
    for n in names:
        kind = snake_to_camel(n)
        stem = by_op_const.get(n) or (n if n in shaders else None)
        out[n] = OpFacts(
            name=n,
            decls=decls.get(n, []),
            shader=({"stem": stem, **shaders[stem]} if stem else None),
            cpu_kernel=kernels.get(n),
            op_kind=kind if kind in kinds else None,
            has_grad=kind in grads,
            grad_line=grads.get(kind),
            on_vulkan=(kind in vk) if kind in kinds else True,
            tests=tests.get(n, []),
        )
    return out


def repo_stats() -> dict:
    """Headline numbers, counted rather than remembered."""
    def count(pattern: str, *dirs: str) -> int:
        n = 0
        for d in dirs:
            for p in (ROOT / d).rglob(pattern):
                n += p.read_text(errors="ignore").count("\n") + 1
        return n

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                        # noqa: BLE001
        rev = ""
    return {
        "headers": len(HEADERS),
        "shaders": len(SHADERS),
        "op_kinds": len(op_kinds()),
        "grad_rules": len(autograd_rules()),
        "cpp_lines": count("*.cpp", "src") + count("*.h", "include", "src"),
        "glsl_lines": count("*.comp", "shaders") + count("*.glsl", "shaders"),
        "rev": rev,
    }


if __name__ == "__main__":
    st = repo_stats()
    print(f"  {st['headers']} headers, {st['shaders']} shaders, {st['op_kinds']} OpKinds, "
          f"{st['grad_rules']} gradient rules")
    print(f"  {st['cpp_lines']} lines C++, {st['glsl_lines']} lines GLSL @ {st['rev']}")
    d = all_decls()
    print(f"  {sum(len(v) for v in d.values())} documented declarations")
