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
    glsl: dict | None = None          # the `///` block on the GLSL function
    cpu_doc: dict | None = None       # the `///` block on the CPU kernel


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

    glsl_fns = all_shader_functions()

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
            # A GLSL helper is named `<op>_op` by convention; fall back to the
            # bare name for the ones that are not element-wise.
            glsl=glsl_fns.get(f"{n}_op") or glsl_fns.get(n),
            cpu_doc=cpu_kernel_doc(n),
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


# ------------------------------------------------- GLSL function doc blocks --

def shader_functions(path: Path) -> dict[str, dict]:
    """`///` blocks attached to GLSL functions.

    The shaders carry the same house style as the headers, and for the
    element-wise family they carry the BEST prose in the repository: why tanh
    clamps its input, why gelu goes through erfc, why relu is written as
    `x <= 0 ? 0 : x` and not `x > 0 ? x : 0`. That reasoning was written by
    whoever hit the bug it prevents, and it belongs in the reference.
    """
    out: dict[str, dict] = {}
    lines = path.read_text().split("\n")
    doc: list[str] = []
    for i, raw in enumerate(lines):
        t = raw.strip()
        if t.startswith("///"):
            doc.append(t[3:].lstrip())
            continue
        m = re.match(r"^(\w[\w ]*?)\s+(\w+)\s*\(([^)]*)\)\s*\{", t)
        if m and doc:
            out[m.group(2)] = {
                "returns": m.group(1),
                "args": m.group(3),
                "doc": "\n".join(doc).strip(),
                "line": i + 1,
                "path": rel(path),
                "url": src_link(path, i + 1),
            }
        if not t.startswith("//"):
            doc = []
    return out


def all_shader_functions() -> dict[str, dict]:
    out = {}
    for s in SHADERS + sorted((ROOT / "shaders").glob("*.glsl")):
        for name, info in shader_functions(s).items():
            out.setdefault(name, info)
    return out


def cpu_kernel_doc(name: str) -> dict | None:
    """The comment block above a `k_<name>` CPU kernel, if it has one."""
    for f in CPU_KERNELS:
        lines = f.read_text().split("\n")
        doc: list[str] = []
        for i, raw in enumerate(lines):
            t = raw.strip()
            if t.startswith("///"):
                doc.append(t[3:].lstrip())
                continue
            if re.search(rf"\bk_{re.escape(name)}\s*\(", t) and doc:
                return {"doc": "\n".join(doc).strip(), "line": i + 1,
                        "path": rel(f), "url": src_link(f, i + 1)}
            if not t.startswith("//"):
                doc = []
    return None


# ------------------------------------------------------- shader internals ---

# vkML binds NO descriptor sets. All 24 shaders take their buffers as
# `uint64_t` device addresses inside the push-constant block, which is why
# bufferDeviceAddress is a REQUIRED feature (vk_device.cpp) and why the
# 128-byte push budget is tight: every operand costs 8 bytes before any
# metadata. Verified by counting -- 0/24 shaders declare a binding.
GLSL_SCALAR_BYTES = {
    "float": 4, "int": 4, "uint": 4, "bool": 4,
    "uint64_t": 8, "int64_t": 8, "double": 8,
    "vec2": 8, "vec4": 16, "uvec2": 8, "uvec4": 16, "ivec4": 16,
    "Operand": 32,          # 4 extents + 4 strides, uint each
}


def push_block(path: Path) -> dict | None:
    """The push-constant struct: fields, types, comments and total bytes.

    The byte total is computed from the declared types under `scalar` layout,
    which is what every shader here requests. It is an ESTIMATE and is labelled
    as one on the page -- the authority is the C++ struct the host packs, and
    where the two disagree the C++ side wins.
    """
    text = path.read_text()
    m = re.search(r"layout\(push_constant[^)]*\)\s*uniform\s+\w+\s*\{(.*?)\}\s*(\w+)\s*;",
                  text, re.S)
    if not m:
        return None
    fields, total = [], 0
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        fm = re.match(r"([\w:]+)\s+(\w+)\s*(\[\s*(\d+)\s*\])?\s*;\s*(?://\s*(.*))?", line)
        if not fm:
            continue
        ctype, fname, count = fm.group(1), fm.group(2), int(fm.group(4) or 1)
        size = GLSL_SCALAR_BYTES.get(ctype, 4) * count
        total += size
        fields.append({"type": ctype, "name": fname, "bytes": size,
                       "comment": (fm.group(5) or "").strip()})
    return {"name": m.group(2), "fields": fields, "bytes": total}


def shader_detail(path: Path) -> dict:
    """Everything structural about one shader."""
    text = path.read_text()
    shared = re.findall(r"^\s*shared\s+(\w+)\s+(\w+)\s*\[([^\]]+)\]", text, re.M)
    return {
        "path": rel(path),
        "url": src_link(path),
        "lines": text.count("\n") + 1,
        # The workgroup width is NOT a literal in any shader. common.glsl
        # declares `layout(local_size_x_id = 0) in;` once and every shader
        # inherits it, so the size is a SPECIALISATION CONSTANT resolved at
        # pipeline creation. That is the mechanism the host uses to clamp to
        # min(256, maxComputeWorkGroupInvocations) on a minimum-spec device
        # without recompiling GLSL (issue #21).
        "workgroup": ("specialisation constant 0"
                      if re.search(r"local_size_x_id\s*=\s*(\d+)", text)
                      or "common.glsl" in text
                      else (re.search(r"local_size_x\s*=\s*(\w+)", text) or [None, None])[1]),
        "spec_constants": re.findall(
            r"layout\(constant_id\s*=\s*(\d+)\)\s*const\s+(\w+)\s+(\w+)\s*=\s*([^;]+);", text),
        "push": push_block(path),
        "shared_mem": [{"type": t, "name": n, "extent": e.strip()} for t, n, e in shared],
        "barriers": len(re.findall(r"\bbarrier\s*\(", text)),
        "subgroup_ops": sorted(set(re.findall(r"\b(subgroup\w+)\s*\(", text))),
        "ops": sorted(set(re.findall(r"#define\s+(OP_\w+)\s", text))),
        "bindings": len(re.findall(r"layout\([^)]*binding\s*=", text)),
        "functions": shader_functions(path),
    }


def all_shader_details() -> dict[str, dict]:
    return {s.stem: shader_detail(s) for s in SHADERS}


# --------------------------------------------------- dispatch and pipelines --

def pipeline_map() -> dict[str, list[dict]]:
    """OpKind -> the pipelines its `case` arm can create.

    Walks VulkanBackend::compute's switch, tracking which `case OpKind::X:`
    labels are in scope, and records every `pipes.get("name", ...)` under them.
    A case with several entries genuinely dispatches several kernels -- matmul
    is the clearest, with gemv, three GEMM variants and a split-K reduction.

    Line-scanned rather than parsed. The switch is 1300 lines of a single
    function and a real parse would need a C++ front end for no additional
    truth: the two forms it has to recognise are `case OpKind::X:` and
    `pipes.get("literal"`, and both are unambiguous.
    """
    path = ROOT / "src" / "backend" / "vulkan" / "vulkan_backend.cpp"
    lines = path.read_text().split("\n")

    start = next(i for i, l in enumerate(lines) if "void VulkanBackend::compute(" in l)
    out: dict[str, list[dict]] = {}
    active: list[str] = []
    for i in range(start, len(lines)):
        line = lines[i]
        if line.startswith("void ") and i > start:
            break                                   # left compute()

        # `case OpKind::X:` -- several may stack before one body.
        for m in re.finditer(r"case OpKind::(\w+)\s*:", line):
            active.append(m.group(1))

        m = re.search(r'pipes\.get\(\s*"([\w.]+)"', line)
        if m and active:
            entry = {"pipeline": m.group(1), "line": i + 1,
                     "url": src_link(path, i + 1)}
            for kind in active:
                if entry["pipeline"] not in [e["pipeline"] for e in out.setdefault(kind, [])]:
                    out[kind].append(entry)

        # A `break;` at case depth ends the group of labels.
        if re.match(r"\s{16}break;", line) or re.match(r"\s{12}\}", line):
            active = []
    return out


def push_structs() -> dict[str, dict]:
    """The C++ push-constant structs, with their compile-time size assertions.

    The GLSL side is checked by scripts/check_push_constants.py; this is the
    other half. Each struct carries a static_assert against
    kGuaranteedPushConstantBytes, so the budget is enforced at COMPILE time on
    the host as well -- 14 assertions, one per block.
    """
    path = ROOT / "src" / "backend" / "vulkan" / "vulkan_backend.cpp"
    text = path.read_text()
    out = {}
    for m in re.finditer(r"struct\s+(\w*Push)\s*\{(.*?)\}\s*;", text, re.S):
        name, body = m.group(1), m.group(2)
        fields = re.findall(r"^\s*([\w:]+)\s+(\w+)\s*(?:\[(\d+)\])?\s*(?:=[^;]*)?;", body, re.M)
        out[name] = {
            "fields": [{"type": t, "name": n, "count": int(c or 1)} for t, n, c in fields],
            "asserted": f"sizeof({name}) <= kGuaranteedPushConstantBytes" in text,
            "line": text[:m.start()].count("\n") + 1,
        }
    return out


# ------------------------------------------------------------- class surface --

@dataclass
class Member:
    """One member of a class: a method, an operator, or a data field."""
    name: str
    signature: str
    doc: str
    kind: str                 # method | field | ctor | operator | static
    access: str               # public | protected | private
    line: int


@dataclass
class ClassDoc:
    name: str
    kind: str                 # class | struct
    bases: str
    doc: str
    file: Path
    line: int
    members: list[Member] = field(default_factory=list)

    @property
    def path(self) -> str:
        return rel(self.file)

    @property
    def url(self) -> str:
        return src_link(self.file, self.line)

    def public(self) -> list[Member]:
        return [m for m in self.members if m.access == "public"]


# Lines that open a class body. `final`, a base list and an attribute may all
# appear, so the base clause is captured loosely and reported verbatim.
CLASS_OPEN = re.compile(r"^(class|struct)\s+(\w+)\s*(?:final\s*)?(?::([^{]*))?\{?\s*$")


def parse_classes(path: Path) -> list[ClassDoc]:
    """Classes in a header, with their public members.

    Brace-counted rather than parsed. The headers here declare one class per
    block with no nested types, so counting braces from the opening line is
    exact for this codebase -- and where it would not be, the member simply
    lands in the wrong class rather than being invented.
    """
    lines = path.read_text().split("\n")
    out: list[ClassDoc] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = CLASS_OPEN.match(stripped)
        if not m or stripped.endswith(";"):        # a forward declaration
            i += 1
            continue

        # The `///` block immediately above.
        doc: list[str] = []
        j = i - 1
        while j >= 0 and lines[j].strip().startswith("///"):
            doc.insert(0, lines[j].strip()[3:].lstrip())
            j -= 1

        cls = ClassDoc(name=m.group(2), kind=m.group(1),
                       bases=(m.group(3) or "").strip(), doc="\n".join(doc).strip(),
                       file=path, line=i + 1)

        # A struct is public by default, a class private.
        access = "public" if m.group(1) == "struct" else "private"
        depth, k, mdoc = 0, i, []
        while k < len(lines):
            line = lines[k]
            depth += line.count("{") - line.count("}")
            if k > i and depth <= 0:
                break
            k += 1
            body = line.strip()

            if body.startswith("///"):
                mdoc.append(body[3:].lstrip())
                continue
            acc = re.match(r"(public|protected|private)\s*:", body)
            if acc:
                access, mdoc = acc.group(1), []
                continue
            if not body or body.startswith(("//", "#", "}")) or body == "{":
                if not body:
                    mdoc = []
                continue
            if k == i + 1:                          # the class line itself
                continue

            # A member declaration: ends in ; or { and is not a statement.
            if not (body.endswith((";", "{", ")")) or "(" in body):
                mdoc = []
                continue

            # JOIN A WRAPPED DECLARATION before classifying it. A signature
            # spanning lines otherwise leaks its continuation as a phantom
            # member -- `Tensor full(span, DType dtype = ..., Device device =
            # ...)` produced members called `dtype` and `device`, which exist
            # nowhere.
            joined, look = body, k
            # `}` terminates too: an inline body -- `bool defined() const
            # noexcept { return node_ != nullptr; }` -- ends with a brace, and
            # without this the joiner ran on and SWALLOWED the next
            # declaration. That is how shape() disappeared from Tensor.
            while (not joined.rstrip().endswith((";", "{", "}"))
                   and look < len(lines) and look - k < 8):
                joined += " " + lines[look].strip()
                look += 1
            if look > k:
                depth += sum(lines[x].count("{") - lines[x].count("}")
                             for x in range(k, look))
                k = look

            sig = re.sub(r"\s+", " ", joined).rstrip("{").strip()
            # `operator` is checked FIRST: `Tensor& operator=(const Tensor&)`
            # also satisfies the field pattern, and was being recorded as a
            # field named `operator`.
            om = re.search(r"\boperator\s*([^\s(]+)", sig)
            if om:
                cls.members.append(Member(name=f"operator{om.group(1)}", signature=sig,
                                          doc="\n".join(mdoc).strip(), kind="operator",
                                          access=access, line=k))
                mdoc = []
                continue
            # `~` is allowed before the name so a DESTRUCTOR is matched.
            # Without it `~Tensor();` was silently dropped -- the tilde is not
            # a word character and was not in the preceding-character set.
            fm = re.search(r"(?:^|[\s*&])(~?\w+)\s*\(", body)
            if fm:
                name = fm.group(1)
                if name in {"if", "for", "while", "switch", "return", "sizeof"}:
                    mdoc = []
                    continue
                if name == cls.name or name == f"~{cls.name}":
                    kind = "ctor"
                elif name.startswith("operator"):
                    kind = "operator"
                elif body.startswith("static") or " static " in body:
                    kind = "static"
                else:
                    kind = "method"
            else:
                dm = re.match(r"(?:mutable\s+)?[\w:<>,\s*&]+?\b(\w+)\s*(?:=[^;]*)?;$", body)
                if not dm:
                    mdoc = []
                    continue
                name, kind = dm.group(1), "field"

            cls.members.append(Member(name=name, signature=sig, doc="\n".join(mdoc).strip(),
                                      kind=kind, access=access, line=k))
            mdoc = []
        out.append(cls)
        i = k
    return out


def all_classes() -> dict[str, ClassDoc]:
    out = {}
    for h in HEADERS:
        for c in parse_classes(h):
            out.setdefault(c.name, c)
    return out


PY_PACKAGE = ROOT / "python" / "vkml"
BINDINGS = ROOT / "bindings" / "module.cpp"


def python_classes() -> dict[str, ClassDoc]:
    """Classes defined in the Python package, via ast.

    A real parse here rather than a line scan, because Python ships one and the
    Python surface is where the nn modules, optimisers and data pipeline live --
    the parts a user meets first. There is no reason to approximate what the
    standard library will do exactly.
    """
    import ast

    out: dict[str, ClassDoc] = {}
    for f in sorted(PY_PACKAGE.glob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            cls = ClassDoc(name=node.name, kind="class", bases=bases,
                           doc=(ast.get_docstring(node) or "").strip(),
                           file=f, line=node.lineno)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # A leading underscore marks internals; __init__ and the
                    # dunder protocol methods are part of the surface.
                    private = item.name.startswith("_") and not item.name.startswith("__")
                    sig = f"{item.name}{ast.unparse(item.args)}"
                    returns = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                    cls.members.append(Member(
                        name=item.name,
                        signature=f"def {sig}{returns}",
                        doc=(ast.get_docstring(item) or "").strip(),
                        kind="ctor" if item.name == "__init__" else "method",
                        access="private" if private else "public",
                        line=item.lineno))
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    cls.members.append(Member(
                        name=item.target.id,
                        signature=f"{item.target.id}: {ast.unparse(item.annotation)}",
                        doc="", kind="field", access="public", line=item.lineno))
            out[node.name] = cls
    return out


# The forms nanobind uses to name a member, and to register a class. The quoted
# string is the PYTHON name, which is the only reliable key -- it is frequently
# not the C++ one. `\s*` spans newlines because the binding TU wraps long
# `.def(` calls with the name on the following line, and anchoring to `.def(`
# rather than to a bare quoted string is what keeps `"shape"_a` parameter
# annotations and string literals in unrelated expressions out of the results.
NB_BIND = re.compile(
    r"\.def(?:_prop_ro|_prop_rw|_static|_ro|_rw)?\(\s*\"(?P<name>\w+)\""
    r"|nb::class_<[^>]+>\s*\w*\s*\(\s*\w+\s*,\s*\"(?P<cls>\w+)\"")


def _nb_binding_lines() -> dict[str, int]:
    """Python member name -> the line in the binding TU that defines it.

    Found by scanning the binding source, because that is the ONLY place the
    Python name is decided and it routinely differs from the C++ one --
    `.def_prop_ro("size", &Tensor::numel)` being the case that motivated this.
    Linking a Python member to its C++ declaration would therefore link to the
    wrong function; linking to the binding shows the reader the rename itself.
    """
    text = BINDINGS.read_text()
    out: dict[str, int] = {}
    for m in NB_BIND.finditer(text):
        name = m.group("name") or m.group("cls")
        out.setdefault(name, text.count("\n", 0, m.start()) + 1)
    return out


def nanobind_class(name: str, extra: tuple[str, ...] = ()) -> ClassDoc | None:
    """A class defined in C++ through nanobind, introspected from the built module.

    WHY THIS IS NOT python_classes(). That function parses `python/vkml/*.py`
    with ast, so it can only see classes written in Python. `Tensor` is defined
    in C++ and injected by nanobind, so it is invisible to a source parse at
    every level -- which is why the site had 25 Python class pages and no page
    for the type all 99 operators return.

    The built extension is therefore the source of truth here, and it is a good
    one: nanobind embeds the real signature in the first line of each member's
    docstring, so signatures still cannot go stale. What it does NOT carry is a
    source location, so lines come from the binding TU by name.
    """
    import importlib
    try:
        mod = importlib.import_module("vkml")
        cls = getattr(mod, name)
    except (ImportError, AttributeError):
        return None

    lines = _nb_binding_lines()
    doc = ClassDoc(name=name, kind="class", bases="",
                   doc=(cls.__doc__ or "").strip(),
                   file=BINDINGS, line=lines.get(name, 1))

    for attr in sorted(set(dir(cls)) - {"__init__"} | set(extra)):
        if attr.startswith("_") and attr not in extra:
            continue
        obj = getattr(cls, attr, None)
        if obj is None:
            continue
        # nanobind puts the signature on the first line and any docstring after
        # it -- properties as "(self) -> T", methods as "name(self, ...) -> T".
        # A property given an explicit docstring in the binding gets that text
        # ALONE, with no signature line at all, so the leading "(" is what
        # distinguishes the two cases rather than the member's kind.
        head, _, rest = (obj.__doc__ or "").strip().partition("\n")
        is_prop = isinstance(obj, property)
        if is_prop:
            sig, body = (f"{attr}{head}", rest) if head.startswith("(") \
                else (attr, f"{head}\n{rest}")
        else:
            sig, body = head, rest
        doc.members.append(Member(
            name=attr, signature=sig or attr, doc=body.strip(),
            kind="field" if is_prop else "method",
            access="public", line=lines.get(attr, doc.line)))
    return doc


# ------------------------------------------------------- environment switches --

ENV_CALL = re.compile(
    r"env_(?P<how>flag|int|value)\(\s*\"(?P<name>VKML_\w+)\"(?:\s*,\s*(?P<default>[^)]+?))?\s*\)")


def env_switches() -> dict[str, dict]:
    """Every VKML_* variable the C++ reads, found at its call site.

    Generated rather than listed, so a new switch appears in the documentation
    the moment it is added. The default is captured verbatim from the call
    because it is often an expression rather than a literal, and paraphrasing
    it would be the drift this is guarding against.

    Only src/ is scanned: these are the switches the LIBRARY reads. Build
    options and test-harness variables are a separate surface and are collected
    by cmake_options() and listed by hand where they are used.
    """
    out: dict[str, dict] = {}
    for path in sorted((ROOT / "src").rglob("*.cpp")) + sorted((ROOT / "src").rglob("*.h")):
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            for m in ENV_CALL.finditer(line):
                name = m.group("name")
                entry = out.setdefault(name, {
                    "name": name, "kind": m.group("how"),
                    "default": (m.group("default") or "").strip(),
                    "sites": [],
                })
                entry["sites"].append({"path": rel(path), "line": i + 1,
                                       "url": src_link(path, i + 1)})
    return out


def cmake_options() -> list[dict]:
    """Project options from the top-level CMakeLists, with their real defaults."""
    text = (ROOT / "CMakeLists.txt").read_text()
    out = []
    for m in re.finditer(r'option\(\s*(VKML_\w+)\s+"([^"]*)"\s+(\w+)\s*\)', text):
        out.append({"name": m.group(1), "help": m.group(2), "default": m.group(3)})
    return out


# ------------------------------------------------------- torch equivalences --

TESTS_DIR = ROOT / "tests" / "python"


def torch_equivalents() -> dict[str, list[str]]:
    """vkML name -> the torch functions its tests are checked against.

    NOT written down anywhere by hand, and deliberately not extracted from
    prose. The test suite states the correspondence and then PROVES it: every
    pair below is an assertion that vkml's output matches torch's within a
    declared tolerance. Prose mentions torch fourteen times; the suite pairs 85.

    Parsed with `ast` rather than imported, because PyTorch is a TEST-ONLY
    dependency -- the documentation says so on its own pages -- and importing
    the test modules to build the site would quietly make it a build dependency
    too.

    Reads any tuple literal that contains both a `V.<name>` and a `torch.<name>`
    attribute, which is the shape every table in the suite uses:

        ("add", V.add, torch.add, "any")
        ("less", V.less, torch.lt)
    """
    import ast

    out: dict[str, set[str]] = {}
    for path in sorted(TESTS_DIR.glob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List)):
                continue
            ours: list[str] = []
            theirs: list[str] = []
            for el in node.elts:
                if not isinstance(el, ast.Attribute):
                    continue
                root, parts = el, []
                while isinstance(root, ast.Attribute):
                    parts.append(root.attr)
                    root = root.value
                if not isinstance(root, ast.Name):
                    continue
                dotted = ".".join(reversed(parts))
                if root.id == "V":
                    ours.append(dotted)
                elif root.id == "torch":
                    theirs.append(f"torch.{dotted}")
            # One of each: a pair. More than one of either is a table row that
            # happens to mention several, and guessing which maps to which would
            # invent a correspondence the tests do not make.
            if len(ours) == 1 and len(theirs) == 1:
                out.setdefault(ours[0], set()).add(theirs[0])
    return {k: sorted(v) for k, v in sorted(out.items())}
