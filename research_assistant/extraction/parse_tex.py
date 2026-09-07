"""Parse arXiv TeX sources into a PaperDocument."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from research_assistant.extraction.fetch import FetchedSource, unpack_tex_source
from research_assistant.extraction.normalize import normalize_text
from research_assistant.extraction.types import PaperDocument, Section
from research_assistant.config import ExtractionConfig

logger = logging.getLogger(__name__)

_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_SECTION = re.compile(
    r"\\((?:sub)*section)\*?\s*\{|\\chapter\*?\{",
)
_BEGIN_ABSTRACT = re.compile(r"\\begin\{abstract\}", re.IGNORECASE)
_END_ABSTRACT = re.compile(r"\\end\{abstract\}", re.IGNORECASE)
_BEGIN_DOC = re.compile(r"\\begin\{document\}")
_END_DOC = re.compile(r"\\end\{document\}")


class ParseError(RuntimeError):
    pass


def parse_tex_source(source: FetchedSource, config: ExtractionConfig) -> PaperDocument:
    root = unpack_tex_source(source, config)
    main = find_main_tex(root)
    seen: set[Path] = {main.resolve()}
    merged = inline_inputs(main.read_text(encoding="utf-8", errors="replace"), main.parent, seen)
    merged = strip_tex_comments(merged)
    title = braced_command("title", merged) or ""
    abstract, body = split_abstract(merged)
    body = _body_only(body)
    sections = split_sections(body)
    if abstract:
        sections = [Section(heading="Abstract", level=0, text=latex_to_text(abstract))] + [
            s for s in sections if normalize_text(s.heading).lower() != "abstract"
        ]
    if not sections:
        text = latex_to_text(body)
        if not text:
            raise ParseError(f"Empty TeX body for {source.arxiv_id}")
        sections = [Section(heading="Body", level=1, text=text)]
    return PaperDocument(
        arxiv_id=source.arxiv_id,
        version=source.version,
        source_kind="tex",
        title=latex_to_text(title),
        abstract=latex_to_text(abstract) if abstract else "",
        sections=sections,
    )


def find_main_tex(root: Path) -> Path:
    files = list(root.rglob("*.tex"))
    if not files:
        raise ParseError(f"No .tex files under {root}")
    scored: list[tuple[int, int, Path]] = []
    preferred = {"main", "paper", "ms", "manuscript", "arxiv"}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        score = 0
        if r"\documentclass" in text:
            score += 10
        if r"\begin{document}" in text:
            score += 10
        if path.stem.lower() in preferred:
            score += 5
        scored.append((score, -len(path.parts), path))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored[0][0] < 10:
        logger.warning("No obvious main TeX file in %s; using %s", root, scored[0][2])
    return scored[0][2]


def inline_inputs(tex: str, base_dir: Path, seen: set[Path], depth: int = 0, max_depth: int = 8) -> str:
    if depth >= max_depth:
        return tex

    def replacer(match: re.Match[str]) -> str:
        rel = match.group(1).strip()
        path = _resolve_input(base_dir, rel)
        if path is None:
            return ""
        resolved = path.resolve()
        if resolved in seen:
            return ""
        seen.add(resolved)
        content = path.read_text(encoding="utf-8", errors="replace")
        return inline_inputs(content, path.parent, seen, depth + 1, max_depth)

    return _INPUT.sub(replacer, tex)


def _resolve_input(base_dir: Path, rel: str) -> Path | None:
    rel = rel.strip().strip("'\"")
    candidate = Path(rel)
    if candidate.suffix.lower() not in {"", ".tex"}:
        return None
    options = []
    if candidate.suffix:
        options.append(base_dir / candidate)
    else:
        options.append(base_dir / f"{rel}.tex")
        options.append(base_dir / rel)
    for path in options:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if not _under(resolved, base_dir.resolve()):
            continue
        return resolved
    return None


def _under(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def strip_tex_comments(tex: str) -> str:
    lines: list[str] = []
    for line in tex.splitlines():
        out: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            out.append(ch)
            i += 1
        lines.append("".join(out).rstrip())
    return "\n".join(lines)


def braced_command(name: str, tex: str) -> str | None:
    match = re.search(rf"\\{name}\s*\{{", tex)
    if not match:
        return None
    return _read_balanced(tex, match.end() - 1)


def _read_balanced(tex: str, open_idx: int) -> str:
    depth = 0
    out: list[str] = []
    for i in range(open_idx, len(tex)):
        ch = tex[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        if depth >= 1:
            out.append(ch)
    return "".join(out)


def split_abstract(tex: str) -> tuple[str, str]:
    begin = _BEGIN_ABSTRACT.search(tex)
    if not begin:
        simple = braced_command("abstract", tex)
        return (simple or "", tex)
    end = _END_ABSTRACT.search(tex, begin.end())
    if not end:
        return "", tex
    abstract = tex[begin.end() : end.start()]
    body = tex[: begin.start()] + tex[end.end() :]
    return abstract, body


def _body_only(tex: str) -> str:
    start = _BEGIN_DOC.search(tex)
    if start:
        tex = tex[start.end() :]
    end = _END_DOC.search(tex)
    if end:
        tex = tex[: end.start()]
    return tex


def split_sections(tex: str) -> list[Section]:
    matches = list(_SECTION.finditer(tex))
    if not matches:
        text = latex_to_text(tex)
        return [Section(heading="Body", level=1, text=text)] if text else []

    preamble = tex[: matches[0].start()]
    sections: list[Section] = []
    preamble_text = latex_to_text(preamble)
    if preamble_text:
        sections.append(Section(heading="Preamble", level=1, text=preamble_text))

    for i, match in enumerate(matches):
        heading = _read_balanced(tex, match.end() - 1)
        start = match.end() - 1
        # Skip the heading braces.
        heading_end = start + 1 + len(heading) + 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
        body = tex[heading_end:end]
        level = 1
        token = match.group(0)
        if "subsubsection" in token:
            level = 3
        elif "subsection" in token:
            level = 2
        elif "chapter" in token:
            level = 1
        text = latex_to_text(body)
        if text:
            sections.append(Section(heading=latex_to_text(heading) or "Section", level=level, text=text))
    return sections


def latex_to_text(tex: str) -> str:
    tex = tex.strip()
    if not tex:
        return ""
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        converter = LatexNodes2Text(keep_comments=False, math_mode="text")
        converted = converter.latex_to_text(tex)
    except Exception:
        converted = _cheap_latex_strip(tex)
    return normalize_text(converted)


def _cheap_latex_strip(tex: str) -> str:
    text = re.sub(r"\\(?:cite|ref|label|includegraphics)\*?\{[^}]*\}", " ", tex)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return text
