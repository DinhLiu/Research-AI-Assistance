"""Deterministic citations and scoped caveats; raw text cannot inject Markdown."""
import re
from urllib.parse import quote

from research_assistant.writing.locales import VI


def escape(text):
    text = re.sub(r"[\r\n]+", " ", str(text))
    return re.sub(r"([\\`*_{}\[\]<>#|!])", r"\\\1", text)


def render_markdown(result):
    vi = result.plan.language == "vi"
    title = VI["title"] if vi else "Literature review"
    lines = [f"# {title}: {escape(result.topic or '')}", "",
             f"{VI['status'] if vi else 'Status'}: {result.generation_status}.", ""]
    lines += [
        (VI["scope_caveat"]
         if vi else "This review describes the extracted snapshot only. Citations are structurally checked; semantic support is not verified. Method groups use a provisional clustering threshold."), "",
        (VI["template_caveat"]
         if vi else "Template statements preserve the original evidence language."), ""]
    if result.plan.evidence_warnings:
        lines += [(VI["input_warning"].format(count=len(result.plan.evidence_warnings))
                   if vi else f"Input warning: excluded {len(result.plan.evidence_warnings)} document-formatting evidence units; see JSON provenance."), ""]
    bibliography = {item["paper_key"]: item for item in result.bibliography}
    numbers = {key: i + 1 for i, key in enumerate(sorted(bibliography))}
    claims = {c.claim_id: c for c in result.claims}
    for section in result.plan.sections:
        members = [claims[cid] for cid in section.claim_ids if cid in claims]
        if not members:
            continue
        lines.extend([f"## {escape(section.title)}", ""])
        for claim in members:
            refs = " ".join(f"[{numbers[key]}](#ref-{numbers[key]})" for key in claim.subject_paper_keys if key in numbers)
            prefix = "[Evidence] " if claim.generation == "template" else ""
            lines.extend([f"{prefix}{escape(claim.text)} {refs}", ""])
    lines.extend(["## " + (VI["verification_title"] if vi else "Scope and verification needs"), "",
                  (VI["verification_caveat"]
                   if vi else "Missing fields do not establish absence of research. Dataset/metric mentions do not establish evaluation relationships or comparable results. Documented limitations require verification before being framed as research gaps."), "",
                  f"Evidence coverage: {result.coverage.get('supported_papers', 0)}/{result.coverage.get('total_papers', 0)} papers; "
                  f"LLM prose coverage: {result.coverage.get('llm_claims', 0)}/{len(result.claims)} claims.", ""])
    if result.coverage.get("unsupported_paper_keys"):
        lines += ["Papers without selected support: " + ", ".join(escape(k) for k in result.coverage["unsupported_paper_keys"]), ""]
    lines.extend(["## " + (VI["conclusion_title"] if vi else "Conclusion"), "",
                  (VI["conclusion"].format(count=result.coverage.get("supported_papers", 0))
                   if vi else f"This review covers selected evidence for {result.coverage.get('supported_papers', 0)} papers in the snapshot. The method groups and limitations above provide a basis for verification; they do not establish review completeness or the novelty of a research direction."), ""])
    if bibliography:
        lines.extend(["## " + (VI["references"] if vi else "References"), ""])
        for key in sorted(bibliography):
            item = bibliography[key]
            lines.extend([f'<a id="ref-{numbers[key]}"></a>',
                          f"{numbers[key]}. {escape(item['title'])}. [{escape(key)}]({item['url']}).", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_bundle(result, *, index_name="review.md", evidence_chunk_size=25):
    """Render a linked multi-file review; filenames and links are code-owned."""
    if evidence_chunk_size <= 0:
        raise ValueError("evidence_chunk_size must be positive")
    if index_name in {"overview.md", "evidence.md", "references.md"} or re.fullmatch(
        r"(?:section|evidence)-\d+\.md", index_name
    ):
        raise ValueError("index_name conflicts with a generated bundle file")
    vi = result.plan.language == "vi"
    title = VI["title"] if vi else "Literature review"
    bibliography = {item["paper_key"]: item for item in result.bibliography}
    numbers = {key: index + 1 for index, key in enumerate(sorted(bibliography))}
    claims = {claim.claim_id: claim for claim in result.claims}
    sections = [
        (section, [claims[cid] for cid in section.claim_ids if cid in claims])
        for section in result.plan.sections
    ]
    sections = [(section, members) for section, members in sections if members]
    section_files = {section.section_id: f"section-{index:02d}.md"
                     for index, (section, _members) in enumerate(sections, 1)}
    index_link = quote(index_name)

    evidence = list(result.evidence)
    evidence_files: dict[str, str] = {}
    evidence_chunks = [evidence[i:i + evidence_chunk_size]
                       for i in range(0, len(evidence), evidence_chunk_size)]
    if len(evidence_chunks) <= 1:
        for unit in evidence:
            evidence_files[unit["unit_id"]] = "evidence.md"
    else:
        for index, chunk in enumerate(evidence_chunks, 1):
            for unit in chunk:
                evidence_files[unit["unit_id"]] = f"evidence-{index:02d}.md"

    files = {}
    nav = _nav(index_link, vi)
    index_lines = [f"# {title}: {escape(result.topic or '')}", "",
                   f"{VI['status'] if vi else 'Status'}: {result.generation_status}.", "",
                   "## " + ("Nội dung" if vi else "Contents"), "",
                   f"- [{'Tổng quan' if vi else 'Overview'}](overview.md)"]
    index_lines.extend(f"- [{escape(section.title)}]({section_files[section.section_id]})"
                       for section, _members in sections)
    index_lines.extend([
        f"- [{'Evidence' if vi else 'Evidence'}](evidence.md)",
        f"- [{VI['references'] if vi else 'References'}](references.md)",
    ])
    files[index_name] = _document(index_lines)

    overview_lines = [f"# {'Tổng quan' if vi else 'Overview'}", "", nav, "",
                      VI["scope_caveat"] if vi else
                      "This review describes the extracted snapshot only. Citations are structurally checked; semantic support is not verified. Method groups use a provisional clustering threshold.",
                      "", VI["template_caveat"] if vi else
                      "Template statements preserve the original evidence language.", ""]
    if result.plan.evidence_warnings:
        overview_lines.extend([
            VI["input_warning"].format(count=len(result.plan.evidence_warnings)) if vi else
            f"Input warning: excluded {len(result.plan.evidence_warnings)} document-formatting evidence units; see JSON provenance.",
            "",
        ])
    overview_lines.extend([
        f"## {VI['verification_title'] if vi else 'Scope and verification needs'}", "",
        VI["verification_caveat"] if vi else
        "Missing fields do not establish absence of research. Dataset/metric mentions do not establish evaluation relationships or comparable results. Documented limitations require verification before being framed as research gaps.",
        "",
        f"Evidence coverage: {result.coverage.get('supported_papers', 0)}/{result.coverage.get('total_papers', 0)} papers; "
        f"LLM prose coverage: {result.coverage.get('llm_claims', 0)}/{len(result.claims)} claims.", "",
        f"## {VI['conclusion_title'] if vi else 'Conclusion'}", "",
        VI["conclusion"].format(count=result.coverage.get("supported_papers", 0)) if vi else
        f"This review covers selected evidence for {result.coverage.get('supported_papers', 0)} papers in the snapshot. The method groups and limitations provide a basis for verification; they do not establish review completeness or novelty.",
    ])
    files["overview.md"] = _document(overview_lines)

    for section, members in sections:
        section_lines = [f"# {escape(section.title)}", "", nav, ""]
        for claim in members:
            citations = " ".join(
                f"[{numbers[key]}](references.md#ref-{numbers[key]})"
                for key in claim.subject_paper_keys if key in numbers
            )
            evidence_links = ", ".join(
                f"[{escape(ref)}]({evidence_files.get(ref, 'evidence.md')}#evidence-{ref})"
                for ref in claim.support_refs
            )
            prefix = "[Evidence] " if claim.generation == "template" else ""
            section_lines.extend([
                f'<a id="claim-{claim.claim_id}"></a>',
                f"{prefix}{escape(claim.text)} {citations}", "",
                f"{'Evidence' if not vi else 'Bằng chứng'}: {evidence_links}", "",
            ])
        files[section_files[section.section_id]] = _document(section_lines)

    used_by = {}
    for section, members in sections:
        for claim in members:
            for ref in claim.support_refs:
                used_by.setdefault(ref, []).append((section, claim))

    if len(evidence_chunks) <= 1:
        files["evidence.md"] = _render_evidence_page(
            evidence, nav, numbers, used_by, section_files, vi, "Evidence"
        )
    else:
        evidence_index = ["# Evidence", "", nav, ""]
        for index, chunk in enumerate(evidence_chunks, 1):
            filename = f"evidence-{index:02d}.md"
            evidence_index.append(f"- [Evidence {index}: {len(chunk)} units]({filename})")
            files[filename] = _render_evidence_page(
                chunk, nav, numbers, used_by, section_files, vi, f"Evidence {index}"
            )
        files["evidence.md"] = _document(evidence_index)

    reference_lines = [f"# {VI['references'] if vi else 'References'}", "", nav, ""]
    for key in sorted(bibliography):
        item = bibliography[key]
        reference_lines.extend([
            f'<a id="ref-{numbers[key]}"></a>',
            f"{numbers[key]}. {escape(item['title'])}. [{escape(key)}]({item['url']}).", "",
        ])
    files["references.md"] = _document(reference_lines)
    return files


def _render_evidence_page(units, nav, numbers, used_by, section_files, vi, title):
    lines = [f"# {title}", "", nav, ""]
    for unit in units:
        ref = unit["unit_id"]
        paper = unit["paper_key"]
        citation = (f"[{numbers[paper]}](references.md#ref-{numbers[paper]})"
                    if paper in numbers else escape(paper))
        backlinks = ", ".join(
            f"[{escape(section.title)}]({section_files[section.section_id]}#claim-{claim.claim_id})"
            for section, claim in used_by.get(ref, [])
        )
        lines.extend([
            f'<a id="evidence-{ref}"></a>',
            f"## {escape(ref)}", "",
            f"{'Tài liệu' if vi else 'Paper'}: {citation} · "
            f"{'Loại' if vi else 'Kind'}: {escape(unit['kind'])} · "
            f"{'Trường' if vi else 'Field'}: {escape(unit['field_path'])}", "",
            escape(unit.get("quote") or unit.get("text") or ""), "",
            f"{'Được dùng tại' if vi else 'Used by'}: {backlinks or '—'}", "",
        ])
    return _document(lines)


def _nav(index_link, vi):
    labels = ("Mục lục", "Tổng quan", "Evidence", "Tài liệu tham khảo") if vi else (
        "Index", "Overview", "Evidence", "References"
    )
    return " · ".join((
        f"[{labels[0]}]({index_link})",
        f"[{labels[1]}](overview.md)",
        f"[{labels[2]}](evidence.md)",
        f"[{labels[3]}](references.md)",
    ))


def _document(lines):
    return "\n".join(lines).rstrip() + "\n"
