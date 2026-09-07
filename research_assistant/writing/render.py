"""Deterministic citations and scoped caveats; raw text cannot inject Markdown."""
import re


def escape(text):
    text = re.sub(r"[\r\n]+", " ", str(text))
    return re.sub(r"([\\`*_{}\[\]<>#|!])", r"\\\1", text)


def render_markdown(result):
    vi = result.plan.language == "vi"
    title = "Tổng quan tài liệu" if vi else "Literature review"
    lines = [f"# {title}: {escape(result.topic or '')}", "",
             f"{'Trạng thái' if vi else 'Status'}: {result.generation_status}.", ""]
    lines += [
        ("Bản tổng hợp chỉ mô tả snapshot đã trích xuất. Citation được kiểm tra cấu trúc; nội dung chưa được xác minh ngữ nghĩa. Các nhóm phương pháp dùng ngưỡng phân cụm tạm thời."
         if vi else "This review describes the extracted snapshot only. Citations are structurally checked; semantic support is not verified. Method groups use a provisional clustering threshold."), "",
        ("Phần template giữ nguyên văn evidence gốc, có thể khác ngôn ngữ được yêu cầu."
         if vi else "Template statements preserve the original evidence language."), ""]
    if result.plan.evidence_warnings:
        lines += [(f"Cảnh báo đầu vào: đã loại {len(result.plan.evidence_warnings)} evidence hướng dẫn định dạng tài liệu; xem JSON provenance."
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
    lines.extend(["## " + ("Phạm vi và điểm cần kiểm chứng" if vi else "Scope and verification needs"), "",
                  ("Không suy ra 'chưa có nghiên cứu' từ trường dữ liệu còn thiếu. Dataset/metric là mentions, không chứng minh quan hệ đánh giá hay khả năng so sánh kết quả. Các hạn chế cần được kiểm chứng trước khi kết luận research gap."
                   if vi else "Missing fields do not establish absence of research. Dataset/metric mentions do not establish evaluation relationships or comparable results. Documented limitations require verification before being framed as research gaps."), "",
                  f"Evidence coverage: {result.coverage.get('supported_papers', 0)}/{result.coverage.get('total_papers', 0)} papers; "
                  f"LLM prose coverage: {result.coverage.get('llm_claims', 0)}/{len(result.claims)} claims.", ""])
    if result.coverage.get("unsupported_paper_keys"):
        lines += ["Papers without selected support: " + ", ".join(escape(k) for k in result.coverage["unsupported_paper_keys"]), ""]
    lines.extend(["## " + ("Kết luận" if vi else "Conclusion"), "",
                  (f"Bản tổng hợp bao phủ evidence được chọn cho {result.coverage.get('supported_papers', 0)} tài liệu trong snapshot. Các nhóm phương pháp và hạn chế ở trên là cơ sở để đọc kiểm chứng; chưa xác lập tính đầy đủ của tổng quan hoặc tính mới của một hướng nghiên cứu."
                   if vi else f"This review covers selected evidence for {result.coverage.get('supported_papers', 0)} papers in the snapshot. The method groups and limitations above provide a basis for verification; they do not establish review completeness or the novelty of a research direction."), ""])
    if bibliography:
        lines.extend(["## " + ("Tài liệu tham khảo" if vi else "References"), ""])
        for key in sorted(bibliography):
            item = bibliography[key]
            lines.extend([f'<a id="ref-{numbers[key]}"></a>',
                          f"{numbers[key]}. {escape(item['title'])}. [{escape(key)}]({item['url']}).", ""])
    return "\n".join(lines).rstrip() + "\n"
