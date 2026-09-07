"""Isolated pipeline worker. Never persist provider credentials in run artifacts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from research_assistant.ui.settings import ENV_DEFAULTS, validate


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run_pipeline(topic, configs, directory, update, services=None):
    if services is None:
        from research_assistant.retrieval.pipeline import RetrievalPipeline
        from research_assistant.extraction.pipeline import extract_papers
        from research_assistant.synthesis.pipeline import synthesize
        from research_assistant.writing.pipeline import write_review
        from research_assistant.writing.render import render_markdown
        services = (lambda topic, cfg: RetrievalPipeline(cfg).search(topic),
                    extract_papers, synthesize, write_review, render_markdown)
    retrieve, extract, synthesize, write, render = services
    stages = ["pending"] * 4
    summaries = [""] * 4
    warnings = []
    files = []

    def emit(status="running", error=None):
        update(dict(status=status, topic=topic, stages=stages[:], summaries=summaries[:],
                    warnings=warnings[:], files=files[:], error=error))

    try:
        result = None
        for index, (stage, function) in enumerate(zip(configs, (retrieve, extract, synthesize, write))):
            stages[index] = "running"
            emit()
            result = function(topic if index == 0 else result, configs[stage])
            name = f"stage-{index + 1}-{stage}.json"
            (directory / name).write_text(result.model_dump_json(indent=2), encoding="utf-8")
            files.append(name)
            if index == 0:
                summaries[index] = f"{len(result.papers)} bài báo"
                if not result.papers:
                    raise ValueError("Không tìm thấy bài báo. Hãy đổi chủ đề hoặc nới bộ lọc.")
            elif index == 1:
                usable = sum(r.status != "skipped" for r in result.records)
                summaries[index] = f"{usable}/{len(result.records)} bài được trích xuất"
                if usable < len(result.records):
                    warnings.append("Một số bài bị bỏ qua ở stage 2; xem JSON để biết lý do.")
                if not usable:
                    raise ValueError("Không trích xuất được bài nào để tổng hợp.")
            elif index == 2:
                summaries[index] = f"{len(result.assignments)} nhóm · {result.execution.summary_status}"
                if not result.paper_manifest:
                    raise ValueError("Stage 3 không có bài đủ điều kiện để viết review.")
                if configs[stage].summarize and result.execution.summary_status != "complete":
                    warnings.append(f"Tổng hợp: {result.execution.summary_status}")
            else:
                summaries[index] = result.generation_status
                (directory / "review.md").write_text(render(result), encoding="utf-8")
                files.append("review.md")
                if result.generation_status != "complete":
                    warnings.append(f"Bản review: {result.generation_status} ({result.execution.stop_reason or 'xem kết quả'}).")
                if result.generation_status == "failed":
                    raise ValueError("Stage 4 không tạo được bản review hợp lệ.")
            stages[index] = "complete"
            emit()
        emit("completed_with_warnings" if warnings else "completed")
    except Exception as exc:
        stages[index] = "failed"
        for i in range(index + 1, 4):
            stages[i] = "skipped"
        message = str(exc)
        for key in ENV_DEFAULTS:
            secret = os.environ.get(key)
            if key.endswith("API_KEY") and secret:
                message = message.replace(secret, "[hidden]")
        emit("failed", message[:2000])


def main():
    directory = Path(sys.argv[1])
    # Importing config loads .env. Apply the immutable run snapshot AFTER imports.
    payload = json.load(sys.stdin)
    for key in ENV_DEFAULTS:
        value = payload["env"].get(key, "")
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    _, configs = validate(payload["env"], payload["configs"])
    run_pipeline(payload["topic"], configs, directory,
                 lambda state: atomic_json(directory / "status.json", state))


if __name__ == "__main__":
    main()
