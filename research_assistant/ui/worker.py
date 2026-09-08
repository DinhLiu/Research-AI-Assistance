"""Isolated pipeline worker. Never persist provider credentials in run artifacts."""
from __future__ import annotations

from contextlib import contextmanager

import json
import logging
import os
import sys
import time
from pathlib import Path

from research_assistant.ui.settings import ENV_DEFAULTS, validate, stage_environment

logger = logging.getLogger("research_assistant.pipeline")


class RedactingFormatter(logging.Formatter):
    """Remove configured credentials from messages and tracebacks."""

    def __init__(self, fmt, secrets):
        super().__init__(fmt)
        self.secrets = tuple(value for value in secrets if value)

    def _redact(self, text):
        for value in self.secrets:
            text = text.replace(value, "[REDACTED]")
        return text

    def format(self, record):
        return self._redact(super().format(record))

    def formatException(self, exc_info):
        return self._redact(super().formatException(exc_info))


def configure_logging():
    """Send pipeline and dependency logs to the stream captured by the UI server."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        [os.environ.get(key, "") for key in ENV_DEFAULTS if key.endswith("API_KEY")],
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


@contextmanager
def stage_profile(stage):
    # The UI runs stages sequentially in a dedicated worker process.
    original = dict(os.environ)
    effective = stage_environment(stage, original)
    try:
        os.environ.update(effective)
        yield
    finally:
        for key in effective:
            if key not in original:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original[key]


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
    files = ["run.log"] if (directory / "run.log").is_file() else []

    def emit(status="running", error=None):
        update(dict(status=status, topic=topic, stages=stages[:], summaries=summaries[:],
                    warnings=warnings[:], files=files[:], error=error))

    try:
        logger.info("Pipeline started | topic=%r", topic)
        result = None
        for index, (stage, function) in enumerate(zip(configs, (retrieve, extract, synthesize, write))):
            stage_started = time.perf_counter()
            stages[index] = "running"
            emit()
            logger.info("Stage %s/4 started | name=%s", index + 1, stage)
            with stage_profile(stage):
                result = function(topic if index == 0 else result, configs[stage])
            name = f"stage-{index + 1}-{stage}.json"
            (directory / name).write_text(result.model_dump_json(indent=2), encoding="utf-8")
            files.append(name)
            if index == 0:
                summaries[index] = f"{len(result.papers)} papers"
                if not result.papers:
                    raise ValueError("No papers were found. Try another topic or broaden the filters.")
            elif index == 1:
                usable = sum(r.status != "skipped" for r in result.records)
                summaries[index] = f"{usable}/{len(result.records)} papers extracted"
                if usable < len(result.records):
                    warnings.append("Some papers were skipped in stage 2; see the JSON output for details.")
                if not usable:
                    raise ValueError("No papers could be extracted for synthesis.")
            elif index == 2:
                summaries[index] = f"{len(result.assignments)} clusters · {result.execution.summary_status}"
                if not result.paper_manifest:
                    raise ValueError("Stage 3 found no eligible papers for the review.")
                if configs[stage].summarize and result.execution.summary_status != "complete":
                    warnings.append(f"Synthesis: {result.execution.summary_status}")
            else:
                summaries[index] = result.generation_status
                (directory / "review.md").write_text(render(result), encoding="utf-8")
                files.append("review.md")
                if result.generation_status != "complete":
                    warnings.append(f"Review: {result.generation_status} ({result.execution.stop_reason or 'see results'}).")
                if result.generation_status == "failed":
                    raise ValueError("Stage 4 could not create a valid review.")
            stages[index] = "complete"
            logger.info("Stage %s/4 completed | name=%s | seconds=%.2f | summary=%s",
                        index + 1, stage, time.perf_counter() - stage_started, summaries[index])
            emit()
        logger.info("Pipeline completed | status=%s", "completed_with_warnings" if warnings else "completed")
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
        logger.exception("Pipeline failed | stage=%s | error=%s",
                         list(configs)[index] if "index" in locals() else "initialization", message)
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
    configure_logging()
    logger.info("Worker initialized | pid=%s | run_directory=%s", os.getpid(), directory)
    _, configs = validate(payload["env"], payload["configs"])
    run_pipeline(payload["topic"], configs, directory,
                 lambda state: atomic_json(directory / "status.json", state))


if __name__ == "__main__":
    main()
