"""Download arXiv e-prints with polite throttling and safe tar extraction."""

from __future__ import annotations

import gzip
import logging
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from research_assistant.arxiv_ids import format_version, normalize_arxiv_id, split_arxiv_id
from research_assistant.config import ExtractionConfig
from research_assistant.llm.client import USER_AGENT

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".tex", ".bbl", ".bib", ".sty", ".cls", ".clo", ".txt"}
FetchedKind = Literal["tex", "pdf"]

_fetch_lock = threading.Lock()
_last_fetch_mono = 0.0


class FetchError(RuntimeError):
    pass


@dataclass
class FetchedSource:
    arxiv_id: str
    version: str
    kind: FetchedKind
    path: Path


def paper_cache_dir(config: ExtractionConfig, arxiv_id: str, version: str) -> Path:
    return Path(config.cache_dir) / normalize_arxiv_id(arxiv_id) / format_version(version)


def source_dir(config: ExtractionConfig, arxiv_id: str, version: str) -> Path:
    return paper_cache_dir(config, arxiv_id, version) / "source"


def parsed_path(
    config: ExtractionConfig, arxiv_id: str, version: str, source_kind: str = "tex"
) -> Path:
    return paper_cache_dir(config, arxiv_id, version) / f"parsed_{config.parsed_version}_{source_kind}.json"


def candidates_path(
    config: ExtractionConfig, arxiv_id: str, version: str, source_kind: str = "tex"
) -> Path:
    ver = getattr(config, "candidate_version", "1")
    return paper_cache_dir(config, arxiv_id, version) / f"candidates_{ver}_{source_kind}.json"


def extraction_dir(config: ExtractionConfig, arxiv_id: str, version: str) -> Path:
    return paper_cache_dir(config, arxiv_id, version) / "extraction"


def _throttle(delay_s: float) -> None:
    global _last_fetch_mono
    with _fetch_lock:
        wait = delay_s - (time.monotonic() - _last_fetch_mono)
        if wait > 0:
            time.sleep(wait)
        _last_fetch_mono = time.monotonic()


def sniff_bytes(data: bytes) -> str:
    head = data[:8]
    if data.startswith(b"%PDF"):
        return "pdf"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "tar"
    lowered = data[:256].lower()
    if lowered.lstrip().startswith(b"<!doctype") or b"<html" in lowered:
        return "html"
    if data.lstrip().startswith(b"\\"):
        return "tex"
    return "unknown"


def fetch_eprint(
    arxiv_id: str,
    version: str | None,
    config: ExtractionConfig,
    *,
    client: httpx.Client | None = None,
) -> FetchedSource:
    """Fetch /e-print then fall back to /pdf. Reuses on-disk source files."""
    core = normalize_arxiv_id(arxiv_id)
    ver = format_version(version)
    dest = source_dir(config, core, ver)
    dest.mkdir(parents=True, exist_ok=True)

    tar_path = dest / "source.tar.gz"
    pdf_path = dest / "paper.pdf"
    tex_path = dest / "main.tex"
    if tar_path.is_file() and tar_path.stat().st_size > 0:
        return FetchedSource(core, ver, "tex", tar_path)
    if tex_path.is_file() and tex_path.stat().st_size > 0:
        return FetchedSource(core, ver, "tex", tex_path)
    if pdf_path.is_file() and pdf_path.stat().st_size > 0:
        return FetchedSource(core, ver, "pdf", pdf_path)

    id_for_url = f"{core}{ver}"
    own_client = client is None
    http = client or httpx.Client(
        timeout=config.fetch_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        data, status = _get_bytes(http, f"https://arxiv.org/e-print/{id_for_url}", config)
        if status == 404 or sniff_bytes(data) == "html":
            data, status = _get_bytes(http, f"https://arxiv.org/pdf/{id_for_url}.pdf", config)
            if status >= 400 or sniff_bytes(data) in {"html", "unknown"}:
                raise FetchError(f"Failed to fetch arXiv source for {id_for_url} (HTTP {status})")
            pdf_path.write_bytes(data)
            return FetchedSource(core, ver, "pdf", pdf_path)
        return _store_payload(data, dest, core, ver)
    finally:
        if own_client:
            http.close()


def fetch_pdf(
    arxiv_id: str,
    version: str | None,
    config: ExtractionConfig,
    *,
    client: httpx.Client | None = None,
) -> FetchedSource:
    core = normalize_arxiv_id(arxiv_id)
    ver = format_version(version)
    dest = source_dir(config, core, ver)
    dest.mkdir(parents=True, exist_ok=True)
    pdf_path = dest / "paper.pdf"
    if pdf_path.is_file() and pdf_path.stat().st_size > 0:
        return FetchedSource(core, ver, "pdf", pdf_path)

    id_for_url = f"{core}{ver}"
    own_client = client is None
    http = client or httpx.Client(
        timeout=config.fetch_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        data, status = _get_bytes(http, f"https://arxiv.org/pdf/{id_for_url}.pdf", config)
        if status >= 400 or sniff_bytes(data) != "pdf":
            raise FetchError(f"Failed to fetch PDF for {id_for_url} (HTTP {status})")
        pdf_path.write_bytes(data)
        return FetchedSource(core, ver, "pdf", pdf_path)
    finally:
        if own_client:
            http.close()


def _get_bytes(client: httpx.Client, url: str, config: ExtractionConfig) -> tuple[bytes, int]:
    _throttle(config.arxiv_delay_s)
    logger.info("GET %s", url)
    response = client.get(url)
    return response.content or b"", response.status_code


def _store_payload(data: bytes, dest: Path, arxiv_id: str, version: str) -> FetchedSource:
    kind = sniff_bytes(data)
    if kind == "gzip":
        inner = gzip.decompress(data)
        inner_kind = sniff_bytes(inner)
        if inner_kind == "pdf":
            path = dest / "paper.pdf"
            path.write_bytes(inner)
            return FetchedSource(arxiv_id, version, "pdf", path)
        if inner_kind in {"tar", "unknown"}:
            # Unknown gzip is usually a tar archive from arXiv.
            path = dest / "source.tar.gz"
            path.write_bytes(data)
            return FetchedSource(arxiv_id, version, "tex", path)
        if inner_kind == "tex":
            path = dest / "main.tex"
            path.write_text(inner.decode("utf-8", errors="replace"), encoding="utf-8")
            return FetchedSource(arxiv_id, version, "tex", path)
        path = dest / "source.tar.gz"
        path.write_bytes(data)
        return FetchedSource(arxiv_id, version, "tex", path)
    if kind == "pdf":
        path = dest / "paper.pdf"
        path.write_bytes(data)
        return FetchedSource(arxiv_id, version, "pdf", path)
    if kind == "tar":
        path = dest / "source.tar.gz"
        path.write_bytes(data)
        return FetchedSource(arxiv_id, version, "tex", path)
    if kind == "tex":
        path = dest / "main.tex"
        path.write_text(data.decode("utf-8", errors="replace"), encoding="utf-8")
        return FetchedSource(arxiv_id, version, "tex", path)
    raise FetchError(f"Unrecognized e-print payload for {arxiv_id} ({kind})")


def safe_extract_tar(
    archive: Path,
    dest: Path,
    *,
    max_files: int = 400,
    max_bytes: int = 80_000_000,
    max_nesting: int = 8,
) -> list[Path]:
    """Extract regular text-like files only. Rejects path traversal and special files."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    used = 0
    opener: str | io.BytesIO | Path = archive
    try:
        tf = tarfile.open(archive, "r:*")
    except tarfile.ReadError:
        # Single gzipped file saved with a .tar.gz suffix.
        raise
    with tf:
        for member in tf:
            if len(written) >= max_files:
                logger.warning("Tar file cap reached (%s); stopping extract", max_files)
                break
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                continue
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            path_obj = Path(name)
            if path_obj.is_absolute() or ".." in path_obj.parts:
                logger.warning("Skipping unsafe tar member %s", name)
                continue
            if len(path_obj.parts) > max_nesting:
                continue
            suffix = path_obj.suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            target = (dest / path_obj).resolve()
            if not _is_relative_to(target, dest):
                logger.warning("Skipping tar member outside dest: %s", name)
                continue
            size = member.size or 0
            if used + size > max_bytes:
                logger.warning("Tar byte cap reached; skipping remaining members")
                break
            handle = tf.extractfile(member)
            if handle is None:
                continue
            payload = handle.read()
            used += len(payload)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            written.append(target)
    return written


def unpack_tex_source(source: FetchedSource, config: ExtractionConfig) -> Path:
    """Return a directory containing extracted .tex files."""
    dest = source.path.parent / "unpacked"
    if dest.exists() and any(dest.rglob("*.tex")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if source.path.suffix.lower() == ".tex":
        src_dir = source.path.parent
        for path in src_dir.iterdir():
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                (dest / path.name).write_bytes(path.read_bytes())
        return dest
    try:
        safe_extract_tar(
            source.path,
            dest,
            max_files=config.max_tar_files,
            max_bytes=config.max_tar_bytes,
            max_nesting=config.max_tar_nesting,
        )
    except tarfile.ReadError:
        # Gzip of a single .tex, stored as source.tar.gz.
        raw = source.path.read_bytes()
        if sniff_bytes(raw) == "gzip":
            inner = gzip.decompress(raw)
            (dest / "main.tex").write_bytes(inner)
        else:
            raise
    return dest


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def parse_id_and_version(raw: str, fallback_version: str | None = None) -> tuple[str, str]:
    core, version = split_arxiv_id(raw)
    return core, format_version(version or fallback_version)
