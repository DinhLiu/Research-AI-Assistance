from __future__ import annotations

import tarfile
from pathlib import Path

from research_assistant.extraction.fetch import safe_extract_tar, sniff_bytes


def test_sniff_pdf_and_gzip():
    assert sniff_bytes(b"%PDF-1.4 leftover") == "pdf"
    assert sniff_bytes(b"\x1f\x8bxxxx") == "gzip"
    assert sniff_bytes(b"\\documentclass{article}") == "tex"


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "src.tar"
    dest = tmp_path / "out"
    with tarfile.open(archive, "w") as tf:
        evil = tmp_path / "payload.tex"
        evil.write_text("\\section{Hack}", encoding="utf-8")
        tf.add(evil, arcname="../outside.tex")
        good = tmp_path / "ok.tex"
        good.write_text("\\section{Ok}", encoding="utf-8")
        tf.add(good, arcname="paper/ok.tex")
    written = safe_extract_tar(archive, dest, max_files=20, max_bytes=10_000, max_nesting=8)
    names = {path.name for path in written}
    assert "ok.tex" in names
    assert not (tmp_path / "outside.tex").exists()
    assert all(path.resolve().is_relative_to(dest.resolve()) for path in written)


def test_safe_extract_skips_symlink(tmp_path: Path):
    archive = tmp_path / "src.tar"
    dest = tmp_path / "out"
    target = tmp_path / "secret.tex"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.tex"
    try:
        link.symlink_to(target)
    except OSError:
        return
    with tarfile.open(archive, "w") as tf:
        tf.add(link, arcname="link.tex")
    written = safe_extract_tar(archive, dest)
    assert written == []
