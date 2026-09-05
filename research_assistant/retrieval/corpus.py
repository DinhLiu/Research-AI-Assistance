from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METADATA_COLUMNS = [
    "arxiv_id",
    "title",
    "abstract",
    "authors",
    "year",
    "matched_categories",
    "latest_version",
    "doi",
]


class ArtifactCorpus:
    """FAISS index + sharded metadata/embeddings from a SPECTER2 embedding run."""

    def __init__(self, artifacts_dir: Path, rebuild_index_if_missing: bool = True):
        self.root = Path(artifacts_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.dim = int(self.manifest["embedding_dim"])
        self.ntotal = int(self.manifest["processed_papers"])
        self.shards = list(self.manifest["shards"])
        self.index_path = self.root / "index" / "papers_flatip.faiss"

        if not self.index_path.exists():
            if not rebuild_index_if_missing:
                raise FileNotFoundError(self.index_path)
            logger.info("FAISS index missing; building from embedding shards")
            self._build_index()

        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != self.ntotal:
            raise RuntimeError(
                f"FAISS ntotal={self.index.ntotal} != manifest processed_papers={self.ntotal}"
            )
        if self.index.d != self.dim:
            raise RuntimeError(f"FAISS dim={self.index.d} != manifest dim={self.dim}")

        self._arxiv_to_row: dict[str, int] | None = None

    def _build_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(self.dim)
        for shard in self.shards:
            emb = np.load(self.root / shard["embedding_file"], mmap_mode="r", allow_pickle=False)
            vec = np.asarray(emb, dtype=np.float32)
            faiss.normalize_L2(vec)
            index.add(vec)
        if index.ntotal != self.ntotal:
            raise RuntimeError("Rebuilt index size does not match manifest")
        faiss.write_index(index, str(self.index_path))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        faiss.normalize_L2(q)
        k = min(k, self.ntotal)
        return self.index.search(q, k)

    def arxiv_id_to_row(self) -> dict[str, int]:
        if self._arxiv_to_row is None:
            mapping: dict[str, int] = {}
            for shard in self.shards:
                df = pd.read_parquet(
                    self.root / shard["metadata_file"],
                    columns=["arxiv_id"],
                )
                start = int(shard["row_start"])
                for offset, arxiv_id in enumerate(df["arxiv_id"].tolist()):
                    mapping[str(arxiv_id)] = start + offset
            if len(mapping) != self.ntotal:
                logger.warning(
                    "arxiv_id map has %s unique ids for %s rows (duplicates possible)",
                    f"{len(mapping):,}",
                    f"{self.ntotal:,}",
                )
            self._arxiv_to_row = mapping
            logger.info("Indexed %s arxiv_id → row_id mappings", f"{len(mapping):,}")
        return self._arxiv_to_row

    def lookup_metadata(self, row_ids: list[int]) -> list[dict]:
        wanted = [int(x) for x in row_ids]
        by_id: dict[int, dict] = {}
        for shard in self.shards:
            start, end = int(shard["row_start"]), int(shard["row_end"])
            local = [g for g in wanted if start <= g < end]
            if not local:
                continue
            path = self.root / shard["metadata_file"]
            df = pd.read_parquet(path, columns=METADATA_COLUMNS)
            for g in local:
                row = df.iloc[g - start]
                record = {k: _jsonable(row[k]) for k in METADATA_COLUMNS}
                record["row_id"] = g
                by_id[g] = record
        missing = [g for g in wanted if g not in by_id]
        if missing:
            raise KeyError(f"row_ids not in metadata shards: {missing[:10]}")
        return [by_id[g] for g in wanted]

    def lookup_embeddings(self, row_ids: list[int]) -> np.ndarray:
        wanted = [int(x) for x in row_ids]
        out = np.empty((len(wanted), self.dim), dtype=np.float32)
        pos = {rid: i for i, rid in enumerate(wanted)}
        seen: set[int] = set()
        for shard in self.shards:
            start, end = int(shard["row_start"]), int(shard["row_end"])
            local = [rid for rid in wanted if start <= rid < end]
            if not local:
                continue
            emb = np.load(
                self.root / shard["embedding_file"],
                mmap_mode="r",
                allow_pickle=False,
            )
            for rid in local:
                vec = np.asarray(emb[rid - start], dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                out[pos[rid]] = vec
                seen.add(rid)
        missing = [rid for rid in wanted if rid not in seen]
        if missing:
            raise KeyError(f"row_ids not in embedding shards: {missing[:10]}")
        return out


def _jsonable(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value
