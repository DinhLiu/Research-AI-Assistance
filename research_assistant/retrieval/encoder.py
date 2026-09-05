from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from research_assistant.config import BASE_MODEL, MAX_LENGTH, QUERY_ADAPTER

logger = logging.getLogger(__name__)


class Specter2QueryEncoder:
    """Short-text encoder: specter2_base + adhoc_query adapter.

    Paper vectors in the FAISS index were encoded with the proximity adapter.
    Short user queries must use this encoder, not proximity.
    """

    def __init__(self, device: str = "auto", use_fp16: bool = True):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        logger.info("Loading SPECTER2 query encoder on %s", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoAdapterModel.from_pretrained(BASE_MODEL)
        adapter_name = model.load_adapter(QUERY_ADAPTER, source="hf", load_as="adhoc_query")
        model.active_adapters = adapter_name
        model.to(self.device)
        model.eval()
        mismatched = [
            (name, str(p.device))
            for name, p in model.named_parameters()
            if p.device != self.device
        ]
        if mismatched:
            raise RuntimeError(f"Query encoder parameters not all on {self.device}: {mismatched[:5]}")
        active = str(model.active_adapters)
        if "adhoc_query" not in active:
            raise RuntimeError(f"Query adapter not active: {active}")
        logger.info("Query encoder ready | adapter=%s", active)
        self.model = model

    @torch.inference_mode()
    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_fp16,
        ):
            out = self.model(**inputs)
            q = out.last_hidden_state[:, 0, :]
        q = F.normalize(q.float(), p=2, dim=1)
        return q.cpu().numpy().astype(np.float32)
