import asyncio
import logging
import os
from typing import Dict, List

import torch
from transformers import pipeline

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logger = logging.getLogger(__name__)

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

_pipe = None


def _load_pipe():
    global _pipe
    if _pipe is None:
        logger.info(f"⏳ Loading FinBERT locally on {DEVICE_NAME}...")
        _pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            device=DEVICE,
        )
        logger.info("✅ FinBERT loaded")
    return _pipe


LABEL_MAP = {
    "positive": "positive", "negative": "negative", "neutral": "neutral",
    "POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral",
}


class FinBERTService:
    """
    Local FinBERT inference on GPU (RTX 3050) via HuggingFace transformers pipeline.
    No API calls, no rate limits, no token required.
    """

    BATCH_SIZE = 32

    async def classify(self, text: str) -> Dict:
        text = text.strip()[:512]
        if not text:
            return self._neutral()
        results = await self.classify_batch([text])
        return results[0]

    async def classify_batch(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []

        cleaned = [t.strip()[:512] if t else "" for t in texts]
        non_empty_idx = [i for i, t in enumerate(cleaned) if t]
        non_empty_texts = [cleaned[i] for i in non_empty_idx]

        results = [self._neutral()] * len(cleaned)

        if not non_empty_texts:
            return results

        loop = asyncio.get_event_loop()
        try:
            pipe = await loop.run_in_executor(None, _load_pipe)
            raw_outputs = await loop.run_in_executor(
                None,
                lambda: pipe(non_empty_texts, batch_size=self.BATCH_SIZE),
            )
            for list_idx, original_idx in enumerate(non_empty_idx):
                results[original_idx] = self._parse(raw_outputs[list_idx])
        except Exception as e:
            logger.error(f"FinBERT local inference error: {e}")

        return results

    def _parse(self, raw) -> Dict:
        try:
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict):
                items = [raw]
            else:
                return self._neutral()

            scores = {}
            for item in items:
                label = LABEL_MAP.get(item.get("label", ""), "neutral")
                scores[label] = float(item.get("score", 0.0))

            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)
            label = max(scores, key=scores.get) if scores else "neutral"

            return {
                "sentiment_label": label,
                "positive_score":  round(pos, 6),
                "negative_score":  round(neg, 6),
                "neutral_score":   round(neu, 6),
                "compound_score":  round(pos - neg, 6),
            }
        except Exception as e:
            logger.error(f"FinBERT parse error: {e}")
            return self._neutral()

    def _neutral(self) -> Dict:
        return {
            "sentiment_label": "neutral",
            "positive_score":  0.0,
            "negative_score":  0.0,
            "neutral_score":   1.0,
            "compound_score":  0.0,
        }
