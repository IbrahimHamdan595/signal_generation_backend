import httpx
import asyncio
import logging
from typing import Dict, List

from app.core.config import settings

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "positive": "positive", "negative": "negative", "neutral": "neutral",
    "LABEL_0":  "positive", "LABEL_1":  "negative", "LABEL_2": "neutral",
    "POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral",
}


class FinBERTService:
    """
    Calls HuggingFace Router API for ProsusAI/finbert sentiment analysis.
    URL: https://router.huggingface.co/pipeline/sentiment-analysis/ProsusAI/finbert
    """

    def __init__(self):
        self.url     = settings.HF_FINBERT_URL
        self.headers = {
            "Authorization": f"Bearer {settings.HF_API_TOKEN}",
            "Content-Type":  "application/json",
        }

    async def classify(self, text: str) -> Dict:
        text = text.strip()[:512]
        if not text:
            return self._neutral()
        return await self._call_api(text)

    async def _call_api(self, text: str, retries: int = 3) -> Dict:
        payload = {"inputs": text}

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                    )

                    if response.status_code == 503:
                        wait = 10 * (attempt + 1)
                        logger.warning(f"⏳ FinBERT loading, retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code == 429:
                        logger.warning("⚠️  Rate limited, waiting 15s...")
                        await asyncio.sleep(15)
                        continue

                    if response.status_code == 401:
                        logger.error("❌ Invalid HF_API_TOKEN — check your .env")
                        return self._neutral()

                    if response.status_code != 200:
                        logger.error(f"FinBERT error {response.status_code}: {response.text}")
                        return self._neutral()

                    return self._parse(response.json())

            except httpx.ConnectError as e:
                logger.error(f"FinBERT connection error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(3)
            except httpx.TimeoutException:
                logger.error(f"FinBERT timeout (attempt {attempt+1})")
                if attempt < retries - 1:
                    await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"FinBERT unexpected error: {e}")
                return self._neutral()

        logger.error("FinBERT: all retries exhausted")
        return self._neutral()

    async def classify_batch(self, texts: List[str]) -> List[Dict]:
        results = []
        for i, text in enumerate(texts):
            result = await self._call_api(text.strip()[:512] if text else "")
            results.append(result)
            if i < len(texts) - 1:
                await asyncio.sleep(0.5)
        return results

    def _parse(self, raw) -> Dict:
        """
        Handles all HuggingFace response formats:

        Format 1 — router API (new):
            [{"label": "positive", "score": 0.97}, ...]

        Format 2 — inference API (old, nested):
            [[{"label": "positive", "score": 0.97}, ...]]

        Format 3 — all scores in one list (top-1 only):
            {"label": "positive", "score": 0.97}
        """
        try:
            if isinstance(raw, list) and len(raw) > 0:
                inner = raw[0]
                if isinstance(inner, list):
                    items = inner
                elif isinstance(inner, dict) and "label" in inner:
                    items = raw
                else:
                    items = raw
            elif isinstance(raw, dict) and "label" in raw:
                items = [raw]
            else:
                logger.error(f"Unrecognised FinBERT response: {raw}")
                return self._neutral()

            scores = {}
            for item in items:
                label = LABEL_MAP.get(item.get("label", ""), "neutral")
                scores[label] = float(item.get("score", 0.0))

            pos   = scores.get("positive", 0.0)
            neg   = scores.get("negative", 0.0)
            neu   = scores.get("neutral",  0.0)

            if len(scores) == 1:
                label = list(scores.keys())[0]
            else:
                label = max(scores, key=scores.get)

            return {
                "sentiment_label": label,
                "positive_score":  round(pos, 6),
                "negative_score":  round(neg, 6),
                "neutral_score":   round(neu, 6),
                "compound_score":  round(pos - neg, 6),
            }

        except Exception as e:
            logger.error(f"FinBERT parse error: {e} | raw: {raw}")
            return self._neutral()

    def _neutral(self) -> Dict:
        return {
            "sentiment_label": "neutral",
            "positive_score":  0.0,
            "negative_score":  0.0,
            "neutral_score":   1.0,
            "compound_score":  0.0,
        }
