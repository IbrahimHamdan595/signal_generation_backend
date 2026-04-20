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

    Supports batch inference: sends up to BATCH_SIZE texts in a single API call,
    reducing total calls by ~16× compared to one-at-a-time scoring.
    """

    BATCH_SIZE = 16  # HF router supports up to 32; 16 is safe

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

    # ── Single-text API call (used by classify()) ─────────────────────────────

    async def _call_api(self, text: str, retries: int = 5) -> Dict:
        payload = {"inputs": text}

        for attempt in range(retries):
            timeout = 60 if attempt == 0 else 90
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        self.url, headers=self.headers, json=payload
                    )

                    if response.status_code == 503:
                        wait = 20 * (attempt + 1)
                        logger.warning(f"⏳ FinBERT model loading, retrying in {wait}s... (attempt {attempt+1}/{retries})")
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code == 429:
                        logger.warning("⚠️  Rate limited, waiting 30s...")
                        await asyncio.sleep(30)
                        continue

                    if response.status_code == 401:
                        logger.error("❌ Invalid HF_API_TOKEN — check your .env")
                        return self._neutral()

                    if response.status_code != 200:
                        logger.error(f"FinBERT error {response.status_code}: {response.text}")
                        return self._neutral()

                    return self._parse(response.json())

            except httpx.ConnectError as e:
                wait = 5 * (attempt + 1)
                logger.error(f"FinBERT connection error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
            except httpx.TimeoutException:
                wait = 15 * (attempt + 1)
                logger.warning(f"⏳ FinBERT timeout (attempt {attempt+1}/{retries}) — retrying in {wait}s")
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"FinBERT unexpected error: {e}")
                return self._neutral()

        logger.error("FinBERT: all retries exhausted — returning neutral")
        return self._neutral()

    # ── Batch API call (main path for bulk scoring) ───────────────────────────

    async def _call_api_batch(self, texts: List[str], retries: int = 5) -> List[Dict]:
        """
        Send a list of texts in a single HF inference API request.
        HF returns a list of per-text results (same order as input).
        Falls back to neutral list on persistent failure.
        """
        payload = {"inputs": texts}

        for attempt in range(retries):
            # Batch requests need more time (multiple texts, bigger response)
            timeout = 90 if attempt == 0 else 120
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        self.url, headers=self.headers, json=payload
                    )

                    if response.status_code == 503:
                        wait = 20 * (attempt + 1)
                        logger.warning(f"⏳ FinBERT loading, retrying in {wait}s... (attempt {attempt+1}/{retries})")
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code == 429:
                        logger.warning("⚠️  Rate limited, waiting 30s...")
                        await asyncio.sleep(30)
                        continue

                    if response.status_code == 401:
                        logger.error("❌ Invalid HF_API_TOKEN")
                        return [self._neutral()] * len(texts)

                    if response.status_code != 200:
                        logger.error(f"FinBERT batch error {response.status_code}: {response.text[:300]}")
                        return [self._neutral()] * len(texts)

                    return self._parse_batch(response.json(), len(texts))

            except httpx.ConnectError as e:
                wait = 5 * (attempt + 1)
                logger.error(f"FinBERT batch connect error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
            except httpx.TimeoutException:
                wait = 15 * (attempt + 1)
                logger.warning(f"⏳ FinBERT batch timeout (attempt {attempt+1}/{retries}), retrying in {wait}s")
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"FinBERT batch unexpected error: {e}")
                return [self._neutral()] * len(texts)

        logger.error("FinBERT batch: all retries exhausted")
        return [self._neutral()] * len(texts)

    # ── Public batch entry point ──────────────────────────────────────────────

    async def classify_batch(self, texts: List[str]) -> List[Dict]:
        """
        Score a list of texts using batch HF API calls (BATCH_SIZE texts per request).

        For 1500 articles this makes ~94 API calls instead of 1500 — ~16× faster.
        Empty strings are short-circuited to neutral without hitting the API.
        """
        if not texts:
            return []

        cleaned = [t.strip()[:512] if t else "" for t in texts]
        results: List[Dict] = []

        for batch_start in range(0, len(cleaned), self.BATCH_SIZE):
            batch = cleaned[batch_start : batch_start + self.BATCH_SIZE]

            # Separate non-empty texts; keep track of original positions
            non_empty_idx   = [i for i, t in enumerate(batch) if t]
            non_empty_texts = [batch[i] for i in non_empty_idx]

            if not non_empty_texts:
                results.extend([self._neutral()] * len(batch))
                continue

            batch_sentiments = await self._call_api_batch(non_empty_texts)

            # Re-insert neutral placeholders for empty slots
            batch_output = [self._neutral()] * len(batch)
            for list_idx, original_idx in enumerate(non_empty_idx):
                if list_idx < len(batch_sentiments):
                    batch_output[original_idx] = batch_sentiments[list_idx]

            results.extend(batch_output)

            # Small delay between batch requests to stay inside rate limits
            if batch_start + self.BATCH_SIZE < len(cleaned):
                await asyncio.sleep(0.5)

        return results

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_batch(self, raw, expected_count: int) -> List[Dict]:
        """
        Parse HF batch response.

        Batch format — list of per-text results:
            [
              [{"label": "positive", "score": 0.97}, ...],   ← text 0
              [{"label": "negative", "score": 0.85}, ...],   ← text 1
              ...
            ]
        """
        try:
            if not isinstance(raw, list):
                logger.error(f"Unexpected FinBERT batch response type: {type(raw)}")
                return [self._neutral()] * expected_count

            results = [self._parse(item) for item in raw]

            # Pad with neutral if HF returned fewer results than we sent
            while len(results) < expected_count:
                results.append(self._neutral())

            return results[:expected_count]

        except Exception as e:
            logger.error(f"FinBERT batch parse error: {e} | raw snippet: {str(raw)[:200]}")
            return [self._neutral()] * expected_count

    def _parse(self, raw) -> Dict:
        """
        Handles all HuggingFace response formats:

        Format 1 — router API (new):
            [{"label": "positive", "score": 0.97}, ...]

        Format 2 — inference API (old, nested):
            [[{"label": "positive", "score": 0.97}, ...]]

        Format 3 — single dict (top-1 only):
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

            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)

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
