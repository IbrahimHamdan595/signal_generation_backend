import os
import logging

logger = logging.getLogger(__name__)

BUCKET = "ml-checkpoints"

# Files that must survive a server restart
CHECKPOINT_FILES = [
    "best_model.pt",
    "model_config.json",
    "scaler_params.json",
    "eval_report.json",
    "versions.json",
]


def _client():
    from app.core.config import settings
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_SERVICE_KEY
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        logger.warning(f"⚠️  Supabase client init failed: {e}")
        return None


def upload(filename: str, local_path: str) -> bool:
    client = _client()
    if not client:
        return False
    if not os.path.exists(local_path):
        logger.warning(f"⚠️  upload skipped — {local_path} not found locally")
        return False
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        client.storage.from_(BUCKET).upload(
            path=filename, file=data, file_options={"upsert": "true"}
        )
        logger.info(f"☁️  Uploaded {filename} → Supabase Storage")
        return True
    except Exception as e:
        logger.error(f"❌ Storage upload failed for {filename}: {e}")
        return False


def download(filename: str, local_path: str) -> bool:
    client = _client()
    if not client:
        return False
    try:
        data = client.storage.from_(BUCKET).download(filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        logger.info(f"☁️  Downloaded {filename} ← Supabase Storage")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Storage download failed for {filename}: {e}")
        return False


def sync_from_cloud():
    """Download any checkpoint files missing locally — called at startup."""
    logger.info("☁️  Syncing checkpoints from Supabase Storage...")
    os.makedirs("checkpoints", exist_ok=True)
    for filename in CHECKPOINT_FILES:
        local_path = os.path.join("checkpoints", filename)
        if not os.path.exists(local_path):
            download(filename, local_path)


def upload_all():
    """Upload all checkpoint files to Supabase Storage — called after training."""
    for filename in CHECKPOINT_FILES:
        local_path = os.path.join("checkpoints", filename)
        upload(filename, local_path)
