import os
import logging

logger = logging.getLogger(__name__)

BUCKET = "ml_checkpoints"

# Files that must survive a server restart. The post-FX layout namespaces
# checkpoints under per-asset-class subfolders (`equities/`, `fx/`); legacy
# pre-FX deploys stored these flat at the bucket root. Sync logic handles
# both layouts so old cloud state still bootstraps onto a fresh disk.
CHECKPOINT_FILES = [
    "best_model.pt",
    "model_config.json",
    "scaler_params.json",
    "eval_report.json",
    "versions.json",
]

ASSET_CLASS_SUBFOLDERS = ("equities", "fx", "equities_1h", "fx_1h")


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
    """Download any checkpoint files missing locally — called at startup.

    Tries the per-asset-class layout (`equities/best_model.pt`, etc.) first;
    if a file is also missing under that path, falls back to the legacy flat
    layout (`best_model.pt` at the bucket root) and writes it into the
    equities subfolder locally. This preserves any pre-FX cloud state from
    older deploys.
    """
    logger.info("Syncing checkpoints from Supabase Storage...")
    os.makedirs("checkpoints", exist_ok=True)

    for ac in ASSET_CLASS_SUBFOLDERS:
        local_dir = os.path.join("checkpoints", ac)
        os.makedirs(local_dir, exist_ok=True)
        for filename in CHECKPOINT_FILES:
            local_path = os.path.join(local_dir, filename)
            if os.path.exists(local_path):
                continue
            # Try per-asset-class cloud path first.
            if download(f"{ac}/{filename}", local_path):
                continue
            # Legacy flat fallback only for equities — there was no FX before.
            if ac == "equities":
                download(filename, local_path)


def upload_all(asset_class: str = "equities"):
    """Upload all checkpoint files for a given asset class — called after training."""
    if asset_class not in ASSET_CLASS_SUBFOLDERS:
        raise ValueError(f"asset_class must be one of {ASSET_CLASS_SUBFOLDERS}")
    for filename in CHECKPOINT_FILES:
        local_path = os.path.join("checkpoints", asset_class, filename)
        upload(f"{asset_class}/{filename}", local_path)
