import logging
import tomllib
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger("tsfm_zoo")

MANIFEST = Path(__file__).with_name("checkpoints.toml")
_licences_shown: set[str] = set()


def entry(family: str, variant: str) -> dict:
    manifest = tomllib.loads(MANIFEST.read_text())
    try:
        return manifest[family][variant]
    except KeyError:
        variants = sorted(manifest.get(family, {}))
        raise KeyError(f"no checkpoint {family}/{variant}; variants: {variants}") from None


def is_cached(family: str, variant: str) -> bool:
    e = entry(family, variant)
    try:
        snapshot_download(e["repo"], revision=e["revision"], local_files_only=True)
        return True
    except Exception:
        return False


def fetch(family: str, variant: str) -> Path:
    """Local directory of the pinned checkpoint, downloading it into HF_HOME if needed."""
    e = entry(family, variant)
    if family not in _licences_shown:
        logger.warning("%s/%s licence: %s", family, variant, e["license"])
        _licences_shown.add(family)
    return Path(snapshot_download(e["repo"], revision=e["revision"]))
