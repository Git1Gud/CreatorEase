from pathlib import Path
from typing import Callable, Optional, TypeVar


T = TypeVar("T")


def resolve_device(raw_device: Optional[str]) -> Optional[str]:
    if not raw_device:
        return None
    value = raw_device.strip().lower()
    if value in {"", "auto", "none"}:
        return None
    return value


def compute_type_for_device(device: Optional[str]) -> str:
    if device and device.startswith("cuda"):
        return "float16"
    return "int8"


def parse_numeric(value: Optional[str], caster: Callable[[str], T], default: T) -> T:
    if value is None:
        return default
    try:
        return caster(value)
    except (TypeError, ValueError):
        return default


def parse_boolean(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def save_upload(storage, dest_folder: Path, prefix: str) -> Path:
    dest_folder.mkdir(parents=True, exist_ok=True)
    filename = storage.filename or f"{prefix}.bin"
    safe_name = f"{prefix}_{Path(filename).name}"
    path = dest_folder / safe_name
    storage.save(str(path))
    return path
