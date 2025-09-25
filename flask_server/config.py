from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable


@dataclass(frozen=True)
class SyncDefaults:
    """Default correlation/synchronization settings."""

    target_fps: float = 30.0
    corr_sr: int = 8000
    max_shift: float = 10.0
    max_seconds: float = 60.0


@dataclass(frozen=True)
class DataPaths:
    """Canonical dataset and artifact paths."""

    qa: Path = Path("data") / "youtube_shorts_podcast_dataset_with_qa.csv"
    base: Path = Path("data") / "youtube_shorts_podcast_dataset.csv"
    model: Path = Path("models") / "random_forest_views_rating_model.pkl"
    feature_importance_image: Path = Path("uploads") / "images" / "feature_importance.png"


@dataclass(frozen=True)
class RouteDefaults:
    """Reusable defaults for API routes."""

    model_size: str = "medium"
    direction: str = "ltr"
    overlap: float = 0.3
    expected_speakers: int = 2


@dataclass
class Settings:
    """Application level configuration and constants."""

    upload_folder: Path = Path("uploads")
    multicam_folder: Path = Path("uploads") / "multicam"
    sync_folder: Path = Path("uploads") / "sync"
    caption_output_filename: str = "output_with_captions.mp4"
    sample_delivery_url: str = (
        "http://res.cloudinary.com/dxt0biqah/video/upload/v1758811700/videos/clvcuwbqvemtw6bbzwau.mp4"
    )
    defaults: RouteDefaults = field(default_factory=RouteDefaults)
    sync_defaults: SyncDefaults = field(default_factory=SyncDefaults)
    data_paths: DataPaths = field(default_factory=DataPaths)

    def iter_directories(self) -> Iterable[Path]:
        yield self.upload_folder
        yield self.multicam_folder
        yield self.sync_folder
        yield self.data_paths.feature_importance_image.parent


settings = Settings()


def ensure_directories() -> None:
    """Ensure folders referenced by the application exist."""

    for folder in settings.iter_directories():
        folder.mkdir(parents=True, exist_ok=True)


def flask_config() -> Dict[str, str]:
    """Return configuration values suitable for Flask's config mapping."""

    return {
        "UPLOAD_FOLDER": str(settings.upload_folder),
        "MULTICAM_FOLDER": str(settings.multicam_folder),
        "SYNC_FOLDER": str(settings.sync_folder),
    }
