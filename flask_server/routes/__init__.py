from .process_video import process_video_bp
from .subtitles import subtitles_bp
from .multicam import multicam_bp
from .sync import sync_bp
from .transcipt import transcript_bp
from .health import health_bp

ALL_BLUEPRINTS = (
    process_video_bp,
    subtitles_bp,
    multicam_bp,
    sync_bp,
    health_bp,
    transcript_bp,
)
