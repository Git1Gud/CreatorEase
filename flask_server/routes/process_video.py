import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import settings
from utils.segment_pipeline import process_and_save_video_with_segments

process_video_bp = Blueprint("process_video", __name__)


@process_video_bp.route("/process_video", methods=["POST"])
def process_video() -> tuple:
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file_storage = request.files["file"]
    if not file_storage or file_storage.filename == "":
        return jsonify({"error": "No file selected"}), 400

    destination: Path = settings.upload_folder / Path(file_storage.filename).name
    file_storage.save(str(destination))

    start_time = time.time()
    try:
        urls = process_and_save_video_with_segments(
            str(destination),
            str(settings.upload_folder),
            model_size="small",
            device="cuda",
            style="modern",
        )
    except Exception as exc:
        return jsonify({"error": f"Video processing failed: {exc}"}), 500
    processing_time = time.time() - start_time

    return (
        jsonify(
            {
                "message": "Video processed successfully",
                "processing_time": f"{processing_time:.2f} seconds",
                "urls": urls,
            }
        ),
        200,
    )
