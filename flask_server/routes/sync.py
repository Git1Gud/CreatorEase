import os
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import settings
from utils.api_helpers import parse_numeric, save_upload
from utils.video_sync import VideoSyncError, sync_videos_with_reference_audio

sync_bp = Blueprint("sync", __name__)


@sync_bp.route("/sync_videos", methods=["POST"])
def sync_videos():
    required_fields = {"video1", "video2", "audio"}
    if not request.files:
        return (
            jsonify(
                {
                    "error": "No files found in request. Ensure you're sending multipart/form-data with fields: video1, video2, audio.",
                }
            ),
            400,
        )

    missing = [field for field in sorted(required_fields) if not request.files.get(field)]
    if missing:
        return (
            jsonify(
                {
                    "error": f"Missing or empty file fields: {', '.join(missing)}",
                    "received_fields": list(request.files.keys()),
                }
            ),
            400,
        )

    video1_file = request.files["video1"]
    video2_file = request.files["video2"]
    audio_file = request.files["audio"]

    sync_defaults = settings.sync_defaults
    target_fps = parse_numeric(request.form.get("target_fps"), float, sync_defaults.target_fps)
    corr_sr = parse_numeric(request.form.get("corr_sr"), int, sync_defaults.corr_sr)
    max_shift = parse_numeric(request.form.get("max_shift"), float, sync_defaults.max_shift)
    max_seconds = parse_numeric(request.form.get("max_seconds"), float, sync_defaults.max_seconds)

    if target_fps <= 0:
        return jsonify({"error": "target_fps must be positive"}), 400
    if corr_sr <= 0:
        return jsonify({"error": "corr_sr must be positive"}), 400

    job_id = uuid.uuid4().hex
    job_dir = settings.sync_folder / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video1_path = save_upload(video1_file, job_dir, "video1")
    video2_path = save_upload(video2_file, job_dir, "video2")
    audio_path = save_upload(audio_file, job_dir, "audio")

    stitched_output_path = job_dir / f"synced_{uuid.uuid4().hex}.mp4"

    try:
        result = sync_videos_with_reference_audio(
            str(video1_path),
            str(video2_path),
            str(audio_path),
            output_dir=str(job_dir),
            target_fps=target_fps,
            corr_sr=corr_sr,
            max_shift_s=max_shift,
            max_seconds=max_seconds,
            generate_stitched=True,
            stitched_filename=str(stitched_output_path),
        )
    except VideoSyncError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Sync failed: {exc}"}), 500

    response_payload = {
        "message": "Sync completed",
        "offsets": result.offsets,
        "target_fps": result.target_fps,
        "timeline_start": result.t0,
        "job_id": job_id,
        "aligned_videos": {
            "video1": result.aligned_video_paths[0],
            "video2": result.aligned_video_paths[1],
        },
        "aligned_audio": result.aligned_audio_path,
    }

    try:
        response_payload["aligned_videos_relative"] = {
            "video1": os.path.relpath(result.aligned_video_paths[0], start=os.getcwd()),
            "video2": os.path.relpath(result.aligned_video_paths[1], start=os.getcwd()),
        }
        response_payload["aligned_audio_relative"] = os.path.relpath(
            result.aligned_audio_path,
            start=os.getcwd(),
        )
    except ValueError:
        pass

    if result.output_path:
        response_payload["output"] = result.output_path
        try:
            response_payload["output_relative"] = os.path.relpath(result.output_path, start=os.getcwd())
        except ValueError:
            pass

    return jsonify(response_payload)
