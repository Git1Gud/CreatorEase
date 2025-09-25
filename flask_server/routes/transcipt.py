from flask import Blueprint, jsonify, request

from config import settings
from utils.api_helpers import compute_type_for_device, resolve_device
from utils.transcription_utils import transcribe_audio_with_whisperx

from pathlib import Path

transcript_bp = Blueprint("transcript", __name__)

@transcript_bp.route("/transcript", methods=["POST"])
def transcript():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file_storage = request.files["file"]
    if not file_storage or file_storage.filename == "":
        return jsonify({"error": "No file selected"}), 400

    model_size = request.form.get("model_size", settings.defaults.model_size)
    device = resolve_device(request.form.get("device"))
    compute_type = compute_type_for_device(device)

    video_path = settings.upload_folder / Path(file_storage.filename).name
    file_storage.save(str(video_path))


    try:
        words = transcribe_audio_with_whisperx(
            str(video_path),
            model_name=model_size,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    return jsonify({"message": "Transcript generated successfully", "output": words})

