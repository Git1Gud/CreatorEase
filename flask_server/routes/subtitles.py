import json
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import settings
from utils.api_helpers import compute_type_for_device, resolve_device
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.transcription_utils import transcribe_audio_with_whisperx

subtitles_bp = Blueprint("subtitles", __name__)


@subtitles_bp.route("/add_subtitles", methods=["POST"])
def add_subtitles():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file_storage = request.files["file"]
    if not file_storage or file_storage.filename == "":
        return jsonify({"error": "No file selected"}), 400

    style = request.form.get("style", "modern")
    model_size = request.form.get("model_size", settings.defaults.model_size)
    device = resolve_device(request.form.get("device"))
    compute_type = compute_type_for_device(device)

    video_path = settings.upload_folder / Path(file_storage.filename).name
    file_storage.save(str(video_path))

    words_json = request.form.get("words_json")
    if words_json:
        try:
            words = json.loads(words_json)
        except Exception as exc:
            return jsonify({"error": f"Invalid words_json: {exc}"}), 400
    else:
        try:
            words = transcribe_audio_with_whisperx(
                str(video_path),
                model_name=model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            return jsonify({"error": f"Transcription failed: {exc}"}), 500

    name = video_path.stem
    out_path = settings.upload_folder / f"{name}_with_captions{video_path.suffix}"

    try:
        add_dynamic_subtitles_to_video(str(video_path), words, str(out_path), style=style)
    except Exception as exc:
        return jsonify({"error": f"Captioning failed: {exc}"}), 500

    return jsonify({"message": "Subtitles added successfully", "output": str(out_path)})
