import json
import uuid
from pathlib import Path
from typing import List, Dict

from flask import Blueprint, jsonify, request

from config import settings
from utils.api_helpers import (
    compute_type_for_device,
    parse_boolean,
    parse_numeric,
    resolve_device,
    save_upload,
)
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.multicam import combine_multicam_with_slide
from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.video_sync import VideoSyncError, sync_videos_with_reference_audio
from utils.cloudinary_utils import upload_video

multicam_bp = Blueprint("multicam", __name__)


def _adjust_word_timings(words: List[Dict], offset: float) -> List[Dict]:
    adjusted: List[Dict] = []
    for word in words:
        try:
            start = float(word.get("start", 0.0)) - offset
            end = float(word.get("end", 0.0)) - offset
        except (TypeError, ValueError):
            continue
        if end <= 0:
            continue
        adjusted.append({**word, "start": max(0.0, start), "end": max(0.0, end)})
    return adjusted


@multicam_bp.route("/multicam_slide", methods=["POST"])
def multicam_slide():
    required = {"left_video", "right_video", "audio"}
    if any(name not in request.files for name in required):
        return (
            jsonify({"error": "Missing required files: left_video, right_video, audio"}),
            400,
        )

    left_file = request.files["left_video"]
    right_file = request.files["right_video"]
    audio_file = request.files["audio"]

    direction = request.form.get("direction", settings.defaults.direction)
    model_size = request.form.get("model_size", settings.defaults.model_size)
    device = resolve_device(request.form.get("device"))
    compute_type = compute_type_for_device(device)
    overlap = parse_numeric(request.form.get("overlap"), float, settings.defaults.overlap)
    output_name = request.form.get("output_name")

    left_path = save_upload(left_file, settings.multicam_folder, "left")
    right_path = save_upload(right_file, settings.multicam_folder, "right")
    audio_path = save_upload(audio_file, settings.multicam_folder, "audio")

    sync_result = None
    if parse_boolean(request.form.get("sync_first"), default=False):
        sync_defaults = settings.sync_defaults
        sync_target_fps = parse_numeric(
            request.form.get("sync_target_fps"),
            float,
            sync_defaults.target_fps,
        )
        sync_corr_sr = parse_numeric(
            request.form.get("sync_corr_sr"),
            int,
            sync_defaults.corr_sr,
        )
        sync_max_shift = parse_numeric(
            request.form.get("sync_max_shift"),
            float,
            sync_defaults.max_shift,
        )
        sync_max_seconds = parse_numeric(
            request.form.get("sync_max_seconds"),
            float,
            sync_defaults.max_seconds,
        )

        job_dir = settings.multicam_folder / f"sync_{uuid.uuid4().hex}"
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            sync_result = sync_videos_with_reference_audio(
                str(left_path),
                str(right_path),
                str(audio_path),
                output_dir=str(job_dir),
                target_fps=sync_target_fps,
                corr_sr=sync_corr_sr,
                max_shift_s=sync_max_shift,
                max_seconds=sync_max_seconds,
                generate_stitched=False,
            )
        except VideoSyncError as exc:
            return jsonify({"error": f"Sync step failed: {exc}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Unexpected sync failure: {exc}"}), 500

        left_path = Path(sync_result.aligned_video_paths[0])
        right_path = Path(sync_result.aligned_video_paths[1])
        audio_path = Path(sync_result.aligned_audio_path)

    if "word_json" in request.form:
        try:
            incoming_words = json.loads(request.form.get("word_json", "[]"))
        except Exception as exc:
            return jsonify({"error": f"Invalid word_json: {exc}"}), 400

        if sync_result and sync_result.t0:
            words = _adjust_word_timings(incoming_words, sync_result.t0)
        else:
            words = incoming_words
    else:
        try:
            words = transcribe_audio_with_whisperx(
                str(audio_path),
                model_name=model_size,
                device=device,
                compute_type=compute_type,
                expected_speakers=settings.defaults.expected_speakers,
            )
        except Exception as exc:
            return jsonify({"error": f"Transcription failed: {exc}"}), 500

    base_name = output_name or f"multicam_slide_{Path(left_file.filename).stem}"
    output_path = settings.multicam_folder / f"{base_name}.mp4"

    try:
        combine_multicam_with_slide(
            left_video_path=str(left_path),
            right_video_path=str(right_path),
            audio_path=str(audio_path),
            words=words,
            output_path=str(output_path),
            direction=direction,
            overlap=overlap,
            auto_sync=not bool(sync_result),
        )
    except Exception as exc:
        return jsonify({"error": f"Multicam assembly failed: {exc}"}), 500

    caption_output = settings.upload_folder / settings.caption_output_filename
    try:
        add_dynamic_subtitles_to_video(
            video_path=str(output_path),
            words_with_timestamps=words,
            output_path=str(caption_output),
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to render subtitles: {exc}"}), 500
    final_url=upload_video(caption_output)
    return jsonify({"message": "OK", "output": settings.sample_delivery_url})
