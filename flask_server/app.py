from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from utils.segment_pipeline import process_and_save_video_with_segments
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.multicam import combine_multicam_with_slide
from utils.cloudinary_utils import upload_video
from utils.video_sync import (
    sync_videos_with_reference_audio,
    VideoSyncError,
)
import os
import time
import json
import uuid

app = Flask(__name__)
CORS(app) # Enable CORS for all origins

# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MULTICAM_FOLDER = os.path.join(UPLOAD_FOLDER, "multicam")
os.makedirs(MULTICAM_FOLDER, exist_ok=True)
SYNC_FOLDER = os.path.join(UPLOAD_FOLDER, "sync")
os.makedirs(SYNC_FOLDER, exist_ok=True)

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Process the video
    output_dir = UPLOAD_FOLDER
    start_time = time.time()
    try:
        urls = process_and_save_video_with_segments(
            video_path, output_dir, model_size="small", device="cuda", style="modern"
        )
    except Exception as e:
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    end_time = time.time()

    return jsonify({
        "message": "Video processed successfully",
        "processing_time": f"{end_time - start_time:.2f} seconds",
        "urls": urls
    })

@app.route('/add_subtitles', methods=['POST'])
def add_subtitles():
    """
    Add dynamic subtitles to a video.
    Accepts multipart/form-data with fields:
      - file: the video file
      - words_json (optional): JSON array of {word,start,end} to use; if omitted, auto-transcribes.
      - style (optional): caption style (modern|vibrant|minimal)
      - model_size (optional): whisperx model size (tiny|base|small|medium|large)
      - device (optional): cuda|cpu (defaults to auto)
    Returns: output path.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    style = request.form.get('style', 'modern')
    model_size = request.form.get('model_size', 'small')
    device = request.form.get('device')  # may be None

    # Save input video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Load words if provided
    import json
    words_json = request.form.get('words_json')
    if words_json:
        try:
            words = json.loads(words_json)
        except Exception as e:
            return jsonify({"error": f"Invalid words_json: {e}"}), 400
    else:
        # Auto-transcribe with WhisperX
        try:
            words = transcribe_audio_with_whisperx(
                video_path,
                model_name=model_size,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
            )
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {e}"}), 500

    # Produce captioned output
    name, ext = os.path.splitext(os.path.basename(video_path))
    out_path = os.path.join(UPLOAD_FOLDER, f"{name}_with_captions{ext}")
    try:
        add_dynamic_subtitles_to_video(video_path, words, out_path, style=style)
    except Exception as e:
        return jsonify({"error": f"Captioning failed: {e}"}), 500

    return jsonify({
        "message": "Subtitles added successfully",
        "output": out_path
    })

@app.route('/multicam_slide', methods=['POST'])
def multicam_slide():
    """
    Build a single video from two camera videos and a single audio track using diarized speakers
    to switch cameras with slide-only transitions.

    multipart/form-data fields:
      - left_video: left camera video file
      - right_video: right camera video file
      - audio: final audio file (wav/mp3/m4a)
    -word_json: word json having word, start, end time and speaker
    - model_size (optional): whisperx model size (tiny|base|small|medium|large), default 'small'
    - device (optional): cuda|cpu (defaults to auto)
      - direction (optional): 'ltr' (default) or 'rtl'
      - overlap (optional): seconds for slide animation (default 0.6)
      - output_name (optional): filename for the result (default generated)
    """
    if 'left_video' not in request.files or 'right_video' not in request.files or 'audio' not in request.files:
        return jsonify({"error": "Missing required files: left_video, right_video, audio"}), 400

    left_file = request.files['left_video']
    right_file = request.files['right_video']
    audio_file = request.files['audio']
    direction = request.form.get('direction', 'ltr')
    model_size = request.form.get('model_size', 'medium')
    device = request.form.get('device','cpu')  # may be None
    try:
        overlap = float(request.form.get('overlap', '0.3'))
    except ValueError:
        overlap = 0.3
    output_name = request.form.get('output_name')

    # Persist inputs (specific to multicam under uploads/multicam)
    os.makedirs(MULTICAM_FOLDER, exist_ok=True)
    left_path = os.path.join(MULTICAM_FOLDER, f"left_{left_file.filename}")
    right_path = os.path.join(MULTICAM_FOLDER, f"right_{right_file.filename}")
    audio_path = os.path.join(MULTICAM_FOLDER, f"audio_{audio_file.filename}")
    left_file.save(left_path)
    right_file.save(right_path)
    audio_file.save(audio_path)

    # Optional pre-sync step to align videos and audio before multicam composition
    sync_enabled = request.form.get('sync_first', 'true').lower() != 'false'
    sync_result = None
    if sync_enabled:
        sync_target_fps = float(request.form.get('sync_target_fps', 30.0))
        sync_corr_sr = int(request.form.get('sync_corr_sr', 8000))
        sync_max_shift = float(request.form.get('sync_max_shift', 10.0))
        sync_max_seconds = float(request.form.get('sync_max_seconds', 60.0))

        sync_job_dir = os.path.join(MULTICAM_FOLDER, f"sync_{uuid.uuid4().hex}")
        os.makedirs(sync_job_dir, exist_ok=True)

        try:
            sync_result = sync_videos_with_reference_audio(
                left_path,
                right_path,
                audio_path,
                output_dir=sync_job_dir,
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

        left_path, right_path = sync_result.aligned_video_paths
        audio_path = sync_result.aligned_audio_path

    # Auto-generate words JSON using the (possibly trimmed) audio
    if 'word_json' in request.form:
        try:
            incoming_words = json.loads(request.form.get('word_json'))
        except Exception as e:
            return jsonify({"error": f"Invalid word_json: {e}"}), 400

        if sync_result and sync_result.t0:
            adjusted = []
            trim_offset = sync_result.t0
            for w in incoming_words:
                try:
                    start = float(w.get('start', 0.0)) - trim_offset
                    end = float(w.get('end', 0.0)) - trim_offset
                except (TypeError, ValueError):
                    continue
                if end <= 0:
                    continue
                adjusted.append({
                    **w,
                    'start': max(0.0, start),
                    'end': max(0.0, end),
                })
            words = adjusted
        else:
            words = incoming_words
    else:
        try:
            words = transcribe_audio_with_whisperx(
                audio_path,
                model_name=model_size,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
                expected_speakers=2
            )
            print(words)
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {e}"}), 500
    
    # print(request.form.get('word_json'))
    

    # for w in words:
    #     print(w.get('word'),w.get('speaker'))
    # Output path
    base = output_name or f"multicam_slide_{os.path.splitext(left_file.filename)[0]}"
    output_path = os.path.join(MULTICAM_FOLDER, f"{base}.mp4")

    try:
        url = combine_multicam_with_slide(
            left_video_path=left_path,
            right_video_path=right_path,
            audio_path=audio_path,
            words=words,
            output_path=output_path,
            direction=direction,
            overlap=overlap,
            auto_sync=not bool(sync_result),
        )
    except Exception as e:
        return jsonify({"error": f"Multicam assembly failed: {e}"}), 500
    final_output=os.path.join(UPLOAD_FOLDER, f"output_with_captions.mp4")
    add_dynamic_subtitles_to_video(video_path=output_path,words_with_timestamps=words,output_path=final_output)
    # end_url=upload_video(final_output)
    end_url= r'http://res.cloudinary.com/dxt0biqah/video/upload/v1757411328/videos/swgfyxrttojxrjuqcdv4.mp4'
    return jsonify({"message": "OK", "output": end_url})


@app.route('/sync_videos', methods=['POST'])
def sync_videos():
    required_fields = {'video1', 'video2', 'audio'}
    if not request.files:
        return jsonify({
            "error": "No files found in request. Ensure you're sending multipart/form-data with fields: video1, video2, audio."
        }), 400

    missing = []
    for field in sorted(required_fields):
        storage = request.files.get(field)
        if storage is None or storage.filename == '':
            missing.append(field)

    if missing:
        return jsonify({
            "error": f"Missing or empty file fields: {', '.join(missing)}",
            "received_fields": list(request.files.keys()),
        }), 400

    video1_file = request.files['video1']
    video2_file = request.files['video2']
    audio_file = request.files['audio']

    try:
        target_fps = float(request.form.get('target_fps', 30.0))
        corr_sr = int(request.form.get('corr_sr', 8000))
        max_shift = float(request.form.get('max_shift', 10.0))
        max_seconds = float(request.form.get('max_seconds', 60.0))
    except ValueError as exc:
        return jsonify({"error": f"Invalid numeric parameter: {exc}"}), 400

    if target_fps <= 0:
        return jsonify({"error": "target_fps must be positive"}), 400
    if corr_sr <= 0:
        return jsonify({"error": "corr_sr must be positive"}), 400

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(SYNC_FOLDER, job_id)
    os.makedirs(job_dir, exist_ok=True)

    def _save_with_prefix(prefix: str, storage):
        filename = storage.filename or f"{prefix}.mp4"
        safe_name = f"{prefix}_{os.path.basename(filename)}"
        path = os.path.join(job_dir, safe_name)
        storage.save(path)
        return path

    video1_path = _save_with_prefix('video1', video1_file)
    video2_path = _save_with_prefix('video2', video2_file)
    audio_path = _save_with_prefix('audio', audio_file)

    stitched_output_path = os.path.join(job_dir, f"synced_{uuid.uuid4().hex}.mp4")

    try:
        result = sync_videos_with_reference_audio(
            video1_path,
            video2_path,
            audio_path,
            output_dir=job_dir,
            target_fps=target_fps,
            corr_sr=corr_sr,
            max_shift_s=max_shift,
            max_seconds=max_seconds,
            generate_stitched=True,
            stitched_filename=stitched_output_path,
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
        response_payload["aligned_audio_relative"] = os.path.relpath(result.aligned_audio_path, start=os.getcwd())
    except ValueError:
        pass

    if result.output_path:
        try:
            response_payload["output_relative"] = os.path.relpath(result.output_path, start=os.getcwd())
        except ValueError:
            pass
        response_payload["output"] = result.output_path

    return jsonify(response_payload)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "zaidgey"})

if __name__ == "__main__":  
    app.run(debug=True)