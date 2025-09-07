from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from utils.segment_pipeline import process_and_save_video_with_segments
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.transcription_utils import transcribe_audio_with_whisperx
import os
import time

app = Flask(__name__)
CORS(app) # Enable CORS for all origins

# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "zaidgey"})

if __name__ == "__main__":
    app.run(debug=True)