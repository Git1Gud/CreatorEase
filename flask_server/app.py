from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from utils.segment_pipeline import process_and_save_video_with_segments
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.multicam import combine_multicam_with_slide
from utils.cloudinary_utils import upload_video
import os
import time
import json

app = Flask(__name__)
CORS(app) # Enable CORS for all origins

# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MULTICAM_FOLDER = os.path.join(UPLOAD_FOLDER, "multicam")
os.makedirs(MULTICAM_FOLDER, exist_ok=True)

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

    # Auto-generate words JSON using your transcription utility on the provided audio

    if 'word_json' in request.form:
        words=json.loads(request.form.get('word_json'))
    else :
        try:
            words = transcribe_audio_with_whisperx(
                audio_path,
                model_name=model_size,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
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
        )
    except Exception as e:
        return jsonify({"error": f"Multicam assembly failed: {e}"}), 500
    final_output=os.path.join(UPLOAD_FOLDER, f"output_with_captions.mp4")
    add_dynamic_subtitles_to_video(video_path=output_path,words_with_timestamps=words,output_path=final_output)
    end_url=upload_video(final_output)
    # end_url= r'http://res.cloudinary.com/dxt0biqah/video/upload/v1757411328/videos/swgfyxrttojxrjuqcdv4.mp4'
    return jsonify({"message": "OK", "output": end_url})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "zaidgey"})

if __name__ == "__main__":  
    app.run(debug=True)