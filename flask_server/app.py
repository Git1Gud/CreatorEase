from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from utils.segment_pipeline import process_and_save_video_with_segments
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "zaidgey"})

if __name__ == "__main__":
    app.run(debug=True)