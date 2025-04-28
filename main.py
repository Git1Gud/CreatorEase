from utils.s3_utils import upload_to_s3
from utils.segment_pipeline import process_and_save_video_with_segments
import time
import os

if __name__ == "__main__":
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, "test.mp4")
    output_dir = uploads_dir
    start_time = time.time()
    process_and_save_video_with_segments(
        video_path, output_dir, model_size="small", device="cuda", style="modern"
    )
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")