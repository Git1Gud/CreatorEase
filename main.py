from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.segment_utils import group_segments_two_speakers, print_speaker_segments,format_speaker_segments
from utils.caption_utils import add_dynamic_subtitles_to_video
import time
import os

def process_and_save_video_with_segments(
    video_path, output_path, model_size="small", device=None, style="modern"
):
    start_time = time.time()
    words_with_timestamps = transcribe_audio_with_whisperx(
        video_path,
        model_name=model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    segments = group_segments_two_speakers(words_with_timestamps)
    formatted_segments=format_speaker_segments(segments)
    print("Formatted Segments:",formatted_segments[:5])
    add_dynamic_subtitles_to_video(video_path, words_with_timestamps, output_path, style=style)
    end_time = time.time()
    print(f"Total processing time: {end_time-start_time:.2f} seconds")

if __name__ == "__main__":
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, "test.mp4")
    output_path = os.path.join(uploads_dir, "new_output_with_captions.mp4")
    process_and_save_video_with_segments(
        video_path, output_path, model_size="small", device="cuda", style="modern"
    )