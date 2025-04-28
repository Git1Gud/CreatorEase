from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.segment_utils import group_segments_two_speakers, format_speaker_segments
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.predict import predict_rating_for_segments
import time
import os
import numpy as np
from moviepy.editor import VideoFileClip
import boto3

def upload_to_s3(file_path, bucket_name, s3_key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    s3.upload_file(file_path, bucket_name, s3_key, ExtraArgs={'ACL': 'public-read'})
    url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
    return url

def process_and_save_video_with_segments(
    video_path, output_dir, model_size="small", device=None, style="modern"
):
    # Transcribe and segment
    words_with_timestamps = transcribe_audio_with_whisperx(
        video_path,
        model_name=model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    segments = group_segments_two_speakers(words_with_timestamps)
    formatted_segments = format_speaker_segments(segments)
    ratings = predict_rating_for_segments(
        video_path, formatted_segments, model_path="random_forest_views_rating_model.pkl"
    )
    print("Ratings:", ratings)

    # Get top 3 segments based on ratings
    top_indices = np.argsort(ratings)[-3:][::-1]
    top_segments = [formatted_segments[i] for i in top_indices]
    top_ratings = [ratings[i] for i in top_indices]
    print("Top 3 Segments:")
    for seg, rating in zip(top_segments, top_ratings):
        print(f"Segment: {seg}\nPredicted rating (1-100): {rating}\n")

    # Extract segment timestamps
    top_segment_times = []
    for seg in top_segments:
        time_part = seg.split('(')[1].split(')')[0]
        seg_start, seg_end = map(float, time_part.split('-'))
        top_segment_times.append((seg_start, seg_end))
    print("Top Segment Times:", top_segment_times)

    # Group words for each segment
    segment_words = [[] for _ in range(len(top_segment_times))]
    for word in words_with_timestamps:
        word_time = word['start']
        for i, (start, end) in enumerate(top_segment_times):
            if start <= word_time <= end:
                segment_words[i].append(word)
                break

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    original_clip = VideoFileClip(video_path)

    # Clip and subtitle each segment
    for i, (seg_start, seg_end) in enumerate(top_segment_times):
        buffer = 0.0
        clip_start = max(0, seg_start - buffer)
        clip_end = min(original_clip.duration, seg_end + buffer)
        segment_clip = original_clip.subclip(clip_start, clip_end)
        segment_path = os.path.join(output_dir, f"segment{i+1}.mp4")
        segment_clip.write_videofile(
            segment_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            threads=4
        )
        # Add subtitles to the segment
        output_captioned_path = os.path.join(output_dir, f"segment{i+1}_with_captions.mp4")
        segment_words_with_timestamps=transcribe_audio_with_whisperx(
            segment_path,
            model_name=model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
        add_dynamic_subtitles_to_video(segment_path, segment_words_with_timestamps, output_captioned_path, style=style)
        print(f"Saved segment {i+1} to {segment_path} and captioned to {output_captioned_path}")

    original_clip.close()

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