from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.segment_utils import group_segments_two_speakers, format_speaker_segments_with_neighbors
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.predict import predict_rating_for_segments
from utils.s3_utils import upload_to_s3
from utils.llm import EngagementQuestionGenerator
from utils.audio_generate import get_narration
import os
import numpy as np
from moviepy.editor import VideoFileClip

def process_and_save_video_with_segments(
    video_path, output_dir, model_size="small", device=None, style="modern"
):
    # Transcribe and segment
    generator = EngagementQuestionGenerator(api_key=os.getenv("GROQ_API_KEY"))
    urls=[]
    words_with_timestamps = transcribe_audio_with_whisperx(
        video_path,
        model_name=model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    segments = group_segments_two_speakers(words_with_timestamps)
    formatted_segments = format_speaker_segments_with_neighbors(segments)
    # Remove duplicates while preserving order
    seen = set()
    formatted_segments = [x for x in formatted_segments if not (x in seen or seen.add(x))]
    # print(formatted_segments)
    ratings = predict_rating_for_segments(
        video_path, formatted_segments, model_path=os.path.join("models", "random_forest_views_rating_model.pkl"),
    )
    print("Ratings:", ratings)

    # Get top 3 segments based on ratings
    top_indices = np.argsort(ratings)[-1:][::-1]
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

    # Prepare output directories
    segments_dir = os.path.join(output_dir, "segments")
    segment_output_dir = os.path.join(output_dir, "segment_output")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(segment_output_dir, exist_ok=True)
    original_clip = VideoFileClip(video_path)

    # Clip and subtitle each segment
    for i, (seg_start, seg_end) in enumerate(top_segment_times):
        buffer = 0.05
        clip_start = max(0, seg_start - buffer)
        clip_end = min(original_clip.duration, seg_end + buffer)
        segment_clip = original_clip.subclip(clip_start, clip_end)
        segment_path = os.path.join(segments_dir, f"segment{i+1}.mp4")
        segment_clip.write_videofile(
            segment_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            threads=4
        )
        
        print(f"Words in segment {i+1}: {[w['word'] for w in segment_words[i]]}")
        # Adjust timestamps to be relative to segment start
        segment_words_with_timestamps = [
            {
                **word,
                "start": word["start"] - seg_start,
                "end": word["end"] - seg_start
            }
            for word in segment_words[i]
        ]
        output_captioned_path = os.path.join(segment_output_dir, f"segment{i+1}_with_captions.mp4")
        add_dynamic_subtitles_to_video(segment_path, segment_words_with_timestamps, output_captioned_path, style=style)
        print(f"Saved segment {i+1} to {segment_path} and captioned to {output_captioned_path}")


        hook = generator.generate_question(top_segments[i], formatted_segments)
        print(f"Generated Hook: {hook}")
        audio_path=get_narration(hook)
        print(f"Generated Audio Path: {audio_path}")

        # urls.append(upload_to_s3(output_captioned_path,output_captioned_path.split("\\")[-1]))

    original_clip.close()
    return urls
