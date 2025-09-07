from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.segment_utils import group_segments_two_speakers, format_speaker_segments_with_neighbors
from utils.caption_utils import add_dynamic_subtitles_to_video
from utils.predict import predict_rating_for_segments
from utils.s3_utils import upload_to_s3
from utils.llm import EngagementQuestionGenerator
from utils.audio_generate import get_narration
import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips
import concurrent.futures as cf


def _parse_segment_times(segment_text):
    """Extract (start, end) floats from a formatted segment string like '... (12.3-18.7)'"""
    time_part = segment_text.split('(')[1].split(')')[0]
    seg_start, seg_end = map(float, time_part.split('-'))
    return seg_start, seg_end


def _select_top_segments(formatted_segments, ratings, k=3):
    """Return top-k segments and their (start, end) times based on ratings."""
    top_indices = np.argsort(ratings)[-k:][::-1]
    top_segments = [formatted_segments[i] for i in top_indices]
    top_times = [_parse_segment_times(s) for s in top_segments]
    top_ratings = [ratings[i] for i in top_indices]
    return top_indices, top_segments, top_times, top_ratings


def _filter_words_for_segment(words_with_timestamps, start, end):
    """Filter words that lie within [start, end], inclusive."""
    return [w for w in words_with_timestamps if start <= w['start'] <= end]


def _clip_and_caption_segment(original_clip, seg_start, seg_end, segment_words, segments_dir, out_dir, idx, style):
    """Clip a segment from the original video, then overlay dynamic captions for that segment.

    Returns: (segment_path, captioned_segment_path)
    """
    buffer = 0.05
    clip_start = max(0, seg_start - buffer)
    clip_end = min(original_clip.duration, seg_end + buffer)

    segment_clip = original_clip.subclip(clip_start, clip_end)
    segment_path = os.path.join(segments_dir, f"segment{idx}.mp4")
    temp_audio = os.path.join(segments_dir, f"temp-audio-seg{idx}-{os.getpid()}.m4a")
    segment_clip.write_videofile(
        segment_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=temp_audio,
        remove_temp=True,
        threads=4
    )
    # Explicitly close to release file handles on Windows
    try:
        segment_clip.close()
    except Exception:
        pass

    # Adjust timestamps to be relative to segment start for caption overlay
    rel_words = [
        {**w, "start": w["start"] - seg_start, "end": w["end"] - seg_start}
        for w in segment_words
    ]

    captioned_path = os.path.join(out_dir, f"segment{idx}_with_captions.mp4")
    add_dynamic_subtitles_to_video(segment_path, rel_words, captioned_path, style=style)
    return segment_path, captioned_path


def _create_captioned_hook_video(hook_text, base_video_path, out_dir, idx, style):
    """TTS the hook, create a blank video sized like base_video_path, overlay dynamic captions, and return its path."""
    # Generate narration and timestamps
    audio_path, hook_words = get_narration(hook_text)

    # Use base video properties (size/fps)
    from moviepy.editor import ColorClip
    with VideoFileClip(base_video_path) as base_clip:
        with AudioFileClip(audio_path) as audio_clip:
            blank = ColorClip(size=base_clip.size, color=(0, 0, 0), duration=audio_clip.duration).set_fps(base_clip.fps)
            blank = blank.set_audio(audio_clip)
            temp_blank_path = os.path.join(out_dir, f"segment{idx}_hook_blank.mp4")
            blank.write_videofile(temp_blank_path, codec='libx264', audio_codec='aac', remove_temp=True, threads=4)

    # Add captions to the blank video
    subtitled_blank_path = os.path.join(out_dir, f"segment{idx}_hook_with_captions.mp4")
    add_dynamic_subtitles_to_video(temp_blank_path, hook_words, subtitled_blank_path, style=style)

    # Cleanup temp blank
    if os.path.exists(temp_blank_path):
        os.remove(temp_blank_path)

    return subtitled_blank_path


def _create_captioned_hook_from_audio(audio_path, base_video_path, out_dir, idx, style, hook_words):
    """Like _create_captioned_hook_video but uses pre-generated audio and provided word timestamps."""
    from moviepy.editor import ColorClip
    with VideoFileClip(base_video_path) as base_clip:
        with AudioFileClip(audio_path) as audio_clip:
            blank = ColorClip(size=base_clip.size, color=(0, 0, 0), duration=audio_clip.duration).set_fps(base_clip.fps)
            blank = blank.set_audio(audio_clip)
            temp_blank_path = os.path.join(out_dir, f"segment{idx}_hook_blank.mp4")
            blank.write_videofile(temp_blank_path, codec='libx264', audio_codec='aac', remove_temp=True, threads=4)

    subtitled_blank_path = os.path.join(out_dir, f"segment{idx}_hook_with_captions.mp4")
    add_dynamic_subtitles_to_video(temp_blank_path, hook_words, subtitled_blank_path, style=style)
    if os.path.exists(temp_blank_path):
        os.remove(temp_blank_path)
    return subtitled_blank_path


def _process_segment_task(args):
    """Worker to process one segment: clip+caption, create captioned hook from audio, and concatenate."""
    (
        idx,
        video_path,
        seg_start,
        seg_end,
        segment_words,
        hook_audio_path,
        hook_words,
        segments_dir,
        segment_output_dir,
        style,
    ) = args

    # 1) Clip and caption the segment
    buffer = 0.05
    clip_start = max(0, seg_start - buffer)
    clip_end = seg_end + buffer
    segment_path = os.path.join(segments_dir, f"segment{idx}.mp4")
    captioned_path = os.path.join(segment_output_dir, f"segment{idx}_with_captions.mp4")

    with VideoFileClip(video_path) as original_clip:
        clip_end = min(original_clip.duration, clip_end)
        segment_clip = original_clip.subclip(clip_start, clip_end)
        temp_audio = os.path.join(segments_dir, f"temp-audio-seg{idx}-{os.getpid()}.m4a")
        segment_clip.write_videofile(
            segment_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=temp_audio,
            remove_temp=True,
            threads=4
        )
        try:
            segment_clip.close()
        except Exception:
            pass

    # Adjust timestamps relative to segment start
    rel_words = [
        {**w, "start": w["start"] - seg_start, "end": w["end"] - seg_start}
        for w in segment_words
    ]
    add_dynamic_subtitles_to_video(segment_path, rel_words, captioned_path, style=style)

    # 2) Create captioned hook video using pre-generated TTS
    hook_captioned_path = _create_captioned_hook_from_audio(
        hook_audio_path, captioned_path, segment_output_dir, idx, style, hook_words
    )

    # 3) Concatenate
    final_output_path = os.path.join(segment_output_dir, f"segment{idx}_final.mp4")
    with VideoFileClip(captioned_path) as seg_clip, VideoFileClip(hook_captioned_path) as hook_clip:
        final_video = concatenate_videoclips([seg_clip, hook_clip])
        final_video.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

    # Optionally cleanup intermediate hook file
    if os.path.exists(hook_captioned_path):
        os.remove(hook_captioned_path)

    return final_output_path

def process_and_save_video_with_segments(
    video_path, output_dir, model_size="small", device=None, style="modern"
):
    # Transcribe and segment
    generator = EngagementQuestionGenerator(api_key=os.getenv("GEMINI_API_KEY"))
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
    top_indices, top_segments, top_segment_times, top_ratings = _select_top_segments(formatted_segments, ratings, k=3)
    print("Top 3 Segments:")
    for seg, rating in zip(top_segments, top_ratings):
        print(f"Segment: {seg}\nPredicted rating (1-100): {rating}\n")
    print("Top Segment Times:", top_segment_times)

    # Group words for each segment
    segment_words = [[] for _ in range(len(top_segment_times))]
    for i, (start, end) in enumerate(top_segment_times):
        segment_words[i] = _filter_words_for_segment(words_with_timestamps, start, end)

    # Prepare output directories
    segments_dir = os.path.join(output_dir, "segments")
    segment_output_dir = os.path.join(output_dir, "segment_output")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(segment_output_dir, exist_ok=True)
    # Pre-generate hooks and Kokoro audio sequentially (avoid loading models in multiple processes)
    hooks = []
    hook_audio_and_words = []
    for i, seg in enumerate(top_segments, start=1):
        hook_text = generator.generate_question(seg, formatted_segments)
        hooks.append(hook_text)
        print(f"Generated Hook for segment {i}: {hook_text}")
        audio_path, hook_words = get_narration(hook_text)
        hook_audio_and_words.append((audio_path, hook_words))

    # Batch process segments in parallel
    tasks = []
    for i, (seg_times, words) in enumerate(zip(top_segment_times, segment_words), start=1):
        seg_start, seg_end = seg_times
        audio_path, hook_words = hook_audio_and_words[i-1]
        tasks.append((
            i,
            video_path,
            seg_start,
            seg_end,
            words,
            audio_path,
            hook_words,
            segments_dir,
            segment_output_dir,
            style,
        ))

    max_workers = max(1, min(len(tasks), (os.cpu_count() or 2) // 2 or 1))
    results = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_segment_task, t) for t in tasks]
        for fut in futures:
            results.append(fut.result())

        # urls.append(upload_to_s3(final_output_path,final_output_path.split("\\")[-1]))

    return results
