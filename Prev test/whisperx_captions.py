import os
import time
import numpy as np
import cv2
import random
import colorsys
import platform
import gc
import whisperx
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

load_dotenv()

def check_gpu_availability():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def transcribe_audio_with_whisperx(audio_file, model_name="small", device=None, compute_type="float16", batch_size=16):
    if device is None:
        device = check_gpu_availability()
    try:
        print(f"Loading {model_name} model...")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        print("Transcribing with Whisper...")
        result = model.transcribe(audio, batch_size=batch_size)
        print(f"Detected language: {result['language']}")
        del model; gc.collect()
        if device == "cuda":
            import torch; torch.cuda.empty_cache()
        print("Aligning words...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a; gc.collect()
        if device == "cuda":
            import torch; torch.cuda.empty_cache()
        try:
            if os.getenv('HUGGINGFACE_TOKEN'):
                print("Identifying speakers...")
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HUGGINGFACE_TOKEN'), device=device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print(f"Speaker diarization failed: {e}. Continuing without speaker labels.")
        words_with_timestamps = []
        for segment in result["segments"]:
            speaker = segment.get("speaker", "")
            for word in segment.get("words", []):
                if "start" in word and "end" in word:
                    words_with_timestamps.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"],
                        "speaker": speaker
                    })
        return words_with_timestamps
    except Exception as e:
        print(f"Error in transcription: {e}")
        raise

def generate_vibrant_colors(num_colors=5):
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = random.random()
    for _ in range(num_colors):
        hue += golden_ratio_conjugate
        hue %= 1.0
        saturation = 0.85 + random.random() * 0.15
        value = 0.95 + random.random() * 0.05
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

def get_system_fonts():
    system = platform.system()
    if system == "Windows":
        font_dir = Path("C:/Windows/Fonts")
        return [font_dir / "arialbd.ttf", font_dir / "Arial.ttf"]
    elif system == "Darwin":
        return [
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf")
        ]
    else:
        return [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        ]

def load_font(size, bold=True):
    font_paths = get_system_fonts()
    if bold:
        font_paths = [p for p in font_paths if "bold" in p.name.lower() or "bd" in p.name.lower()] + \
                     [p for p in font_paths if "bold" not in p.name.lower() and "bd" not in p.name.lower()]
    for font_path in font_paths:
        try:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()

def get_text_dimensions(text, font):
    if font is None:
        return len(text) * 8, 16
    try:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        elif hasattr(font, "getsize"):
            return font.getsize(text)
        else:
            return len(text) * (font.size // 2), font.size
    except:
        return len(text) * 8, 16

def create_captionsai_style_frame(frame, words_to_display, current_word_idx, frame_width, frame_height, font=None, color_scheme=None, speaker_colors=None):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if color_scheme is None:
            text_color = (255, 255, 255, 255)
            highlight_color = (255, 230, 0, 255)
            shadow_color = (0, 0, 0, 180)
        else:
            text_color = (*color_scheme["text"], 255)
            highlight_color = (*color_scheme["highlight"], 255)
            shadow_color = (*color_scheme["shadow"], 180)
        full_text = " ".join([word["word"] for word in words_to_display])
        max_width = int(frame_width * 0.8)
        wrapped_lines = []
        words = full_text.split()
        current_line = []
        current_width = 0
        for word in words:
            word_width, _ = get_text_dimensions(word + " ", font)
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                wrapped_lines.append((current_line, current_width))
                current_line = [word]
                current_width = word_width
        if current_line:
            wrapped_lines.append((current_line, current_width))
        _, line_height = get_text_dimensions("Ay", font)
        text_block_height = len(wrapped_lines) * line_height * 1.2
        bottom_padding = int(frame_height * 0.1)
        y_position = frame_height - bottom_padding - text_block_height
        global_word_idx = 0
        for line_idx, (line_words, line_width) in enumerate(wrapped_lines):
            x_position = (frame_width - line_width) // 2
            line_y = y_position + line_idx * line_height * 1.2
            joined_line = " ".join(line_words)
            for offset_x, offset_y in [(2, 2), (-2, 2), (2, -2), (-2, -2)]:
                draw.text(
                    (x_position + offset_x, line_y + offset_y),
                    joined_line,
                    font=font,
                    fill=shadow_color
                )
            for word in line_words:
                word_with_space = word + " "
                word_width, _ = get_text_dimensions(word_with_space, font)
                is_current = (global_word_idx == current_word_idx)
                if speaker_colors and global_word_idx < len(words_to_display):
                    speaker = words_to_display[global_word_idx].get("speaker", "")
                    word_color = speaker_colors.get(speaker, text_color)
                    color = highlight_color if is_current else word_color
                else:
                    color = highlight_color if is_current else text_color
                if is_current:
                    glow_padding = 3
                    word_box = [
                        x_position - glow_padding,
                        line_y - glow_padding,
                        x_position + word_width + glow_padding,
                        line_y + line_height + glow_padding
                    ]
                    try:
                        draw.rounded_rectangle(
                            word_box,
                            radius=8,
                            fill=(highlight_color[0], highlight_color[1], highlight_color[2], 60)
                        )
                    except AttributeError:
                        draw.rectangle(
                            word_box,
                            fill=(highlight_color[0], highlight_color[1], highlight_color[2], 60)
                        )
                draw.text((x_position, line_y), word_with_space, font=font, fill=color)
                x_position += word_width
                global_word_idx += 1
        frame_with_overlay = Image.alpha_composite(pil_frame.convert('RGBA'), overlay)
        result = np.array(frame_with_overlay.convert('RGB'))
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error creating captioned frame: {e}")
        return frame

def group_words_into_chunks(words_with_timestamps, min_chunk_duration=1.0):
    chunks = []
    current_chunk = []
    chunk_start = None
    for word in words_with_timestamps:
        if not current_chunk:
            chunk_start = word["start"]
        current_chunk.append(word)
        if (word["end"] - chunk_start >= min_chunk_duration) or (word == words_with_timestamps[-1]):
            chunks.append({
                "words": current_chunk,
                "start": chunk_start,
                "end": word["end"]
            })
            current_chunk = []
    return chunks

def add_dynamic_subtitles_to_video(video_path, words_with_timestamps, output_path, style="modern"):
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_output = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
        if not output_video.isOpened():
            raise ValueError(f"Could not create output video file: {temp_output}")
        vibrant_colors = generate_vibrant_colors(5)
        color_schemes = {
            "modern": {"text": (255, 255, 255), "highlight": (255, 230, 0), "shadow": (0, 0, 0)},
            "vibrant": {"text": (255, 255, 255), "highlight": vibrant_colors[0], "shadow": (0, 0, 0)},
            "minimal": {"text": (255, 255, 255), "highlight": (255, 255, 255), "shadow": (0, 0, 0)}
        }
        color_scheme = color_schemes.get(style, color_schemes["modern"])
        speakers = set(word.get("speaker", "") for word in words_with_timestamps if word.get("speaker"))
        speaker_colors = {speaker: vibrant_colors[i % len(vibrant_colors)] for i, speaker in enumerate(speakers) if speaker}
        font_size = int(frame_height * 0.055)
        font = load_font(font_size, bold=True)
        chunks = group_words_into_chunks(words_with_timestamps, min_chunk_duration=1.0)
        frame_idx = 0
        chunk_idx = 0
        while frame_idx < frame_count:
            ret, frame = video.read()
            if not ret:
                break
            current_time = frame_idx / fps
            while (chunk_idx + 1 < len(chunks)) and (current_time > chunks[chunk_idx]["end"]):
                chunk_idx += 1
            chunk = chunks[chunk_idx]
            highlight_idx = None
            for i, word in enumerate(chunk["words"]):
                if word["start"] <= current_time <= word["end"]:
                    highlight_idx = i
                    break
            frame = create_captionsai_style_frame(
                frame,
                chunk["words"],
                highlight_idx,
                frame_width,
                frame_height,
                font=font,
                color_scheme=color_scheme,
                speaker_colors=speaker_colors if speakers else None
            )
            output_video.write(frame)
            frame_idx += 1
        video.release()
        output_video.release()
        print("Adding audio to final video...")
        with VideoFileClip(video_path) as video_clip, \
             VideoFileClip(temp_output) as processed_clip:
            if video_clip.audio:
                if video_clip.audio.duration > processed_clip.duration:
                    audio_clip = video_clip.audio.subclip(0, processed_clip.duration)
                else:
                    audio_clip = video_clip.audio
                video_with_audio = processed_clip.set_audio(audio_clip)
                video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None, threads=4)
            else:
                processed_clip.write_videofile(output_path, codec='libx264', verbose=False, logger=None, threads=4)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"Subtitles added successfully! Output: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error adding subtitles to video: {e}")
        raise

def group_by_speaker_segments(words_with_timestamps):
    segments = []
    if not words_with_timestamps:
        return segments
    current_speaker = words_with_timestamps[0].get("speaker", "unknown")
    current_segment = {
        "speaker": current_speaker,
        "words": [],
        "start": words_with_timestamps[0]["start"],
        "end": words_with_timestamps[0]["end"]
    }
    for word in words_with_timestamps:
        speaker = word.get("speaker", "unknown")
        if speaker != current_speaker:
            current_segment["end"] = current_segment["words"][-1]["end"]
            segments.append(current_segment)
            current_segment = {
                "speaker": speaker,
                "words": [],
                "start": word["start"],
                "end": word["end"]
            }
            current_speaker = speaker
        current_segment["words"].append(word)
        current_segment["end"] = word["end"]
    if current_segment["words"]:
        segments.append(current_segment)
    return segments

def group_segments_two_speakers(words_with_timestamps):
    """
    Group words into segments, each segment containing up to two consecutive speakers.
    Add a small differentiator (e.g., '||') in the text when the speaker changes within a segment.
    """
    segments = []
    if not words_with_timestamps:
        return segments

    i = 0
    n = len(words_with_timestamps)
    while i < n:
        seg_words = []
        speakers_in_seg = []
        start = words_with_timestamps[i]["start"]
        current_speaker = words_with_timestamps[i].get("speaker", "unknown")
        speakers_in_seg.append(current_speaker)
        seg_words.append(words_with_timestamps[i])
        i += 1

        # Collect words from the first speaker
        while i < n and words_with_timestamps[i].get("speaker", "unknown") == current_speaker:
            seg_words.append(words_with_timestamps[i])
            i += 1

        # If there is a second speaker, collect their words too
        speaker_change_idx = None
        if i < n:
            next_speaker = words_with_timestamps[i].get("speaker", "unknown")
            if next_speaker != current_speaker:
                speakers_in_seg.append(next_speaker)
                # Mark the index where the speaker changes
                speaker_change_idx = len(seg_words)
                seg_words.append(words_with_timestamps[i])
                i += 1
                while i < n and words_with_timestamps[i].get("speaker", "unknown") == next_speaker:
                    seg_words.append(words_with_timestamps[i])
                    i += 1

        end = seg_words[-1]["end"]
        segments.append({
            "speakers": speakers_in_seg,
            "words": seg_words,
            "start": start,
            "end": end,
            "speaker_change_idx": speaker_change_idx  # None if only one speaker
        })
    return segments

def print_speaker_segments(segments):
    """
    Print segments with speakers, timestamps, and a differentiator when the speaker changes.
    """
    for seg in segments:
        speakers_str = " & ".join(seg['speakers'])
        text = []
        for idx, w in enumerate(seg['words']):
            # Insert differentiator when speaker changes
            if seg.get("speaker_change_idx") is not None and idx == seg["speaker_change_idx"]:
                text.append("||")  # Differentiator
            text.append(w['word'])
        print(f"{speakers_str} ({seg['start']:.2f}-{seg['end']:.2f}): {' '.join(text)}")


def process_and_save_video_with_segments(
    video_path, output_path, model_size="small", device=None, style="modern"
):
    """
    Transcribe, segment, print speaker segments, and save video with captions.
    """
    start_time = time.time()
    words_with_timestamps = transcribe_audio_with_whisperx(
        video_path,
        model_name=model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    segments = group_segments_two_speakers(words_with_timestamps)
    print_speaker_segments(segments)
    add_dynamic_subtitles_to_video(video_path, words_with_timestamps, output_path, style=style)
    end_time = time.time()
    print(f"Total processing time: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    video_path = r"D:\Self\Gen ai\subtitle test\uploads\segment1.mp4"
    output_path = r"D:\Self\Gen ai\subtitle test\uploads\test_output_segment.mp4"
    process_and_save_video_with_segments(
        video_path, output_path, model_size="small", device="cuda", style="modern"
    )