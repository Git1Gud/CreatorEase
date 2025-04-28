import os
import time
import numpy as np
import cv2
import random
import colorsys
import platform
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip
from faster_whisper import WhisperModel
import gc

def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """Extract audio from video file."""
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

def check_gpu_availability():
    """Check if CUDA is available"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def transcribe_audio_with_timestamps(audio_path, model_size="base", device=None):
    """Transcribe audio and get word-level timestamps using faster-whisper"""
    if device is None:
        device = check_gpu_availability()
    try:
        compute_type = "float16" if device == "cuda" else "int8"
        start = time.time()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        words_with_timestamps = [
            {"word": word.word.strip(), "start": word.start, "end": word.end}
            for segment in segments for word in segment.words
        ]
        del model
        gc.collect()
        print(f"Transcription completed in {time.time()-start:.2f} seconds")
        return words_with_timestamps
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

def generate_vibrant_colors(num_colors=3):
    """Generate visually pleasing colors"""
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
    """Get available system fonts based on OS"""
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
    """Load system font with fallback"""
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
    """Get text dimensions with fallback"""
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

def create_captionsai_style_frame(frame, words_to_display, current_word_idx, frame_width, frame_height, font=None, color_scheme=None):
    """Create a frame with captions.ai style subtitles"""
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
        avg_char_width, _ = get_text_dimensions("x", font)
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
    """Group words into chunks so each caption stays longer on screen."""
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
    """Add captions.ai style dynamic subtitles to video (optimized for speed and readability)"""
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
        vibrant_colors = generate_vibrant_colors(3)
        color_schemes = {
            "modern": {"text": (255, 255, 255), "highlight": (255, 230, 0), "shadow": (0, 0, 0)},
            "vibrant": {"text": (255, 255, 255), "highlight": vibrant_colors[0], "shadow": (0, 0, 0)},
            "minimal": {"text": (255, 255, 255), "highlight": (255, 255, 255), "shadow": (0, 0, 0)}
        }
        color_scheme = color_schemes.get(style, color_schemes["modern"])
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
                color_scheme=color_scheme
            )
            output_video.write(frame)
            frame_idx += 1
        video.release()
        output_video.release()
        print("Adding audio to final video...")
        add_audio_to_video(video_path, temp_output, output_path)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"Subtitles added successfully! Output: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error adding subtitles to video: {e}")
        raise

def add_audio_to_video(original_video_path: str, video_without_audio_path: str, output_path: str):
    """Add audio from original video to processed video."""
    try:
        with VideoFileClip(video_without_audio_path) as video_clip, \
             AudioFileClip(original_video_path) as audio_clip:
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            video_with_audio = video_clip.set_audio(audio_clip)
            video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None, threads=4)
    except Exception as e:
        print(f"Error adding audio to video: {e}")
        raise

def check_model_size_for_device(device):
    """Determine appropriate model size based on available memory"""
    try:
        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                free_memory_bytes = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                free_memory = free_memory_bytes / (1024**3)
                if free_memory < 2.0:
                    return "tiny"
                elif free_memory < 4.0:
                    return "base"
                elif free_memory < 6.0:
                    return "small"
                else:
                    return "medium"
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 4:
            return "tiny"
        elif available_gb < 8:
            return "base"
        else:
            return "small"
    except Exception:
        return "base"

def main(video_path: str, output_path: str, model_size: str = None, device: str = None, style: str = "modern") -> bool:
    """Main function to process video and add captions.ai style subtitles."""
    audio_path = "temp_audio.mp3"
    try:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        device = device or check_gpu_availability()
        model_size = model_size or check_model_size_for_device(device)
        print(f"Using {model_size} model on {device}")
        print("Extracting audio from video...")
        extract_audio_from_video(video_path, audio_path)
        print("Transcribing audio with faster-whisper...")
        words_with_timestamps = transcribe_audio_with_timestamps(audio_path, model_size, device)
        print(f"Adding {style} captions.ai style subtitles to video...")
        add_dynamic_subtitles_to_video(video_path, words_with_timestamps, output_path, style=style)
        print(f"Done! Output saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error in main processing: {e}")
        return False
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    video_path = "Shortened video.mp4"  # Replace with your video path
    output_path = "output_with_captions.mp4"
    start_time = time.time()
    main(video_path, output_path, style="modern")
    end_time = time.time()
    print(f"Total processing time: {end_time-start_time:.2f} seconds")