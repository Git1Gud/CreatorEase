import os
import time
import numpy as np
import cv2
import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip
import gc
import random
import colorsys
import platform
from pathlib import Path
from faster_whisper import WhisperModel
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from faster_whisper import WhisperModel

def process_frame(args):
    """Process a single frame with subtitles"""
    frame, words_with_timestamps, frame_idx, fps, frame_width, frame_height, font, color_scheme = args
    
    # Determine which word should be highlighted based on timestamp
    current_time = frame_idx / fps
    current_word_idx = None
    for idx, word in enumerate(words_with_timestamps):
        if word["start"] <= current_time <= word["end"]:
            current_word_idx = idx
            break
    
    return create_captionsai_style_frame(frame, words_with_timestamps, current_word_idx, frame_width, frame_height, font, color_scheme)

def add_dynamic_subtitles_parallel(video_path, words_with_timestamps, output_path, style="modern"):
    """Add captions to video using parallel processing"""
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = output_path + ".temp.mp4"
    output_video = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
    
    if not output_video.isOpened():
        raise ValueError(f"Could not create output video file: {temp_output}")

    # Define color scheme
    color_schemes = {
        "modern": {"text": (255, 255, 255), "highlight": (255, 230, 0), "shadow": (0, 0, 0)},
        "minimal": {"text": (255, 255, 255), "highlight": (255, 255, 255), "shadow": (0, 0, 0)}
    }
    color_scheme = color_schemes.get(style, color_schemes["modern"])

    # Load font
    font_size = int(frame_height * 0.05)
    font = load_font(font_size)

    # Parallel processing setup
    num_workers = mp.cpu_count() - 1  # Use all but one core
    pool = mp.Pool(num_workers)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append((frame, words_with_timestamps, frame_idx, fps, frame_width, frame_height, font, color_scheme))
        frame_idx += 1

    # Process frames in parallel
    processed_frames = pool.map(process_frame, frames)

    # Write processed frames to video
    for processed_frame in processed_frames:
        output_video.write(processed_frame)

    # Clean up
    video.release()
    output_video.release()
    pool.close()
    pool.join()

    # Replace original audio
    video_clip = VideoFileClip(video_path)
    video_with_audio = VideoFileClip(temp_output).set_audio(video_clip.audio)
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None)

    # Remove temp file
    os.remove(temp_output)
    print("Subtitles added successfully!")


def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file"""
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
        video.close()
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

def check_gpu_availability():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"

def transcribe_audio_with_timestamps(audio_path, model_size="base", device=None):
    """Transcribe audio and get word-level timestamps using faster-whisper"""
    if device is None:
        device = "cuda" if "cuda" in str(check_gpu_availability()) else "cpu"
    
    try:
        compute_type = "float16" if device == "cuda" else "int8"
        start=time.time()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        
        words_with_timestamps = []
        for segment in segments:
            for word in segment.words:
                words_with_timestamps.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
        
        del model
        gc.collect()
        end=time.time()
        print(f"Transcription completed in {end-start} seconds")
        return words_with_timestamps
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

def generate_vibrant_colors(num_colors=5):
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
    font_paths = []
    
    if system == "Windows":
        font_dir = Path("C:/Windows/Fonts")
        font_paths = [font_dir / "Arial.ttf", font_dir / "arialbd.ttf"]
    elif system == "Darwin":  # macOS
        font_paths = [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        ]
    else:  # Linux
        font_paths = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        ]
    
    return font_paths

def load_font(size, bold=True):
    """Load system font with fallback"""
    font_paths = get_system_fonts()
    
    if bold:
        font_paths = [p for p in font_paths if "bold" in p.name.lower() or "bd" in p.name.lower()] + [p for p in font_paths if "bold" not in p.name.lower() and "bd" not in p.name.lower()]
    
    for font_path in font_paths:
        try:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError):
            continue
    
    return ImageFont.load_default()

def get_text_dimensions(text, font):
    """Get text dimensions with fallback"""
    text.strip()
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
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        
        # Create transparent overlay
        overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Use provided color scheme or default
        if color_scheme is None:
            text_color = (255, 255, 255, 255)
            highlight_color = (255, 230, 0, 255)
            shadow_color = (0, 0, 0, 180)
        else:
            text_color = (*color_scheme["text"], 255)
            highlight_color = (*color_scheme["highlight"], 255)
            shadow_color = (*color_scheme["shadow"], 180)
        
        # Combine words into one string
        full_text = " ".join([word["word"] for word in words_to_display])
        
        # Calculate text wrapping
        max_width = int(frame_width * 0.8)
        wrapped_lines = []
        
        # Calculate average character width
        avg_char_width, _ = get_text_dimensions("x", font)
        
        # Simple word wrapping
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
        
        # Calculate text block height
        _, line_height = get_text_dimensions("Ay", font)
        text_block_height = len(wrapped_lines) * line_height * 1.2
        
        # Position text block
        bottom_padding = int(frame_height * 0.1)
        y_position = frame_height - bottom_padding - text_block_height
        
        # Track current word position
        global_word_idx = 0
        
        # Draw each line
        for line_idx, (line_words, line_width) in enumerate(wrapped_lines):
            # Center the line
            x_position = (frame_width - line_width) // 2
            line_y = y_position + line_idx * line_height * 1.2
            
            # Draw shadow for the line
            joined_line = " ".join(line_words)
            for offset_x, offset_y in [(2, 2), (-2, 2), (2, -2), (-2, -2)]:
                draw.text(
                    (x_position + offset_x, line_y + offset_y),
                    joined_line,
                    font=font,
                    fill=shadow_color
                )
            
            # Draw each word
            for word in line_words:
                word_with_space = word + " "
                word_width, _ = get_text_dimensions(word_with_space, font)
                
                # Determine if this is the current word
                is_current = (global_word_idx == current_word_idx)
                
                # Choose color based on whether this is the current word
                color = highlight_color if is_current else text_color
                
                # Add highlight effect for current word
                if is_current:
                    glow_padding = 3
                    word_box = [
                        x_position - glow_padding,
                        line_y - glow_padding,
                        x_position + word_width + glow_padding,
                        line_y + line_height + glow_padding
                    ]
                    
                    # Draw rounded rectangle behind current word
                    try:
                        draw.rounded_rectangle(
                            word_box,
                            radius=8,
                            fill=(highlight_color[0], highlight_color[1], highlight_color[2], 60)
                        )
                    except AttributeError:
                        # Fallback for older PIL versions
                        draw.rectangle(
                            word_box,
                            fill=(highlight_color[0], highlight_color[1], highlight_color[2], 60)
                        )
                
                # Draw the word
                draw.text((x_position, line_y), word_with_space, font=font, fill=color)
                
                # Move to next word position
                x_position += word_width
                global_word_idx += 1
        
        # Composite overlay with original frame
        frame_with_overlay = Image.alpha_composite(pil_frame.convert('RGBA'), overlay)
        result = np.array(frame_with_overlay.convert('RGB'))
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        print(f"Error creating captioned frame: {e}")
        return frame

def add_dynamic_subtitles_to_video(video_path, words_with_timestamps, output_path, batch_size=10, style="modern"):
    """Add captions.ai style dynamic subtitles to video"""
    try:
        # Open video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        temp_output = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(
            temp_output, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )
        
        if not output_video.isOpened():
            raise ValueError(f"Could not create output video file: {temp_output}")
        
        # Generate color schemes
        vibrant_colors = generate_vibrant_colors(3)
        color_schemes = {
            "modern": {"text": (255, 255, 255), "highlight": (255, 230, 0), "shadow": (0, 0, 0)},
            "vibrant": {"text": (255, 255, 255), "highlight": vibrant_colors[0], "shadow": (0, 0, 0)},
            "minimal": {"text": (255, 255, 255), "highlight": (255, 255, 255), "shadow": (0, 0, 0)}
        }
        
        # Select color scheme
        color_scheme = color_schemes.get(style, color_schemes["modern"])
        
        # Set up font
        font_size = int(frame_height * 0.055)
        font = load_font(font_size, bold=True)
        
        # Process frames in batches
        frame_idx = 0
        while frame_idx < frame_count:
            batch_frames = []
            batch_times = []
            
            # Read a batch of frames
            for _ in range(min(batch_size, frame_count - frame_idx)):
                ret, frame = video.read()
                if not ret:
                    break
                
                current_time = frame_idx / fps
                batch_frames.append(frame)
                batch_times.append(current_time)
                frame_idx += 1
            
            # Process each frame in the batch
            for frame, current_time in zip(batch_frames, batch_times):
                # Find the current word based on timestamp
                current_word_idx = next(
                    (i for i, word in enumerate(words_with_timestamps) 
                     if word["start"] <= current_time <= word["end"]), 
                    None
                )
                
                # Determine context window for display
                if current_word_idx is not None:
                    window_size = 8
                    window_start = max(0, current_word_idx - window_size // 2)
                    window_end = min(len(words_with_timestamps), window_start + window_size)
                    words_to_display = words_with_timestamps[window_start:window_end]
                    
                    # Calculate relative position of current word
                    relative_current_idx = current_word_idx - window_start
                    
                    # Add subtitles to frame
                    frame = create_captionsai_style_frame(
                        frame, 
                        words_to_display, 
                        relative_current_idx, 
                        frame_width, 
                        frame_height,
                        font=font,
                        color_scheme=color_scheme
                    )
                
                # Write the frame
                output_video.write(frame)
            
            # Clear batch from memory
            del batch_frames
            del batch_times
            gc.collect()
        
        # Release resources
        video.release()
        output_video.release()
        
        # Add audio back to video
        print("Adding audio to final video...")
        add_audio_to_video(video_path, temp_output, output_path)
        
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        return output_path
    
    except Exception as e:
        print(f"Error adding subtitles to video: {e}")
        raise

def add_audio_to_video(original_video_path, video_without_audio_path, output_path):
    """Add audio from original video to processed video"""
    try:
        video_clip = VideoFileClip(video_without_audio_path)
        audio_clip = AudioFileClip(original_video_path)
        
        # Ensure audio duration matches video
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        
        video_with_audio = video_clip.set_audio(audio_clip)
        video_with_audio.write_videofile(output_path, codec='libx264', 
                                        audio_codec='aac', verbose=False, 
                                        logger=None, threads=4)
        
        # Release memory
        video_clip.close()
        audio_clip.close()
        video_with_audio.close()
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
            
        # For CPU or if CUDA check failed
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

def main(video_path, output_path, model_size=None, device=None, style="modern", batch_size=10):
    """Main function to process video and add captions.ai style subtitles"""
    try:
        # Validate input
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Check device
        if device is None:
            device = check_gpu_availability()
            
        # Determine model size
        if model_size is None:
            model_size = check_model_size_for_device(device)
        
        print(f"Using {model_size} model on {device}")
        
        # Extract audio
        print("Extracting audio from video...")
        audio_path = "temp_audio.mp3"
        extract_audio_from_video(video_path, audio_path)
        
        # Transcribe
        print("Transcribing audio with faster-whisper...")
        words_with_timestamps = transcribe_audio_with_timestamps(audio_path, model_size, device)
        
        # Add subtitles
        print(f"Adding {style} captions.ai style subtitles to video...")
        add_dynamic_subtitles_to_video(video_path, words_with_timestamps, output_path, batch_size=batch_size, style=style)
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"Done! Output saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error in main processing: {e}")
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
        return False

# Example usage
if __name__ == "__main__":
    video_path = "Shortened video.mp4"  # Replace with your video path
    output_path = "output_with_captions.mp4"
    start_time=time.time()
    main(video_path, output_path, style="modern", batch_size=8)
    end_time=time.time()
    print(f"Total time taken: {end_time - start_time} seconds")