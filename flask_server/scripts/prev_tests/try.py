import os
import subprocess
import argparse
from pathlib import Path
import tempfile
from faster_whisper import WhisperModel

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file using ffmpeg"""
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-q:a", "0", "-map", "a", 
        "-vn", output_audio_path,
        "-y"  # Overwrite if exists
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_audio_path

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
        device = check_gpu_availability()
    
    compute_type = "float16" if device == "cuda" else "int8"
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
    
    return words_with_timestamps

def format_ass_timestamp(seconds):
    """Format seconds as H:MM:SS.cc for ASS format"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_part = seconds % 60
    centiseconds = int((seconds_part - int(seconds_part)) * 100)
    
    return f"{hours}:{minutes:02d}:{int(seconds_part):02d}.{centiseconds:02d}"

def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=width,height", 
        "-of", "csv=p=0", video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    width, height = map(int, result.stdout.strip().split(','))
    return width, height

def generate_ass_from_words(words_with_timestamps, output_path, video_width, video_height, style="modern"):
    """Convert word-level timestamps to ASS format with highlighting"""
    # Define styles based on selected style
    if style == "modern":
        text_color = "FFFFFF"
        highlight_color = "00FFFF"  # FFFF00 in RGB is yellow, but ASS uses BBGGRR
    elif style == "vibrant":
        text_color = "FFFFFF"
        highlight_color = "00AAFF"  # FFAA00 in RGB is orange
    elif style == "minimal":
        text_color = "FFFFFF"
        highlight_color = "FFFFFF"
    else:
        text_color = "FFFFFF"
        highlight_color = "00FFFF"
    
    # Calculate font size based on video height
    font_size = int(video_height * 0.055)
    
    # ASS header
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00{text_color},&H00{text_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1
Style: Highlight,Arial,{font_size},&H00{highlight_color},&H00{highlight_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    
    # Calculate word positions for windowing
    for i, word in enumerate(words_with_timestamps):
        start_time = format_ass_timestamp(word["start"])
        end_time = format_ass_timestamp(word["end"])
        
        # Create a window of words around the current word
        window_size = 8
        window_start = max(0, i - window_size // 2)
        window_end = min(len(words_with_timestamps), window_start + window_size)
        
        # Build the display line with highlighting
        line = ""
        for j in range(window_start, window_end):
            if j == i:  # Current word - highlight it
                line += "{\\1c&H" + highlight_color + "&}" + words_with_timestamps[j]["word"] + "{\\1c&H" + text_color + "&} "
            else:
                line += words_with_timestamps[j]["word"] + " "
        
        # Add background highlight/glow effect for the current word
        # The \4a is the alpha for the shadow/box (BackColour)
        line = "{\\bord2\\shad0\\4a&H60&\\4c&H" + highlight_color + "&}" + line.strip() + "{\\4a&HFF&}"
        
        # Center the text at the bottom of the screen
        events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{{\\an2}}{line}")
    
    # Write ASS file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n".join(events))
    
    return output_path

def add_subtitles_with_ffmpeg(video_path, subtitle_path, output_path, font=None):
    """Burn subtitles into video using ffmpeg"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"ass={subtitle_path}",
        "-c:a", "copy",
        "-c:v", "libx264", "-crf", "18",
        "-preset", "faster",
        output_path,
        "-y"  # Overwrite if exists
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Add captions to video using ffmpeg")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Path to output video file")
    parser.add_argument("--style", choices=["modern", "vibrant", "minimal"], default="modern", 
                        help="Caption style (modern, vibrant, minimal)")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="base",
                        help="Whisper model size")
    args = parser.parse_args()
    
    # Set output path if not specified
    if not args.output:
        input_path = Path(args.video_path)
        args.output = str(input_path.with_stem(input_path.stem + "_captioned"))
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio
        print("Extracting audio...")
        audio_path = os.path.join(temp_dir, "audio.mp3")
        extract_audio_from_video(args.video_path, audio_path)
        
        # Transcribe audio
        print(f"Transcribing with {args.model} model...")
        words_with_timestamps = transcribe_audio_with_timestamps(audio_path, args.model)
        
        # Get video dimensions
        video_width, video_height = get_video_dimensions(args.video_path)
        
        # Generate ASS subtitles
        print("Generating subtitles...")
        subtitle_path = os.path.join(temp_dir, "subtitles.ass")
        generate_ass_from_words(words_with_timestamps, subtitle_path, video_width, video_height, args.style)
        
        # Add subtitles to video
        print("Adding subtitles to video...")
        add_subtitles_with_ffmpeg(args.video_path, subtitle_path, args.output)
        
        print(f"Done! Output saved to {args.output}")

if __name__ == "__main__":
    main()