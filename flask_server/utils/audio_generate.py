from dotenv import load_dotenv
import os
import shutil
from kokoro import KPipeline
import soundfile as sf
import torch
import numpy as np

load_dotenv()

def get_narration(text):
    """
    Obtain narration audio for the given text using Kokoro TTS model.
    Also returns approximate word-level timestamps for dynamic captions.
    Returns: (audio_path, words_with_timestamps)
    """
    print(text)
    # Load Hugging Face token if needed (optional for public models)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    # Initialize Kokoro pipeline for American English
    pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a')  # 'a' for American English

    # Generate audio
    generator = pipeline(
        text,
        voice='af_heart',  # Default voice; change as needed
        speed=1,
        split_pattern=r'\n+'
    )

    # Collect all audio segments and build naive word-level timestamps
    sr = 24000
    audio_chunks = []
    words_with_timestamps = []
    t_cursor = 0.0

    for gs, ps, audio in generator:
        # Normalize audio to numpy array
        if isinstance(audio, np.ndarray):
            arr = audio
        elif torch.is_tensor(audio):
            arr = audio.detach().cpu().numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)

        audio_chunks.append(arr)
        seg_dur = len(arr) / sr

        # Split the provided text for this chunk into words and distribute time evenly
        chunk_text = str(gs).strip()
        words = [w for w in chunk_text.split() if w]
        if not words:
            # Fallback: treat whole chunk as one word span
            words = [chunk_text or "..."]

        per_word = seg_dur / len(words)
        for w in words:
            start = t_cursor
            end = start + per_word
            words_with_timestamps.append({"word": w, "start": start, "end": end})
            t_cursor = end

    if not audio_chunks:
        raise Exception("No audio generated.")

    # Concatenate audio
    final_audio = audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks)

    # Save to temporary WAV file
    temp_audio_file = f"temp_{abs(hash(text))}.wav"
    sf.write(temp_audio_file, final_audio, sr)

    # Define the destination directory
    upload_dir = "uploads/audio"

    # Create the directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Define the destination file path
    destination_path = os.path.join(upload_dir, temp_audio_file)

    # Move the temporary file to the destination
    shutil.move(temp_audio_file, destination_path)

    return destination_path, words_with_timestamps

# Example usage (uncomment to test)
# audio_path = get_narration("Hello, this is a test narration.")
# print(f"Audio saved to: {audio_path}")