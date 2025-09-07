from dotenv import load_dotenv
import os
import shutil
from kokoro import KPipeline
import soundfile as sf
import torch

load_dotenv()

def get_narration(text):
    """
    Obtain narration audio for the given text using Kokoro TTS model.
    """
    print(text)
    # Load Hugging Face token if needed (optional for public models)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    
    # Initialize Kokoro pipeline for American English
    pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M',lang_code='a')  # 'a' for American English
    
    # Generate audio
    generator = pipeline(
        text, voice='af_heart',  # Default voice; change as needed
        speed=1, split_pattern=r'\n+'
    )
    
    # Collect all audio segments
    all_audio = []
    for gs, ps, audio in generator:
        all_audio.append(audio)
    
    # Concatenate if multiple segments
    if all_audio:
        final_audio = torch.cat(all_audio, dim=0).numpy() if len(all_audio) > 1 else all_audio[0]
    else:
        raise Exception("No audio generated.")
    
    # Save to temporary WAV file
    temp_audio_file = f"temp_{abs(hash(text))}.wav"
    sf.write(temp_audio_file, final_audio, 24000)
    
    # Define the destination directory
    upload_dir = "uploads/audio"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Define the destination file path
    destination_path = os.path.join(upload_dir, temp_audio_file)
    
    # Move the temporary file to the destination
    shutil.move(temp_audio_file, destination_path)
        
    return destination_path

# Example usage (uncomment to test)
# audio_path = get_narration("Hello, this is a test narration.")
# print(f"Audio saved to: {audio_path}")