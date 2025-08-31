from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
import requests
import shutil  # Import the shutil module

load_dotenv()

def get_narration(text):
    """
    Obtain narration audio for the given text using Eleven Labs API.
    """
    # Check both possible environment variable names
    api_key = os.environ.get("ELEVEN_LAB_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise Exception("Please set the ELEVEN_LAB_API_KEY environment variable.")
    
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Your Eleven Labs voice ID
    
    # Note the updated URL format with voice_id in the path
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1"  # Adding the model ID which is often required
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error from Eleven Labs API: {response.text}")

    temp_audio_file = f"temp_{abs(hash(text))}.mp3"
    with open(temp_audio_file, "wb") as f:
        f.write(response.content)
    
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

# audio_path = get_narration("Hello, this is a test narration.")
# print(f"Audio saved to: {audio_path}")