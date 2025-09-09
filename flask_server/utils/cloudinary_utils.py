import os
import cloudinary
from cloudinary.uploader import upload
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary with environment variables
cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
api_key = os.getenv('CLOUDINARY_API_KEY')
api_secret = os.getenv('CLOUDINARY_API_SECRET')

if not all([cloud_name, api_key, api_secret]):
    raise ValueError("Cloudinary credentials not found. Please set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET environment variables.")

cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret
)

def upload_video(file_path, folder='videos', public_id=None, **kwargs):
    """
    Upload a video file to Cloudinary.

    Args:
        file_path (str): Path to the video file to upload.
        folder (str): Folder in Cloudinary to upload to (default: 'videos').
        public_id (str): Optional public ID for the uploaded file.
        **kwargs: Additional parameters for the upload (e.g., transformation options).

    Returns:
        dict: Upload result containing 'url', 'public_id', 'secure_url', etc.

    Raises:
        Exception: If upload fails.
    """
    try:
        result = cloudinary.uploader.upload(
            file_path,
            resource_type='video',
            folder=folder,
            public_id=public_id,
            **kwargs
        )
        print(f"Video uploaded successfully: {result['url']}")
        return result['url']
    except Exception as e:
        print(f"Error uploading video: {e}")
        raise

def get_video_url(public_id, **kwargs):
    """
    Generate a URL for a video stored in Cloudinary.

    Args:
        public_id (str): Public ID of the video.
        **kwargs: Additional parameters (e.g., format, quality).

    Returns:
        str: The video URL.
    """
    return cloudinary.utils.cloudinary_url(public_id, resource_type='video', **kwargs)[0]

