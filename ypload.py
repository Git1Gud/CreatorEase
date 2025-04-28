import boto3
import os
from dotenv import load_dotenv
load_dotenv()

def upload_to_s3(file_path, bucket_name, s3_key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    s3.upload_file(file_path, bucket_name, s3_key)
    url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
    return url

# Example usage inside your loop:
bucket_name = os.getenv("BUCKET_NAME")
aws_access_key_id = os.getenv('ACCESS_KEY')
aws_secret_access_key = os.getenv('SECRET_KEY')
region_name = os.getenv('REGION_NAME')

url=upload_to_s3(
    file_path=r"D:\Self\Gen ai\subtitle test\uploads\segment2.mp4",
    bucket_name=bucket_name,
    s3_key="segment2.mp4",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

print(url)