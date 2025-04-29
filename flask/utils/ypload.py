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
    s3.upload_file(file_path, bucket_name, s3_key,ExtraArgs={'ContentType': 'video/mp4'})
    url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"
    return url

# Example usage inside your loop:
bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
aws_access_key_id = os.getenv('AWS_S3_ACCESS_KEY')
aws_secret_access_key = os.getenv('AWS_S3_SECRET_ACCESS_KEY')
region_name = os.getenv('ASW_S3_REGION_NAME')

url=upload_to_s3(
    file_path=r"C:\Users\Admin\Desktop\CreatorEase\uploads\segments\segment2.mp4",
    bucket_name=bucket_name,
    s3_key="segment2.mp4",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

print(url)