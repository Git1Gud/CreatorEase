import boto3

def upload_to_s3(file_path, bucket_name, s3_key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    s3.upload_file(file_path, bucket_name, s3_key, ExtraArgs={'ACL': 'public-read'})
    url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
    return url
