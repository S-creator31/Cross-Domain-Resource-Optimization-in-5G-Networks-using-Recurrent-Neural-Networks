# utils/aws_utils.py
# Centralizes all AWS S3 interactions.
# Both the model (.pth) and the scaler (.pkl) are stored and retrieved from S3.
# This way, whatever was used during training is exactly what runs in production.

import boto3
import os
from config import AWS_CONFIG, LOCAL_PATHS


def get_s3_client():
    """Return a boto3 S3 client for the configured region."""
    return boto3.client("s3", region_name=AWS_CONFIG["region_name"])


def download_file_from_s3(s3_key: str, local_path: str) -> bool:
    """
    Download a single file from S3.
    Returns True on success, False on failure.
    """
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3.download_file(AWS_CONFIG["bucket_name"], s3_key, local_path)
        print(f"Downloaded s3://{AWS_CONFIG['bucket_name']}/{s3_key} → {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        return False


def upload_file_to_s3(local_path: str, s3_key: str) -> bool:
    """
    Upload a single file to S3.
    Call this after training to store both best_model.pth and scaler.pkl.
    """
    s3 = get_s3_client()
    try:
        s3.upload_file(local_path, AWS_CONFIG["bucket_name"], s3_key)
        print(f"Uploaded {local_path} → s3://{AWS_CONFIG['bucket_name']}/{s3_key}")
        return True
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")
        return False


def download_model_and_scaler() -> bool:
    """
    Convenience function: download both the model weights and the scaler.
    Call this at API startup.
    """
    model_ok = download_file_from_s3(AWS_CONFIG["model_key"], LOCAL_PATHS["model"])
    scaler_ok = download_file_from_s3(AWS_CONFIG["scaler_key"], LOCAL_PATHS["scaler"])
    return model_ok and scaler_ok


def upload_model_and_scaler() -> bool:
    """
    Convenience function: upload both the model weights and the scaler after training.
    """
    model_ok = upload_file_to_s3(LOCAL_PATHS["model"], AWS_CONFIG["model_key"])
    scaler_ok = upload_file_to_s3(LOCAL_PATHS["scaler"], AWS_CONFIG["scaler_key"])
    return model_ok and scaler_ok
