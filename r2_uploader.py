"""
Cloudflare R2 uploader for MANTIS miner payloads
"""

import json
import logging
import os
from typing import Dict

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class R2Uploader:
    """Handles uploading payloads to Cloudflare R2 bucket"""
    
    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID", "70449db619e94a3e183a3d466127a407")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID", "015b36e900568c4907f86f8b71efcf48")
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY", "804d03d190a7922a190572f83fa1d048e6062be63cb952d74ce617b139e5e189")
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "mantis")
        self.public_url_base = os.getenv("R2_PUBLIC_URL_BASE", "https://pub-97dfad3ad4c74c3a94f508d81e57681d.r2.dev")
        
        if not all([self.account_id, self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError("Missing required R2 configuration. Check your .env file.")
        
        # Initialize S3 client for R2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name='auto',
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3}
            )
        )
        
        logger.info(f"R2Uploader initialized for bucket: {self.bucket_name}")
    
    async def upload_json(self, data: Dict, filename: str) -> str:
        """
        Upload JSON data to R2 bucket and return public URL
        
        Args:
            data: Dictionary to upload as JSON
            filename: Name of the file (should be hotkey)
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data, indent=2)
            
            # Upload to R2
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=json_data.encode('utf-8'),
                ContentType='application/json',
                CacheControl='no-cache'
            )
            
            # Construct public URL
            if self.public_url_base:
                public_url = f"{self.public_url_base.rstrip('/')}/{filename}"
            else:
                public_url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{filename}"
            
            logger.info(f"Successfully uploaded {filename} to R2")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload {filename} to R2: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test R2 connection by listing bucket contents"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            logger.info("R2 connection test successful")
            return True
        except Exception as e:
            logger.error(f"R2 connection test failed: {e}")
            return False
