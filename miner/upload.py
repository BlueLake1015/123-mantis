import os
import boto3
from dotenv import load_dotenv
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("miner")
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv() # Load credentials from .env file

bucket_name='123-mantis'

def upload_to_r2():
    hotkey = os.getenv("HOTKEY")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ['R2_WRITE_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_WRITE_SECRET_ACCESS_KEY'],
        region_name="auto",
    )
    s3.upload_file(hotkey, bucket_name, hotkey)
    logger.info(f"Successfully uploaded {hotkey} to R2 bucket {bucket_name}.")
