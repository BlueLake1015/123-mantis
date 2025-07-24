import numpy as np
import json
import time
import secrets
import requests
from timelock import Timelock
from config import ASSETS, ASSET_EMBEDDING_DIMS # Assume a local config.py
import argparse
import bittensor as bt
from loguru import logger
import boto3
import os

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

def generate_data():
    # Generate embeddings for each asset (replace with your model outputs)
    # The order must match the order in config.ASSETS
    multi_asset_embedding = [
        np.random.uniform(-1, 1, size=ASSET_EMBEDDING_DIMS[asset]).tolist()
        for asset in ASSETS
    ]

def upload_to_r2(hotkey_address):
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://70449db619e94a3e183a3d466127a407.r2.cloudflarestorage.com",
        aws_access_key_id="015b36e900568c4907f86f8b71efcf48",
        aws_secret_access_key="804d03d190a7922a190572f83fa1d048e6062be63cb952d74ce617b139e5e189",
        region_name="auto",
    )
    bucket_name = "mantis"
    s3.upload_file(hotkey_address, bucket_name, hotkey_address)
    logger.info(f"Successfully uploaded {hotkey_address} to R2 bucket {bucket_name}.")

def generate_and_upload(config):
    # Configure your wallet and the subtensor
    try:
        wallet_name = getattr(config, 'wallet.name')
        wallet_hotkey = getattr(config, 'wallet.hotkey')
        wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        hotkey_address = wallet.hotkey.ss58_address
        
        # Generate random embeddings for each asset
        multi_asset_embedding = [
            np.random.uniform(-1, 1, size=ASSET_EMBEDDING_DIMS[asset]).tolist()
            for asset in ASSETS
        ]

        # Fetch beacon info to calculate a future round
        info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
        future_time = time.time() + 30  # Target a round ~30 seconds in the future
        target_round = int((future_time - info["genesis_time"]) // info["period"])

        # Create the plaintext by joining embeddings and the hotkey
        plaintext = f"{str(multi_asset_embedding)}:::{hotkey_address}"

        # Encrypt the plaintext for the target round
        tlock = Timelock(DRAND_PUBLIC_KEY)
        salt = secrets.token_bytes(32)
        ciphertext_hex = tlock.tle(target_round, plaintext, salt).hex()

        filename = hotkey_address 
        payload = {
            "round": target_round,
            "ciphertext": ciphertext_hex,
        }

        with open(filename, "w") as f:
            json.dump(payload, f)
        
        upload_to_r2(filename)

        # Delete the file after uploading
        if os.path.exists(filename):
            os.remove(filename)
        logger.info(f"Uploaded at round: {target_round}")

    except Exception as e:
        logger.error(f"Error: {e}")

def main(config):
    wallet_name = getattr(config, 'wallet.name')
    wallet_hotkey = getattr(config, 'wallet.hotkey')
    logger.info(f"Miner started for {wallet_name} {wallet_hotkey}")
    try:
        while True:
            generate_and_upload(config)
            time.sleep(60)
    except KeyboardInterrupt:
        logger.warning("CTRL+C detected. Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MANTIS commit tool"
    )
    parser.add_argument('--wallet.name', type=str, required=True, help='Bittensor wallet coldkey name that will be used to register')
    parser.add_argument('--wallet.hotkey', type=str, required=True, help='Bittensor wallet hotkey name that will be used to register')
    config = parser.parse_args()
    main(config)
