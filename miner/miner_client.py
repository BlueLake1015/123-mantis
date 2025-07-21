# miner_client.py
import asyncio
from  upload import upload_to_r2
import logging
import json
import random
import secrets
import time
import requests
import os
from dotenv import load_dotenv
import aiohttp
import pandas as pd
import ta
import numpy as np

from timelock import Timelock

from datetime import datetime
from typing import List

from config import ASSET_EMBEDDING_DIMS, ASSETS, TASK_INTERVAL

from utils import get_asset_prices, fetch_recent_prices, compute_garch_sigma, denoise_wavelet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("miner")


load_dotenv()

FILENAME = os.getenv("HOTKEY")
LOCK_TIME_SECONDS = 30
TIME_INTERVAL = int(os.getenv("TIME_INTERVAL"))

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

async def generate_embedding() -> List[float]:
    async with aiohttp.ClientSession() as session:
        prices = await get_asset_prices(session)
        live_price = prices.get("BTC")

    history = fetch_recent_prices("BTC", TIME_INTERVAL)
    # Ensure it's a Series:
    if isinstance(history, pd.Series):
        df = history.to_frame(name='price')
    else:
        df = history  # already a DataFrame
    df['denoised'] = denoise_wavelet(df['price'])

    # Price features: ensure bfill on series level
    df['return_1']      = df['denoised'].pct_change(1).bfill()
    df['return_5']      = df['denoised'].pct_change(5).bfill()
    df['return_15']     = df['denoised'].pct_change(15).bfill()
    df['volatility5']   = df['denoised'].rolling(5).std().bfill()
    df['atr']           = ta.volatility.average_true_range(df['price'], df['price'], df['price'], window=14).bfill()
    df['bb_h']          = ta.volatility.bollinger_hband(df['price'], window=20).bfill()
    df['bb_l']          = ta.volatility.bollinger_lband(df['price'], window=20).bfill()
    df['ma_7']          = df['price'].rolling(7).mean().bfill()
    df['ma_21']         = df['price'].rolling(21).mean().bfill()

    # ARCH with no rescale
    returns = df['denoised'].pct_change().dropna()
    df['garch_sigma'] = compute_garch_sigma(returns).reindex(df.index).bfill()


    # Time encodings
    now = datetime.utcnow()
    df['dow_sin'] = np.sin(2*np.pi*now.weekday()/7)
    df['dow_cos'] = np.cos(2*np.pi*now.weekday()/7)

    # Build embedding vector
    feat_cols = ['return_1','return_5','return_15','volatility5',
                 'atr','bb_h','bb_l','ma_7','ma_21','garch_sigma',
                 'dow_sin','dow_cos']
    embedding = df[feat_cols].iloc[-1].tolist()


    # Live log return
    if live_price is not None:
        last = df['denoised'].iloc[-1]
        embedding.append(np.log(live_price / last + 1e-8))

    # Returns sequence window
    seq = returns.values[-32:]
    if len(seq) < 32:
        # seq = np.pad(seq, (32-len(seq),0))
        pad = np.random.uniform(-1e-4, 1e-4, size=(32 - len(seq),))
        seq = np.concatenate([pad, seq])

    embedding.extend(seq.tolist())

    # Pad/truncate & normalize
    embedding = np.array(embedding, dtype=float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=1, neginf=-1)
    norm = np.max(np.abs(embedding)) or 1.0
    embedding = (embedding / norm).tolist()
    # print(type(embedding))
    # THEN pad if needed:
    # if len(embedding) < 100:
    #     embedding.extend(np.random.uniform(-1e-4,1e-4,100 - len(embedding)).tolist())
    
    return embedding

async def generate_asset_embedding(asset_name: str, live_price: float) -> List[float]:
    try:
        df = fetch_recent_prices(asset_name, TIME_INTERVAL)
        # Ensure it's a Series:
        if isinstance(df, pd.Series):
            df = df.to_frame(name='price')
        else:
            df = df  # already a DataFrame
        if asset_name == "BTC":
            df['denoised'] = denoise_wavelet(df['price'])

            # Price features: ensure bfill on series level
            df['return_1']      = df['denoised'].pct_change(1).bfill()
            df['return_5']      = df['denoised'].pct_change(5).bfill()
            df['return_15']     = df['denoised'].pct_change(15).bfill()
            df['volatility5']   = df['denoised'].rolling(5).std().bfill()
            df['atr']           = ta.volatility.average_true_range(df['price'], df['price'], df['price'], window=14).bfill()
            df['bb_h']          = ta.volatility.bollinger_hband(df['price'], window=20).bfill()
            df['bb_l']          = ta.volatility.bollinger_lband(df['price'], window=20).bfill()
            df['ma_7']          = df['price'].rolling(7).mean().bfill()
            df['ma_21']         = df['price'].rolling(21).mean().bfill()

            # ARCH with no rescale
            returns = df['denoised'].pct_change().dropna()
            df['garch_sigma'] = compute_garch_sigma(returns).reindex(df.index).bfill()


            # Time encodings
            now = datetime.utcnow()
            df['dow_sin'] = np.sin(2*np.pi*now.weekday()/7)
            df['dow_cos'] = np.cos(2*np.pi*now.weekday()/7)

            # Build embedding vector
            feat_cols = ['return_1','return_5','return_15','volatility5',
                        'atr','bb_h','bb_l','ma_7','ma_21','garch_sigma',
                        'dow_sin','dow_cos']
            embedding = df[feat_cols].iloc[-1].tolist()


            # Live log return
            if live_price is not None:
                last = df['denoised'].iloc[-1]
                embedding.append(np.log(live_price / last + 1e-8))

            # Returns sequence window
            seq = returns.values[-32:]
            if len(seq) < 32:
                # seq = np.pad(seq, (32-len(seq),0))
                pad = np.random.uniform(-1e-4, 1e-4, size=(32 - len(seq),))
                seq = np.concatenate([pad, seq])

            embedding.extend(seq.tolist())

            # Pad/truncate & normalize
            embedding = np.array(embedding, dtype=float)
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1, neginf=-1)
            norm = np.max(np.abs(embedding)) or 1.0
            embedding = (embedding / norm).tolist()
            # THEN pad if needed:
            if len(embedding) < 100:
                embedding.extend(np.random.uniform(-1e-4,1e-4,100 - len(embedding)).tolist())
            
            return embedding
        else:
            df['return'] = df['price'].pct_change()
            return_15 = df['price'].pct_change(15).iloc[-1]
            vol_15 = df['return'].rolling(15).std().iloc[-1]
            
            # Ensure valid floats and plain list
            vec = [
                float(np.nan_to_num(return_15).item()),
                float(np.nan_to_num(vol_15).item())
            ]
            return vec

    except Exception as e:
        logger.warning(f"Embedding generation failed for {asset_name}: {e}")
        return [0.0] * ASSET_EMBEDDING_DIMS[asset_name]

async def generate_multi_asset_embeddings() -> List[List[float]]:
    embeddings: List[List[float]] = []
    async with aiohttp.ClientSession() as session:
        prices = await get_asset_prices(session)

    for asset in ASSETS:
        live_price = prices.get(asset)
        embedding = await generate_asset_embedding(asset, live_price)
        assert len(embedding) == ASSET_EMBEDDING_DIMS[asset]
        embeddings.append(embedding)
    return embeddings

async def generate_and_encrypt(hotkey: str, filename: str | None = None):
    if filename is None:
        filename = hotkey
    logger.info("--- Starting Payload Generation ---")

    embeddings: List[List[float]] = await generate_multi_asset_embeddings()

    # embeddings = await generate_embedding()
    logger.info(f"Generated embedding vector length: {len(embeddings)}")
    logger.info(f"Generated embedding vector: {embeddings}")

    try:
        info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
        future_time = time.time() + LOCK_TIME_SECONDS
        round_num = int((future_time - info["genesis_time"]) // info["period"])
    except Exception as e:
        print(f"Error fetching Drand info: {e}")
        return None

    try:
        tlock = Timelock(DRAND_PUBLIC_KEY)
        plaintext = f"{str(embeddings)}:::{hotkey}"
        salt = secrets.token_bytes(32)
        
        ciphertext_hex = tlock.tle(round_num, plaintext, salt).hex()
    except Exception as e:
        print(f"Error during encryption: {e}")
        return None

    payload_dict = {"round": round_num, "ciphertext": ciphertext_hex}
    
    if filename:
        try:
            with open(filename, "w") as f:
                json.dump(payload_dict, f, indent=2)
            print(f"Encrypted payload saved to: {filename}")
        except Exception as e:
            print(f"Error saving to file {filename}: {e}")

    return payload_dict


async def main():
    logger.info("🌱 Miner server started... awaiting DRAND round")

    while True:
        payload = await generate_and_encrypt(os.getenv("HOTKEY"), FILENAME)
        logger.info("🔐 Encrypted payload: %s", payload["ciphertext"][:8])
        upload_to_r2()
        logger.info("🔐 payload: %s ...", payload["ciphertext"][:8])


        await asyncio.sleep(TASK_INTERVAL)  # wait for next round

if __name__ == "__main__":
    asyncio.run(main())
