"""
MANTIS Competitive Miner
========================

A comprehensive miner for Bittensor subnet 123 (MANTIS) that generates
high-scoring multi-asset embeddings optimized for the validator's 
counterfactual scoring system.

Features:
- Advanced BTC feature generation (100 dimensions)
- Multi-asset support with optimized 2D features
- Timelock encryption with Drand
- R2 bucket integration
- Continuous operation with error recovery
- Competitive feature engineering strategies
"""

import argparse
import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from scipy import stats
from sklearn.preprocessing import StandardScaler
from timelock import Timelock

from config import ASSETS, ASSET_EMBEDDING_DIMS, SAMPLE_STEP
from r2_uploader import R2Uploader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Drand configuration
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class FeatureEngine:
    """Advanced feature engineering for multi-asset predictions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_cache = {}
        self.feature_history = {}
        
    def fetch_market_data(self, symbol: str, period: str = "5d", interval: str = "1m") -> pd.DataFrame:
        """Fetch market data with caching"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        try:
            if cache_key in self.price_cache:
                cached_data, timestamp = self.price_cache[cache_key]
                if time.time() - timestamp < 60:  # Cache for 1 minute
                    return cached_data
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            self.price_cache[cache_key] = (data, time.time())
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators optimized for 1-hour prediction"""
        if data.empty or len(data) < 20:
            return {}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        indicators = {}
        
        try:
            # Momentum indicators (optimized for 1-hour horizon)
            indicators['rsi_14'] = self._rsi(close, 14)
            indicators['rsi_7'] = self._rsi(close, 7)  # Shorter period for hourly
            indicators['momentum_10'] = (close.iloc[-1] / close.iloc[-11] - 1) if len(close) > 10 else 0
            indicators['momentum_30'] = (close.iloc[-1] / close.iloc[-31] - 1) if len(close) > 30 else 0
            
            # MACD with shorter periods for hourly prediction
            indicators['macd'], indicators['macd_signal'] = self._macd(close, 8, 17, 9)
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._bollinger_bands(close, 20, 2)
            indicators['bb_position'] = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
            
            # Volume indicators
            indicators['volume_sma_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if len(volume) > 20 else 1
            indicators['volume_momentum'] = (volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1] - 1) if len(volume) > 20 else 0
            
            # Volatility measures
            returns = close.pct_change().dropna()
            if len(returns) > 20:
                indicators['volatility_20'] = returns.rolling(20).std().iloc[-1]
                indicators['volatility_regime'] = 1 if indicators['volatility_20'] > returns.rolling(60).std().iloc[-1] else -1
            
            # Support/Resistance
            indicators['distance_to_high'] = (high.rolling(20).max().iloc[-1] - close.iloc[-1]) / close.iloc[-1]
            indicators['distance_to_low'] = (close.iloc[-1] - low.rolling(20).min().iloc[-1]) / close.iloc[-1]
            
            # Price action patterns
            indicators['higher_highs'] = self._count_higher_highs(high, 10)
            indicators['lower_lows'] = self._count_lower_lows(low, 10)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return indicators
    
    def calculate_cross_asset_features(self, btc_data: pd.DataFrame, other_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate cross-asset correlation and divergence features"""
        features = {}
        
        if btc_data.empty:
            return features
            
        btc_returns = btc_data['Close'].pct_change().dropna()
        
        try:
            # BTC dominance proxy
            if 'ETH-USD' in other_data and not other_data['ETH-USD'].empty:
                eth_returns = other_data['ETH-USD']['Close'].pct_change().dropna()
                min_len = min(len(btc_returns), len(eth_returns))
                if min_len > 10:
                    btc_ret = btc_returns.iloc[-min_len:]
                    eth_ret = eth_returns.iloc[-min_len:]
                    correlation = btc_ret.corr(eth_ret)
                    features['btc_eth_correlation'] = correlation if not np.isnan(correlation) else 0
                    features['btc_eth_divergence'] = btc_ret.iloc[-1] - eth_ret.iloc[-1]
            
            # Risk-on/Risk-off sentiment (using SPY as proxy)
            spy_data = self.fetch_market_data('SPY', period='2d', interval='1m')
            if not spy_data.empty:
                spy_returns = spy_data['Close'].pct_change().dropna()
                min_len = min(len(btc_returns), len(spy_returns))
                if min_len > 10:
                    btc_ret = btc_returns.iloc[-min_len:]
                    spy_ret = spy_returns.iloc[-min_len:]
                    correlation = btc_ret.corr(spy_ret)
                    features['btc_spy_correlation'] = correlation if not np.isnan(correlation) else 0
                    features['risk_sentiment'] = spy_ret.rolling(20).mean().iloc[-1] if len(spy_ret) > 20 else 0
            
            # Dollar strength impact
            dxy_data = self.fetch_market_data('DX-Y.NYB', period='2d', interval='1m')
            if not dxy_data.empty:
                dxy_returns = dxy_data['Close'].pct_change().dropna()
                min_len = min(len(btc_returns), len(dxy_returns))
                if min_len > 10:
                    btc_ret = btc_returns.iloc[-min_len:]
                    dxy_ret = dxy_returns.iloc[-min_len:]
                    correlation = btc_ret.corr(dxy_ret)
                    features['btc_dxy_correlation'] = correlation if not np.isnan(correlation) else 0
                    
        except Exception as e:
            logger.error(f"Error calculating cross-asset features: {e}")
            
        return features
    
    def calculate_time_based_features(self) -> Dict[str, float]:
        """Calculate time-based features for session and cycle analysis"""
        now = datetime.utcnow()
        features = {}
        
        # Trading session indicators
        hour = now.hour
        features['asian_session'] = 1 if 0 <= hour < 8 else 0
        features['london_session'] = 1 if 8 <= hour < 16 else 0
        features['ny_session'] = 1 if 13 <= hour < 21 else 0
        features['overlap_session'] = 1 if 13 <= hour < 16 else 0
        
        # Day of week effects
        weekday = now.weekday()
        features['monday_effect'] = 1 if weekday == 0 else 0
        features['friday_effect'] = 1 if weekday == 4 else 0
        features['weekend_approach'] = 1 if weekday >= 4 else 0
        
        # Hour of day cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        return features
    
    def calculate_regime_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market regime indicators"""
        features = {}
        
        if data.empty or len(data) < 50:
            return features
            
        close = data['Close']
        returns = close.pct_change().dropna()
        
        try:
            # Trend regime
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            features['trend_regime'] = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            
            # Volatility regime
            vol_short = returns.rolling(10).std()
            vol_long = returns.rolling(30).std()
            features['vol_regime'] = 1 if vol_short.iloc[-1] > vol_long.iloc[-1] else -1
            
            # Mean reversion vs momentum regime
            autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1)).iloc[-1]
            features['momentum_regime'] = 1 if autocorr > 0 else -1
            
            # Market stress indicator
            if len(returns) > 20:
                recent_vol = returns.rolling(5).std().iloc[-1]
                normal_vol = returns.rolling(20).std().iloc[-1]
                features['stress_indicator'] = recent_vol / normal_vol if normal_vol > 0 else 1
                
        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
            
        return features
    
    def generate_btc_features(self) -> List[float]:
        """Generate comprehensive 100-dimensional BTC feature vector"""
        logger.info("Generating BTC features...")
        
        # Fetch BTC data
        btc_data = self.fetch_market_data('BTC-USD', period='5d', interval='1m')
        if btc_data.empty:
            logger.warning("No BTC data available, using zero vector")
            return [0.0] * 100
        
        # Fetch related asset data for cross-asset features
        other_data = {
            'ETH-USD': self.fetch_market_data('ETH-USD', period='2d', interval='1m'),
            'SPY': self.fetch_market_data('SPY', period='2d', interval='1m'),
        }
        
        features = []
        
        # Technical indicators (30 features)
        tech_indicators = self.calculate_technical_indicators(btc_data)
        tech_features = [
            tech_indicators.get('rsi_14', 0.5),
            tech_indicators.get('rsi_7', 0.5),
            tech_indicators.get('momentum_10', 0),
            tech_indicators.get('momentum_30', 0),
            tech_indicators.get('macd', 0),
            tech_indicators.get('macd_signal', 0),
            tech_indicators.get('macd_histogram', 0),
            tech_indicators.get('bb_position', 0.5),
            tech_indicators.get('bb_width', 0),
            tech_indicators.get('volume_sma_ratio', 1),
            tech_indicators.get('volume_momentum', 0),
            tech_indicators.get('volatility_20', 0),
            tech_indicators.get('volatility_regime', 0),
            tech_indicators.get('distance_to_high', 0),
            tech_indicators.get('distance_to_low', 0),
            tech_indicators.get('higher_highs', 0),
            tech_indicators.get('lower_lows', 0),
        ]
        
        # Pad or truncate to 30 features
        tech_features = (tech_features + [0] * 30)[:30]
        features.extend(tech_features)
        
        # Cross-asset features (20 features)
        cross_asset = self.calculate_cross_asset_features(btc_data, other_data)
        cross_features = [
            cross_asset.get('btc_eth_correlation', 0),
            cross_asset.get('btc_eth_divergence', 0),
            cross_asset.get('btc_spy_correlation', 0),
            cross_asset.get('risk_sentiment', 0),
            cross_asset.get('btc_dxy_correlation', 0),
        ]
        cross_features = (cross_features + [0] * 20)[:20]
        features.extend(cross_features)
        
        # Time-based features (20 features)
        time_features_dict = self.calculate_time_based_features()
        time_features = [
            time_features_dict.get('asian_session', 0),
            time_features_dict.get('london_session', 0),
            time_features_dict.get('ny_session', 0),
            time_features_dict.get('overlap_session', 0),
            time_features_dict.get('monday_effect', 0),
            time_features_dict.get('friday_effect', 0),
            time_features_dict.get('weekend_approach', 0),
            time_features_dict.get('hour_sin', 0),
            time_features_dict.get('hour_cos', 0),
        ]
        time_features = (time_features + [0] * 20)[:20]
        features.extend(time_features)
        
        # Regime features (15 features)
        regime_features_dict = self.calculate_regime_features(btc_data)
        regime_features = [
            regime_features_dict.get('trend_regime', 0),
            regime_features_dict.get('vol_regime', 0),
            regime_features_dict.get('momentum_regime', 0),
            regime_features_dict.get('stress_indicator', 1),
        ]
        regime_features = (regime_features + [0] * 15)[:15]
        features.extend(regime_features)
        
        # Advanced momentum features (15 features)
        if not btc_data.empty:
            close = btc_data['Close']
            advanced_features = []
            
            # Multiple timeframe momentum
            for period in [5, 15, 30, 60]:
                if len(close) > period:
                    momentum = (close.iloc[-1] / close.iloc[-period-1] - 1)
                    advanced_features.append(momentum)
                else:
                    advanced_features.append(0)
            
            # Momentum acceleration
            if len(close) > 10:
                mom_5 = (close.iloc[-1] / close.iloc[-6] - 1)
                mom_10 = (close.iloc[-6] / close.iloc[-11] - 1)
                acceleration = mom_5 - mom_10
                advanced_features.append(acceleration)
            else:
                advanced_features.append(0)
                
            advanced_features = (advanced_features + [0] * 15)[:15]
        else:
            advanced_features = [0] * 15
            
        features.extend(advanced_features)
        
        # Ensure exactly 100 features
        features = (features + [0] * 100)[:100]
        
        # Apply smoothing to maintain consistency
        features = self._smooth_features('BTC', features)
        
        # Clip to [-1, 1] range
        features = np.clip(features, -1, 1).tolist()
        
        logger.info(f"Generated {len(features)} BTC features")
        return features
    
    def generate_asset_features(self, asset: str) -> List[float]:
        """Generate 2-dimensional features for non-BTC assets"""
        logger.info(f"Generating features for {asset}")
        
        # Map asset to Yahoo Finance symbol
        symbol_map = {
            'ETH': 'ETH-USD',
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'CADUSD': 'CADUSD=X',
            'NZDUSD': 'NZDUSD=X',
            'CHFUSD': 'CHFUSD=X',
            'XAUUSD': 'GC=F',  # Gold futures
            'XAGUSD': 'SI=F',  # Silver futures
        }
        
        symbol = symbol_map.get(asset, f'{asset}-USD')
        
        # Fetch asset data
        asset_data = self.fetch_market_data(symbol, period='2d', interval='1m')
        btc_data = self.fetch_market_data('BTC-USD', period='2d', interval='1m')
        
        features = [0.0, 0.0]  # Default values
        
        try:
            if not asset_data.empty and not btc_data.empty:
                # Feature 1: Momentum relative to BTC
                asset_returns = asset_data['Close'].pct_change().dropna()
                btc_returns = btc_data['Close'].pct_change().dropna()
                
                min_len = min(len(asset_returns), len(btc_returns))
                if min_len > 10:
                    asset_momentum = asset_returns.rolling(10).mean().iloc[-1]
                    btc_momentum = btc_returns.rolling(10).mean().iloc[-1]
                    features[0] = asset_momentum - btc_momentum
                
                # Feature 2: Asset-specific factor
                if asset == 'ETH':
                    # ETH-specific: correlation with BTC
                    if min_len > 20:
                        correlation = asset_returns.iloc[-20:].corr(btc_returns.iloc[-20:])
                        features[1] = correlation if not np.isnan(correlation) else 0
                        
                elif asset in ['EURUSD', 'GBPUSD', 'CADUSD', 'NZDUSD', 'CHFUSD']:
                    # Forex: volatility regime
                    if len(asset_returns) > 20:
                        recent_vol = asset_returns.rolling(5).std().iloc[-1]
                        normal_vol = asset_returns.rolling(20).std().iloc[-1]
                        features[1] = (recent_vol / normal_vol - 1) if normal_vol > 0 else 0
                        
                elif asset in ['XAUUSD', 'XAGUSD']:
                    # Precious metals: inverse correlation with dollar strength
                    dxy_data = self.fetch_market_data('DX-Y.NYB', period='2d', interval='1m')
                    if not dxy_data.empty:
                        dxy_returns = dxy_data['Close'].pct_change().dropna()
                        min_len_dxy = min(len(asset_returns), len(dxy_returns))
                        if min_len_dxy > 10:
                            correlation = asset_returns.iloc[-min_len_dxy:].corr(dxy_returns.iloc[-min_len_dxy:])
                            features[1] = -correlation if not np.isnan(correlation) else 0
                            
        except Exception as e:
            logger.error(f"Error generating features for {asset}: {e}")
        
        # Apply smoothing
        features = self._smooth_features(asset, features)
        
        # Clip to [-1, 1] range
        features = np.clip(features, -1, 1).tolist()
        
        logger.info(f"Generated features for {asset}: {features}")
        return features
    
    def _smooth_features(self, asset: str, features: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing to maintain feature consistency"""
        if asset not in self.feature_history:
            self.feature_history[asset] = features
            return features
        
        previous = self.feature_history[asset]
        smoothed = []
        
        for i, (new_val, prev_val) in enumerate(zip(features, previous)):
            smoothed_val = alpha * new_val + (1 - alpha) * prev_val
            smoothed.append(smoothed_val)
        
        self.feature_history[asset] = smoothed
        return smoothed
    
    # Technical indicator helper methods
    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 0.5
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return (rsi.iloc[-1] / 100 - 0.5) * 2  # Normalize to [-1, 1]
    
    def _macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return 0, 0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        # Normalize
        macd_norm = macd.iloc[-1] / prices.iloc[-1] * 100
        signal_norm = macd_signal.iloc[-1] / prices.iloc[-1] * 100
        
        return np.clip(macd_norm, -1, 1), np.clip(signal_norm, -1, 1)
    
    def _bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices.iloc[-1], prices.iloc[-1], prices.iloc[-1]
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    def _count_higher_highs(self, highs: pd.Series, period: int) -> float:
        """Count higher highs in recent period"""
        if len(highs) < period:
            return 0
        
        recent_highs = highs.iloc[-period:]
        count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs.iloc[i] > recent_highs.iloc[i-1]:
                count += 1
        
        return (count / (period - 1) - 0.5) * 2  # Normalize to [-1, 1]
    
    def _count_lower_lows(self, lows: pd.Series, period: int) -> float:
        """Count lower lows in recent period"""
        if len(lows) < period:
            return 0
        
        recent_lows = lows.iloc[-period:]
        count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows.iloc[i] < recent_lows.iloc[i-1]:
                count += 1
        
        return (count / (period - 1) - 0.5) * 2  # Normalize to [-1, 1]


class MANTISMiner:
    """Main MANTIS miner class"""
    
    def __init__(self, wallet_name: str, wallet_hotkey: str, netuid: int = 123):
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.netuid = netuid
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.r2_uploader = R2Uploader()
        self.timelock = Timelock(DRAND_PUBLIC_KEY)
        
        # Initialize Bittensor components
        self.wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.subtensor = bt.subtensor(network="finney")
        
        # State
        self.is_committed = False
        self.last_submission_time = 0
        
        logger.info(f"MANTIS Miner initialized for wallet {wallet_name}/{wallet_hotkey}")
    
    async def generate_multi_asset_embeddings(self) -> List[List[float]]:
        """Generate embeddings for all assets"""
        logger.info("Generating multi-asset embeddings...")
        
        embeddings = []
        
        for asset in ASSETS:
            if asset == "BTC":
                features = self.feature_engine.generate_btc_features()
            else:
                features = self.feature_engine.generate_asset_features(asset)
            
            embeddings.append(features)
            logger.info(f"Generated {len(features)} features for {asset}")
        
        return embeddings
    
    def encrypt_payload(self, embeddings: List[List[float]], lock_time_seconds: int = 30) -> Dict:
        """Encrypt embeddings with timelock"""
        logger.info("Encrypting payload...")
        
        try:
            # Get Drand info for future round calculation
            info_response = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10)
            info_response.raise_for_status()
            info = info_response.json()
            
            # Calculate target round
            future_time = time.time() + lock_time_seconds
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            # Create plaintext with hotkey signature
            plaintext = f"{str(embeddings)}:::{self.wallet.hotkey.ss58_address}"
            logger.info(f"Plaintext: {plaintext}")
            # Encrypt
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.timelock.tle(target_round, plaintext, salt).hex()
            
            payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
            logger.info(f"Payload encrypted for round {target_round}")
            return payload
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def upload_payload(self, payload: Dict) -> str:
        """Upload payload to R2 and return public URL"""
        logger.info("Uploading payload to R2...")
        
        try:
            # Use hotkey as filename
            filename = self.wallet.hotkey.ss58_address
            
            # Upload to R2
            public_url = await self.r2_uploader.upload_json(payload, filename)
            
            logger.info(f"Payload uploaded to: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def commit_url_to_subnet(self, public_url: str):
        """Commit the public URL to the subnet (one-time setup)"""
        if self.is_committed:
            logger.info("URL already committed to subnet")
            return
        
        try:
            logger.info(f"Committing URL to subnet: {public_url}")
            
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.netuid,
                data=public_url
            )
            
            self.is_committed = True
            logger.info("URL successfully committed to subnet")
            
        except Exception as e:
            logger.error(f"Failed to commit URL: {e}")
            raise
    
    async def mining_cycle(self):
        """Execute one complete mining cycle"""
        try:
            start_time = time.time()
            
            # Generate embeddings
            embeddings = await self.generate_multi_asset_embeddings()
            
            # Encrypt payload
            payload = self.encrypt_payload(embeddings)
            
            # Upload to R2
            public_url = await self.upload_payload(payload)
            
            # Commit URL if not already done
            if not self.is_committed:
                self.commit_url_to_subnet(public_url)
            
            self.last_submission_time = time.time()
            cycle_time = self.last_submission_time - start_time
            
            logger.info(f"Mining cycle completed in {cycle_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Mining cycle failed: {e}")
            raise
    
    async def run(self, update_interval: int = 60):
        """Run the miner continuously"""
        logger.info(f"Starting MANTIS miner with {update_interval}s update interval")
        
        while True:
            try:
                start_time = time.time()
                await self.mining_cycle()
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Miner stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                logger.info("Retrying in 30 seconds...")
                await asyncio.sleep(30)


async def main():
    parser = argparse.ArgumentParser(description="MANTIS Competitive Miner")
    parser.add_argument("--wallet.name", required=True, help="Wallet name")
    parser.add_argument("--wallet.hotkey", required=True, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=123, help="Subnet netuid")
    parser.add_argument("--update-interval", type=int, default=60, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize miner
    miner = MANTISMiner(
        wallet_name=getattr(args, 'wallet.name'),
        wallet_hotkey=getattr(args, 'wallet.hotkey'),
        netuid=args.netuid
    )
    
    # Run miner
    await miner.run(update_interval=args.update_interval)


if __name__ == "__main__":
    asyncio.run(main())
