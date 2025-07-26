"""
MANTIS Advanced Miner - Generates unique, competitive embeddings for multi-asset prediction
"""

import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import bittensor as bt
from timelock import Timelock
from dotenv import load_dotenv
import aiohttp

from r2_uploader import R2Uploader
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Timelock configuration
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class TechnicalIndicators:
    """Manual implementation of technical indicators without talib"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        return np.mean(prices[-period:])
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        
        alpha = 2 / (period + 1)
        ema_val = prices[0]
        
        for price in prices[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        
        return ema_val
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # For signal line, we'd need historical MACD values
        # Simplified: use a shorter EMA of recent price changes
        signal_line = TechnicalIndicators.ema(prices[-signal:], signal // 2) if len(prices) >= signal else macd_line
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands"""
        if len(prices) < period:
            price = prices[-1] if len(prices) > 0 else 0.0
            return price, price, price
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        if len(close) < k_period:
            return 50.0, 50.0
        
        lowest_low = np.min(low[-k_period:])
        highest_high = np.max(high[-k_period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Simplified D% as moving average of recent K% values
        d_percent = k_percent  # In practice, you'd average recent K% values
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(close) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0
        
        return np.mean(true_ranges[-period:])
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> float:
        """On-Balance Volume"""
        if len(close) < 2 or len(volume) == 0:
            return 0.0
        
        obv_val = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_val += volume[i]
            elif close[i] < close[i-1]:
                obv_val -= volume[i]
        
        return obv_val

class AdvancedDataCollector:
    """Collects data from multiple free sources for comprehensive market analysis"""
    
    def __init__(self):
        self.session = None
        self.scalers = {}
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("Data collector initialized with free APIs")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def get_crypto_data(self, symbol: str = 'bitcoin') -> Dict:
        """Get comprehensive crypto data from free APIs"""
        data = {}
        
        try:
            # CoinGecko API for price and market data
            coingecko_id = 'bitcoin' if symbol == 'BTC' else 'ethereum'
            
            # Current price and market data
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coingecko_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    price_data = await resp.json()
                    coin_data = price_data.get(coingecko_id, {})
                    
                    data['price'] = coin_data.get('usd', 0)
                    data['market_cap'] = coin_data.get('usd_market_cap', 0)
                    data['volume_24h'] = coin_data.get('usd_24h_vol', 0)
                    data['change_24h'] = coin_data.get('usd_24h_change', 0)
                    data['last_updated'] = coin_data.get('last_updated_at', 0)
            
            # Historical data for technical analysis
            hist_url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
            hist_params = {
                'vs_currency': 'usd',
                'days': '1',
                'interval': 'hourly'
            }
            
            async with self.session.get(hist_url, params=hist_params) as resp:
                if resp.status == 200:
                    hist_data = await resp.json()
                    prices = hist_data.get('prices', [])
                    volumes = hist_data.get('total_volumes', [])
                    
                    if prices:
                        # Convert to OHLCV format for technical analysis
                        df_data = []
                        for i, (timestamp, price) in enumerate(prices):
                            volume = volumes[i][1] if i < len(volumes) else 0
                            df_data.append({
                                'timestamp': timestamp,
                                'open': price,
                                'high': price,
                                'low': price,
                                'close': price,
                                'volume': volume
                            })
                        
                        data['ohlcv'] = pd.DataFrame(df_data)
            
            # Alternative free crypto API - CryptoCompare
            try:
                crypto_symbol = 'BTC' if symbol == 'BTC' else 'ETH'
                cc_url = f"https://min-api.cryptocompare.com/data/v2/histohour"
                cc_params = {
                    'fsym': crypto_symbol,
                    'tsym': 'USD',
                    'limit': 100
                }
                
                async with self.session.get(cc_url, params=cc_params) as resp:
                    if resp.status == 200:
                        cc_data = await resp.json()
                        if cc_data.get('Response') == 'Success':
                            hist_data = cc_data.get('Data', {}).get('Data', [])
                            if hist_data:
                                # Use this as backup OHLCV data
                                cc_df = pd.DataFrame(hist_data)
                                if 'ohlcv' not in data and not cc_df.empty:
                                    cc_df['timestamp'] = pd.to_datetime(cc_df['time'], unit='s')
                                    data['ohlcv'] = cc_df[['timestamp', 'open', 'high', 'low', 'close', 'volumefrom']].rename(
                                        columns={'volumefrom': 'volume'}
                                    )
            except Exception as e:
                logger.debug(f"CryptoCompare backup failed: {e}")
            
            # Fear & Greed Index (Bitcoin only)
            if symbol == 'BTC':
                try:
                    fng_url = "https://api.alternative.me/fng/"
                    async with self.session.get(fng_url) as resp:
                        if resp.status == 200:
                            fng_data = await resp.json()
                            if fng_data.get('data'):
                                data['fear_greed_index'] = float(fng_data['data'][0]['value'])
                except Exception as e:
                    logger.debug(f"Fear & Greed Index failed: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting crypto data for {symbol}: {e}")
            
        return data
    
    async def get_forex_data(self, pair: str) -> Dict:
        """Get forex data from free APIs"""
        data = {}
        
        try:
            # Use exchangerate-api.com (free tier)
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            # Current rate
            url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    rate_data = await resp.json()
                    rates = rate_data.get('rates', {})
                    if quote_currency in rates:
                        data['price'] = rates[quote_currency]
                        data['last_updated'] = rate_data.get('time_last_updated', 0)
            
            # Use yfinance as backup
            if 'price' not in data:
                try:
                    yahoo_symbol = f"{pair}=X"
                    ticker = yf.Ticker(yahoo_symbol)
                    hist = ticker.history(period="1d", interval="1h")
                    if not hist.empty:
                        data['price'] = float(hist['Close'].iloc[-1])
                        data['volume'] = float(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                        data['ohlcv'] = hist.tail(100)
                        data['change_1h'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) if len(hist) >= 2 else 0
                        data['volatility'] = hist['Close'].pct_change().std() * np.sqrt(24)  # Daily vol
                except Exception as e:
                    logger.debug(f"Yahoo Finance backup failed for {pair}: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to get forex data for {pair}: {e}")
            
        return data
    
    async def get_commodities_data(self, symbol: str) -> Dict:
        """Get commodities data from free APIs"""
        data = {}
        
        try:
            # Use metals-api.com free tier for precious metals
            if symbol in ['XAUUSD', 'XAGUSD']:
                metal = 'XAU' if symbol == 'XAUUSD' else 'XAG'
                
                # Try metals-api (free tier available)
                try:
                    metals_url = f"https://api.metals.live/v1/spot/{metal.lower()}"
                    async with self.session.get(metals_url) as resp:
                        if resp.status == 200:
                            metals_data = await resp.json()
                            if isinstance(metals_data, list) and metals_data:
                                data['price'] = metals_data[0].get('price', 0)
                except Exception as e:
                    logger.debug(f"Metals API failed: {e}")
                
                # Backup: Use yfinance
                if 'price' not in data:
                    try:
                        yahoo_symbol = "GC=F" if symbol == "XAUUSD" else "SI=F"
                        ticker = yf.Ticker(yahoo_symbol)
                        hist = ticker.history(period="1d", interval="1h")
                        if not hist.empty:
                            data['price'] = float(hist['Close'].iloc[-1])
                            data['volume'] = float(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                            data['ohlcv'] = hist.tail(100)
                            data['change_1h'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) if len(hist) >= 2 else 0
                            data['volatility'] = hist['Close'].pct_change().std() * np.sqrt(24)
                    except Exception as e:
                        logger.debug(f"Yahoo Finance backup failed for {symbol}: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to get commodities data for {symbol}: {e}")
            
        return data
    
    async def get_market_sentiment_data(self) -> Dict:
        """Get market sentiment from free sources"""
        data = {}
        
        try:
            # Fear & Greed Index
            fng_url = "https://api.alternative.me/fng/"
            async with self.session.get(fng_url) as resp:
                if resp.status == 200:
                    fng_data = await resp.json()
                    if fng_data.get('data'):
                        data['fear_greed_index'] = float(fng_data['data'][0]['value'])
                        data['fear_greed_classification'] = fng_data['data'][0]['value_classification']
            
            # Bitcoin dominance from CoinGecko
            dom_url = "https://api.coingecko.com/api/v3/global"
            async with self.session.get(dom_url) as resp:
                if resp.status == 200:
                    global_data = await resp.json()
                    market_data = global_data.get('data', {})
                    data['btc_dominance'] = market_data.get('market_cap_percentage', {}).get('btc', 0)
                    data['total_market_cap'] = market_data.get('total_market_cap', {}).get('usd', 0)
                    data['total_volume'] = market_data.get('total_volume', {}).get('usd', 0)
            
        except Exception as e:
            logger.debug(f"Error getting sentiment data: {e}")
            
        return data

class AdvancedFeatureEngine:
    """Generates sophisticated features for price prediction"""
    
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.scaler = RobustScaler()
        self.tech_indicators = TechnicalIndicators()
        
    def technical_indicators(self, ohlcv_df: pd.DataFrame) -> Dict[str, float]:
        """Generate technical indicators using manual implementations"""
        if len(ohlcv_df) < 20:
            return {}
            
        close = ohlcv_df['close'].values
        high = ohlcv_df['high'].values
        low = ohlcv_df['low'].values
        volume = ohlcv_df['volume'].values
        
        features = {}
        
        try:
            # RSI indicators
            features['rsi_14'] = self.tech_indicators.rsi(close, 14) / 100.0
            features['rsi_7'] = self.tech_indicators.rsi(close, 7) / 100.0
            
            # Moving averages
            features['sma_20'] = self.tech_indicators.sma(close, 20) / close[-1] if close[-1] != 0 else 1.0
            features['ema_12'] = self.tech_indicators.ema(close, 12) / close[-1] if close[-1] != 0 else 1.0
            
            # MACD
            macd, signal, histogram = self.tech_indicators.macd(close)
            features['macd'] = macd / close[-1] if close[-1] != 0 else 0.0
            features['macd_signal'] = signal / close[-1] if close[-1] != 0 else 0.0
            features['macd_histogram'] = histogram / close[-1] if close[-1] != 0 else 0.0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.tech_indicators.bollinger_bands(close)
            if bb_upper != bb_lower:
                features['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower)
            else:
                features['bb_position'] = 0.5
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0.0
            
            # Stochastic
            stoch_k, stoch_d = self.tech_indicators.stochastic(high, low, close)
            features['stoch_k'] = stoch_k / 100.0
            features['stoch_d'] = stoch_d / 100.0
            
            # ATR (Average True Range)
            atr = self.tech_indicators.atr(high, low, close)
            features['atr'] = atr / close[-1] if close[-1] != 0 else 0.0
            
            # Volume indicators
            if not np.all(volume == 0):
                obv = self.tech_indicators.obv(close, volume)
                features['obv'] = obv / np.mean(volume) if np.mean(volume) != 0 else 0.0
                
                # Volume-price trend
                features['volume_trend'] = np.corrcoef(close[-20:], volume[-20:])[0, 1] if len(close) >= 20 else 0.0
            
            # Price momentum features
            for period in [5, 10, 20]:
                if len(close) >= period:
                    features[f'momentum_{period}'] = (close[-1] / close[-period] - 1) if close[-period] != 0 else 0.0
                    features[f'volatility_{period}'] = np.std(close[-period:]) / np.mean(close[-period:]) if np.mean(close[-period:]) != 0 else 0.0
            
            # Price position features
            if len(close) >= 50:
                features['price_percentile_50'] = np.percentile(close[-50:], 50)
                features['price_position'] = (close[-1] - features['price_percentile_50']) / features['price_percentile_50'] if features['price_percentile_50'] != 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            
        # Clean and normalize features
        cleaned_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
                cleaned_features[key] = np.clip(value, -10, 10)  # Clip extreme values
                
        return cleaned_features
    
    def cross_asset_features(self, asset_data: Dict) -> Dict[str, float]:
        """Generate cross-asset correlation and momentum features"""
        features = {}
        
        try:
            # Extract price changes for correlation analysis
            price_changes = {}
            
            for asset, data in asset_data.items():
                if 'change_1h' in data:
                    price_changes[asset] = data['change_1h']
                elif 'change_24h' in data:
                    price_changes[asset] = data['change_24h'] / 24  # Approximate hourly change
            
            # Calculate momentum comparisons
            if 'BTC' in price_changes and 'ETH' in price_changes:
                btc_momentum = price_changes['BTC']
                eth_momentum = price_changes['ETH']
                features['btc_eth_momentum_diff'] = btc_momentum - eth_momentum
                features['btc_eth_momentum_ratio'] = btc_momentum / eth_momentum if eth_momentum != 0 else 1.0
                
            # Risk-on/Risk-off sentiment
            if 'XAUUSD' in price_changes and 'BTC' in price_changes:
                features['risk_sentiment'] = price_changes['BTC'] - price_changes['XAUUSD']
                
            # Currency strength analysis
            usd_pairs = ['EURUSD', 'GBPUSD', 'CADUSD', 'NZDUSD']
            usd_strength = 0
            pair_count = 0
            
            for pair in usd_pairs:
                if pair in price_changes:
                    usd_strength -= price_changes[pair]  # Negative because USD is quote currency
                    pair_count += 1
                    
            if pair_count > 0:
                features['usd_strength'] = usd_strength / pair_count
                
            # Market correlation features
            if len(price_changes) >= 3:
                changes = list(price_changes.values())
                features['market_correlation'] = np.corrcoef(changes, changes)[0, 1] if len(changes) > 1 else 0.0
                features['market_volatility'] = np.std(changes)
                
        except Exception as e:
            logger.warning(f"Error calculating cross-asset features: {e}")
            
        return features
    
    def time_features(self) -> Dict[str, float]:
        """Generate time-based features"""
        now = datetime.utcnow()
        
        features = {
            'hour_sin': np.sin(2 * np.pi * now.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24),
            'day_sin': np.sin(2 * np.pi * now.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * now.weekday() / 7),
            'is_weekend': float(now.weekday() >= 5),
            'is_market_hours_ny': float(9 <= now.hour <= 16),
            'is_market_hours_london': float(8 <= now.hour <= 17),
            'is_market_hours_asia': float(0 <= now.hour <= 8),
        }
        
        return features

class MANTISMiner:
    """Advanced MANTIS miner with sophisticated feature generation"""
    
    def __init__(self, wallet_name: str, hotkey_name: str, netuid: int = 123):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        
        # Initialize components
        self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        self.subtensor = bt.subtensor(network="finney")
        self.data_collector = AdvancedDataCollector()
        self.feature_engine = AdvancedFeatureEngine()
        self.r2_uploader = R2Uploader()
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        
        # Get hotkey for encryption
        self.hotkey = self.wallet.hotkey.ss58_address
        
        # State tracking
        self.public_url = None
        self.last_commit_time = 0
        
        logger.info(f"MANTIS Miner initialized for hotkey: {self.hotkey}")
    
    async def commit_public_url(self):
        """Commit public URL to the subnet"""
        try:
            # Test R2 connection first
            if not self.r2_uploader.test_connection():
                raise Exception("R2 connection test failed")
            
            # Construct public URL
            self.public_url = f"{self.r2_uploader.public_url_base.rstrip('/')}/{self.hotkey}"
            
            # Commit to subtensor
            success = self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.netuid,
                data=self.public_url
            )
            
            if success:
                logger.info(f"Successfully committed public URL: {self.public_url}")
                self.last_commit_time = time.time()
                return True
            else:
                logger.error("Failed to commit public URL to subtensor")
                return False
                
        except Exception as e:
            logger.error(f"Error committing public URL: {e}")
            return False
    
    async def collect_all_data(self) -> Dict:
        """Collect data for all assets using free APIs"""
        all_data = {}
        
        # Collect crypto data
        btc_data = await self.data_collector.get_crypto_data('BTC')
        if btc_data:
            all_data['BTC'] = btc_data
        
        eth_data = await self.data_collector.get_crypto_data('ETH')
        if eth_data:
            all_data['ETH'] = eth_data
        
        # Collect forex data
        forex_pairs = ['EURUSD', 'GBPUSD', 'CADUSD', 'NZDUSD', 'CHFUSD']
        for pair in forex_pairs:
            forex_data = await self.data_collector.get_forex_data(pair)
            if forex_data:
                all_data[pair] = forex_data
        
        # Collect commodities data
        commodities = ['XAUUSD', 'XAGUSD']
        for commodity in commodities:
            commodity_data = await self.data_collector.get_commodities_data(commodity)
            if commodity_data:
                all_data[commodity] = commodity_data
        
        # Add market sentiment data
        sentiment_data = await self.data_collector.get_market_sentiment_data()
        if sentiment_data:
            all_data['sentiment'] = sentiment_data
        
        return all_data
    
    def generate_asset_embedding(self, asset: str, asset_data: Dict, all_data: Dict) -> List[float]:
        """Generate embedding for a specific asset using free API data"""
        target_dim = config.ASSET_EMBEDDING_DIMS[asset]
        features = {}
        
        try:
            # Technical indicators from OHLCV data
            if 'ohlcv' in asset_data and not asset_data['ohlcv'].empty:
                tech_features = self.feature_engine.technical_indicators(asset_data['ohlcv'])
                features.update(tech_features)
            
            # Basic price and market features
            if 'price' in asset_data:
                features['price_normalized'] = np.tanh(asset_data['price'] / 50000 if asset == 'BTC' else asset_data['price'])
            
            if 'volume_24h' in asset_data:
                features['volume_24h_normalized'] = np.tanh(asset_data['volume_24h'] / 1e9)
            
            if 'change_24h' in asset_data:
                features['change_24h'] = np.tanh(asset_data['change_24h'] / 100)
            
            if 'market_cap' in asset_data:
                features['market_cap_normalized'] = np.tanh(asset_data['market_cap'] / 1e12)
            
            # Sentiment features (mainly for BTC)
            if asset == 'BTC' and 'sentiment' in all_data:
                sentiment = all_data['sentiment']
                if 'fear_greed_index' in sentiment:
                    features['fear_greed'] = (sentiment['fear_greed_index'] - 50) / 50  # Normalize to [-1, 1]
                if 'btc_dominance' in sentiment:
                    features['btc_dominance'] = (sentiment['btc_dominance'] - 50) / 50
            
            # Cross-asset features
            cross_features = self.feature_engine.cross_asset_features(all_data)
            features.update(cross_features)
            
            # Time features
            time_features = self.feature_engine.time_features()
            features.update(time_features)
            
            # Convert to list and pad/truncate to target dimension
            feature_values = list(features.values())
            
            # Ensure all values are finite and in range
            feature_values = [
                np.clip(float(v), -1.0, 1.0) if np.isfinite(v) else 0.0 
                for v in feature_values
            ]
            
            if len(feature_values) < target_dim:
                # Pad with zeros
                feature_values.extend([0.0] * (target_dim - len(feature_values)))
            elif len(feature_values) > target_dim:
                # Use PCA or truncate intelligently
                if target_dim >= 10:  # Use PCA for larger dimensions
                    try:
                        feature_array = np.array(feature_values).reshape(1, -1)
                        pca = PCA(n_components=target_dim)
                        reduced = pca.fit_transform(feature_array)[0]
                        feature_values = [np.clip(float(v), -1.0, 1.0) for v in reduced]
                    except:
                        feature_values = feature_values[:target_dim]
                else:
                    # For small dimensions, select most important features
                    feature_values = feature_values[:target_dim]
            
            return feature_values
            
        except Exception as e:
            logger.error(f"Error generating embedding for {asset}: {e}")
            return [0.0] * target_dim
    
    def generate_multi_asset_embeddings(self, all_data: Dict) -> List[List[float]]:
        """Generate embeddings for all assets"""
        embeddings = []
        
        for asset in config.ASSETS:
            asset_data = all_data.get(asset, {})
            embedding = self.generate_asset_embedding(asset, asset_data, all_data)
            embeddings.append(embedding)
            
            logger.info(f"Generated {len(embedding)}-dim embedding for {asset}")
        
        return embeddings
    
    def convert_np_float64_to_float(self, data):
        if isinstance(data, list):
            return [self.convert_np_float64_to_float(item) for item in data]
        elif isinstance(data, np.float64):
            return float(data)
        else:
            return data

    async def create_encrypted_payload(self, embeddings: List[List[float]]) -> Dict:
        """Create timelock-encrypted payload"""
        try:
            # Get future round for timelock
            info_response = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10)
            info = info_response.json()
            
            future_time = time.time() + 30  # 30 seconds in future
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            # Create plaintext with hotkey signature
            cleaned_embeddings = self.convert_np_float64_to_float(embeddings)
            plaintext = f"{str(cleaned_embeddings)}:::{self.hotkey}"
            
            logger.info(f"Plaintext: {plaintext}")
            # Encrypt
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.tlock.tle(target_round, plaintext, salt).hex()
            
            payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
            return payload
            
        except Exception as e:
            logger.error(f"Error creating encrypted payload: {e}")
            raise
    
    async def mine_once(self):
        """Perform one mining iteration"""
        try:
            logger.info("Starting mining iteration...")
            
            # Collect data
            all_data = await self.collect_all_data()
            if not all_data:
                logger.warning("No data collected, skipping iteration")
                return
            
            logger.info(f"Collected data for {len(all_data)} assets")
            
            # Generate embeddings
            embeddings = self.generate_multi_asset_embeddings(all_data)
            
            # Create encrypted payload
            payload = await self.create_encrypted_payload(embeddings)
            
            # Upload to R2
            public_url = await self.r2_uploader.upload_json(payload, self.hotkey)
            
            logger.info(f"Successfully uploaded payload to: {public_url}")
            
        except Exception as e:
            logger.error(f"Error in mining iteration: {e}")
    
    async def run(self, mining_interval: int = 60):
        """Run the miner continuously"""
        logger.info("Starting MANTIS Advanced Miner...")
        
        # Initialize data collector
        await self.data_collector.initialize()
        
        try:
            # Commit public URL at startup
            if not await self.commit_public_url():
                logger.error("Failed to commit public URL, exiting")
                return
            
            # Main mining loop
            while True:
                try:
                    # Re-commit URL periodically (every 24 hours)
                    if time.time() - self.last_commit_time > 86400:
                        await self.commit_public_url()
                    
                    start_time = time.time()
                    await self.mine_once()
                    elapsed = time.time() - start_time
                    sleep_time = max(0, mining_interval - elapsed)
                    if sleep_time > 0:
                        logger.info(f"Sleeping for {sleep_time} seconds...")
                        await asyncio.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
        finally:
            await self.data_collector.cleanup()

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MANTIS Advanced Miner")
    parser.add_argument("--wallet.name", required=True, help="Wallet name")
    parser.add_argument("--wallet.hotkey", required=True, help="Hotkey name")
    parser.add_argument("--netuid", type=int, default=123, help="Subnet netuid")
    parser.add_argument("--interval", type=int, default=60, help="Mining interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run miner
    miner = MANTISMiner(
        wallet_name=getattr(args, 'wallet.name'),
        hotkey_name=getattr(args, 'wallet.hotkey'),
        netuid=args.netuid
    )
    
    await miner.run(mining_interval=args.interval)

if __name__ == "__main__":
    asyncio.run(main())
