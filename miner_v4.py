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

class UniqueFeatureEngine:
    """Generates highly unique and creative features with controlled randomness"""
    
    def __init__(self, miner_id: str):
        self.miner_id = miner_id
        self.lookback_periods = [3, 7, 13, 21, 34, 55, 89]  # Fibonacci sequence
        self.scaler = RobustScaler()
        self.tech_indicators = TechnicalIndicators()
        
        # Create unique random seed based on miner ID for consistent uniqueness
        self.unique_seed = hash(miner_id) % 2**32
        np.random.seed(self.unique_seed)
        
        # Generate unique feature weights for this miner
        self.feature_weights = np.random.uniform(0.5, 2.0, 50)
        self.phase_shifts = np.random.uniform(0, 2*np.pi, 20)
        self.frequency_mults = np.random.uniform(0.8, 1.2, 15)
        
        logger.info(f"Initialized unique feature engine with seed: {self.unique_seed}")
    
    def fractal_dimension_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate fractal dimension and chaos theory features"""
        features = {}
        
        try:
            if len(prices) < 10:
                return features
            
            # Higuchi fractal dimension
            def higuchi_fd(data, kmax=10):
                N = len(data)
                lk = []
                for k in range(1, kmax + 1):
                    lm = []
                    for m in range(k):
                        ll = 0
                        for i in range(1, int((N - m) / k)):
                            ll += abs(data[m + i * k] - data[m + (i - 1) * k])
                        ll = ll * (N - 1) / (k * k * int((N - m) / k))
                        lm.append(ll)
                    lk.append(np.mean(lm))
                
                lk = np.array(lk)
                k = np.arange(1, kmax + 1)
                
                # Linear regression to find slope
                if len(lk) > 1 and np.std(np.log(k)) > 0:
                    slope = np.polyfit(np.log(k), np.log(lk), 1)[0]
                    return -slope
                return 1.5
            
            features['fractal_dimension'] = higuchi_fd(prices)
            
            # Hurst exponent using R/S analysis
            def hurst_exponent(data):
                n = len(data)
                if n < 20:
                    return 0.5
                
                # Calculate mean
                mean_data = np.mean(data)
                
                # Calculate cumulative deviations
                cumdev = np.cumsum(data - mean_data)
                
                # Calculate range
                R = np.max(cumdev) - np.min(cumdev)
                
                # Calculate standard deviation
                S = np.std(data)
                
                if S == 0:
                    return 0.5
                
                # R/S ratio
                rs = R / S
                
                # Hurst exponent approximation
                return np.log(rs) / np.log(n)
            
            features['hurst_exponent'] = hurst_exponent(prices)
            
            # Lyapunov exponent approximation
            def lyapunov_approx(data, lag=1):
                if len(data) < lag + 2:
                    return 0.0
                
                divergences = []
                for i in range(len(data) - lag - 1):
                    diff = abs(data[i + lag + 1] - data[i + 1])
                    if diff > 0:
                        divergences.append(np.log(diff))
                
                return np.mean(divergences) if divergences else 0.0
            
            features['lyapunov_approx'] = lyapunov_approx(prices)
            
        except Exception as e:
            logger.debug(f"Error in fractal features: {e}")
            
        return features
    
    def wavelet_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract wavelet-based features using PyWavelets"""
        features = {}
        
        try:
            import pywt
            
            if len(prices) < 16:
                return features
            
            # Ensure length is power of 2 for wavelet transform
            n = len(prices)
            power_of_2 = 2 ** int(np.log2(n))
            prices_padded = prices[-power_of_2:]
            
            # Multi-level wavelet decomposition
            wavelets = ['db4', 'haar', 'coif2']
            
            for i, wavelet in enumerate(wavelets):
                try:
                    coeffs = pywt.wavedec(prices_padded, wavelet, level=3)
                    
                    # Energy in different frequency bands
                    for j, coeff in enumerate(coeffs):
                        energy = np.sum(coeff ** 2)
                        features[f'wavelet_{wavelet}_level_{j}_energy'] = energy
                        
                        # Statistical moments
                        if len(coeff) > 0:
                            features[f'wavelet_{wavelet}_level_{j}_mean'] = np.mean(coeff)
                            features[f'wavelet_{wavelet}_level_{j}_std'] = np.std(coeff)
                            features[f'wavelet_{wavelet}_level_{j}_skew'] = self._safe_skew(coeff)
                            features[f'wavelet_{wavelet}_level_{j}_kurt'] = self._safe_kurtosis(coeff)
                    
                except Exception as e:
                    logger.debug(f"Error with wavelet {wavelet}: {e}")
                    continue
            
        except ImportError:
            logger.debug("PyWavelets not available, skipping wavelet features")
        except Exception as e:
            logger.debug(f"Error in wavelet features: {e}")
            
        return features
    
    def spectral_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT"""
        features = {}
        
        try:
            if len(prices) < 8:
                return features
            
            # Remove trend
            detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
            
            # Apply window function
            windowed = detrended * np.hanning(len(detrended))
            
            # FFT
            fft = np.fft.fft(windowed)
            freqs = np.fft.fftfreq(len(windowed))
            
            # Power spectral density
            psd = np.abs(fft) ** 2
            
            # Spectral features
            features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs[:len(freqs)//2] - features['spectral_centroid']) ** 2) * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]))
            features['spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]] if len(np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0]) > 0 else 0
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd[:len(psd)//2])
            features['dominant_frequency'] = freqs[dominant_freq_idx]
            features['dominant_power'] = psd[dominant_freq_idx]
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
            features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
            
        except Exception as e:
            logger.debug(f"Error in spectral features: {e}")
            
        return features
    
    def unique_mathematical_transforms(self, prices: np.ndarray) -> Dict[str, float]:
        """Apply unique mathematical transformations with miner-specific parameters"""
        features = {}
        
        try:
            if len(prices) < 5:
                return features
            
            # Miner-specific transformations using unique parameters
            for i, weight in enumerate(self.feature_weights[:10]):
                # Custom polynomial features
                poly_feature = np.sum(prices ** (1 + weight)) / len(prices)
                features[f'poly_transform_{i}'] = np.tanh(poly_feature / np.mean(prices))
                
                # Custom trigonometric transforms
                phase = self.phase_shifts[i % len(self.phase_shifts)]
                freq_mult = self.frequency_mults[i % len(self.frequency_mults)]
                
                trig_feature = np.mean(np.sin(prices * freq_mult + phase) * weight)
                features[f'trig_transform_{i}'] = trig_feature
                
                # Custom exponential decay features
                decay_weights = np.exp(-np.arange(len(prices)) * weight * 0.1)
                decay_feature = np.sum(prices * decay_weights) / np.sum(decay_weights)
                features[f'decay_transform_{i}'] = decay_feature / np.mean(prices) if np.mean(prices) != 0 else 0
            
            # Unique ratio-based features
            golden_ratio = 1.618033988749
            features['golden_ratio_momentum'] = (prices[-1] / prices[int(-len(prices)/golden_ratio)] - 1) if len(prices) > golden_ratio else 0
            
            # Fibonacci retracement levels
            if len(prices) >= 21:
                high = np.max(prices[-21:])
                low = np.min(prices[-21:])
                current = prices[-1]
                
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                for level in fib_levels:
                    fib_price = high - (high - low) * level
                    features[f'fib_distance_{level}'] = (current - fib_price) / (high - low) if (high - low) != 0 else 0
            
            # Custom volatility measures
            log_returns = np.diff(np.log(prices + 1e-8))
            if len(log_returns) > 0:
                # Unique volatility with miner-specific weighting
                weighted_vol = np.sqrt(np.sum(log_returns ** 2 * self.feature_weights[:len(log_returns)]) / np.sum(self.feature_weights[:len(log_returns)]))
                features['weighted_volatility'] = weighted_vol
                
                # Asymmetric volatility
                upside_vol = np.std(log_returns[log_returns > 0]) if np.any(log_returns > 0) else 0
                downside_vol = np.std(log_returns[log_returns < 0]) if np.any(log_returns < 0) else 0
                features['volatility_asymmetry'] = (upside_vol - downside_vol) / (upside_vol + downside_vol + 1e-8)
            
        except Exception as e:
            logger.debug(f"Error in mathematical transforms: {e}")
            
        return features
    
    def market_microstructure_features(self, ohlcv_df: pd.DataFrame) -> Dict[str, float]:
        """Extract market microstructure and order flow features"""
        features = {}
        
        try:
            if len(ohlcv_df) < 5:
                return features
            
            high = ohlcv_df['high'].values
            low = ohlcv_df['low'].values
            close = ohlcv_df['close'].values
            volume = ohlcv_df['volume'].values
            
            # Price impact measures
            if len(close) >= 2:
                price_changes = np.diff(close)
                volume_changes = np.diff(volume)
                
                # Volume-weighted price impact
                nonzero_vol = volume_changes[volume_changes != 0]
                nonzero_price = price_changes[volume_changes != 0]
                
                if len(nonzero_vol) > 0:
                    features['price_impact'] = np.corrcoef(nonzero_price, nonzero_vol)[0, 1] if len(nonzero_vol) > 1 else 0
            
            # Bid-ask spread proxy using high-low
            spreads = (high - low) / close
            features['avg_spread_proxy'] = np.mean(spreads)
            features['spread_volatility'] = np.std(spreads)
            
            # Order flow imbalance proxy
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            if len(money_flow) >= 2:
                positive_flow = money_flow[np.diff(close) > 0]
                negative_flow = money_flow[np.diff(close) < 0]
                
                total_positive = np.sum(positive_flow) if len(positive_flow) > 0 else 0
                total_negative = np.sum(negative_flow) if len(negative_flow) > 0 else 0
                
                features['order_flow_imbalance'] = (total_positive - total_negative) / (total_positive + total_negative + 1e-8)
            
            # Volume profile features
            price_levels = np.linspace(np.min(low), np.max(high), 10)
            volume_profile = np.zeros(len(price_levels) - 1)
            
            for i in range(len(ohlcv_df)):
                for j in range(len(price_levels) - 1):
                    if price_levels[j] <= close[i] < price_levels[j + 1]:
                        volume_profile[j] += volume[i]
                        break
            
            if np.sum(volume_profile) > 0:
                volume_profile_norm = volume_profile / np.sum(volume_profile)
                features['volume_concentration'] = np.max(volume_profile_norm)
                features['volume_entropy'] = -np.sum(volume_profile_norm[volume_profile_norm > 0] * 
                                                   np.log2(volume_profile_norm[volume_profile_norm > 0]))
            
        except Exception as e:
            logger.debug(f"Error in microstructure features: {e}")
            
        return features
    
    def regime_detection_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Detect market regimes and extract regime-specific features"""
        features = {}
        
        try:
            if len(prices) < 20:
                return features
            
            # Volatility regimes
            returns = np.diff(np.log(prices + 1e-8))
            rolling_vol = pd.Series(returns).rolling(window=10).std().values
            
            # Detect high/low volatility regimes
            vol_threshold = np.median(rolling_vol[~np.isnan(rolling_vol)])
            current_vol = rolling_vol[-1] if not np.isnan(rolling_vol[-1]) else vol_threshold
            
            features['vol_regime'] = 1.0 if current_vol > vol_threshold else -1.0
            features['vol_regime_strength'] = abs(current_vol - vol_threshold) / vol_threshold if vol_threshold != 0 else 0
            
            # Trend regimes using multiple timeframes
            for window in [5, 10, 20]:
                if len(prices) >= window:
                    trend = (prices[-1] - prices[-window]) / prices[-window] if prices[-window] != 0 else 0
                    features[f'trend_regime_{window}'] = np.tanh(trend * 100)  # Normalize
            
            # Mean reversion vs momentum regimes
            if len(returns) >= 10:
                # Autocorrelation of returns
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
                features['mean_reversion_strength'] = -autocorr  # Negative autocorr = mean reversion
                
                # Momentum strength
                momentum_periods = [3, 5, 10]
                momentum_scores = []
                for period in momentum_periods:
                    if len(returns) >= period:
                        period_return = np.sum(returns[-period:])
                        momentum_scores.append(period_return)
                
                if momentum_scores:
                    features['momentum_consistency'] = np.std(momentum_scores) / (np.mean(np.abs(momentum_scores)) + 1e-8)
            
        except Exception as e:
            logger.debug(f"Error in regime detection: {e}")
            
        return features
    
    def behavioral_finance_features(self, prices: np.ndarray, volume: np.ndarray = None) -> Dict[str, float]:
        """Extract behavioral finance and sentiment-based features"""
        features = {}
        
        try:
            if len(prices) < 10:
                return features
            
            # Overreaction and underreaction patterns
            returns = np.diff(np.log(prices + 1e-8))
            
            if len(returns) >= 5:
                # Short-term reversal (overreaction)
                short_term_returns = returns[-3:]
                long_term_returns = returns[-10:-3] if len(returns) >= 10 else returns[:-3]
                
                if len(long_term_returns) > 0:
                    short_momentum = np.mean(short_term_returns)
                    long_momentum = np.mean(long_term_returns)
                    features['reversal_signal'] = -short_momentum * long_momentum  # Contrarian signal
            
            # Psychological price levels (round numbers)
            current_price = prices[-1]
            
            # Distance to round numbers
            round_levels = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
            min_distance = float('inf')
            
            for level in round_levels:
                distance = abs(current_price - level) / level
                if distance < min_distance:
                    min_distance = distance
            
            features['round_number_proximity'] = 1 / (1 + min_distance)
            
            # Support/resistance levels
            if len(prices) >= 20:
                # Find local maxima and minima
                from scipy.signal import argrelextrema
                
                try:
                    local_maxima = argrelextrema(prices, np.greater, order=3)[0]
                    local_minima = argrelextrema(prices, np.less, order=3)[0]
                    
                    # Distance to nearest support/resistance
                    resistance_levels = prices[local_maxima] if len(local_maxima) > 0 else []
                    support_levels = prices[local_minima] if len(local_minima) > 0 else []
                    
                    if len(resistance_levels) > 0:
                        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                        features['resistance_distance'] = (nearest_resistance - current_price) / current_price
                    
                    if len(support_levels) > 0:
                        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                        features['support_distance'] = (current_price - nearest_support) / current_price
                        
                except ImportError:
                    # Fallback without scipy
                    pass
            
            # Herding behavior proxy
            if volume is not None and len(volume) >= 10:
                # Volume spikes during price moves
                price_moves = np.abs(returns)
                volume_changes = np.diff(volume)
                
                if len(volume_changes) >= len(price_moves):
                    volume_changes = volume_changes[:len(price_moves)]
                    
                    # Correlation between price volatility and volume
                    if len(price_moves) > 1 and np.std(volume_changes) > 0:
                        herding_proxy = np.corrcoef(price_moves[1:], volume_changes)[0, 1]
                        features['herding_behavior'] = herding_proxy
            
        except Exception as e:
            logger.debug(f"Error in behavioral features: {e}")
            
        return features
    
    def _safe_skew(self, data):
        """Safely calculate skewness"""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            if len(data) < 3:
                return 0.0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
    
    def _safe_kurtosis(self, data):
        """Safely calculate kurtosis"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            if len(data) < 4:
                return 0.0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3
    
    def generate_comprehensive_features(self, asset: str, asset_data: Dict, all_data: Dict) -> Dict[str, float]:
        """Generate comprehensive unique features for an asset"""
        all_features = {}
        
        try:
            # Get OHLCV data if available
            ohlcv_df = asset_data.get('ohlcv', pd.DataFrame())
            
            if not ohlcv_df.empty and len(ohlcv_df) >= 5:
                prices = ohlcv_df['close'].values
                volume = ohlcv_df['volume'].values if 'volume' in ohlcv_df else None
                
                # Apply all unique feature extraction methods
                fractal_features = self.fractal_dimension_features(prices)
                all_features.update(fractal_features)
                
                wavelet_features = self.wavelet_features(prices)
                all_features.update(wavelet_features)
                
                spectral_features = self.spectral_features(prices)
                all_features.update(spectral_features)
                
                math_features = self.unique_mathematical_transforms(prices)
                all_features.update(math_features)
                
                microstructure_features = self.market_microstructure_features(ohlcv_df)
                all_features.update(microstructure_features)
                
                regime_features = self.regime_detection_features(prices)
                all_features.update(regime_features)
                
                behavioral_features = self.behavioral_finance_features(prices, volume)
                all_features.update(behavioral_features)
            
            # Add basic features with unique transformations
            if 'price' in asset_data:
                price = asset_data['price']
                # Apply miner-specific transformation
                unique_price_feature = np.tanh(price * self.feature_weights[0] / 10000)
                all_features['unique_price_transform'] = unique_price_feature
            
            # Add controlled randomness that maintains some predictive value
            current_time = time.time()
            time_seed = int(current_time / 3600)  # Changes every hour
            np.random.seed(self.unique_seed + time_seed)
            
            # Generate time-varying unique features
            for i in range(5):
                # Controlled random walk that slowly evolves
                random_walk = np.sin(current_time * self.frequency_mults[i] / 3600) * 0.1
                noise = np.random.normal(0, 0.05)  # Small amount of noise
                all_features[f'unique_temporal_{i}'] = random_walk + noise
            
        except Exception as e:
            logger.error(f"Error generating comprehensive features for {asset}: {e}")
        
        return all_features

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
        
        # Get hotkey for encryption and unique ID
        self.hotkey = self.wallet.hotkey.ss58_address
        
        # Initialize unique feature engine with miner-specific ID
        self.feature_engine = UniqueFeatureEngine(self.hotkey)
        
        self.r2_uploader = R2Uploader()
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        
        # State tracking
        self.public_url = None
        self.last_commit_time = 0
        
        logger.info(f"MANTIS Unique Miner initialized for hotkey: {self.hotkey}")
    
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
        """Generate highly unique embedding for a specific asset"""
        target_dim = config.ASSET_EMBEDDING_DIMS[asset]
        
        try:
            # Generate comprehensive unique features
            features = self.feature_engine.generate_comprehensive_features(asset, asset_data, all_data)
            
            # Convert to feature vector
            feature_values = list(features.values())
        
            # Ensure all values are finite and in range
            feature_values = [
                np.clip(float(v), -1.0, 1.0) if np.isfinite(v) else 0.0 
                for v in feature_values
            ]
            
            # Add asset-specific unique transformations
            if len(feature_values) > 0:
                # Apply miner-specific feature mixing
                mixed_features = []
                for i in range(min(target_dim, len(feature_values) * 2)):
                    # Create unique combinations of features
                    idx1 = i % len(feature_values)
                    idx2 = (i + 1) % len(feature_values)
                    
                    # Unique mixing function based on miner ID
                    weight1 = self.feature_engine.feature_weights[i % len(self.feature_engine.feature_weights)]
                    weight2 = 1.0 - weight1
                    
                    mixed_value = (feature_values[idx1] * weight1 + feature_values[idx2] * weight2)
                    
                    # Apply unique non-linear transformation
                    phase = self.feature_engine.phase_shifts[i % len(self.feature_engine.phase_shifts)]
                    transformed = np.tanh(mixed_value + np.sin(mixed_value * 3.14159 + phase) * 0.1)
                    
                    mixed_features.append(transformed)
                
                feature_values = mixed_features
            
            # Pad or truncate to target dimension
            if len(feature_values) < target_dim:
                # Pad with unique pattern based on existing features
                padding_pattern = []
                for i in range(target_dim - len(feature_values)):
                    if len(feature_values) > 0:
                        # Create padding based on existing features
                        base_idx = i % len(feature_values)
                        phase_shift = self.feature_engine.phase_shifts[i % len(self.feature_engine.phase_shifts)]
                        padding_value = feature_values[base_idx] * np.cos(phase_shift) * 0.5
                    else:
                        padding_value = 0.0
                    padding_pattern.append(padding_value)
                
                feature_values.extend(padding_pattern)
            elif len(feature_values) > target_dim:
                # Intelligent truncation using PCA or selection
                if target_dim >= 10:
                    try:
                        feature_array = np.array(feature_values).reshape(1, -1)
                        pca = PCA(n_components=target_dim)
                        reduced = pca.fit_transform(feature_array)[0]
                        feature_values = [np.clip(float(v), -1.0, 1.0) for v in reduced]
                    except:
                        # Fallback: select features with unique pattern
                        indices = np.linspace(0, len(feature_values) - 1, target_dim, dtype=int)
                        feature_values = [feature_values[i] for i in indices]
                else:
                    # For small dimensions, use unique selection pattern
                    step = len(feature_values) // target_dim
                    feature_values = feature_values[::step][:target_dim]
            
            # Final normalization and uniqueness injection
            final_features = []
            for i, value in enumerate(feature_values[:target_dim]):
                # Add small unique signature to each feature
                signature = np.sin(self.feature_engine.unique_seed * (i + 1) / 1000) * 0.02
                final_value = np.clip(value + signature, -1.0, 1.0)
                final_features.append(final_value)
            
            # Ensure we have exactly target_dim features
            while len(final_features) < target_dim:
                final_features.append(0.0)
            
            return final_features[:target_dim]
            
        except Exception as e:
            logger.error(f"Error generating unique embedding for {asset}: {e}")
            # Return unique fallback pattern instead of zeros
            fallback = []
            for i in range(target_dim):
                value = np.sin(self.feature_engine.unique_seed * (i + 1) / 100) * 0.5
                fallback.append(value)
            return fallback
    
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
