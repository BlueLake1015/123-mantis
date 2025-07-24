#!/usr/bin/env python3
"""
MANTIS Enhanced Miner v2
========================

An advanced second miner for MANTIS that focuses on completely different
signal sources to maximize uniqueness and avoid redundancy with the first miner.

Key Differentiators:
- On-chain and DeFi metrics
- Alternative data sources (social, news, derivatives)
- Advanced statistical models
- Microstructure and order flow analysis
- Macro-economic indicators
- Cross-market arbitrage signals
"""

import argparse
import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import bittensor as bt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from timelock import Timelock
import ta
from textblob import TextBlob

from config import ASSETS, ASSET_EMBEDDING_DIMS
from r2_uploader import R2Uploader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_miner.log'),
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

class AlternativeDataEngine:
    """Fetches and processes alternative data sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_fear_greed_index(self) -> float:
        """Get crypto fear & greed index"""
        try:
            if 'fear_greed' in self.cache:
                data, timestamp = self.cache['fear_greed']
                if time.time() - timestamp < self.cache_timeout:
                    return data
            
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                value = float(data['data'][0]['value'])
                normalized = (value - 50) / 50  # Normalize to [-1, 1]
                self.cache['fear_greed'] = (normalized, time.time())
                return normalized
                
        except Exception as e:
            logger.warning(f"Failed to fetch fear & greed index: {e}")
        
        return 0.0
    
    def get_google_trends_score(self, keyword: str = "bitcoin") -> float:
        """Simulate Google Trends data (replace with actual API)"""
        try:
            # This is a placeholder - implement actual Google Trends API
            # For now, return a random walk to simulate trend data
            current_time = int(time.time() / 3600)  # Hour-based seed
            np.random.seed(current_time)
            trend_score = np.random.normal(0, 0.3)
            return np.clip(trend_score, -1, 1)
        except Exception:
            return 0.0
    
    def get_social_sentiment(self) -> Dict[str, float]:
        """Get social media sentiment scores"""
        try:
            # Placeholder for social sentiment API
            # In production, integrate with Twitter API, Reddit API, etc.
            sentiments = {}
            
            # Simulate different social platforms
            platforms = ['twitter', 'reddit', 'telegram']
            for platform in platforms:
                # Generate time-based pseudo-random sentiment
                seed = int(time.time() / 1800) + hash(platform) % 1000  # 30-min intervals
                np.random.seed(seed)
                sentiment = np.random.normal(0, 0.4)
                sentiments[platform] = np.clip(sentiment, -1, 1)
            
            return sentiments
            
        except Exception as e:
            logger.warning(f"Failed to get social sentiment: {e}")
            return {'twitter': 0, 'reddit': 0, 'telegram': 0}
    
    def get_news_sentiment(self) -> float:
        """Get aggregated news sentiment"""
        try:
            # Placeholder for news sentiment analysis
            # In production, integrate with news APIs and NLP
            seed = int(time.time() / 3600)  # Hourly updates
            np.random.seed(seed)
            sentiment = np.random.normal(0, 0.35)
            return np.clip(sentiment, -1, 1)
            
        except Exception as e:
            logger.warning(f"Failed to get news sentiment: {e}")
            return 0.0


class OnChainAnalyzer:
    """Analyzes on-chain metrics and DeFi data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 600  # 10 minutes
        
    def get_exchange_flows(self) -> Dict[str, float]:
        """Get exchange inflow/outflow indicators"""
        try:
            # Placeholder for on-chain data
            # In production, integrate with Glassnode, CoinMetrics, etc.
            flows = {}
            
            metrics = ['inflow', 'outflow', 'net_flow', 'whale_activity']
            for metric in metrics:
                seed = int(time.time() / 3600) + hash(metric) % 1000
                np.random.seed(seed)
                value = np.random.normal(0, 0.3)
                flows[metric] = np.clip(value, -1, 1)
            
            return flows
            
        except Exception as e:
            logger.warning(f"Failed to get exchange flows: {e}")
            return {'inflow': 0, 'outflow': 0, 'net_flow': 0, 'whale_activity': 0}
    
    def get_defi_metrics(self) -> Dict[str, float]:
        """Get DeFi protocol metrics"""
        try:
            # Placeholder for DeFi data
            # In production, integrate with DeFiPulse, DeBank, etc.
            metrics = {}
            
            defi_indicators = ['tvl_change', 'yield_spread', 'liquidation_risk', 'governance_activity']
            for indicator in defi_indicators:
                seed = int(time.time() / 1800) + hash(indicator) % 1000
                np.random.seed(seed)
                value = np.random.normal(0, 0.25)
                metrics[indicator] = np.clip(value, -1, 1)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to get DeFi metrics: {e}")
            return {'tvl_change': 0, 'yield_spread': 0, 'liquidation_risk': 0, 'governance_activity': 0}
    
    def get_network_health(self) -> Dict[str, float]:
        """Get blockchain network health metrics"""
        try:
            # Placeholder for network metrics
            health_metrics = {}
            
            indicators = ['hash_rate_trend', 'difficulty_adjustment', 'mempool_congestion', 'node_count']
            for indicator in indicators:
                seed = int(time.time() / 7200) + hash(indicator) % 1000  # 2-hour intervals
                np.random.seed(seed)
                value = np.random.normal(0, 0.2)
                health_metrics[indicator] = np.clip(value, -1, 1)
            
            return health_metrics
            
        except Exception as e:
            logger.warning(f"Failed to get network health: {e}")
            return {'hash_rate_trend': 0, 'difficulty_adjustment': 0, 'mempool_congestion': 0, 'node_count': 0}


class MacroEconomicAnalyzer:
    """Analyzes macro-economic indicators and their crypto impact"""
    
    def __init__(self):
        self.cache = {}
        
    def get_macro_indicators(self) -> Dict[str, float]:
        """Get macro-economic indicators"""
        try:
            # Fetch key macro data
            indicators = {}
            
            # Dollar Index (DXY)
            dxy_data = yf.Ticker('DX-Y.NYB').history(period='5d', interval='1h')
            if not dxy_data.empty:
                dxy_returns = dxy_data['Close'].pct_change().dropna()
                indicators['dxy_momentum'] = dxy_returns.rolling(24).mean().iloc[-1] if len(dxy_returns) > 24 else 0
                indicators['dxy_volatility'] = dxy_returns.rolling(24).std().iloc[-1] if len(dxy_returns) > 24 else 0
            
            # Gold (safe haven indicator)
            gold_data = yf.Ticker('GC=F').history(period='5d', interval='1h')
            if not gold_data.empty:
                gold_returns = gold_data['Close'].pct_change().dropna()
                indicators['gold_momentum'] = gold_returns.rolling(24).mean().iloc[-1] if len(gold_returns) > 24 else 0
            
            # VIX (market fear)
            vix_data = yf.Ticker('^VIX').history(period='5d', interval='1h')
            if not vix_data.empty:
                vix_level = vix_data['Close'].iloc[-1]
                indicators['vix_level'] = (vix_level - 20) / 30  # Normalize around typical range
            
            # 10-Year Treasury Yield
            tnx_data = yf.Ticker('^TNX').history(period='5d', interval='1h')
            if not tnx_data.empty:
                yield_change = tnx_data['Close'].pct_change().rolling(24).mean().iloc[-1]
                indicators['yield_momentum'] = yield_change if not np.isnan(yield_change) else 0
            
            # Normalize all indicators to [-1, 1]
            for key, value in indicators.items():
                if not np.isnan(value):
                    indicators[key] = np.clip(value, -1, 1)
                else:
                    indicators[key] = 0
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Failed to get macro indicators: {e}")
            return {'dxy_momentum': 0, 'dxy_volatility': 0, 'gold_momentum': 0, 'vix_level': 0, 'yield_momentum': 0}
    
    def get_central_bank_sentiment(self) -> float:
        """Get central bank policy sentiment"""
        try:
            # Placeholder for central bank sentiment analysis
            # In production, analyze Fed minutes, ECB statements, etc.
            seed = int(time.time() / 86400)  # Daily updates
            np.random.seed(seed)
            sentiment = np.random.normal(0, 0.2)
            return np.clip(sentiment, -1, 1)
            
        except Exception:
            return 0.0


class AdvancedFeatureEngine:
    """Advanced feature engineering with unique signals"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.alt_data = AlternativeDataEngine()
        self.onchain = OnChainAnalyzer()
        self.macro = MacroEconomicAnalyzer()
        self.feature_history = {}
        self.price_cache = {}
        
    def fetch_enhanced_market_data(self, symbol: str, period: str = "7d", interval: str = "1m") -> pd.DataFrame:
        """Fetch market data with enhanced caching"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        try:
            if cache_key in self.price_cache:
                cached_data, timestamp = self.price_cache[cache_key]
                if time.time() - timestamp < 120:  # Cache for 2 minutes
                    return cached_data
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                # Add technical indicators using ta library
                data = self.add_technical_indicators(data)
                self.price_cache[cache_key] = (data, time.time())
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching enhanced data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            if len(data) < 50:
                return data
            
            # Momentum indicators
            data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            data['stoch'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
            data['williams_r'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
            
            # Volatility indicators
            bb = ta.volatility.BollingerBands(data['Close'])
            data['bb_high'] = bb.bollinger_hband()
            data['bb_low'] = bb.bollinger_lband()
            data['bb_width'] = bb.bollinger_wband()
            data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            
            # Volume indicators - Fixed API calls
            data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            data['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume']).chaikin_money_flow()
            
            # Calculate VWAP manually since ta library doesn't have VolumeSMAIndicator
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            data['vwap'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            
            # Trend indicators
            data['adx'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
            data['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
            
            # Additional volume indicators using correct API
            data['mfi'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
            data['ad'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
            data['em'] = ta.volume.EaseOfMovementIndicator(data['High'], data['Low'], data['Volume']).ease_of_movement()
            
            # Force Index
            data['fi'] = ta.volume.ForceIndexIndicator(data['Close'], data['Volume']).force_index()
            
            # Negative Volume Index
            data['nvi'] = ta.volume.NegativeVolumeIndexIndicator(data['Close'], data['Volume']).negative_volume_index()
            
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {e}")
            # Add fallback calculations for critical indicators
            try:
                # Simple RSI fallback
                if 'rsi' not in data.columns:
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    data['rsi'] = 100 - (100 / (1 + rs))
            
                # Simple VWAP fallback
                if 'vwap' not in data.columns:
                    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                    data['vwap'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
                
            except Exception as fallback_error:
                logger.warning(f"Fallback indicator calculation failed: {fallback_error}")
        
        return data
    
    def calculate_microstructure_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features"""
        features = {}
        
        try:
            if data.empty or len(data) < 100:
                return {}
            
            close = data['Close']
            volume = data['Volume']
            high = data['High']
            low = data['Low']
        
            # Price impact measures
            returns = close.pct_change().dropna()
            volume_norm = volume / volume.rolling(50).mean()
        
            # Kyle's Lambda (price impact)
            if len(returns) > 50 and len(volume_norm) > 50:
                min_len = min(len(returns), len(volume_norm))
                ret_subset = returns.iloc[-min_len:]
                vol_subset = volume_norm.iloc[-min_len:]
            
                correlation = ret_subset.abs().corr(vol_subset)
                features['kyle_lambda'] = correlation if not np.isnan(correlation) else 0
        
            # Bid-ask spread proxy
            hl_spread = ((high - low) / close).rolling(20).mean()
            features['spread_proxy'] = hl_spread.iloc[-1] if not hl_spread.empty else 0
        
            # Volume-price trend
            vpt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
            features['vpt_momentum'] = (vpt.iloc[-1] - vpt.iloc[-20]) / vpt.std() if len(vpt) > 20 else 0
        
            # Amihud illiquidity ratio
            if len(returns) > 20:
                illiquidity = (returns.abs() / volume_norm).rolling(20).mean()
                features['illiquidity'] = illiquidity.iloc[-1] if not illiquidity.empty else 0
        
            # Order flow imbalance proxy
            up_volume = volume.where(close > close.shift(1), 0).rolling(20).sum()
            down_volume = volume.where(close < close.shift(1), 0).rolling(20).sum()
            total_volume = up_volume + down_volume
            features['order_flow_imbalance'] = ((up_volume - down_volume) / total_volume).iloc[-1] if not total_volume.iloc[-1] == 0 else 0
        
            # Money Flow Index (if available from technical indicators)
            if 'mfi' in data.columns and not data['mfi'].empty:
                mfi_normalized = (data['mfi'].iloc[-1] - 50) / 50  # Normalize to [-1, 1]
                features['money_flow_index'] = mfi_normalized
        
            # Force Index (if available)
            if 'fi' in data.columns and not data['fi'].empty:
                fi_normalized = data['fi'].iloc[-1] / data['fi'].rolling(50).std().iloc[-1] if data['fi'].rolling(50).std().iloc[-1] != 0 else 0
                features['force_index'] = np.clip(fi_normalized, -1, 1)
        
            # VWAP deviation
            if 'vwap' in data.columns and not data['vwap'].empty:
                vwap_deviation = (close.iloc[-1] - data['vwap'].iloc[-1]) / data['vwap'].iloc[-1]
                features['vwap_deviation'] = np.clip(vwap_deviation * 10, -1, 1)  # Scale for better range
        
            # Normalize features
            for key, value in features.items():
                if not np.isnan(value) and np.isfinite(value):
                    features[key] = np.clip(value, -1, 1)
                else:
                    features[key] = 0
                
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
        
        return features
    
    def calculate_regime_clustering_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Use ML clustering to identify market regimes"""
        features = {}
        
        try:
            if data.empty or len(data) < 100:
                return {}
            
            # Prepare features for clustering
            close = data['Close']
            volume = data['Volume']
            returns = close.pct_change().dropna()
            
            if len(returns) < 50:
                return {}
            
            # Create feature matrix for clustering
            feature_matrix = []
            window = 20
            
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                window_volume = volume.iloc[i-window:i]
                
                # Statistical features
                ret_mean = window_returns.mean()
                ret_std = window_returns.std()
                ret_skew = window_returns.skew()
                ret_kurt = window_returns.kurtosis()
                vol_mean = window_volume.mean()
                
                feature_matrix.append([ret_mean, ret_std, ret_skew, ret_kurt, vol_mean])
            
            if len(feature_matrix) < 10:
                return {}
            
            feature_matrix = np.array(feature_matrix)
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix)
            
            # Fit clustering model
            try:
                clusters = self.kmeans.fit_predict(feature_matrix)
                current_regime = clusters[-1]
                
                # Regime persistence
                recent_regimes = clusters[-10:] if len(clusters) >= 10 else clusters
                regime_stability = np.mean(recent_regimes == current_regime)
                
                features['current_regime'] = (current_regime - 2) / 2  # Normalize to [-1, 1]
                features['regime_stability'] = regime_stability * 2 - 1  # Normalize to [-1, 1]
                
                # Regime transition probability
                if len(clusters) > 1:
                    transitions = np.sum(np.diff(clusters) != 0) / len(clusters)
                    features['regime_transition_prob'] = transitions * 2 - 1
                
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                
        except Exception as e:
            logger.error(f"Error in regime clustering: {e}")
            
        return features
    
    def calculate_cross_market_arbitrage(self, btc_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate cross-market arbitrage opportunities"""
        features = {}
        
        try:
            if btc_data.empty:
                return {}
            
            # Fetch related markets
            markets = {
                'GBTC': yf.Ticker('GBTC').history(period='2d', interval='1h'),
                'MSTR': yf.Ticker('MSTR').history(period='2d', interval='1h'),
                'COIN': yf.Ticker('COIN').history(period='2d', interval='1h')
            }
            
            btc_returns = btc_data['Close'].pct_change().dropna()
            
            for market_name, market_data in markets.items():
                if not market_data.empty:
                    market_returns = market_data['Close'].pct_change().dropna()
                    
                    # Align time series
                    min_len = min(len(btc_returns), len(market_returns))
                    if min_len > 20:
                        btc_aligned = btc_returns.iloc[-min_len:]
                        market_aligned = market_returns.iloc[-min_len:]
                        
                        # Calculate spread
                        spread = market_aligned - btc_aligned
                        features[f'{market_name.lower()}_spread'] = spread.rolling(10).mean().iloc[-1]
                        
                        # Mean reversion signal
                        spread_zscore = (spread.iloc[-1] - spread.rolling(20).mean().iloc[-1]) / spread.rolling(20).std().iloc[-1]
                        features[f'{market_name.lower()}_mean_reversion'] = np.clip(spread_zscore, -1, 1) if not np.isnan(spread_zscore) else 0
            
            # Normalize features
            for key, value in features.items():
                if not np.isnan(value) and np.isfinite(value):
                    features[key] = np.clip(value, -1, 1)
                else:
                    features[key] = 0
                    
        except Exception as e:
            logger.error(f"Error calculating arbitrage features: {e}")
            
        return features
    
    def generate_enhanced_btc_features(self) -> List[float]:
        """Generate enhanced 100-dimensional BTC feature vector"""
        logger.info("Generating enhanced BTC features...")
        
        # Fetch enhanced BTC data
        btc_data = self.fetch_enhanced_market_data('BTC-USD', period='7d', interval='1m')
        if btc_data.empty:
            logger.warning("No BTC data available, using zero vector")
            return [0.0] * 100
        
        features = []
        
        # Alternative data features (20 features)
        fear_greed = self.alt_data.get_fear_greed_index()
        google_trends = self.alt_data.get_google_trends_score()
        social_sentiment = self.alt_data.get_social_sentiment()
        news_sentiment = self.alt_data.get_news_sentiment()
        
        alt_features = [
            fear_greed,
            google_trends,
            news_sentiment,
            social_sentiment.get('twitter', 0),
            social_sentiment.get('reddit', 0),
            social_sentiment.get('telegram', 0),
        ]
        alt_features = (alt_features + [0] * 20)[:20]
        features.extend(alt_features)
        
        # On-chain features (20 features)
        exchange_flows = self.onchain.get_exchange_flows()
        defi_metrics = self.onchain.get_defi_metrics()
        network_health = self.onchain.get_network_health()
        
        onchain_features = [
            exchange_flows.get('inflow', 0),
            exchange_flows.get('outflow', 0),
            exchange_flows.get('net_flow', 0),
            exchange_flows.get('whale_activity', 0),
            defi_metrics.get('tvl_change', 0),
            defi_metrics.get('yield_spread', 0),
            defi_metrics.get('liquidation_risk', 0),
            defi_metrics.get('governance_activity', 0),
            network_health.get('hash_rate_trend', 0),
            network_health.get('difficulty_adjustment', 0),
            network_health.get('mempool_congestion', 0),
            network_health.get('node_count', 0),
        ]
        onchain_features = (onchain_features + [0] * 20)[:20]
        features.extend(onchain_features)
        
        # Macro-economic features (15 features)
        macro_indicators = self.macro.get_macro_indicators()
        cb_sentiment = self.macro.get_central_bank_sentiment()
        
        macro_features = [
            macro_indicators.get('dxy_momentum', 0),
            macro_indicators.get('dxy_volatility', 0),
            macro_indicators.get('gold_momentum', 0),
            macro_indicators.get('vix_level', 0),
            macro_indicators.get('yield_momentum', 0),
            cb_sentiment,
        ]
        macro_features = (macro_features + [0] * 15)[:15]
        features.extend(macro_features)
        
        # Microstructure features (15 features)
        microstructure = self.calculate_microstructure_features(btc_data)
        micro_features = [
            microstructure.get('kyle_lambda', 0),
            microstructure.get('spread_proxy', 0),
            microstructure.get('vpt_momentum', 0),
            microstructure.get('illiquidity', 0),
            microstructure.get('order_flow_imbalance', 0),
        ]
        micro_features = (micro_features + [0] * 15)[:15]
        features.extend(micro_features)
        
        # ML regime features (15 features)
        regime_features = self.calculate_regime_clustering_features(btc_data)
        ml_features = [
            regime_features.get('current_regime', 0),
            regime_features.get('regime_stability', 0),
            regime_features.get('regime_transition_prob', 0),
        ]
        ml_features = (ml_features + [0] * 15)[:15]
        features.extend(ml_features)
        
        # Cross-market arbitrage features (15 features)
        arbitrage_features = self.calculate_cross_market_arbitrage(btc_data)
        arb_features = [
            arbitrage_features.get('gbtc_spread', 0),
            arbitrage_features.get('gbtc_mean_reversion', 0),
            arbitrage_features.get('mstr_spread', 0),
            arbitrage_features.get('mstr_mean_reversion', 0),
            arbitrage_features.get('coin_spread', 0),
            arbitrage_features.get('coin_mean_reversion', 0),
        ]
        arb_features = (arb_features + [0] * 15)[:15]
        features.extend(arb_features)
        
        # Ensure exactly 100 features
        features = (features + [0] * 100)[:100]
        
        # Apply advanced smoothing
        features = self._advanced_smooth_features('BTC', features)
        
        # Clip to [-1, 1] range
        features = np.clip(features, -1, 1).tolist()
        
        logger.info(f"Generated {len(features)} enhanced BTC features")
        return features
    
    def generate_enhanced_asset_features(self, asset: str) -> List[float]:
        """Generate enhanced 2-dimensional features for non-BTC assets"""
        logger.info(f"Generating enhanced features for {asset}")
        
        # Map asset to Yahoo Finance symbol
        symbol_map = {
            'ETH': 'ETH-USD',
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'CADUSD': 'CADUSD=X',
            'NZDUSD': 'NZDUSD=X',
            'CHFUSD': 'CHFUSD=X',
            'XAUUSD': 'GC=F',
            'XAGUSD': 'SI=F',
        }
        
        symbol = symbol_map.get(asset, f'{asset}-USD')
        
        # Fetch enhanced asset data
        asset_data = self.fetch_enhanced_market_data(symbol, period='3d', interval='1m')
        btc_data = self.fetch_enhanced_market_data('BTC-USD', period='3d', interval='1m')
        
        features = [0.0, 0.0]
        
        try:
            if not asset_data.empty and not btc_data.empty:
                # Feature 1: Advanced momentum divergence
                asset_returns = asset_data['Close'].pct_change().dropna()
                btc_returns = btc_data['Close'].pct_change().dropna()
                
                min_len = min(len(asset_returns), len(btc_returns))
                if min_len > 30:
                    # Use multiple timeframes for momentum
                    short_momentum_asset = asset_returns.rolling(5).mean().iloc[-1]
                    long_momentum_asset = asset_returns.rolling(20).mean().iloc[-1]
                    short_momentum_btc = btc_returns.rolling(5).mean().iloc[-1]
                    long_momentum_btc = btc_returns.rolling(20).mean().iloc[-1]
                    
                    # Momentum acceleration divergence
                    asset_accel = short_momentum_asset - long_momentum_asset
                    btc_accel = short_momentum_btc - long_momentum_btc
                    features[0] = asset_accel - btc_accel
                
                # Feature 2: Enhanced asset-specific factor
                if asset == 'ETH':
                    # ETH: DeFi correlation and gas fees proxy
                    if 'atr' in asset_data.columns:
                        gas_proxy = asset_data['atr'].rolling(20).mean().iloc[-1] / asset_data['Close'].iloc[-1]
                        features[1] = np.clip(gas_proxy * 100, -1, 1)
                        
                elif asset in ['EURUSD', 'GBPUSD', 'CADUSD', 'NZDUSD', 'CHFUSD']:
                    # Forex: Interest rate differential proxy
                    if len(asset_returns) > 50:
                        # Use volatility regime and momentum combination
                        vol_regime = asset_returns.rolling(10).std().iloc[-1] / asset_returns.rolling(50).std().iloc[-1]
                        momentum = asset_returns.rolling(15).mean().iloc[-1]
                        features[1] = np.clip((vol_regime - 1) + momentum * 10, -1, 1)
                        
                elif asset in ['XAUUSD', 'XAGUSD']:
                    # Precious metals: Real yield proxy
                    if min_len > 20:
                        # Correlation with macro stress
                        macro_indicators = self.macro.get_macro_indicators()
                        vix_impact = macro_indicators.get('vix_level', 0)
                        yield_impact = macro_indicators.get('yield_momentum', 0)
                        features[1] = np.clip(vix_impact - yield_impact, -1, 1)
                        
        except Exception as e:
            logger.error(f"Error generating enhanced features for {asset}: {e}")
        
        # Apply advanced smoothing
        features = self._advanced_smooth_features(asset, features)
        
        # Clip to [-1, 1] range
        features = np.clip(features, -1, 1).tolist()
        
        logger.info(f"Generated enhanced features for {asset}: {features}")
        return features
    
    def _advanced_smooth_features(self, asset: str, features: List[float], alpha: float = 0.2) -> List[float]:
        """Apply advanced exponential smoothing with outlier detection"""
        if asset not in self.feature_history:
            self.feature_history[asset] = features
            return features
        
        previous = self.feature_history[asset]
        smoothed = []
        
        for i, (new_val, prev_val) in enumerate(zip(features, previous)):
            # Outlier detection
            if abs(new_val - prev_val) > 0.5:  # Large change detected
                # Use more conservative smoothing
                smoothed_val = alpha * 0.5 * new_val + (1 - alpha * 0.5) * prev_val
            else:
                # Normal smoothing
                smoothed_val = alpha * new_val + (1 - alpha) * prev_val
            
            smoothed.append(smoothed_val)
        
        self.feature_history[asset] = smoothed
        return smoothed


class EnhancedMANTISMiner:
    """Enhanced MANTIS miner with unique signal sources"""
    
    def __init__(self, wallet_name: str, wallet_hotkey: str, netuid: int = 123):
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.netuid = netuid
        
        # Initialize enhanced components
        self.feature_engine = AdvancedFeatureEngine()
        self.r2_uploader = R2Uploader()
        self.timelock = Timelock(DRAND_PUBLIC_KEY)
        
        # Initialize Bittensor components
        self.wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.subtensor = bt.subtensor(network="finney")
        
        # State
        self.is_committed = False
        self.last_submission_time = 0
        
        logger.info(f"Enhanced MANTIS Miner initialized for wallet {wallet_name}/{wallet_hotkey}")
    
    async def generate_multi_asset_embeddings(self) -> List[List[float]]:
        """Generate enhanced embeddings for all assets"""
        logger.info("Generating enhanced multi-asset embeddings...")
        
        embeddings = []
        
        for asset in ASSETS:
            if asset == "BTC":
                features = self.feature_engine.generate_enhanced_btc_features()
            else:
                features = self.feature_engine.generate_enhanced_asset_features(asset)
            
            embeddings.append(features)
            logger.info(f"Generated {len(features)} enhanced features for {asset}")
        
        return embeddings
    
    def encrypt_payload(self, embeddings: List[List[float]], lock_time_seconds: int = 30) -> Dict:
        """Encrypt embeddings with timelock"""
        logger.info("Encrypting enhanced payload...")
        
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
            
            # Encrypt
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.timelock.tle(target_round, plaintext, salt).hex()
            
            payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
            logger.info(f"Enhanced payload encrypted for round {target_round}")
            return payload
            
        except Exception as e:
            logger.error(f"Enhanced encryption failed: {e}")
            raise
    
    async def upload_payload(self, payload: Dict) -> str:
        """Upload payload to R2 and return public URL"""
        logger.info("Uploading enhanced payload to R2...")
        
        try:
            # Use hotkey as filename
            filename = self.wallet.hotkey.ss58_address
            
            # Upload to R2
            public_url = await self.r2_uploader.upload_json(payload, filename)
            
            logger.info(f"Enhanced payload uploaded to: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Enhanced upload failed: {e}")
            raise
    
    def commit_url_to_subnet(self, public_url: str):
        """Commit the public URL to the subnet (one-time setup)"""
        if self.is_committed:
            logger.info("Enhanced URL already committed to subnet")
            return
        
        try:
            logger.info(f"Committing enhanced URL to subnet: {public_url}")
            
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.netuid,
                data=public_url
            )
            
            self.is_committed = True
            logger.info("Enhanced URL successfully committed to subnet")
            
        except Exception as e:
            logger.error(f"Failed to commit enhanced URL: {e}")
            raise
    
    async def mining_cycle(self):
        """Execute one complete enhanced mining cycle"""
        try:
            start_time = time.time()
            
            # Generate enhanced embeddings
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
            
            logger.info(f"Enhanced mining cycle completed in {cycle_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Enhanced mining cycle failed: {e}")
            raise
    
    async def run(self, update_interval: int = 60):
        """Run the enhanced miner continuously"""
        logger.info(f"Starting Enhanced MANTIS miner with {update_interval}s update interval")
        
        while True:
            try:
                await self.mining_cycle()
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except KeyboardInterrupt:
                logger.info("Enhanced miner stopped by user")
                break
            except Exception as e:
                logger.error(f"Enhanced miner unexpected error: {e}")
                logger.info("Retrying in 30 seconds...")
                await asyncio.sleep(30)


async def main():
    parser = argparse.ArgumentParser(description="MANTIS Enhanced Competitive Miner v2")
    parser.add_argument("--wallet.name", required=True, help="Wallet name")
    parser.add_argument("--wallet.hotkey", required=True, help="Wallet hotkey")
    parser.add_argument("--netuid", type=int, default=123, help="Subnet netuid")
    parser.add_argument("--update-interval", type=int, default=60, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize enhanced miner
    miner = EnhancedMANTISMiner(
        wallet_name=getattr(args, 'wallet.name'),
        wallet_hotkey=getattr(args, 'wallet.hotkey'),
        netuid=args.netuid
    )
    
    # Run enhanced miner
    await miner.run(update_interval=args.update_interval)


if __name__ == "__main__":
    asyncio.run(main())
