import pywt
from arch import arch_model
import yfinance as yf
import pandas as pd
import aiohttp
import asyncio
from pycoingecko import CoinGeckoAPI
import numpy as np

cg = CoinGeckoAPI()

async def _fetch_price_source(session, url, parse_json=True):
    async with session.get(url, timeout=5) as resp:
        resp.raise_for_status()
        if parse_json:
            return await resp.json()
        else:
            return await resp.text()

async def _get_price_from_sources(session, source_list):
    for name, url, parser in source_list:
        try:
            parse_json = not url.endswith("e=csv")
            data = await _fetch_price_source(session, url, parse_json=parse_json)
            price = parser(data)
            if price is not None:
                return price
        except Exception:
            continue
    return None

async def get_asset_prices(session: aiohttp.ClientSession) -> dict[str, float] | None:
    sources = {
        "BTC": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=BTC-USD",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("CoinGecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            lambda data: data["bitcoin"]["usd"] if "bitcoin" in data else None),
            ("Bitstamp", "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            lambda data: float(data["last"]) if "last" in data else None)
        ],
        "ETH": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=ETH-USD",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("CoinGecko", "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
            lambda data: data["ethereum"]["usd"] if "ethereum" in data else None),
            ("Bitstamp", "https://www.bitstamp.net/api/v2/ticker/ethusd/",
            lambda data: float(data["last"]) if "last" in data else None)
        ],
        "EURUSD": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=EURUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("FreeForex", "https://www.freeforexapi.com/api/live?pairs=EURUSD",
            lambda data: data["rates"]["EURUSD"]["rate"] if "rates" in data else None),
            ("Stooq", "https://stooq.com/q/l/?s=eurusd&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else float(text.split(',')[6]))
        ],
        "GBPUSD": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=GBPUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("FreeForex", "https://www.freeforexapi.com/api/live?pairs=GBPUSD",
            lambda data: data["rates"]["GBPUSD"]["rate"] if "rates" in data else None),
            ("Stooq", "https://stooq.com/q/l/?s=gbpusd&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else float(text.split(',')[6]))
        ],
        "CADUSD": [  # CADUSD = 1 CAD in USD, invert USD/CAD
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=CADUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("FreeForex", "https://www.freeforexapi.com/api/live?pairs=USDCAD",
            lambda data: (1/ data["rates"]["USDCAD"]["rate"]) if "rates" in data else None),
            ("Stooq", "https://stooq.com/q/l/?s=usdcad&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else 1/ float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else 1/ float(text.split(',')[6]))
        ],
        "NZDUSD": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=NZDUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("FreeForex", "https://www.freeforexapi.com/api/live?pairs=NZDUSD",
            lambda data: data["rates"]["NZDUSD"]["rate"] if "rates" in data else None),
            ("Stooq", "https://stooq.com/q/l/?s=nzdusd&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else float(text.split(',')[6]))
        ],
        "CHFUSD": [  # CHFUSD = 1 CHF in USD, invert USD/CHF
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=CHFUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("FreeForex", "https://www.freeforexapi.com/api/live?pairs=USDCHF",
            lambda data: (1/ data["rates"]["USDCHF"]["rate"]) if "rates" in data else None),
            ("Stooq", "https://stooq.com/q/l/?s=usdchf&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else 1/ float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else 1/ float(text.split(',')[6]))
        ],
        "XAUUSD": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=XAUUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("Stooq", "https://stooq.com/q/l/?s=xauusd&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else float(text.split(',')[6])),
        ],
        "XAGUSD": [
            ("Yahoo", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=XAGUSD=X",
            lambda data: data["quoteResponse"]["result"][0]["regularMarketPrice"] if data["quoteResponse"]["result"] else None),
            ("Stooq", "https://stooq.com/q/l/?s=xagusd&f=sd2t2ohlcv&e=csv",
            lambda text: None if not text or "N/D" in text
                        else float(text.splitlines()[1].split(',')[6]) if text.startswith("Symbol")
                        else float(text.split(',')[6])),
        ]
    }

    prices = {}
    tasks = {asset: asyncio.create_task(_get_price_from_sources(session, srcs)) for asset, srcs in sources.items()}
    for asset, task in tasks.items():
        price = await task
        if price is not None:
            prices[asset] = price

    print(f"Fetched prices for {len(prices)} assets: {prices}")
    return prices

def fetch_recent_prices(asset: str, minutes: int = 120) -> pd.DataFrame:
    if asset == "BTC":
        cg_id = "bitcoin"
        vs = cg.get_coin_market_chart_by_id(id=cg_id, vs_currency='usd', days=1)
        df = pd.DataFrame(vs['prices'], columns=['time', 'price'])
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df[-minutes:]

    # Map to Yahoo Finance symbols
    yahoo_symbols = {
        "ETH": "ETH-USD",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "CADUSD": "CADUSD=X",
        "NZDUSD": "NZDUSD=X",
        "CHFUSD": "CHFUSD=X",
        "XAUUSD": "GC=F",   # Gold Futures
        "XAGUSD": "SI=F",   # Silver Futures
    }

    symbol = yahoo_symbols.get(asset)
    if not symbol:
        raise ValueError(f"fetch_recent_prices not supported for asset: {asset}")

    # Download 1-day of data at 1-minute intervals
    df = yf.download(symbol, interval="1m", period="1d", progress=False, auto_adjust=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No price data for {asset} from Yahoo Finance.")
    
    df = df[['Close']].rename(columns={'Close': 'price'})
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    return df[-minutes:]

def denoise_wavelet(ts: pd.Series) -> pd.Series:
    coeffs = pywt.wavedec(ts.values, 'db4', level=3)
    cleaned = pywt.waverec(coeffs[:-1] + [np.zeros_like(coeffs[-1])], 'db4')
    return pd.Series(cleaned, index=ts.index)

def compute_garch_sigma(returns: pd.Series) -> pd.Series:
    res = arch_model(returns * 100, p=1, o=0, q=1, vol='GARCH', dist='normal', rescale=False).fit(disp='off')
    return res.conditional_volatility / 100
