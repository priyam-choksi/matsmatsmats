"""
Market Data Collector for LLM Trading Arena
Collects price data, technicals, news, and SEC filings for all tickers
Uses multiple data sources with fallbacks
"""

import yfinance as yf
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Load env from TradingAgent root
# Path: TradingAgent/agents/llm_arena/data_collector.py
# .env:  TradingAgent/.env
ROOT_DIR = Path(__file__).parent.parent.parent  # Go up 3 levels to TradingAgent
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: try current directory
    load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS = [
    "AAPL", "AMZN", "CVX", "GOOGL", "GS", "JNJ", "JPM", "KO", "LLY", "META",
    "MSFT", "NVDA", "PG", "QQQ", "SPY", "TSLA", "UNH", "V", "WMT", "XOM"
]

# Map tickers to CIK numbers for SEC EDGAR
TICKER_TO_CIK = {
    "AAPL": "0000320193", "AMZN": "0001018724", "CVX": "0000093410",
    "GOOGL": "0001652044", "GS": "0000886982", "JNJ": "0000200406",
    "JPM": "0000019617", "KO": "0000021344", "LLY": "0000059478",
    "META": "0001326801", "MSFT": "0000789019", "NVDA": "0001045810",
    "PG": "0000080424", "TSLA": "0001318605", "UNH": "0000731766",
    "V": "0001403161", "WMT": "0000104169", "XOM": "0000034088",
    "QQQ": None, "SPY": None  # ETFs don't have SEC filings
}

OUTPUT_DIR = ROOT_DIR / "outputs" / "market_data"
INTERVAL_DAYS = 3

# API Keys - check multiple possible env var names
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY") or os.getenv("VITE_FINNHUB_KEY") or ""
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY") or os.getenv("VITE_NEWSAPI_KEY") or ""
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY") or os.getenv("ALPHAVANTAGE_API_KEY") or ""
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "TradingArena research@example.com")


# =============================================================================
# MACRO DATA (Market-wide context)
# =============================================================================

def get_macro_data(target_date: str) -> dict:
    """Get macro indicators - SPY, VIX, Treasury yields, etc."""
    try:
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=30)
        
        macro = {}
        
        # SPY - S&P 500 proxy
        spy = yf.download("SPY", start=start_date, end=end_date + timedelta(days=1), progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy = spy[spy.index <= pd.to_datetime(target_date)]
            if not spy.empty:
                macro["spy"] = {
                    "price": round(float(spy['Close'].iloc[-1]), 2),
                    "change_1d_pct": round(((float(spy['Close'].iloc[-1]) / float(spy['Close'].iloc[-2])) - 1) * 100, 2) if len(spy) > 1 else None,
                    "change_7d_pct": round(((float(spy['Close'].iloc[-1]) / float(spy['Close'].iloc[-7])) - 1) * 100, 2) if len(spy) > 7 else None,
                }
        
        # VIX - Volatility index
        vix = yf.download("^VIX", start=start_date, end=end_date + timedelta(days=1), progress=False)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix = vix[vix.index <= pd.to_datetime(target_date)]
            if not vix.empty:
                vix_level = float(vix['Close'].iloc[-1])
                macro["vix"] = {
                    "level": round(vix_level, 2),
                    "signal": "HIGH_FEAR" if vix_level > 25 else "ELEVATED" if vix_level > 20 else "LOW_FEAR" if vix_level < 15 else "NORMAL"
                }
        
        # 10-Year Treasury Yield
        tny = yf.download("^TNX", start=start_date, end=end_date + timedelta(days=1), progress=False)
        if not tny.empty:
            if isinstance(tny.columns, pd.MultiIndex):
                tny.columns = tny.columns.get_level_values(0)
            tny = tny[tny.index <= pd.to_datetime(target_date)]
            if not tny.empty:
                macro["treasury_10y"] = {
                    "yield_pct": round(float(tny['Close'].iloc[-1]), 2)
                }
        
        # Dollar Index
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date + timedelta(days=1), progress=False)
        if not dxy.empty:
            if isinstance(dxy.columns, pd.MultiIndex):
                dxy.columns = dxy.columns.get_level_values(0)
            dxy = dxy[dxy.index <= pd.to_datetime(target_date)]
            if not dxy.empty:
                macro["dollar_index"] = {
                    "level": round(float(dxy['Close'].iloc[-1]), 2)
                }
        
        # Gold (fear indicator)
        gold = yf.download("GC=F", start=start_date, end=end_date + timedelta(days=1), progress=False)
        if not gold.empty:
            if isinstance(gold.columns, pd.MultiIndex):
                gold.columns = gold.columns.get_level_values(0)
            gold = gold[gold.index <= pd.to_datetime(target_date)]
            if not gold.empty:
                macro["gold"] = {
                    "price": round(float(gold['Close'].iloc[-1]), 2)
                }
        
        # Determine market regime
        if macro.get("spy") and macro.get("vix"):
            spy_trend = macro["spy"].get("change_7d_pct", 0) or 0
            vix_level = macro["vix"]["level"]
            
            if spy_trend > 2 and vix_level < 18:
                regime = "BULL_LOW_VOL"
            elif spy_trend > 0 and vix_level < 22:
                regime = "BULL_NORMAL"
            elif spy_trend < -2 and vix_level > 25:
                regime = "BEAR_HIGH_VOL"
            elif spy_trend < 0:
                regime = "BEAR_NORMAL"
            else:
                regime = "SIDEWAYS"
            
            macro["regime"] = regime
        
        return macro
        
    except Exception as e:
        print(f"    ⚠ Macro data error: {e}")
        return {}


# =============================================================================
# FUNDAMENTAL DATA
# =============================================================================

def get_fundamentals(ticker: str) -> dict:
    """Get fundamental data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            # Valuation
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            
            # Profitability
            "profit_margin": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else None,
            "operating_margin": round(info.get("operatingMargins", 0) * 100, 2) if info.get("operatingMargins") else None,
            "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
            "roa": round(info.get("returnOnAssets", 0) * 100, 2) if info.get("returnOnAssets") else None,
            
            # Growth
            "revenue_growth": round(info.get("revenueGrowth", 0) * 100, 2) if info.get("revenueGrowth") else None,
            "earnings_growth": round(info.get("earningsGrowth", 0) * 100, 2) if info.get("earningsGrowth") else None,
            
            # Financial Health
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            
            # Dividends
            "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
            
            # Price Context
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_day_avg": info.get("fiftyDayAverage"),
            "two_hundred_day_avg": info.get("twoHundredDayAverage"),
            
            # Analyst Views
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            
            # Company Info
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
        
        # Calculate some derived metrics
        if fundamentals["fifty_two_week_high"] and fundamentals["fifty_two_week_low"]:
            current = info.get("currentPrice") or info.get("regularMarketPrice")
            if current:
                high = fundamentals["fifty_two_week_high"]
                low = fundamentals["fifty_two_week_low"]
                fundamentals["pct_from_52w_high"] = round(((current - high) / high) * 100, 2)
                fundamentals["pct_from_52w_low"] = round(((current - low) / low) * 100, 2)
        
        # Clean up None values
        return {k: v for k, v in fundamentals.items() if v is not None}
        
    except Exception as e:
        print(f"    ⚠ {ticker}: Fundamentals error - {e}")
        return {}


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series) -> tuple:
    """Calculate MACD indicator"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> tuple:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, sma, lower


# =============================================================================
# PRICE DATA (yfinance)
# =============================================================================

def get_price_data(ticker: str, target_date: str) -> Optional[dict]:
    """Get OHLCV data and technicals from yfinance"""
    try:
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=90)
        
        df = yf.download(
            ticker, 
            start=start_date.strftime("%Y-%m-%d"), 
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False
        )
        
        if df.empty or len(df) < 20:
            print(f"    ⚠ {ticker}: Insufficient price data")
            return None
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Filter to target date
        df.index = pd.to_datetime(df.index)
        target_dt = pd.to_datetime(target_date)
        df = df[df.index <= target_dt]
        
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        latest_date = df.index[-1]
        close_price = float(latest['Close'])
        
        # Calculate all technicals
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean() if len(df) >= 200 else None
        df['rsi'] = calculate_rsi(df['Close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['Close'])
        
        # Volume analysis
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        volume_ratio = float(latest['Volume']) / float(df['volume_sma'].iloc[-1]) if pd.notna(df['volume_sma'].iloc[-1]) else 1.0
        
        # Price changes
        def safe_pct_change(current, periods_back):
            if len(df) > periods_back:
                prev = float(df['Close'].iloc[-periods_back-1])
                return round(((current - prev) / prev) * 100, 2)
            return None
        
        return {
            "open": round(float(latest['Open']), 2),
            "high": round(float(latest['High']), 2),
            "low": round(float(latest['Low']), 2),
            "close": close_price,
            "volume": int(latest['Volume']),
            "actual_date": latest_date.strftime("%Y-%m-%d"),
            "change_1d_pct": safe_pct_change(close_price, 1),
            "change_7d_pct": safe_pct_change(close_price, 7),
            "change_30d_pct": safe_pct_change(close_price, 30),
            "technicals": {
                "rsi_14": round(float(df['rsi'].iloc[-1]), 2) if pd.notna(df['rsi'].iloc[-1]) else None,
                "sma_20": round(float(df['sma_20'].iloc[-1]), 2) if pd.notna(df['sma_20'].iloc[-1]) else None,
                "sma_50": round(float(df['sma_50'].iloc[-1]), 2) if pd.notna(df['sma_50'].iloc[-1]) else None,
                "macd": round(float(df['macd'].iloc[-1]), 3) if pd.notna(df['macd'].iloc[-1]) else None,
                "macd_signal": round(float(df['macd_signal'].iloc[-1]), 3) if pd.notna(df['macd_signal'].iloc[-1]) else None,
                "macd_histogram": round(float(df['macd_hist'].iloc[-1]), 3) if pd.notna(df['macd_hist'].iloc[-1]) else None,
                "bb_upper": round(float(df['bb_upper'].iloc[-1]), 2) if pd.notna(df['bb_upper'].iloc[-1]) else None,
                "bb_lower": round(float(df['bb_lower'].iloc[-1]), 2) if pd.notna(df['bb_lower'].iloc[-1]) else None,
                "price_vs_sma20_pct": round(((close_price / float(df['sma_20'].iloc[-1])) - 1) * 100, 2) if pd.notna(df['sma_20'].iloc[-1]) else None,
                "price_vs_sma50_pct": round(((close_price / float(df['sma_50'].iloc[-1])) - 1) * 100, 2) if pd.notna(df['sma_50'].iloc[-1]) else None,
                "volume_ratio": round(volume_ratio, 2),
            }
        }
        
    except Exception as e:
        print(f"    ✗ {ticker}: Price error - {e}")
        return None


# =============================================================================
# NEWS DATA (Finnhub + NewsAPI fallback)
# =============================================================================

def get_finnhub_news(ticker: str, target_date: str) -> list:
    """Get news from Finnhub"""
    if not FINNHUB_API_KEY:
        return []
    
    try:
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=7)
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": target_date,
            "token": FINNHUB_API_KEY
        }
        
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code == 429:
            time.sleep(1)
            return []
        
        resp.raise_for_status()
        articles = resp.json()
        
        return [{
            "headline": a.get("headline", "")[:200],
            "source": a.get("source", "finnhub"),
            "datetime": a.get("datetime", 0),
            "summary": a.get("summary", "")[:300],
            "url": a.get("url", "")
        } for a in articles[:5]]
        
    except Exception as e:
        return []


def get_newsapi_news(ticker: str, company_name: str, target_date: str) -> list:
    """Get news from NewsAPI (fallback)"""
    if not NEWSAPI_KEY:
        return []
    
    try:
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=7)
        
        # NewsAPI free tier only goes back ~1 month
        if (datetime.now() - end_date).days > 30:
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{ticker} OR {company_name}",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": target_date,
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": NEWSAPI_KEY
        }
        
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        return [{
            "headline": a.get("title", "")[:200],
            "source": a.get("source", {}).get("name", "newsapi"),
            "datetime": a.get("publishedAt", ""),
            "summary": a.get("description", "")[:300],
            "url": a.get("url", "")
        } for a in data.get("articles", [])[:5]]
        
    except Exception:
        return []


# Company names for NewsAPI search
TICKER_TO_COMPANY = {
    "AAPL": "Apple", "AMZN": "Amazon", "CVX": "Chevron", "GOOGL": "Google Alphabet",
    "GS": "Goldman Sachs", "JNJ": "Johnson Johnson", "JPM": "JPMorgan", "KO": "Coca-Cola",
    "LLY": "Eli Lilly", "META": "Meta Facebook", "MSFT": "Microsoft", "NVDA": "Nvidia",
    "PG": "Procter Gamble", "QQQ": "Nasdaq ETF", "SPY": "S&P 500 ETF", "TSLA": "Tesla",
    "UNH": "UnitedHealth", "V": "Visa", "WMT": "Walmart", "XOM": "Exxon"
}


def get_news(ticker: str, target_date: str) -> list:
    """Get news with fallbacks"""
    # Try Finnhub first
    news = get_finnhub_news(ticker, target_date)
    
    # If no news, try NewsAPI
    if not news:
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        news = get_newsapi_news(ticker, company, target_date)
    
    return news


# =============================================================================
# SEC EDGAR FILINGS
# =============================================================================

def get_sec_filings(ticker: str, target_date: str) -> list:
    """Get recent SEC filings from EDGAR"""
    cik = TICKER_TO_CIK.get(ticker)
    if not cik:
        return []
    
    try:
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=90)  # Last 90 days of filings
        
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {"User-Agent": SEC_USER_AGENT}
        
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocument", [])
        accessions = recent.get("accessionNumber", [])
        
        for i, (form, date, desc, acc) in enumerate(zip(forms, dates, descriptions, accessions)):
            filing_date = datetime.strptime(date, "%Y-%m-%d")
            
            # Only include filings before target date and within range
            if filing_date <= end_date and filing_date >= start_date:
                # Only include significant filings
                if form in ["10-K", "10-Q", "8-K", "4", "13F-HR"]:
                    filings.append({
                        "form": form,
                        "date": date,
                        "description": desc[:100],
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc.replace('-', '')}/{desc}"
                    })
            
            if len(filings) >= 5:
                break
        
        return filings
        
    except Exception as e:
        return []


# =============================================================================
# MAIN COLLECTION LOGIC
# =============================================================================

def collect_for_date(target_date: str, force: bool = False) -> dict:
    """Collect data for all tickers for a specific date"""
    date_dir = OUTPUT_DIR / target_date
    date_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"success": [], "failed": [], "skipped": []}
    
    print(f"\n{'='*60}")
    print(f"Collecting data for {target_date}")
    print(f"{'='*60}")
    
    # Get macro data once for the date (shared across all tickers)
    print("  Fetching macro data...")
    macro_data = get_macro_data(target_date)
    if macro_data:
        regime = macro_data.get("regime", "UNKNOWN")
        vix = macro_data.get("vix", {}).get("level", "N/A")
        print(f"  ✓ Macro: Regime={regime} | VIX={vix}")
    
    # Save macro data separately
    macro_path = date_dir / "_macro.json"
    with open(macro_path, "w") as f:
        json.dump({"date": target_date, "macro": macro_data}, f, indent=2)
    
    print()
    for ticker in TICKERS:
        output_path = date_dir / f"{ticker}.json"
        
        if output_path.exists() and not force:
            results["skipped"].append(ticker)
            continue
        
        # Get price data (required)
        price_data = get_price_data(ticker, target_date)
        if not price_data:
            results["failed"].append(ticker)
            continue
        
        # Get fundamentals
        fundamentals = get_fundamentals(ticker)
        
        # Get news (optional)
        news = get_news(ticker, target_date)
        
        # Get SEC filings (optional)
        sec_filings = get_sec_filings(ticker, target_date)
        
        # Build complete record
        data = {
            "ticker": ticker,
            "target_date": target_date,
            "actual_date": price_data.pop("actual_date"),
            "collected_at": datetime.now().isoformat(),
            "price": {
                "open": price_data["open"],
                "high": price_data["high"],
                "low": price_data["low"],
                "close": price_data["close"],
                "volume": price_data["volume"],
                "change_1d_pct": price_data["change_1d_pct"],
                "change_7d_pct": price_data["change_7d_pct"],
                "change_30d_pct": price_data["change_30d_pct"],
            },
            "technicals": price_data["technicals"],
            "fundamentals": fundamentals,
            "macro": macro_data,  # Include macro context
            "news": news,
            "news_count": len(news),
            "sec_filings": sec_filings,
            "sec_filings_count": len(sec_filings),
            "data_quality": {
                "has_price": True,
                "has_technicals": price_data["technicals"]["rsi_14"] is not None,
                "has_fundamentals": len(fundamentals) > 5,
                "has_macro": len(macro_data) > 0,
                "has_news": len(news) > 0,
                "has_sec_filings": len(sec_filings) > 0
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Status output
        pe = fundamentals.get("pe_ratio", "N/A")
        pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A"
        
        results["success"].append(ticker)
        print(f"  ✓ {ticker}: ${price_data['close']:.2f} | RSI:{price_data['technicals']['rsi_14']:.0f} | P/E:{pe_str} | News:{len(news)}")
    
    print(f"\nSummary: {len(results['success'])} collected, {len(results['skipped'])} skipped, {len(results['failed'])} failed")
    
    return results


def generate_dates(start_date: str, end_date: str, interval_days: int = 3) -> list:
    """Generate list of trading dates"""
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current <= end:
        if current.weekday() < 5:  # Weekdays only
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    
    return dates


def backfill(start_date: str, end_date: str, interval_days: int = 3):
    """Backfill market data"""
    dates = generate_dates(start_date, end_date, interval_days)
    
    print(f"\n{'#'*60}")
    print(f"BACKFILL: {start_date} to {end_date}")
    print(f"Interval: {interval_days} days | Total dates: {len(dates)}")
    print(f"Tickers: {len(TICKERS)}")
    print(f"{'#'*60}")
    print(f"\nNote: News data may be sparse for dates > 1 month old")
    print(f"      SEC filings should be available for all dates")
    
    all_results = {"total_success": 0, "total_failed": 0, "total_skipped": 0}
    
    for i, date in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}]", end="")
        results = collect_for_date(date)
        
        all_results["total_success"] += len(results["success"])
        all_results["total_failed"] += len(results["failed"])
        all_results["total_skipped"] += len(results["skipped"])
    
    print(f"\n{'#'*60}")
    print(f"BACKFILL COMPLETE")
    print(f"Total: {all_results['total_success']} collected, {all_results['total_skipped']} skipped, {all_results['total_failed']} failed")
    print(f"{'#'*60}")
    
    return all_results


def collect_latest():
    """Collect data for today"""
    today = datetime.now()
    if today.weekday() == 5:
        today -= timedelta(days=1)
    elif today.weekday() == 6:
        today -= timedelta(days=2)
    
    collect_for_date(today.strftime("%Y-%m-%d"))


def get_collection_status() -> dict:
    """Get collection status"""
    if not OUTPUT_DIR.exists():
        return {"dates_collected": 0, "date_range": None}
    
    dates = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    
    if not dates:
        return {"dates_collected": 0, "date_range": None}
    
    # Check data quality for latest date
    latest_dir = OUTPUT_DIR / dates[-1]
    files_count = len(list(latest_dir.glob("*.json")))
    
    return {
        "dates_collected": len(dates),
        "date_range": {"start": dates[0], "end": dates[-1]},
        "latest_date": dates[-1],
        "latest_tickers": files_count,
        "output_dir": str(OUTPUT_DIR)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Data Collector")
    parser.add_argument("--backfill", action="store_true", help="Backfill historical data")
    parser.add_argument("--start", type=str, default="2024-06-20", help="Start date")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument("--interval", type=int, default=3, help="Days between samples")
    parser.add_argument("--latest", action="store_true", help="Collect latest only")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--date", type=str, help="Collect specific date")
    parser.add_argument("--force", action="store_true", help="Force re-collection")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_collection_status()
        print(json.dumps(status, indent=2))
    elif args.backfill:
        backfill(args.start, args.end, args.interval)
    elif args.latest:
        collect_latest()
    elif args.date:
        collect_for_date(args.date, force=args.force)
    else:
        print("Market Data Collector")
        print("=" * 40)
        status = get_collection_status()
        print(f"Dates collected: {status['dates_collected']}")
        if status.get('date_range'):
            print(f"Range: {status['date_range']['start']} to {status['date_range']['end']}")
        print("\nAPI Status:")
        print(f"  Finnhub: {'✓ Configured' if FINNHUB_API_KEY else '✗ Not set'}")
        print(f"  NewsAPI: {'✓ Configured' if NEWSAPI_KEY else '✗ Not set'}")
        print(f"  SEC EDGAR: ✓ No key needed")
        print(f"  yfinance: ✓ No key needed")
        print("\nUsage:")
        print("  python data_collector.py --backfill")
        print("  python data_collector.py --latest")
        print("  python data_collector.py --date 2024-07-15")