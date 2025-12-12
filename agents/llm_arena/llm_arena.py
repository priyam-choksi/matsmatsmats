"""
LLM Trading Arena - Production Version
======================================
Features:
- Checkpoint/Resume: Crash at round 300? Restart from 300
- Progress tracking with ETA
- Rate limiting with exponential backoff
- Graceful degradation: If one provider fails, others continue
- Configurable timing
- Mid-run performance stats

Data Structure:
- Rounds 1-90: Rich workflow data (6-phase pipeline with debates)
- Rounds 91-473: Lite market data (technicals, fundamentals, news)

Usage:
    python llm_arena_production.py --run                    # Run all rounds
    python llm_arena_production.py --run --rounds 50        # Run 50 rounds
    python llm_arena_production.py --resume                 # Resume from checkpoint
    python llm_arena_production.py --status                 # Show status
    python llm_arena_production.py --reset                  # Clear checkpoint
"""

import json
import os
import time
import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import requests
from dotenv import load_dotenv

load_dotenv(r"F:\DAMG 7374_GENAI\TradingAgent\.env")  # ADD THIS LINE


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.parent
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""

# Data paths
WORKFLOW_DATA_DIR = ROOT_DIR / "outputs" / "workflows"
MARKET_DATA_DIR = ROOT_DIR / "outputs" / "market_data"
ARENA_OUTPUT_DIR = ROOT_DIR / "outputs" / "llm_arena"
ARENA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trading configuration
TICKERS = [
    "AAPL", "AMZN", "CVX", "GOOGL", "GS", "JNJ", "JPM", "KO", "LLY", "META",
    "MSFT", "NVDA", "PG", "QQQ", "SPY", "TSLA", "UNH", "V", "WMT", "XOM"
]
STARTING_CAPITAL = 250_000
MAX_POSITION_PCT = 0.40

# =============================================================================
# TIMING CONFIGURATION - ADJUST THESE FOR YOUR NEEDS
# =============================================================================

TIMING = {
    "inter_model_delay": 3,       # Seconds between each model
    "inter_round_delay": 3,      # Seconds after each round
    "request_timeout": 90,        # API timeout
    "max_retries": 3,             # Retries per failed call
    "checkpoint_every": 5,        # Save checkpoint every N rounds
    "failure_cooldown": 300,      # Wait 5 minutes on failure before retry (if --wait mode)
    "max_failure_retries": 3,     # Max times to retry a failed round (if --wait mode)
}

# =============================================================================
# MODEL CONFIGURATION - OPTIMIZED FOR RATE LIMITS
# =============================================================================

# Provider-specific delays (seconds between requests)
PROVIDER_DELAYS = {
    "openai": 0.5,      # Paid - generous limits
    "groq": 2,        # 30-60 req/min depending on model
}

# OPTION 1: Best for 473 rounds (avoids Google entirely)
ARENA_MODELS = {
    "GPT-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "Llama-3.3-70B": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Llama-4-Maverick": {"provider": "groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct"},
    "Kimi-K2": {"provider": "groq", "model": "moonshotai/kimi-k2-instruct-0905"},  # Updated!
    "Qwen3-32B": {"provider": "groq", "model": "qwen/qwen3-32b"},
    "GPT-OSS-120B": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "Allam-2-7B": {"provider": "groq", "model": "allam-2-7b"},  # Arabic-focused but multilingual
}
# =============================================================================
# RATE LIMITING & RETRIES
# =============================================================================

_last_request_time = {}
_provider_stats = {p: {"requests": 0, "failures": 0} for p in PROVIDER_DELAYS}


class ModelFailedError(Exception):
    """Raised when a model fails after all retries - signals to stop the round"""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"{provider}: {message}")


def call_with_retry(provider: str, api_func, *args, **kwargs) -> str:
    """
    Call API with rate limiting and retry logic.
    RAISES ModelFailedError if all retries fail - we don't want to skip any model!
    """
    
    # Enforce delay between requests to same provider
    last_time = _last_request_time.get(provider, 0)
    min_delay = PROVIDER_DELAYS.get(provider, 2.0)
    elapsed = time.time() - last_time
    
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)
    
    last_error = None
    
    # Retry loop
    for attempt in range(TIMING["max_retries"]):
        try:
            _last_request_time[provider] = time.time()
            result = api_func(*args, **kwargs)
            _provider_stats[provider]["requests"] += 1
            return result
            
        except requests.exceptions.HTTPError as e:
            _provider_stats[provider]["failures"] += 1
            last_error = str(e)
            
            if "429" in str(e):
                wait = (60 * (attempt + 1)) + random.uniform(0, 10)  # Longer waits
                print(f"        ⚠️ Rate limited ({provider}), waiting {wait:.0f}s (attempt {attempt+1}/{TIMING['max_retries']})")
                time.sleep(wait)
            else:
                print(f"        ⚠️ HTTP error: {e} (attempt {attempt+1}/{TIMING['max_retries']})")
                time.sleep(10 * (attempt + 1))
                
        except requests.exceptions.Timeout:
            _provider_stats[provider]["failures"] += 1
            last_error = "Timeout"
            wait = 20 * (attempt + 1)
            print(f"        ⚠️ Timeout ({provider}), retrying in {wait}s (attempt {attempt+1}/{TIMING['max_retries']})")
            time.sleep(wait)
            
        except requests.exceptions.ConnectionError as e:
            _provider_stats[provider]["failures"] += 1
            last_error = f"Connection error: {e}"
            wait = 30 * (attempt + 1)
            print(f"        ⚠️ Connection error ({provider}), retrying in {wait}s")
            time.sleep(wait)
            
        except Exception as e:
            _provider_stats[provider]["failures"] += 1
            last_error = f"{type(e).__name__}: {e}"
            print(f"        ⚠️ Error ({provider}): {last_error} (attempt {attempt+1}/{TIMING['max_retries']})")
            time.sleep(15 * (attempt + 1))
    
    # All retries failed - raise error to stop the round
    raise ModelFailedError(provider, f"All {TIMING['max_retries']} retries failed. Last error: {last_error}")


# =============================================================================
# LLM API CALLS
# =============================================================================

def _google_api(prompt: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2500}}
    resp = requests.post(url, headers={"Content-Type": "application/json"},
                        params={"key": GOOGLE_API_KEY}, json=payload,
                        timeout=TIMING["request_timeout"])
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _groq_api(prompt: str, model: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7, "max_tokens": 2500}
    resp = requests.post(url, headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"}, json=payload,
                        timeout=TIMING["request_timeout"])
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _mistral_api(prompt: str, model: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7, "max_tokens": 2500}
    resp = requests.post(url, headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json"}, json=payload,
                        timeout=TIMING["request_timeout"])
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _openai_api(prompt: str, model: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7, "max_tokens": 2500}
    resp = requests.post(url, headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"}, json=payload,
                        timeout=TIMING["request_timeout"])
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm(provider: str, model: str, prompt: str) -> str:
    """
    Route to correct provider with retry logic.
    RAISES ModelFailedError if the call fails - we need ALL models to participate!
    """
    # All Groq-hosted models (including qwen, kimi, llama-4, etc.)
    groq_models = ["qwen/", "moonshotai/", "meta-llama/", "llama-", "openai/gpt-oss", "groq/", "allam"]
    
    if provider == "groq" or any(model.startswith(prefix) for prefix in groq_models):
        if GROQ_API_KEY:
            return call_with_retry("groq", _groq_api, prompt, model)
        else:
            raise ModelFailedError("groq", "No API key configured for groq")
    elif provider == "google" and GOOGLE_API_KEY:
        return call_with_retry(provider, _google_api, prompt, model)
    elif provider == "mistral" and MISTRAL_API_KEY:
        return call_with_retry(provider, _mistral_api, prompt, model)
    elif provider == "openai" and OPENAI_API_KEY:
        return call_with_retry(provider, _openai_api, prompt, model)
    else:
        raise ModelFailedError(provider, f"No API key configured for {provider}")


# =============================================================================
# PORTFOLIO CLASSES
# =============================================================================

@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.shares * self.avg_cost
        return ((self.market_value - cost) / cost * 100) if cost else 0


@dataclass
class Trade:
    timestamp: str
    ticker: str
    action: str
    shares: float
    price: float
    amount_usd: float
    reasoning: str
    portfolio_value_after: float
    model_reasoning: str = ""


@dataclass
class Portfolio:
    name: str
    cash: float = STARTING_CAPITAL
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    
    def update_prices(self, prices: dict):
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.current_price = prices[ticker]
    
    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value
    
    @property
    def return_pct(self) -> float:
        return ((self.total_value - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    
    def execute_buy(self, ticker: str, amount: float, price: float,
                    reasoning: str, timestamp: str, model_reasoning: str = ""):
        amount = min(amount, self.cash)
        if amount < 500:
            return None
        
        # Check position limit
        if ticker in self.positions:
            current_pct = self.positions[ticker].market_value / self.total_value
        else:
            current_pct = 0
        max_buy = (MAX_POSITION_PCT - current_pct) * self.total_value
        amount = min(amount, max_buy)
        if amount < 500:
            return None
        
        shares = amount / price
        
        if ticker in self.positions:
            pos = self.positions[ticker]
            total = pos.shares + shares
            pos.avg_cost = ((pos.shares * pos.avg_cost) + (shares * price)) / total
            pos.shares = total
            pos.current_price = price
        else:
            self.positions[ticker] = Position(ticker, shares, price, price)
        
        self.cash -= amount
        trade = Trade(timestamp, ticker, "BUY", round(shares, 4), price,
                     round(amount, 2), reasoning[:500], round(self.total_value, 2),
                     model_reasoning[:2000] if model_reasoning else "")
        self.trades.append(trade)
        return trade
    
    def execute_sell(self, ticker: str, amount: float, price: float,
                     reasoning: str, timestamp: str, model_reasoning: str = ""):
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        pos.current_price = price
        amount = min(amount, pos.market_value)
        if amount < 10:
            return None
        
        shares = min(amount / price, pos.shares)
        pos.shares -= shares
        self.cash += shares * price
        
        if pos.shares < 0.0001:
            del self.positions[ticker]
        
        trade = Trade(timestamp, ticker, "SELL", round(shares, 4), price,
                     round(shares * price, 2), reasoning[:500], round(self.total_value, 2),
                     model_reasoning[:2000] if model_reasoning else "")
        self.trades.append(trade)
        return trade
    
    def record_equity(self, timestamp: str):
        self.equity_curve.append({
            "timestamp": timestamp, "value": round(self.total_value, 2),
            "cash": round(self.cash, 2), "positions_value": round(self.positions_value, 2)
        })
    
    def to_dict(self) -> dict:
        return {
            "name": self.name, "cash": round(self.cash, 2),
            "total_value": round(self.total_value, 2),
            "return_pct": round(self.return_pct, 2),
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
            "equity_curve": self.equity_curve
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Portfolio':
        p = cls(name=data["name"], cash=data["cash"])
        for ticker, pos_data in data.get("positions", {}).items():
            p.positions[ticker] = Position(**pos_data)
        for t in data.get("trades", []):
            p.trades.append(Trade(**t))
        p.equity_curve = data.get("equity_curve", [])
        return p


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    def __init__(self):
        self.filepath = ARENA_OUTPUT_DIR / "checkpoint.json"
    
    def save(self, portfolios: dict, completed_dates: list, reasoning: dict):
        data = {
            "timestamp": datetime.now().isoformat(),
            "completed_count": len(completed_dates),
            "completed_dates": completed_dates,
            "portfolios": {n: p.to_dict() for n, p in portfolios.items()},
            "reasoning": reasoning
        }
        temp = self.filepath.with_suffix('.tmp')
        with open(temp, 'w') as f:
            json.dump(data, f)
        temp.replace(self.filepath)
    
    def load(self) -> Optional[dict]:
        if not self.filepath.exists():
            return None
        try:
            with open(self.filepath) as f:
                return json.load(f)
        except:
            return None
    
    def clear(self):
        if self.filepath.exists():
            self.filepath.unlink()
    
    def exists(self) -> bool:
        return self.filepath.exists()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_workflow_data(ticker: str, sample_dir: Path) -> Optional[dict]:
    """Load rich pipeline data for a ticker from workflow outputs"""
    try:
        # Check for portfolio subdirectory (some structures have this)
        portfolio_dir = sample_dir / "portfolio_100000"
        if not portfolio_dir.exists():
            portfolio_dir = sample_dir
        
        data = {"ticker": ticker, "data_type": "rich"}
        
        # Files to look for
        files_to_load = [
            ("discussion_points", "discussion_points.json"),
            ("market_context", "market_context.json"),
            ("bull_thesis", "bull_thesis.json"),
            ("bear_thesis", "bear_thesis.json"),
            ("research_synthesis", "research_synthesis.json"),
        ]
        
        for key, fname in files_to_load:
            # Try both locations
            for search_dir in [portfolio_dir, sample_dir]:
                fpath = search_dir / fname
                if fpath.exists():
                    with open(fpath, encoding='utf-8', errors='ignore') as f:
                        data[key] = json.load(f)
                    break
        
        # Need at least some data beyond ticker and data_type
        if len(data) > 2:
            return data
        
        # If no standard files found, try to load any JSON files
        json_files = list(sample_dir.glob("*.json")) + list(portfolio_dir.glob("*.json")) if portfolio_dir != sample_dir else list(sample_dir.glob("*.json"))
        for jf in json_files[:5]:  # Load up to 5 JSON files
            key = jf.stem  # filename without extension
            if key not in data:
                try:
                    with open(jf, encoding='utf-8', errors='ignore') as f:
                        data[key] = json.load(f)
                except:
                    pass
        
        return data if len(data) > 2 else None
        
    except Exception as e:
        print(f"      ⚠️ Error loading {ticker}: {e}")
        return None


def load_market_data(date: str) -> dict:
    date_dir = MARKET_DATA_DIR / date
    data = {}
    if date_dir.exists():
        for ticker in TICKERS:
            fpath = date_dir / f"{ticker}.json"
            if fpath.exists():
                with open(fpath, encoding='utf-8', errors='ignore') as f:
                    ticker_data = json.load(f)
                    ticker_data["data_type"] = "lite"
                    data[ticker] = ticker_data
    return data


def get_all_rounds() -> list:
    """Get all rounds from market data only"""
    rounds = []
    
    if MARKET_DATA_DIR.exists():
        market_dates = sorted([d.name for d in MARKET_DATA_DIR.iterdir() if d.is_dir()])
        for date in market_dates:
            rounds.append({"date": date, "phase": 1, "data_type": "lite"})
    
    return rounds

# =============================================================================
# PROMPT BUILDING
# =============================================================================

def format_portfolio(portfolio: Portfolio) -> str:
    lines = [f"Cash: ${portfolio.cash:,.2f}",
             f"Total Value: ${portfolio.total_value:,.2f}",
             f"Return: {portfolio.return_pct:+.2f}%", ""]
    
    if portfolio.positions:
        lines.append("Positions:")
        for ticker, pos in portfolio.positions.items():
            pct = pos.market_value / portfolio.total_value * 100 if portfolio.total_value else 0
            lines.append(f"  {ticker}: {pos.shares:.2f} @ ${pos.avg_cost:.2f} | "
                        f"Now: ${pos.current_price:.2f} | P&L: {pos.unrealized_pnl_pct:+.1f}% | {pct:.1f}%")
    else:
        lines.append("Positions: None (100% cash)")
    return "\n".join(lines)


def format_market_data(market_data: dict, data_type: str) -> str:
    lines = []
    for ticker, data in market_data.items():
        if data_type == "rich" and "discussion_points" in data:
            dp = data.get("discussion_points", {})
            summary = dp.get("summary", {})
            recs = summary.get("recommendations", {})
            lines.append(f"\n### {ticker}")
            lines.append(f"Signals: Tech={recs.get('technical','N/A')}, "
                        f"Fund={recs.get('fundamental','N/A')}, "
                        f"News={recs.get('news','N/A')}, Macro={recs.get('macro','N/A')}")
            lines.append(f"Sentiment: {summary.get('net_sentiment', 'N/A')}")
            if "llm_synthesis" in dp:
                lines.append(f"Analysis: {dp['llm_synthesis'][:300]}...")
            mc = data.get("market_context", {})
            if "price_data" in mc:
                pd = mc["price_data"]
                lines.append(f"Price: ${pd.get('close',0):.2f} | Change: {pd.get('daily_return',0)*100:+.2f}%")
        else:
            price = data.get("price", {})
            tech = data.get("technicals", {})
            macro = data.get("macro", {})
            lines.append(f"\n### {ticker}")
            lines.append(f"Price: ${price.get('close',0):.2f} | "
                        f"1D: {price.get('change_1d_pct',0):+.2f}% | "
                        f"7D: {price.get('change_7d_pct',0):+.2f}%")
            lines.append(f"RSI: {tech.get('rsi_14','N/A')} | "
                        f"vs SMA20: {tech.get('price_vs_sma20_pct','N/A')}%")
            if macro:
                lines.append(f"Regime: {macro.get('regime','N/A')} | VIX: {macro.get('vix',{}).get('level','N/A')}")
    return "\n".join(lines)


def build_prompt(portfolio: Portfolio, market_data: dict, date: str, data_type: str) -> str:
    return f"""You are an autonomous portfolio manager in a trading competition.

DATE: {date}
DATA TYPE: {"Rich multi-agent analysis" if data_type == "rich" else "Standard market data"}

YOUR PORTFOLIO:
{format_portfolio(portfolio)}

TICKERS: {', '.join(TICKERS)}

MARKET DATA:
{format_market_data(market_data, data_type)}

RULES: Max 40% per stock. Min trade $500.

RESPOND WITH:
<reasoning>
[Your analysis: sentiment, opportunities, risks, rationale]
</reasoning>

<json>
{{
  "trades": [
    {{"ticker": "NVDA", "action": "BUY", "amount_usd": 5000, "reasoning": "Strong momentum"}},
    {{"ticker": "AAPL", "action": "SELL", "amount_usd": 3000, "reasoning": "Taking profits"}}
  ],
  "market_outlook": "Brief outlook"
}}
</json>

If holding: {{"trades": [], "market_outlook": "Holding because..."}}
"""


# =============================================================================
# DECISION PARSING
# =============================================================================

def get_decision(model_name: str, config: dict, portfolio: Portfolio,
                 market_data: dict, date: str, data_type: str) -> tuple:
    """Get trading decision from LLM. Returns (trades, reasoning)"""
    
    # Update portfolio prices
    prices = {}
    for ticker, data in market_data.items():
        if "price" in data:
            prices[ticker] = data["price"].get("close", 0)
        elif "market_context" in data:
            prices[ticker] = data["market_context"].get("price_data", {}).get("close", 0)
    portfolio.update_prices(prices)
    
    # Call LLM
    prompt = build_prompt(portfolio, market_data, date, data_type)
    response = call_llm(config["provider"], config["model"], prompt)
    
    if not response:
        return [], ""
    
    # Parse reasoning
    reasoning = ""
    match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        print(f"      💭 {reasoning[:100].replace(chr(10), ' ')}...")
    
    # Parse trades
    trades = []
    json_match = re.search(r'<json>(.*?)</json>', response, re.DOTALL)
    if json_match:
        try:
            decision = json.loads(json_match.group(1))
            trades = decision.get("trades", [])
            outlook = decision.get("market_outlook", "")
            if outlook:
                print(f"      📊 {outlook[:80]}...")
        except json.JSONDecodeError:
            pass
    else:
        # Fallback
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                trades = json.loads(match.group()).get("trades", [])
        except:
            pass
    
    return trades, reasoning


# =============================================================================
# ARENA
# =============================================================================

class TradingArena:
    def __init__(self):
        self.portfolios = {name: Portfolio(name=name) for name in ARENA_MODELS}
        self.completed_dates = []
        self.reasoning = {}
        self.checkpoint = CheckpointManager()
        self.start_time = None
    
    def restore_from_checkpoint(self) -> bool:
        """Restore state from checkpoint. Returns True if restored."""
        data = self.checkpoint.load()
        if not data:
            return False
        
        self.completed_dates = data["completed_dates"]
        self.reasoning = data.get("reasoning", {})
        
        for name, pdata in data["portfolios"].items():
            if name in self.portfolios:
                self.portfolios[name] = Portfolio.from_dict(pdata)
        
        print(f"✅ Restored from checkpoint: {len(self.completed_dates)} rounds completed")
        return True
    
    def run_round(self, round_info: dict, market_data: dict) -> bool:
        """
        Run one round of trading. 
        Returns True if successful, False if a model failed (round incomplete).
        
        IMPORTANT: If any model fails, we stop the round and DON'T save it.
        This ensures all models participate in every round.
        """
        date = round_info["date"]
        data_type = round_info["data_type"]
        
        # Get prices
        prices = {}
        for ticker, data in market_data.items():
            if "price" in data:
                prices[ticker] = data["price"].get("close", 0)
            elif "market_context" in data:
                prices[ticker] = data["market_context"].get("price_data", {}).get("close", 0)
        
        # Temporary storage for this round (only commit if ALL models succeed)
        round_results = {}
        round_reasoning = {}
        
        for idx, (model_name, config) in enumerate(ARENA_MODELS.items()):
            print(f"\n  🤖 {model_name}:")
            portfolio = self.portfolios[model_name]
            portfolio.update_prices(prices)
            
            try:
                trades, reasoning = get_decision(model_name, config, portfolio,
                                                market_data, date, data_type)
            except ModelFailedError as e:
                print(f"\n  ❌ ROUND FAILED: {e.provider} is unavailable")
                print(f"     Error: {e.message}")
                print(f"\n  ⏸️  Stopping here. No trades from this round will be saved.")
                print(f"     Run with --resume after the API recovers.\n")
                return False  # Round failed - don't save anything
            
            round_reasoning[model_name] = reasoning
            
            # Execute trades (store results temporarily)
            executed_trades = []
            for t in trades:
                ticker = t.get("ticker")
                action = t.get("action", "").upper()
                # Handle string amounts from LLM responses
                amount = t.get("amount_usd", 0)
                try:
                    amount = float(amount) if amount else 0
                except (ValueError, TypeError):
                    amount = 0
                reason = t.get("reasoning", "")
                price = prices.get(ticker, 0)
                if not ticker or not price or amount <= 0:
                    continue
                
                if action == "BUY":
                    result = portfolio.execute_buy(ticker, amount, price, reason, date, reasoning)
                elif action == "SELL":
                    result = portfolio.execute_sell(ticker, amount, price, reason, date, reasoning)
                else:
                    result = None
                
                if result:
                    print(f"      ✓ {action} {ticker}: ${amount:,.0f} @ ${price:.2f}")
                    executed_trades.append(result)
            
            if not executed_trades:
                print(f"      - No trades (holding)")
            
            portfolio.record_equity(date)
            print(f"      💰 ${portfolio.total_value:,.2f} ({portfolio.return_pct:+.2f}%)")
            
            round_results[model_name] = {
                "trades": executed_trades,
                "portfolio_value": portfolio.total_value
            }
            
            # Delay between models
            if idx < len(ARENA_MODELS) - 1:
                time.sleep(TIMING["inter_model_delay"])
        
        # All models succeeded! Save the round
        self.reasoning[date] = round_reasoning
        self.completed_dates.append(date)
        return True
    
    def print_progress(self, current: int, total: int):
        """Print progress bar and ETA"""
        pct = current / total * 100
        elapsed = time.time() - self.start_time
        
        if current > 0:
            per_round = elapsed / current
            remaining = (total - current) * per_round
            eta = timedelta(seconds=int(remaining))
        else:
            eta = "calculating..."
        
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\n📊 Progress: [{bar}] {current}/{total} ({pct:.1f}%) | ETA: {eta}")
    
    def print_standings(self):
        """Print current standings"""
        sorted_p = sorted(self.portfolios.items(), key=lambda x: x[1].total_value, reverse=True)
        print("\n🏆 Current Standings:")
        for rank, (name, p) in enumerate(sorted_p, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"   {medal} {rank}. {name:20} ${p.total_value:>10,.2f} ({p.return_pct:+.2f}%)")
    
    def run(self, max_rounds: Optional[int] = None, resume: bool = False, wait_on_failure: bool = False):
        """
        Run the arena with fail-safe stopping.
        
        Args:
            max_rounds: Limit number of rounds
            resume: Resume from checkpoint
            wait_on_failure: If True, wait and retry on failure instead of stopping
        """
        # Get all rounds
        all_rounds = get_all_rounds()
        
        if max_rounds:
            all_rounds = all_rounds[:max_rounds]
        
        total_rounds = len(all_rounds)
        
        # Resume from checkpoint?
        if resume and self.checkpoint.exists():
            self.restore_from_checkpoint()
            # Filter out completed rounds
            all_rounds = [r for r in all_rounds if r["date"] not in self.completed_dates]
            print(f"📌 Resuming: {len(all_rounds)} rounds remaining")
        
        if not all_rounds:
            print("✅ All rounds already completed!")
            self.print_standings()
            return
        
        # Header
        print(f"\n{'#'*60}")
        print(f"🏟️  LLM TRADING ARENA")
        print(f"{'#'*60}")
        print(f"Total Rounds: {total_rounds} | Remaining: {len(all_rounds)}")
        print(f"Models: {', '.join(ARENA_MODELS.keys())}")
        print(f"Starting Capital: ${STARTING_CAPITAL:,}")
        
        if wait_on_failure:
            print(f"\n🔄 WAIT MODE: Will wait {TIMING['failure_cooldown']}s and retry on failures")
        else:
            print(f"\n⚠️  FAIL-SAFE MODE: If any model fails, arena stops cleanly.")
            print(f"   Use --resume to continue, or --wait to auto-retry.")
        
        # Estimate time
        secs_per_round = (len(ARENA_MODELS) * TIMING["inter_model_delay"]) + TIMING["inter_round_delay"]
        est_time = timedelta(seconds=len(all_rounds) * secs_per_round)
        print(f"\nEstimated Time: {est_time}")
        print(f"{'#'*60}")
        
        self.start_time = time.time()
        stopped_early = False
        
        i = 0
        while i < len(all_rounds):
            round_info = all_rounds[i]
            date = round_info["date"]
            
            # Progress
            completed = len(self.completed_dates)
            self.print_progress(completed, total_rounds)
            
            print(f"\n{'='*60}")
            print(f"ROUND {completed + 1}/{total_rounds}: {date} (Phase {round_info['phase']} - {round_info['data_type']})")
            print(f"{'='*60}")
            
            market_data = load_market_data(date)

            
            # Run round - returns False if a model failed
            success = self.run_round(round_info, market_data)
            
            if not success:
                if wait_on_failure:
                    # Wait and retry
                    for retry in range(TIMING["max_failure_retries"]):
                        cooldown = TIMING["failure_cooldown"]
                        print(f"\n⏳ Waiting {cooldown}s before retry ({retry + 1}/{TIMING['max_failure_retries']})...")
                        
                        # Countdown
                        for remaining in range(cooldown, 0, -30):
                            print(f"   {remaining}s remaining...")
                            time.sleep(min(30, remaining))
                        
                        print(f"\n🔄 Retrying round {date}...")
                        
                        # Reset portfolios to state before this round
                        # (they may have partial trades - reload from checkpoint)
                        if self.checkpoint.exists():
                            self.restore_from_checkpoint()
                        
                        success = self.run_round(round_info, market_data)
                        if success:
                            break
                    
                    if not success:
                        print(f"\n❌ Round failed after {TIMING['max_failure_retries']} retries")
                        stopped_early = True
                        break
                else:
                    # Stop immediately
                    stopped_early = True
                    print(f"\n{'!'*60}")
                    print(f"⏸️  ARENA PAUSED - API failure detected")
                    print(f"{'!'*60}")
                    print(f"\nCompleted rounds: {len(self.completed_dates)}")
                    print(f"Remaining rounds: {total_rounds - len(self.completed_dates)}")
                    print(f"\nTo continue later:")
                    print(f"   python llm_arena_production.py --resume")
                    print(f"\nOr to auto-wait on failures:")
                    print(f"   python llm_arena_production.py --resume --wait")
                    break
            
            # Checkpoint periodically
            if len(self.completed_dates) % TIMING["checkpoint_every"] == 0:
                self.checkpoint.save(self.portfolios, self.completed_dates, self.reasoning)
                print(f"  💾 Checkpoint saved ({len(self.completed_dates)} rounds)")
            
            # Inter-round delay
            time.sleep(TIMING["inter_round_delay"])
            i += 1
        
        # Final save
        self.checkpoint.save(self.portfolios, self.completed_dates, self.reasoning)
        self.save_results()
        self.print_standings()
        
        elapsed = timedelta(seconds=int(time.time() - self.start_time))
        
        if stopped_early:
            print(f"\n⏸️  Paused after {elapsed} ({len(self.completed_dates)} rounds completed)")
        else:
            print(f"\n✅ Completed all {len(self.completed_dates)} rounds in {elapsed}")
    
    def save_results(self):
        """Save final results"""
        sorted_p = sorted(self.portfolios.items(), key=lambda x: x[1].total_value, reverse=True)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "rounds": len(self.completed_dates),
            "leaderboard": [
                {"rank": i+1, "model": n, "value": round(p.total_value, 2),
                 "return_pct": round(p.return_pct, 2), "trades": len(p.trades)}
                for i, (n, p) in enumerate(sorted_p)
            ],
            "portfolios": {n: p.to_dict() for n, p in self.portfolios.items()}
        }
        
        with open(ARENA_OUTPUT_DIR / "arena_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Trade history
        all_trades = []
        for name, p in self.portfolios.items():
            for t in p.trades:
                all_trades.append({"model": name, **asdict(t)})
        all_trades.sort(key=lambda x: x["timestamp"])
        
        with open(ARENA_OUTPUT_DIR / "trade_history.json", "w") as f:
            json.dump(all_trades, f, indent=2)
        
        print(f"\n💾 Results saved to {ARENA_OUTPUT_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def show_status():
    """Show arena status with detailed data diagnostics"""
    print("🏟️  LLM Trading Arena Status")
    print("=" * 60)
    
    # Check workflow data
    print(f"\n📁 Workflow Data: {WORKFLOW_DATA_DIR}")
    print(f"   Exists: {WORKFLOW_DATA_DIR.exists()}")
    
    workflow_dates = set()
    if WORKFLOW_DATA_DIR.exists():
        for ticker in TICKERS:
            ticker_dir = WORKFLOW_DATA_DIR / ticker
            if ticker_dir.exists():
                for d in ticker_dir.iterdir():
                    if d.is_dir() and "sample" in d.name.lower():
                        date_str = d.name.split("_sample")[0]
                        workflow_dates.add(date_str)
        
        # Show sample folder structure
        sample_ticker = WORKFLOW_DATA_DIR / "AAPL"
        if sample_ticker.exists():
            samples = [d.name for d in sample_ticker.iterdir() if d.is_dir()][:3]
            print(f"   Sample folders: {samples}")
    
    print(f"   Unique dates found: {len(workflow_dates)}")
    if workflow_dates:
        sorted_wd = sorted(workflow_dates)
        print(f"   Date range: {sorted_wd[0]} to {sorted_wd[-1]}")
    
    # Check market data
    print(f"\n📁 Market Data: {MARKET_DATA_DIR}")
    print(f"   Exists: {MARKET_DATA_DIR.exists()}")
    
    market_dates = set()
    if MARKET_DATA_DIR.exists():
        market_dates = {d.name for d in MARKET_DATA_DIR.iterdir() if d.is_dir()}
    
    print(f"   Unique dates found: {len(market_dates)}")
    if market_dates:
        sorted_md = sorted(market_dates)
        print(f"   Date range: {sorted_md[0]} to {sorted_md[-1]}")
    
    # Calculate rounds
    only_market = market_dates - workflow_dates
    overlap = market_dates & workflow_dates
    
    print(f"\n📊 Round Calculation:")
    print(f"   Workflow dates (Phase 1 - rich): {len(workflow_dates)}")
    print(f"   Market-only dates (Phase 2 - lite): {len(only_market)}")
    print(f"   Overlap (using workflow): {len(overlap)}")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL ROUNDS: {len(workflow_dates) + len(only_market)}")
    
    rounds = get_all_rounds()
    rich = sum(1 for r in rounds if r["data_type"] == "rich")
    lite = len(rounds) - rich
    
    print(f"\n   Verification from get_all_rounds():")
    print(f"   Rich: {rich} | Lite: {lite} | Total: {len(rounds)}")
    
    print(f"\n🤖 Models ({len(ARENA_MODELS)}):")
    for name, cfg in ARENA_MODELS.items():
        print(f"   {name}: {cfg['provider']}/{cfg['model']}")
    
    print(f"\n🔑 API Keys:")
    print(f"   OpenAI:  {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"   Google:  {'✅' if GOOGLE_API_KEY else '❌'}")
    print(f"   Groq:    {'✅' if GROQ_API_KEY else '❌'}")
    print(f"   Mistral: {'✅' if MISTRAL_API_KEY else '❌'}")
    
    # Checkpoint
    cp = CheckpointManager()
    if cp.exists():
        data = cp.load()
        if data:
            print(f"\n💾 Checkpoint Found:")
            print(f"   Completed: {data['completed_count']} rounds")
            print(f"   Saved: {data['timestamp']}")
    
    # Time estimate
    secs = (len(ARENA_MODELS) * TIMING["inter_model_delay"]) + TIMING["inter_round_delay"]
    total_time = timedelta(seconds=len(rounds) * secs)
    print(f"\n⏱️  Estimated Runtime: {total_time}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Trading Arena - Production Version")
    parser.add_argument("--run", action="store_true", help="Run arena")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--wait", action="store_true", help="Wait and retry on failures (instead of stopping)")
    parser.add_argument("--rounds", type=int, help="Max rounds to run")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.reset:
        CheckpointManager().clear()
        print("✅ Checkpoint cleared - next run will start fresh")
    elif args.run or args.resume:
        arena = TradingArena()
        arena.run(max_rounds=args.rounds, resume=args.resume, wait_on_failure=args.wait)
    else:
        print("🏟️  LLM Trading Arena - Production Version")
        print("=" * 50)
        print("\nUsage:")
        print("  python llm_arena_production.py --status              # Show data & API status")
        print("  python llm_arena_production.py --run                 # Run all rounds")
        print("  python llm_arena_production.py --run --rounds 50     # Run 50 rounds only")
        print("  python llm_arena_production.py --run --wait          # Auto-retry on failures")
        print("  python llm_arena_production.py --resume              # Resume from checkpoint")
        print("  python llm_arena_production.py --resume --wait       # Resume with auto-retry")
        print("  python llm_arena_production.py --reset               # Clear checkpoint")
        print("\nFeatures:")
        print("  ✅ Checkpoint/Resume - Never lose progress")
        print("  ✅ Fail-safe - Stops cleanly if ANY model fails (no skipped results)")
        print("  ✅ Wait mode - Auto-retry after 5 min cooldown")
        print("  ✅ Progress bar with ETA")