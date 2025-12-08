"""
LLM Trading Arena
Multiple LLMs compete as autonomous portfolio managers
Each sees the same market data, makes independent decisions

Enhanced with Chain-of-Thought reasoning for better decision quality
"""

import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import requests
from dotenv import load_dotenv

load_dotenv()

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

# LLM Models configuration - FIXED model names and IDs
ARENA_MODELS = {
    # OpenAI
    "GPT-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    
    # Google Gemini - FREE tier: 15 req/min, 1500 req/day
    "Gemini-Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    
    # Groq - FREE tier: 30 req/min
    "Llama-3.3-70B": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Llama-3.1-8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
   
    # Mistral - FREE tier available
    "Mistral-Small": {"provider": "mistral", "model": "mistral-small-latest"}    
}

# Data paths
WORKFLOW_DATA_DIR = ROOT_DIR / "outputs" / "workflows"
MARKET_DATA_DIR = ROOT_DIR / "outputs" / "market_data"
ARENA_OUTPUT_DIR = ROOT_DIR / "outputs" / "llm_arena"

# Trading configuration
TICKERS = [
    "AAPL", "AMZN", "CVX", "GOOGL", "GS", "JNJ", "JPM", "KO", "LLY", "META",
    "MSFT", "NVDA", "PG", "QQQ", "SPY", "TSLA", "UNH", "V", "WMT", "XOM"
]
STARTING_CAPITAL = 250_000
MAX_POSITION_PCT = 0.25


# =============================================================================
# PORTFOLIO MANAGEMENT
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
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100


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
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
    
    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value
    
    @property
    def return_pct(self) -> float:
        return ((self.total_value - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    
    def get_position_pct(self, ticker: str, current_price: float) -> float:
        if ticker not in self.positions:
            return 0
        position_value = self.positions[ticker].shares * current_price
        return position_value / self.total_value if self.total_value > 0 else 0
    
    def execute_buy(self, ticker: str, amount_usd: float, price: float, 
                    reasoning: str, timestamp: str, model_reasoning: str = ""):
        amount_usd = min(amount_usd, self.cash)
        if amount_usd < 10:
            return None
        
        current_pct = self.get_position_pct(ticker, price)
        max_buy = (MAX_POSITION_PCT - current_pct) * self.total_value
        amount_usd = min(amount_usd, max_buy)
        
        if amount_usd < 10:
            return None
        
        shares = amount_usd / price
        
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            pos.avg_cost = ((pos.shares * pos.avg_cost) + (shares * price)) / total_shares
            pos.shares = total_shares
            pos.current_price = price
        else:
            self.positions[ticker] = Position(ticker=ticker, shares=shares, avg_cost=price, current_price=price)
        
        self.cash -= amount_usd
        
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            action="BUY",
            shares=round(shares, 4),
            price=price,
            amount_usd=round(amount_usd, 2),
            reasoning=reasoning[:500],
            portfolio_value_after=round(self.total_value, 2),
            model_reasoning=model_reasoning[:2000] if model_reasoning else ""
        )
        self.trades.append(trade)
        return trade
    
    def execute_sell(self, ticker: str, amount_usd: float, price: float, 
                     reasoning: str, timestamp: str, model_reasoning: str = ""):
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        pos.current_price = price
        
        max_sell_value = pos.market_value
        amount_usd = min(amount_usd, max_sell_value)
        
        if amount_usd < 10:
            return None
        
        shares = amount_usd / price
        shares = min(shares, pos.shares)
        
        pos.shares -= shares
        self.cash += shares * price
        
        if pos.shares < 0.0001:
            del self.positions[ticker]
        
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            action="SELL",
            shares=round(shares, 4),
            price=price,
            amount_usd=round(shares * price, 2),
            reasoning=reasoning[:500],
            portfolio_value_after=round(self.total_value, 2),
            model_reasoning=model_reasoning[:2000] if model_reasoning else ""
        )
        self.trades.append(trade)
        return trade
    
    def record_equity(self, timestamp: str):
        self.equity_curve.append({
            "timestamp": timestamp,
            "value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(self.positions_value, 2)
        })
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cash": round(self.cash, 2),
            "positions_value": round(self.positions_value, 2),
            "total_value": round(self.total_value, 2),
            "return_pct": round(self.return_pct, 2),
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "trade_count": len(self.trades),
            "equity_curve": self.equity_curve,
            "trades": [asdict(t) for t in self.trades]
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_workflow_data(ticker: str, sample_dir: Path) -> Optional[dict]:
    """Load rich pipeline data for a ticker from workflow outputs"""
    try:
        portfolio_dir = sample_dir / "portfolio_100000"
        if not portfolio_dir.exists():
            portfolio_dir = sample_dir
        
        data = {"ticker": ticker, "data_type": "rich"}
        
        files_to_load = [
            ("discussion_points", "discussion_points.json"),
            ("market_context", "market_context.json"),
            ("bull_thesis", "bull_thesis.json"),
            ("bear_thesis", "bear_thesis.json"),
            ("research_synthesis", "research_synthesis.json"),
        ]
        
        for key, filename in files_to_load:
            filepath = portfolio_dir / filename
            if filepath.exists():
                with open(filepath, encoding='utf-8', errors='ignore') as f:
                    data[key] = json.load(f)
        
        return data if len(data) > 2 else None
        
    except Exception as e:
        print(f"    ⚠ Error loading workflow data for {ticker}: {e}")
        return None


def load_market_data(date: str) -> dict:
    """Load lite API data for all tickers for a specific date"""
    date_dir = MARKET_DATA_DIR / date
    data = {}
    
    if not date_dir.exists():
        return data
    
    for ticker in TICKERS:
        filepath = date_dir / f"{ticker}.json"
        if filepath.exists():
            with open(filepath, encoding='utf-8', errors='ignore') as f:
                ticker_data = json.load(f)
                ticker_data["data_type"] = "lite"
                data[ticker] = ticker_data
    
    return data


def get_available_workflow_dates() -> list:
    """Get list of dates available in workflow data"""
    dates = []
    
    for ticker in TICKERS:
        ticker_dir = WORKFLOW_DATA_DIR / ticker
        if not ticker_dir.exists():
            continue
        
        for sample_dir in ticker_dir.iterdir():
            if sample_dir.is_dir() and "_sample" in sample_dir.name:
                date_str = sample_dir.name.split("_")[0]
                if date_str not in dates:
                    dates.append(date_str)
    
    return sorted(dates)


def get_available_market_dates() -> list:
    """Get list of dates available in market data"""
    if not MARKET_DATA_DIR.exists():
        return []
    return sorted([d.name for d in MARKET_DATA_DIR.iterdir() if d.is_dir()])


def get_all_available_rounds() -> list:
    """Get all available rounds combining workflow and market data"""
    workflow_dates = set(get_available_workflow_dates())
    market_dates = set(get_available_market_dates())
    
    rounds = []
    
    for date in sorted(workflow_dates):
        rounds.append({"date": date, "phase": 1, "data_type": "rich"})
    
    for date in sorted(market_dates - workflow_dates):
        rounds.append({"date": date, "phase": 2, "data_type": "lite"})
    
    rounds.sort(key=lambda x: x["date"])
    return rounds


# =============================================================================
# LLM API CALLS (All providers with correct endpoints)
# =============================================================================

def call_google(prompt: str, model: str = "gemini-2.5-flash") -> Optional[str]:
    """Call Google Gemini API"""
    if not GOOGLE_API_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GOOGLE_API_KEY}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2500}
        }
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"      ⚠ Google API error: {e}")
        return None


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> Optional[str]:
    """Call Groq API"""
    if not GROQ_API_KEY:
        return None
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2500
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"      ⚠ Groq API error: {e}")
        return None


def call_mistral(prompt: str, model: str = "mistral-small-latest") -> Optional[str]:
    """Call Mistral API"""
    if not MISTRAL_API_KEY:
        return None
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2500
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"      ⚠ Mistral API error: {e}")
        return None

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Call OpenAI API"""
    if not OPENAI_API_KEY:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2500
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"      ⚠ OpenAI API error: {e}")
        return None


def call_llm(provider: str, model: str, prompt: str) -> Optional[str]:
    """Route to the correct provider"""
    if provider == "google":
        return call_google(prompt, model)
    elif provider == "groq":
        return call_groq(prompt, model)
    elif provider == "mistral":
        return call_mistral(prompt, model)
    elif provider == "openai":
        return call_openai(prompt, model)
    else:
        print(f"      ⚠ Unknown provider: {provider}")
        return None


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def format_portfolio_for_prompt(portfolio: Portfolio) -> str:
    """Format portfolio state for LLM prompt"""
    lines = [
        f"Cash Available: ${portfolio.cash:,.2f}",
        f"Total Portfolio Value: ${portfolio.total_value:,.2f}",
        f"Overall Return: {portfolio.return_pct:+.2f}%",
        ""
    ]
    
    if portfolio.positions:
        lines.append("Current Positions:")
        for ticker, pos in portfolio.positions.items():
            pct_of_portfolio = (pos.market_value / portfolio.total_value * 100) if portfolio.total_value > 0 else 0
            lines.append(f"  {ticker}: {pos.shares:.2f} shares @ ${pos.avg_cost:.2f} avg | "
                        f"Current: ${pos.current_price:.2f} | P&L: {pos.unrealized_pnl_pct:+.1f}% | "
                        f"Weight: {pct_of_portfolio:.1f}%")
    else:
        lines.append("Current Positions: None (100% cash)")
    
    return "\n".join(lines)


def format_market_data_for_prompt(market_data: dict, data_type: str) -> str:
    """Format market data for LLM prompt"""
    lines = []
    
    for ticker, data in market_data.items():
        if data_type == "rich" and "discussion_points" in data:
            dp = data.get("discussion_points", {})
            summary = dp.get("summary", {})
            recs = summary.get("recommendations", {})
            
            lines.append(f"\n### {ticker}")
            lines.append(f"Analyst Recommendations: Technical={recs.get('technical', 'N/A')}, "
                        f"Fundamental={recs.get('fundamental', 'N/A')}, "
                        f"News={recs.get('news', 'N/A')}, Macro={recs.get('macro', 'N/A')}")
            lines.append(f"Net Sentiment: {summary.get('net_sentiment', 'N/A')}")
            
            if "llm_synthesis" in dp:
                lines.append(f"Analysis Summary: {dp['llm_synthesis'][:400]}...")
            
            mc = data.get("market_context", {})
            if "price_data" in mc:
                pd = mc["price_data"]
                lines.append(f"Price: ${pd.get('close', 0):.2f} | Daily Change: {pd.get('daily_return', 0)*100:+.2f}%")
        
        else:
            price = data.get("price", {})
            tech = data.get("technicals", {})
            fund = data.get("fundamentals", {})
            macro = data.get("macro", {})
            news = data.get("news", [])
            
            lines.append(f"\n### {ticker}")
            lines.append(f"Price: ${price.get('close', 0):.2f} | "
                        f"1D: {price.get('change_1d_pct', 0):+.2f}% | "
                        f"7D: {price.get('change_7d_pct', 0):+.2f}% | "
                        f"30D: {price.get('change_30d_pct', 0):+.2f}%")
            lines.append(f"RSI(14): {tech.get('rsi_14', 'N/A')} | "
                        f"MACD: {tech.get('macd', 'N/A')} | "
                        f"vs SMA20: {tech.get('price_vs_sma20_pct', 'N/A')}% | "
                        f"vs SMA50: {tech.get('price_vs_sma50_pct', 'N/A')}%")
            
            if fund:
                lines.append(f"P/E: {fund.get('pe_ratio', 'N/A')} | "
                            f"Fwd P/E: {fund.get('forward_pe', 'N/A')} | "
                            f"Analyst: {fund.get('analyst_recommendation', 'N/A')}")
            
            if macro:
                regime = macro.get('regime', 'N/A')
                vix = macro.get('vix', {}).get('level', 'N/A')
                lines.append(f"Market Regime: {regime} | VIX: {vix}")
            
            if news:
                lines.append(f"Latest News: {news[0].get('headline', 'N/A')[:80]}...")
    
    return "\n".join(lines)


# =============================================================================
# LLM DECISION ENGINE (Chain-of-Thought)
# =============================================================================

def get_llm_trading_decision(model_name: str, model_config: dict, portfolio: Portfolio, 
                             market_data: dict, current_date: str, data_type: str) -> tuple:
    """
    Ask LLM for trading decisions using Chain-of-Thought prompting.
    Returns: (list of trades, full reasoning string)
    """
    provider = model_config["provider"]
    model = model_config["model"]
    
    # Get current prices
    prices = {}
    for ticker, data in market_data.items():
        if "price" in data:
            prices[ticker] = data["price"].get("close", 0)
        elif "market_context" in data and "price_data" in data["market_context"]:
            prices[ticker] = data["market_context"]["price_data"].get("close", 0)
    
    portfolio.update_prices(prices)
    
    # Chain-of-Thought prompt
    prompt = f"""You are an autonomous portfolio manager competing in a trading arena.
Your goal is to maximize returns while managing risk intelligently.

═══════════════════════════════════════════════════════════════
TODAY'S DATE: {current_date}
DATA QUALITY: {"Full analyst reports with technical, fundamental, news, and macro analysis" if data_type == "rich" else "Market data with technicals, fundamentals, and macro context"}
═══════════════════════════════════════════════════════════════

YOUR PORTFOLIO:
{format_portfolio_for_prompt(portfolio)}

TRADEABLE UNIVERSE: {', '.join(TICKERS)}

TODAY'S MARKET DATA:
{format_market_data_for_prompt(market_data, data_type)}

═══════════════════════════════════════════════════════════════
TRADING RULES:
- Maximum 25% of portfolio in any single stock
- All trades execute at today's closing price
- Minimum trade size: $10
═══════════════════════════════════════════════════════════════

INSTRUCTIONS:
1. First, ANALYZE the market data inside <reasoning> tags:
   - What is the overall market sentiment today?
   - Which stocks show the strongest/weakest signals?
   - What are the key risks to consider?
   - How does your current portfolio positioning align with opportunities?

2. Then, DECIDE on your trades and output them inside <json> tags.

RESPONSE FORMAT:
<reasoning>
[Your detailed market analysis and trade rationale here. Be specific about WHY you're making each decision.]
</reasoning>

<json>
{{
  "trades": [
    {{"ticker": "NVDA", "action": "BUY", "amount_usd": 5000, "reasoning": "Strong momentum with positive analyst sentiment"}},
    {{"ticker": "AAPL", "action": "SELL", "amount_usd": 3000, "reasoning": "Taking profits after 15% gain"}}
  ],
  "market_outlook": "Brief 1-2 sentence market outlook",
  "strategy_note": "Brief note on your current strategy"
}}
</json>

If holding all positions with no trades:
<reasoning>
[Explain why you're choosing to hold]
</reasoning>

<json>
{{
  "trades": [],
  "market_outlook": "...",
  "strategy_note": "Holding because..."
}}
</json>
"""
    
    response = call_llm(provider, model, prompt)
    
    if not response:
        return [], ""
    
    # Parse the reasoning
    agent_thoughts = ""
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if reasoning_match:
        agent_thoughts = reasoning_match.group(1).strip()
        preview = agent_thoughts[:100].replace('\n', ' ')
        print(f"      💭 Thinking: {preview}...")
    
    # Parse the JSON trades
    trades = []
    json_match = re.search(r'<json>(.*?)</json>', response, re.DOTALL)
    
    if json_match:
        try:
            decision = json.loads(json_match.group(1))
            trades = decision.get("trades", [])
            
            outlook = decision.get("market_outlook", "")
            if outlook:
                print(f"      📊 Outlook: {outlook[:80]}...")
                
        except json.JSONDecodeError as e:
            print(f"      ⚠ Failed to parse JSON: {e}")
    else:
        # Fallback: try to find raw JSON
        try:
            json_fallback = re.search(r'\{[\s\S]*\}', response)
            if json_fallback:
                decision = json.loads(json_fallback.group())
                trades = decision.get("trades", [])
        except json.JSONDecodeError:
            pass
    
    return trades, agent_thoughts


# =============================================================================
# ARENA RUNNER
# =============================================================================

class TradingArena:
    def __init__(self):
        self.portfolios = {name: Portfolio(name=name) for name in ARENA_MODELS.keys()}
        self.rounds_completed = 0
        self.round_reasoning = {}
        
        ARENA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def run_round(self, round_info: dict, market_data: dict):
        """Run a single round of trading for all LLMs"""
        date = round_info["date"]
        data_type = round_info["data_type"]
        
        print(f"\n{'='*60}")
        print(f"ROUND {self.rounds_completed + 1}: {date} (Phase {round_info['phase']} - {data_type} data)")
        print(f"{'='*60}")
        
        # Get prices
        prices = {}
        for ticker, data in market_data.items():
            if "price" in data:
                prices[ticker] = data["price"].get("close", 0)
            elif "market_context" in data and "price_data" in data["market_context"]:
                prices[ticker] = data["market_context"]["price_data"].get("close", 0)
        
        self.round_reasoning[date] = {}
        
        # Each LLM makes decisions
        for model_name, model_config in ARENA_MODELS.items():
            print(f"\n  🤖 {model_name}:")
            portfolio = self.portfolios[model_name]
            portfolio.update_prices(prices)
            
            trades, reasoning = get_llm_trading_decision(
                model_name, model_config, portfolio, market_data, date, data_type
            )
            
            self.round_reasoning[date][model_name] = reasoning
            
            # Execute trades
            trades_executed = 0
            for trade in trades:
                ticker = trade.get("ticker")
                action = trade.get("action", "").upper()
                amount = trade.get("amount_usd", 0)
                trade_reasoning = trade.get("reasoning", "")
                price = prices.get(ticker, 0)
                
                if not ticker or not price or amount <= 0:
                    continue
                
                if action == "BUY":
                    result = portfolio.execute_buy(
                        ticker, amount, price, trade_reasoning, date, reasoning
                    )
                    if result:
                        print(f"      ✓ BUY {ticker}: ${amount:,.0f} @ ${price:.2f}")
                        trades_executed += 1
                
                elif action == "SELL":
                    result = portfolio.execute_sell(
                        ticker, amount, price, trade_reasoning, date, reasoning
                    )
                    if result:
                        print(f"      ✓ SELL {ticker}: ${amount:,.0f} @ ${price:.2f}")
                        trades_executed += 1
            
            if trades_executed == 0:
                print(f"      - No trades executed (holding)")
            
            portfolio.record_equity(date)
            print(f"      💰 Portfolio: ${portfolio.total_value:,.2f} ({portfolio.return_pct:+.2f}%)")
            
            time.sleep(1)  # Rate limiting
        
        self.rounds_completed += 1
        self.save_state()
    
    def run_backtest(self, max_rounds: Optional[int] = None):
        """Run through all available rounds"""
        rounds = get_all_available_rounds()
        
        if max_rounds:
            rounds = rounds[:max_rounds]
        
        print(f"\n{'#'*60}")
        print(f"🏟️  LLM TRADING ARENA - BACKTEST")
        print(f"{'#'*60}")
        print(f"Rounds: {len(rounds)} | Models: {len(ARENA_MODELS)}")
        print(f"Starting Capital: ${STARTING_CAPITAL:,}")
        print(f"Models: {', '.join(ARENA_MODELS.keys())}")
        print(f"{'#'*60}")
        
        for round_info in rounds:
            date = round_info["date"]
            
            if round_info["data_type"] == "rich":
                market_data = self._load_workflow_round(date)
            else:
                market_data = load_market_data(date)
            
            if not market_data:
                print(f"\n  ⚠ No data for {date}, skipping")
                continue
            
            self.run_round(round_info, market_data)
        
        self.print_leaderboard()
        self.save_results()
    
    def _load_workflow_round(self, date: str) -> dict:
        """Load workflow data for all tickers for a specific date"""
        market_data = {}
        
        for ticker in TICKERS:
            ticker_dir = WORKFLOW_DATA_DIR / ticker
            if not ticker_dir.exists():
                continue
            
            for sample_dir in ticker_dir.iterdir():
                if sample_dir.is_dir() and sample_dir.name.startswith(date):
                    data = load_workflow_data(ticker, sample_dir)
                    if data:
                        market_data[ticker] = data
                    break
        
        return market_data
    
    def print_leaderboard(self):
        """Print current leaderboard"""
        print(f"\n{'='*60}")
        print("🏆 FINAL LEADERBOARD")
        print(f"{'='*60}")
        
        sorted_portfolios = sorted(
            self.portfolios.items(),
            key=lambda x: x[1].total_value,
            reverse=True
        )
        
        for rank, (name, portfolio) in enumerate(sorted_portfolios, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            win_rate = self._calculate_win_rate(portfolio)
            print(f"{emoji} {rank}. {name:20} | ${portfolio.total_value:>10,.2f} | "
                  f"{portfolio.return_pct:>+6.2f}% | Trades: {len(portfolio.trades):>3} | "
                  f"Win Rate: {win_rate:.1f}%")
    
    def _calculate_win_rate(self, portfolio: Portfolio) -> float:
        """Calculate win rate from trades"""
        if not portfolio.trades:
            return 0
        winning_trades = sum(1 for t in portfolio.trades 
                           if t.action == "SELL" and "profit" in t.reasoning.lower())
        return (winning_trades / len(portfolio.trades)) * 100 if portfolio.trades else 0
    
    def save_state(self):
        """Save current arena state"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "rounds_completed": self.rounds_completed,
            "portfolios": {name: p.to_dict() for name, p in self.portfolios.items()}
        }
        
        with open(ARENA_OUTPUT_DIR / "arena_state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def save_results(self):
        """Save final results"""
        sorted_portfolios = sorted(
            self.portfolios.items(),
            key=lambda x: x[1].total_value,
            reverse=True
        )
        
        leaderboard = []
        for rank, (name, portfolio) in enumerate(sorted_portfolios, 1):
            leaderboard.append({
                "rank": rank,
                "model": name,
                "final_value": round(portfolio.total_value, 2),
                "return_pct": round(portfolio.return_pct, 2),
                "total_trades": len(portfolio.trades),
                "win_rate": round(self._calculate_win_rate(portfolio), 1)
            })
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "rounds_completed": self.rounds_completed,
            "starting_capital": STARTING_CAPITAL,
            "leaderboard": leaderboard,
            "portfolios": {name: p.to_dict() for name, p in self.portfolios.items()}
        }
        
        with open(ARENA_OUTPUT_DIR / "arena_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save trade history
        all_trades = []
        for name, portfolio in self.portfolios.items():
            for trade in portfolio.trades:
                all_trades.append({
                    "model": name,
                    **asdict(trade)
                })
        
        all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
        
        with open(ARENA_OUTPUT_DIR / "trade_history.json", "w") as f:
            json.dump(all_trades, f, indent=2)
        
        # Save reasoning history
        with open(ARENA_OUTPUT_DIR / "reasoning_history.json", "w") as f:
            json.dump(self.round_reasoning, f, indent=2)
        
        print(f"\n✓ Results saved to {ARENA_OUTPUT_DIR}")
        print(f"  - arena_results.json (final standings)")
        print(f"  - trade_history.json (all trades with reasoning)")
        print(f"  - reasoning_history.json (full CoT analysis per round)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Trading Arena")
    parser.add_argument("--run", action="store_true", help="Run the arena backtest")
    parser.add_argument("--rounds", type=int, default=None, help="Max rounds to run")
    parser.add_argument("--status", action="store_true", help="Show arena status")
    
    args = parser.parse_args()
    
    if args.status:
        workflow_dates = get_available_workflow_dates()
        market_dates = get_available_market_dates()
        
        print("🏟️  LLM Trading Arena Status")
        print("=" * 40)
        print(f"Workflow data (rich): {len(workflow_dates)} dates")
        if workflow_dates:
            print(f"  Range: {workflow_dates[0]} to {workflow_dates[-1]}")
        print(f"Market data (lite): {len(market_dates)} dates")
        if market_dates:
            print(f"  Range: {market_dates[0]} to {market_dates[-1]}")
        print(f"\nModels: {', '.join(ARENA_MODELS.keys())}")
        print(f"\nAPI Keys Status:")
        print(f"  OpenAI:  {'✓' if OPENAI_API_KEY else '✗'}")
        print(f"  Google:  {'✓' if GOOGLE_API_KEY else '✗'}")
        print(f"  Groq:    {'✓' if GROQ_API_KEY else '✗'}")
        print(f"  Mistral: {'✓' if MISTRAL_API_KEY else '✗'}")
                
    elif args.run:
        arena = TradingArena()
        arena.run_backtest(max_rounds=args.rounds)
    
    else:
        print("🏟️  LLM Trading Arena")
        print("=" * 40)
        print("\nUsage:")
        print("  python llm_arena.py --status         # Show data & API status")
        print("  python llm_arena.py --run            # Run full backtest")
        print("  python llm_arena.py --run --rounds 5 # Run 5 rounds only")