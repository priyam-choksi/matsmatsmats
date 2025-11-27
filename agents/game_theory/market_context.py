"""
market_context.py - Core Data Structure for Game Theory Tournament

This dataclass holds all information available for a trading decision:
- Market data (OHLCV, daily return)
- Three risk evaluation perspectives (aggressive/neutral/conservative)
- Derived fields (regime, volatility)

This is the input that all strategies receive to make their decisions.

Your data structure:
    outputs/game_theory/{TICKER}/portfolio_100000/sample_{N}/
    ├── date_info.json          # date, sample_number, market_data
    ├── aggressive_eval.json    # stance, position_size, confidence, reasoning
    ├── neutral_eval.json       # stance, position_size, confidence, reasoning
    └── conservative_eval.json  # stance, position_size, confidence, reasoning
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketContext:
    """
    All information available for a trading decision.
    
    This is populated by the DataLoader from your collected workflow data.
    Each strategy receives this context and decides what position to take.
    
    Attributes:
        # Identifiers
        date: Trading date (YYYY-MM-DD format)
        ticker: Stock symbol (e.g., "AAPL")
        sample_num: Sample number within the ticker's dataset
        portfolio_size: Portfolio size in dollars (default 100000)
        
        # Price data from date_info.json -> market_data
        open_price: Day's opening price
        close_price: Day's closing price
        high: Day's high price
        low: Day's low price
        volume: Trading volume
        daily_return: Day's return as decimal (e.g., 0.0123 = 1.23%)
        
        # Aggressive agent evaluation
        aggressive_stance: "BUY", "SELL", or "HOLD"
        aggressive_position: Position size as decimal (0.0 to 1.0)
        aggressive_confidence: "HIGH", "MEDIUM", or "LOW"
        aggressive_reasoning: Full reasoning text
        
        # Neutral agent evaluation
        neutral_stance: "BUY", "SELL", "HOLD", or "SMALL BUY"
        neutral_position: Position size as decimal (0.0 to 1.0)
        neutral_confidence: "HIGH", "MEDIUM", or "LOW"
        neutral_reasoning: Full reasoning text
        
        # Conservative agent evaluation
        conservative_stance: "BUY", "SELL", "HOLD", or "AVOID"
        conservative_position: Position size as decimal (0.0 to 1.0)
        conservative_confidence: "HIGH", "MEDIUM", or "LOW"
        conservative_reasoning: Full reasoning text
        
        # Derived fields (computed by DataLoader/RegimeDetector)
        regime: Market regime - "bull", "bear", or "sideways"
        volatility: Rolling standard deviation of recent returns
    """
    
    # === Identifiers ===
    date: str
    ticker: str
    sample_num: int
    portfolio_size: float = 100000.0
    
    # === Price Data (from date_info.json -> market_data) ===
    open_price: float = 0.0
    close_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    daily_return: float = 0.0  # As decimal: 0.0123 = 1.23%
    
    # === Aggressive Agent Evaluation ===
    aggressive_stance: str = "HOLD"        # BUY, SELL, HOLD
    aggressive_position: float = 0.0       # 0.0 to 1.0 (e.g., 0.195 = 19.5%)
    aggressive_confidence: str = "LOW"     # HIGH, MEDIUM, LOW
    aggressive_reasoning: str = ""
    
    # === Neutral Agent Evaluation ===
    neutral_stance: str = "HOLD"           # BUY, SELL, HOLD, SMALL BUY
    neutral_position: float = 0.0          # 0.0 to 1.0
    neutral_confidence: str = "LOW"        # HIGH, MEDIUM, LOW
    neutral_reasoning: str = ""
    
    # === Conservative Agent Evaluation ===
    conservative_stance: str = "HOLD"      # BUY, SELL, HOLD, AVOID
    conservative_position: float = 0.0     # 0.0 to 1.0
    conservative_confidence: str = "LOW"   # HIGH, MEDIUM, LOW
    conservative_reasoning: str = ""
    
    # === Derived Fields (set by DataLoader) ===
    regime: str = "sideways"               # bull, bear, sideways
    volatility: float = 0.0                # Rolling std of returns
    
    def __post_init__(self):
        """Validate and normalize data after initialization."""
        # Normalize stances to uppercase
        self.aggressive_stance = self._normalize_stance(self.aggressive_stance)
        self.neutral_stance = self._normalize_stance(self.neutral_stance)
        self.conservative_stance = self._normalize_stance(self.conservative_stance)
        
        # Normalize confidence to uppercase
        self.aggressive_confidence = self.aggressive_confidence.upper() if self.aggressive_confidence else "LOW"
        self.neutral_confidence = self.neutral_confidence.upper() if self.neutral_confidence else "LOW"
        self.conservative_confidence = self.conservative_confidence.upper() if self.conservative_confidence else "LOW"
        
        # Clamp position sizes to valid range [0, 1]
        self.aggressive_position = max(0.0, min(1.0, float(self.aggressive_position)))
        self.neutral_position = max(0.0, min(1.0, float(self.neutral_position)))
        self.conservative_position = max(0.0, min(1.0, float(self.conservative_position)))
        
        # Ensure regime is lowercase
        self.regime = self.regime.lower() if self.regime else "sideways"
    
    def _normalize_stance(self, stance: str) -> str:
        """Normalize stance values to standard format."""
        if not stance:
            return "HOLD"
        
        stance_upper = stance.upper().strip()
        
        # Map variations to standard values
        stance_map = {
            "BUY": "BUY",
            "STRONG BUY": "BUY",
            "SMALL BUY": "BUY",  # Treat as BUY for simplicity
            "SELL": "SELL",
            "STRONG SELL": "SELL",
            "HOLD": "HOLD",
            "AVOID": "HOLD",    # Treat AVOID as HOLD (no position)
            "WAIT": "HOLD",
        }
        
        return stance_map.get(stance_upper, "HOLD")
    
    # === Convenience Properties ===
    
    @property
    def average_position(self) -> float:
        """Average position size across all three evaluations."""
        return (self.aggressive_position + self.neutral_position + self.conservative_position) / 3.0
    
    @property
    def position_spread(self) -> float:
        """Spread between highest and lowest position recommendations."""
        positions = [self.aggressive_position, self.neutral_position, self.conservative_position]
        return max(positions) - min(positions)
    
    @property
    def consensus_level(self) -> float:
        """
        Measure of how much the three evaluations agree.
        Returns 0.0 (no consensus) to 1.0 (perfect consensus).
        
        Based on standard deviation of positions - lower std = higher consensus.
        Assumes max reasonable std is ~0.10 (10% spread).
        """
        import statistics
        positions = [self.aggressive_position, self.neutral_position, self.conservative_position]
        
        if len(set(positions)) == 1:  # All same
            return 1.0
        
        std = statistics.stdev(positions)
        # Normalize: std of 0 = consensus 1.0, std of 0.10+ = consensus 0.0
        return max(0.0, 1.0 - std / 0.10)
    
    @property
    def bullish_count(self) -> int:
        """Count of evaluations with BUY stance."""
        count = 0
        if self.aggressive_stance == "BUY":
            count += 1
        if self.neutral_stance == "BUY":
            count += 1
        if self.conservative_stance == "BUY":
            count += 1
        return count
    
    @property
    def high_confidence_count(self) -> int:
        """Count of evaluations with HIGH confidence."""
        count = 0
        if self.aggressive_confidence == "HIGH":
            count += 1
        if self.neutral_confidence == "HIGH":
            count += 1
        if self.conservative_confidence == "HIGH":
            count += 1
        return count
    
    @property
    def market_moved_up(self) -> bool:
        """True if daily return was positive."""
        return self.daily_return > 0
    
    @property
    def intraday_range(self) -> float:
        """Intraday price range as percentage of close."""
        if self.close_price <= 0:
            return 0.0
        return (self.high - self.low) / self.close_price
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "ticker": self.ticker,
            "sample_num": self.sample_num,
            "portfolio_size": self.portfolio_size,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "daily_return": self.daily_return,
            "aggressive": {
                "stance": self.aggressive_stance,
                "position": self.aggressive_position,
                "confidence": self.aggressive_confidence,
            },
            "neutral": {
                "stance": self.neutral_stance,
                "position": self.neutral_position,
                "confidence": self.neutral_confidence,
            },
            "conservative": {
                "stance": self.conservative_stance,
                "position": self.conservative_position,
                "confidence": self.conservative_confidence,
            },
            "regime": self.regime,
            "volatility": self.volatility,
            "derived": {
                "average_position": round(self.average_position, 4),
                "consensus_level": round(self.consensus_level, 4),
                "bullish_count": self.bullish_count,
                "high_confidence_count": self.high_confidence_count,
            }
        }
    
    def __repr__(self) -> str:
        """Readable string representation."""
        return (
            f"MarketContext({self.ticker} {self.date} sample#{self.sample_num} | "
            f"return={self.daily_return:+.2%} regime={self.regime} | "
            f"agg={self.aggressive_position:.0%} neu={self.neutral_position:.0%} "
            f"con={self.conservative_position:.0%})"
        )