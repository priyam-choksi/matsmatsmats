"""
market_context.py - Core Data Structure for Game Theory Tournament

UPDATED: Now includes rich signal data from your workflow:
- Risk/reward ratios
- Expected values
- Bull/bear probabilities
- Confidence levels

This gives strategies much more information to make differentiated decisions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketContext:
    """
    All information available for a trading decision.
    
    ENHANCED with rich signal data from your evaluators.
    """
    
    # === Identifiers ===
    date: str = ""
    ticker: str = ""
    sample_num: int = 0
    portfolio_size: float = 100000.0
    
    # === Price Data ===
    open_price: float = 0.0
    close_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    daily_return: float = 0.0  # Actual market return (decimal)
    
    # === Aggressive Agent Evaluation ===
    aggressive_stance: str = "HOLD"
    aggressive_position: float = 0.0
    aggressive_confidence: str = "LOW"
    aggressive_reasoning: str = ""
    
    # === Neutral Agent Evaluation ===
    neutral_stance: str = "HOLD"
    neutral_position: float = 0.0
    neutral_confidence: str = "LOW"
    neutral_reasoning: str = ""
    
    # === Conservative Agent Evaluation ===
    conservative_stance: str = "HOLD"
    conservative_position: float = 0.0
    conservative_confidence: str = "LOW"
    conservative_reasoning: str = ""
    
    # === NEW: Rich Signal Data (from calculated_metrics) ===
    risk_reward_ratio: float = 0.0
    expected_value: float = 0.0
    bull_prob_pct: float = 33.3
    bear_prob_pct: float = 33.3
    upside_pct: float = 0.0
    downside_pct: float = 0.0
    
    # === Derived Fields ===
    regime: str = "sideways"
    volatility: float = 0.0
    
    def __post_init__(self):
        """Validate and normalize data after initialization."""
        # Normalize stances
        self.aggressive_stance = self._normalize_stance(self.aggressive_stance)
        self.neutral_stance = self._normalize_stance(self.neutral_stance)
        self.conservative_stance = self._normalize_stance(self.conservative_stance)
        
        # Normalize confidence
        self.aggressive_confidence = (self.aggressive_confidence or "LOW").upper()
        self.neutral_confidence = (self.neutral_confidence or "LOW").upper()
        self.conservative_confidence = (self.conservative_confidence or "LOW").upper()
        
        # Clamp positions
        self.aggressive_position = max(0.0, min(1.0, float(self.aggressive_position)))
        self.neutral_position = max(0.0, min(1.0, float(self.neutral_position)))
        self.conservative_position = max(0.0, min(1.0, float(self.conservative_position)))
        
        # Ensure regime is lowercase
        self.regime = (self.regime or "sideways").lower()
    
    def _normalize_stance(self, stance: str) -> str:
        """Normalize stance values to standard format."""
        if not stance:
            return "HOLD"
        
        stance_upper = stance.upper().strip()
        
        stance_map = {
            "BUY": "BUY",
            "STRONG BUY": "BUY",
            "SMALL BUY": "BUY",
            "SELL": "SELL",
            "STRONG SELL": "SELL",
            "HOLD": "HOLD",
            "AVOID": "AVOID",
            "WAIT": "HOLD",
        }
        
        return stance_map.get(stance_upper, "HOLD")
    
    # === Signal Analysis Properties ===
    
    @property
    def bullish_count(self) -> int:
        """Count of bullish stances (BUY)."""
        count = 0
        if self.aggressive_stance == "BUY":
            count += 1
        if self.neutral_stance == "BUY":
            count += 1
        if self.conservative_stance == "BUY":
            count += 1
        return count
    
    @property
    def bearish_count(self) -> int:
        """Count of bearish stances (AVOID, SELL)."""
        count = 0
        if self.aggressive_stance in ["AVOID", "SELL"]:
            count += 1
        if self.neutral_stance in ["AVOID", "SELL"]:
            count += 1
        if self.conservative_stance in ["AVOID", "SELL"]:
            count += 1
        return count
    
    @property
    def neutral_count(self) -> int:
        """Count of neutral stances (HOLD)."""
        return 3 - self.bullish_count - self.bearish_count
    
    @property
    def signal_consensus(self) -> str:
        """
        Overall signal consensus.
        Returns: 'bullish', 'bearish', 'mixed', or 'neutral'
        """
        if self.bullish_count >= 2:
            return "bullish"
        elif self.bearish_count >= 2:
            return "bearish"
        elif self.bullish_count == 1 and self.bearish_count == 1:
            return "mixed"
        else:
            return "neutral"
    
    @property
    def signal_strength(self) -> float:
        """
        Signal strength from -1 (all bearish) to +1 (all bullish).
        """
        return (self.bullish_count - self.bearish_count) / 3.0
    
    @property
    def has_high_confidence(self) -> bool:
        """True if any agent has HIGH confidence."""
        return "HIGH" in [
            self.aggressive_confidence,
            self.neutral_confidence,
            self.conservative_confidence
        ]
    
    @property
    def confidence_score(self) -> float:
        """
        Average confidence as a score (LOW=0.33, MEDIUM=0.66, HIGH=1.0).
        """
        conf_map = {"LOW": 0.33, "MEDIUM": 0.66, "HIGH": 1.0}
        scores = [
            conf_map.get(self.aggressive_confidence, 0.33),
            conf_map.get(self.neutral_confidence, 0.33),
            conf_map.get(self.conservative_confidence, 0.33)
        ]
        return sum(scores) / 3.0
    
    @property
    def average_position(self) -> float:
        """Average recommended position across all agents."""
        return (self.aggressive_position + self.neutral_position + self.conservative_position) / 3.0
    
    @property 
    def is_good_risk_reward(self) -> bool:
        """True if R/R ratio >= 1.5 (favorable)."""
        return self.risk_reward_ratio >= 1.5
    
    @property
    def is_positive_ev(self) -> bool:
        """True if expected value is positive."""
        return self.expected_value > 0
    
    def __repr__(self) -> str:
        return (
            f"MarketContext({self.ticker} {self.date} | "
            f"signal={self.signal_consensus} | "
            f"R/R={self.risk_reward_ratio:.2f} | "
            f"EV={self.expected_value:+.2f}%)"
        )