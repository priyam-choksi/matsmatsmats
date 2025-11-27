"""
base_strategy.py - Abstract Base Class for Trading Strategies

Location: agents/game_theory/base_strategy.py

This module defines:
1. TradeResult - Dataclass holding results of a single trade
2. TradingStrategy - Abstract base class all strategies inherit from

The SCORE SYSTEM is the core mechanic:
- Each strategy tracks a score from -10 to +10, starting at 0
- Score updates AFTER each trade based on outcome
- Each strategy interprets the same score DIFFERENTLY based on personality

Score Update Rules:
    | Market Movement | Position Type      | Outcome      | Points |
    |-----------------|-------------------|--------------|--------|
    | Market UP (>0)  | Aggressive (>50%) | Correct      | +3     |
    | Market UP (>0)  | Conservative (≤50%)| Missed gains | -1     |
    | Market DOWN (≤0)| Conservative (≤50%)| Avoided loss | +2     |
    | Market DOWN (≤0)| Aggressive (>50%) | Wrong        | -3     |

ENHANCED with:
- Position scaling support
- Initial score bias
- Recent performance tracking
- Strategy-specific memory
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from .market_context import MarketContext


@dataclass
class TradeResult:
    """
    Result of a single trade execution.
    
    Created by TradingStrategy.execute_trade() after each decision.
    Stored in strategy.trades list for historical analysis.
    
    Attributes:
        date: Trading date
        sample_num: Sample number for reference
        position_pct: Position size as percentage (0-100)
        market_return: Actual market return that day (decimal)
        trade_return: Strategy's return (market_return * position_pct/100)
        reasoning: Strategy's explanation for the decision
        regime: Market regime at time of trade
        score_before: Strategy's score before this trade
        score_after: Strategy's score after this trade
    """
    date: str
    sample_num: int
    position_pct: float           # 0-100
    market_return: float          # Actual market return (decimal)
    trade_return: float           # Strategy's return (decimal)
    reasoning: str
    regime: str = "sideways"
    score_before: float = 0.0
    score_after: float = 0.0
    
    @property
    def was_profitable(self) -> bool:
        """True if trade made money."""
        return self.trade_return > 0
    
    @property
    def was_aggressive(self) -> bool:
        """True if position was aggressive (>50%)."""
        return self.position_pct > 50
    
    @property
    def market_went_up(self) -> bool:
        """True if market return was positive."""
        return self.market_return > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "sample_num": self.sample_num,
            "position_pct": round(self.position_pct, 2),
            "market_return": round(self.market_return, 4),
            "trade_return": round(self.trade_return, 4),
            "reasoning": self.reasoning,
            "regime": self.regime,
            "score_before": round(self.score_before, 1),
            "score_after": round(self.score_after, 1),
        }


class TradingStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Implements the score-based decision system where each strategy:
    1. Receives MarketContext with all available information
    2. Decides position size (0-100%) based on its personality + score
    3. Gets score updated based on trade outcome
    4. Tracks equity curve, returns, and regime-specific performance
    
    ENHANCEMENTS:
    - position_scale: Multiplier for meaningful position sizes
    - initial_score: Starting bias for strategy differentiation
    - Recent performance tracking (is_hot, is_cold)
    - Strategy memory for learning
    
    Subclasses must implement:
        decide_position(ctx, history) -> (position_pct, reasoning)
    
    Attributes:
        name: Strategy display name
        description: Brief description of strategy philosophy
        position_scale: Multiplier for position sizes
        initial_score: Starting score bias
        score: Current score (-10 to +10)
        score_history: List of score values over time
        equity_curve: Cumulative equity (starts at 1.0)
        daily_returns: List of daily returns
        positions: List of position sizes taken
        trades: List of TradeResult objects
        regime_returns: Returns grouped by regime
        strategy_memory: Dict for strategy-specific learning
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        position_scale: float = 1.0,
        initial_score: float = 0.0
    ):
        """
        Initialize strategy.
        
        Args:
            name: Display name (e.g., "Cooperator")
            description: Brief description of strategy logic
            position_scale: Multiplier for position sizes (default 1.0)
            initial_score: Starting score bias (default 0.0)
        """
        self.name = name
        self.description = description
        self.position_scale = position_scale
        self.initial_score = initial_score
        
        # Score system (-10 to +10)
        self.score: float = initial_score
        self.score_history: List[float] = [initial_score]
        
        # Performance tracking
        self.equity_curve: List[float] = [1.0]  # Starts at 1.0 (100%)
        self.daily_returns: List[float] = []
        self.positions: List[float] = []
        self.trades: List[TradeResult] = []
        
        # Regime-specific tracking
        self.regime_returns: Dict[str, List[float]] = {
            'bull': [],
            'bear': [],
            'sideways': []
        }
        
        # Strategy-specific memory for learning
        self.strategy_memory: Dict[str, any] = {}
    
    @abstractmethod
    def decide_position(
        self, 
        ctx: MarketContext, 
        history: List[TradeResult]
    ) -> Tuple[float, str]:
        """
        Decide position size based on market context and history.
        
        This is the core method each strategy must implement.
        The strategy should use:
        - ctx: Current market data and agent evaluations
        - self.score: Current score (-10 to +10) to adjust behavior
        - history: Past trades for pattern analysis
        
        Args:
            ctx: MarketContext with all available information
            history: List of past TradeResult objects
            
        Returns:
            Tuple of (position_pct, reasoning) where:
            - position_pct: Position size as percentage (0-100)
            - reasoning: String explaining the decision
        """
        pass
    
    def update_score(self, position_pct: float, market_return: float) -> float:
        """
        Update score based on trade outcome.
        
        Score System:
            | Market    | Position      | Outcome      | Points |
            |-----------|---------------|--------------|--------|
            | UP (>0)   | Aggressive    | Correct      | +3     |
            | UP (>0)   | Conservative  | Missed gains | -1     |
            | DOWN (≤0) | Conservative  | Avoided loss | +2     |
            | DOWN (≤0) | Aggressive    | Wrong        | -3     |
        
        Args:
            position_pct: Position size as percentage (0-100)
            market_return: Actual market return (decimal)
            
        Returns:
            The score change that was applied
        """
        aggressive = position_pct > 50
        market_up = market_return > 0
        
        # Determine score change
        if market_up and aggressive:
            change = +3  # Correct: aggressive when market went up
        elif market_up and not aggressive:
            change = -1  # Missed gains: conservative when market went up
        elif not market_up and not aggressive:
            change = +2  # Avoided loss: conservative when market went down
        else:  # market down and aggressive
            change = -3  # Wrong: aggressive when market went down
        
        # Apply change with clamping to [-10, +10]
        old_score = self.score
        self.score = max(-10.0, min(10.0, self.score + change))
        self.score_history.append(self.score)
        
        return change
    
    def execute_trade(self, ctx: MarketContext) -> TradeResult:
        """
        Execute a trade: get decision, calculate return, update score.
        
        This is the main method called by the tournament engine.
        
        Flow:
            1. Call decide_position() to get strategy's decision
            2. Calculate trade return based on position and market move
            3. Update equity curve and tracking lists
            4. Update score based on outcome
            5. Return TradeResult
        
        Args:
            ctx: MarketContext with all information for this period
            
        Returns:
            TradeResult with full details of the trade
        """
        # Store score before trade
        score_before = self.score
        
        # Get strategy's decision
        position_pct, reasoning = self.decide_position(ctx, self.trades)
        
        # Clamp position to valid range [0, 100]
        position_pct = max(0.0, min(100.0, position_pct))
        
        # Calculate trade return
        # If position is 60% and market returns 2%, trade return is 1.2%
        trade_return = ctx.daily_return * (position_pct / 100.0)
        
        # Update tracking lists
        self.daily_returns.append(trade_return)
        self.positions.append(position_pct)
        
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1.0 + trade_return)
        self.equity_curve.append(new_equity)
        
        # Update score AFTER seeing the outcome
        self.update_score(position_pct, ctx.daily_return)
        
        # Track by regime
        regime = ctx.regime.lower() if ctx.regime else "sideways"
        if regime in self.regime_returns:
            self.regime_returns[regime].append(trade_return)
        else:
            self.regime_returns["sideways"].append(trade_return)
        
        # Create trade result
        result = TradeResult(
            date=ctx.date,
            sample_num=ctx.sample_num,
            position_pct=position_pct,
            market_return=ctx.daily_return,
            trade_return=trade_return,
            reasoning=reasoning,
            regime=regime,
            score_before=score_before,
            score_after=self.score
        )
        
        self.trades.append(result)
        return result
    
    def reset(self):
        """
        Reset strategy state for a new tournament.
        
        Call this before running on a new ticker or new simulation.
        Preserves the initial_score bias.
        """
        self.score = self.initial_score
        self.score_history = [self.initial_score]
        self.equity_curve = [1.0]
        self.daily_returns = []
        self.positions = []
        self.trades = []
        self.regime_returns = {
            'bull': [],
            'bear': [],
            'sideways': []
        }
        self.strategy_memory = {}
    
    # === Enhanced Performance Properties ===
    
    @property
    def recent_performance(self) -> float:
        """Get win rate of last 10 trades."""
        if len(self.trades) < 5:
            return 0.5  # Neutral if not enough data
        
        recent = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        wins = sum(1 for t in recent if t.trade_return > 0)
        return wins / len(recent)
    
    @property
    def is_hot(self) -> bool:
        """True if recent performance > 60%"""
        return self.recent_performance > 0.6
    
    @property
    def is_cold(self) -> bool:
        """True if recent performance < 40%"""
        return self.recent_performance < 0.4
    
    @property
    def recent_average_return(self) -> float:
        """Average return of last 10 trades."""
        if not self.daily_returns:
            return 0.0
        recent = self.daily_returns[-10:] if len(self.daily_returns) >= 10 else self.daily_returns
        return np.mean(recent) if recent else 0.0
    
    # === Convenience Properties ===
    
    @property
    def total_return(self) -> float:
        """Total return as decimal (e.g., 0.15 = 15%)."""
        if not self.equity_curve:
            return 0.0
        return self.equity_curve[-1] - 1.0
    
    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        return self.total_return * 100.0
    
    @property
    def num_trades(self) -> int:
        """Number of trades executed."""
        return len(self.trades)
    
    @property
    def avg_position(self) -> float:
        """Average position size."""
        if not self.positions:
            return 0.0
        return sum(self.positions) / len(self.positions)
    
    @property
    def win_rate(self) -> float:
        """Percentage of profitable trades."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.trade_return > 0)
        return wins / len(self.trades) * 100.0
    
    @property
    def score_trend(self) -> str:
        """Current score trend: 'improving', 'declining', or 'stable'."""
        if len(self.score_history) < 5:
            return "stable"
        recent = self.score_history[-5:]
        if recent[-1] > recent[0] + 2:
            return "improving"
        elif recent[-1] < recent[0] - 2:
            return "declining"
        return "stable"
    
    def get_regime_performance(self) -> Dict[str, float]:
        """Get total return by regime."""
        result = {}
        for regime, returns in self.regime_returns.items():
            if returns:
                # Cumulative return for this regime
                cumulative = 1.0
                for r in returns:
                    cumulative *= (1.0 + r)
                result[regime] = (cumulative - 1.0) * 100.0  # As percentage
            else:
                result[regime] = 0.0
        return result
    
    def get_summary(self) -> dict:
        """Get summary statistics for this strategy."""
        return {
            "name": self.name,
            "description": self.description,
            "position_scale": self.position_scale,
            "initial_score": self.initial_score,
            "total_return_pct": round(self.total_return_pct, 2),
            "num_trades": self.num_trades,
            "win_rate": round(self.win_rate, 1),
            "avg_position": round(self.avg_position, 1),
            "final_score": round(self.score, 1),
            "score_trend": self.score_trend,
            "recent_performance": round(self.recent_performance * 100, 1),
            "is_hot": self.is_hot,
            "is_cold": self.is_cold,
            "regime_performance": self.get_regime_performance()
        }
    
    def __repr__(self) -> str:
        """Readable string representation."""
        return (
            f"{self.name}(return={self.total_return_pct:+.2f}%, "
            f"score={self.score:+.1f}, trades={self.num_trades})"
        )


# === Helper Functions ===

def get_score_interpretation(score: float) -> str:
    """
    Get human-readable interpretation of score.
    
    Args:
        score: Score value from -10 to +10
        
    Returns:
        Description of what the score means
    """
    if score >= 7:
        return "Excellent - strategy is performing very well"
    elif score >= 4:
        return "Good - strategy is on a winning streak"
    elif score >= 1:
        return "Slightly positive - recent trades working"
    elif score >= -1:
        return "Neutral - mixed results"
    elif score >= -4:
        return "Slightly negative - recent trades not working"
    elif score >= -7:
        return "Poor - strategy is struggling"
    else:
        return "Very poor - strategy needs to adapt"


def calculate_position_aggressiveness(position_pct: float) -> str:
    """
    Categorize position size.
    
    Args:
        position_pct: Position as percentage (0-100)
        
    Returns:
        Category string
    """
    if position_pct >= 75:
        return "very_aggressive"
    elif position_pct >= 50:
        return "aggressive"
    elif position_pct >= 25:
        return "moderate"
    elif position_pct > 0:
        return "conservative"
    else:
        return "no_position"