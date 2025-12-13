"""
base.py - Base Strategy Class for Capital Allocation Tournament

Location: agents/game_theory/strategies/base.py

Simple, clean base class for all trading strategies.
Unlike the old system, strategies don't track their own scores.
The GAME provides dynamics through capital reallocation.

Each strategy just needs to implement:
    decide_position(market, game) -> float (0-100%)

That's it! The strategy sees:
    - market: LLM signals + market data (MarketContext)
    - game: What others did, current allocations (GameState)

And returns a position from 0-100%.

Usage:
    from game_theory.strategies.base import Strategy
    
    class MyStrategy(Strategy):
        name = "My Strategy"
        description = "Does something cool"
        
        def decide_position(self, market, game):
            # Your logic here
            return 50.0  # 50% position
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..market_context import MarketContext
    from ..game_state import GameState


class Strategy(ABC):
    """
    Abstract base class for capital allocation tournament strategies.
    
    Strategies compete for capital allocation by making position decisions.
    Each round:
        1. Strategy sees market context (LLM signals) and game state (what others did)
        2. Strategy decides position (0-100%)
        3. Market moves
        4. Capital reallocates based on relative performance
    
    The key insight: your decision should consider BOTH the market signals
    AND what other strategies might do, since you're competing for capital.
    
    Attributes:
        name: Strategy display name
        description: Brief explanation of strategy logic
    """
    
    name: str = "Base Strategy"
    description: str = "Abstract base - do not use directly"
    
    def __init__(self):
        """Initialize strategy."""
        # Track decisions for logging
        self.decision_history = []
        self.reasoning_history = []
    
    @abstractmethod
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position based on market context and game state.
        
        This is the ONLY method you need to implement.
        
        Args:
            market: MarketContext with LLM signals and market data
                - market.aggressive_position (0-1)
                - market.neutral_position (0-1)
                - market.conservative_position (0-1)
                - market.daily_return (actual return, for backtesting)
                - market.regime ("bull", "bear", "sideways")
                
            game: GameState with competition info
                - game.last_positions: What each strategy did last round
                - game.last_returns: What each strategy earned last round
                - game.last_winner: Who won last round
                - game.allocations: Current capital per strategy
                - game.round_num: Current round number
        
        Returns:
            Position as percentage (0-100)
            0 = no position, 100 = fully invested
        """
        pass
    
    def get_reasoning(self) -> str:
        """Get reasoning for last decision (for logging)."""
        return self.reasoning_history[-1] if self.reasoning_history else ""
    
    def record_decision(self, position: float, reasoning: str):
        """Record a decision for logging."""
        self.decision_history.append(position)
        self.reasoning_history.append(reasoning)
    
    def reset(self):
        """Reset strategy state for new tournament."""
        self.decision_history = []
        self.reasoning_history = []
    
    # === Helper Methods for Subclasses ===
    
    def _get_llm_signals(self, market: 'MarketContext') -> dict:
        """Extract LLM signals from market context."""
        return {
            "aggressive": market.aggressive_position * 100,  # Convert to %
            "neutral": market.neutral_position * 100,
            "conservative": market.conservative_position * 100,
            "avg": np.mean([
                market.aggressive_position,
                market.neutral_position,
                market.conservative_position
            ]) * 100
        }
    
    def _get_llm_consensus(self, market: 'MarketContext') -> float:
        """
        Calculate consensus among LLM agents.
        
        Returns:
            Consensus level 0-1 (1 = perfect agreement)
        """
        positions = [
            market.aggressive_position,
            market.neutral_position,
            market.conservative_position
        ]
        std = np.std(positions)
        # Normalize: std of 0.1 (10%) = 0 consensus, std of 0 = 1 consensus
        consensus = max(0, 1 - std / 0.10)
        return consensus
    
    def _get_strongest_signal(self, market: 'MarketContext') -> float:
        """Get the strongest (highest) LLM signal as percentage."""
        return max(
            market.aggressive_position,
            market.neutral_position,
            market.conservative_position
        ) * 100
    
    def _get_group_position(self, game: 'GameState') -> float:
        """Get average position of all other strategies last round."""
        if not game.last_positions:
            return 50.0  # Default neutral
        
        others = [v for k, v in game.last_positions.items() if k != self.name]
        return np.mean(others) if others else 50.0
    
    def _clamp_position(self, position: float, min_pos: float = 0, max_pos: float = 100) -> float:
        """Clamp position to valid range."""
        return max(min_pos, min(max_pos, position))
    
    def __repr__(self) -> str:
        return f"{self.name}()"


class StrategyResult:
    """
    Result of a strategy decision for a single round.
    
    Used for detailed logging.
    """
    def __init__(
        self,
        strategy_name: str,
        position: float,
        reasoning: str,
        llm_signals: dict = None,
        group_position: float = None
    ):
        self.strategy_name = strategy_name
        self.position = position
        self.reasoning = reasoning
        self.llm_signals = llm_signals or {}
        self.group_position = group_position
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "position_pct": round(self.position, 2),
            "reasoning": self.reasoning,
            "llm_signals": self.llm_signals,
            "group_position": round(self.group_position, 2) if self.group_position else None
        }


# === Test ===
if __name__ == "__main__":
    print("Strategy base class - cannot test directly")
    print("See individual strategy files for tests")