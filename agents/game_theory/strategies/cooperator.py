"""
cooperator.py - Cooperator Strategy (Follow the Pack)

Location: agents/game_theory/strategies/cooperator.py

The Cooperator takes positions SIMILAR to what other strategies are doing.
This is actual cooperation in game theory terms - aligning with the group.

Philosophy:
    "Safety in numbers. Don't rock the boat."

Game Theory Role:
    - COOPERATES by taking similar positions to others
    - Reduces variance in outcomes (everyone wins/loses together)
    - Won't steal capital, but won't lose it to defectors either
    - Stable, defensive strategy

Behavior:
    - Round 1: Use LLM consensus (no history yet)
    - Round 2+: Blend group average with LLM signal
    - Weight: 70% group, 30% LLM

When It Wins:
    - When the group is right (trending markets)
    - When defectors are wrong
    - In stable, predictable markets

When It Loses:
    - When the group is wrong
    - At turning points (slow to adapt)
    - When defectors correctly call reversals

Usage:
    from game_theory.strategies.cooperator import CooperatorStrategy
    
    strategy = CooperatorStrategy()
    position = strategy.decide_position(market, game)
"""

from typing import TYPE_CHECKING
import numpy as np
from .base import Strategy

if TYPE_CHECKING:
    from ..market_context import MarketContext
    from ..game_state import GameState


class CooperatorStrategy(Strategy):
    """
    Cooperator - Follow the Pack
    
    Takes positions similar to other strategies, creating stability.
    
    Position Logic:
        1. First round: Use LLM consensus position
        2. Later rounds: 70% group average + 30% LLM signal
        3. Gradually adapt to group movements
    
    In capital allocation game:
        - Low variance strategy (similar returns to group)
        - Allocation stays relatively stable
        - Protects against being "wrong alone"
        - Can't outperform much, but won't underperform much either
    """
    
    name = "Cooperator"
    description = "Follows the pack - takes similar positions to others"
    
    # Configuration
    GROUP_WEIGHT = 0.70     # How much to follow group
    LLM_WEIGHT = 0.30       # How much to follow LLM
    SCALE_FACTOR = 3.0      # Scale up LLM signals
    MIN_POSITION = 10.0     # Minimum position
    MAX_POSITION = 85.0     # Maximum position
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position by following the group.
        
        Blends group average position with LLM signal.
        """
        signals = self._get_llm_signals(market)
        consensus = self._get_llm_consensus(market)
        
        # Calculate LLM-based position
        llm_position = self._calculate_llm_position(market)
        
        # First round: just use LLM (no history)
        if game.round_num <= 1 or not game.last_positions:
            position = llm_position
            blend_info = f"Round 1, using LLM only"
        else:
            # Get group average (excluding self)
            group_avg = self._get_group_position(game)
            
            # Blend: group + LLM
            position = (group_avg * self.GROUP_WEIGHT) + (llm_position * self.LLM_WEIGHT)
            blend_info = f"Group={group_avg:.1f}%×{self.GROUP_WEIGHT} + LLM={llm_position:.1f}%×{self.LLM_WEIGHT}"
        
        # Adjust based on consensus
        # High consensus among LLM agents → more confident
        if consensus > 0.7:
            position *= 1.1
            consensus_adj = "+10% (high LLM consensus)"
        elif consensus < 0.3:
            position *= 0.9
            consensus_adj = "-10% (low LLM consensus)"
        else:
            consensus_adj = "none"
        
        # Clamp to valid range
        position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"Cooperator: {blend_info} | "
            f"LLM consensus={consensus:.0%} (adj: {consensus_adj}) | "
            f"Final={position:.1f}%"
        )
        
        self.record_decision(position, reasoning)
        return position
    
    def _calculate_llm_position(self, market: 'MarketContext') -> float:
        """Calculate position based on LLM signals."""
        avg_signal = np.mean([
            market.aggressive_position,
            market.neutral_position,
            market.conservative_position
        ])
        
        # Scale up
        position = avg_signal * 100 * self.SCALE_FACTOR
        
        # Clamp
        return self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
    
    def _get_group_position(self, game: 'GameState') -> float:
        """Get average position of OTHER strategies."""
        if not game.last_positions:
            return 50.0
        
        others = [v for k, v in game.last_positions.items() if k != self.name]
        return np.mean(others) if others else 50.0


# === Test ===
if __name__ == "__main__":
    print("Testing CooperatorStrategy...")
    print("=" * 60)
    
    class MockMarket:
        aggressive_position = 0.15
        neutral_position = 0.10
        conservative_position = 0.05
        regime = "bull"
    
    class MockGame:
        round_num = 1
        last_positions = {}
        allocations = {}
    
    strategy = CooperatorStrategy()
    market = MockMarket()
    game = MockGame()
    
    # Test 1: First round (no history)
    print("\nTest 1: First round (no group history)")
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: With group history - group is bullish
    print("\nTest 2: Group is bullish (avg 70%)")
    game.round_num = 5
    game.last_positions = {
        "Buy-and-Hold": 100,
        "Defector": 30,
        "Tit-for-Tat": 65,
        "Signal Follower": 55,
        "Cooperator": 60  # Self (should be excluded)
    }
    
    position = strategy.decide_position(market, game)
    print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: Group is bearish
    print("\nTest 3: Group is bearish (avg 25%)")
    game.last_positions = {
        "Buy-and-Hold": 100,
        "Defector": 10,
        "Tit-for-Tat": 20,
        "Signal Follower": 15,
        "Cooperator": 30
    }
    
    # LLM also cautious
    market.aggressive_position = 0.05
    market.neutral_position = 0.03
    market.conservative_position = 0.02
    
    position = strategy.decide_position(market, game)
    print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: Group is mixed
    print("\nTest 4: Group is mixed (high variance)")
    game.last_positions = {
        "Buy-and-Hold": 100,
        "Defector": 10,
        "Tit-for-Tat": 50,
        "Signal Follower": 40,
        "Cooperator": 45
    }
    
    position = strategy.decide_position(market, game)
    print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")