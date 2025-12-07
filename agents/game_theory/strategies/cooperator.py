# """
# cooperator.py - Cooperator Strategy (Follow the Pack)

# Location: agents/game_theory/strategies/cooperator.py

# The Cooperator takes positions SIMILAR to what other strategies are doing.
# This is actual cooperation in game theory terms - aligning with the group.

# Philosophy:
#     "Safety in numbers. Don't rock the boat."

# Game Theory Role:
#     - COOPERATES by taking similar positions to others
#     - Reduces variance in outcomes (everyone wins/loses together)
#     - Won't steal capital, but won't lose it to defectors either
#     - Stable, defensive strategy

# Behavior:
#     - Round 1: Use LLM consensus (no history yet)
#     - Round 2+: Blend group average with LLM signal
#     - Weight: 70% group, 30% LLM

# When It Wins:
#     - When the group is right (trending markets)
#     - When defectors are wrong
#     - In stable, predictable markets

# When It Loses:
#     - When the group is wrong
#     - At turning points (slow to adapt)
#     - When defectors correctly call reversals

# Usage:
#     from game_theory.strategies.cooperator import CooperatorStrategy
    
#     strategy = CooperatorStrategy()
#     position = strategy.decide_position(market, game)
# """

# from typing import TYPE_CHECKING
# import numpy as np
# from .base import Strategy

# if TYPE_CHECKING:
#     from ..market_context import MarketContext
#     from ..game_state import GameState


# class CooperatorStrategy(Strategy):
#     """
#     Cooperator - Follow the Pack
    
#     Takes positions similar to other strategies, creating stability.
    
#     Position Logic:
#         1. First round: Use LLM consensus position
#         2. Later rounds: 70% group average + 30% LLM signal
#         3. Gradually adapt to group movements
    
#     In capital allocation game:
#         - Low variance strategy (similar returns to group)
#         - Allocation stays relatively stable
#         - Protects against being "wrong alone"
#         - Can't outperform much, but won't underperform much either
#     """
    
#     name = "Cooperator"
#     description = "Follows the pack - takes similar positions to others"
    
#     # Configuration
#     GROUP_WEIGHT = 0.70     # How much to follow group
#     LLM_WEIGHT = 0.30       # How much to follow LLM
#     SCALE_FACTOR = 3.0      # Scale up LLM signals
#     MIN_POSITION = 10.0     # Minimum position
#     MAX_POSITION = 85.0     # Maximum position
    
#     def decide_position(
#         self,
#         market: 'MarketContext',
#         game: 'GameState'
#     ) -> float:
#         """
#         Decide position by following the group.
        
#         Blends group average position with LLM signal.
#         """
#         signals = self._get_llm_signals(market)
#         consensus = self._get_llm_consensus(market)
        
#         # Calculate LLM-based position
#         llm_position = self._calculate_llm_position(market)
        
#         # First round: just use LLM (no history)
#         if game.round_num <= 1 or not game.last_positions:
#             position = llm_position
#             blend_info = f"Round 1, using LLM only"
#         else:
#             # Get group average (excluding self)
#             group_avg = self._get_group_position(game)
            
#             # Blend: group + LLM
#             position = (group_avg * self.GROUP_WEIGHT) + (llm_position * self.LLM_WEIGHT)
#             blend_info = f"Group={group_avg:.1f}%×{self.GROUP_WEIGHT} + LLM={llm_position:.1f}%×{self.LLM_WEIGHT}"
        
#         # Adjust based on consensus
#         # High consensus among LLM agents → more confident
#         if consensus > 0.7:
#             position *= 1.1
#             consensus_adj = "+10% (high LLM consensus)"
#         elif consensus < 0.3:
#             position *= 0.9
#             consensus_adj = "-10% (low LLM consensus)"
#         else:
#             consensus_adj = "none"
        
#         # Clamp to valid range
#         position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
#         # Build reasoning
#         reasoning = (
#             f"Cooperator: {blend_info} | "
#             f"LLM consensus={consensus:.0%} (adj: {consensus_adj}) | "
#             f"Final={position:.1f}%"
#         )
        
#         self.record_decision(position, reasoning)
#         return position
    
#     def _calculate_llm_position(self, market: 'MarketContext') -> float:
#         """Calculate position based on LLM signals."""
#         avg_signal = np.mean([
#             market.aggressive_position,
#             market.neutral_position,
#             market.conservative_position
#         ])
        
#         # Scale up
#         position = avg_signal * 100 * self.SCALE_FACTOR
        
#         # Clamp
#         return self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
    
#     def _get_group_position(self, game: 'GameState') -> float:
#         """Get average position of OTHER strategies."""
#         if not game.last_positions:
#             return 50.0
        
#         others = [v for k, v in game.last_positions.items() if k != self.name]
#         return np.mean(others) if others else 50.0


# # === Test ===
# if __name__ == "__main__":
#     print("Testing CooperatorStrategy...")
#     print("=" * 60)
    
#     class MockMarket:
#         aggressive_position = 0.15
#         neutral_position = 0.10
#         conservative_position = 0.05
#         regime = "bull"
    
#     class MockGame:
#         round_num = 1
#         last_positions = {}
#         allocations = {}
    
#     strategy = CooperatorStrategy()
#     market = MockMarket()
#     game = MockGame()
    
#     # Test 1: First round (no history)
#     print("\nTest 1: First round (no group history)")
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 2: With group history - group is bullish
#     print("\nTest 2: Group is bullish (avg 70%)")
#     game.round_num = 5
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Defector": 30,
#         "Tit-for-Tat": 65,
#         "Signal Follower": 55,
#         "Cooperator": 60  # Self (should be excluded)
#     }
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 3: Group is bearish
#     print("\nTest 3: Group is bearish (avg 25%)")
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Defector": 10,
#         "Tit-for-Tat": 20,
#         "Signal Follower": 15,
#         "Cooperator": 30
#     }
    
#     # LLM also cautious
#     market.aggressive_position = 0.05
#     market.neutral_position = 0.03
#     market.conservative_position = 0.02
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 4: Group is mixed
#     print("\nTest 4: Group is mixed (high variance)")
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Defector": 10,
#         "Tit-for-Tat": 50,
#         "Signal Follower": 40,
#         "Cooperator": 45
#     }
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     print("\n" + "=" * 60)
#     print("TEST COMPLETE")


"""
cooperator.py - Cooperator Strategy (MOMENTUM)

Location: agents/game_theory/strategies/cooperator.py

UPDATED: Now implements MOMENTUM trading logic while keeping game theory name.

Game Theory Concept: "Cooperate with market direction"
Trading Logic: MOMENTUM - ride the trend, if market going up stay long

Philosophy:
    "The trend is your friend. What's been going up keeps going up."

When It Wins:
    - Trending markets (sustained up or down moves)
    - When momentum persists across multiple rounds
    - Strong directional markets

When It Loses:
    - At market turning points (slow to reverse)
    - In choppy, mean-reverting markets
    - When trends suddenly reverse

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
    Cooperator - MOMENTUM Strategy
    
    Game Theory: Cooperates with market direction (follows the trend)
    Trading Logic: If recent returns positive, stay long. Chase winners.
    
    Position Logic:
        1. Look at last N rounds of market returns
        2. If average return positive → bullish (high position)
        3. If average return negative → bearish (low position)
        4. Scale position based on momentum strength
    
    In capital allocation game:
        - Wins in trending markets
        - Loses at reversals
        - Medium-high variance strategy
    """
    
    name = "Cooperator"
    description = "MOMENTUM: Rides the trend, cooperates with market direction"
    
    # Configuration
    LOOKBACK = 3           # How many rounds to look back
    MIN_POSITION = 15.0    # Minimum position
    MAX_POSITION = 90.0    # Maximum position
    SCALE_FACTOR = 3.5     # Scale up LLM signals for first rounds
    
    # Momentum thresholds (as decimals, e.g., 0.01 = 1%)
    STRONG_UP = 0.015      # Strong uptrend threshold
    MILD_UP = 0.005        # Mild uptrend threshold
    MILD_DOWN = -0.005     # Mild downtrend threshold
    STRONG_DOWN = -0.015   # Strong downtrend threshold
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position based on recent momentum.
        
        Looks at recent market returns and follows the trend.
        """
        # First few rounds: not enough history, use LLM signal
        if len(game.rounds) < self.LOOKBACK:
            position = self._use_llm_signal(market)
            reasoning = (
                f"MOMENTUM (warmup): Only {len(game.rounds)} rounds, need {self.LOOKBACK} | "
                f"Using LLM signal → {position:.1f}%"
            )
            self.record_decision(position, reasoning)
            return position
        
        # Calculate recent momentum (average of last N returns)
        recent_returns = [r.market_return for r in game.rounds[-self.LOOKBACK:]]
        avg_momentum = sum(recent_returns) / len(recent_returns)
        
        # Also get last round's return for recency weighting
        last_return = game.rounds[-1].market_return
        
        # Momentum-based position sizing
        if avg_momentum > self.STRONG_UP:
            # Strong uptrend - go heavy
            position = 85.0
            trend = "STRONG UP"
        elif avg_momentum > self.MILD_UP:
            # Mild uptrend - moderately bullish
            position = 70.0
            trend = "MILD UP"
        elif avg_momentum > self.MILD_DOWN:
            # Sideways/neutral - moderate position
            position = 50.0
            trend = "NEUTRAL"
        elif avg_momentum > self.STRONG_DOWN:
            # Mild downtrend - reduce exposure
            position = 35.0
            trend = "MILD DOWN"
        else:
            # Strong downtrend - defensive
            position = 20.0
            trend = "STRONG DOWN"
        
        # Adjust based on last round's return (recency boost)
        if last_return > 0.02:  # Last round was very positive
            position = min(self.MAX_POSITION, position + 10)
        elif last_return < -0.02:  # Last round was very negative
            position = max(self.MIN_POSITION, position - 10)
        
        # Clamp to valid range
        position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"MOMENTUM: {trend} | "
            f"Avg({self.LOOKBACK}d): {avg_momentum*100:+.2f}% | "
            f"Last: {last_return*100:+.2f}% | "
            f"Position: {position:.1f}%"
        )
        
        self.record_decision(position, reasoning)
        return position
    
    def _use_llm_signal(self, market: 'MarketContext') -> float:
        """Fallback: use LLM signal when not enough history."""
        avg_signal = np.mean([
            market.aggressive_position,
            market.neutral_position,
            market.conservative_position
        ])
        position = avg_signal * 100 * self.SCALE_FACTOR
        return self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)


# === Test ===
if __name__ == "__main__":
    print("Testing CooperatorStrategy (MOMENTUM)...")
    print("=" * 60)
    
    class MockMarket:
        aggressive_position = 0.15
        neutral_position = 0.10
        conservative_position = 0.05
        regime = "bull"
    
    class MockRound:
        def __init__(self, market_return):
            self.market_return = market_return
    
    class MockGame:
        round_num = 1
        rounds = []
        last_positions = {}
        allocations = {}
    
    strategy = CooperatorStrategy()
    market = MockMarket()
    game = MockGame()
    
    # Test 1: First round (no history)
    print("\nTest 1: First round (uses LLM)")
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: Strong uptrend
    print("\nTest 2: Strong uptrend (+2% avg)")
    game.rounds = [MockRound(0.02), MockRound(0.025), MockRound(0.015)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~85-90%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: Mild uptrend
    print("\nTest 3: Mild uptrend (+0.8% avg)")
    game.rounds = [MockRound(0.01), MockRound(0.005), MockRound(0.009)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~70%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: Strong downtrend
    print("\nTest 4: Strong downtrend (-2% avg)")
    game.rounds = [MockRound(-0.02), MockRound(-0.025), MockRound(-0.015)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~15-20%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 5: Choppy/sideways
    print("\nTest 5: Choppy market (0% avg)")
    game.rounds = [MockRound(0.01), MockRound(-0.01), MockRound(0.005)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~50%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")