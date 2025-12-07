# """
# tit_for_tat.py - Tit-for-Tat Adaptive Strategy

# Location: agents/game_theory/strategies/tit_for_tat.py

# The Tit-for-Tat strategy copies what worked last round.
# This is the famous strategy that won Axelrod's tournament.

# Philosophy:
#     "Do what worked. Adapt to the environment."

# Game Theory Role:
#     - Starts by COOPERATING (nice)
#     - Copies the WINNER's approach each round
#     - Adapts quickly to what's working
#     - Balances between cooperation and defection

# Original Tit-for-Tat Properties:
#     1. Nice: Starts by cooperating
#     2. Retaliatory: Copies defection if opponent defected
#     3. Forgiving: Goes back to cooperating if opponent cooperates
#     4. Clear: Easy to understand, predictable

# Our Adaptation:
#     - "Cooperate" = Follow the winner
#     - "Defect" = Go against the winner
#     - We always copy the winner, making us adaptive

# When It Wins:
#     - In changing regimes (quickly adapts)
#     - When there's a clear winning strategy
#     - In predictable markets

# When It Loses:
#     - When winners alternate randomly
#     - When it copies right before a reversal
#     - In choppy, unpredictable markets

# Usage:
#     from game_theory.strategies.tit_for_tat import TitForTatStrategy
    
#     strategy = TitForTatStrategy()
#     position = strategy.decide_position(market, game)
# """

# from typing import TYPE_CHECKING
# import numpy as np
# from .base import Strategy

# if TYPE_CHECKING:
#     from ..market_context import MarketContext
#     from ..game_state import GameState


# class TitForTatStrategy(Strategy):
#     """
#     Tit-for-Tat - Copy What Worked
    
#     Adaptive strategy that copies the winning approach each round.
    
#     Position Logic:
#         1. First round: Cooperate (follow group/LLM blend)
#         2. Later rounds: Copy what the winner did
#             - Winner was bullish? Go bullish
#             - Winner was bearish? Go bearish
#         3. Blend: 80% copy winner, 20% own judgment
    
#     In capital allocation game:
#         - Medium variance strategy
#         - Adapts to winning approaches
#         - Should accumulate capital in trending regimes
#         - May lag at turning points (copies OLD winner)
#     """
    
#     name = "Tit-for-Tat"
#     description = "Copies what worked last round - adaptive learner"
    
#     # Configuration
#     COPY_WEIGHT = 0.80      # How much to copy winner
#     OWN_WEIGHT = 0.20       # How much own judgment
#     MIN_POSITION = 15.0     # Minimum position
#     MAX_POSITION = 85.0     # Maximum position
#     SCALE_FACTOR = 3.0
    
#     def __init__(self):
#         super().__init__()
#         # Track who we're copying
#         self.copying = None
#         self.copy_history = []
    
#     def decide_position(
#         self,
#         market: 'MarketContext',
#         game: 'GameState'
#     ) -> float:
#         """
#         Decide position by copying the last winner.
        
#         Adapts to what's working in the current environment.
#         """
#         signals = self._get_llm_signals(market)
        
#         # Calculate our own judgment (LLM-based)
#         own_position = self._calculate_llm_position(market)
        
#         # First round: cooperate (follow group average / LLM blend)
#         if game.round_num <= 1 or not game.last_winner:
#             position = own_position
#             self.copying = None
#             reasoning = f"Tit-for-Tat: Round 1, starting nice → {position:.1f}%"
#             self.record_decision(position, reasoning)
#             return position
        
#         # Get winner's last position
#         winner = game.last_winner
#         winner_position = game.last_positions.get(winner, 50.0)
        
#         # Track who we're copying
#         self.copying = winner
#         self.copy_history.append(winner)
        
#         # Blend: copy winner + own judgment
#         position = (winner_position * self.COPY_WEIGHT) + (own_position * self.OWN_WEIGHT)
        
#         # Check winner's streak
#         winner_streak = self._get_winner_streak(game)
        
#         # If winner has long streak, follow more strongly
#         if winner_streak >= 3:
#             # Strong streak - follow more closely
#             position = (winner_position * 0.9) + (own_position * 0.1)
#             streak_adj = f"streak={winner_streak} → 90% copy"
#         elif winner_streak == 1:
#             # Just started winning - be cautious
#             position = (winner_position * 0.7) + (own_position * 0.3)
#             streak_adj = f"new winner → 70% copy"
#         else:
#             streak_adj = "normal copy"
        
#         # Clamp position
#         position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
#         # Build reasoning
#         reasoning = (
#             f"Tit-for-Tat: Copying '{winner}' ({winner_position:.1f}%) | "
#             f"{streak_adj} | "
#             f"Own={own_position:.1f}% | "
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
#         position = avg_signal * 100 * self.SCALE_FACTOR
#         return self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
    
#     def _get_winner_streak(self, game: 'GameState') -> int:
#         """Get current winner's streak length."""
#         if not game.rounds:
#             return 0
        
#         current_winner = game.last_winner
#         streak = 0
        
#         for r in reversed(game.rounds):
#             if r.winner == current_winner:
#                 streak += 1
#             else:
#                 break
        
#         return streak
    
#     def reset(self):
#         """Reset strategy state."""
#         super().reset()
#         self.copying = None
#         self.copy_history = []
    
#     def get_copy_stats(self) -> dict:
#         """Get statistics about copying behavior."""
#         if not self.copy_history:
#             return {}
        
#         from collections import Counter
#         counts = Counter(self.copy_history)
#         total = len(self.copy_history)
        
#         return {
#             "total_copies": total,
#             "copy_counts": dict(counts),
#             "copy_rates": {k: v/total for k, v in counts.items()}
#         }


# # === Test ===
# if __name__ == "__main__":
#     print("Testing TitForTatStrategy...")
#     print("=" * 60)
    
#     class MockMarket:
#         aggressive_position = 0.15
#         neutral_position = 0.10
#         conservative_position = 0.05
#         regime = "bull"
    
#     class MockRound:
#         def __init__(self, winner):
#             self.winner = winner
    
#     class MockGame:
#         round_num = 1
#         last_positions = {}
#         last_winner = None
#         rounds = []
#         allocations = {}
    
#     strategy = TitForTatStrategy()
#     market = MockMarket()
#     game = MockGame()
    
#     # Test 1: First round (start nice)
#     print("\nTest 1: First round (start nice)")
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Copying: {strategy.copying}")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 2: Copy Buy-and-Hold (100%)
#     print("\nTest 2: Copy Buy-and-Hold winner (100%)")
#     game.round_num = 5
#     game.last_winner = "Buy-and-Hold"
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Cooperator": 60,
#         "Defector": 30,
#         "Signal Follower": 50,
#         "Tit-for-Tat": 55
#     }
#     game.rounds = [MockRound("Buy-and-Hold")]
    
#     position = strategy.decide_position(market, game)
#     print(f"  Winner: {game.last_winner} (pos={game.last_positions['Buy-and-Hold']}%)")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Copying: {strategy.copying}")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 3: Copy Defector (30%)
#     print("\nTest 3: Copy Defector winner (30%)")
#     game.last_winner = "Defector"
#     game.rounds = [MockRound("Defector")]
    
#     position = strategy.decide_position(market, game)
#     print(f"  Winner: {game.last_winner} (pos={game.last_positions['Defector']}%)")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Copying: {strategy.copying}")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 4: Long winner streak
#     print("\nTest 4: Buy-and-Hold on 5-round streak")
#     game.last_winner = "Buy-and-Hold"
#     game.rounds = [MockRound("Buy-and-Hold") for _ in range(5)]
    
#     position = strategy.decide_position(market, game)
#     print(f"  Winner streak: {strategy._get_winner_streak(game)}")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 5: New winner
#     print("\nTest 5: Signal Follower just started winning")
#     game.last_winner = "Signal Follower"
#     game.rounds = [MockRound("Buy-and-Hold"), MockRound("Buy-and-Hold"), MockRound("Signal Follower")]
    
#     position = strategy.decide_position(market, game)
#     print(f"  Winner streak: {strategy._get_winner_streak(game)}")
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     print("\n" + "=" * 60)
#     print("Copy stats:", strategy.get_copy_stats())
#     print("TEST COMPLETE")


"""
tit_for_tat.py - Tit-for-Tat Adaptive Strategy

Location: agents/game_theory/strategies/tit_for_tat.py

UPDATED: Now learns from WHY the winner won, not just their position.

The classic Tit-for-Tat from Axelrod's tournament, adapted for trading:
- Starts "nice" (cooperative, follows signals)
- Copies the winning APPROACH each round
- Adapts quickly to market regime

Philosophy:
    "Do what worked. But understand WHY it worked."

Smart Adaptation:
    - If Momentum won → market is trending → be bullish
    - If Mean Reversion won → market reversed → be contrarian
    - If Signal Follower won → LLM was right → trust signals
    - Also considers: was the winner bullish or bearish?

When It Wins:
    - When market regime is consistent (winners repeat)
    - When there's a clear pattern to follow
    - In predictable transitions

When It Loses:
    - When winners alternate randomly
    - At sudden regime changes (copies old winner)
    - When all strategies perform similarly

Usage:
    from game_theory.strategies.tit_for_tat import TitForTatStrategy
    
    strategy = TitForTatStrategy()
    position = strategy.decide_position(market, game)
"""

from typing import TYPE_CHECKING
import numpy as np
from .base import Strategy

if TYPE_CHECKING:
    from ..market_context import MarketContext
    from ..game_state import GameState


class TitForTatStrategy(Strategy):
    """
    Tit-for-Tat - Smart Adaptive Strategy
    
    Learns from both WHO won and WHY they won.
    
    Position Logic:
        1. First round: Start nice (follow LLM signals)
        2. Later rounds: Analyze last winner
           - Copy their approach (momentum vs mean reversion)
           - Blend with current market context
           - Weight by winner's streak (longer streak = more confidence)
    
    In capital allocation game:
        - Medium variance strategy
        - Quick to adapt to regime changes
        - Benefits from consistent winners
        - Lags by one round at turning points
    """
    
    name = "Tit-for-Tat"
    description = "Smart adaptive: learns from winner's approach, not just position"
    
    # Configuration
    MIN_POSITION = 20.0
    MAX_POSITION = 85.0
    SCALE_FACTOR = 3.5
    
    # How much to weight winner's approach vs own judgment
    BASE_COPY_WEIGHT = 0.70
    STREAK_BONUS = 0.05  # Extra weight per streak length
    
    def __init__(self):
        super().__init__()
        self.copying = None
        self.copy_history = []
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position by learning from the last winner.
        
        Understands WHY they won and adapts approach accordingly.
        """
        # First round: start nice (follow LLM signals)
        if game.round_num < 1 or not game.last_winner:
            position = self._get_llm_position(market)
            reasoning = f"T4T (start nice): Following LLM signals → {position:.1f}%"
            self.record_decision(position, reasoning)
            self.copying = "LLM"
            self.copy_history.append("LLM")
            return position
        
        # Get winner info
        winner = game.last_winner
        winner_position = game.last_positions.get(winner, 50)
        winner_streak, _ = self._get_winner_streak(game)
        
        # Get last market return to understand context
        last_return = game.rounds[-1].market_return if game.rounds else 0
        
        # Determine what approach won and why
        position, approach = self._learn_from_winner(
            winner, winner_position, last_return, market, game
        )
        
        # Adjust confidence based on streak
        # Longer streak = winner's approach is working = trust it more
        copy_weight = min(0.90, self.BASE_COPY_WEIGHT + winner_streak * self.STREAK_BONUS)
        
        # Blend with our own LLM-based judgment
        own_position = self._get_llm_position(market)
        final_position = copy_weight * position + (1 - copy_weight) * own_position
        
        # Clamp
        final_position = self._clamp_position(final_position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"T4T: Copying {winner} ({approach}) | "
            f"Winner pos: {winner_position:.0f}% | "
            f"Streak: {winner_streak} | "
            f"Last mkt: {last_return*100:+.2f}% | "
            f"Copy weight: {copy_weight:.0%} | "
            f"Position: {final_position:.1f}%"
        )
        
        self.record_decision(final_position, reasoning)
        self.copying = winner
        self.copy_history.append(winner)
        
        return final_position
    
    def _learn_from_winner(
        self,
        winner: str,
        winner_position: float,
        last_return: float,
        market: 'MarketContext',
        game: 'GameState'
    ) -> tuple:
        """
        Learn from WHY the winner won and adapt approach.
        
        Returns:
            (position, approach_description)
        """
        # Analyze what the winner did and why it worked
        
        if winner == "Cooperator":
            # Momentum won - market was trending
            # → I should also ride momentum
            if last_return > 0:
                # Uptrend - be bullish
                position = max(70, winner_position)
                approach = "momentum-bullish"
            else:
                # Downtrend momentum worked - be cautious
                position = min(40, winner_position)
                approach = "momentum-bearish"
                
        elif winner == "Defector":
            # Mean reversion won - market reversed
            # → I should also fade extremes
            if last_return > 0:
                # Market went up but Defector won?
                # They were probably positioned for bounce that happened
                position = 65  # Moderate, expect normalization
                approach = "mean-reversion-neutral"
            else:
                # Market went down, Defector won by being light
                # Expect bounce, position for it
                position = 70
                approach = "mean-reversion-bounce"
                
        elif winner == "Signal Follower":
            # LLM signals were right
            # → Trust the LLM more heavily
            position = self._get_llm_position(market)
            # Boost confidence in LLM
            if position > 50:
                position = min(85, position + 10)
            approach = "trust-llm"
            
        else:
            # Unknown or Tit-for-Tat won (shouldn't happen often)
            # Just copy position
            position = winner_position
            approach = "copy-direct"
        
        return position, approach
    
    def _get_llm_position(self, market: 'MarketContext') -> float:
        """Get position based on LLM signals."""
        avg_signal = np.mean([
            market.aggressive_position,
            market.neutral_position,
            market.conservative_position
        ])
        
        # Scale up to meaningful positions
        # Conservative interpretation: map 0-20% signals to 40-80% positions
        position = 40 + (avg_signal * 100 * 2)  # 0.10 → 60%, 0.20 → 80%
        
        return self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
    
    def _get_winner_streak(self, game: 'GameState') -> tuple:
        """Get current winner's streak length."""
        if not game.rounds:
            return 0, None
        
        current_winner = game.last_winner
        streak = 0
        
        for r in reversed(game.rounds):
            if r.winner == current_winner:
                streak += 1
            else:
                break
        
        return streak, current_winner
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.copying = None
        self.copy_history = []
    
    def get_copy_stats(self) -> dict:
        """Get statistics about copying behavior."""
        if not self.copy_history:
            return {}
        
        from collections import Counter
        counts = Counter(self.copy_history)
        total = len(self.copy_history)
        
        return {
            "total_copies": total,
            "copy_counts": dict(counts),
            "copy_rates": {k: round(v/total, 3) for k, v in counts.items()}
        }


# === Test ===
if __name__ == "__main__":
    print("Testing TitForTatStrategy (Smart Adaptive)...")
    print("=" * 60)
    
    class MockMarket:
        aggressive_position = 0.15
        neutral_position = 0.10
        conservative_position = 0.05
        regime = "bull"
    
    class MockRound:
        def __init__(self, winner, market_return):
            self.winner = winner
            self.market_return = market_return
    
    class MockGame:
        round_num = 0
        last_positions = {}
        last_winner = None
        rounds = []
        allocations = {}
    
    strategy = TitForTatStrategy()
    market = MockMarket()
    game = MockGame()
    
    # Test 1: First round (start nice)
    print("\nTest 1: First round (start nice)")
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: Cooperator (Momentum) won with bullish position
    print("\nTest 2: Cooperator won (85%), market was up +2%")
    game.round_num = 2
    game.last_winner = "Cooperator"
    game.last_positions = {"Cooperator": 85, "Defector": 30, "Signal Follower": 60}
    game.rounds = [MockRound("Cooperator", 0.02)]
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be bullish, ~70-80%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: Defector (Mean Reversion) won
    print("\nTest 3: Defector won (35%), market was down -1.5%")
    game.last_winner = "Defector"
    game.last_positions = {"Cooperator": 80, "Defector": 35, "Signal Follower": 55}
    game.rounds = [MockRound("Defector", -0.015)]
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should expect bounce, ~65-75%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: Signal Follower won
    print("\nTest 4: Signal Follower won (65%), market +1%")
    game.last_winner = "Signal Follower"
    game.last_positions = {"Cooperator": 75, "Defector": 40, "Signal Follower": 65}
    game.rounds = [MockRound("Signal Follower", 0.01)]
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should trust LLM)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 5: Winner on a streak
    print("\nTest 5: Cooperator on 3-round streak")
    game.last_winner = "Cooperator"
    game.last_positions = {"Cooperator": 85, "Defector": 30, "Signal Follower": 60}
    game.rounds = [
        MockRound("Cooperator", 0.01),
        MockRound("Cooperator", 0.015),
        MockRound("Cooperator", 0.02)
    ]
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be very bullish, high copy weight)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Show copy stats
    print(f"\nCopy Stats: {strategy.get_copy_stats()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")