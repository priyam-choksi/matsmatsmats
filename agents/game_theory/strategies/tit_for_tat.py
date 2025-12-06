"""
tit_for_tat.py - Tit-for-Tat Adaptive Strategy

Location: agents/game_theory/strategies/tit_for_tat.py

The Tit-for-Tat strategy copies what worked last round.
This is the famous strategy that won Axelrod's tournament.

Philosophy:
    "Do what worked. Adapt to the environment."

Game Theory Role:
    - Starts by COOPERATING (nice)
    - Copies the WINNER's approach each round
    - Adapts quickly to what's working
    - Balances between cooperation and defection

Original Tit-for-Tat Properties:
    1. Nice: Starts by cooperating
    2. Retaliatory: Copies defection if opponent defected
    3. Forgiving: Goes back to cooperating if opponent cooperates
    4. Clear: Easy to understand, predictable

Our Adaptation:
    - "Cooperate" = Follow the winner
    - "Defect" = Go against the winner
    - We always copy the winner, making us adaptive

When It Wins:
    - In changing regimes (quickly adapts)
    - When there's a clear winning strategy
    - In predictable markets

When It Loses:
    - When winners alternate randomly
    - When it copies right before a reversal
    - In choppy, unpredictable markets

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
    Tit-for-Tat - Copy What Worked
    
    Adaptive strategy that copies the winning approach each round.
    
    Position Logic:
        1. First round: Cooperate (follow group/LLM blend)
        2. Later rounds: Copy what the winner did
            - Winner was bullish? Go bullish
            - Winner was bearish? Go bearish
        3. Blend: 80% copy winner, 20% own judgment
    
    In capital allocation game:
        - Medium variance strategy
        - Adapts to winning approaches
        - Should accumulate capital in trending regimes
        - May lag at turning points (copies OLD winner)
    """
    
    name = "Tit-for-Tat"
    description = "Copies what worked last round - adaptive learner"
    
    # Configuration
    COPY_WEIGHT = 0.80      # How much to copy winner
    OWN_WEIGHT = 0.20       # How much own judgment
    MIN_POSITION = 15.0     # Minimum position
    MAX_POSITION = 85.0     # Maximum position
    SCALE_FACTOR = 3.0
    
    def __init__(self):
        super().__init__()
        # Track who we're copying
        self.copying = None
        self.copy_history = []
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position by copying the last winner.
        
        Adapts to what's working in the current environment.
        """
        signals = self._get_llm_signals(market)
        
        # Calculate our own judgment (LLM-based)
        own_position = self._calculate_llm_position(market)
        
        # First round: cooperate (follow group average / LLM blend)
        if game.round_num <= 1 or not game.last_winner:
            position = own_position
            self.copying = None
            reasoning = f"Tit-for-Tat: Round 1, starting nice → {position:.1f}%"
            self.record_decision(position, reasoning)
            return position
        
        # Get winner's last position
        winner = game.last_winner
        winner_position = game.last_positions.get(winner, 50.0)
        
        # Track who we're copying
        self.copying = winner
        self.copy_history.append(winner)
        
        # Blend: copy winner + own judgment
        position = (winner_position * self.COPY_WEIGHT) + (own_position * self.OWN_WEIGHT)
        
        # Check winner's streak
        winner_streak = self._get_winner_streak(game)
        
        # If winner has long streak, follow more strongly
        if winner_streak >= 3:
            # Strong streak - follow more closely
            position = (winner_position * 0.9) + (own_position * 0.1)
            streak_adj = f"streak={winner_streak} → 90% copy"
        elif winner_streak == 1:
            # Just started winning - be cautious
            position = (winner_position * 0.7) + (own_position * 0.3)
            streak_adj = f"new winner → 70% copy"
        else:
            streak_adj = "normal copy"
        
        # Clamp position
        position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"Tit-for-Tat: Copying '{winner}' ({winner_position:.1f}%) | "
            f"{streak_adj} | "
            f"Own={own_position:.1f}% | "
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
        position = avg_signal * 100 * self.SCALE_FACTOR
        return self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
    
    def _get_winner_streak(self, game: 'GameState') -> int:
        """Get current winner's streak length."""
        if not game.rounds:
            return 0
        
        current_winner = game.last_winner
        streak = 0
        
        for r in reversed(game.rounds):
            if r.winner == current_winner:
                streak += 1
            else:
                break
        
        return streak
    
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
            "copy_rates": {k: v/total for k, v in counts.items()}
        }


# === Test ===
if __name__ == "__main__":
    print("Testing TitForTatStrategy...")
    print("=" * 60)
    
    class MockMarket:
        aggressive_position = 0.15
        neutral_position = 0.10
        conservative_position = 0.05
        regime = "bull"
    
    class MockRound:
        def __init__(self, winner):
            self.winner = winner
    
    class MockGame:
        round_num = 1
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
    print(f"  Copying: {strategy.copying}")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: Copy Buy-and-Hold (100%)
    print("\nTest 2: Copy Buy-and-Hold winner (100%)")
    game.round_num = 5
    game.last_winner = "Buy-and-Hold"
    game.last_positions = {
        "Buy-and-Hold": 100,
        "Cooperator": 60,
        "Defector": 30,
        "Signal Follower": 50,
        "Tit-for-Tat": 55
    }
    game.rounds = [MockRound("Buy-and-Hold")]
    
    position = strategy.decide_position(market, game)
    print(f"  Winner: {game.last_winner} (pos={game.last_positions['Buy-and-Hold']}%)")
    print(f"  Position: {position:.1f}%")
    print(f"  Copying: {strategy.copying}")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: Copy Defector (30%)
    print("\nTest 3: Copy Defector winner (30%)")
    game.last_winner = "Defector"
    game.rounds = [MockRound("Defector")]
    
    position = strategy.decide_position(market, game)
    print(f"  Winner: {game.last_winner} (pos={game.last_positions['Defector']}%)")
    print(f"  Position: {position:.1f}%")
    print(f"  Copying: {strategy.copying}")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: Long winner streak
    print("\nTest 4: Buy-and-Hold on 5-round streak")
    game.last_winner = "Buy-and-Hold"
    game.rounds = [MockRound("Buy-and-Hold") for _ in range(5)]
    
    position = strategy.decide_position(market, game)
    print(f"  Winner streak: {strategy._get_winner_streak(game)}")
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 5: New winner
    print("\nTest 5: Signal Follower just started winning")
    game.last_winner = "Signal Follower"
    game.rounds = [MockRound("Buy-and-Hold"), MockRound("Buy-and-Hold"), MockRound("Signal Follower")]
    
    position = strategy.decide_position(market, game)
    print(f"  Winner streak: {strategy._get_winner_streak(game)}")
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    print("\n" + "=" * 60)
    print("Copy stats:", strategy.get_copy_stats())
    print("TEST COMPLETE")