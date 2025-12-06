"""
buy_hold.py - Buy and Hold Benchmark Strategy

Location: agents/game_theory/strategies/buy_hold.py

The simplest strategy: always 100% invested, no matter what.

This is the BENCHMARK that all other strategies try to beat.
It ignores:
    - LLM signals
    - What other strategies do
    - Market regime
    - Everything

Philosophy:
    "Time in the market beats timing the market."

Game Theory Role:
    - Does NOT participate in cooperation/defection dynamics
    - Pure market exposure baseline
    - If you can't beat this, your strategy adds no value

Usage:
    from game_theory.strategies.buy_hold import BuyHoldStrategy
    
    strategy = BuyHoldStrategy()
    position = strategy.decide_position(market, game)  # Always 100
"""

from typing import TYPE_CHECKING
from .base import Strategy

if TYPE_CHECKING:
    from ..market_context import MarketContext
    from ..game_state import GameState


class BuyHoldStrategy(Strategy):
    """
    Buy and Hold - Always 100% Invested
    
    The benchmark strategy. Takes full market exposure every round.
    
    In capital allocation game:
        - Gets full market return every round (positive or negative)
        - In bull markets: accumulates capital (consistently outperforms cautious strategies)
        - In bear markets: loses capital (consistently underperforms)
        - High variance in allocation due to consistent behavior
    
    Key insight: This strategy DOESN'T play the game. It just invests.
    Other strategies that try to be clever must beat this baseline.
    """
    
    name = "Buy-and-Hold"
    description = "Always 100% invested - the benchmark to beat"
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Always return 100% position.
        
        Ignores all inputs - market signals, game state, everything.
        This IS the benchmark.
        """
        # Build reasoning for logging
        reasoning = (
            f"Buy-and-Hold: ALWAYS 100% invested | "
            f"Round {game.round_num} | "
            f"Regime: {market.regime} (ignored) | "
            f"LLM avg: {self._get_llm_signals(market)['avg']:.1f}% (ignored) | "
            f"Group pos: {self._get_group_position(game):.1f}% (ignored) | "
            f"THIS IS THE BENCHMARK"
        )
        
        self.record_decision(100.0, reasoning)
        return 100.0


# === Test ===
if __name__ == "__main__":
    print("Testing BuyHoldStrategy...")
    print("=" * 60)
    
    # Mock objects for testing
    class MockMarket:
        aggressive_position = 0.15
        neutral_position = 0.10
        conservative_position = 0.05
        regime = "bull"
        daily_return = 0.02
    
    class MockGame:
        round_num = 5
        last_positions = {"Cooperator": 60, "Defector": 30}
        allocations = {"Buy-and-Hold": 250000}
    
    strategy = BuyHoldStrategy()
    market = MockMarket()
    game = MockGame()
    
    position = strategy.decide_position(market, game)
    
    print(f"Strategy: {strategy.name}")
    print(f"Position: {position}%")
    print(f"Reasoning: {strategy.get_reasoning()}")
    
    assert position == 100.0, "Buy-and-Hold should always be 100%"
    print("\n✓ Test passed: Always returns 100%")
    
    # Test across multiple scenarios
    test_cases = [
        {"regime": "bull", "llm": 0.20},
        {"regime": "bear", "llm": 0.02},
        {"regime": "sideways", "llm": 0.10},
    ]
    
    print("\nTesting across scenarios:")
    for tc in test_cases:
        market.regime = tc["regime"]
        market.aggressive_position = tc["llm"]
        pos = strategy.decide_position(market, game)
        print(f"  {tc['regime']:10} regime, LLM={tc['llm']*100:.0f}% → Position: {pos}%")
        assert pos == 100.0
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")