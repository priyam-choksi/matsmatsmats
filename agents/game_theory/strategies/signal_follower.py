"""
signal_follower.py - LLM Signal Follower Strategy

Location: agents/game_theory/strategies/signal_follower.py

Follows the LLM agent signals without considering what other strategies do.
This tests whether the LLM analysis adds value over passive investing.

Philosophy:
    "Trust the AI analysis. It knows something the market doesn't."

Signal Logic:
    - If any agent is bullish (>10%), follow the strongest signal
    - If all agents are cautious, take small position based on average
    - Scale up signals to create meaningful positions

Game Theory Role:
    - Does NOT participate in cooperation/defection
    - Pure test of LLM signal quality
    - Ignores other strategies completely

Usage:
    from game_theory.strategies.signal_follower import SignalFollowerStrategy
    
    strategy = SignalFollowerStrategy()
    position = strategy.decide_position(market, game)
"""

from typing import TYPE_CHECKING
import numpy as np
from .base import Strategy

if TYPE_CHECKING:
    from ..market_context import MarketContext
    from ..game_state import GameState


class SignalFollowerStrategy(Strategy):
    """
    Signal Follower - Trust the LLM Analysis
    
    Uses the three LLM agent signals to determine position:
        - Aggressive agent: Risk-seeking perspective
        - Neutral agent: Balanced perspective  
        - Conservative agent: Risk-averse perspective
    
    Position Logic:
        1. If strongest signal > 10%, use it (scaled up)
        2. Otherwise, use average signal (scaled up)
        3. Scale factor: 3x (so 10% signal → 30% position)
        4. Cap at 80% to leave some room for risk management
    
    In capital allocation game:
        - Performance depends entirely on LLM signal quality
        - Doesn't react to other strategies
        - Win/lose based on market prediction accuracy
    """
    
    name = "Signal Follower"
    description = "Follows LLM agent signals - tests if AI analysis adds value"
    
    # Configuration
    SCALE_FACTOR = 3.0      # Multiply signals by this
    MAX_POSITION = 80.0     # Maximum position
    MIN_POSITION = 5.0      # Minimum position (always have some skin in game)
    BULLISH_THRESHOLD = 0.10  # Signal > this is "bullish"
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position based on LLM signals.
        
        Ignores game state (what others are doing).
        Pure signal-following strategy.
        """
        signals = self._get_llm_signals(market)
        consensus = self._get_llm_consensus(market)
        
        # Get individual signals
        agg = market.aggressive_position
        neu = market.neutral_position
        con = market.conservative_position
        
        # Determine base position
        max_signal = max(agg, neu, con)
        avg_signal = np.mean([agg, neu, con])
        
        if max_signal > self.BULLISH_THRESHOLD:
            # At least one agent is bullish - follow strongest
            base_position = max_signal * 100 * self.SCALE_FACTOR
            signal_type = "max"
        else:
            # All agents cautious - follow average
            base_position = avg_signal * 100 * self.SCALE_FACTOR
            signal_type = "avg"
        
        # Apply consensus adjustment
        # High consensus → more confident → larger position
        # Low consensus → uncertain → smaller position
        consensus_mult = 0.7 + (consensus * 0.6)  # Range: 0.7 to 1.3
        position = base_position * consensus_mult
        
        # Clamp to valid range
        position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"Signal Follower: "
            f"agg={agg*100:.1f}%, neu={neu*100:.1f}%, con={con*100:.1f}% | "
            f"Using {signal_type}={max_signal*100 if signal_type=='max' else avg_signal*100:.1f}% | "
            f"Consensus={consensus:.0%} (mult={consensus_mult:.2f}) | "
            f"Base={base_position:.1f}% → Final={position:.1f}%"
        )
        
        self.record_decision(position, reasoning)
        return position


# === Test ===
if __name__ == "__main__":
    print("Testing SignalFollowerStrategy...")
    print("=" * 60)
    
    class MockMarket:
        aggressive_position = 0.15
        neutral_position = 0.10
        conservative_position = 0.05
        regime = "bull"
    
    class MockGame:
        round_num = 5
        last_positions = {"Cooperator": 60, "Defector": 30}
        allocations = {}
    
    strategy = SignalFollowerStrategy()
    
    # Test 1: Bullish scenario
    print("\nTest 1: Bullish signals (15%, 10%, 5%)")
    market = MockMarket()
    game = MockGame()
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: Cautious scenario
    print("\nTest 2: Cautious signals (5%, 3%, 2%)")
    market.aggressive_position = 0.05
    market.neutral_position = 0.03
    market.conservative_position = 0.02
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: High consensus bullish
    print("\nTest 3: High consensus bullish (20%, 18%, 16%)")
    market.aggressive_position = 0.20
    market.neutral_position = 0.18
    market.conservative_position = 0.16
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: Low consensus (disagreement)
    print("\nTest 4: Low consensus (25%, 10%, 0%)")
    market.aggressive_position = 0.25
    market.neutral_position = 0.10
    market.conservative_position = 0.00
    
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")