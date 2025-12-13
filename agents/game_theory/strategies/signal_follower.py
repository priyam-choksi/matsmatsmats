# """
# signal_follower.py - LLM Signal Follower Strategy

# Location: agents/game_theory/strategies/signal_follower.py

# Follows the LLM agent signals without considering what other strategies do.
# This tests whether the LLM analysis adds value over passive investing.

# Philosophy:
#     "Trust the AI analysis. It knows something the market doesn't."

# Signal Logic:
#     - If any agent is bullish (>10%), follow the strongest signal
#     - If all agents are cautious, take small position based on average
#     - Scale up signals to create meaningful positions

# Game Theory Role:
#     - Does NOT participate in cooperation/defection
#     - Pure test of LLM signal quality
#     - Ignores other strategies completely

# Usage:
#     from game_theory.strategies.signal_follower import SignalFollowerStrategy
    
#     strategy = SignalFollowerStrategy()
#     position = strategy.decide_position(market, game)
# """

# from typing import TYPE_CHECKING
# import numpy as np
# from .base import Strategy

# if TYPE_CHECKING:
#     from ..market_context import MarketContext
#     from ..game_state import GameState


# class SignalFollowerStrategy(Strategy):
#     """
#     Signal Follower - Trust the LLM Analysis
    
#     Uses the three LLM agent signals to determine position:
#         - Aggressive agent: Risk-seeking perspective
#         - Neutral agent: Balanced perspective  
#         - Conservative agent: Risk-averse perspective
    
#     Position Logic:
#         1. If strongest signal > 10%, use it (scaled up)
#         2. Otherwise, use average signal (scaled up)
#         3. Scale factor: 3x (so 10% signal → 30% position)
#         4. Cap at 80% to leave some room for risk management
    
#     In capital allocation game:
#         - Performance depends entirely on LLM signal quality
#         - Doesn't react to other strategies
#         - Win/lose based on market prediction accuracy
#     """
    
#     name = "Signal Follower"
#     description = "Follows LLM agent signals - tests if AI analysis adds value"
    
#     # Configuration
#     SCALE_FACTOR = 3.0      # Multiply signals by this
#     MAX_POSITION = 80.0     # Maximum position
#     MIN_POSITION = 5.0      # Minimum position (always have some skin in game)
#     BULLISH_THRESHOLD = 0.10  # Signal > this is "bullish"
    
#     def decide_position(
#         self,
#         market: 'MarketContext',
#         game: 'GameState'
#     ) -> float:
#         """
#         Decide position based on LLM signals.
        
#         Ignores game state (what others are doing).
#         Pure signal-following strategy.
#         """
#         signals = self._get_llm_signals(market)
#         consensus = self._get_llm_consensus(market)
        
#         # Get individual signals
#         agg = market.aggressive_position
#         neu = market.neutral_position
#         con = market.conservative_position
        
#         # Determine base position
#         max_signal = max(agg, neu, con)
#         avg_signal = np.mean([agg, neu, con])
        
#         if max_signal > self.BULLISH_THRESHOLD:
#             # At least one agent is bullish - follow strongest
#             base_position = max_signal * 100 * self.SCALE_FACTOR
#             signal_type = "max"
#         else:
#             # All agents cautious - follow average
#             base_position = avg_signal * 100 * self.SCALE_FACTOR
#             signal_type = "avg"
        
#         # Apply consensus adjustment
#         # High consensus → more confident → larger position
#         # Low consensus → uncertain → smaller position
#         consensus_mult = 0.7 + (consensus * 0.6)  # Range: 0.7 to 1.3
#         position = base_position * consensus_mult
        
#         # Clamp to valid range
#         position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
#         # Build reasoning
#         reasoning = (
#             f"Signal Follower: "
#             f"agg={agg*100:.1f}%, neu={neu*100:.1f}%, con={con*100:.1f}% | "
#             f"Using {signal_type}={max_signal*100 if signal_type=='max' else avg_signal*100:.1f}% | "
#             f"Consensus={consensus:.0%} (mult={consensus_mult:.2f}) | "
#             f"Base={base_position:.1f}% → Final={position:.1f}%"
#         )
        
#         self.record_decision(position, reasoning)
#         return position


# # === Test ===
# if __name__ == "__main__":
#     print("Testing SignalFollowerStrategy...")
#     print("=" * 60)
    
#     class MockMarket:
#         aggressive_position = 0.15
#         neutral_position = 0.10
#         conservative_position = 0.05
#         regime = "bull"
    
#     class MockGame:
#         round_num = 5
#         last_positions = {"Cooperator": 60, "Defector": 30}
#         allocations = {}
    
#     strategy = SignalFollowerStrategy()
    
#     # Test 1: Bullish scenario
#     print("\nTest 1: Bullish signals (15%, 10%, 5%)")
#     market = MockMarket()
#     game = MockGame()
    
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 2: Cautious scenario
#     print("\nTest 2: Cautious signals (5%, 3%, 2%)")
#     market.aggressive_position = 0.05
#     market.neutral_position = 0.03
#     market.conservative_position = 0.02
    
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 3: High consensus bullish
#     print("\nTest 3: High consensus bullish (20%, 18%, 16%)")
#     market.aggressive_position = 0.20
#     market.neutral_position = 0.18
#     market.conservative_position = 0.16
    
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 4: Low consensus (disagreement)
#     print("\nTest 4: Low consensus (25%, 10%, 0%)")
#     market.aggressive_position = 0.25
#     market.neutral_position = 0.10
#     market.conservative_position = 0.00
    
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     print("\n" + "=" * 60)
#     print("TEST COMPLETE")


"""
signal_follower.py - LLM Signal Follower Strategy

Location: agents/game_theory/strategies/signal_follower.py

UPDATED: Now takes LARGER positions based on LLM signals to compete fairly.

Follows the LLM agent signals as a portfolio manager (not risk advisor).
Interprets LLM recommendations and scales them to real trading positions.

Philosophy:
    "Trust the AI analysis. It knows something the market doesn't."

Signal Interpretation (NEW):
    - Conservative signal → 40-50% position (cautious but still invested)
    - Neutral signal → 55-70% position (moderate conviction)
    - Aggressive signal → 75-90% position (high conviction)
    
    This treats LLM signals as GUIDANCE, not literal position sizes.
    A "10% position recommendation" from a risk advisor becomes 
    "low conviction" for a portfolio manager → 40-50% position.

When It Wins:
    - When LLM analysis correctly predicts direction
    - When agent consensus is high and correct
    - In markets where fundamentals drive prices

When It Loses:
    - When LLM is wrong about direction
    - When markets are driven by sentiment/momentum
    - In highly volatile, unpredictable conditions

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
    
    Interprets LLM agent signals as a portfolio manager would:
    - Takes the signal direction seriously
    - Scales up to meaningful position sizes
    - Uses consensus to gauge conviction
    
    Position Logic (NEW - Portfolio Manager Interpretation):
        1. Calculate average LLM signal (0-1 scale where 0.10 = 10%)
        2. Map signal to position tiers:
           - Signal < 5% → Low conviction → 40-50% position
           - Signal 5-12% → Medium conviction → 55-70% position
           - Signal 12-20% → High conviction → 75-85% position
           - Signal > 20% → Very high conviction → 85-90% position
        3. Adjust based on consensus (high agreement = higher position)
    """
    
    name = "Signal Follower"
    description = "Follows LLM signals as portfolio manager - scaled positions"
    
    # Configuration - Position tiers based on signal interpretation
    MIN_POSITION = 35.0     # Even low conviction gets meaningful position
    MAX_POSITION = 90.0     # Cap at 90% (some risk management)
    
    # Signal thresholds (as decimals from LLM, e.g., 0.10 = 10% recommendation)
    LOW_SIGNAL = 0.05       # Below this = low conviction
    MEDIUM_SIGNAL = 0.12    # Below this = medium conviction
    HIGH_SIGNAL = 0.20      # Below this = high conviction
    # Above HIGH_SIGNAL = very high conviction
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position based on LLM signals.
        
        Interprets signals as a portfolio manager, not literal positions.
        """
        # Get the three LLM signals
        signals = {
            'aggressive': market.aggressive_position,
            'neutral': market.neutral_position,
            'conservative': market.conservative_position
        }
        
        # Average signal and strongest signal
        avg_signal = np.mean(list(signals.values()))
        max_signal = max(signals.values())
        min_signal = min(signals.values())
        
        # Calculate consensus (how much agents agree)
        signal_std = np.std(list(signals.values()))
        consensus = 1.0 - min(1.0, signal_std / 0.10)  # 0-1, higher = more agreement
        
        # Determine conviction tier based on average signal
        if avg_signal < self.LOW_SIGNAL:
            # Low conviction - but still take a position
            base_position = 45.0
            conviction = "LOW"
        elif avg_signal < self.MEDIUM_SIGNAL:
            # Medium conviction - moderate position
            base_position = 62.0
            conviction = "MEDIUM"
        elif avg_signal < self.HIGH_SIGNAL:
            # High conviction - substantial position
            base_position = 78.0
            conviction = "HIGH"
        else:
            # Very high conviction - aggressive position
            base_position = 88.0
            conviction = "VERY HIGH"
        
        # Adjust based on consensus
        # High consensus → increase position (more confident)
        # Low consensus → decrease position (agents disagree)
        consensus_adjustment = (consensus - 0.5) * 15  # -7.5 to +7.5
        position = base_position + consensus_adjustment
        
        # If strongest signal is much higher than average, slight boost
        if max_signal > avg_signal * 1.5:
            position += 5.0
        
        # Clamp to valid range
        position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"SIGNAL FOLLOWER: {conviction} conviction | "
            f"Signals: Agg={signals['aggressive']*100:.1f}%, "
            f"Neu={signals['neutral']*100:.1f}%, "
            f"Con={signals['conservative']*100:.1f}% | "
            f"Avg={avg_signal*100:.1f}% | "
            f"Consensus={consensus:.0%} | "
            f"Position: {position:.1f}%"
        )
        
        self.record_decision(position, reasoning)
        return position


# === Test ===
if __name__ == "__main__":
    print("Testing SignalFollowerStrategy (Portfolio Manager)...")
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
    
    strategy = SignalFollowerStrategy()
    market = MockMarket()
    game = MockGame()
    
    # Test 1: Medium signals (typical)
    print("\nTest 1: Medium signals (Agg=15%, Neu=10%, Con=5%)")
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~60-70%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: Low signals (conservative LLM)
    print("\nTest 2: Low signals (Agg=5%, Neu=3%, Con=2%)")
    market.aggressive_position = 0.05
    market.neutral_position = 0.03
    market.conservative_position = 0.02
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~40-50%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: High signals (bullish LLM)
    print("\nTest 3: High signals (Agg=25%, Neu=18%, Con=12%)")
    market.aggressive_position = 0.25
    market.neutral_position = 0.18
    market.conservative_position = 0.12
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~80-90%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: High consensus (agents agree)
    print("\nTest 4: High consensus (all at 12%)")
    market.aggressive_position = 0.12
    market.neutral_position = 0.12
    market.conservative_position = 0.11
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be higher due to consensus)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 5: Low consensus (agents disagree)
    print("\nTest 5: Low consensus (Agg=25%, Neu=10%, Con=3%)")
    market.aggressive_position = 0.25
    market.neutral_position = 0.10
    market.conservative_position = 0.03
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be lower due to disagreement)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")