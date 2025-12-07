# """
# defector.py - Defector Strategy (Go Against the Pack)

# Location: agents/game_theory/strategies/defector.py

# The Defector takes positions OPPOSITE to what other strategies are doing.
# This is actual defection in game theory terms - breaking from the group.

# Philosophy:
#     "The crowd is usually wrong at extremes. Be contrarian."

# Game Theory Role:
#     - DEFECTS by taking opposite positions to others
#     - High variance strategy (big wins or big losses)
#     - Can steal capital when right, loses capital when wrong
#     - High risk, high reward

# Behavior:
#     - Round 1: Use LLM signal (no history yet)
#     - Round 2+: Go opposite of group average
#     - If group > 60%: Go bearish (low position)
#     - If group < 40%: Go bullish (high position)
#     - If group neutral: Use LLM signal

# When It Wins:
#     - At market turning points
#     - When the group is wrong
#     - In mean-reverting markets

# When It Loses:
#     - In trending markets (fighting the trend)
#     - When crowd is actually right
#     - In momentum-driven markets

# Usage:
#     from game_theory.strategies.defector import DefectorStrategy
    
#     strategy = DefectorStrategy()
#     position = strategy.decide_position(market, game)
# """

# from typing import TYPE_CHECKING
# import numpy as np
# from .base import Strategy

# if TYPE_CHECKING:
#     from ..market_context import MarketContext
#     from ..game_state import GameState


# class DefectorStrategy(Strategy):
#     """
#     Defector - Go Against the Pack
    
#     Takes opposite positions to other strategies, betting on reversals.
    
#     Position Logic:
#         1. First round: Use LLM signal
#         2. Later rounds: Opposite of group average
#             - Group bullish (>60%): Go bearish (20-40%)
#             - Group bearish (<40%): Go bullish (60-80%)
#             - Group neutral: Use LLM signal
    
#     In capital allocation game:
#         - High variance strategy
#         - Can steal significant capital when right
#         - Can lose significant capital when wrong
#         - Works best at regime turning points
#     """
    
#     name = "Defector"
#     description = "Goes against the pack - contrarian positions"
    
#     # Configuration
#     BULLISH_THRESHOLD = 60.0   # Group above this = they're bullish
#     BEARISH_THRESHOLD = 40.0   # Group below this = they're bearish
    
#     # Position ranges for contrarian plays
#     CONTRARIAN_BEARISH_MIN = 15.0   # When going against bullish crowd
#     CONTRARIAN_BEARISH_MAX = 35.0
#     CONTRARIAN_BULLISH_MIN = 65.0   # When going against bearish crowd
#     CONTRARIAN_BULLISH_MAX = 85.0
    
#     SCALE_FACTOR = 3.0
    
#     def decide_position(
#         self,
#         market: 'MarketContext',
#         game: 'GameState'
#     ) -> float:
#         """
#         Decide position by going against the group.
        
#         Contrarian strategy - opposite of what others are doing.
#         """
#         signals = self._get_llm_signals(market)
        
#         # First round: use LLM signal
#         if game.round_num <= 1 or not game.last_positions:
#             position = self._calculate_llm_position(market)
#             reasoning = f"Defector: Round 1, using LLM signal → {position:.1f}%"
#             self.record_decision(position, reasoning)
#             return position
        
#         # Get group average (excluding self)
#         group_avg = self._get_group_position(game)
        
#         # Decide contrarian position
#         if group_avg > self.BULLISH_THRESHOLD:
#             # Group is bullish → go bearish
#             # The more bullish they are, the more bearish we go
#             excess = group_avg - self.BULLISH_THRESHOLD
#             scale = excess / (100 - self.BULLISH_THRESHOLD)  # 0 to 1
#             position = self.CONTRARIAN_BEARISH_MAX - (scale * (self.CONTRARIAN_BEARISH_MAX - self.CONTRARIAN_BEARISH_MIN))
#             action = f"DEFECT: Group bullish ({group_avg:.1f}%) → go bearish"
            
#         elif group_avg < self.BEARISH_THRESHOLD:
#             # Group is bearish → go bullish
#             excess = self.BEARISH_THRESHOLD - group_avg
#             scale = excess / self.BEARISH_THRESHOLD  # 0 to 1
#             position = self.CONTRARIAN_BULLISH_MIN + (scale * (self.CONTRARIAN_BULLISH_MAX - self.CONTRARIAN_BULLISH_MIN))
#             action = f"DEFECT: Group bearish ({group_avg:.1f}%) → go bullish"
            
#         else:
#             # Group is neutral → use LLM signal
#             position = self._calculate_llm_position(market)
#             action = f"NEUTRAL: Group neutral ({group_avg:.1f}%) → use LLM"
        
#         # Adjust based on LLM consensus
#         # If LLM agrees with our contrarian view, be more aggressive
#         llm_avg = signals['avg']
#         if group_avg > self.BULLISH_THRESHOLD and llm_avg < 30:
#             # We're going bearish AND LLM is bearish → more conviction
#             position *= 0.8  # Even lower position
#             llm_adj = "LLM agrees (bearish) → more conviction"
#         elif group_avg < self.BEARISH_THRESHOLD and llm_avg > 40:
#             # We're going bullish AND LLM is bullish → more conviction
#             position = min(90, position * 1.2)
#             llm_adj = "LLM agrees (bullish) → more conviction"
#         else:
#             llm_adj = "no LLM adjustment"
        
#         # Clamp position
#         position = self._clamp_position(position, 10.0, 90.0)
        
#         # Build reasoning
#         reasoning = (
#             f"Defector: {action} | "
#             f"LLM avg={llm_avg:.1f}% ({llm_adj}) | "
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
#         return self._clamp_position(position, 10.0, 80.0)
    
#     def _get_group_position(self, game: 'GameState') -> float:
#         """Get average position of OTHER strategies."""
#         if not game.last_positions:
#             return 50.0
#         others = [v for k, v in game.last_positions.items() if k != self.name]
#         return np.mean(others) if others else 50.0


# # === Test ===
# if __name__ == "__main__":
#     print("Testing DefectorStrategy...")
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
    
#     strategy = DefectorStrategy()
#     market = MockMarket()
#     game = MockGame()
    
#     # Test 1: First round
#     print("\nTest 1: First round (no history)")
#     position = strategy.decide_position(market, game)
#     print(f"  Position: {position:.1f}%")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 2: Group is bullish
#     print("\nTest 2: Group is bullish (avg 75%)")
#     game.round_num = 5
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Cooperator": 70,
#         "Tit-for-Tat": 65,
#         "Signal Follower": 60,
#         "Defector": 30
#     }
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}% (should be bearish)")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 3: Group is bearish
#     print("\nTest 3: Group is bearish (avg 25%)")
#     game.last_positions = {
#         "Buy-and-Hold": 100,  # Still 100 but only one
#         "Cooperator": 20,
#         "Tit-for-Tat": 15,
#         "Signal Follower": 10,
#         "Defector": 70
#     }
    
#     # LLM is also bearish
#     market.aggressive_position = 0.05
#     market.neutral_position = 0.03
#     market.conservative_position = 0.02
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}% (should be bullish)")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 4: Group is neutral
#     print("\nTest 4: Group is neutral (avg 50%)")
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Cooperator": 45,
#         "Tit-for-Tat": 50,
#         "Signal Follower": 40,
#         "Defector": 55
#     }
    
#     market.aggressive_position = 0.15
#     market.neutral_position = 0.10
#     market.conservative_position = 0.05
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}% (should follow LLM)")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     # Test 5: Extreme bullish group
#     print("\nTest 5: Extreme bullish group (avg 90%)")
#     game.last_positions = {
#         "Buy-and-Hold": 100,
#         "Cooperator": 85,
#         "Tit-for-Tat": 90,
#         "Signal Follower": 80,
#         "Defector": 20
#     }
    
#     position = strategy.decide_position(market, game)
#     print(f"  Group avg (excl self): {strategy._get_group_position(game):.1f}%")
#     print(f"  Position: {position:.1f}% (should be very bearish)")
#     print(f"  Reasoning: {strategy.get_reasoning()}")
    
#     print("\n" + "=" * 60)
#     print("TEST COMPLETE")


"""
defector.py - Defector Strategy (MEAN REVERSION)

Location: agents/game_theory/strategies/defector.py

UPDATED: Now implements MEAN REVERSION trading logic while keeping game theory name.

Game Theory Concept: "Defect from the crowd when they're extreme"
Trading Logic: MEAN REVERSION - fade big moves, buy dips, sell rips

Philosophy:
    "What goes up must come down. Extremes revert to the mean."

When It Wins:
    - At market turning points
    - In choppy, range-bound markets
    - After overextended moves

When It Loses:
    - In trending markets (fighting the trend)
    - When momentum persists
    - In breakout scenarios

Usage:
    from game_theory.strategies.defector import DefectorStrategy
    
    strategy = DefectorStrategy()
    position = strategy.decide_position(market, game)
"""

from typing import TYPE_CHECKING
import numpy as np
from .base import Strategy

if TYPE_CHECKING:
    from ..market_context import MarketContext
    from ..game_state import GameState


class DefectorStrategy(Strategy):
    """
    Defector - MEAN REVERSION Strategy
    
    Game Theory: Defects from crowd when market has moved too far
    Trading Logic: Fade extremes - buy after drops, sell after rips
    
    Position Logic:
        1. Look at last round's market return
        2. If big up move → expect pullback → reduce position
        3. If big down move → expect bounce → increase position
        4. If small move → neutral position
    
    In capital allocation game:
        - Wins at reversals
        - Loses in trends
        - High variance strategy (contrarian)
    """
    
    name = "Defector"
    description = "MEAN REVERSION: Fades extremes, defects from momentum"
    
    # Configuration
    MIN_POSITION = 15.0    # Minimum position
    MAX_POSITION = 90.0    # Maximum position
    SCALE_FACTOR = 3.5     # Scale up LLM signals for first rounds
    
    # Mean reversion thresholds (as decimals)
    BIG_UP = 0.02          # Big up move (>2%)
    MODERATE_UP = 0.01     # Moderate up move (1-2%)
    MODERATE_DOWN = -0.01  # Moderate down move (-1% to -2%)
    BIG_DOWN = -0.02       # Big down move (<-2%)
    
    # Also look at 2-day cumulative for stronger signals
    LOOKBACK = 2
    CUMULATIVE_EXTREME = 0.03  # 3% move over 2 days is extreme
    
    def decide_position(
        self,
        market: 'MarketContext',
        game: 'GameState'
    ) -> float:
        """
        Decide position by fading extreme moves.
        
        Contrarian strategy - goes against recent momentum.
        """
        # First round: no history, use neutral position
        if len(game.rounds) < 1:
            position = 50.0  # Start neutral
            reasoning = "MEAN REVERSION (warmup): No history, starting neutral at 50%"
            self.record_decision(position, reasoning)
            return position
        
        # Get last round's return
        last_return = game.rounds[-1].market_return
        
        # Calculate cumulative return if we have enough history
        if len(game.rounds) >= self.LOOKBACK:
            recent_returns = [r.market_return for r in game.rounds[-self.LOOKBACK:]]
            cumulative = sum(recent_returns)
        else:
            cumulative = last_return
        
        # Mean reversion logic - FADE the move
        if last_return > self.BIG_UP:
            # Big up move → expect pullback → go light
            position = 25.0
            action = "FADE BIG UP"
            
            # Even lighter if cumulative is extreme
            if cumulative > self.CUMULATIVE_EXTREME:
                position = 15.0
                action = "FADE EXTREME UP"
                
        elif last_return > self.MODERATE_UP:
            # Moderate up → slight fade
            position = 40.0
            action = "FADE MODERATE UP"
            
        elif last_return < self.BIG_DOWN:
            # Big down move → expect bounce → go heavy
            position = 85.0
            action = "BUY THE DIP"
            
            # Even heavier if cumulative is extreme
            if cumulative < -self.CUMULATIVE_EXTREME:
                position = 90.0
                action = "BUY EXTREME DIP"
                
        elif last_return < self.MODERATE_DOWN:
            # Moderate down → slight buy
            position = 70.0
            action = "MILD DIP BUY"
            
        else:
            # Small move → no clear signal → slightly bullish bias
            position = 55.0
            action = "NEUTRAL (slight long bias)"
        
        # Clamp to valid range
        position = self._clamp_position(position, self.MIN_POSITION, self.MAX_POSITION)
        
        # Build reasoning
        reasoning = (
            f"MEAN REVERSION: {action} | "
            f"Last: {last_return*100:+.2f}% | "
            f"Cumul({self.LOOKBACK}d): {cumulative*100:+.2f}% | "
            f"Position: {position:.1f}%"
        )
        
        self.record_decision(position, reasoning)
        return position


# === Test ===
if __name__ == "__main__":
    print("Testing DefectorStrategy (MEAN REVERSION)...")
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
    
    strategy = DefectorStrategy()
    market = MockMarket()
    game = MockGame()
    
    # Test 1: First round (no history)
    print("\nTest 1: First round (start neutral)")
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be 50%)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 2: Big up move - should FADE
    print("\nTest 2: After big up move (+3%)")
    game.rounds = [MockRound(0.03)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~25% - fading)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 3: Extreme up - multiple days
    print("\nTest 3: Extreme up (2 days of +2%)")
    game.rounds = [MockRound(0.02), MockRound(0.02)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~15% - extreme fade)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 4: Big down move - should BUY
    print("\nTest 4: After big down move (-3%)")
    game.rounds = [MockRound(-0.03)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~85% - buying dip)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 5: Extreme down - multiple days
    print("\nTest 5: Extreme down (2 days of -2%)")
    game.rounds = [MockRound(-0.02), MockRound(-0.02)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be 90% - extreme buy)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    # Test 6: Small move - neutral
    print("\nTest 6: Small move (+0.5%)")
    game.rounds = [MockRound(0.005)]
    position = strategy.decide_position(market, game)
    print(f"  Position: {position:.1f}% (should be ~55% - neutral)")
    print(f"  Reasoning: {strategy.get_reasoning()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")