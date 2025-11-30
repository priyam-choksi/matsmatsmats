"""
defector.py - Contrarian Strategy

Location: agents/game_theory/strategies/defector.py

This strategy fades the crowd at extremes. When everyone agrees,
it reduces exposure (doesn't invert!), believing extreme consensus 
often marks turning points.

Philosophy:
    "The crowd is wrong at extremes. Be cautious when everyone agrees."

Score Behavior:
    Score determines HOW contrarian to be (threshold for fading)
    - High score (+5 to +10): Threshold = 70% (more willing to fade consensus)
    - Neutral (-2 to +5):     Threshold = 85% (only extreme consensus)
    - Low score (-10 to -2):  Threshold = 110% (never contrarian - market is trending)

Position Logic:
    1. Calculate consensus from position std
    2. If consensus >= threshold: FADE to small position (NOT invert!)
    3. If consensus < threshold:  Take larger position (disagreement = opportunity)
    4. Apply 5x position scaling for meaningful trades
    5. Adjust for regime and hot/cold streaks

Why This Matters:
    - Tests if fading extreme consensus adds value
    - In trending markets, contrarian = losing
    - At turning points, contrarian = winning
    - Score adapts: stops being contrarian when it's not working

ENHANCED with:
    - 5x position scaling for meaningful positions
    - Starting score bias of -1 (skeptical)
    - Regime awareness (less contrarian in strong trends)
    - Tracking of when contrarian vs follow worked
"""

from typing import List, Tuple
import numpy as np

from ..base_strategy import TradingStrategy, TradeResult
from ..market_context import MarketContext


class DefectorStrategy(TradingStrategy):
    """
    CONTRARIAN - Fade the Crowd at Extremes
    
    This strategy believes that when everyone agrees, the market
    has likely already priced in that view, and it's time to be cautious.
    
    Core Idea:
        - Extreme consensus = potential turning point = reduce position
        - Mixed signals = opportunity = take larger position
        - NOT about inverting, but about fading extremes
        - If contrarian isn't working (low score), follow the trend
    
    Contrarian Logic:
        When consensus >= threshold (extreme agreement):
            - Take small position (fade the extreme)
            - Logic: extremes often reverse
            
        When consensus < threshold (disagreement):
            - Take larger position based on average
            - Logic: uncertainty creates opportunity
    
    Score Interpretation:
        Score adjusts the threshold for going contrarian:
        
        | Score Range | Threshold | Interpretation |
        |-------------|-----------|----------------|
        | +5 to +10   | 70%       | "Contrarian working, do it more" |
        | -2 to +5    | 85%       | "Only at extreme consensus" |
        | -10 to -2   | 110%      | "Stop being contrarian, market trending" |
        
        Key insight: When score is very low, the >100% threshold
        means we NEVER go contrarian. This is the adaptation.
    
    When It Wins:
        - At market turning points
        - When consensus marks extremes (tops/bottoms)
        - Mean-reverting markets
    
    When It Loses:
        - Strong trending markets
        - When consensus is actually right
        - When it keeps fading a strong trend
    """
    
    # Position limits
    MIN_POSITION = 20.0
    MAX_POSITION = 80.0
    # Position scaling for meaningful trades
    POSITION_SCALE = 5.0
    
    def __init__(self):
        super().__init__(
            name="Defector",
            description="Contrarian - fades crowd at extremes based on score",
            position_scale=self.POSITION_SCALE,
            initial_score=-1.0  # Skeptical starting bias
        )
        
        # Track contrarian trade outcomes for analysis
        self.contrarian_trades = 0
        self.contrarian_wins = 0
        self.follow_trades = 0
        self.follow_wins = 0
        
        # Track regime-specific contrarian performance
        self.contrarian_by_regime = {
            'bull': {'trades': 0, 'wins': 0},
            'bear': {'trades': 0, 'wins': 0},
            'sideways': {'trades': 0, 'wins': 0}
        }
    
    def reset(self):
        """Reset including contrarian tracking."""
        super().reset()
        self.contrarian_trades = 0
        self.contrarian_wins = 0
        self.follow_trades = 0
        self.follow_wins = 0
        self.contrarian_by_regime = {
            'bull': {'trades': 0, 'wins': 0},
            'bear': {'trades': 0, 'wins': 0},
            'sideways': {'trades': 0, 'wins': 0}
        }
    
    def _get_contrarian_threshold(self) -> float:
        """
        Get consensus threshold for going contrarian.
        
        Higher score = lower threshold (more willing to be contrarian)
        Lower score = higher threshold (less contrarian, market is trending)
        
        Returns:
            Consensus threshold (0.0 to 1.1)
            Note: >1.0 means NEVER go contrarian
        """
        if self.score >= 5:
            # Contrarian is working - be more willing
            return 0.70
        
        elif self.score >= -2:
            # Normal - only at extreme consensus
            return 0.85
        
        else:
            # Contrarian not working - market is trending
            # Threshold > 1.0 means we never trigger contrarian
            return 1.10
    
    def _calculate_consensus(self, positions: List[float]) -> float:
        """
        Calculate consensus level from position recommendations.
        
        Args:
            positions: List of position recommendations (decimals)
            
        Returns:
            Consensus level (0.0 to 1.0)
        """
        if len(positions) < 2:
            return 1.0
        
        if len(set(positions)) == 1:
            return 1.0
        
        std_position = np.std(positions)
        consensus = max(0.0, 1.0 - std_position / 0.10)
        
        return consensus
    
    def _get_regime_adjustment(self, ctx: MarketContext, is_contrarian: bool) -> float:
        """
        Adjust contrarian behavior based on regime.
        
        In strong trends (bull/bear), reduce contrarian.
        In sideways markets, contrarian works better.
        
        Returns:
            Adjustment factor (0.7 to 1.3)
        """
        if not is_contrarian:
            return 1.0  # No adjustment for follow mode
        
        regime = ctx.regime.lower()
        
        # Check if contrarian has been working in this regime
        regime_stats = self.contrarian_by_regime.get(regime, {'trades': 0, 'wins': 0})
        if regime_stats['trades'] >= 5:
            win_rate = regime_stats['wins'] / regime_stats['trades']
            if win_rate < 0.3:
                return 0.7  # Contrarian failing in this regime
            elif win_rate > 0.7:
                return 1.3  # Contrarian working great in this regime
        
        # Default regime adjustments
        if regime == 'bull' and self.score < 0:
            return 0.8  # Reduce contrarian in bull trend when losing
        elif regime == 'bear' and self.score < 0:
            return 0.8  # Reduce contrarian in bear trend when losing
        elif regime == 'sideways':
            return 1.1  # Slightly favor contrarian in sideways
        
        return 1.0
    
    def decide_position(
        self, 
        ctx: MarketContext, 
        history: List[TradeResult]
    ) -> Tuple[float, str]:
        """
        Decide position - fade extremes, take opportunity when mixed.
        
        Args:
            ctx: MarketContext with agent evaluations
            history: Past trades
            
        Returns:
            Tuple of (position_pct, reasoning)
        """
        # Get all three position recommendations
        positions = [
            ctx.aggressive_position,
            ctx.neutral_position,
            ctx.conservative_position
        ]
        
        avg_position = np.mean(positions)
        consensus = self._calculate_consensus(positions)
        threshold = self._get_contrarian_threshold()
        
        # Check if ALL agents are strongly bullish or bearish
        all_bullish = all(p > 0.10 for p in positions)
        all_bearish = all(p < 0.03 for p in positions)
        extreme_consensus = (consensus >= threshold) and (all_bullish or all_bearish)
        
        # Decision: contrarian or opportunistic?
        if extreme_consensus:
            # HIGH CONSENSUS - Be contrarian (fade to small position)
            # NOT inverting! Just taking small position when everyone agrees
            base_position_pct = 25.0  # Fixed small position for contrarian
            
            # Apply position scaling (but reduced for contrarian)
            scaled_position = base_position_pct * (self.position_scale * 0.5)  # Only 50% of normal scaling
            
            # Apply regime adjustment
            regime_adj = self._get_regime_adjustment(ctx, True)
            position_pct = scaled_position * regime_adj
            
            # Clamp to reasonable range
            position_pct = max(self.MIN_POSITION, min(self.MAX_POSITION, position_pct))
            
            reasoning = (
                f"Defector: consensus {consensus:.0%} >= {threshold:.0%} "
                f"({'all bullish' if all_bullish else 'all bearish'}) "
                f"[score={self.score:+.0f}] -> FADE to {position_pct:.1f}%"
            )
            
            if regime_adj != 1.0:
                reasoning += f" (regime adj {regime_adj:.1f})"
            
            self.contrarian_trades += 1
            
        else:
            # MIXED SIGNALS - Take opportunity
            # When agents disagree, there's opportunity
            if consensus < 0.5:
                # Very low consensus = high opportunity
                multiplier = 2.0
                logic = "high disagreement = opportunity"
            else:
                # Moderate consensus = normal following
                multiplier = 1.5
                logic = "moderate consensus"
            
            base_position_pct = avg_position * 100.0 * multiplier
            
            # Apply position scaling
            position_pct = base_position_pct * self.position_scale
            
            # Apply hot/cold adjustment for follow mode
            if self.is_hot:
                position_pct *= 1.1
            elif self.is_cold:
                position_pct *= 0.9
            
            # Clamp to max
            position_pct = min(self.MAX_POSITION, position_pct)
            
            reasoning = (
                f"Defector: consensus {consensus:.0%} < {threshold:.0%} "
                f"({logic}) "
                f"[score={self.score:+.0f}] -> opportunistic {position_pct:.1f}%"
            )
            
            if self.is_hot:
                reasoning += " [HOT]"
            elif self.is_cold:
                reasoning += " [COLD]"
            
            self.follow_trades += 1
        
        return position_pct, reasoning
    
    def execute_trade(self, ctx: MarketContext) -> TradeResult:
        """
        Execute trade and track contrarian vs follow outcomes by regime.
        
        Overrides base to add contrarian tracking.
        """
        # Determine if this will be a contrarian trade
        positions = [ctx.aggressive_position, ctx.neutral_position, ctx.conservative_position]
        consensus = self._calculate_consensus(positions)
        threshold = self._get_contrarian_threshold()
        all_bullish = all(p > 0.10 for p in positions)
        all_bearish = all(p < 0.03 for p in positions)
        is_contrarian = (consensus >= threshold) and (all_bullish or all_bearish)
        
        # Execute the trade
        result = super().execute_trade(ctx)
        
        # Track outcome
        regime = ctx.regime.lower()
        if is_contrarian:
            if result.trade_return > 0:
                self.contrarian_wins += 1
                self.contrarian_by_regime[regime]['wins'] += 1
            self.contrarian_by_regime[regime]['trades'] += 1
        else:
            if result.trade_return > 0:
                self.follow_wins += 1
        
        # Store in strategy memory
        self.strategy_memory['last_was_contrarian'] = is_contrarian
        self.strategy_memory['contrarian_success_rate'] = (
            self.contrarian_wins / self.contrarian_trades
            if self.contrarian_trades > 0 else 0.5
        )
        
        return result
    
    def get_contrarian_stats(self) -> dict:
        """Get statistics about contrarian vs follow trades."""
        stats = {
            "contrarian_trades": self.contrarian_trades,
            "contrarian_wins": self.contrarian_wins,
            "contrarian_win_rate": (
                self.contrarian_wins / self.contrarian_trades * 100 
                if self.contrarian_trades > 0 else 0
            ),
            "follow_trades": self.follow_trades,
            "follow_wins": self.follow_wins,
            "follow_win_rate": (
                self.follow_wins / self.follow_trades * 100 
                if self.follow_trades > 0 else 0
            ),
        }
        
        # Add regime-specific stats
        for regime, data in self.contrarian_by_regime.items():
            if data['trades'] > 0:
                stats[f"contrarian_{regime}_win_rate"] = data['wins'] / data['trades'] * 100
                stats[f"contrarian_{regime}_trades"] = data['trades']
        
        return stats


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing DefectorStrategy...")
    print("=" * 50)
    
    strategy = DefectorStrategy()
    print(f"Initial score: {strategy.score} (skeptical bias)")
    print(f"Position scale: {strategy.POSITION_SCALE}x")
    
    # Test Case 1: High consensus - should fade (not invert!)
    print("\nTest 1: High Consensus (should FADE to small position)")
    ctx1 = MarketContext(
        date="2024-03-15",
        ticker="AAPL",
        sample_num=1,
        daily_return=0.02,
        aggressive_position=0.18,    # 18%
        aggressive_stance="BUY",
        aggressive_confidence="HIGH",
        neutral_position=0.16,       # 16%
        neutral_stance="BUY",
        neutral_confidence="HIGH",
        conservative_position=0.14,  # 14%
        conservative_stance="BUY",
        conservative_confidence="HIGH",
        regime="bull"
    )
    
    positions = [ctx1.aggressive_position, ctx1.neutral_position, ctx1.conservative_position]
    consensus = strategy._calculate_consensus(positions)
    avg = np.mean(positions)
    position, reasoning = strategy.decide_position(ctx1, [])
    
    print(f"  Positions: {[f'{p:.0%}' for p in positions]}")
    print(f"  Average: {avg:.0%}")
    print(f"  Consensus: {consensus:.0%}")
    print(f"  Threshold: {strategy._get_contrarian_threshold():.0%}")
    print(f"  Decision: {position:.1f}% (faded, not inverted!)")
    print(f"  {reasoning}")
    
    # Test Case 2: Low consensus - should see opportunity
    print("\nTest 2: Low Consensus (disagreement = opportunity)")
    ctx2 = MarketContext(
        date="2024-03-16",
        ticker="AAPL",
        sample_num=2,
        daily_return=-0.01,
        aggressive_position=0.20,    # 20%
        aggressive_stance="BUY",
        aggressive_confidence="HIGH",
        neutral_position=0.05,       # 5%
        neutral_stance="HOLD",
        neutral_confidence="LOW",
        conservative_position=0.00,  # 0%
        conservative_stance="AVOID",
        conservative_confidence="LOW",
        regime="sideways"
    )
    
    positions = [ctx2.aggressive_position, ctx2.neutral_position, ctx2.conservative_position]
    consensus = strategy._calculate_consensus(positions)
    avg = np.mean(positions)
    position, reasoning = strategy.decide_position(ctx2, [])
    
    print(f"  Positions: {[f'{p:.0%}' for p in positions]}")
    print(f"  Average: {avg:.0%}")
    print(f"  Consensus: {consensus:.0%}")
    print(f"  Decision: {position:.1f}% (opportunistic)")
    print(f"  {reasoning}")
    
    # Test Case 3: Score effect on threshold
    print("\nTest 3: Score Effect on Threshold")
    
    test_scores = [-8, -5, -2, 0, 3, 5, 8, 10]
    for score in test_scores:
        strategy.score = score
        thresh = strategy._get_contrarian_threshold()
        mode = "NEVER contrarian" if thresh > 1.0 else f"contrarian at {thresh:.0%}"
        print(f"  Score {score:+3d} -> {mode}")
    
    print("\n" + "=" * 50)
    print("DefectorStrategy test complete!")