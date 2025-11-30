"""
cooperator.py - Consensus Follower Strategy

Location: agents/game_theory/strategies/cooperator.py

This strategy trusts collective wisdom and scales position
with how much the agents agree (consensus).

Philosophy:
    "When agents agree, amplify. When uncertain, reduce."

Score Behavior:
    Score DIRECTLY multiplies position size
    - High score (+5 to +10): Multiplier = 1.2 to 1.5 (scale up)
    - Neutral (-2 to +5):     Multiplier = 1.0 (normal)
    - Low score (-10 to -2):  Multiplier = 0.5 to 0.8 (protect capital)

Position Logic:
    1. Calculate consensus (how much agents agree) from position std
    2. Base position = average * (0.5 + 0.5 * consensus)
    3. Apply 4x position scaling for meaningful trades
    4. Apply score multiplier
    5. Check if hot/cold and adjust
    6. Cap at 85%

Why This Matters:
    - Tests if consensus is a useful signal
    - When all agents agree, there's likely something there
    - When agents disagree, uncertainty is high - reduce exposure

ENHANCED with:
    - 4x position scaling for meaningful trades
    - Starting score bias of +2 (optimistic)
    - Consensus success tracking
    - Hot/cold streak awareness
"""

from typing import List, Tuple
import numpy as np

from ..base_strategy import TradingStrategy, TradeResult
from ..market_context import MarketContext


class CooperatorStrategy(TradingStrategy):
    """
    CONSENSUS FOLLOWER - Trust Collective Wisdom, Scale with Momentum
    
    This strategy believes that when multiple independent agents
    reach similar conclusions, the signal is stronger.
    
    Core Idea:
        - High consensus (agents agree) = amplify the signal
        - Low consensus (agents disagree) = reduce exposure
        - Score provides momentum overlay
        - Learn from whether consensus has been working
    
    Consensus Calculation:
        consensus = 1.0 - (std of positions / 0.10)
        
        If std = 0 (all same): consensus = 1.0
        If std = 0.10 (spread out): consensus = 0.0
    
    Score Interpretation:
        Score directly scales position size:
        
        | Score Range | Multiplier | Effect |
        |-------------|------------|--------|
        | +5 to +10   | 1.2 - 1.5  | "Hot hand, scale up" |
        | -2 to +5    | 1.0        | "Normal" |
        | -10 to -2   | 0.5 - 0.8  | "Cold streak, protect" |
    
    When It Wins:
        - When consensus actually predicts outcomes
        - Trending markets where agents align
        - When protecting capital during losing streaks helps
    
    When It Loses:
        - When agents agree but are wrong together
        - Choppy markets with false consensus
        - When scaling up on winning streaks leads to overexposure
    """
    
    # Maximum position size
    MAX_POSITION = 85.0
    # Position scaling for meaningful trades
    POSITION_SCALE = 4.0
    
    def __init__(self):
        super().__init__(
            name="Cooperator",
            description="Consensus follower - scales position with agreement and score",
            position_scale=self.POSITION_SCALE,
            initial_score=2.0  # Optimistic starting bias
        )
        
        # Track consensus success for learning
        self.consensus_history = []  # List of (consensus_level, worked_bool)
        self.high_consensus_wins = 0
        self.high_consensus_trades = 0
        self.low_consensus_wins = 0
        self.low_consensus_trades = 0
    
    def reset(self):
        """Reset strategy state including consensus tracking."""
        super().reset()
        self.consensus_history = []
        self.high_consensus_wins = 0
        self.high_consensus_trades = 0
        self.low_consensus_wins = 0
        self.low_consensus_trades = 0
    
    def _get_score_multiplier(self) -> float:
        """
        Get position multiplier based on current score.
        
        High score = scale up (momentum)
        Low score = scale down (protect capital)
        
        Returns:
            Multiplier for position size (0.5 to 1.5)
        """
        if self.score >= 5:
            # High score: 1.2 to 1.5
            # Linear interpolation from score 5->10 maps to 1.2->1.5
            return 1.2 + (self.score - 5) * 0.06
        
        elif self.score >= -2:
            # Neutral: 1.0
            return 1.0
        
        else:
            # Low score: 0.5 to 0.8
            # Linear interpolation from score -10->-2 maps to 0.5->0.8
            # At -10: 0.5, at -2: 0.8
            return 0.5 + (self.score + 10) * 0.0375
    
    def _calculate_consensus(self, positions: List[float]) -> float:
        """
        Calculate consensus level from position recommendations.
        
        High consensus = agents agree (low std)
        Low consensus = agents disagree (high std)
        
        Args:
            positions: List of position recommendations (decimals)
            
        Returns:
            Consensus level (0.0 to 1.0)
        """
        if len(positions) < 2:
            return 1.0
        
        # If all positions are the same
        if len(set(positions)) == 1:
            return 1.0
        
        std_position = np.std(positions)
        
        # Normalize: assume max reasonable std is ~0.10 (10% spread)
        # std of 0 = consensus 1.0
        # std of 0.10+ = consensus 0.0
        consensus = max(0.0, 1.0 - std_position / 0.10)
        
        return consensus
    
    def _get_consensus_adjustment(self) -> float:
        """
        Adjust position based on whether consensus has been working.
        
        Returns:
            Adjustment factor (0.8 to 1.2)
        """
        # Need enough history
        if len(self.consensus_history) < 10:
            return 1.0
        
        # Look at recent high consensus trades
        recent_high = [(c, w) for c, w in self.consensus_history[-20:] if c > 0.7]
        if len(recent_high) >= 5:
            success_rate = sum(w for _, w in recent_high) / len(recent_high)
            if success_rate > 0.7:
                return 1.2  # Consensus working great
            elif success_rate < 0.3:
                return 0.8  # Consensus failing
        
        return 1.0
    
    def decide_position(
        self, 
        ctx: MarketContext, 
        history: List[TradeResult]
    ) -> Tuple[float, str]:
        """
        Decide position based on consensus and score.
        
        Formula:
            base_position = avg_position * (0.5 + 0.5 * consensus)
            scaled_position = base_position * position_scale
            final_position = scaled_position * score_multiplier * adjustments
        
        Args:
            ctx: MarketContext with agent evaluations
            history: Past trades (for context)
            
        Returns:
            Tuple of (position_pct, reasoning)
        """
        # Get all three position recommendations
        positions = [
            ctx.aggressive_position,
            ctx.neutral_position,
            ctx.conservative_position
        ]
        
        # Calculate average and consensus
        avg_position = np.mean(positions)
        consensus = self._calculate_consensus(positions)
        
        # Base position scales with consensus
        # Low consensus (0.0) -> 50% of avg
        # High consensus (1.0) -> 100% of avg
        base_position_pct = avg_position * 100.0 * (0.5 + 0.5 * consensus)
        
        # Apply position scaling for meaningful trades
        scaled_position = base_position_pct * self.position_scale
        
        # Apply score multiplier
        score_mult = self._get_score_multiplier()
        position_pct = scaled_position * score_mult
        
        # Apply consensus success adjustment
        consensus_adj = self._get_consensus_adjustment()
        position_pct = position_pct * consensus_adj
        
        # Apply hot/cold adjustment
        if self.is_hot:
            position_pct *= 1.1
        elif self.is_cold:
            position_pct *= 0.9
        
        # Cap at maximum
        position_pct = min(self.MAX_POSITION, position_pct)
        
        # Build reasoning
        reasoning = (
            f"Cooperator: consensus={consensus:.0%}, "
            f"avg={avg_position:.1%}, "
            f"base={base_position_pct:.1f}%, "
            f"scaled={scaled_position:.1f}%, "
            f"mult={score_mult:.2f}x "
        )
        
        if consensus_adj != 1.0:
            reasoning += f"cons_adj={consensus_adj:.1f}x "
        
        if self.is_hot:
            reasoning += "[HOT] "
        elif self.is_cold:
            reasoning += "[COLD] "
            
        reasoning += f"[score={self.score:+.0f}] -> {position_pct:.1f}%"
        
        return position_pct, reasoning
    
    def execute_trade(self, ctx: MarketContext) -> TradeResult:
        """
        Override to track consensus success.
        """
        # Calculate consensus before trade
        positions = [
            ctx.aggressive_position,
            ctx.neutral_position,
            ctx.conservative_position
        ]
        consensus = self._calculate_consensus(positions)
        
        # Execute the trade
        result = super().execute_trade(ctx)
        
        # Track consensus success
        won = result.trade_return > 0
        self.consensus_history.append((consensus, won))
        
        # Track high/low consensus separately
        if consensus > 0.7:
            self.high_consensus_trades += 1
            if won:
                self.high_consensus_wins += 1
        elif consensus < 0.3:
            self.low_consensus_trades += 1
            if won:
                self.low_consensus_wins += 1
        
        # Keep history manageable
        if len(self.consensus_history) > 50:
            self.consensus_history.pop(0)
        
        # Store in strategy memory for reference
        self.strategy_memory['last_consensus'] = consensus
        self.strategy_memory['consensus_win_rate'] = (
            self.high_consensus_wins / self.high_consensus_trades 
            if self.high_consensus_trades > 0 else 0.5
        )
        
        return result
    
    def get_consensus_stats(self) -> dict:
        """Get statistics about consensus from trade history."""
        
        stats = {
            "high_consensus_win_rate": (
                self.high_consensus_wins / self.high_consensus_trades * 100
                if self.high_consensus_trades > 0 else 0
            ),
            "low_consensus_win_rate": (
                self.low_consensus_wins / self.low_consensus_trades * 100
                if self.low_consensus_trades > 0 else 0
            ),
            "high_consensus_trades": self.high_consensus_trades,
            "low_consensus_trades": self.low_consensus_trades,
        }
        
        # Add recent consensus performance
        if len(self.consensus_history) >= 10:
            recent = self.consensus_history[-10:]
            recent_high = [(c, w) for c, w in recent if c > 0.7]
            recent_low = [(c, w) for c, w in recent if c < 0.3]
            
            if recent_high:
                stats["recent_high_consensus_win_rate"] = sum(w for _, w in recent_high) / len(recent_high) * 100
            if recent_low:
                stats["recent_low_consensus_win_rate"] = sum(w for _, w in recent_low) / len(recent_low) * 100
        
        return stats


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing CooperatorStrategy...")
    print("=" * 50)
    
    strategy = CooperatorStrategy()
    print(f"Initial score: {strategy.score} (optimistic bias)")
    print(f"Position scale: {strategy.POSITION_SCALE}x")
    
    # Test Case 1: High consensus (agents agree)
    print("\nTest 1: High Consensus (agents agree)")
    ctx1 = MarketContext(
        date="2024-03-15",
        ticker="AAPL",
        sample_num=1,
        daily_return=0.02,
        aggressive_position=0.15,    # 15%
        aggressive_stance="BUY",
        aggressive_confidence="HIGH",
        neutral_position=0.12,       # 12%
        neutral_stance="BUY",
        neutral_confidence="MEDIUM",
        conservative_position=0.10,  # 10%
        conservative_stance="HOLD",
        conservative_confidence="LOW",
        regime="bull"
    )
    
    positions = [ctx1.aggressive_position, ctx1.neutral_position, ctx1.conservative_position]
    consensus = strategy._calculate_consensus(positions)
    position, reasoning = strategy.decide_position(ctx1, [])
    
    print(f"  Positions: {[f'{p:.0%}' for p in positions]}")
    print(f"  Average: {np.mean(positions):.1%}")
    print(f"  Consensus: {consensus:.0%}")
    print(f"  Decision: {position:.1f}%")
    print(f"  {reasoning}")
    
    # Test Case 2: Low consensus (agents disagree)
    print("\nTest 2: Low Consensus (agents disagree)")
    ctx2 = MarketContext(
        date="2024-03-16",
        ticker="AAPL",
        sample_num=2,
        daily_return=-0.01,
        aggressive_position=0.25,    # 25%
        aggressive_stance="BUY",
        aggressive_confidence="HIGH",
        neutral_position=0.08,       # 8%
        neutral_stance="HOLD",
        neutral_confidence="LOW",
        conservative_position=0.00,  # 0%
        conservative_stance="AVOID",
        conservative_confidence="LOW",
        regime="sideways"
    )
    
    positions = [ctx2.aggressive_position, ctx2.neutral_position, ctx2.conservative_position]
    consensus = strategy._calculate_consensus(positions)
    position, reasoning = strategy.decide_position(ctx2, [])
    
    print(f"  Positions: {[f'{p:.0%}' for p in positions]}")
    print(f"  Average: {np.mean(positions):.1%}")
    print(f"  Consensus: {consensus:.0%}")
    print(f"  Decision: {position:.1f}%")
    print(f"  {reasoning}")
    
    # Test Case 3: Score effect on multiplier
    print("\nTest 3: Score Effect on Multiplier")
    
    test_scores = [-8, -5, -2, 0, 3, 5, 8, 10]
    for score in test_scores:
        strategy.score = score
        mult = strategy._get_score_multiplier()
        print(f"  Score {score:+3d} -> Multiplier: {mult:.2f}x")
    
    print("\n" + "=" * 50)
    print("CooperatorStrategy test complete!")