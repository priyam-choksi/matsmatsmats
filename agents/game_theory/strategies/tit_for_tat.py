"""
tit_for_tat.py - Adaptive Learner Strategy

Location: agents/game_theory/strategies/tit_for_tat.py

This strategy mirrors what worked last round. It adapts by
copying the approach of the winning strategy, believing that
recent success is predictive of near-term success.

Philosophy:
    "Different regimes favor different strategies. Adapt to what's working."

Score Behavior:
    Score determines ADAPTATION STRENGTH (how much to mirror)
    - High score (+5 to +10): Adaptation = 100% (full mirror of winner)
    - Neutral (-2 to +5):     Adaptation = 70% (partial mirror)
    - Low score (-10 to -2):  Adaptation = 40% (mostly hedge/blend)

Position Logic:
    1. Track what each approach would suggest
    2. Mirror last round's winning approach with adaptation strength
    3. Blend with neutral (average) position based on adaptation

IMPORTANT:
    Requires external call to update_winner() after each round.
    The tournament engine must call this with the name of the
    strategy that performed best in the previous round.

ENHANCED with:
    - Variable position scaling based on who it's copying
    - Memory of last 3 winners (not just 1)
    - Initial neutral score (0)
    - Hot/cold streak awareness
"""

from typing import List, Tuple, Dict
import numpy as np

from ..base_strategy import TradingStrategy, TradeResult
from ..market_context import MarketContext


class TitForTatStrategy(TradingStrategy):
    """
    ADAPTIVE LEARNER - Mirror What Worked Last Round
    
    This strategy is inspired by the famous Tit-for-Tat strategy
    from game theory, which won Axelrod's tournament by being
    simple, nice, and retaliatory.
    
    In this trading context:
        - "Cooperate" = follow the winning approach
        - "Defect" = if mirroring isn't working, blend more
    
    Core Idea:
        Different market regimes favor different strategies:
        - Bull markets might favor aggressive approaches
        - Bear markets might favor conservative approaches
        - Sideways markets might favor contrarian approaches
        
        By mirroring the recent winner, we adapt to the regime.
    
    Approach Types:
        We map strategies to approach types:
        - "cooperator": Follow consensus, scale with agreement
        - "defector": Contrarian at extremes  
        - "conservative": Follow conservative agent
        - "aggressive": Follow aggressive agent
        - "neutral": Average of all agents
    
    Score Interpretation:
        Score determines how strongly to mirror:
        
        | Score Range | Adaptation | Interpretation |
        |-------------|------------|----------------|
        | +5 to +10   | 100%       | "Full mirror, it's working" |
        | -2 to +5    | 70%        | "Partial mirror" |
        | -10 to -2   | 40%        | "Mostly blend, mirroring isn't working" |
    
    When It Wins:
        - Regime changes where adapting matters
        - When recent performance predicts future performance
        - Markets with momentum in strategy effectiveness
    
    When It Loses:
        - Rapidly changing conditions (always one step behind)
        - When the best strategy keeps changing
        - Mean-reverting strategy effectiveness
    """
    
    # Position scaling varies based on who we're copying
    POSITION_SCALE_MAP = {
        'cooperator': 4.0,    # Same as cooperator
        'defector': 5.0,      # Same as defector
        'conservative': 3.0,  # Conservative scaling
        'aggressive': 6.0,    # Most aggressive scaling
        'neutral': 3.5        # Moderate scaling
    }
    
    def __init__(self):
        super().__init__(
            name="Tit-for-Tat",
            description="Adaptive learner - mirrors winning strategies",
            position_scale=4.0,  # Default, changes based on who we copy
            initial_score=0.0  # Neutral starting point
        )
        
        # Start by mirroring cooperator (follow consensus)
        self.last_winning_approach = "cooperator"
        
        # Track last 3 winners for better adaptation
        self.winner_history = ["cooperator", "cooperator", "cooperator"]
        
        # Track wins by approach type
        self.approach_wins: Dict[str, int] = {
            "cooperator": 0,
            "defector": 0,
            "conservative": 0,
            "aggressive": 0,
            "neutral": 0
        }
        
        # Track which approach we mirrored each round
        self.approach_history: List[str] = []
    
    def reset(self):
        """Reset including adaptation state."""
        super().reset()
        self.last_winning_approach = "cooperator"
        self.winner_history = ["cooperator", "cooperator", "cooperator"]
        self.approach_wins = {k: 0 for k in self.approach_wins}
        self.approach_history = []
    
    def update_winner(self, winner_name: str):
        """
        Update with the strategy that won last round.
        
        Call this after each round with the name of the strategy
        that had the highest return.
        
        Args:
            winner_name: Name of winning strategy (e.g., "Cooperator")
        
        Maps strategy names to approach types:
            - "Cooperator", "consensus" -> "cooperator"
            - "Defector", "contrarian" -> "defector"
            - "Conservative", "Buy-and-Hold" -> "conservative"
            - "Actual Market" -> "neutral"
            - Others -> "aggressive"
        """
        name_lower = winner_name.lower()
        
        # Map strategy name to approach type
        if "cooperator" in name_lower or "consensus" in name_lower:
            approach = "cooperator"
        elif "defector" in name_lower or "contrar" in name_lower:
            approach = "defector"
        elif "conserv" in name_lower or "hold" in name_lower or "patient" in name_lower:
            approach = "conservative"
        elif "actual" in name_lower or "market" in name_lower or "control" in name_lower:
            approach = "neutral"
        else:
            approach = "aggressive"
        
        self.last_winning_approach = approach
        
        # Update winner history (keep last 3)
        self.winner_history.append(approach)
        if len(self.winner_history) > 3:
            self.winner_history.pop(0)
        
        # Track wins
        if approach in self.approach_wins:
            self.approach_wins[approach] += 1
    
    def _get_dominant_approach(self) -> str:
        """
        Get the most common winner from recent history.
        
        Returns:
            Most common approach from last 3 winners
        """
        from collections import Counter
        counts = Counter(self.winner_history)
        return counts.most_common(1)[0][0]
    
    def _get_adaptation_strength(self) -> float:
        """
        Get how strongly to mirror based on score.
        
        Higher score = stronger mirror (it's working)
        Lower score = weaker mirror (blend more)
        
        Returns:
            Adaptation strength (0.0 to 1.0)
        """
        if self.score >= 5:
            return 1.0   # Full mirror
        elif self.score >= -2:
            return 0.7   # Partial mirror
        else:
            return 0.4   # Mostly blend
    
    def _calculate_approach_positions(self, ctx: MarketContext) -> Dict[str, float]:
        """
        Calculate what each approach would suggest.
        
        Args:
            ctx: MarketContext with agent evaluations
            
        Returns:
            Dictionary mapping approach -> suggested position (percentage)
        """
        # Get raw positions
        positions = [
            ctx.aggressive_position,
            ctx.neutral_position,
            ctx.conservative_position
        ]
        avg_position = np.mean(positions)
        avg_pct = avg_position * 100.0
        
        # Calculate consensus for cooperator approach
        std_pos = np.std(positions)
        consensus = max(0.0, 1.0 - std_pos / 0.10)
        
        # Cooperator approach: consensus-scaled
        cooperator_base = avg_pct * (0.5 + 0.5 * consensus)
        cooperator_pos = cooperator_base * self.POSITION_SCALE_MAP['cooperator']
        
        # Defector approach: fade if high consensus, follow if low
        if consensus > 0.8:
            # High consensus - fade to small position
            defector_base = 30.0 * 0.4  # Small contrarian position
        else:
            # Low consensus - follow aggressively
            defector_base = avg_pct * 2.0
        defector_pos = defector_base * self.POSITION_SCALE_MAP['defector']
        
        # Conservative approach: follow conservative agent with scaling
        conservative_base = ctx.conservative_position * 100.0
        conservative_pos = conservative_base * self.POSITION_SCALE_MAP['conservative']
        
        # Aggressive approach: follow aggressive agent with scaling
        aggressive_base = ctx.aggressive_position * 100.0
        aggressive_pos = aggressive_base * self.POSITION_SCALE_MAP['aggressive']
        
        # Neutral approach: simple average with scaling
        neutral_pos = avg_pct * self.POSITION_SCALE_MAP['neutral']
        
        return {
            "cooperator": min(85.0, cooperator_pos),
            "defector": max(20.0, min(80.0, defector_pos)),
            "conservative": min(70.0, conservative_pos),
            "aggressive": min(90.0, aggressive_pos),
            "neutral": min(75.0, neutral_pos)
        }
    
    def decide_position(
        self, 
        ctx: MarketContext, 
        history: List[TradeResult]
    ) -> Tuple[float, str]:
        """
        Decide position by mirroring the last winning approach.
        
        Blends the target approach with neutral based on adaptation strength.
        
        Args:
            ctx: MarketContext with agent evaluations
            history: Past trades
            
        Returns:
            Tuple of (position_pct, reasoning)
        """
        # Get adaptation strength based on score
        adaptation = self._get_adaptation_strength()
        
        # Get dominant approach from recent winners
        dominant = self._get_dominant_approach()
        
        # Calculate what each approach would suggest
        approach_positions = self._calculate_approach_positions(ctx)
        
        # Get target position from dominant winning approach
        target_pos = approach_positions.get(dominant, approach_positions["neutral"])
        
        # Get neutral position for blending
        neutral_pos = approach_positions["neutral"]
        
        # Blend: adaptation% of target + (1-adaptation)% of neutral
        position_pct = target_pos * adaptation + neutral_pos * (1.0 - adaptation)
        
        # Apply hot/cold adjustment
        if self.is_hot:
            position_pct *= 1.1
        elif self.is_cold:
            position_pct *= 0.9
        
        # Update position scale to match who we're copying
        self.position_scale = self.POSITION_SCALE_MAP.get(dominant, 4.0)
        
        # Clamp to reasonable range
        position_pct = max(15.0, min(85.0, position_pct))
        
        # Track which approach we're mirroring
        self.approach_history.append(dominant)
        
        # Build reasoning
        reasoning = (
            f"TitForTat: mirroring '{dominant}' "
            f"(recent: {'/'.join(self.winner_history[-3:])}) "
            f"target={target_pos:.0f}%, adapt={adaptation:.0%} "
        )
        
        if self.is_hot:
            reasoning += "[HOT] "
        elif self.is_cold:
            reasoning += "[COLD] "
            
        reasoning += f"[score={self.score:+.0f}] -> {position_pct:.0f}%"
        
        return position_pct, reasoning
    
    def get_adaptation_stats(self) -> dict:
        """Get statistics about adaptation behavior."""
        total_wins = sum(self.approach_wins.values())
        
        stats = {
            "approach_wins": self.approach_wins,
            "total_rounds_tracked": total_wins,
            "most_successful_approach": (
                max(self.approach_wins, key=self.approach_wins.get)
                if total_wins > 0 else "none"
            ),
            "current_mirroring": self.last_winning_approach,
            "recent_winners": self.winner_history[-3:],
            "approach_history": self.approach_history[-10:]  # Last 10
        }
        
        # Calculate which approach we've mirrored most
        if self.approach_history:
            from collections import Counter
            mirror_counts = Counter(self.approach_history)
            stats["most_mirrored"] = mirror_counts.most_common(1)[0][0]
            stats["mirror_distribution"] = dict(mirror_counts)
        
        return stats


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing TitForTatStrategy...")
    print("=" * 50)
    
    strategy = TitForTatStrategy()
    print(f"Initial score: {strategy.score} (neutral)")
    print(f"Initial approach: {strategy.last_winning_approach}")
    
    # Test Case 1: Initial state (mirrors cooperator by default)
    print("\nTest 1: Initial State")
    ctx1 = MarketContext(
        date="2024-03-15",
        ticker="AAPL",
        sample_num=1,
        daily_return=0.02,
        aggressive_position=0.20,
        aggressive_stance="BUY",
        aggressive_confidence="HIGH",
        neutral_position=0.10,
        neutral_stance="HOLD",
        neutral_confidence="MEDIUM",
        conservative_position=0.05,
        conservative_stance="HOLD",
        conservative_confidence="LOW",
        regime="bull"
    )
    
    position, reasoning = strategy.decide_position(ctx1, [])
    print(f"  Mirroring: {strategy.last_winning_approach}")
    print(f"  Decision: {position:.1f}%")
    print(f"  {reasoning}")
    
    # Test Case 2: Update winner and see change
    print("\nTest 2: After Defector Wins")
    strategy.update_winner("Defector")
    
    position, reasoning = strategy.decide_position(ctx1, [])
    print(f"  Mirroring: {strategy.last_winning_approach}")
    print(f"  Decision: {position:.1f}%")
    print(f"  {reasoning}")
    
    # Test Case 3: Multiple winner updates
    print("\nTest 3: After Multiple Winners")
    strategy.update_winner("Cooperator")
    strategy.update_winner("Buy-and-Hold")
    
    dominant = strategy._get_dominant_approach()
    position, reasoning = strategy.decide_position(ctx1, [])
    print(f"  Winner history: {strategy.winner_history}")
    print(f"  Dominant approach: {dominant}")
    print(f"  Decision: {position:.1f}%")
    print(f"  {reasoning}")
    
    # Test Case 4: Score effect on adaptation
    print("\nTest 4: Score Effect on Adaptation")
    
    test_scores = [-8, -5, -2, 0, 3, 5, 8, 10]
    for score in test_scores:
        strategy.score = score
        adapt = strategy._get_adaptation_strength()
        print(f"  Score {score:+3d} -> Adaptation: {adapt:.0%}")
    
    print("\n" + "=" * 50)
    print("TitForTatStrategy test complete!")