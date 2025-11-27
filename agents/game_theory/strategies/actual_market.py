"""
actual_market.py - Control Group Strategy (Faithful Executor)

Location: agents/game_theory/strategies/actual_market.py

This strategy is the CONTROL GROUP for your experiment.
It faithfully executes the average recommendation from your 
multi-agent system without any modification based on score.

Philosophy:
    "Trust the system. Execute what the agents recommend."

Score Behavior:
    Tracks score but NEVER changes behavior based on it.
    This is intentional - we need a pure baseline to compare against.

Position Logic:
    Simply averages the three agent recommendations and scales up:
    position = average * 3.0 (scaling for meaningful positions)

Why This Matters:
    - Provides baseline to measure if game theory strategies add value
    - If other strategies can't beat this, the game theory layer is useless
    - If other strategies DO beat this, you've proven the value of the layer

ENHANCED with:
    - 3x position scaling for meaningful trades
    - No adaptation (pure control)
"""

from typing import List, Tuple
import numpy as np

from ..base_strategy import TradingStrategy, TradeResult
from ..market_context import MarketContext


class ActualMarketStrategy(TradingStrategy):
    """
    CONTROL GROUP - The Faithful Executor
    
    This strategy represents what happens if you just follow
    the multi-agent system's recommendations with appropriate scaling
    but without any game theory adaptation.
    
    Behavior:
        - Takes the average of all three risk evaluations
        - Scales up by 3x for meaningful positions
        - NEVER modifies based on score
    
    Score Interpretation:
        Score is tracked but completely ignored. This is intentional.
        We need a pure baseline that doesn't adapt.
    
    Position Scaling:
        The AI agents give conservative positions (typically 0-20%).
        We scale by 3x to get meaningful game theory positions (0-60%).
        This allows for actual competition between strategies.
    
    When It Wins:
        - When the multi-agent system is well-calibrated
        - When market conditions match agent assumptions
        - When adaptation adds noise rather than signal
    
    When It Loses:
        - When agents are systematically biased (too conservative)
        - When market regime changes require adaptation
        - When contrarian signals would have helped
    """
    
    # Position scaling for meaningful trades
    POSITION_SCALE = 3.0
    
    def __init__(self):
        super().__init__(
            name="Actual Market",
            description="Control group - executes scaled system recommendation without modification",
            position_scale=self.POSITION_SCALE,
            initial_score=0.0  # Neutral starting point (though ignored)
        )
    
    def decide_position(
        self, 
        ctx: MarketContext, 
        history: List[TradeResult]
    ) -> Tuple[float, str]:
        """
        Decide position by averaging the three agent recommendations.
        
        This is intentionally simple - no score adjustment, no adaptation.
        Pure execution of the average recommendation, scaled up.
        
        Args:
            ctx: MarketContext with agent evaluations
            history: Past trades (ignored by this strategy)
            
        Returns:
            Tuple of (position_pct, reasoning)
        """
        # Get position recommendations from all three agents
        # These are decimals (0.0 to 1.0) from your JSON files
        aggressive_pos = ctx.aggressive_position
        neutral_pos = ctx.neutral_position
        conservative_pos = ctx.conservative_position
        
        # Simple average
        avg_position = np.mean([aggressive_pos, neutral_pos, conservative_pos])
        
        # Convert to percentage and scale up
        base_position_pct = avg_position * 100.0
        scaled_position_pct = base_position_pct * self.position_scale
        
        # Cap at 100%
        position_pct = min(100.0, scaled_position_pct)
        
        # Build reasoning string
        reasoning = (
            f"Control: avg("
            f"agg={aggressive_pos:.1%}, "
            f"neu={neutral_pos:.1%}, "
            f"con={conservative_pos:.1%}) = "
            f"{avg_position:.1%} → scaled {self.position_scale}x = "
            f"{position_pct:.1f}% "
            f"[score={self.score:+.0f}, ignored]"
        )
        
        return position_pct, reasoning


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing ActualMarketStrategy...")
    print("=" * 50)
    
    # Create mock context
    ctx = MarketContext(
        date="2024-03-15",
        ticker="AAPL",
        sample_num=1,
        portfolio_size=100000,
        daily_return=0.015,  # 1.5% market return
        aggressive_position=0.20,   # 20%
        aggressive_stance="BUY",
        aggressive_confidence="HIGH",
        neutral_position=0.05,      # 5%
        neutral_stance="SMALL BUY",
        neutral_confidence="LOW",
        conservative_position=0.01,  # 1%
        conservative_stance="HOLD",
        conservative_confidence="LOW",
        regime="bull"
    )
    
    strategy = ActualMarketStrategy()
    
    # Test decision
    position, reasoning = strategy.decide_position(ctx, [])
    print(f"\nContext:")
    print(f"  Aggressive: {ctx.aggressive_position:.1%}")
    print(f"  Neutral: {ctx.neutral_position:.1%}")
    print(f"  Conservative: {ctx.conservative_position:.1%}")
    print(f"\nDecision:")
    print(f"  Average: {np.mean([ctx.aggressive_position, ctx.neutral_position, ctx.conservative_position]):.1%}")
    print(f"  Position: {position:.1f}%")
    print(f"  Reasoning: {reasoning}")
    
    # Execute trade
    result = strategy.execute_trade(ctx)
    print(f"\nTrade Result:")
    print(f"  Market return: {result.market_return:.2%}")
    print(f"  Trade return: {result.trade_return:.2%}")
    print(f"  Score after: {strategy.score:+.0f}")
    
    # Simulate a few more trades to show score tracking (but not affecting decisions)
    print(f"\nSimulating 5 more trades...")
    returns = [0.02, -0.01, 0.015, -0.025, 0.01]
    for i, ret in enumerate(returns):
        ctx.daily_return = ret
        ctx.sample_num = i + 2
        result = strategy.execute_trade(ctx)
        print(f"  Trade {i+2}: market={ret:+.2%}, position={result.position_pct:.1f}%, score={strategy.score:+.0f}")
    
    print(f"\nFinal Summary:")
    print(f"  Total Return: {strategy.total_return_pct:+.2f}%")
    print(f"  Win Rate: {strategy.win_rate:.1f}%")
    print(f"  Final Score: {strategy.score:+.0f} (tracked but ignored)")
    print(f"  Position always: avg * {strategy.POSITION_SCALE}x")
    
    print("\n" + "=" * 50)
    print("ActualMarketStrategy test complete!")