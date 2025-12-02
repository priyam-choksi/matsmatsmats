"""
buy_hold.py - Buy & Hold Benchmark Strategy
"""

from typing import List, Tuple
import numpy as np

from ..base_strategy import TradingStrategy, TradeResult
from ..market_context import MarketContext


class BuyHoldStrategy(TradingStrategy):
    """
    BUY & HOLD - Always 100% invested, the benchmark to beat.
    """
    
    def __init__(self):
        super().__init__(
            name="Buy-and-Hold",
            description="Always 100% invested - the benchmark to beat",
            position_scale=1.0,
            initial_score=0.0
        )
    
    def decide_position(
        self, 
        ctx: MarketContext, 
        history: List[TradeResult]
    ) -> Tuple[float, str]:
        """Always returns 100% position."""
        avg_pos = np.mean([
            ctx.aggressive_position, 
            ctx.neutral_position, 
            ctx.conservative_position
        ])
        
        reasoning = (
            f"Buy-and-Hold: 100% invested | "
            f"Score={self.score:+.1f} (ignored) | "
            f"Regime={ctx.regime} (ignored) | "
            f"Agent avg={avg_pos:.1%} (ignored) | "
            f"THIS IS THE BENCHMARK"
        )
        
        return 100.0, reasoning