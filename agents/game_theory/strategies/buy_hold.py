# agents/game_theory/strategies/buy_hold.py

"""
Buy & Hold Strategy - The Patient Optimist

Personality: "Time in market > timing the market"

Philosophy:
- Ignores short-term noise
- Only acts on VERY strong signals (>80% confidence)
- Never sells in bull markets
- Minimizes transaction costs

Game Theory Role: The "always cooperate" player - minimizes friction
"""

from typing import Optional, List, Dict
from ..strategy_interface import TradingStrategy, StrategyDecision
from ..market_context import MarketContext


class BuyHoldStrategy(TradingStrategy):
    """
    Patient long-term strategy that filters out weak signals.
    """
    
    def __init__(self, confidence_threshold: float = 0.80):
        """
        Args:
            confidence_threshold: Only act if confidence exceeds this (default 80%)
        """
        super().__init__(name="Buy & Hold")
        self.confidence_threshold = confidence_threshold
    
    def make_decision(self,
                     context: MarketContext,
                     tournament_history: Optional[List[Dict]] = None) -> StrategyDecision:
        """
        Filter decision through long-term lens.
        
        Logic:
        1. If confidence < threshold → HOLD
        2. If SELL in bull market → HOLD
        3. Otherwise → follow base decision
        """
        
        base = context.base_decision
        action = base.action
        reasoning_parts = []
        
        # Rule 1: Confidence filter
        if base.confidence < self.confidence_threshold:
            action = 'HOLD'
            reasoning_parts.append(
                f"Confidence {base.confidence:.1%} below Buy&Hold threshold "
                f"{self.confidence_threshold:.1%}"
            )
        
        # Rule 2: Never sell in bull markets
        elif action == 'SELL' and context.regime in ['bull_trend', 'momentum']:
            action = 'HOLD'
            reasoning_parts.append(
                f"Buy&Hold doesn't sell in {context.regime} regime"
            )
        
        # Rule 3: Accept strong signal
        else:
            reasoning_parts.append(
                f"Strong signal ({base.confidence:.1%}) in {context.regime} - "
                f"following base recommendation"
            )
        
        reasoning = "; ".join(reasoning_parts)
        
        return self.create_decision(
            action=action,
            position_multiplier=1.0,  # Same position size as base
            confidence_multiplier=1.0,  # Same confidence
            reasoning=reasoning,
            context=context,
            metadata={
                'confidence_threshold': self.confidence_threshold,
                'original_action': base.action
            }
        )