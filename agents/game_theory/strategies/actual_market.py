# agents/game_theory/strategies/actual_market.py

"""
Actual Market Strategy - The Control Group

Personality: "I trust the multi-agent system completely"

This strategy uses your Risk Manager's decision with NO modifications.
It serves as the control/baseline for comparison.

Game Theory Role: The "truthful" player with no strategic manipulation
"""

from typing import Optional, List, Dict
from ..strategy_interface import TradingStrategy, StrategyDecision
from ..market_context import MarketContext


class ActualMarketStrategy(TradingStrategy):
    """
    Uses the base decision directly with no modifications.
    This is your control group - pure multi-agent system output.
    """
    
    def __init__(self):
        super().__init__(name="Actual Market")
    
    def make_decision(self,
                     context: MarketContext,
                     tournament_history: Optional[List[Dict]] = None) -> StrategyDecision:
        """
        Simply pass through the base decision unchanged.
        
        Args:
            context: Market context with base decision
            tournament_history: Not used (but required by interface)
            
        Returns:
            StrategyDecision matching the base decision exactly
        """
        
        base = context.base_decision
        
        return StrategyDecision(
            strategy_name=self.name,
            action=base.action,
            position_size=base.position_size_dollars or 0.0,
            confidence=base.confidence,
            reasoning="Using Risk Manager's decision directly (control strategy)",
            metadata={
                'regime': context.regime,
                'consensus': context.analyst_consensus
            }
        )