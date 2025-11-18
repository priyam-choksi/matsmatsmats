# agents/game_theory/strategies/cooperator.py

"""
Cooperator Strategy - The Momentum Amplifier

Personality: "When everyone's bullish, I'm MORE bullish. Ride the wave!"

Philosophy:
- Amplifies decisions when market agrees (high consensus + aligned regime)
- Reduces position when market is uncertain
- Profits from trend continuation
- Rewards coordination

Game Theory Role: The "reward cooperation" player - profits when system and market align
"""

from typing import Optional, List, Dict
from ..strategy_interface import TradingStrategy, StrategyDecision
from ..market_context import MarketContext


class CooperatorStrategy(TradingStrategy):
    """
    Amplifies signals when consensus is high and market aligns.
    Reduces exposure when market is uncertain.
    """
    
    def __init__(self):
        super().__init__(name="Cooperator")
    
    def make_decision(self,
                     context: MarketContext,
                     tournament_history: Optional[List[Dict]] = None) -> StrategyDecision:
        """
        Adjust position based on alignment and consensus.
        
        Logic:
        - High alignment (>0.5) + High consensus (>0.7) → Amplify by 1.3x
        - Low alignment (<0) → Reduce by 0.7x
        - Neutral → Keep same
        """
        
        alignment = context.alignment_score()
        consensus = context.analyst_consensus
        
        # Calculate position multiplier
        if alignment > 0.5 and consensus > 0.7:
            # Strong agreement - be aggressive
            position_multiplier = 1.0 + (alignment * 0.5)  # 1.0 to 1.5x
            confidence_multiplier = 1.15
            reasoning = (f"Amplifying: High alignment ({alignment:.2f}) + "
                        f"high consensus ({consensus:.1%})")
        
        elif alignment < 0:
            # Misalignment - be cautious
            position_multiplier = 0.7
            confidence_multiplier = 0.9
            reasoning = f"Reducing: Misalignment detected ({alignment:.2f})"
        
        else:
            # Neutral - follow base
            position_multiplier = 1.0
            confidence_multiplier = 1.0
            reasoning = f"Neutral alignment ({alignment:.2f}), following base decision"
        
        return self.create_decision(
            action=context.base_decision.action,
            position_multiplier=position_multiplier,
            confidence_multiplier=confidence_multiplier,
            reasoning=reasoning,
            context=context,
            metadata={
                'alignment': alignment,
                'consensus': consensus,
                'amplification_factor': position_multiplier
            }
        )