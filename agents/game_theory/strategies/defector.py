# agents/game_theory/strategies/defector.py

"""
Defector Strategy - The Contrarian Skeptic

Personality: "Everyone's greedy? I'm fearful. Everyone's fearful? I'm greedy."

Philosophy:
- Bets AGAINST consensus when confidence is moderate
- Exploits herd behavior and overreaction
- Profits from mean reversion
- Skeptical of extreme sentiment

Game Theory Role: The "exploit herd behavior" player - profits from mean reversion
"""

from typing import Optional, List, Dict
from ..strategy_interface import TradingStrategy, StrategyDecision
from ..market_context import MarketContext


class DefectorStrategy(TradingStrategy):
    """
    Contrarian strategy that bets against strong consensus.
    Exploits potential overreactions in the market.
    """
    
    def __init__(self):
        super().__init__(name="Defector")
    
    def make_decision(self,
                     context: MarketContext,
                     tournament_history: Optional[List[Dict]] = None) -> StrategyDecision:
        """
        Invert signals when consensus is too strong (potential overreaction).
        
        Logic:
        1. High consensus (>75%) + moderate confidence (<90%) → INVERT
        2. Extreme sentiment (>80% or <20%) → FADE
        3. Otherwise → FOLLOW
        """
        
        consensus = context.analyst_consensus
        sentiment = context.sentiment_score
        confidence = context.confidence
        base_action = context.base_decision.action
        
        action = base_action
        position_multiplier = 1.0
        reasoning_parts = []
        
        # Rule 1: Strong consensus but not extreme confidence = potential overreaction
        if consensus > 0.75 and confidence < 0.9:
            action = self._invert_action(base_action)
            position_multiplier = 0.6  # Conservative contrarian bet
            reasoning_parts.append(
                f"Contrarian: High consensus ({consensus:.1%}) suggests overreaction"
            )
        
        # Rule 2: Extreme sentiment = fade
        elif sentiment > 0.8 or sentiment < 0.2:
            action = self._invert_action(base_action)
            position_multiplier = 0.7
            reasoning_parts.append(
                f"Fading extreme sentiment ({sentiment:.1%})"
            )
        
        # Rule 3: No clear overreaction = follow
        else:
            reasoning_parts.append(
                "No clear overreaction, following market"
            )
        
        reasoning = "; ".join(reasoning_parts)
        
        return self.create_decision(
            action=action,
            position_multiplier=position_multiplier,
            confidence_multiplier=0.9,  # Contrarian is inherently uncertain
            reasoning=reasoning,
            context=context,
            metadata={
                'consensus': consensus,
                'sentiment': sentiment,
                'inverted': action != base_action
            }
        )
    
    def _invert_action(self, action: str) -> str:
        """
        Invert action: BUY→SELL, SELL→BUY, HOLD→HOLD
        """
        if action == 'BUY':
            return 'SELL'
        elif action == 'SELL':
            return 'BUY'
        return 'HOLD'