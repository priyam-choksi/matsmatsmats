# agents/game_theory/strategies/tit_for_tat.py

"""
Tit-for-Tat Strategy - The Adaptive Learner

Personality: "I'll trust you until you burn me, then I adapt"

Philosophy:
- Mirrors the PREVIOUS WINNING STRATEGY from tournament history
- Starts cooperative (Cooperator strategy)
- Learns which personality works in current conditions
- Evolves based on what's actually working

Game Theory Role: Evolution - learns which personality works in current regime
"""

from typing import Optional, List, Dict
from ..strategy_interface import TradingStrategy, StrategyDecision
from ..market_context import MarketContext


class TitForTatStrategy(TradingStrategy):
    """
    Adaptive strategy that mirrors the previous tournament winner.
    Starts cooperative, then learns from results.
    """
    
    def __init__(self):
        super().__init__(name="Tit-for-Tat")
    
    def make_decision(self,
                     context: MarketContext,
                     tournament_history: Optional[List[Dict]] = None) -> StrategyDecision:
        """
        Mirror the strategy that won the last tournament.
        
        Logic:
        1. First round (no history) → Use Cooperator strategy
        2. After round 1 → Mirror whatever strategy won last time
        3. Apply that strategy's logic to current context
        """
        
        # First round: Start cooperative
        if not tournament_history or len(tournament_history) == 0:
            return self._use_cooperator_logic(context, is_first_round=True)
        
        # Get last winner
        last_result = tournament_history[-1]
        last_winner = last_result.get('winner', 'actual_market')
        
        # Mirror that strategy's logic
        if last_winner == 'cooperator':
            decision = self._use_cooperator_logic(context)
        elif last_winner == 'defector':
            decision = self._use_defector_logic(context)
        elif last_winner == 'buy_hold':
            decision = self._use_buy_hold_logic(context)
        else:  # actual_market
            decision = self._use_actual_market_logic(context)
        
        # Update reasoning to show we're mirroring
        decision.reasoning = f"Mirroring '{last_winner}': {decision.reasoning}"
        decision.strategy_name = self.name  # But keep our name
        
        decision.metadata = decision.metadata or {}
        decision.metadata['mirroring'] = last_winner
        
        return decision
    
    # ===== Mirror Logic Methods ===== #
    
    def _use_cooperator_logic(self, context: MarketContext, is_first_round: bool = False) -> StrategyDecision:
        """Apply Cooperator's logic"""
        alignment = context.alignment_score()
        consensus = context.analyst_consensus
        
        if alignment > 0.5 and consensus > 0.7:
            position_multiplier = 1.0 + (alignment * 0.5)
            confidence_multiplier = 1.15
            reasoning = "Amplifying (Cooperator logic): High alignment + consensus"
        elif alignment < 0:
            position_multiplier = 0.7
            confidence_multiplier = 0.9
            reasoning = "Reducing (Cooperator logic): Misalignment detected"
        else:
            position_multiplier = 1.0
            confidence_multiplier = 1.0
            reasoning = "Neutral (Cooperator logic): Following base"
        
        if is_first_round:
            reasoning = "First round: Starting cooperative " + reasoning
        
        return self.create_decision(
            action=context.base_decision.action,
            position_multiplier=position_multiplier,
            confidence_multiplier=confidence_multiplier,
            reasoning=reasoning,
            context=context
        )
    
    def _use_defector_logic(self, context: MarketContext) -> StrategyDecision:
        """Apply Defector's logic"""
        consensus = context.analyst_consensus
        sentiment = context.sentiment_score
        base_action = context.base_decision.action
        
        # Invert if strong consensus
        if consensus > 0.75 and context.confidence < 0.9:
            action = self._invert_action(base_action)
            position_multiplier = 0.6
            reasoning = "Inverting (Defector logic): High consensus"
        elif sentiment > 0.8 or sentiment < 0.2:
            action = self._invert_action(base_action)
            position_multiplier = 0.7
            reasoning = "Fading (Defector logic): Extreme sentiment"
        else:
            action = base_action
            position_multiplier = 1.0
            reasoning = "Following (Defector logic): No overreaction"
        
        return self.create_decision(
            action=action,
            position_multiplier=position_multiplier,
            confidence_multiplier=0.9,
            reasoning=reasoning,
            context=context
        )
    
    def _use_buy_hold_logic(self, context: MarketContext) -> StrategyDecision:
        """Apply Buy & Hold logic"""
        base = context.base_decision
        action = base.action
        
        # Only act on high confidence
        if base.confidence < 0.80:
            action = 'HOLD'
            reasoning = "Holding (Buy&Hold logic): Confidence too low"
        elif action == 'SELL' and context.regime in ['bull_trend', 'momentum']:
            action = 'HOLD'
            reasoning = "Holding (Buy&Hold logic): Don't sell in bull market"
        else:
            reasoning = "Acting (Buy&Hold logic): Strong signal"
        
        return self.create_decision(
            action=action,
            position_multiplier=1.0,
            confidence_multiplier=1.0,
            reasoning=reasoning,
            context=context
        )
    
    def _use_actual_market_logic(self, context: MarketContext) -> StrategyDecision:
        """Apply Actual Market logic (just follow base)"""
        return self.create_decision(
            action=context.base_decision.action,
            position_multiplier=1.0,
            confidence_multiplier=1.0,
            reasoning="Following (Actual Market logic): Trust base system",
            context=context
        )
    
    def _invert_action(self, action: str) -> str:
        """Invert action for Defector logic"""
        if action == 'BUY':
            return 'SELL'
        elif action == 'SELL':
            return 'BUY'
        return 'HOLD'