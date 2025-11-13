# agents/game_theory/strategy_interface.py

"""
Base interface for all game theory trading strategies.
All strategies must inherit from TradingStrategy and implement make_decision().

Example:
    class MyStrategy(TradingStrategy):
        def make_decision(self, context, tournament_history):
            # Your strategy logic here
            return self.create_decision(
                action=context.base_decision.action,
                position_multiplier=1.0,
                reasoning="My reasoning"
            )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
from .market_context import MarketContext


@dataclass
class StrategyDecision:
    """
    A trading decision made by a strategy.
    
    Attributes:
        strategy_name: Name of the strategy
        action: BUY, SELL, or HOLD
        position_size: Dollar amount
        confidence: 0.0 to 1.0
        reasoning: Explanation of the decision
        metadata: Additional strategy-specific data
    """
    strategy_name: str
    action: str
    position_size: float
    confidence: float
    reasoning: str
    metadata: Optional[Dict] = None
    
    def __str__(self):
        return (f"{self.strategy_name}: {self.action} ${self.position_size:,.0f} "
                f"(confidence: {self.confidence:.1%})")


class TradingStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Each strategy represents a different "personality" or approach to trading:
    - Actual Market: Trusts the base system
    - Buy & Hold: Patient long-term
    - Cooperator: Amplifies when consensus high
    - Defector: Contrarian when herd strong
    - Tit-for-Tat: Adapts to what works
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Human-readable strategy name
        """
        self.name = name
    
    @abstractmethod
    def make_decision(self, 
                     context: MarketContext,
                     tournament_history: Optional[List[Dict]] = None) -> StrategyDecision:
        """
        Make a trading decision based on market context.
        
        Args:
            context: Current market conditions and base decision
            tournament_history: Past tournament results (for adaptive strategies)
            
        Returns:
            StrategyDecision with action, position size, confidence, reasoning
        """
        pass
    
    # Helper methods that strategies can use
    
    def create_decision(self,
                       action: str,
                       position_multiplier: float,
                       confidence_multiplier: float,
                       reasoning: str,
                       context: MarketContext,
                       metadata: Optional[Dict] = None) -> StrategyDecision:
        """
        Helper to create a StrategyDecision.
        
        Args:
            action: BUY/SELL/HOLD
            position_multiplier: Multiply base position by this (e.g., 1.3 = 30% bigger)
            confidence_multiplier: Multiply base confidence by this
            reasoning: Why this decision was made
            context: Market context
            metadata: Additional data
            
        Returns:
            StrategyDecision
        """
        
        base = context.base_decision
        
        # Calculate position size
        if base.position_size_dollars:
            position_size = base.position_size_dollars * position_multiplier
        else:
            position_size = 0.0
        
        # Calculate confidence
        confidence = min(base.confidence * confidence_multiplier, 1.0)
        
        return StrategyDecision(
            strategy_name=self.name,
            action=action,
            position_size=position_size,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata
        )