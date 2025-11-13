# agents/game_theory/tournament_engine.py

"""
Tournament Engine - Runs all 5 strategies in competition

Takes base decision + market context, runs all 5 strategies,
compares their decisions, determines optimal strategy.

Example usage:
    from game_theory.tournament_engine import TournamentEngine
    from game_theory.market_context import MarketContext, MarketContextBuilder
    
    engine = TournamentEngine()
    results = engine.run_tournament(context)
    
    print(f"Winner: {results['recommended_strategy']}")
    for name, decision in results['strategies'].items():
        print(f"{name}: {decision}")
"""

from typing import Dict, List, Optional
from datetime import datetime
from .strategies import (
    ActualMarketStrategy,
    BuyHoldStrategy,
    CooperatorStrategy,
    DefectorStrategy,
    TitForTatStrategy
)
from .market_context import MarketContext
from .strategy_interface import StrategyDecision


class TournamentEngine:
    """
    Runs tournament with all 5 trading strategies.
    Compares their decisions and selects optimal strategy.
    """
    
    def __init__(self):
        """Initialize with all 5 strategies"""
        self.strategies = {
            'actual_market': ActualMarketStrategy(),
            'buy_hold': BuyHoldStrategy(),
            'cooperator': CooperatorStrategy(),
            'defector': DefectorStrategy(),
            'tit_for_tat': TitForTatStrategy()
        }
        self.history: List[Dict] = []
    
    def run_tournament(self, context: MarketContext, symbol: str = "") -> Dict:
        """
        Run tournament with all 5 strategies.
        
        Args:
            context: Market context with base decision and regime
            symbol: Stock ticker (for logging)
            
        Returns:
            Dictionary with:
                - 'strategies': Dict of strategy_name → StrategyDecision
                - 'recommended_strategy': Name of optimal strategy
                - 'context': Original market context
                - 'symbol': Stock ticker
                - 'timestamp': When tournament ran
        """
        
        # Run each strategy
        strategy_decisions = {}
        for name, strategy in self.strategies.items():
            decision = strategy.make_decision(
                context=context,
                tournament_history=self.history
            )
            strategy_decisions[name] = decision
        
        # Determine recommended strategy (simple heuristic for now)
        recommended = self._select_strategy(context, strategy_decisions)
        
        # Create tournament result
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'strategies': strategy_decisions,
            'recommended_strategy': recommended,
            'winner': recommended  # Alias for compatibility
        }
        
        # Add to history
        self.history.append(result)
        
        return result
    
    def _select_strategy(self, 
                        context: MarketContext,
                        decisions: Dict[str, StrategyDecision]) -> str:
        """
        Select optimal strategy based on market conditions.
        
        This is a simple heuristic. In Phase 4 (meta-learning),
        this will be replaced with learned strategy selection.
        
        Rules:
        - High consensus (>75%) → Cooperator
        - Low consensus (<30%) → Defector
        - Sideways regime → Buy & Hold
        - Default → Actual Market
        """
        
        consensus = context.analyst_consensus
        regime = context.regime
        
        # High consensus → Cooperator amplifies
        if consensus > 0.75:
            return 'cooperator'
        
        # Low consensus → Defector contrarian
        elif consensus < 0.30:
            return 'defector'
        
        # Sideways market → Buy & Hold avoids overtrading
        elif regime == 'sideways':
            return 'buy_hold'
        
        # Default: Use actual market decision
        else:
            return 'actual_market'
    
    def get_history(self) -> List[Dict]:
        """Get tournament history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear tournament history"""
        self.history.clear()
    
    def print_results(self, result: Dict):
        """
        Pretty print tournament results.
        
        Args:
            result: Tournament result from run_tournament()
        """
        print("\n" + "="*60)
        print(f"🏆 TOURNAMENT RESULTS: {result['symbol']}")
        print("="*60)
        
        context = result['context']
        print(f"\n📊 Market Context:")
        print(f"  Regime: {context.regime}")
        print(f"  Consensus: {context.analyst_consensus:.1%}")
        print(f"  Sentiment: {context.sentiment_score:.1%}")
        print(f"  Base Action: {context.base_decision.action}")
        print(f"  Base Confidence: {context.confidence:.1%}")
        
        print(f"\n🎯 Strategy Decisions:")
        for name, decision in result['strategies'].items():
            marker = "✅" if name == result['recommended_strategy'] else "  "
            print(f"{marker} {name.upper()}:")
            print(f"      Action: {decision.action}")
            print(f"      Position: ${decision.position_size:,.0f}")
            print(f"      Confidence: {decision.confidence:.1%}")
            print(f"      Reasoning: {decision.reasoning}")
        
        winner = result['recommended_strategy']
        winner_decision = result['strategies'][winner]
        print(f"\n⭐ RECOMMENDED: {winner.upper()}")
        print(f"   Final Action: {winner_decision.action}")
        print(f"   Final Position: ${winner_decision.position_size:,.0f}")
        print(f"   Final Confidence: {winner_decision.confidence:.1%}")
        print("\n" + "="*60 + "\n")