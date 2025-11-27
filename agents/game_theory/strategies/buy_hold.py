from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from ..strategy_interface import TradingStrategy, StrategyDecision
from ..market_context import MarketContext


@dataclass
class TradeResult:
    """Result of a single trade for score calculation"""
    actual_return: float          # What the market did (%)
    strategy_return: float        # What our strategy made (%)
    was_correct: bool             # Did we profit?
    position_was_aggressive: bool # Was position > 50%?
    position_pct: float           # Actual position taken
    regime: str                   # Market regime during trade


class BuyHoldStrategy(TradingStrategy):
    """
    Buy & Hold: The Eternal Optimist / Market Benchmark
    
    "Time in the market beats timing the market."
    
    This strategy:
    1. ALWAYS takes 100% position
    2. NEVER reduces based on signals
    3. NEVER increases based on signals
    4. Represents passive investing
    5. Is THE benchmark to beat
    
    Score System (tracked but doesn't affect behavior):
    - Score represents how well "stay invested" has worked recently
    - High score (+5 to +10): Market trending up, Buy & Hold winning
    - Low score (-5 to -10): Market choppy/down, Buy & Hold struggling
    - But we NEVER change behavior based on score
    """
    
    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    
    def __init__(self, portfolio_value: float = 100000):
        """
        Initialize Buy & Hold strategy.
        
        Args:
            portfolio_value: Starting portfolio value (default $100k)
        """
        super().__init__(name="Buy-and-Hold")
        
        # Portfolio tracking
        self.portfolio_value = portfolio_value
        self.initial_portfolio = portfolio_value
        
        # Score system (-10 to +10)
        self.score = 0.0
        self.score_history: List[float] = [0.0]
        
        # Performance tracking
        self.returns_history: List[float] = []
        self.equity_curve: List[float] = [portfolio_value]
        self.decisions_history: List[Dict] = []
        self.trade_results: List[TradeResult] = []
        
        # Buy & Hold ALWAYS takes 100% position
        self.position_pct = 100.0
        
        # Regime performance tracking (for analysis, not behavior)
        self.regime_performance: Dict[str, List[float]] = {
            'bull': [],
            'bear': [],
            'sideways': [],
            'volatile': []
        }
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = portfolio_value
    
    # ==========================================================================
    # SCORE INTERPRETATION (Tracked but doesn't change behavior)
    # ==========================================================================
    
    def interpret_score(self) -> str:
        """
        Interpret current score - FOR DISPLAY ONLY.
        Buy & Hold NEVER changes behavior based on score.
        
        Returns:
            Human-readable interpretation with emoji
        """
        if self.score >= 8:
            return "📈 Market strongly trending up - Buy & Hold thriving!"
        elif self.score >= 5:
            return "📈 Market trending up - staying invested pays off"
        elif self.score >= 2:
            return "📊 Market slightly positive - steady as she goes"
        elif self.score >= -2:
            return "📊 Market flat - patience is a virtue"
        elif self.score >= -5:
            return "📉 Market struggling - but we hold through adversity"
        elif self.score >= -8:
            return "📉 Market declining - true test of conviction"
        else:
            return "🔥 Market in freefall - diamond hands activated"
    
    def get_position_bias(self) -> float:
        """
        Get position bias based on score.
        
        FOR BUY & HOLD: ALWAYS RETURNS 1.0
        We NEVER adjust position based on score - that's the whole point.
        
        Returns:
            1.0 always (no adjustment)
        """
        return 1.0  # NEVER changes
    
    # ==========================================================================
    # MAIN DECISION METHOD
    # ==========================================================================
    
    def make_decision(
        self,
        context: MarketContext,
        tournament_history: Optional[List[Dict]] = None
    ) -> StrategyDecision:
        """
        Make trading decision - ALWAYS 100% invested.
        
        This is the core of Buy & Hold:
        - We don't care about consensus
        - We don't care about sentiment
        - We don't care about regime
        - We don't care about score
        - We ALWAYS take 100% position
        
        Args:
            context: Market context with base decision and signals
            tournament_history: Past results (ignored by Buy & Hold)
        
        Returns:
            StrategyDecision with 100% BUY position
        """
        
        # Build reasoning (for display/logging)
        reasoning_parts = []
        reasoning_parts.append("Buy & Hold: Always 100% invested")
        reasoning_parts.append(f"Current score: {self.score:+.1f} ({self.interpret_score()})")
        reasoning_parts.append(f"Market regime: {context.regime} (ignored)")
        reasoning_parts.append(f"Consensus: {context.analyst_consensus:.0%} (ignored)")
        reasoning_parts.append("This IS the benchmark - beat me or go home")
        
        reasoning = " | ".join(reasoning_parts)
        
        # Calculate 100% position size
        position_size = self.portfolio_value  # 100% of portfolio
        
        # Track this decision
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'action': 'BUY',
            'position_pct': 100.0,
            'position_size': position_size,
            'score': self.score,
            'regime': context.regime,
            'consensus': context.analyst_consensus,
            'reasoning': reasoning
        }
        self.decisions_history.append(decision_record)
        
        # Create and return decision
        return StrategyDecision(
            strategy_name=self.name,
            action='BUY',  # ALWAYS BUY
            position_size=position_size,  # 100% of portfolio
            confidence=1.0,  # Maximum confidence in staying invested
            reasoning=reasoning,
            metadata={
                'position_pct': 100.0,
                'score': self.score,
                'score_interpretation': self.interpret_score(),
                'regime': context.regime,
                'regime_ignored': True,
                'consensus_ignored': True,
                'strategy_type': 'benchmark',
                'is_passive': True
            }
        )
    
    # ==========================================================================
    # SCORE UPDATE SYSTEM
    # ==========================================================================
    
    def update_score(self, result: TradeResult) -> float:
        """
        Update score based on trade result.
        
        Score System:
        - Market up + we're 100% invested = good (+2 to +3)
        - Market down + we're 100% invested = bad (-2 to -3)
        
        This tracks whether "stay invested" is working, but
        we NEVER change behavior based on it.
        
        Args:
            result: TradeResult with actual returns
            
        Returns:
            New score value
        """
        
        old_score = self.score
        
        # Since we're ALWAYS 100% invested (aggressive)
        if result.actual_return > 0:
            # Market went up - we captured it
            if result.actual_return > 2.0:
                delta = +3  # Big up day
            elif result.actual_return > 0.5:
                delta = +2  # Normal up day
            else:
                delta = +1  # Small up day
        else:
            # Market went down - we took the hit
            if result.actual_return < -2.0:
                delta = -3  # Big down day
            elif result.actual_return < -0.5:
                delta = -2  # Normal down day
            else:
                delta = -1  # Small down day
        
        # Apply delta with bounds
        self.score = max(-10, min(10, self.score + delta))
        self.score_history.append(self.score)
        
        return self.score
    
    # ==========================================================================
    # PERFORMANCE TRACKING
    # ==========================================================================
    
    def update_performance(self, actual_return: float, regime: str = 'unknown') -> Dict:
        """
        Update performance metrics after a trade.
        
        Args:
            actual_return: What the market actually did (%)
            regime: Market regime during this period
            
        Returns:
            Dictionary with performance update details
        """
        
        # Since we're 100% invested, our return = market return
        strategy_return = actual_return * (self.position_pct / 100.0)
        
        # Create trade result
        result = TradeResult(
            actual_return=actual_return,
            strategy_return=strategy_return,
            was_correct=(actual_return > 0),
            position_was_aggressive=True,  # 100% is always aggressive
            position_pct=100.0,
            regime=regime
        )
        self.trade_results.append(result)
        
        # Update score
        self.update_score(result)
        
        # Update returns history
        self.returns_history.append(strategy_return)
        self.total_return = (1 + self.total_return / 100) * (1 + strategy_return / 100) - 1
        self.total_return *= 100
        
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + strategy_return / 100)
        self.equity_curve.append(new_equity)
        self.portfolio_value = new_equity
        
        # Update peak and drawdown
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        current_drawdown = (self.peak_equity - new_equity) / self.peak_equity * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Update regime performance
        if regime in self.regime_performance:
            self.regime_performance[regime].append(strategy_return)
        
        # Update trade counts
        self.total_trades += 1
        if strategy_return > 0:
            self.winning_trades += 1
        
        return {
            'strategy_return': strategy_return,
            'new_equity': new_equity,
            'new_score': self.score,
            'total_return': self.total_return,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        }
    
    # ==========================================================================
    # ANALYTICS METHODS
    # ==========================================================================
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics for this strategy.
        
        Returns:
            Dictionary with all performance metrics
        """
        
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        # Calculate Sharpe Ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate regime-specific returns
        regime_stats = {}
        for regime, rets in self.regime_performance.items():
            if rets:
                regime_stats[regime] = {
                    'count': len(rets),
                    'avg_return': np.mean(rets),
                    'total_return': sum(rets),
                    'win_rate': sum(1 for r in rets if r > 0) / len(rets)
                }
            else:
                regime_stats[regime] = {'count': 0, 'avg_return': 0, 'total_return': 0, 'win_rate': 0}
        
        return {
            'strategy_name': self.name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_return': self.total_return,
            'avg_return': np.mean(returns) if len(returns) > 0 else 0,
            'std_return': np.std(returns) if len(returns) > 1 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown,
            'current_equity': self.portfolio_value,
            'current_score': self.score,
            'avg_position': 100.0,  # Always 100%
            'regime_performance': regime_stats,
            'is_benchmark': True
        }
    
    def get_equity_curve(self) -> List[float]:
        """Get the equity curve for charting."""
        return self.equity_curve.copy()
    
    def get_score_history(self) -> List[float]:
        """Get score history for charting."""
        return self.score_history.copy()
    
    # ==========================================================================
    # RESET AND UTILITY METHODS
    # ==========================================================================
    
    def reset(self, portfolio_value: Optional[float] = None):
        """
        Reset strategy to initial state.
        
        Args:
            portfolio_value: New starting portfolio (or use initial)
        """
        if portfolio_value:
            self.portfolio_value = portfolio_value
            self.initial_portfolio = portfolio_value
        else:
            self.portfolio_value = self.initial_portfolio
        
        self.score = 0.0
        self.score_history = [0.0]
        self.returns_history = []
        self.equity_curve = [self.portfolio_value]
        self.decisions_history = []
        self.trade_results = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.portfolio_value
        
        for regime in self.regime_performance:
            self.regime_performance[regime] = []
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"BuyHoldStrategy(\n"
            f"  name='{self.name}',\n"
            f"  position=100% (always),\n"
            f"  score={self.score:+.1f},\n"
            f"  total_return={self.total_return:+.2f}%,\n"
            f"  win_rate={self.winning_trades}/{self.total_trades},\n"
            f"  interpretation='{self.interpret_score()}'\n"
            f")"
        )
    
    def __repr__(self) -> str:
        """Repr representation."""
        return f"BuyHoldStrategy(score={self.score:+.1f}, equity=${self.portfolio_value:,.0f})"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_buy_hold_strategy(portfolio_value: float = 100000) -> BuyHoldStrategy:
    """
    Factory function to create a Buy & Hold strategy.
    
    Args:
        portfolio_value: Starting portfolio value
        
    Returns:
        Configured BuyHoldStrategy instance
    """
    return BuyHoldStrategy(portfolio_value=portfolio_value)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BUY & HOLD STRATEGY TEST")
    print("=" * 70)
    
    # Create strategy
    strategy = BuyHoldStrategy(portfolio_value=100000)
    print(f"\nInitialized: {strategy}")
    
    # Simulate some market moves
    market_moves = [
        (2.5, 'bull'),    # Market up 2.5%
        (1.2, 'bull'),    # Market up 1.2%
        (-0.8, 'sideways'),  # Market down 0.8%
        (-2.1, 'bear'),   # Market down 2.1%
        (3.0, 'bull'),    # Market up 3.0%
        (-1.5, 'volatile'),  # Market down 1.5%
        (0.5, 'sideways'),   # Market up 0.5%
    ]
    
    print("\nSimulating market moves:")
    print("-" * 50)
    
    for i, (move, regime) in enumerate(market_moves, 1):
        result = strategy.update_performance(move, regime)
        print(f"Day {i}: Market {move:+.1f}% ({regime})")
        print(f"        Strategy: {result['strategy_return']:+.2f}% | "
              f"Score: {result['new_score']:+.1f} | "
              f"Equity: ${result['new_equity']:,.0f}")
    
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    
    stats = strategy.get_statistics()
    print(f"Total Return: {stats['total_return']:+.2f}%")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"Final Score: {stats['current_score']:+.1f}")
    print(f"Score Interpretation: {strategy.interpret_score()}")
    
    print("\nRegime Performance:")
    for regime, data in stats['regime_performance'].items():
        if data['count'] > 0:
            print(f"  {regime}: {data['count']} trades, "
                  f"avg {data['avg_return']:+.2f}%, "
                  f"win rate {data['win_rate']:.0%}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)