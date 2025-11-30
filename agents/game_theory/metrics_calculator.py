"""
metrics_calculator.py - Financial Performance Metrics

Location: agents/game_theory/metrics_calculator.py

This module calculates comprehensive financial metrics for strategies:
- Return metrics: Total return, annualized return
- Risk metrics: Volatility, max drawdown, VaR
- Risk-adjusted: Sharpe ratio, Sortino ratio, Calmar ratio
- Trade statistics: Win rate, profit factor, avg win/loss

These metrics help answer your research question by providing
standardized ways to compare strategy performance.

Usage:
    from game_theory.metrics_calculator import MetricsCalculator, StrategyMetrics
    
    calc = MetricsCalculator()
    metrics = calc.calculate(strategy)
    
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.1f}%")
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Avoid circular import
if TYPE_CHECKING:
    from .base_strategy import TradingStrategy


@dataclass
class StrategyMetrics:
    """
    Complete metrics for a strategy's performance.
    
    All percentage values are stored as percentages (e.g., 15.5 means 15.5%).
    Ratios are stored as raw values (e.g., 1.5 means 1.5).
    
    Attributes:
        name: Strategy name
        
        # Return Metrics
        total_return: Total cumulative return (%)
        annualized_return: Annualized return (%)
        
        # Risk Metrics
        volatility: Annualized volatility (%)
        max_drawdown: Maximum drawdown (%)
        
        # Risk-Adjusted Metrics
        sharpe_ratio: Risk-adjusted return (return / volatility)
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: Return / max drawdown
        
        # Trade Statistics
        win_rate: Percentage of winning trades (%)
        avg_win: Average winning trade return (%)
        avg_loss: Average losing trade return (%)
        profit_factor: Gross profit / gross loss
        
        # Position Metrics
        avg_position: Average position size (%)
        total_trades: Number of trades executed
        
        # Score
        final_score: Final score from -10 to +10
        
        # Regime Performance
        bull_return: Total return in bull markets (%)
        bear_return: Total return in bear markets (%)
        sideways_return: Total return in sideways markets (%)
    """
    name: str
    
    # Return Metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    
    # Risk Metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk-Adjusted Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade Statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Position Metrics
    avg_position: float = 0.0
    total_trades: int = 0
    
    # Score
    final_score: float = 0.0
    
    # Regime Performance
    bull_return: float = 0.0
    bear_return: float = 0.0
    sideways_return: float = 0.0


class MetricsCalculator:
    """
    Calculate comprehensive financial metrics for trading strategies.
    
    Handles edge cases like:
    - Empty return series
    - All positive/negative returns (for Sortino)
    - Zero drawdown (for Calmar)
    - Division by zero throughout
    
    Attributes:
        annualization_factor: Factor to annualize returns.
            Default 252/3 assumes ~3 samples per trading day equivalent.
            Adjust based on your data frequency.
    """
    
    def __init__(self, annualization_factor: float = 252 / 3):
        """
        Initialize MetricsCalculator.
        
        Args:
            annualization_factor: Factor for annualizing returns.
                Default 252/3 ≈ 84, assumes ~3 samples per trading day.
                For daily data, use 252.
                For weekly data, use 52.
        """
        self.annualization_factor = annualization_factor
    
    def calculate(self, strategy: 'TradingStrategy') -> Optional[StrategyMetrics]:
        """
        Calculate all metrics for a strategy.
        
        Args:
            strategy: TradingStrategy instance with trade history
            
        Returns:
            StrategyMetrics object, or None if insufficient data
        """
        # Check for sufficient data
        if not strategy.daily_returns or len(strategy.daily_returns) < 2:
            return None
        
        returns = np.array(strategy.daily_returns)
        equity = np.array(strategy.equity_curve)
        n_samples = len(returns)
        
        # === Return Metrics ===
        total_return = (equity[-1] - 1.0) * 100.0
        
        # Annualized return using compound growth
        if equity[-1] > 0:
            annualized_return = (
                (equity[-1] ** (self.annualization_factor / n_samples)) - 1.0
            ) * 100.0
        else:
            annualized_return = -100.0
        
        # === Risk Metrics ===
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(self.annualization_factor) * 100.0
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(equity)
        
        # === Risk-Adjusted Metrics ===
        # Sharpe Ratio (assuming 0 risk-free rate)
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (downside deviation)
        sortino_ratio = self._calculate_sortino(returns, annualized_return)
        
        # Calmar Ratio (return / max drawdown)
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = float('inf') if annualized_return > 0 else 0.0
        
        # === Trade Statistics ===
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        # Win rate
        win_rate = (len(wins) / n_samples * 100.0) if n_samples > 0 else 0.0
        
        # Average win/loss
        avg_win = (np.mean(wins) * 100.0) if len(wins) > 0 else 0.0
        avg_loss = (np.mean(losses) * 100.0) if len(losses) > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        # === Position Metrics ===
        avg_position = np.mean(strategy.positions) if strategy.positions else 0.0
        
        # === Regime Performance ===
        bull_return = self._calculate_regime_return(strategy.regime_returns.get('bull', []))
        bear_return = self._calculate_regime_return(strategy.regime_returns.get('bear', []))
        sideways_return = self._calculate_regime_return(strategy.regime_returns.get('sideways', []))
        
        return StrategyMetrics(
            name=strategy.name,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_position=avg_position,
            total_trades=n_samples,
            final_score=strategy.score,
            bull_return=bull_return,
            bear_return=bear_return,
            sideways_return=sideways_return
        )
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity: Equity curve array
            
        Returns:
            Maximum drawdown as percentage (positive value)
        """
        if len(equity) < 2:
            return 0.0
        
        # Running maximum
        peak = np.maximum.accumulate(equity)
        
        # Drawdown at each point
        drawdown = (equity - peak) / peak
        
        # Maximum drawdown (as positive percentage)
        max_dd = abs(np.min(drawdown)) * 100.0
        
        return max_dd
    
    def _calculate_sortino(self, returns: np.ndarray, annualized_return: float) -> float:
        """
        Calculate Sortino ratio using downside deviation.
        
        Args:
            returns: Array of returns
            annualized_return: Annualized return for numerator
            
        Returns:
            Sortino ratio
        """
        # Get only negative returns
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            # No negative returns - infinite Sortino if positive return
            return float('inf') if annualized_return > 0 else 0.0
        
        # Downside deviation (annualized)
        downside_std = np.std(downside_returns) * np.sqrt(self.annualization_factor) * 100.0
        
        if downside_std > 0:
            return annualized_return / downside_std
        else:
            return 0.0
    
    def _calculate_regime_return(self, returns: List[float]) -> float:
        """
        Calculate total return for a regime.
        
        Args:
            returns: List of returns in this regime
            
        Returns:
            Total return as percentage
        """
        if not returns:
            return 0.0
        
        # Cumulative return
        cumulative = 1.0
        for r in returns:
            cumulative *= (1.0 + r)
        
        return (cumulative - 1.0) * 100.0
    
    def to_dict(self, metrics: StrategyMetrics) -> Dict:
        """
        Convert StrategyMetrics to dictionary for JSON serialization.
        
        Handles special values like infinity.
        
        Args:
            metrics: StrategyMetrics object
            
        Returns:
            Dictionary with all metrics
        """
        def safe_value(v):
            """Convert infinity to string for JSON."""
            if isinstance(v, float):
                if np.isinf(v):
                    return "inf" if v > 0 else "-inf"
                if np.isnan(v):
                    return None
            return v
        
        return {
            'name': metrics.name,
            'total_return': round(metrics.total_return, 2),
            'annualized_return': round(metrics.annualized_return, 2),
            'volatility': round(metrics.volatility, 2),
            'max_drawdown': round(metrics.max_drawdown, 2),
            'sharpe_ratio': round(safe_value(metrics.sharpe_ratio), 3),
            'sortino_ratio': round(safe_value(metrics.sortino_ratio), 3) if not np.isinf(metrics.sortino_ratio) else 'inf',
            'calmar_ratio': round(safe_value(metrics.calmar_ratio), 2) if not np.isinf(metrics.calmar_ratio) else 'inf',
            'win_rate': round(metrics.win_rate, 1),
            'avg_win': round(metrics.avg_win, 3),
            'avg_loss': round(metrics.avg_loss, 3),
            'profit_factor': round(safe_value(metrics.profit_factor), 2) if not np.isinf(metrics.profit_factor) else 'inf',
            'avg_position': round(metrics.avg_position, 1),
            'total_trades': metrics.total_trades,
            'final_score': round(metrics.final_score, 1),
            'bull_return': round(metrics.bull_return, 2),
            'bear_return': round(metrics.bear_return, 2),
            'sideways_return': round(metrics.sideways_return, 2),
        }
    
    def compare_strategies(
        self, 
        metrics_list: List[StrategyMetrics]
    ) -> Dict[str, StrategyMetrics]:
        """
        Compare multiple strategies and rank them.
        
        Args:
            metrics_list: List of StrategyMetrics objects
            
        Returns:
            Dictionary with rankings by different criteria
        """
        if not metrics_list:
            return {}
        
        # Sort by different criteria
        by_return = sorted(metrics_list, key=lambda m: m.total_return, reverse=True)
        by_sharpe = sorted(metrics_list, key=lambda m: m.sharpe_ratio if not np.isinf(m.sharpe_ratio) else -999, reverse=True)
        by_drawdown = sorted(metrics_list, key=lambda m: m.max_drawdown)
        by_win_rate = sorted(metrics_list, key=lambda m: m.win_rate, reverse=True)
        
        return {
            'by_total_return': [m.name for m in by_return],
            'by_sharpe_ratio': [m.name for m in by_sharpe],
            'by_max_drawdown': [m.name for m in by_drawdown],
            'by_win_rate': [m.name for m in by_win_rate],
            'best_overall': by_return[0].name if by_return else None,
            'best_risk_adjusted': by_sharpe[0].name if by_sharpe else None,
            'lowest_risk': by_drawdown[0].name if by_drawdown else None,
        }


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing MetricsCalculator...")
    print("=" * 60)
    
    # Create a mock strategy with some data
    class MockStrategy:
        def __init__(self):
            self.name = "Test Strategy"
            self.daily_returns = [
                0.01, -0.005, 0.015, 0.008, -0.012,
                0.02, -0.008, 0.012, -0.003, 0.018,
                0.005, -0.015, 0.01, 0.007, -0.009,
                0.025, -0.01, 0.008, 0.012, -0.005,
            ]
            
            # Build equity curve
            self.equity_curve = [1.0]
            for r in self.daily_returns:
                self.equity_curve.append(self.equity_curve[-1] * (1 + r))
            
            self.positions = [50.0] * len(self.daily_returns)
            self.score = 3.5
            self.regime_returns = {
                'bull': [0.01, 0.015, 0.02, 0.018, 0.025, 0.012],
                'bear': [-0.005, -0.012, -0.008, -0.015, -0.01, -0.005],
                'sideways': [0.008, 0.012, -0.003, 0.005, 0.01, 0.007, -0.009, 0.008],
            }
    
    strategy = MockStrategy()
    calc = MetricsCalculator()
    
    metrics = calc.calculate(strategy)
    
    if metrics:
        print(f"\nStrategy: {metrics.name}")
        print("-" * 40)
        
        print("\nReturn Metrics:")
        print(f"  Total Return:      {metrics.total_return:+.2f}%")
        print(f"  Annualized Return: {metrics.annualized_return:+.2f}%")
        
        print("\nRisk Metrics:")
        print(f"  Volatility:        {metrics.volatility:.2f}%")
        print(f"  Max Drawdown:      {metrics.max_drawdown:.2f}%")
        
        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio:     {metrics.sortino_ratio:.3f}")
        print(f"  Calmar Ratio:      {metrics.calmar_ratio:.2f}")
        
        print("\nTrade Statistics:")
        print(f"  Win Rate:          {metrics.win_rate:.1f}%")
        print(f"  Avg Win:           {metrics.avg_win:+.3f}%")
        print(f"  Avg Loss:          {metrics.avg_loss:.3f}%")
        print(f"  Profit Factor:     {metrics.profit_factor:.2f}")
        
        print("\nRegime Performance:")
        print(f"  Bull Return:       {metrics.bull_return:+.2f}%")
        print(f"  Bear Return:       {metrics.bear_return:+.2f}%")
        print(f"  Sideways Return:   {metrics.sideways_return:+.2f}%")
        
        print("\nOther:")
        print(f"  Avg Position:      {metrics.avg_position:.1f}%")
        print(f"  Total Trades:      {metrics.total_trades}")
        print(f"  Final Score:       {metrics.final_score:+.1f}")
        
        # Test JSON conversion
        print("\nJSON conversion:")
        metrics_dict = calc.to_dict(metrics)
        print(f"  Keys: {list(metrics_dict.keys())}")
    
    print("\n" + "=" * 60)
    print("MetricsCalculator test complete!")