"""
Game Theory Trading Tournament System

Location: agents/game_theory/__init__.py

A comprehensive system for running game theory strategy competitions
on trading data from your multi-agent AI trading system.

This package includes:
- 5 trading strategies with score-based adaptation
- Financial metrics calculation (Sharpe, Sortino, Drawdown, etc.)
- Monte Carlo simulations for robustness testing
- Animated visualizations and static charts
- Complete tournament engine

Research Question:
    "Which game theory strategy performs best in different market regimes?"

Quick Start:
    from game_theory import TournamentEngine
    
    engine = TournamentEngine()
    engine.run_ticker("AAPL")
    # or
    engine.run_all_tickers()

CLI Usage:
    python -m agents.game_theory.run_analysis --ticker AAPL
    python -m agents.game_theory.run_analysis --all
    python -m agents.game_theory.run_analysis --list

Package Structure:
    game_theory/
    ├── __init__.py              # This file
    ├── market_context.py        # MarketContext dataclass
    ├── data_loader.py           # Load collected workflow data
    ├── regime_detector.py       # Classify bull/bear/sideways
    ├── base_strategy.py         # Abstract base with score system
    ├── metrics_calculator.py    # Financial metrics
    ├── monte_carlo_engine.py    # Bootstrap simulations
    ├── visualization_engine.py  # Charts and GIFs
    ├── tournament_engine.py     # Main orchestrator
    ├── run_analysis.py          # CLI entry point
    └── strategies/
        ├── __init__.py
        ├── actual_market.py     # Control group
        ├── buy_hold.py          # Patient investor
        ├── cooperator.py        # Consensus follower
        ├── defector.py          # Contrarian
        └── tit_for_tat.py       # Adaptive learner

Strategies Overview:
    1. Actual Market (Control)
       - Faithfully executes system recommendation
       - Score tracked but ignored
       
    2. Buy-and-Hold (Patient Investor)
       - Only acts on high confidence signals
       - Score adjusts confidence threshold
       
    3. Cooperator (Consensus Follower)
       - Scales position with agent agreement
       - Score multiplies position size
       
    4. Defector (Contrarian)
       - Fades crowd at extremes
       - Score determines contrarian threshold
       
    5. Tit-for-Tat (Adaptive Learner)
       - Mirrors last round's winning approach
       - Score determines adaptation strength

Score System:
    Each strategy tracks a score from -10 to +10:
    
    | Market | Position    | Outcome      | Points |
    |--------|-------------|--------------|--------|
    | UP     | Aggressive  | Correct      | +3     |
    | UP     | Conservative| Missed gains | -1     |
    | DOWN   | Conservative| Avoided loss | +2     |
    | DOWN   | Aggressive  | Wrong        | -3     |

Author: Priyam (DAMG 7374 GenAI Course Project)
"""

# Core data structures
from .market_context import MarketContext
from .data_loader import DataLoader, load_ticker

# Regime detection
from .regime_detector import RegimeDetector

# Strategy base and implementations
from .base_strategy import TradingStrategy, TradeResult
from .strategies import (
    ActualMarketStrategy,
    BuyHoldStrategy,
    CooperatorStrategy,
    DefectorStrategy,
    TitForTatStrategy,
    get_all_strategies,
    get_strategy_by_name,
    ALL_STRATEGIES
)

# Metrics and analysis
from .metrics_calculator import MetricsCalculator, StrategyMetrics
from .monte_carlo_engine import MonteCarloEngine, MonteCarloResult

# Visualization
from .visualization_engine import VisualizationEngine

# Main engine
from .tournament_engine import TournamentEngine


# Package metadata
__version__ = "1.0.0"
__author__ = "Priyam"
__description__ = "Game Theory Trading Tournament System"


# Convenience function for quick analysis
def run_quick_analysis(ticker: str, monte_carlo: bool = False):
    """
    Run a quick analysis on a single ticker.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        monte_carlo: Whether to run Monte Carlo (slower)
        
    Returns:
        Dict of strategy metrics
        
    Example:
        from game_theory import run_quick_analysis
        metrics = run_quick_analysis("AAPL")
    """
    engine = TournamentEngine()
    return engine.run_ticker(
        ticker, 
        run_monte_carlo=monte_carlo,
        generate_gifs=False
    )


# Export list
__all__ = [
    # Core
    'MarketContext',
    'DataLoader',
    'load_ticker',
    'RegimeDetector',
    
    # Strategies
    'TradingStrategy',
    'TradeResult',
    'ActualMarketStrategy',
    'BuyHoldStrategy',
    'CooperatorStrategy',
    'DefectorStrategy',
    'TitForTatStrategy',
    'get_all_strategies',
    'get_strategy_by_name',
    'ALL_STRATEGIES',
    
    # Analysis
    'MetricsCalculator',
    'StrategyMetrics',
    'MonteCarloEngine',
    'MonteCarloResult',
    
    # Visualization
    'VisualizationEngine',
    
    # Main engine
    'TournamentEngine',
    
    # Convenience
    'run_quick_analysis',
    
    # Metadata
    '__version__',
    '__author__',
]