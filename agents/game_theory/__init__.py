"""
game_theory/__init__.py - Game Theory Package Initialization

Location: agents/game_theory/__init__.py

Capital Allocation Tournament System
=====================================

This package implements a game theory tournament where trading strategies
compete for capital allocation. Strategies that outperform gain capital
from underperformers, creating actual game dynamics.

Main Components:
    - GTEngine: Main tournament engine
    - GameState: Tracks allocations, positions, history
    - Strategy classes: BuyHold, SignalFollower, Cooperator, Defector, TitForTat
    - DataLoader: Loads collected workflow data
    - MarketContext: Market data structure

Quick Start:
    from game_theory import GTEngine
    
    engine = GTEngine()
    engine.run_ticker("AAPL")
    # or
    engine.run_all_tickers()

CLI Usage:
    python -m agents.game_theory.run_analysis --ticker AAPL
    python -m agents.game_theory.run_analysis --all
    python -m agents.game_theory.run_analysis --list
"""

# Core components
from .market_context import MarketContext
from .game_state import GameState, RoundResult
from .data_loader import DataLoader

# Main engine
from .gt_engine import GTEngine

# Strategies
from .strategies import (
    Strategy,
    BuyHoldStrategy,
    SignalFollowerStrategy,
    CooperatorStrategy,
    DefectorStrategy,
    TitForTatStrategy,
    get_all_strategies,
    get_strategy_by_name,
    ALL_STRATEGIES
)

# Supporting modules (optional imports - may not all exist yet)
try:
    from .metrics_calculator import MetricsCalculator, StrategyMetrics
except ImportError:
    pass

try:
    from .monte_carlo_engine import MonteCarloEngine, MonteCarloResult
except ImportError:
    pass

try:
    from .visualization_engine import VisualizationEngine
except ImportError:
    pass

try:
    from .regime_detector import RegimeDetector
except ImportError:
    pass


# Package metadata
__version__ = "2.0.0"
__author__ = "Priyam"
__description__ = "Game Theory Capital Allocation Tournament"


# What gets imported with "from game_theory import *"
__all__ = [
    # Core
    "MarketContext",
    "GameState",
    "RoundResult",
    "DataLoader",
    
    # Engine
    "GTEngine",
    
    # Strategies
    "Strategy",
    "BuyHoldStrategy",
    "SignalFollowerStrategy", 
    "CooperatorStrategy",
    "DefectorStrategy",
    "TitForTatStrategy",
    "get_all_strategies",
    "get_strategy_by_name",
    "ALL_STRATEGIES",
]