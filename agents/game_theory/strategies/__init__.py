"""
strategies/__init__.py - Strategy Package Initialization

Location: agents/game_theory/strategies/__init__.py

This file makes the strategies folder a Python package and provides
convenient imports for all strategy classes.

Usage:
    # Import individual strategies
    from game_theory.strategies import CooperatorStrategy
    from game_theory.strategies import DefectorStrategy
    
    # Import all strategies at once
    from game_theory.strategies import (
        ActualMarketStrategy,
        BuyHoldStrategy,
        CooperatorStrategy,
        DefectorStrategy,
        TitForTatStrategy
    )
    
    # Get all strategy classes as a list
    from game_theory.strategies import ALL_STRATEGIES
    strategies = [S() for S in ALL_STRATEGIES]
    
    # Get strategy by name
    from game_theory.strategies import get_strategy_by_name
    cooperator = get_strategy_by_name("Cooperator")
"""

from .actual_market import ActualMarketStrategy
from .buy_hold import BuyHoldStrategy
from .cooperator import CooperatorStrategy
from .defector import DefectorStrategy
from .tit_for_tat import TitForTatStrategy


# List of all strategy classes (for easy iteration)
ALL_STRATEGIES = [
    ActualMarketStrategy,
    BuyHoldStrategy,
    CooperatorStrategy,
    DefectorStrategy,
    TitForTatStrategy,
]

# Strategy name to class mapping
STRATEGY_MAP = {
    "actual_market": ActualMarketStrategy,
    "actual market": ActualMarketStrategy,
    "control": ActualMarketStrategy,
    "buy_hold": BuyHoldStrategy,
    "buy-and-hold": BuyHoldStrategy,
    "buyhold": BuyHoldStrategy,
    "patient": BuyHoldStrategy,
    "cooperator": CooperatorStrategy,
    "consensus": CooperatorStrategy,
    "defector": DefectorStrategy,
    "contrarian": DefectorStrategy,
    "tit_for_tat": TitForTatStrategy,
    "tit-for-tat": TitForTatStrategy,
    "titfortat": TitForTatStrategy,
    "adaptive": TitForTatStrategy,
}


def get_strategy_by_name(name: str):
    """
    Get a strategy instance by name.
    
    Args:
        name: Strategy name (case-insensitive, flexible matching)
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name not recognized
        
    Examples:
        >>> get_strategy_by_name("Cooperator")
        CooperatorStrategy(...)
        
        >>> get_strategy_by_name("tit-for-tat")
        TitForTatStrategy(...)
    """
    name_lower = name.lower().strip()
    
    if name_lower in STRATEGY_MAP:
        return STRATEGY_MAP[name_lower]()
    
    # Try partial matching
    for key, strategy_class in STRATEGY_MAP.items():
        if name_lower in key or key in name_lower:
            return strategy_class()
    
    available = list(set(STRATEGY_MAP.keys()))
    raise ValueError(
        f"Unknown strategy: '{name}'. "
        f"Available strategies: {sorted(available)}"
    )


def get_all_strategies():
    """
    Get instances of all available strategies.
    
    Returns:
        List of strategy instances, one of each type
        
    Example:
        >>> strategies = get_all_strategies()
        >>> for s in strategies:
        ...     print(s.name)
        Actual Market
        Buy-and-Hold
        Cooperator
        Defector
        Tit-for-Tat
    """
    return [S() for S in ALL_STRATEGIES]


def get_strategy_descriptions() -> dict:
    """
    Get descriptions of all strategies.
    
    Returns:
        Dictionary mapping strategy name to description
    """
    descriptions = {}
    for S in ALL_STRATEGIES:
        instance = S()
        descriptions[instance.name] = instance.description
    return descriptions


# Export list for "from strategies import *"
__all__ = [
    # Strategy classes
    'ActualMarketStrategy',
    'BuyHoldStrategy',
    'CooperatorStrategy',
    'DefectorStrategy',
    'TitForTatStrategy',
    # Lists and maps
    'ALL_STRATEGIES',
    'STRATEGY_MAP',
    # Helper functions
    'get_strategy_by_name',
    'get_all_strategies',
    'get_strategy_descriptions',
]