"""
strategies/__init__.py - Strategy Package Initialization

Location: agents/game_theory/strategies/__init__.py

UPDATED: Buy-and-Hold is now a BENCHMARK, not a tournament participant.
Tournament strategies compete against each other, and results are compared
to the Buy-and-Hold benchmark.

Usage:
    # Get tournament strategies (4 strategies that compete)
    from game_theory.strategies import get_tournament_strategies
    strategies = get_tournament_strategies()
    
    # Get benchmark strategy (for comparison)
    from game_theory.strategies import get_benchmark_strategy
    benchmark = get_benchmark_strategy()
    
    # Get all strategies including benchmark
    from game_theory.strategies import get_all_strategies
    all_strategies = get_all_strategies()
"""

from .base import Strategy, StrategyResult
from .buy_hold import BuyHoldStrategy
from .signal_follower import SignalFollowerStrategy
from .cooperator import CooperatorStrategy
from .defector import DefectorStrategy
from .tit_for_tat import TitForTatStrategy


# Tournament strategies - these COMPETE for capital allocation
TOURNAMENT_STRATEGIES = [
    SignalFollowerStrategy,
    CooperatorStrategy,
    DefectorStrategy,
    TitForTatStrategy,
]

# Benchmark strategy - calculated separately for comparison
BENCHMARK_STRATEGY = BuyHoldStrategy

# All strategies (for backward compatibility)
ALL_STRATEGIES = [BuyHoldStrategy] + TOURNAMENT_STRATEGIES


# Strategy name to class mapping (flexible matching)
STRATEGY_MAP = {
    # Buy-and-Hold (Benchmark)
    "buy-and-hold": BuyHoldStrategy,
    "buy_and_hold": BuyHoldStrategy,
    "buyhold": BuyHoldStrategy,
    "bnh": BuyHoldStrategy,
    "benchmark": BuyHoldStrategy,
    
    # Signal Follower
    "signal follower": SignalFollowerStrategy,
    "signal_follower": SignalFollowerStrategy,
    "signalfollower": SignalFollowerStrategy,
    "signal": SignalFollowerStrategy,
    "llm": SignalFollowerStrategy,
    
    # Cooperator
    "cooperator": CooperatorStrategy,
    "coop": CooperatorStrategy,
    "cooperative": CooperatorStrategy,
    "follow": CooperatorStrategy,
    
    # Defector
    "defector": DefectorStrategy,
    "defect": DefectorStrategy,
    "contrarian": DefectorStrategy,
    "contrary": DefectorStrategy,
    
    # Tit-for-Tat
    "tit-for-tat": TitForTatStrategy,
    "tit_for_tat": TitForTatStrategy,
    "titfortat": TitForTatStrategy,
    "tft": TitForTatStrategy,
    "adaptive": TitForTatStrategy,
}


def get_tournament_strategies():
    """
    Get instances of tournament strategies (excludes benchmark).
    
    These are the strategies that compete for capital allocation.
    Buy-and-Hold is NOT included - it's tracked separately as benchmark.
    
    Returns:
        List of 4 strategy instances: SignalFollower, Cooperator, Defector, TitForTat
    """
    return [cls() for cls in TOURNAMENT_STRATEGIES]


def get_benchmark_strategy():
    """
    Get the benchmark strategy (Buy-and-Hold).
    
    This is calculated separately and used for comparison,
    not as a tournament participant.
    
    Returns:
        BuyHoldStrategy instance
    """
    return BENCHMARK_STRATEGY()


def get_all_strategies():
    """
    Get instances of ALL strategies including benchmark.
    
    For backward compatibility. Prefer get_tournament_strategies() + get_benchmark_strategy()
    
    Returns:
        List of all strategy instances
    """
    return [cls() for cls in ALL_STRATEGIES]


def get_strategy_by_name(name: str) -> Strategy:
    """
    Get a strategy instance by name.
    
    Args:
        name: Strategy name (case-insensitive, flexible matching)
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name not recognized
    """
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in STRATEGY_MAP:
        return STRATEGY_MAP[name_lower]()
    
    # Partial match
    for key, cls in STRATEGY_MAP.items():
        if name_lower in key or key in name_lower:
            return cls()
    
    # Not found
    available = sorted(set(STRATEGY_MAP.keys()))
    raise ValueError(
        f"Unknown strategy: '{name}'. "
        f"Available: {available}"
    )


def get_strategy_names() -> list:
    """Get list of all strategy names."""
    return [cls().name for cls in ALL_STRATEGIES]


def get_tournament_strategy_names() -> list:
    """Get list of tournament strategy names (excludes benchmark)."""
    return [cls().name for cls in TOURNAMENT_STRATEGIES]


# === Test ===
if __name__ == "__main__":
    print("Testing strategies package...")
    print("=" * 60)
    
    # Test get_tournament_strategies
    print("\nTournament strategies (compete for capital):")
    tournament = get_tournament_strategies()
    for s in tournament:
        print(f"  - {s.name}: {s.description}")
    
    # Test get_benchmark_strategy
    print("\nBenchmark strategy (for comparison):")
    benchmark = get_benchmark_strategy()
    print(f"  - {benchmark.name}: {benchmark.description}")
    
    # Test get_all_strategies
    print(f"\nAll strategies: {len(get_all_strategies())}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")