"""
game_state.py - Game State Tracking for Capital Allocation Tournament

Location: agents/game_theory/game_state.py

This module tracks the state of the capital allocation game:
- How much capital each strategy has
- What positions each strategy took
- Who won each round
- Full history for analysis

The GameState is passed to each strategy so they can see what others are doing
and make strategic decisions based on the competitive landscape.

Usage:
    from game_theory.game_state import GameState
    
    # Initialize game
    game = GameState(
        strategy_names=["Buy-and-Hold", "Cooperator", "Defector", "Tit-for-Tat", "Signal Follower"],
        total_capital=1_000_000
    )
    
    # After each round, update state
    game.update_round(
        positions={"Cooperator": 45.0, "Defector": 80.0, ...},
        returns={"Cooperator": 0.012, "Defector": -0.005, ...},
        market_return=0.015
    )
    
    # Strategies can see game state
    print(game.last_positions)  # What others did
    print(game.allocations)     # Current capital distribution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


@dataclass
class RoundResult:
    """Result of a single round in the tournament."""
    round_num: int
    date: str
    regime: str
    market_return: float
    
    # What each strategy did
    positions: Dict[str, float]
    
    # What each strategy earned (dollar amount)
    dollar_returns: Dict[str, float]
    
    # What each strategy earned (percentage of their allocation)
    pct_returns: Dict[str, float]
    
    # Allocations AFTER reallocation
    allocations_after: Dict[str, float]
    
    # Who won this round
    winner: str
    
    # Was this a "defection" scenario? (high position variance)
    high_variance_round: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "round_num": self.round_num,
            "date": self.date,
            "regime": self.regime,
            "market_return": round(self.market_return, 6),
            "market_return_pct": round(self.market_return * 100, 4),
            "positions": {k: round(v, 2) for k, v in self.positions.items()},
            "dollar_returns": {k: round(v, 2) for k, v in self.dollar_returns.items()},
            "pct_returns": {k: round(v, 6) for k, v in self.pct_returns.items()},
            "pct_returns_display": {k: round(v * 100, 4) for k, v in self.pct_returns.items()},
            "allocations_after": {k: round(v, 2) for k, v in self.allocations_after.items()},
            "winner": self.winner,
            "high_variance_round": self.high_variance_round
        }


@dataclass 
class GameState:
    """
    Complete state of the capital allocation game.
    
    This is passed to strategies so they can make decisions based on:
    - What other strategies did last round
    - Current capital distribution
    - Who's winning/losing
    - Historical patterns
    
    Attributes:
        strategy_names: List of strategy names in the game
        total_capital: Total capital in the game (always sums to this)
        reallocation_rate: How aggressively capital shifts (0.1 = 10% per std dev)
        
        round_num: Current round number (0 = before first round)
        allocations: Current capital per strategy
        last_positions: Positions taken last round (0-100%)
        last_returns: Returns earned last round (as decimal)
        last_winner: Strategy that won last round
        
        allocation_history: Full history of allocations
        position_history: Full history of positions
        return_history: Full history of returns
        rounds: List of RoundResult objects
    """
    strategy_names: List[str]
    total_capital: float = 1_000_000
    reallocation_rate: float = 0.10  # 10% shift per std dev
    
    # Current state
    round_num: int = 0
    allocations: Dict[str, float] = field(default_factory=dict)
    last_positions: Dict[str, float] = field(default_factory=dict)
    last_returns: Dict[str, float] = field(default_factory=dict)
    last_winner: Optional[str] = None
    
    # History
    allocation_history: List[Dict[str, float]] = field(default_factory=list)
    position_history: List[Dict[str, float]] = field(default_factory=list)
    return_history: List[Dict[str, float]] = field(default_factory=list)
    rounds: List[RoundResult] = field(default_factory=list)
    
    # Metadata
    ticker: str = ""
    start_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize allocations equally if not set."""
        if not self.allocations:
            initial = self.total_capital / len(self.strategy_names)
            self.allocations = {name: initial for name in self.strategy_names}
        
        # Record initial allocation
        if not self.allocation_history:
            self.allocation_history.append(dict(self.allocations))
        
        if not self.start_time:
            self.start_time = datetime.now()
    
    def get_allocation_pct(self, strategy_name: str) -> float:
        """Get strategy's allocation as percentage of total."""
        return (self.allocations.get(strategy_name, 0) / self.total_capital) * 100
    
    def get_others_positions(self, strategy_name: str) -> Dict[str, float]:
        """Get positions of all OTHER strategies (for decision making)."""
        return {k: v for k, v in self.last_positions.items() if k != strategy_name}
    
    def get_others_avg_position(self, strategy_name: str) -> float:
        """Get average position of other strategies."""
        others = self.get_others_positions(strategy_name)
        if not others:
            return 50.0  # Default to neutral
        return np.mean(list(others.values()))
    
    def get_winner_streak(self) -> tuple:
        """Get current winner and their streak length."""
        if not self.rounds:
            return None, 0
        
        current_winner = self.rounds[-1].winner
        streak = 0
        for r in reversed(self.rounds):
            if r.winner == current_winner:
                streak += 1
            else:
                break
        return current_winner, streak
    
    def update_round(
        self,
        positions: Dict[str, float],
        market_return: float,
        date: str = "",
        regime: str = "sideways"
    ) -> RoundResult:
        """
        Update game state after a round.
        
        Args:
            positions: Position each strategy took (0-100%)
            market_return: Actual market return (as decimal, e.g., 0.02 = 2%)
            date: Trading date
            regime: Market regime
            
        Returns:
            RoundResult with full details
        """
        self.round_num += 1
        
        # Calculate returns for each strategy
        dollar_returns = {}
        pct_returns = {}
        
        for name in self.strategy_names:
            position = positions.get(name, 0)
            allocation = self.allocations[name]
            
            # Dollar return = allocation * (position/100) * market_return
            invested = allocation * (position / 100)
            dollar_ret = invested * market_return
            dollar_returns[name] = dollar_ret
            
            # Percentage return (of their allocation)
            pct_ret = dollar_ret / allocation if allocation > 0 else 0
            pct_returns[name] = pct_ret
        
        # Determine winner (highest dollar return this round)
        winner = max(dollar_returns, key=dollar_returns.get)
        
        # Check if high variance round (strategies took very different positions)
        pos_values = list(positions.values())
        high_variance = np.std(pos_values) > 25 if len(pos_values) > 1 else False
        
        # Reallocate capital based on relative performance
        self._reallocate(pct_returns)
        
        # Create round result
        result = RoundResult(
            round_num=self.round_num,
            date=date,
            regime=regime,
            market_return=market_return,
            positions=dict(positions),
            dollar_returns=dollar_returns,
            pct_returns=pct_returns,
            allocations_after=dict(self.allocations),
            winner=winner,
            high_variance_round=high_variance
        )
        
        # Update state
        self.last_positions = dict(positions)
        self.last_returns = pct_returns
        self.last_winner = winner
        
        # Record history
        self.allocation_history.append(dict(self.allocations))
        self.position_history.append(dict(positions))
        self.return_history.append(pct_returns)
        self.rounds.append(result)
        
        return result
    
    def _reallocate(self, returns: Dict[str, float]):
        """
        Reallocate capital based on relative performance.
        
        Uses z-score of returns to determine reallocation.
        Winners gain capital, losers lose capital.
        Total always sums to total_capital.
        """
        ret_values = list(returns.values())
        avg_ret = np.mean(ret_values)
        std_ret = np.std(ret_values)
        
        # Apply reallocation
        for name in self.allocations:
            if std_ret > 1e-10:  # Avoid division by zero
                z_score = (returns[name] - avg_ret) / std_ret
            else:
                z_score = 0
            
            # Reallocation: shift by rate * z_score
            # Positive z = outperformed = gain capital
            # Negative z = underperformed = lose capital
            multiplier = 1 + (z_score * self.reallocation_rate)
            
            # Clamp multiplier to prevent extreme swings
            multiplier = max(0.8, min(1.2, multiplier))
            
            self.allocations[name] *= multiplier
        
        # Normalize to total capital
        total = sum(self.allocations.values())
        if total > 0:
            for name in self.allocations:
                self.allocations[name] *= (self.total_capital / total)
    
    def get_cooperation_rate(self) -> float:
        """
        Calculate how often strategies took similar positions.
        
        Returns:
            Rate from 0-1 where 1 = perfect cooperation (all same position)
        """
        if not self.position_history:
            return 0.0
        
        cooperation_scores = []
        for positions in self.position_history:
            values = list(positions.values())
            if len(values) > 1:
                # Cooperation = 1 - normalized std dev
                std = np.std(values)
                max_std = 50  # Max possible std if positions are 0 and 100
                cooperation = 1 - (std / max_std)
                cooperation_scores.append(max(0, cooperation))
        
        return np.mean(cooperation_scores) if cooperation_scores else 0.0
    
    def get_defection_stats(self) -> Dict:
        """Get statistics about defection (contrarian) behavior."""
        if not self.rounds:
            return {}
        
        defection_rounds = [r for r in self.rounds if r.high_variance_round]
        defection_wins = [r for r in defection_rounds if r.winner == "Defector"]
        
        return {
            "total_rounds": len(self.rounds),
            "defection_rounds": len(defection_rounds),
            "defection_rate": len(defection_rounds) / len(self.rounds) if self.rounds else 0,
            "defector_wins_in_defection": len(defection_wins),
            "defector_win_rate_when_defecting": (
                len(defection_wins) / len(defection_rounds) 
                if defection_rounds else 0
            )
        }
    
    def get_allocation_gini(self) -> float:
        """
        Calculate Gini coefficient of current allocation.
        
        0 = perfect equality (everyone has same)
        1 = perfect inequality (one has everything)
        """
        values = sorted(self.allocations.values())
        n = len(values)
        if n == 0:
            return 0
        
        cumsum = np.cumsum(values)
        return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    def get_summary(self) -> Dict:
        """Get summary statistics for the game."""
        if not self.rounds:
            return {"status": "no rounds played"}
        
        # Count wins per strategy
        wins = {name: 0 for name in self.strategy_names}
        for r in self.rounds:
            wins[r.winner] = wins.get(r.winner, 0) + 1
        
        # Calculate total returns per strategy
        total_returns = {name: 0.0 for name in self.strategy_names}
        for r in self.rounds:
            for name, ret in r.pct_returns.items():
                total_returns[name] = (1 + total_returns[name]) * (1 + ret) - 1
        
        return {
            "ticker": self.ticker,
            "total_rounds": len(self.rounds),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "total_capital": self.total_capital,
            "reallocation_rate": self.reallocation_rate,
            "final_allocations": {k: round(v, 2) for k, v in self.allocations.items()},
            "allocation_pcts": {k: round(v / self.total_capital * 100, 2) for k, v in self.allocations.items()},
            "wins_per_strategy": wins,
            "win_rates": {k: round(v / len(self.rounds) * 100, 1) for k, v in wins.items()},
            "total_returns_pct": {k: round(v * 100, 2) for k, v in total_returns.items()},
            "cooperation_rate": round(self.get_cooperation_rate(), 3),
            "allocation_gini": round(self.get_allocation_gini(), 3),
            "defection_stats": self.get_defection_stats()
        }
    
    def to_dict(self) -> Dict:
        """Convert full game state to dictionary for JSON serialization."""
        return {
            "meta": {
                "ticker": self.ticker,
                "total_capital": self.total_capital,
                "reallocation_rate": self.reallocation_rate,
                "total_rounds": len(self.rounds),
                "strategy_names": self.strategy_names,
                "start_time": self.start_time.isoformat() if self.start_time else None
            },
            "final_state": {
                "allocations": {k: round(v, 2) for k, v in self.allocations.items()},
                "allocation_pcts": {k: round(v / self.total_capital * 100, 2) for k, v in self.allocations.items()},
                "last_positions": self.last_positions,
                "last_winner": self.last_winner
            },
            "rounds": [r.to_dict() for r in self.rounds],
            "summary": self.get_summary()
        }
    
    def reset(self, ticker: str = ""):
        """Reset game state for a new tournament."""
        initial = self.total_capital / len(self.strategy_names)
        self.allocations = {name: initial for name in self.strategy_names}
        self.round_num = 0
        self.last_positions = {}
        self.last_returns = {}
        self.last_winner = None
        self.allocation_history = [dict(self.allocations)]
        self.position_history = []
        self.return_history = []
        self.rounds = []
        self.ticker = ticker
        self.start_time = datetime.now()


# === Test ===
if __name__ == "__main__":
    print("Testing GameState...")
    print("=" * 60)
    
    # Create game
    strategies = ["Buy-and-Hold", "Cooperator", "Defector", "Tit-for-Tat", "Signal Follower"]
    game = GameState(strategy_names=strategies, total_capital=1_000_000)
    game.ticker = "TEST"
    
    print(f"Initial allocations: {game.allocations}")
    print(f"Total: ${sum(game.allocations.values()):,.0f}")
    
    # Simulate a few rounds
    test_rounds = [
        {"market": 0.02, "positions": {"Buy-and-Hold": 100, "Cooperator": 60, "Defector": 30, "Tit-for-Tat": 55, "Signal Follower": 50}},
        {"market": -0.01, "positions": {"Buy-and-Hold": 100, "Cooperator": 55, "Defector": 70, "Tit-for-Tat": 60, "Signal Follower": 45}},
        {"market": 0.015, "positions": {"Buy-and-Hold": 100, "Cooperator": 65, "Defector": 25, "Tit-for-Tat": 70, "Signal Follower": 55}},
    ]
    
    for i, round_data in enumerate(test_rounds):
        result = game.update_round(
            positions=round_data["positions"],
            market_return=round_data["market"],
            date=f"2024-01-{i+1:02d}",
            regime="bull" if round_data["market"] > 0 else "bear"
        )
        print(f"\nRound {result.round_num}: Market {round_data['market']*100:+.1f}%")
        print(f"  Winner: {result.winner}")
        print(f"  Allocations: {', '.join(f'{k}: ${v:,.0f}' for k, v in result.allocations_after.items())}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    summary = game.get_summary()
    print(f"  Final allocations: {summary['allocation_pcts']}")
    print(f"  Win rates: {summary['win_rates']}")
    print(f"  Cooperation rate: {summary['cooperation_rate']:.1%}")
    print(f"  Gini coefficient: {summary['allocation_gini']:.3f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")