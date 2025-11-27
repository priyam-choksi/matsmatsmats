"""
monte_carlo_engine.py - Bootstrap Monte Carlo Simulations

Location: agents/game_theory/monte_carlo_engine.py

This module runs Monte Carlo simulations to test strategy robustness.
Instead of just looking at one historical path, we resample returns
with replacement to see the distribution of possible outcomes.

Why Monte Carlo?
    - Single backtest can be lucky/unlucky
    - Shows range of possible outcomes
    - Calculates probability of beating benchmark
    - Provides confidence intervals

How It Works:
    1. Take strategy's actual returns
    2. Resample with replacement (bootstrap)
    3. Calculate metrics for each simulation
    4. Report distribution of outcomes

Usage:
    from game_theory.monte_carlo_engine import MonteCarloEngine
    
    engine = MonteCarloEngine(n_simulations=1000)
    
    # Single strategy
    result = engine.run_bootstrap(strategy.daily_returns)
    print(f"Mean return: {result.mean_return:.2f}%")
    print(f"95% CI: [{result.ci_lower_95:.2f}%, {result.ci_upper_95:.2f}%]")
    
    # Compare multiple strategies
    returns_dict = {s.name: s.daily_returns for s in strategies}
    results = engine.compare_strategies(returns_dict)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """
    Results from Monte Carlo simulation for one strategy.
    
    Attributes:
        strategy_name: Name of the strategy
        n_simulations: Number of simulations run
        
        # Return Distribution
        mean_return: Mean total return across simulations (%)
        std_return: Standard deviation of returns (%)
        median_return: Median total return (%)
        ci_lower_95: 2.5th percentile (lower bound of 95% CI)
        ci_upper_95: 97.5th percentile (upper bound of 95% CI)
        
        # Probabilities
        prob_positive: Probability of positive return
        prob_beat_benchmark: Probability of beating benchmark
        
        # Risk Metrics
        var_95: Value at Risk at 95% (5th percentile)
        mean_max_drawdown: Average maximum drawdown
        
        # Sharpe Distribution
        mean_sharpe: Mean Sharpe ratio
        std_sharpe: Std dev of Sharpe ratio
    """
    strategy_name: str
    n_simulations: int
    
    # Return Distribution
    mean_return: float = 0.0
    std_return: float = 0.0
    median_return: float = 0.0
    ci_lower_95: float = 0.0
    ci_upper_95: float = 0.0
    
    # Probabilities
    prob_positive: float = 0.0
    prob_beat_benchmark: float = 0.0
    
    # Risk Metrics
    var_95: float = 0.0
    mean_max_drawdown: float = 0.0
    
    # Sharpe Distribution
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0


class MonteCarloEngine:
    """
    Bootstrap Monte Carlo simulation for strategy robustness testing.
    
    Uses resampling with replacement to generate thousands of
    possible return paths from actual strategy returns.
    
    Attributes:
        n_simulations: Number of Monte Carlo simulations to run
        rng: Numpy random generator (can be seeded for reproducibility)
    """
    
    def __init__(
        self, 
        n_simulations: int = 1000, 
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_simulations: Number of simulations to run (default 1000)
            random_seed: Optional seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)
    
    def run_bootstrap(
        self, 
        returns: List[float],
        annualization_factor: float = 252 / 3
    ) -> MonteCarloResult:
        """
        Run bootstrap simulation on a single strategy's returns.
        
        Resamples returns with replacement and calculates 
        distribution of outcomes.
        
        Args:
            returns: List of daily returns (decimals)
            annualization_factor: Factor for annualizing metrics
            
        Returns:
            MonteCarloResult with distribution statistics
        """
        returns = np.array(returns)
        n = len(returns)
        
        # Need sufficient data
        if n < 5:
            return self._empty_result("insufficient_data")
        
        # Storage for simulation results
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        # Run simulations
        for _ in range(self.n_simulations):
            # Resample with replacement
            sampled = self.rng.choice(returns, size=n, replace=True)
            
            # Calculate equity curve
            equity = np.cumprod(1.0 + sampled)
            
            # Total return
            total_ret = (equity[-1] - 1.0) * 100.0
            total_returns.append(total_ret)
            
            # Annualized return and volatility for Sharpe
            ann_ret = ((equity[-1]) ** (annualization_factor / n) - 1.0) * 100.0
            vol = np.std(sampled) * np.sqrt(annualization_factor) * 100.0
            
            if vol > 0:
                sharpe_ratios.append(ann_ret / vol)
            else:
                sharpe_ratios.append(0.0)
            
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_drawdowns.append(abs(np.min(dd)) * 100.0)
        
        # Convert to arrays
        total_returns = np.array(total_returns)
        sharpe_ratios = np.array(sharpe_ratios)
        max_drawdowns = np.array(max_drawdowns)
        
        return MonteCarloResult(
            strategy_name="",  # Set externally
            n_simulations=self.n_simulations,
            mean_return=float(np.mean(total_returns)),
            std_return=float(np.std(total_returns)),
            median_return=float(np.median(total_returns)),
            ci_lower_95=float(np.percentile(total_returns, 2.5)),
            ci_upper_95=float(np.percentile(total_returns, 97.5)),
            prob_positive=float(np.mean(total_returns > 0)),
            prob_beat_benchmark=0.0,  # Set in compare_strategies
            var_95=float(np.percentile(total_returns, 5)),
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            mean_sharpe=float(np.mean(sharpe_ratios)),
            std_sharpe=float(np.std(sharpe_ratios))
        )
    
    def compare_strategies(
        self, 
        strategy_returns: Dict[str, List[float]],
        benchmark_name: str = "Actual Market"
    ) -> Dict[str, MonteCarloResult]:
        """
        Run Monte Carlo on all strategies and calculate probability 
        of beating the benchmark.
        
        Args:
            strategy_returns: Dict mapping strategy name to returns list
            benchmark_name: Name of benchmark strategy to compare against
            
        Returns:
            Dict mapping strategy name to MonteCarloResult
        """
        results = {}
        
        # Find benchmark returns
        benchmark_returns = strategy_returns.get(benchmark_name)
        
        # Try alternative benchmark names if not found
        if benchmark_returns is None:
            for alt_name in ["Buy-and-Hold", "BuyAndHold", "Market", "Control"]:
                if alt_name in strategy_returns:
                    benchmark_returns = strategy_returns[alt_name]
                    benchmark_name = alt_name
                    break
        
        # Run simulation for each strategy
        for name, returns in strategy_returns.items():
            mc_result = self.run_bootstrap(returns)
            mc_result.strategy_name = name
            
            # Calculate probability of beating benchmark
            if benchmark_returns is not None and name != benchmark_name:
                prob_beat = self._calculate_prob_beat_benchmark(
                    returns, 
                    benchmark_returns
                )
                mc_result.prob_beat_benchmark = prob_beat
            
            results[name] = mc_result
        
        return results
    
    def _calculate_prob_beat_benchmark(
        self, 
        strategy_returns: List[float],
        benchmark_returns: List[float]
    ) -> float:
        """
        Calculate probability that strategy beats benchmark.
        
        Uses paired bootstrap - samples same indices for both
        to preserve correlation structure.
        
        Args:
            strategy_returns: Strategy's returns
            benchmark_returns: Benchmark's returns
            
        Returns:
            Probability of beating benchmark (0.0 to 1.0)
        """
        strat_arr = np.array(strategy_returns)
        bench_arr = np.array(benchmark_returns)
        
        # Use minimum length
        n = min(len(strat_arr), len(bench_arr))
        
        if n < 5:
            return 0.5  # Not enough data
        
        strat_arr = strat_arr[:n]
        bench_arr = bench_arr[:n]
        
        beat_count = 0
        
        for _ in range(self.n_simulations):
            # Same indices for paired comparison
            idx = self.rng.choice(n, size=n, replace=True)
            
            strat_equity = np.prod(1.0 + strat_arr[idx])
            bench_equity = np.prod(1.0 + bench_arr[idx])
            
            if strat_equity > bench_equity:
                beat_count += 1
        
        return beat_count / self.n_simulations
    
    def _empty_result(self, name: str) -> MonteCarloResult:
        """Return empty result for insufficient data."""
        return MonteCarloResult(
            strategy_name=name,
            n_simulations=0,
            mean_return=0.0,
            std_return=0.0,
            median_return=0.0,
            ci_lower_95=0.0,
            ci_upper_95=0.0,
            prob_positive=0.0,
            prob_beat_benchmark=0.0,
            var_95=0.0,
            mean_max_drawdown=0.0,
            mean_sharpe=0.0,
            std_sharpe=0.0
        )
    
    def to_dict(self, result: MonteCarloResult) -> Dict:
        """
        Convert MonteCarloResult to dictionary for JSON serialization.
        
        Args:
            result: MonteCarloResult object
            
        Returns:
            Dictionary with all results
        """
        return {
            'strategy': result.strategy_name,
            'simulations': result.n_simulations,
            'mean_return': round(result.mean_return, 2),
            'std_return': round(result.std_return, 2),
            'median_return': round(result.median_return, 2),
            'ci_95': [
                round(result.ci_lower_95, 2), 
                round(result.ci_upper_95, 2)
            ],
            'prob_positive': round(result.prob_positive, 3),
            'prob_beat_benchmark': round(result.prob_beat_benchmark, 3),
            'var_95': round(result.var_95, 2),
            'mean_max_drawdown': round(result.mean_max_drawdown, 2),
            'mean_sharpe': round(result.mean_sharpe, 3),
            'std_sharpe': round(result.std_sharpe, 3)
        }
    
    def print_results(self, results: Dict[str, MonteCarloResult]):
        """
        Print formatted Monte Carlo results.
        
        Args:
            results: Dict mapping strategy name to MonteCarloResult
        """
        print("\n" + "=" * 70)
        print("MONTE CARLO SIMULATION RESULTS")
        print(f"Simulations: {self.n_simulations}")
        print("=" * 70)
        
        # Header
        print(f"\n{'Strategy':<18} {'Mean':>8} {'Median':>8} {'95% CI':>18} {'P(>0)':>8} {'P(Beat)':>8}")
        print("-" * 70)
        
        # Sort by mean return
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1].mean_return, 
            reverse=True
        )
        
        for name, r in sorted_results:
            ci_str = f"[{r.ci_lower_95:+.1f}, {r.ci_upper_95:+.1f}]"
            print(
                f"{name:<18} "
                f"{r.mean_return:>+7.1f}% "
                f"{r.median_return:>+7.1f}% "
                f"{ci_str:>18} "
                f"{r.prob_positive:>7.0%} "
                f"{r.prob_beat_benchmark:>7.0%}"
            )
        
        print("-" * 70)
        
        # Risk metrics
        print(f"\n{'Strategy':<18} {'VaR 95%':>10} {'Avg MaxDD':>10} {'Mean Sharpe':>12}")
        print("-" * 50)
        
        for name, r in sorted_results:
            print(
                f"{name:<18} "
                f"{r.var_95:>+9.1f}% "
                f"{r.mean_max_drawdown:>9.1f}% "
                f"{r.mean_sharpe:>+11.3f}"
            )
        
        print("=" * 70 + "\n")


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing MonteCarloEngine...")
    print("=" * 60)
    
    # Create sample returns for testing
    np.random.seed(42)
    
    # Simulated strategy returns
    strategy_returns = {
        "Actual Market": list(np.random.normal(0.001, 0.015, 50)),
        "Cooperator": list(np.random.normal(0.0015, 0.012, 50)),
        "Defector": list(np.random.normal(0.0008, 0.020, 50)),
        "Buy-and-Hold": list(np.random.normal(0.0012, 0.010, 50)),
        "Tit-for-Tat": list(np.random.normal(0.0013, 0.014, 50)),
    }
    
    # Run Monte Carlo
    engine = MonteCarloEngine(n_simulations=1000, random_seed=42)
    
    # Test single strategy
    print("\nSingle Strategy Bootstrap:")
    print("-" * 40)
    
    result = engine.run_bootstrap(strategy_returns["Cooperator"])
    result.strategy_name = "Cooperator"
    
    print(f"Strategy: {result.strategy_name}")
    print(f"Mean Return: {result.mean_return:+.2f}%")
    print(f"Median Return: {result.median_return:+.2f}%")
    print(f"95% CI: [{result.ci_lower_95:+.2f}%, {result.ci_upper_95:+.2f}%]")
    print(f"P(Return > 0): {result.prob_positive:.1%}")
    print(f"VaR 95%: {result.var_95:+.2f}%")
    print(f"Mean Max DD: {result.mean_max_drawdown:.2f}%")
    print(f"Mean Sharpe: {result.mean_sharpe:.3f}")
    
    # Test comparison
    print("\n" + "=" * 60)
    print("Strategy Comparison:")
    
    results = engine.compare_strategies(strategy_returns, benchmark_name="Actual Market")
    engine.print_results(results)
    
    # Test JSON conversion
    print("JSON conversion test:")
    json_dict = engine.to_dict(results["Cooperator"])
    print(f"Keys: {list(json_dict.keys())}")
    
    print("\n" + "=" * 60)
    print("MonteCarloEngine test complete!")