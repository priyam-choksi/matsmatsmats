"""
monte_carlo_analysis.py - Monte Carlo Simulation for Strategy Robustness

This module runs Monte Carlo simulations to:
1. Test if strategy outperformance is statistically significant
2. Calculate confidence intervals on returns
3. Bootstrap p-values for strategy comparisons
4. Generate probability distributions

Usage:
    python monte_carlo_analysis.py AAPL
    python monte_carlo_analysis.py --all --simulations 1000
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from main analysis
from game_theory_analysis import (
    DataLoader, GameTheoryTournament, MarketContext,
    CooperatorStrategy, DefectorStrategy, TitForTatStrategy,
    ConservativeBaselineStrategy, AggressiveBaselineStrategy, BuyAndHoldStrategy
)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    strategy_name: str
    mean_return: float
    std_return: float
    ci_lower: float  # 95% CI
    ci_upper: float
    percentile_5: float
    percentile_95: float
    prob_positive: float
    prob_beat_market: float
    sharpe_mean: float
    sharpe_std: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.
    
    Methods:
    1. Bootstrap resampling - resample historical returns
    2. Return shuffling - randomize order to test path dependency
    3. Noise injection - add random noise to returns
    """
    
    def __init__(self, output_dir: Path = None):
        self.loader = DataLoader()
        
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.loader.project_root / "outputs" / "monte_carlo" / timestamp
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Monte Carlo Simulator initialized")
        print(f"Output: {self.output_dir}")
    
    def run_bootstrap_simulation(self, ticker: str, n_simulations: int = 1000,
                                 portfolio: int = 100000) -> Dict[str, MonteCarloResult]:
        """
        Bootstrap simulation - resample daily returns with replacement.
        Tests robustness to specific sequence of returns.
        """
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP MONTE CARLO: {ticker}")
        print(f"Simulations: {n_simulations}")
        print(f"{'='*60}")
        
        # Load original data
        contexts = self.loader.load_ticker_data(ticker, portfolio)
        if not contexts:
            print(f"No data for {ticker}")
            return {}
        
        # Extract market returns
        market_returns = [ctx.daily_return for ctx in contexts]
        
        # Initialize result storage
        strategies = [
            CooperatorStrategy(),
            DefectorStrategy(),
            TitForTatStrategy(),
            ConservativeBaselineStrategy(),
            AggressiveBaselineStrategy(),
            BuyAndHoldStrategy()
        ]
        
        simulation_results = {s.name: {'returns': [], 'sharpes': []} for s in strategies}
        
        # Run simulations
        for sim in range(n_simulations):
            if (sim + 1) % 100 == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}")
            
            # Bootstrap resample returns
            resampled_indices = np.random.choice(len(contexts), size=len(contexts), replace=True)
            
            # Reset strategies
            for strategy in strategies:
                strategy.reset()
            
            # Run with resampled data
            for idx in resampled_indices:
                ctx = contexts[idx]
                for strategy in strategies:
                    strategy.execute_trade(ctx)
            
            # Record results
            for strategy in strategies:
                metrics = strategy.calculate_metrics()
                if metrics:
                    simulation_results[strategy.name]['returns'].append(metrics.total_return)
                    simulation_results[strategy.name]['sharpes'].append(metrics.sharpe_ratio)
        
        # Calculate statistics
        results = {}
        bh_returns = simulation_results['Buy-and-Hold']['returns']
        
        for name, data in simulation_results.items():
            returns = np.array(data['returns'])
            sharpes = np.array(data['sharpes'])
            
            # Probability of beating market
            prob_beat = np.mean(returns > np.array(bh_returns)) if name != 'Buy-and-Hold' else 0
            
            results[name] = MonteCarloResult(
                strategy_name=name,
                mean_return=np.mean(returns),
                std_return=np.std(returns),
                ci_lower=np.percentile(returns, 2.5),
                ci_upper=np.percentile(returns, 97.5),
                percentile_5=np.percentile(returns, 5),
                percentile_95=np.percentile(returns, 95),
                prob_positive=np.mean(returns > 0),
                prob_beat_market=prob_beat,
                sharpe_mean=np.mean(sharpes),
                sharpe_std=np.std(sharpes)
            )
        
        # Generate visualizations
        self._plot_return_distributions(ticker, simulation_results)
        self._plot_confidence_intervals(ticker, results)
        self._save_results(ticker, results, simulation_results)
        
        return results
    
    def run_significance_test(self, ticker: str, n_permutations: int = 5000,
                             portfolio: int = 100000) -> Dict[str, float]:
        """
        Permutation test for statistical significance.
        Tests: Is strategy performance significantly different from random?
        
        Returns p-values for each strategy vs Buy-and-Hold.
        """
        print(f"\n{'='*60}")
        print(f"PERMUTATION TEST: {ticker}")
        print(f"Permutations: {n_permutations}")
        print(f"{'='*60}")
        
        # Load data and get actual performance
        contexts = self.loader.load_ticker_data(ticker, portfolio)
        if not contexts:
            return {}
        
        # Get actual strategy returns
        strategies = [
            CooperatorStrategy(),
            DefectorStrategy(),
            TitForTatStrategy(),
            ConservativeBaselineStrategy(),
            AggressiveBaselineStrategy(),
            BuyAndHoldStrategy()
        ]
        
        for ctx in contexts:
            for strategy in strategies:
                strategy.execute_trade(ctx)
        
        actual_returns = {s.name: s.calculate_metrics().total_return for s in strategies}
        actual_diff = {s.name: actual_returns[s.name] - actual_returns['Buy-and-Hold'] 
                      for s in strategies if s.name != 'Buy-and-Hold'}
        
        print(f"\nActual performance difference vs Buy-and-Hold:")
        for name, diff in actual_diff.items():
            print(f"  {name}: {diff:+.2f}%")
        
        # Permutation test
        permuted_diffs = {name: [] for name in actual_diff.keys()}
        
        for perm in range(n_permutations):
            if (perm + 1) % 500 == 0:
                print(f"  Permutation {perm + 1}/{n_permutations}")
            
            # Shuffle return order
            shuffled_contexts = contexts.copy()
            np.random.shuffle(shuffled_contexts)
            
            # Reset and run
            for strategy in strategies:
                strategy.reset()
            
            for ctx in shuffled_contexts:
                for strategy in strategies:
                    strategy.execute_trade(ctx)
            
            # Record differences
            perm_returns = {s.name: s.calculate_metrics().total_return for s in strategies}
            for name in actual_diff.keys():
                diff = perm_returns[name] - perm_returns['Buy-and-Hold']
                permuted_diffs[name].append(diff)
        
        # Calculate p-values (two-tailed)
        p_values = {}
        for name, actual in actual_diff.items():
            perm_array = np.array(permuted_diffs[name])
            # Two-tailed: probability of seeing result as extreme
            p_value = np.mean(np.abs(perm_array) >= np.abs(actual))
            p_values[name] = p_value
        
        print(f"\nP-values (vs Buy-and-Hold):")
        for name, p in p_values.items():
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"  {name}: p = {p:.4f} {sig}")
        
        # Save results
        with open(self.output_dir / f'{ticker}_significance.json', 'w') as f:
            json.dump({
                'actual_diff': actual_diff,
                'p_values': p_values,
                'n_permutations': n_permutations
            }, f, indent=2)
        
        return p_values
    
    def run_all_tickers(self, n_simulations: int = 500) -> Dict[str, Dict[str, MonteCarloResult]]:
        """Run Monte Carlo on all available tickers"""
        tickers = self.loader.get_available_tickers()
        
        if not tickers:
            print("No tickers found for Monte Carlo analysis")
            return {}
        
        all_results = {}
        
        for ticker in tickers:
            results = self.run_bootstrap_simulation(ticker, n_simulations)
            if results:
                all_results[ticker] = results
        
        # Generate combined analysis only if we have results
        if all_results:
            self._generate_combined_mc_analysis(all_results)
        else:
            print("No results to generate combined analysis")
        
        return all_results
    
    def _plot_return_distributions(self, ticker: str, simulation_results: Dict):
        """Plot distribution of returns for each strategy"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {
            'Cooperator': '#2ecc71',
            'Defector': '#e74c3c',
            'Tit-for-Tat': '#3498db',
            'Conservative': '#9b59b6',
            'Aggressive': '#f39c12',
            'Buy-and-Hold': '#2c3e50'
        }
        
        for idx, (name, data) in enumerate(simulation_results.items()):
            ax = axes[idx]
            returns = data['returns']
            
            ax.hist(returns, bins=50, color=colors.get(name, '#888'), 
                   alpha=0.7, edgecolor='white')
            ax.axvline(x=np.mean(returns), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(returns):.1f}%')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}', fontweight='bold')
            ax.legend(fontsize=9)
        
        plt.suptitle(f'{ticker} - Monte Carlo Return Distributions\n(n=1000 bootstrap samples)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_intervals(self, ticker: str, results: Dict[str, MonteCarloResult]):
        """Plot 95% confidence intervals for each strategy"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        strategies = list(results.keys())
        means = [results[s].mean_return for s in strategies]
        ci_lower = [results[s].ci_lower for s in strategies]
        ci_upper = [results[s].ci_upper for s in strategies]
        
        # Calculate error bars
        lower_err = [means[i] - ci_lower[i] for i in range(len(strategies))]
        upper_err = [ci_upper[i] - means[i] for i in range(len(strategies))]
        
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2c3e50']
        
        bars = ax.barh(strategies, means, xerr=[lower_err, upper_err],
                      color=colors[:len(strategies)], capsize=5, alpha=0.8)
        
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Return (%)', fontsize=12)
        ax.set_title(f'{ticker} - 95% Confidence Intervals on Returns', fontsize=14, fontweight='bold')
        
        # Add probability annotations
        for i, (strat, result) in enumerate(results.items()):
            ax.text(ci_upper[i] + 1, i, 
                   f'P(>0): {result.prob_positive:.0%}',
                   va='center', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_confidence_intervals.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, ticker: str, results: Dict[str, MonteCarloResult], 
                     simulation_results: Dict):
        """Save Monte Carlo results"""
        # Summary
        summary = {name: vars(r) for name, r in results.items()}
        with open(self.output_dir / f'{ticker}_mc_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Raw data for further analysis
        raw_data = {name: {
            'returns': data['returns'],
            'sharpes': data['sharpes']
        } for name, data in simulation_results.items()}
        
        # Convert to DataFrame and save
        for name, data in raw_data.items():
            df = pd.DataFrame({
                'return': data['returns'],
                'sharpe': data['sharpes']
            })
            df.to_csv(self.output_dir / f'{ticker}_{name}_simulations.csv', index=False)
    
    def _generate_combined_mc_analysis(self, all_results: Dict[str, Dict[str, MonteCarloResult]]):
        """Generate analysis across all tickers"""
        print(f"\n{'='*60}")
        print("COMBINED MONTE CARLO ANALYSIS")
        print(f"{'='*60}")
        
        # Aggregate probabilities
        strategy_probs = {}
        
        for ticker, results in all_results.items():
            for name, mc_result in results.items():
                if name not in strategy_probs:
                    strategy_probs[name] = {
                        'prob_positive': [],
                        'prob_beat_market': [],
                        'mean_returns': [],
                        'sharpe_means': []
                    }
                strategy_probs[name]['prob_positive'].append(mc_result.prob_positive)
                strategy_probs[name]['prob_beat_market'].append(mc_result.prob_beat_market)
                strategy_probs[name]['mean_returns'].append(mc_result.mean_return)
                strategy_probs[name]['sharpe_means'].append(mc_result.sharpe_mean)
        
        # Summary table
        if not strategy_probs:
            print("No Monte Carlo data to summarize")
            return
            
        print(f"\n{'Strategy':<20} {'Avg P(>0)':>12} {'Avg P(Beat Mkt)':>16} {'Avg Return':>12}")
        print("-" * 65)
        
        for name, probs in sorted(strategy_probs.items(), 
                                  key=lambda x: np.mean(x[1]['mean_returns']), reverse=True):
            avg_pos = np.mean(probs['prob_positive'])
            avg_beat = np.mean(probs['prob_beat_market'])
            avg_ret = np.mean(probs['mean_returns'])
            print(f"{name:<20} {avg_pos:>11.1%} {avg_beat:>15.1%} {avg_ret:>+11.2f}%")
        
        # Plot combined results
        self._plot_combined_mc_results(strategy_probs)
        
        # Save combined results
        combined = {
            name: {
                'avg_prob_positive': np.mean(probs['prob_positive']),
                'avg_prob_beat_market': np.mean(probs['prob_beat_market']),
                'avg_mean_return': np.mean(probs['mean_returns']),
                'avg_sharpe': np.mean(probs['sharpe_means'])
            }
            for name, probs in strategy_probs.items()
        }
        
        with open(self.output_dir / 'combined_mc_analysis.json', 'w') as f:
            json.dump(combined, f, indent=2)
    
    def _plot_combined_mc_results(self, strategy_probs: Dict):
        """Plot combined Monte Carlo results"""
        if not strategy_probs:
            print("No data to plot for combined MC results")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        strategies = list(strategy_probs.keys())
        if not strategies:
            plt.close()
            return
            
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2c3e50']
        
        # Probability of positive return
        ax = axes[0]
        probs = [np.mean(strategy_probs[s]['prob_positive']) * 100 for s in strategies]
        bars = ax.bar(strategies, probs, color=colors[:len(strategies)])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Probability of Positive Return', fontweight='bold')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}%', ha='center', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Probability of beating market
        ax = axes[1]
        beat_probs = [np.mean(strategy_probs[s]['prob_beat_market']) * 100 
                     for s in strategies if s != 'Buy-and-Hold']
        strat_names = [s for s in strategies if s != 'Buy-and-Hold']
        bars = ax.bar(strat_names, beat_probs, color=colors[:len(strat_names)])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Probability of Beating Buy-and-Hold', fontweight='bold')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, beat_probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}%', ha='center', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Distribution of mean returns across tickers
        ax = axes[2]
        data_to_plot = [strategy_probs[s]['mean_returns'] for s in strategies if strategy_probs[s]['mean_returns']]
        valid_strategies = [s for s in strategies if strategy_probs[s]['mean_returns']]
        
        if data_to_plot and valid_strategies:
            bp = ax.boxplot(data_to_plot, labels=[s[:8] for s in valid_strategies], patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(valid_strategies)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Return (%)')
        ax.set_title('Return Distribution Across Tickers', fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Monte Carlo Analysis - All Tickers Combined', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_mc_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Analysis")
    parser.add_argument('ticker', nargs='?', default=None, help='Single ticker to analyze')
    parser.add_argument('--all', action='store_true', help='Run on all tickers')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--significance', action='store_true', help='Run significance test')
    parser.add_argument('--portfolio', type=int, default=100000, help='Portfolio size')
    
    args = parser.parse_args()
    
    simulator = MonteCarloSimulator()
    
    if args.all:
        simulator.run_all_tickers(n_simulations=args.simulations)
    elif args.ticker:
        simulator.run_bootstrap_simulation(args.ticker, n_simulations=args.simulations)
        if args.significance:
            simulator.run_significance_test(args.ticker)
    else:
        print("Please specify --ticker AAPL or --all")
        return
    
    print("✅ Monte Carlo analysis complete!")


if __name__ == "__main__":
    main()