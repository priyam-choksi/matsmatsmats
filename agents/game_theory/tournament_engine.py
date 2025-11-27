"""
tournament_engine.py - Main Tournament Orchestrator

Location: agents/game_theory/tournament_engine.py

This is the main engine that brings everything together:
1. Loads data from your collected workflows
2. Runs all strategies through the tournament
3. Calculates comprehensive metrics
4. Runs Monte Carlo simulations
5. Generates visualizations
6. Saves results to JSON

This is what you'll use to answer your research question:
"Which game theory strategy performs best in different market regimes?"

Usage:
    from game_theory.tournament_engine import TournamentEngine
    
    engine = TournamentEngine()
    
    # Run single ticker
    metrics = engine.run_ticker("AAPL")
    
    # Run all tickers
    all_results = engine.run_all_tickers()

CLI Usage (via run_analysis.py):
    python -m agents.game_theory.run_analysis --ticker AAPL
    python -m agents.game_theory.run_analysis --all
    python -m agents.game_theory.run_analysis --all --quick
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from .data_loader import DataLoader
from .market_context import MarketContext
from .metrics_calculator import MetricsCalculator, StrategyMetrics
from .monte_carlo_engine import MonteCarloEngine, MonteCarloResult
from .visualization_engine import VisualizationEngine
from .strategies import (
    ActualMarketStrategy,
    BuyHoldStrategy,
    CooperatorStrategy,
    DefectorStrategy,
    TitForTatStrategy,
    get_all_strategies
)


class TournamentEngine:
    """
    Main engine to run game theory tournaments.
    
    Orchestrates the entire analysis pipeline:
    1. Load data -> DataLoader
    2. Run tournament -> Strategies
    3. Calculate metrics -> MetricsCalculator
    4. Monte Carlo -> MonteCarloEngine
    5. Visualize -> VisualizationEngine
    6. Save results -> JSON files
    
    Attributes:
        loader: DataLoader instance
        output_dir: Directory for all outputs
        metrics_calc: MetricsCalculator instance
        monte_carlo: MonteCarloEngine instance
        viz: VisualizationEngine instance
        strategies: List of strategy instances
        all_results: Accumulated results across tickers
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize TournamentEngine.
        
        Args:
            output_dir: Custom output directory. If None, creates
                       timestamped folder in outputs/gt_analysis/
        """
        # Initialize data loader
        self.loader = DataLoader()
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = (
                self.loader.project_root / "outputs" / "gt_analysis" / timestamp
            )
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "by_ticker").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Initialize components
        self.metrics_calc = MetricsCalculator()
        self.monte_carlo = MonteCarloEngine(n_simulations=1000)
        self.viz = VisualizationEngine(self.output_dir / "visualizations")
        
        # Initialize strategies
        self.strategies = get_all_strategies()
        
        # Results storage
        self.all_results: Dict[str, Dict[str, StrategyMetrics]] = {}
        
        print(f"Tournament Engine initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Strategies: {[s.name for s in self.strategies]}")
    
    def run_ticker(
        self, 
        ticker: str, 
        run_monte_carlo: bool = True,
        generate_gifs: bool = True
    ) -> Dict[str, StrategyMetrics]:
        """
        Run tournament for a single ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            run_monte_carlo: Whether to run Monte Carlo simulations
            generate_gifs: Whether to generate animated GIFs
            
        Returns:
            Dict mapping strategy name to StrategyMetrics
        """
        print(f"\n{'='*60}")
        print(f"RUNNING TOURNAMENT: {ticker}")
        print('='*60)
        
        # Load data
        contexts = self.loader.load_ticker_data(ticker)
        
        if not contexts:
            print(f"No data found for {ticker}")
            return {}
        
        print(f"Loaded {len(contexts)} samples")
        
        # Reset all strategies
        for s in self.strategies:
            s.reset()
        
        # Run tournament
        print("\nExecuting tournament...")
        for i, ctx in enumerate(contexts):
            # Execute trades for all strategies
            round_returns = {}
            
            for strategy in self.strategies:
                result = strategy.execute_trade(ctx)
                round_returns[strategy.name] = result.trade_return
            
            # Update Tit-for-Tat with round winner
            winner = max(round_returns, key=round_returns.get)
            for s in self.strategies:
                if hasattr(s, 'update_winner'):
                    s.update_winner(winner)
            
            # Progress indicator
            if (i + 1) % 20 == 0 or i == len(contexts) - 1:
                print(f"  Completed {i+1}/{len(contexts)} samples")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics: Dict[str, StrategyMetrics] = {}
        
        for s in self.strategies:
            m = self.metrics_calc.calculate(s)
            if m:
                metrics[s.name] = m
                print(
                    f"  {s.name:15} | "
                    f"Return: {m.total_return:+7.2f}% | "
                    f"Sharpe: {m.sharpe_ratio:+6.3f} | "
                    f"Score: {m.final_score:+5.1f}"
                )
        
        # Save ticker results
        self._save_ticker_results(ticker, metrics)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        ticker_viz_dir = self.output_dir / "by_ticker" / ticker
        ticker_viz_dir.mkdir(exist_ok=True)
        ticker_viz = VisualizationEngine(ticker_viz_dir)
        
        ticker_viz.create_equity_curves(self.strategies, ticker)
        ticker_viz.create_strategy_comparison(metrics)
        ticker_viz.create_regime_heatmap(metrics)
        ticker_viz.create_score_history_chart(self.strategies, ticker)
        
        if generate_gifs:
            print("  Creating animated GIFs (this may take a moment)...")
            ticker_viz.create_equity_race_gif(self.strategies, ticker)
            ticker_viz.create_score_evolution_gif(self.strategies, ticker)
        
        # Monte Carlo
        if run_monte_carlo:
            print("\nRunning Monte Carlo simulations...")
            returns_dict = {s.name: s.daily_returns for s in self.strategies}
            mc_results = self.monte_carlo.compare_strategies(returns_dict)
            
            ticker_viz.create_monte_carlo_chart(mc_results)
            
            # Save MC results
            mc_dict = {
                name: self.monte_carlo.to_dict(r) 
                for name, r in mc_results.items()
            }
            mc_path = ticker_viz_dir / "monte_carlo.json"
            with open(mc_path, 'w', encoding='utf-8') as f:
                json.dump(mc_dict, f, indent=2)
            print(f"  Saved: {mc_path}")
        
        # Store results
        self.all_results[ticker] = metrics
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {ticker}")
        print('='*60)
        
        return metrics
    
    def run_all_tickers(
        self, 
        run_monte_carlo: bool = True,
        generate_gifs: bool = False
    ) -> Dict[str, Dict[str, StrategyMetrics]]:
        """
        Run tournament for all available tickers.
        
        Args:
            run_monte_carlo: Whether to run Monte Carlo for each ticker
            generate_gifs: Whether to generate animated GIFs (slower)
            
        Returns:
            Dict mapping ticker -> (strategy name -> StrategyMetrics)
        """
        tickers = self.loader.get_available_tickers()
        
        print(f"\n{'#'*60}")
        print(f"TOURNAMENT: ALL TICKERS")
        print(f"Found {len(tickers)} tickers: {tickers}")
        print('#'*60)
        
        for ticker in tickers:
            metrics = self.run_ticker(
                ticker, 
                run_monte_carlo=run_monte_carlo,
                generate_gifs=generate_gifs
            )
            if metrics:
                self.all_results[ticker] = metrics
        
        # Generate combined analysis
        if self.all_results:
            self._generate_combined_analysis()
        
        return self.all_results
    
    def _save_ticker_results(
        self, 
        ticker: str, 
        metrics: Dict[str, StrategyMetrics]
    ):
        """Save metrics JSON for a ticker."""
        ticker_dir = self.output_dir / "by_ticker" / ticker
        ticker_dir.mkdir(exist_ok=True)
        
        # Convert to dict
        metrics_dict = {
            name: self.metrics_calc.to_dict(m) 
            for name, m in metrics.items()
        }
        
        # Save
        output_path = ticker_dir / "metrics.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"  Saved: {output_path}")
    
    def _generate_combined_analysis(self):
        """Generate analysis combining all tickers."""
        print(f"\n{'='*60}")
        print("GENERATING COMBINED ANALYSIS")
        print('='*60)
        
        # Aggregate by strategy
        strategy_totals: Dict[str, List[StrategyMetrics]] = {
            s.name: [] for s in self.strategies
        }
        
        for ticker, metrics in self.all_results.items():
            for name, m in metrics.items():
                if name in strategy_totals:
                    strategy_totals[name].append(m)
        
        # Calculate averages
        combined = {}
        for name, metrics_list in strategy_totals.items():
            if metrics_list:
                n = len(metrics_list)
                combined[name] = {
                    'avg_return': sum(m.total_return for m in metrics_list) / n,
                    'avg_sharpe': sum(m.sharpe_ratio for m in metrics_list) / n,
                    'avg_max_dd': sum(m.max_drawdown for m in metrics_list) / n,
                    'avg_win_rate': sum(m.win_rate for m in metrics_list) / n,
                    'avg_score': sum(m.final_score for m in metrics_list) / n,
                    'total_tickers': n,
                    'win_count': sum(1 for m in metrics_list if m.total_return > 0),
                    'avg_bull_return': sum(m.bull_return for m in metrics_list) / n,
                    'avg_bear_return': sum(m.bear_return for m in metrics_list) / n,
                    'avg_sideways_return': sum(m.sideways_return for m in metrics_list) / n,
                }
        
        # Save combined metrics
        combined_path = self.output_dir / "combined_metrics.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2)
        print(f"Saved: {combined_path}")
        
        # Create combined visualizations
        self._create_combined_visualizations(combined)
        
        # Print summary
        self._print_summary(combined)
    
    def _create_combined_visualizations(self, combined: Dict):
        """Create visualizations for combined results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        names = list(combined.keys())
        
        # 1. Average Return by Strategy
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = [combined[n]['avg_return'] for n in names]
        colors = [self.viz._get_color(n) for n in names]
        
        bars = ax.bar(names, returns, color=colors)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Average Return (%)')
        ax.set_title('Average Return Across All Tickers', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for bar, val in zip(bars, returns):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height(),
                f'{val:+.1f}%', 
                ha='center', 
                va='bottom'
            )
        
        plt.tight_layout()
        fig.savefig(
            self.output_dir / "visualizations" / "combined_returns.png",
            dpi=150
        )
        plt.close(fig)
        
        # 2. Regime Performance Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        width = 0.25
        
        bull = [combined[n]['avg_bull_return'] for n in names]
        bear = [combined[n]['avg_bear_return'] for n in names]
        sideways = [combined[n]['avg_sideways_return'] for n in names]
        
        ax.bar(x - width, bull, width, label='Bull', color='#27ae60')
        ax.bar(x, bear, width, label='Bear', color='#e74c3c')
        ax.bar(x + width, sideways, width, label='Sideways', color='#3498db')
        
        ax.set_ylabel('Average Return (%)')
        ax.set_title('Average Return by Market Regime', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        fig.savefig(
            self.output_dir / "visualizations" / "combined_regime_performance.png",
            dpi=150
        )
        plt.close(fig)
        
        print("Created combined visualizations")
    
    def _print_summary(self, combined: Dict):
        """Print final summary to console."""
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY")
        print('='*70)
        
        # Sort by average return
        sorted_strats = sorted(
            combined.items(), 
            key=lambda x: x[1]['avg_return'], 
            reverse=True
        )
        
        print(f"\n{'Strategy':<15} {'Avg Return':>12} {'Avg Sharpe':>12} "
              f"{'Avg Score':>10} {'Win Rate':>10}")
        print("-" * 60)
        
        for name, m in sorted_strats:
            print(
                f"{name:<15} "
                f"{m['avg_return']:>+11.2f}% "
                f"{m['avg_sharpe']:>+11.3f} "
                f"{m['avg_score']:>+9.1f} "
                f"{m['avg_win_rate']:>9.1f}%"
            )
        
        # Regime breakdown
        print(f"\n{'Strategy':<15} {'Bull':>12} {'Bear':>12} {'Sideways':>12}")
        print("-" * 55)
        
        for name, m in sorted_strats:
            print(
                f"{name:<15} "
                f"{m['avg_bull_return']:>+11.2f}% "
                f"{m['avg_bear_return']:>+11.2f}% "
                f"{m['avg_sideways_return']:>+11.2f}%"
            )
        
        # Winner announcement
        winner = sorted_strats[0][0]
        print(f"\n{'='*70}")
        print(f"🏆 BEST OVERALL STRATEGY: {winner}")
        print(f"   Average Return: {sorted_strats[0][1]['avg_return']:+.2f}%")
        print(f"   Average Sharpe: {sorted_strats[0][1]['avg_sharpe']:+.3f}")
        print('='*70)
        
        print(f"\nResults saved to: {self.output_dir}")
        print('='*70 + "\n")
    
    def get_results_summary(self) -> Dict:
        """Get summary of all results as dictionary."""
        if not self.all_results:
            return {}
        
        summary = {
            'tickers_analyzed': list(self.all_results.keys()),
            'strategies': [s.name for s in self.strategies],
            'output_directory': str(self.output_dir),
            'results_by_ticker': {}
        }
        
        for ticker, metrics in self.all_results.items():
            summary['results_by_ticker'][ticker] = {
                name: self.metrics_calc.to_dict(m)
                for name, m in metrics.items()
            }
        
        return summary


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing TournamentEngine...")
    print("=" * 60)
    
    try:
        engine = TournamentEngine()
        
        # Show available data
        engine.loader.print_data_summary()
        
        tickers = engine.loader.get_available_tickers()
        
        if tickers:
            # Run on first available ticker
            test_ticker = tickers[0]
            print(f"\nRunning test tournament on {test_ticker}...")
            
            metrics = engine.run_ticker(
                test_ticker,
                run_monte_carlo=True,
                generate_gifs=False  # Skip GIFs for quick test
            )
            
            if metrics:
                print("\nTest completed successfully!")
                print(f"Results saved to: {engine.output_dir}")
        else:
            print("\nNo ticker data found!")
            print("Make sure you have run the data collection workflow first.")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure you're running from the project directory.")
    
    print("\n" + "=" * 60)
    print("TournamentEngine test complete!")