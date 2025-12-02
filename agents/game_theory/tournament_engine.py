"""
FULL REPLACEMENT for tournament_engine.py
Location: agents/game_theory/tournament_engine.py

Just replace the entire file with this.
"""

import json
import logging
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
    Main engine to run game theory tournaments with detailed logging.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize TournamentEngine."""
        # Find data directory
        self.loader = DataLoader()
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("outputs") / "tournament_results" / timestamp
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        
        # Create logs folder structure: logs/YYYYMMDD_HHMMSS/text/ and logs/YYYYMMDD_HHMMSS/json/
        self._log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_base_dir = self.output_dir / "logs" / self._log_timestamp
        self._log_text_dir = self._log_base_dir / "text"
        self._log_json_dir = self._log_base_dir / "json"
        self._log_text_dir.mkdir(parents=True, exist_ok=True)
        self._log_json_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_calc = MetricsCalculator()
        self.monte_carlo = MonteCarloEngine()
        self.viz = VisualizationEngine(self.output_dir / "visualizations")
        
        # Initialize strategies
        self.strategies = get_all_strategies()
        
        # Results storage
        self.all_results: Dict[str, Dict[str, StrategyMetrics]] = {}
        
        # Logger reference
        self._current_log_file = None
        
        print(f"TournamentEngine initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Strategies: {[s.name for s in self.strategies]}")
    
    def _init_logger(self, ticker: str) -> logging.Logger:
        """Initialize a detailed logger for this ticker."""
        logger = logging.getLogger(f"tournament_{ticker}_{datetime.now().timestamp()}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        log_file = self._log_text_dir / f"{ticker}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        self._current_log_file = log_file
        return logger
    
    def _log_header(self, logger: logging.Logger, ticker: str, contexts: list):
        """Log header information."""
        logger.info("=" * 100)
        logger.info("GAME THEORY TOURNAMENT - DETAILED EXECUTION LOG")
        logger.info("=" * 100)
        logger.info(f"Ticker: {ticker}")
        logger.info(f"Run Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Total Samples: {len(contexts)}")
        logger.info(f"Date Range: {contexts[0].date} to {contexts[-1].date}")
        logger.info("")
        logger.info("STRATEGIES CONFIGURATION:")
        logger.info("-" * 100)
        for s in self.strategies:
            logger.info(f"  {s.name}:")
            logger.info(f"    Description: {s.description}")
            logger.info(f"    Position Scale: {getattr(s, 'position_scale', 1.0)}x")
            logger.info(f"    Initial Score: {getattr(s, 'initial_score', 0.0)}")
            if hasattr(s, 'MIN_POSITION'):
                logger.info(f"    Min Position: {s.MIN_POSITION}%")
            if hasattr(s, 'MAX_POSITION'):
                logger.info(f"    Max Position: {s.MAX_POSITION}%")
            if hasattr(s, 'POSITION_SCALE'):
                logger.info(f"    POSITION_SCALE constant: {s.POSITION_SCALE}x")
        logger.info("")
        logger.info("=" * 100)
        logger.info("")
    
    def run_ticker(
        self,
        ticker: str,
        run_monte_carlo: bool = True,
        generate_gifs: bool = False
    ) -> Dict[str, StrategyMetrics]:
        """Run tournament for a single ticker with detailed logging."""
        print(f"\n{'='*60}")
        print(f"RUNNING TOURNAMENT: {ticker}")
        print('='*60)
        
        contexts = self.loader.load_ticker_data(ticker)
        if not contexts:
            print(f"No data found for {ticker}")
            return {}
        
        print(f"Loaded {len(contexts)} samples")
        
        # Initialize logger
        logger = self._init_logger(ticker)
        self._log_header(logger, ticker, contexts)
        
        # Reset all strategies
        for s in self.strategies:
            s.reset()
        
        # ========================================
        # RUN TOURNAMENT WITH DETAILED LOGGING
        # ========================================
        print("\nExecuting tournament (logging every decision)...")
        logger.info("SAMPLE-BY-SAMPLE EXECUTION LOG")
        logger.info("=" * 100)
        
        for i, ctx in enumerate(contexts):
            sample_num = i + 1
            
            # Log sample header
            logger.info("")
            logger.info(f"{'━' * 100}")
            logger.info(f"SAMPLE {sample_num}/{len(contexts)}: {ctx.date} | {ticker}")
            logger.info(f"{'━' * 100}")
            
            # Log market context
            logger.info("")
            logger.info("┌─ MARKET DATA ─────────────────────────────────────────────────────────────────┐")
            logger.info(f"│  Daily Return: {ctx.daily_return:+.6f} ({ctx.daily_return*100:+.3f}%)")
            logger.info(f"│  Regime: {ctx.regime.upper()}")
            logger.info(f"│  Open: ${ctx.open_price:.2f}  Close: ${ctx.close_price:.2f}  Change: ${ctx.close_price - ctx.open_price:+.2f}")
            logger.info(f"│  High: ${ctx.high:.2f}  Low: ${ctx.low:.2f}  Range: ${ctx.high - ctx.low:.2f}")
            logger.info(f"│  Volume: {ctx.volume:,.0f}")
            logger.info("└───────────────────────────────────────────────────────────────────────────────┘")
            
            # Log agent evaluations
            logger.info("")
            logger.info("┌─ RISK AGENT EVALUATIONS ──────────────────────────────────────────────────────┐")
            logger.info(f"│  AGGRESSIVE:   Stance={ctx.aggressive_stance:8}  Position={ctx.aggressive_position*100:5.1f}%  Confidence={ctx.aggressive_confidence}")
            logger.info(f"│  NEUTRAL:      Stance={ctx.neutral_stance:8}  Position={ctx.neutral_position*100:5.1f}%  Confidence={ctx.neutral_confidence}")
            logger.info(f"│  CONSERVATIVE: Stance={ctx.conservative_stance:8}  Position={ctx.conservative_position*100:5.1f}%  Confidence={ctx.conservative_confidence}")
            
            # Calculate derived values
            positions = [ctx.aggressive_position, ctx.neutral_position, ctx.conservative_position]
            avg_pos = sum(positions) / 3
            std_pos = (sum((p - avg_pos)**2 for p in positions) / 3) ** 0.5
            consensus = max(0.0, 1.0 - std_pos / 0.10)
            all_bullish = all(p > 0.10 for p in positions)
            all_bearish = all(p < 0.03 for p in positions)
            
            logger.info("│  ─────────────────────────────────────────────────────────────────────────────")
            logger.info(f"│  DERIVED: Avg={avg_pos*100:.1f}%  StdDev={std_pos*100:.2f}%  Consensus={consensus*100:.0f}%")
            logger.info(f"│  FLAGS: AllBullish={all_bullish}  AllBearish={all_bearish}")
            logger.info("└───────────────────────────────────────────────────────────────────────────────┘")
            
            # Execute trades and log each strategy decision
            logger.info("")
            logger.info("┌─ STRATEGY DECISIONS ──────────────────────────────────────────────────────────┐")
            round_returns = {}
            
            for strategy in self.strategies:
                score_before = strategy.score
                equity_before = strategy.equity_curve[-1] if strategy.equity_curve else 1.0
                
                # Execute trade
                result = strategy.execute_trade(ctx)
                round_returns[strategy.name] = result.trade_return
                
                # Log detailed decision
                logger.info(f"│")
                logger.info(f"│  ▶ {strategy.name.upper()}")
                logger.info(f"│    Score: {score_before:+.1f} → {result.score_after:+.1f} (Δ {result.score_after - score_before:+.1f})")
                logger.info(f"│    Position: {result.position_pct:.1f}%")
                logger.info(f"│    Reasoning: {result.reasoning}")
                logger.info(f"│    Market Return: {ctx.daily_return*100:+.3f}%")
                logger.info(f"│    Trade Return: {result.trade_return*100:+.4f}%")
                logger.info(f"│    Equity: {equity_before:.4f} → {strategy.equity_curve[-1]:.4f}")
                logger.info(f"│    Cumulative: {strategy.total_return_pct:+.2f}%")
            
            logger.info("└───────────────────────────────────────────────────────────────────────────────┘")
            
            # Log round summary
            winner = max(round_returns, key=round_returns.get)
            loser = min(round_returns, key=round_returns.get)
            
            logger.info("")
            logger.info(f"  ROUND RESULT: Winner={winner} ({round_returns[winner]*100:+.4f}%)  Loser={loser} ({round_returns[loser]*100:+.4f}%)")
            
            # Update Tit-for-Tat
            for s in self.strategies:
                if hasattr(s, 'update_winner'):
                    s.update_winner(winner)
            
            # Progress (console)
            if (i + 1) % 20 == 0 or i == len(contexts) - 1:
                print(f"  Completed {i+1}/{len(contexts)} samples")
        
        # ========================================
        # LOG FINAL SUMMARY
        # ========================================
        logger.info("")
        logger.info("")
        logger.info("=" * 100)
        logger.info("TOURNAMENT COMPLETE - FINAL RESULTS SUMMARY")
        logger.info("=" * 100)
        logger.info("")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics: Dict[str, StrategyMetrics] = {}
        
        # Results table
        logger.info("PERFORMANCE METRICS:")
        logger.info("-" * 100)
        logger.info(f"{'Strategy':<15} {'Return':>12} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10} {'FinalScore':>12} {'Trades':>8}")
        logger.info("-" * 100)
        
        for s in self.strategies:
            m = self.metrics_calc.calculate(s)
            if m:
                metrics[s.name] = m
                print(f"  {s.name:15} | Return: {m.total_return:+7.2f}% | Sharpe: {m.sharpe_ratio:+6.3f} | Score: {m.final_score:+5.1f}")
                logger.info(f"{s.name:<15} {m.total_return:>+11.2f}% {m.sharpe_ratio:>+9.3f} {m.max_drawdown:>+9.2f}% {m.win_rate:>9.1f}% {m.final_score:>+11.1f} {s.num_trades:>8}")
        
        logger.info("-" * 100)
        logger.info("")
        
        # Regime breakdown
        logger.info("REGIME BREAKDOWN:")
        logger.info("-" * 70)
        logger.info(f"{'Strategy':<15} {'Bull Return':>15} {'Bear Return':>15} {'Sideways':>15}")
        logger.info("-" * 70)
        for name, m in metrics.items():
            logger.info(f"{name:<15} {m.bull_return:>+14.2f}% {m.bear_return:>+14.2f}% {m.sideways_return:>+14.2f}%")
        logger.info("")
        
        # Position statistics
        logger.info("POSITION STATISTICS:")
        logger.info("-" * 100)
        for s in self.strategies:
            if s.positions:
                avg_p = sum(s.positions) / len(s.positions)
                min_p = min(s.positions)
                max_p = max(s.positions)
                logger.info(f"{s.name:<15}  Avg: {avg_p:5.1f}%  Min: {min_p:5.1f}%  Max: {max_p:5.1f}%  Trades: {len(s.positions)}")
                
                # Strategy-specific
                if hasattr(s, 'get_contrarian_stats'):
                    stats = s.get_contrarian_stats()
                    logger.info(f"                 Contrarian: {stats.get('contrarian_trades', 0)} trades, {stats.get('contrarian_win_rate', 0):.1f}% win")
                    logger.info(f"                 Follow: {stats.get('follow_trades', 0)} trades, {stats.get('follow_win_rate', 0):.1f}% win")
        
        logger.info("")
        logger.info("=" * 100)
        logger.info(f"LOG FILE: {self._current_log_file}")
        logger.info("=" * 100)
        
        # Close logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        print(f"\n  ✓ Detailed log: {self._current_log_file}")
        
        # Save JSON log
        json_log_file = self._save_json_log(ticker, contexts, metrics)
        print(f"  ✓ JSON log: {json_log_file}")
        
        # Save results, visualizations, etc.
        self._save_ticker_results(ticker, metrics)
        
        print("\nGenerating visualizations...")
        ticker_viz_dir = self.output_dir / "by_ticker" / ticker
        ticker_viz_dir.mkdir(parents=True, exist_ok=True)
        ticker_viz = VisualizationEngine(ticker_viz_dir)
        
        ticker_viz.create_equity_curves(self.strategies, ticker)
        ticker_viz.create_strategy_comparison(metrics)
        ticker_viz.create_regime_heatmap(metrics)
        ticker_viz.create_score_history_chart(self.strategies, ticker)
        
        if generate_gifs:
            print("  Creating animated GIFs...")
            ticker_viz.create_equity_race_gif(self.strategies, ticker)
            ticker_viz.create_score_evolution_gif(self.strategies, ticker)
        
        if run_monte_carlo:
            print("\nRunning Monte Carlo simulations...")
            returns_dict = {s.name: s.daily_returns for s in self.strategies}
            mc_results = self.monte_carlo.compare_strategies(returns_dict)
            ticker_viz.create_monte_carlo_chart(mc_results)
            mc_dict = {name: self.monte_carlo.to_dict(r) for name, r in mc_results.items()}
            mc_path = ticker_viz_dir / "monte_carlo.json"
            with open(mc_path, 'w', encoding='utf-8') as f:
                json.dump(mc_dict, f, indent=2)
            print(f"  Saved: {mc_path}")
        
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
        """Run tournament for all available tickers (one log per ticker)."""
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
        
        if self.all_results:
            self._generate_combined_analysis()
        
        return self.all_results
    
    def _save_ticker_results(self, ticker: str, metrics: Dict[str, StrategyMetrics]):
        """Save metrics JSON for a ticker."""
        ticker_dir = self.output_dir / "by_ticker" / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        metrics_dict = {name: self.metrics_calc.to_dict(m) for name, m in metrics.items()}
        output_path = ticker_dir / "metrics.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"  Saved: {output_path}")
    
    def _save_json_log(self, ticker: str, contexts: list, metrics: Dict[str, StrategyMetrics]) -> Path:
        """Save detailed JSON log with all decisions."""
        log_data = {
            "meta": {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(contexts),
                "date_range": {
                    "start": contexts[0].date,
                    "end": contexts[-1].date
                }
            },
            "strategies_config": {},
            "samples": [],
            "summary": {}
        }
        
        # Strategy configs
        for s in self.strategies:
            log_data["strategies_config"][s.name] = {
                "description": s.description,
                "position_scale": getattr(s, 'position_scale', 1.0),
                "initial_score": getattr(s, 'initial_score', 0.0),
                "min_position": getattr(s, 'MIN_POSITION', None),
                "max_position": getattr(s, 'MAX_POSITION', None),
                "position_scale_constant": getattr(s, 'POSITION_SCALE', None)
            }
        
        # Sample-by-sample data from stored trades
        for s in self.strategies:
            s._json_trades = {t.sample_num: t for t in s.trades}
        
        for i, ctx in enumerate(contexts):
            sample_num = i + 1
            positions = [ctx.aggressive_position, ctx.neutral_position, ctx.conservative_position]
            avg_pos = sum(positions) / 3
            std_pos = (sum((p - avg_pos)**2 for p in positions) / 3) ** 0.5
            consensus = max(0.0, 1.0 - std_pos / 0.10)
            
            sample_data = {
                "sample_num": sample_num,
                "date": ctx.date,
                "market": {
                    "daily_return": round(ctx.daily_return, 6),
                    "daily_return_pct": round(ctx.daily_return * 100, 3),
                    "regime": ctx.regime,
                    "open": ctx.open_price,
                    "close": ctx.close_price,
                    "high": ctx.high,
                    "low": ctx.low,
                    "volume": ctx.volume
                },
                "agent_evaluations": {
                    "aggressive": {
                        "stance": ctx.aggressive_stance,
                        "position": round(ctx.aggressive_position * 100, 1),
                        "confidence": ctx.aggressive_confidence
                    },
                    "neutral": {
                        "stance": ctx.neutral_stance,
                        "position": round(ctx.neutral_position * 100, 1),
                        "confidence": ctx.neutral_confidence
                    },
                    "conservative": {
                        "stance": ctx.conservative_stance,
                        "position": round(ctx.conservative_position * 100, 1),
                        "confidence": ctx.conservative_confidence
                    }
                },
                "derived": {
                    "avg_position": round(avg_pos * 100, 1),
                    "std_dev": round(std_pos * 100, 2),
                    "consensus": round(consensus * 100, 0),
                    "all_bullish": all(p > 0.10 for p in positions),
                    "all_bearish": all(p < 0.03 for p in positions)
                },
                "strategy_decisions": {}
            }
            
            # Add each strategy's decision
            for s in self.strategies:
                trade = s._json_trades.get(sample_num)
                if trade:
                    sample_data["strategy_decisions"][s.name] = {
                        "position_pct": round(trade.position_pct, 1),
                        "reasoning": trade.reasoning,
                        "score_before": round(trade.score_before, 1),
                        "score_after": round(trade.score_after, 1),
                        "trade_return": round(trade.trade_return, 6),
                        "trade_return_pct": round(trade.trade_return * 100, 4)
                    }
            
            log_data["samples"].append(sample_data)
        
        # Clean up temp attribute
        for s in self.strategies:
            if hasattr(s, '_json_trades'):
                delattr(s, '_json_trades')
        
        # Summary
        for s in self.strategies:
            m = metrics.get(s.name)
            if m:
                log_data["summary"][s.name] = {
                    "total_return": round(m.total_return, 2),
                    "sharpe_ratio": round(m.sharpe_ratio, 3),
                    "max_drawdown": round(m.max_drawdown, 2),
                    "win_rate": round(m.win_rate, 1),
                    "final_score": round(m.final_score, 1),
                    "bull_return": round(m.bull_return, 2),
                    "bear_return": round(m.bear_return, 2),
                    "sideways_return": round(m.sideways_return, 2),
                    "avg_position": round(sum(s.positions)/len(s.positions), 1) if s.positions else 0,
                    "min_position": round(min(s.positions), 1) if s.positions else 0,
                    "max_position": round(max(s.positions), 1) if s.positions else 0
                }
                
                # Strategy-specific stats
                if hasattr(s, 'get_contrarian_stats'):
                    stats = s.get_contrarian_stats()
                    log_data["summary"][s.name]["contrarian_stats"] = stats
        
        # Save
        json_path = self.output_dir / "logs" / f"tournament_{ticker}_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
        
        return json_path
    
    def _generate_combined_analysis(self):
        """Generate analysis combining all tickers."""
        print(f"\n{'='*60}")
        print("GENERATING COMBINED ANALYSIS")
        print('='*60)
        
        strategy_totals: Dict[str, List[StrategyMetrics]] = {s.name: [] for s in self.strategies}
        
        for ticker, metrics in self.all_results.items():
            for name, m in metrics.items():
                if name in strategy_totals:
                    strategy_totals[name].append(m)
        
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
        
        combined_path = self.output_dir / "combined_metrics.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2)
        print(f"Saved: {combined_path}")
        
        self._create_combined_visualizations(combined)
        self._print_summary(combined)
    
    def _create_combined_visualizations(self, combined: Dict):
        """Create visualizations for combined results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        names = list(combined.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = [combined[n]['avg_return'] for n in names]
        colors = [self.viz._get_color(n) for n in names]
        bars = ax.bar(names, returns, color=colors)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Average Return (%)')
        ax.set_title('Average Return Across All Tickers', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        for bar, val in zip(bars, returns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:+.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        fig.savefig(self.output_dir / "visualizations" / "combined_returns.png", dpi=150)
        plt.close(fig)
        
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
        fig.savefig(self.output_dir / "visualizations" / "combined_regime_performance.png", dpi=150)
        plt.close(fig)
        print("Created combined visualizations")
    
    def _print_summary(self, combined: Dict):
        """Print final summary to console."""
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY")
        print('='*70)
        
        sorted_strats = sorted(combined.items(), key=lambda x: x[1]['avg_return'], reverse=True)
        
        print(f"\n{'Strategy':<15} {'Avg Return':>12} {'Avg Sharpe':>12} {'Avg Score':>10} {'Win Rate':>10}")
        print("-" * 60)
        for name, m in sorted_strats:
            print(f"{name:<15} {m['avg_return']:>+11.2f}% {m['avg_sharpe']:>+11.3f} {m['avg_score']:>+9.1f} {m['avg_win_rate']:>9.1f}%")
        
        print(f"\n{'Strategy':<15} {'Bull':>12} {'Bear':>12} {'Sideways':>12}")
        print("-" * 55)
        for name, m in sorted_strats:
            print(f"{name:<15} {m['avg_bull_return']:>+11.2f}% {m['avg_bear_return']:>+11.2f}% {m['avg_sideways_return']:>+11.2f}%")
        
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
            summary['results_by_ticker'][ticker] = {name: self.metrics_calc.to_dict(m) for name, m in metrics.items()}
        return summary


if __name__ == "__main__":
    print("Testing TournamentEngine...")
    print("=" * 60)
    try:
        engine = TournamentEngine()
        engine.loader.print_data_summary()
        tickers = engine.loader.get_available_tickers()
        if tickers:
            test_ticker = tickers[0]
            print(f"\nRunning test tournament on {test_ticker}...")
            metrics = engine.run_ticker(test_ticker, run_monte_carlo=True, generate_gifs=False)
            if metrics:
                print("\nTest completed successfully!")
                print(f"Results saved to: {engine.output_dir}")
        else:
            print("\nNo ticker data found!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    print("\n" + "=" * 60)
    print("TournamentEngine test complete!")