"""
gt_engine.py - Game Theory Tournament Engine (Capital Allocation)

Location: agents/game_theory/gt_engine.py

FEATURES:
- Organized output structure by ticker
- High-resolution dashboards and visualizations
- Combined cross-ticker analysis
- Human-readable summary tables
- CSV exports for easy analysis
"""

import json
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from .data_loader import DataLoader
from .market_context import MarketContext
from .game_state import GameState, RoundResult
from .metrics_calculator import MetricsCalculator
from .strategies import get_tournament_strategies, get_benchmark_strategy, Strategy


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class GTEngine:
    """
    Game Theory Tournament Engine
    
    Output Structure:
        outputs/game_theory_results/gt_tournament_YYYYMMDD_HHMMSS/
        ├── by_ticker/
        │   ├── AAPL/
        │   │   ├── AAPL_detailed.json
        │   │   ├── AAPL_summary.json
        │   │   ├── AAPL_trades.csv
        │   │   ├── AAPL_readable.txt
        │   │   ├── AAPL.log
        │   │   ├── AAPL_dashboard.png
        │   │   ├── AAPL_decisions.png
        │   │   └── AAPL_race.gif
        │   └── .../
        ├── combined/
        │   ├── all_tickers_summary.json
        │   ├── all_trades.csv
        │   ├── results_table.txt
        │   ├── results_table.csv
        │   ├── cross_ticker_dashboard.png
        │   └── strategy_rankings.png
        └── README.txt
    """
    
    COLORS = {
        'Buy-and-Hold': '#2c3e50',
        'Signal Follower': '#9b59b6',
        'Cooperator': '#27ae60',
        'Defector': '#e74c3c',
        'Tit-for-Tat': '#3498db',
    }
    
    def __init__(
        self,
        total_capital: float = 1_000_000,
        reallocation_rate: float = 0.10,
        output_dir: Optional[Path] = None
    ):
        """Initialize GTEngine."""
        self.total_capital = total_capital
        self.reallocation_rate = reallocation_rate
        self.loader = DataLoader()
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("outputs") / "game_theory_results" / f"gt_tournament_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "by_ticker").mkdir(exist_ok=True)
        (self.output_dir / "combined").mkdir(exist_ok=True)
        
        self.strategies = get_tournament_strategies()
        self.strategy_names = [s.name for s in self.strategies]
        self.benchmark = get_benchmark_strategy()
        self.metrics_calc = MetricsCalculator()
        self.all_results: Dict[str, dict] = {}
        
        self._write_readme()
        
        print(f"\n{'='*70}")
        print(f"  GAME THEORY TOURNAMENT ENGINE")
        print(f"{'='*70}")
        print(f"  Total capital: ${total_capital:,.0f}")
        print(f"  Reallocation rate: {reallocation_rate:.0%}")
        print(f"  Output: {self.output_dir}")
        print(f"  Tournament strategies: {self.strategy_names}")
        print(f"  Benchmark: {self.benchmark.name}")
        print(f"{'='*70}\n")
    
    def _write_readme(self):
        """Write README explaining output structure."""
        readme = f"""
GAME THEORY TOURNAMENT RESULTS
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OUTPUT STRUCTURE:
-----------------
by_ticker/          - Individual ticker results
  AAPL/
    AAPL_dashboard.png    - Main visual summary
    AAPL_readable.txt     - Human-readable round-by-round log
    AAPL_detailed.json    - Complete data for analysis
    AAPL_summary.json     - Summary metrics
    AAPL_trades.csv       - Trade data for import
    AAPL_decisions.png    - Position vs return analysis
    AAPL_race.gif         - Animated capital race

combined/           - Cross-ticker analysis
  results_table.txt       - Easy-to-read summary table
  results_table.csv       - For spreadsheet import
  all_tickers_summary.json - Complete combined data
  all_trades.csv          - All trades across tickers
  cross_ticker_dashboard.png - Visual comparison
  strategy_rankings.png   - Who won where

STRATEGIES:
-----------
{chr(10).join(f'  - {s.name}: {s.description}' for s in self.strategies)}

BENCHMARK: {self.benchmark.name}

CONFIGURATION:
--------------
  Total Capital: ${self.total_capital:,.0f}
  Reallocation Rate: {self.reallocation_rate:.0%}
  Initial Per Strategy: ${self.total_capital / len(self.strategies):,.0f}
"""
        with open(self.output_dir / "README.txt", 'w', encoding='utf-8') as f:
            f.write(readme)
    
    def _get_ticker_dir(self, ticker: str) -> Path:
        """Get/create ticker-specific output directory."""
        ticker_dir = self.output_dir / "by_ticker" / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        return ticker_dir
    
    def run_ticker(self, ticker: str) -> dict:
        """Run tournament for a single ticker."""
        print(f"\n{'='*70}")
        print(f"  TOURNAMENT: {ticker}")
        print(f"{'='*70}")
        
        contexts = self.loader.load_ticker_data(ticker)
        if not contexts:
            print(f"  ✗ No data found for {ticker}")
            return None
        
        print(f"  Loaded {len(contexts)} samples ({contexts[0].date} to {contexts[-1].date})")
        
        ticker_dir = self._get_ticker_dir(ticker)
        
        game = GameState(
            strategy_names=self.strategy_names,
            total_capital=self.total_capital,
            reallocation_rate=self.reallocation_rate
        )
        game.ticker = ticker
        
        for s in self.strategies:
            s.reset()
        self.benchmark.reset()
        
        # benchmark_returns, benchmark_equity = [], [self.total_capital]
        # capital_history = {name: [self.total_capital / len(self.strategies)] for name in self.strategy_names}
        # capital_history['Buy-and-Hold'] = [self.total_capital]
        
        benchmark_starting = self.total_capital / len(self.strategies)  # Same as each strategy: $250K
        benchmark_returns, benchmark_equity = [], [benchmark_starting]
        capital_history = {name: [benchmark_starting] for name in self.strategy_names}
        capital_history['Buy-and-Hold'] = [benchmark_starting]
        
        logger = self._init_logger(ticker, ticker_dir)
        readable_log = open(ticker_dir / f"{ticker}_readable.txt", 'w', encoding='utf-8')
        
        self._log_header(logger, ticker, contexts, game)
        self._write_readable_header(readable_log, ticker, contexts, game)
        
        detailed_rounds = []
        
        for i, ctx in enumerate(contexts):
            positions, decisions = {}, {}
            
            for s in self.strategies:
                pos = max(0, min(100, s.decide_position(ctx, game)))
                positions[s.name] = pos
                decisions[s.name] = {"position": pos, "reasoning": s.get_reasoning()}
            
            benchmark_pos = self.benchmark.decide_position(ctx, game)
            benchmark_ret = benchmark_pos / 100 * ctx.daily_return
            benchmark_returns.append(benchmark_ret)
            benchmark_equity.append(benchmark_equity[-1] * (1 + benchmark_ret))
            
            allocations_before = dict(game.allocations)
            result = game.update_round(positions=positions, market_return=ctx.daily_return, date=ctx.date, regime=ctx.regime)
            
            for name in self.strategy_names:
                capital_history[name].append(game.allocations[name])
            capital_history['Buy-and-Hold'].append(benchmark_equity[-1])
            
            self._log_round(logger, i+1, ctx, decisions, result, game, benchmark_ret)
            self._write_readable_round(readable_log, i+1, ctx, decisions, result, game, allocations_before, benchmark_equity[-2], benchmark_equity[-1], benchmark_ret)
            detailed_rounds.append(self._create_round_detail(i+1, ctx, decisions, result, game, benchmark_ret))
        
        benchmark_total_return = (benchmark_equity[-1] / benchmark_equity[0] - 1) * 100
        
        self._log_summary(logger, game, benchmark_total_return)
        self._write_readable_summary(readable_log, game, benchmark_total_return, capital_history)
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        readable_log.close()
        
        results = {
            'game': game, 
            'benchmark': {'name': self.benchmark.name, 'returns': benchmark_returns, 'equity': benchmark_equity, 'total_return_pct': benchmark_total_return, 'final_equity': benchmark_equity[-1]},
            'contexts': contexts, 
            'detailed_rounds': detailed_rounds, 
            'capital_history': capital_history
        }
        
        self._save_ticker_outputs(ticker, ticker_dir, results)
        self._generate_ticker_visualizations(ticker, ticker_dir, results)
        
        self.all_results[ticker] = results
        self._print_ticker_summary(ticker, results)
        
        return results
    
    def run_all_tickers(self) -> Dict[str, dict]:
        """Run tournament for all available tickers."""
        tickers = self.loader.get_available_tickers()
        print(f"\nFound {len(tickers)} tickers: {tickers}\n")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            self.run_ticker(ticker)
        
        print(f"\n{'='*70}")
        print(f"  GENERATING COMBINED ANALYSIS")
        print(f"{'='*70}")
        
        self._save_combined_outputs()
        self._generate_combined_visualizations()
        self._print_final_summary()
        
        return self.all_results
    
    # ==================== TICKER-LEVEL OUTPUTS ====================
    
    def _save_ticker_outputs(self, ticker: str, ticker_dir: Path, results: dict):
        """Save all outputs for a single ticker."""
        game, benchmark = results['game'], results['benchmark']
        
        # Detailed JSON
        data = {
            "meta": {"ticker": ticker, "timestamp": datetime.now().isoformat(), "total_rounds": len(results['detailed_rounds']), "benchmark": benchmark['name'], "tournament_strategies": self.strategy_names},
            "rounds": results['detailed_rounds'],
            "summary": convert_numpy_types(game.get_summary()),
            "benchmark_summary": {"total_return_pct": benchmark['total_return_pct'], "final_equity": benchmark['final_equity']}
        }
        with open(ticker_dir / f"{ticker}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(data), f, indent=2)
        
        # Summary JSON
        summary_data = game.get_summary()
        summary_data["benchmark"] = {"name": benchmark['name'], "total_return_pct": benchmark['total_return_pct'], "final_equity": benchmark['final_equity']}
        with open(ticker_dir / f"{ticker}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(summary_data), f, indent=2)
        
        # Trades CSV
        self._save_trades_csv(ticker, ticker_dir, results)
        
        print(f"  ✓ Saved outputs to: {ticker_dir}")
    
    def _save_trades_csv(self, ticker: str, ticker_dir: Path, results: dict):
        """Export trade decisions to CSV."""
        game, benchmark, contexts = results['game'], results['benchmark'], results['contexts']
        
        with open(ticker_dir / f"{ticker}_trades.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Round', 'Date', 'Ticker', 'Regime', 'Market_Return_Pct']
            for name in self.strategy_names:
                safe = name.replace(' ', '_').replace('-', '_')
                header.extend([f'{safe}_Position', f'{safe}_Return', f'{safe}_Capital'])
            header.extend(['Benchmark_Return', 'Benchmark_Capital', 'Winner'])
            writer.writerow(header)
            
            for i, rnd in enumerate(game.rounds):
                bench_ret = benchmark['returns'][i] * 100 if i < len(benchmark['returns']) else 0
                bench_cap = benchmark['equity'][i+1] if i+1 < len(benchmark['equity']) else benchmark['equity'][-1]
                
                row = [i+1, rnd.date, ticker, rnd.regime, round(rnd.market_return * 100, 4)]
                for name in self.strategy_names:
                    row.extend([round(rnd.positions.get(name, 0), 2), round(rnd.pct_returns.get(name, 0) * 100, 4), round(rnd.allocations_after.get(name, 0), 2)])
                row.extend([round(bench_ret, 4), round(bench_cap, 2), rnd.winner])
                writer.writerow(row)
    
    # ==================== COMBINED OUTPUTS ====================
    
    def _save_combined_outputs(self):
        """Save combined analysis across all tickers."""
        if not self.all_results:
            return
        
        combined_dir = self.output_dir / "combined"
        self._create_results_table(combined_dir)
        self._save_combined_json(combined_dir)
        self._save_all_trades_csv(combined_dir)
        print(f"  ✓ Combined outputs saved to: {combined_dir}")
    
    def _create_results_table(self, combined_dir: Path):
        """Create human-readable results table."""
        rows = []
        for ticker, results in self.all_results.items():
            game, benchmark = results['game'], results['benchmark']
            summary = game.get_summary()
            
            best_strat, best_ret = None, float('-inf')
            for name in self.strategy_names:
                ret = summary['total_returns_pct'].get(name, 0)
                if ret > best_ret:
                    best_ret, best_strat = ret, name
            
            total_mkt_return = sum(r.market_return for r in game.rounds) * 100
            if total_mkt_return > 5:
                trend = "BULL"
            elif total_mkt_return < -5:
                trend = "BEAR"
            else:
                trend = "SIDEWAYS"
            
            rows.append({
                'ticker': ticker, 'rounds': summary['total_rounds'], 'trend': trend,
                'market_return': total_mkt_return, 'benchmark_return': benchmark['total_return_pct'],
                'winner': best_strat, 'winner_return': best_ret,
                'beat_benchmark': best_ret > benchmark['total_return_pct'],
                **{f'{name}_return': summary['total_returns_pct'].get(name, 0) for name in self.strategy_names},
                **{f'{name}_wins': summary['wins_per_strategy'].get(name, 0) for name in self.strategy_names},
                'cooperation': summary['cooperation_rate'], 'gini': summary['allocation_gini']
            })
        
        rows.sort(key=lambda x: x['benchmark_return'], reverse=True)
        
        # Write readable table
        with open(combined_dir / "results_table.txt", 'w', encoding='utf-8') as f:
            f.write("=" * 140 + "\n")
            f.write("                              GAME THEORY TOURNAMENT - CROSS-TICKER RESULTS\n")
            f.write("=" * 140 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tickers: {len(rows)} | Strategies: {', '.join(self.strategy_names)} | Benchmark: Buy-and-Hold\n\n")
            
            f.write("-" * 140 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 140 + "\n\n")
            
            f.write(f"{'Ticker':<8} {'Trend':<8} {'Rounds':>6} {'Benchmark':>10} │")
            for name in self.strategy_names:
                f.write(f" {name[:10]:>10}")
            f.write(f" │ {'Winner':<15} {'Beat?':>5}\n")
            f.write("-" * 140 + "\n")
            
            beat_count = {name: 0 for name in self.strategy_names}
            win_count = {name: 0 for name in self.strategy_names}
            
            for row in rows:
                f.write(f"{row['ticker']:<8} {row['trend']:<8} {row['rounds']:>6} {row['benchmark_return']:>+9.2f}% │")
                for name in self.strategy_names:
                    ret = row[f'{name}_return']
                    f.write(f" {ret:>+9.2f}%")
                    if ret > row['benchmark_return']:
                        beat_count[name] += 1
                beat = "✓" if row['beat_benchmark'] else "✗"
                f.write(f" │ {row['winner']:<15} {beat:>5}\n")
                win_count[row['winner']] += 1
            
            f.write("-" * 140 + "\n\n")
            
            # Strategy Rankings
            f.write("=" * 80 + "\n")
            f.write("STRATEGY RANKINGS\n")
            f.write("=" * 80 + "\n\n")
            
            avg_returns = {name: np.mean([r[f'{name}_return'] for r in rows]) for name in self.strategy_names}
            avg_benchmark = np.mean([r['benchmark_return'] for r in rows])
            sorted_strats = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"{'Rank':<6} {'Strategy':<18} {'Avg Return':>12} {'vs Benchmark':>14} {'Ticker Wins':>12} {'Beat Bench':>12}\n")
            f.write("-" * 80 + "\n")
            
            for rank, (name, avg_ret) in enumerate(sorted_strats, 1):
                excess = avg_ret - avg_benchmark
                wins = win_count.get(name, 0)
                beats = beat_count.get(name, 0)
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
                f.write(f"{medal} {rank:<4} {name:<18} {avg_ret:>+11.2f}% {excess:>+13.2f}% {wins:>12} {beats:>8}/{len(rows)}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"       {'BENCHMARK':<18} {avg_benchmark:>+11.2f}%\n\n")
            
            # Wins by Ticker
            f.write("=" * 100 + "\n")
            f.write("WINS BY TICKER\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'Strategy':<18} │")
            for row in rows:
                f.write(f" {row['ticker']:>6}")
            f.write(f" │ {'TOTAL':>6}\n")
            f.write("-" * 100 + "\n")
            
            for name in self.strategy_names:
                f.write(f"{name:<18} │")
                total = 0
                for row in rows:
                    if row['winner'] == name:
                        f.write(f" {'★':>6}")
                        total += 1
                    else:
                        mark = "✓" if row[f'{name}_return'] > row['benchmark_return'] else "·"
                        f.write(f" {mark:>6}")
                f.write(f" │ {total:>6}\n")
            
            f.write("-" * 100 + "\n")
            f.write("Legend: ★ = Winner | ✓ = Beat Benchmark | · = Lost to Benchmark\n\n")
            
            # Key Findings
            f.write("=" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 80 + "\n\n")
            
            overall_best = sorted_strats[0]
            most_consistent = max(beat_count.items(), key=lambda x: x[1])
            most_wins = max(win_count.items(), key=lambda x: x[1])
            
            f.write(f"  📊 Best Average Return: {overall_best[0]} ({overall_best[1]:+.2f}%)\n")
            f.write(f"  🎯 Most Consistent: {most_consistent[0]} ({most_consistent[1]}/{len(rows)} tickers)\n")
            f.write(f"  🏆 Most Tournament Wins: {most_wins[0]} ({most_wins[1]}/{len(rows)} tickers)\n")
            f.write(f"  📈 Average Benchmark Return: {avg_benchmark:+.2f}%\n")
            f.write(f"  🤝 Average Cooperation Rate: {np.mean([r['cooperation'] for r in rows]):.1%}\n")
            f.write(f"  📉 Average Gini Coefficient: {np.mean([r['gini'] for r in rows]):.3f}\n")
            f.write("\n" + "=" * 140 + "\n")
        
        # CSV version
        with open(combined_dir / "results_table.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Ticker', 'Trend', 'Rounds', 'Market_Return', 'Benchmark_Return', 'Winner', 'Beat_Benchmark']
            for name in self.strategy_names:
                safe = name.replace(' ', '_')
                header.extend([f'{safe}_Return', f'{safe}_Wins'])
            header.extend(['Cooperation_Rate', 'Gini'])
            writer.writerow(header)
            
            for row in rows:
                csv_row = [row['ticker'], row['trend'], row['rounds'], round(row['market_return'], 2), round(row['benchmark_return'], 2), row['winner'], row['beat_benchmark']]
                for name in self.strategy_names:
                    csv_row.extend([round(row[f'{name}_return'], 2), row[f'{name}_wins']])
                csv_row.extend([round(row['cooperation'] * 100, 1), round(row['gini'], 4)])
                writer.writerow(csv_row)
        
        print(f"    ✓ Results table saved")
    
    def _save_combined_json(self, combined_dir: Path):
        """Save combined JSON summary."""
        combined = {
            "timestamp": datetime.now().isoformat(),
            "total_tickers": len(self.all_results),
            "tickers": list(self.all_results.keys()),
            "benchmark": self.benchmark.name,
            "tournament_strategies": self.strategy_names,
            "by_ticker": {},
            "aggregated": {}
        }
        
        all_returns = {name: [] for name in self.strategy_names}
        all_wins = {name: 0 for name in self.strategy_names}
        benchmark_returns = []
        total_rounds = 0
        
        for ticker, results in self.all_results.items():
            game, benchmark = results['game'], results['benchmark']
            summary = game.get_summary()
            summary['benchmark_return_pct'] = benchmark['total_return_pct']
            combined["by_ticker"][ticker] = convert_numpy_types(summary)
            
            benchmark_returns.append(benchmark['total_return_pct'])
            total_rounds += summary['total_rounds']
            
            for name in self.strategy_names:
                all_returns[name].append(summary['total_returns_pct'].get(name, 0))
                all_wins[name] += summary['wins_per_strategy'].get(name, 0)
        
        combined["aggregated"] = {
            "total_rounds": total_rounds,
            "avg_benchmark_return": round(np.mean(benchmark_returns), 2),
            "avg_returns": {name: round(np.mean(vals), 2) for name, vals in all_returns.items()},
            "total_wins": all_wins,
            "overall_win_rate": {name: round(wins/total_rounds*100, 1) for name, wins in all_wins.items()}
        }
        
        with open(combined_dir / "all_tickers_summary.json", 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(combined), f, indent=2)
        print(f"    ✓ Combined JSON saved")
    
    def _save_all_trades_csv(self, combined_dir: Path):
        """Save all trades across all tickers."""
        with open(combined_dir / "all_trades.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Round', 'Date', 'Ticker', 'Regime', 'Market_Return_Pct']
            for name in self.strategy_names:
                safe = name.replace(' ', '_').replace('-', '_')
                header.extend([f'{safe}_Position', f'{safe}_Return', f'{safe}_Capital'])
            header.extend(['Benchmark_Return', 'Benchmark_Capital', 'Winner'])
            writer.writerow(header)
            
            for ticker, results in self.all_results.items():
                game, benchmark = results['game'], results['benchmark']
                for i, rnd in enumerate(game.rounds):
                    bench_ret = benchmark['returns'][i] * 100 if i < len(benchmark['returns']) else 0
                    bench_cap = benchmark['equity'][i+1] if i+1 < len(benchmark['equity']) else benchmark['equity'][-1]
                    
                    row = [i+1, rnd.date, ticker, rnd.regime, round(rnd.market_return * 100, 4)]
                    for name in self.strategy_names:
                        row.extend([round(rnd.positions.get(name, 0), 2), round(rnd.pct_returns.get(name, 0) * 100, 4), round(rnd.allocations_after.get(name, 0), 2)])
                    row.extend([round(bench_ret, 4), round(bench_cap, 2), rnd.winner])
                    writer.writerow(row)
        print(f"    ✓ All trades CSV saved")
    
    
    # ==================== VISUALIZATIONS ====================
    
    def _generate_ticker_visualizations(self, ticker: str, ticker_dir: Path, results: dict):
        """Generate visualizations for a single ticker."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            self._create_ticker_dashboard(ticker, ticker_dir, results)
            self._create_decisions_scatter(ticker, ticker_dir, results)
            self._create_race_gif(ticker, ticker_dir, results)
            
            print(f"  ✓ Visualizations saved")
        except Exception as e:
            print(f"  ⚠ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_combined_visualizations(self):
        """Generate cross-ticker visualizations."""
        if not self.all_results:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            combined_dir = self.output_dir / "combined"
            self._create_cross_ticker_dashboard(combined_dir)
            self._create_strategy_rankings_chart(combined_dir)
            print(f"    ✓ Combined visualizations saved")
        except Exception as e:
            print(f"    ⚠ Combined visualization error: {e}")
            import traceback
            traceback.print_exc()

    # Strategy short names for charts (add this as class attribute)
    SHORT_NAMES = {
        'Signal Follower': 'Signal',
        'Cooperator': 'Coop',
        'Defector': 'Defect',
        'Tit-for-Tat': 'TFT',
        'Buy-and-Hold': 'B&H'
    }
    
    def _get_short_name(self, name: str) -> str:
        """Get shortened strategy name for charts."""
        return self.SHORT_NAMES.get(name, name[:8])
    
    def _shade_regimes(self, ax, contexts: list, max_x: int):
        """Add shaded background regions for market regimes."""
        regime_colors = {
            'bull': '#27ae6018',
            'bear': '#e74c3c18',
            'sideways': '#f39c1212'
        }
        if not contexts:
            return
        current_regime = contexts[0].regime if contexts else 'sideways'
        start_idx = 0
        for i, ctx in enumerate(contexts):
            regime = ctx.regime if ctx.regime in regime_colors else 'sideways'
            if regime != current_regime or i == len(contexts) - 1:
                end_idx = i if regime != current_regime else i + 1
                color = regime_colors.get(current_regime, '#f8f9fa10')
                ax.axvspan(start_idx, end_idx, facecolor=color, edgecolor='none')
                current_regime = regime
                start_idx = i
    
    def _create_ticker_dashboard(self, ticker: str, ticker_dir: Path, results: dict):
        """Create high-quality dashboard for a single ticker."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        game, benchmark = results['game'], results['benchmark']
        summary = game.get_summary()
        contexts = results['contexts']
        
        # Larger figure with better proportions
        fig = plt.figure(figsize=(22, 14), dpi=150)
        fig.patch.set_facecolor('white')
        
        # More space for right column (table)
        gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.35,
                      left=0.05, right=0.98, top=0.90, bottom=0.08,
                      width_ratios=[1, 1, 1, 0.1, 1.2])
        
        # === MAIN CHART: Capital Over Time (spans 3 columns) ===
        ax_main = fig.add_subplot(gs[0:2, 0:3])
        
        rounds = range(len(game.allocation_history))
        initial_alloc = self.total_capital / len(self.strategy_names)
        
        self._shade_regimes(ax_main, contexts, len(rounds))
        
        for name in self.strategy_names:
            allocations = [h[name] for h in game.allocation_history]
            color = self.COLORS.get(name, '#95a5a6')
            ax_main.plot(rounds, allocations, label=name, linewidth=2.5, color=color, alpha=0.9)
        
        bench_equity = benchmark['equity'][:len(rounds)+1] if len(benchmark['equity']) > len(rounds) else benchmark['equity']
        ax_main.plot(range(len(bench_equity)), bench_equity, label='Buy-and-Hold (Benchmark)', 
                    linewidth=3, color=self.COLORS['Buy-and-Hold'], linestyle='--', alpha=0.8)
        
        ax_main.axhline(y=initial_alloc, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
        ax_main.set_xlabel('Round', fontsize=12, fontweight='medium')
        ax_main.set_ylabel('Capital ($)', fontsize=12, fontweight='medium')
        ax_main.set_title(f'{ticker} - Capital Progression Through Tournament', fontsize=15, fontweight='bold', pad=10)
        ax_main.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax_main.tick_params(axis='both', labelsize=10)
        ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_main.set_facecolor('#fafafa')
        
        # === RIGHT SIDE: Final Standings Table ===
        ax_table = fig.add_subplot(gs[0:2, 4])
        ax_table.axis('off')
        
        benchmark_ret = benchmark['total_return_pct']
        strat_returns = [(name, summary['total_returns_pct'].get(name, 0)) for name in self.strategy_names]
        strat_returns.sort(key=lambda x: x[1], reverse=True)
        
        # Build table with better formatting
        table_data = []
        cell_colors = []
        
        for name, ret in strat_returns:
            excess = ret - benchmark_ret
            win_rate = summary['win_rates'].get(name, 0)
            final_alloc = summary['allocation_pcts'].get(name, 0)
            status = "Y" if excess > 0 else "N"
            
            short_name = self._get_short_name(name)
            table_data.append([short_name, f"{ret:+.1f}%", f"{excess:+.1f}%", f"{win_rate:.0f}%", f"{final_alloc:.1f}%", status])
            
            if excess > 2:
                row_color = ['#c8e6c9'] * 6
            elif excess > 0:
                row_color = ['#b3e5fc'] * 6
            elif excess > -5:
                row_color = ['#fff9c4'] * 6
            else:
                row_color = ['#ffcdd2'] * 6
            cell_colors.append(row_color)
        
        table_data.append([self._get_short_name('Buy-and-Hold'), f"{benchmark_ret:+.1f}%", "---", "---", "100%", "REF"])
        cell_colors.append(['#e0e0e0'] * 6)
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Strategy', 'Return', 'Excess', 'Wins', 'Alloc', 'Beat'],
            loc='center',
            cellLoc='center',
            colWidths=[0.22, 0.16, 0.16, 0.14, 0.16, 0.12],
            cellColours=cell_colors
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.0)
        
        for j in range(6):
            table[(0, j)].set_facecolor('#37474f')
            table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=10)
        
        ax_table.set_title('Final Standings', fontsize=14, fontweight='bold', pad=15, loc='center')
        
        # === BOTTOM ROW ===
        
        # Bottom Left: Win Distribution
        ax_wins = fig.add_subplot(gs[2, 0])
        
        wins = [summary['wins_per_strategy'].get(name, 0) for name in self.strategy_names]
        colors = [self.COLORS.get(name, '#95a5a6') for name in self.strategy_names]
        short_names = [self._get_short_name(n) for n in self.strategy_names]
        
        bars = ax_wins.bar(short_names, wins, color=colors, edgecolor='white', linewidth=2, alpha=0.9)
        ax_wins.set_ylabel('Round Wins', fontsize=11)
        ax_wins.set_title('Wins Distribution', fontsize=12, fontweight='bold', pad=8)
        ax_wins.tick_params(axis='x', labelsize=10)
        ax_wins.set_facecolor('#fafafa')
        ax_wins.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, wins):
            ax_wins.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(int(val)), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Bottom Middle-Left: vs Benchmark
        ax_perf = fig.add_subplot(gs[2, 1])
        
        excess_returns = [summary['total_returns_pct'].get(name, 0) - benchmark_ret for name in self.strategy_names]
        bar_colors = ['#43a047' if r > 0 else '#e53935' for r in excess_returns]
        
        bars = ax_perf.barh(short_names, excess_returns, color=bar_colors, edgecolor='white', linewidth=2, alpha=0.9, height=0.6)
        ax_perf.axvline(x=0, color='#212121', linewidth=2)
        ax_perf.set_xlabel('Excess Return (%)', fontsize=11)
        ax_perf.set_title('vs Benchmark', fontsize=12, fontweight='bold', pad=8)
        ax_perf.tick_params(axis='both', labelsize=10)
        ax_perf.set_facecolor('#fafafa')
        ax_perf.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, excess_returns):
            x_pos = val + (0.5 if val >= 0 else -0.5)
            ha = 'left' if val >= 0 else 'right'
            ax_perf.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%', 
                        ha=ha, va='center', fontsize=10, fontweight='bold')
        
        # Bottom Middle-Right: Metrics Box (plain ASCII)
        ax_metrics = fig.add_subplot(gs[2, 2])
        ax_metrics.axis('off')
        
        total_rounds = summary['total_rounds']
        cooperation_rate = summary['cooperation_rate']
        gini = summary['allocation_gini']
        
        winner_name = max(summary['total_returns_pct'].items(), key=lambda x: x[1])[0]
        winner_ret = summary['total_returns_pct'][winner_name]
        
        regime_counts = {}
        for ctx in contexts:
            r = ctx.regime if ctx.regime in ['bull', 'bear', 'sideways'] else 'sideways'
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        # Plain ASCII box - no special characters
        metrics_text = f"""
+----------------------------------+
|      TOURNAMENT METRICS          |
+----------------------------------+
|  Total Rounds:    {total_rounds:>12}  |
|  Cooperation:     {cooperation_rate:>11.1%}  |
|  Gini Coeff:      {gini:>12.3f}  |
+----------------------------------+
|      MARKET REGIMES              |
+----------------------------------+
|  Bull:     {regime_counts.get('bull', 0):>4} ({regime_counts.get('bull', 0)/total_rounds*100:>5.1f}%)    |
|  Bear:     {regime_counts.get('bear', 0):>4} ({regime_counts.get('bear', 0)/total_rounds*100:>5.1f}%)    |
|  Sideways: {regime_counts.get('sideways', 0):>4} ({regime_counts.get('sideways', 0)/total_rounds*100:>5.1f}%)    |
+----------------------------------+
|      WINNER                      |
|  {self._get_short_name(winner_name):<12} {winner_ret:>+10.2f}%   |
+----------------------------------+
"""
        
        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='center', horizontalalignment='center',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#bdbdbd', linewidth=2))
        
        # Bottom Right: Position Distribution Box Plot
        ax_pos = fig.add_subplot(gs[2, 4])
        
        position_data = []
        for name in self.strategy_names:
            positions = [r.positions.get(name, 50) for r in game.rounds]
            position_data.append(positions)
        
        bp = ax_pos.boxplot(position_data, labels=short_names, patch_artist=True, notch=False)
        
        for patch, name in zip(bp['boxes'], self.strategy_names):
            patch.set_facecolor(self.COLORS.get(name, '#95a5a6') + '99')
            patch.set_edgecolor(self.COLORS.get(name, '#95a5a6'))
            patch.set_linewidth(2)
        
        for whisker in bp['whiskers']:
            whisker.set(color='#666666', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='#666666', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='#000000', linewidth=2)
        
        ax_pos.set_ylabel('Position (%)', fontsize=11)
        ax_pos.set_title('Position Ranges', fontsize=12, fontweight='bold', pad=8)
        ax_pos.tick_params(axis='x', labelsize=10)
        ax_pos.set_facecolor('#fafafa')
        ax_pos.grid(True, alpha=0.3, axis='y')
        ax_pos.set_ylim(-5, 105)
        ax_pos.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Main title (no emojis)
        fig.suptitle(f'Game Theory Tournament Dashboard: {ticker}', fontsize=18, fontweight='bold', y=0.96)
        
        # Subtitle
        subtitle = f'Capital: ${self.total_capital:,.0f} | Realloc: {self.reallocation_rate:.0%} | Benchmark: Buy-and-Hold'
        fig.text(0.5, 0.92, subtitle, ha='center', fontsize=11, color='#666666', style='italic')
        
        plt.savefig(ticker_dir / f'{ticker}_dashboard.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"    [OK] Dashboard saved (high-res)")
    
    def _create_decisions_scatter(self, ticker: str, ticker_dir: Path, results: dict):
        """Create position vs return scatter plot."""
        import matplotlib.pyplot as plt
        
        game, benchmark = results['game'], results['benchmark']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
        fig.patch.set_facecolor('white')
        
        # LEFT: Position vs Return scatter
        for name in self.strategy_names:
            positions = [r.positions.get(name, 50) for r in game.rounds]
            returns = [r.pct_returns.get(name, 0) * 100 for r in game.rounds]
            color = self.COLORS.get(name, '#95a5a6')
            ax1.scatter(positions, returns, label=name, alpha=0.6, s=60, color=color, edgecolors='white', linewidth=0.5)
        
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.axvline(x=50, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax1.set_xlabel('Position (%)', fontsize=12, fontweight='medium')
        ax1.set_ylabel('Return (%)', fontsize=12, fontweight='medium')
        ax1.set_title('Position vs Return (each dot = one round)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#fafafa')
        
        # RIGHT: Risk-Return Profile
        avg_positions, total_returns = [], []
        for name in self.strategy_names:
            positions = [r.positions.get(name, 50) for r in game.rounds]
            ret_pct = sum(r.pct_returns.get(name, 0) for r in game.rounds) * 100
            avg_positions.append(np.mean(positions))
            total_returns.append(ret_pct)
        
        colors = [self.COLORS.get(name, '#95a5a6') for name in self.strategy_names]
        ax2.scatter(avg_positions, total_returns, s=400, c=colors, edgecolors='white', linewidths=3, zorder=5)
        
        for i, name in enumerate(self.strategy_names):
            ax2.annotate(name.replace(' ', '\n'), (avg_positions[i], total_returns[i]),
                        textcoords="offset points", xytext=(0, 18), ha='center', fontsize=10, fontweight='bold')
        
        bench_ret = benchmark['total_return_pct']
        ax2.scatter([100], [bench_ret], s=400, c=[self.COLORS['Buy-and-Hold']], marker='D', edgecolors='white', linewidths=3, zorder=5)
        ax2.annotate('Buy-and-\nHold', (100, bench_ret), textcoords="offset points", xytext=(0, 18), ha='center', fontsize=10, fontweight='bold')
        
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.axhline(y=bench_ret, color=self.COLORS['Buy-and-Hold'], linewidth=2, linestyle='--', alpha=0.6, label='Benchmark')
        ax2.set_xlabel('Average Position (%)', fontsize=12, fontweight='medium')
        ax2.set_ylabel('Total Return (%)', fontsize=12, fontweight='medium')
        ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.set_facecolor('#fafafa')
        
        plt.suptitle(f'{ticker} — Strategy Decision Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(ticker_dir / f'{ticker}_decisions.png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_race_gif(self, ticker: str, ticker_dir: Path, results: dict):
        """Create animated capital race GIF."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        game, benchmark = results['game'], results['benchmark']
        n_rounds = len(game.allocation_history)
        
        capital_data = {name: [h[name] for h in game.allocation_history] for name in self.strategy_names}
        capital_data['Buy-and-Hold'] = benchmark['equity'][:n_rounds+1] if len(benchmark['equity']) > n_rounds else benchmark['equity']
        
        min_len = min(len(v) for v in capital_data.values())
        for name in capital_data:
            capital_data[name] = capital_data[name][:min_len]
        
        fig, (ax_race, ax_bars) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
        fig.patch.set_facecolor('white')
        
        def animate(frame):
            ax_race.clear()
            ax_bars.clear()
            
            current = min(frame, min_len - 1)
            
            for name in self.strategy_names + ['Buy-and-Hold']:
                data = capital_data[name][:current+1]
                style = '--' if name == 'Buy-and-Hold' else '-'
                lw = 3 if name == 'Buy-and-Hold' else 2.5
                ax_race.plot(range(len(data)), data, label=name, color=self.COLORS.get(name, '#95a5a6'), linestyle=style, linewidth=lw)
                if data:
                    ax_race.scatter([len(data)-1], [data[-1]], color=self.COLORS.get(name, '#95a5a6'), s=120, zorder=5, edgecolors='white', linewidth=2)
            
            ax_race.set_xlim(0, min_len)
            y_min = min(min(v) for v in capital_data.values()) * 0.95
            y_max = max(max(v) for v in capital_data.values()) * 1.05
            ax_race.set_ylim(y_min, y_max)
            ax_race.set_xlabel('Round', fontsize=12)
            ax_race.set_ylabel('Capital ($)', fontsize=12)
            ax_race.set_title(f'Round {current + 1} of {min_len}', fontsize=14, fontweight='bold')
            ax_race.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            ax_race.legend(loc='upper left', fontsize=9)
            ax_race.grid(True, alpha=0.3)
            ax_race.set_facecolor('#fafafa')
            
            current_values = [(name, capital_data[name][current]) for name in self.strategy_names + ['Buy-and-Hold']]
            current_values.sort(key=lambda x: x[1], reverse=True)
            
            names = [x[0] for x in current_values]
            values = [x[1] for x in current_values]
            colors = [self.COLORS.get(n, '#95a5a6') for n in names]
            
            bars = ax_bars.barh(range(len(names)), values, color=colors, edgecolor='white', linewidth=2)
            ax_bars.set_yticks(range(len(names)))
            ax_bars.set_yticklabels(names, fontsize=11)
            ax_bars.set_xlabel('Capital ($)', fontsize=12)
            ax_bars.set_title('Current Standings', fontsize=14, fontweight='bold')
            ax_bars.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            ax_bars.set_facecolor('#fafafa')
            
            for bar, val in zip(bars, values):
                ax_bars.text(val + 5000, bar.get_y() + bar.get_height()/2,
                           f'${val/1000:.0f}K', ha='left', va='center', fontsize=10, fontweight='bold')
            
            plt.suptitle(f'{ticker} — Capital Race', fontsize=16, fontweight='bold')
            plt.tight_layout()
        
        frames = list(range(0, min_len, max(1, min_len // 50))) + [min_len - 1]
        anim = FuncAnimation(fig, animate, frames=frames, interval=100)
        anim.save(ticker_dir / f'{ticker}_race.gif', writer=PillowWriter(fps=10))
        plt.close()
        print(f"    ✓ Race animation saved")
    
    def _create_cross_ticker_dashboard(self, combined_dir: Path):
        """Create cross-ticker comparison dashboard."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(24, 16), dpi=150)
        fig.patch.set_facecolor('white')
        
        gs = GridSpec(3, 4, figure=fig, hspace=0.32, wspace=0.28,
                      left=0.05, right=0.95, top=0.91, bottom=0.06)
        
        tickers = list(self.all_results.keys())
        
        # Collect data
        data = []
        for ticker in tickers:
            results = self.all_results[ticker]
            game, benchmark = results['game'], results['benchmark']
            summary = game.get_summary()
            
            total_mkt = sum(r.market_return for r in game.rounds) * 100
            trend = "BULL" if total_mkt > 5 else ("BEAR" if total_mkt < -5 else "SIDE")
            
            data.append({
                'ticker': ticker,
                'trend': trend,
                'benchmark': benchmark['total_return_pct'],
                **{name: summary['total_returns_pct'].get(name, 0) for name in self.strategy_names}
            })
        
        # === TOP: Returns by Ticker ===
        ax1 = fig.add_subplot(gs[0, 0:3])
        
        x = np.arange(len(tickers))
        width = 0.15
        all_names = ['Buy-and-Hold'] + self.strategy_names
        n_strats = len(all_names)
        
        for i, name in enumerate(all_names):
            key = 'benchmark' if name == 'Buy-and-Hold' else name
            values = [d[key] for d in data]
            offset = (i - n_strats/2 + 0.5) * width
            color = self.COLORS.get(name, '#95a5a6')
            ax1.bar(x + offset, values, width, label=self._get_short_name(name), color=color, edgecolor='white', linewidth=1)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{d['ticker']}\n({d['trend']})" for d in data], fontsize=10)
        ax1.set_ylabel('Return (%)', fontsize=12)
        ax1.set_title('Strategy Returns by Ticker', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=9, ncol=3)
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor('#fafafa')
        
        # === TOP RIGHT: Win Distribution Pie ===
        ax2 = fig.add_subplot(gs[0, 3])
        
        win_counts = {name: 0 for name in self.strategy_names}
        for ticker, results in self.all_results.items():
            summary = results['game'].get_summary()
            best = max(summary['total_returns_pct'].items(), key=lambda x: x[1])[0]
            win_counts[best] += 1
        
        labels = [f"{self._get_short_name(n)}\n({c})" for n, c in win_counts.items() if c > 0]
        sizes = [c for c in win_counts.values() if c > 0]
        colors = [self.COLORS.get(n, '#95a5a6') for n, c in win_counts.items() if c > 0]
        
        if sizes:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', 
                                               startangle=90, textprops={'fontsize': 10},
                                               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            ax2.set_title('Tournament Wins', fontsize=13, fontweight='bold', pad=10)
        
        # === MIDDLE: Excess Returns Heatmap ===
        ax3 = fig.add_subplot(gs[1, :])
        
        excess_data = np.array([[d[name] - d['benchmark'] for name in self.strategy_names] for d in data])
        vmax = max(abs(excess_data.min()), abs(excess_data.max()), 10)
        
        im = ax3.imshow(excess_data.T, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
        ax3.set_xticks(range(len(tickers)))
        ax3.set_xticklabels(tickers, fontsize=11)
        ax3.set_yticks(range(len(self.strategy_names)))
        ax3.set_yticklabels([self._get_short_name(n) for n in self.strategy_names], fontsize=11)
        ax3.set_title('Excess Return vs Benchmark (%)  |  Green = Beat  |  Red = Lost', fontsize=13, fontweight='bold', pad=10)
        
        for i in range(len(self.strategy_names)):
            for j in range(len(tickers)):
                val = excess_data[j, i]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax3.text(j, i, f'{val:+.1f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax3, shrink=0.6, pad=0.02)
        cbar.ax.set_ylabel('Excess Return (%)', fontsize=10)
        
        # === BOTTOM LEFT: Average Returns ===
        ax4 = fig.add_subplot(gs[2, 0])
        
        avg_returns = {name: np.mean([d[name] for d in data]) for name in self.strategy_names}
        avg_benchmark = np.mean([d['benchmark'] for d in data])
        
        all_avgs = [(n, avg_benchmark if n == 'Buy-and-Hold' else avg_returns[n]) for n in all_names]
        all_avgs.sort(key=lambda x: x[1], reverse=True)
        
        bar_names = [self._get_short_name(x[0]) for x in all_avgs]
        bar_vals = [x[1] for x in all_avgs]
        bar_colors = [self.COLORS.get(x[0], '#95a5a6') for x in all_avgs]
        
        bars = ax4.barh(bar_names, bar_vals, color=bar_colors, edgecolor='white', linewidth=2, height=0.6)
        ax4.axvline(x=0, color='black', linewidth=1)
        ax4.set_xlabel('Avg Return (%)', fontsize=11)
        ax4.set_title('Average Return', fontsize=12, fontweight='bold', pad=8)
        ax4.set_facecolor('#fafafa')
        ax4.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, bar_vals):
            x_pos = val + 0.3 if val >= 0 else val - 0.3
            ax4.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%', 
                    ha='left' if val >= 0 else 'right', va='center', fontsize=10, fontweight='bold')
        
        # === BOTTOM MIDDLE-LEFT: Beat Benchmark Count ===
        ax5 = fig.add_subplot(gs[2, 1])
        
        beat_counts = {name: sum(1 for d in data if d[name] > d['benchmark']) for name in self.strategy_names}
        
        bc_names = [self._get_short_name(n) for n in self.strategy_names]
        bc_vals = [beat_counts[n] for n in self.strategy_names]
        bc_colors = [self.COLORS.get(n, '#95a5a6') for n in self.strategy_names]
        
        bars = ax5.bar(bc_names, bc_vals, color=bc_colors, edgecolor='white', linewidth=2)
        ax5.axhline(y=len(tickers)/2, color='#666666', linestyle='--', alpha=0.7, linewidth=2)
        ax5.set_ylabel(f'Beat Count (/{len(tickers)})', fontsize=11)
        ax5.set_title('Consistency', fontsize=12, fontweight='bold', pad=8)
        ax5.set_facecolor('#fafafa')
        ax5.set_ylim(0, len(tickers) + 1)
        
        for bar, val in zip(bars, bc_vals):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(val), ha='center', fontsize=11, fontweight='bold')
        
        # === BOTTOM MIDDLE-RIGHT: Rankings ===
        ax6 = fig.add_subplot(gs[2, 2])
        
        rankings = {name: [] for name in self.strategy_names}
        for ticker, results in self.all_results.items():
            summary = results['game'].get_summary()
            sorted_strats = sorted(summary['total_returns_pct'].items(), key=lambda x: x[1], reverse=True)
            for rank, (name, _) in enumerate(sorted_strats, 1):
                rankings[name].append(rank)
        
        avg_ranks = sorted([(name, np.mean(ranks)) for name, ranks in rankings.items()], key=lambda x: x[1])
        
        rank_names = [self._get_short_name(x[0]) for x in avg_ranks]
        rank_vals = [x[1] for x in avg_ranks]
        rank_colors = [self.COLORS.get(x[0], '#95a5a6') for x in avg_ranks]
        
        bars = ax6.barh(rank_names, rank_vals, color=rank_colors, edgecolor='white', linewidth=2, height=0.6)
        ax6.set_xlabel('Avg Rank (lower=better)', fontsize=11)
        ax6.set_title('Rankings', fontsize=12, fontweight='bold', pad=8)
        ax6.set_xlim(0, len(self.strategy_names) + 0.5)
        ax6.set_facecolor('#fafafa')
        ax6.grid(True, alpha=0.3, axis='x')
        
        medals = ['1st', '2nd', '3rd', '4th']
        for i, (bar, val) in enumerate(zip(bars, rank_vals)):
            medal = medals[i] if i < len(medals) else ''
            ax6.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{medal} ({val:.2f})', 
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        # === BOTTOM RIGHT: Summary ===
        ax7 = fig.add_subplot(gs[2, 3])
        ax7.axis('off')
        
        best_avg = max(avg_returns.items(), key=lambda x: x[1])
        most_consistent = max(beat_counts.items(), key=lambda x: x[1])
        most_wins = max(win_counts.items(), key=lambda x: x[1])
        best_rank = avg_ranks[0]
        
        summary_text = f"""
+------------------------------------+
|        TOURNAMENT SUMMARY          |
+------------------------------------+
|                                    |
|  Best Avg Return:                  |
|    {self._get_short_name(best_avg[0]):<14} {best_avg[1]:>+8.2f}%     |
|                                    |
|  Most Consistent:                  |
|    {self._get_short_name(most_consistent[0]):<14} {most_consistent[1]:>3}/{len(tickers)} beat     |
|                                    |
|  Most Wins:                        |
|    {self._get_short_name(most_wins[0]):<14} {most_wins[1]:>3}/{len(tickers)} won      |
|                                    |
|  Best Rank:                        |
|    {self._get_short_name(best_rank[0]):<14} {best_rank[1]:>8.2f}      |
|                                    |
|  Benchmark Avg:   {avg_benchmark:>+8.2f}%         |
|  Total Tickers:   {len(tickers):>8}          |
+------------------------------------+
"""
        
        ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=11, 
                verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#757575', linewidth=2))
        
        fig.suptitle('Game Theory Tournament - Cross-Ticker Analysis', fontsize=20, fontweight='bold', y=0.96)
        subtitle = f'{len(tickers)} Tickers | {len(self.strategy_names)} Strategies | Benchmark: Buy-and-Hold'
        fig.text(0.5, 0.925, subtitle, ha='center', fontsize=12, color='#666666', style='italic')
        
        plt.savefig(combined_dir / 'cross_ticker_dashboard.png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    [OK] Cross-ticker dashboard saved (high-res)")
    
    def _create_strategy_rankings_chart(self, combined_dir: Path):
        """Create strategy rankings visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        fig.patch.set_facecolor('white')
        
        tickers = list(self.all_results.keys())
        
        rankings = {name: [] for name in self.strategy_names}
        for ticker, results in self.all_results.items():
            summary = results['game'].get_summary()
            sorted_strats = sorted(summary['total_returns_pct'].items(), key=lambda x: x[1], reverse=True)
            for rank, (name, _) in enumerate(sorted_strats, 1):
                rankings[name].append(rank)
        
        # 1. Average Rank
        ax1 = axes[0, 0]
        avg_ranks = {name: np.mean(ranks) for name, ranks in rankings.items()}
        sorted_by_rank = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        names = [x[0] for x in sorted_by_rank]
        ranks = [x[1] for x in sorted_by_rank]
        colors = [self.COLORS.get(n, '#95a5a6') for n in names]
        
        bars = ax1.barh(names, ranks, color=colors, edgecolor='white', linewidth=2)
        ax1.set_xlabel('Average Rank (lower is better)', fontsize=12)
        ax1.set_title('Average Ranking Across Tickers', fontsize=14, fontweight='bold')
        ax1.set_facecolor('#fafafa')
        ax1.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, ranks)):
            medals = ['🥇', '🥈', '🥉', '']
            medal = medals[i] if i < 3 else ''
            ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{medal} {val:.2f}', va='center', fontsize=11, fontweight='bold')
        
        # 2. Rank Distribution (box plot)
        ax2 = axes[0, 1]
        rank_data = [rankings[name] for name in self.strategy_names]
        bp = ax2.boxplot(rank_data, labels=[n[:10] for n in self.strategy_names], patch_artist=True, notch=True)
        
        for patch, name in zip(bp['boxes'], self.strategy_names):
            patch.set_facecolor(self.COLORS.get(name, '#95a5a6') + '80')
            patch.set_edgecolor(self.COLORS.get(name, '#95a5a6'))
            patch.set_linewidth(2)
        
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
        
        ax2.set_ylabel('Rank', fontsize=12)
        ax2.set_title('Rank Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylim(0.5, len(self.strategy_names) + 0.5)
        ax2.invert_yaxis()
        ax2.set_facecolor('#fafafa')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. First Place Finishes
        ax3 = axes[1, 0]
        first_places = {name: sum(1 for r in ranks if r == 1) for name, ranks in rankings.items()}
        
        fp_names = list(first_places.keys())
        fp_counts = list(first_places.values())
        fp_colors = [self.COLORS.get(n, '#95a5a6') for n in fp_names]
        
        bars = ax3.bar(fp_names, fp_counts, color=fp_colors, edgecolor='white', linewidth=2)
        ax3.set_ylabel('# First Place Finishes', fontsize=12)
        ax3.set_title('Tournament Victories', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=20)
        ax3.set_facecolor('#fafafa')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, fp_counts):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(val), ha='center', fontsize=12, fontweight='bold')
        
        # 4. Rank Timeline
        ax4 = axes[1, 1]
        
        for name in self.strategy_names:
            color = self.COLORS.get(name, '#95a5a6')
            ax4.plot(range(len(tickers)), rankings[name], 'o-', label=name, color=color, linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        ax4.set_xticks(range(len(tickers)))
        ax4.set_xticklabels(tickers, rotation=45, ha='right', fontsize=10)
        ax4.set_ylabel('Rank', fontsize=12)
        ax4.set_title('Rank by Ticker', fontsize=14, fontweight='bold')
        ax4.set_ylim(0.5, len(self.strategy_names) + 0.5)
        ax4.invert_yaxis()
        ax4.legend(loc='upper right', fontsize=10)
        ax4.set_facecolor('#fafafa')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Strategy Rankings Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(combined_dir / 'strategy_rankings.png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    ✓ Strategy rankings chart saved")
    
    # === Logging Methods ===
    
    def _init_logger(self, ticker: str, ticker_dir: Path) -> logging.Logger:
        logger = logging.getLogger(f"gt_{ticker}_{datetime.now().timestamp()}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        handler = logging.FileHandler(ticker_dir / f"{ticker}.log", encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        return logger
    
    def _log_header(self, logger, ticker, contexts, game):
        logger.info("=" * 100)
        logger.info("GAME THEORY CAPITAL ALLOCATION TOURNAMENT")
        logger.info("=" * 100)
        logger.info(f"Ticker: {ticker}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Total Samples: {len(contexts)}")
        logger.info(f"Date Range: {contexts[0].date} to {contexts[-1].date}")
        logger.info("")
        logger.info("GAME CONFIGURATION:")
        logger.info(f"  Total Capital: ${game.total_capital:,.0f}")
        logger.info(f"  Reallocation Rate: {game.reallocation_rate:.0%}")
        logger.info(f"  Initial Allocation: ${game.total_capital / len(self.strategies):,.0f} per strategy")
        logger.info("")
        logger.info("TOURNAMENT STRATEGIES:")
        for s in self.strategies:
            logger.info(f"  {s.name}: {s.description}")
        logger.info("")
        logger.info(f"BENCHMARK: {self.benchmark.name}")
        logger.info("=" * 100)
        logger.info("")
    
    def _log_round(self, logger, round_num, ctx, decisions, result, game, benchmark_ret):
        logger.info(f"ROUND {round_num}: {ctx.date} | Regime: {ctx.regime}")
        logger.info("-" * 80)
        logger.info(f"Market Return: {ctx.daily_return*100:+.2f}%")
        logger.info(f"Benchmark Return: {benchmark_ret*100:+.2f}%")
        logger.info("")
        logger.info("STRATEGY DECISIONS:")
        for name, dec in decisions.items():
            logger.info(f"  {name}: {dec['position']:.1f}% - {dec['reasoning']}")
        logger.info("")
        logger.info(f"WINNER: {result.winner}")
        logger.info("")
    
    def _log_summary(self, logger, game, benchmark_return):
        summary = game.get_summary()
        logger.info("=" * 100)
        logger.info("TOURNAMENT SUMMARY")
        logger.info("=" * 100)
        logger.info(f"BENCHMARK ({self.benchmark.name}): {benchmark_return:+.2f}%")
        logger.info("")
        logger.info("TOURNAMENT RESULTS:")
        for name, ret in sorted(summary['total_returns_pct'].items(), key=lambda x: -x[1]):
            excess = ret - benchmark_return
            beat = "BEAT" if excess > 0 else "LOST TO"
            logger.info(f"  {name}: {ret:+.2f}% ({beat} benchmark by {excess:+.2f}%)")
        logger.info("=" * 100)
    
    def _write_readable_header(self, f, ticker, contexts, game):
        initial_alloc = self.total_capital / len(self.strategies)
        f.write("=" * 90 + "\n")
        f.write("                    GAME THEORY TOURNAMENT - DETAILED RESULTS\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Date Range: {contexts[0].date} to {contexts[-1].date}\n")
        f.write(f"Total Rounds: {len(contexts)}\n\n")
        f.write("STARTING CAPITAL:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total Pool: ${self.total_capital:,.0f}\n\n")
        f.write("  Tournament Strategies (competing for capital):\n")
        for name in self.strategy_names:
            f.write(f"    * {name}: ${initial_alloc:,.0f}\n")
        # f.write(f"\n  Benchmark (tracked separately):\n")
        # f.write(f"    * Buy-and-Hold: ${self.total_capital:,.0f}\n")
        # 250k change
        f.write(f"\n  Benchmark (tracked separately, same starting capital):\n")
        f.write(f"    * Buy-and-Hold: ${initial_alloc:,.0f}\n")
        f.write("\n" + "=" * 90 + "\n\n")
    
    def _write_readable_round(self, f, round_num, ctx, decisions, result, game, alloc_before, bench_before, bench_after, benchmark_ret):
        market_dir = "UP" if ctx.daily_return >= 0 else "DOWN"
        f.write(f"ROUND {round_num}: {ctx.date}\n")
        f.write(f"Market: {market_dir} {ctx.daily_return*100:+.2f}% | Regime: {ctx.regime}\n")
        f.write("-" * 90 + "\n\n")
        
        f.write("+------------------+------------+--------------+--------------+--------------+--------------+\n")
        f.write("| Strategy         |  Position  | Capital Start|  $ Invested  |   $ Return   |  Capital End |\n")
        f.write("+------------------+------------+--------------+--------------+--------------+--------------+\n")
        
        returns_list = [(name, result.pct_returns[name]) for name in self.strategy_names]
        returns_list.append(('Buy-and-Hold', benchmark_ret))
        sorted_returns = sorted(returns_list, key=lambda x: x[1], reverse=True)
        winner = sorted_returns[0][0]
        loser = sorted_returns[-1][0]
        
        for name in self.strategy_names:
            pos = decisions[name]["position"]
            cap_start = alloc_before[name]
            invested = cap_start * (pos / 100)
            dollar_ret = result.dollar_returns[name]
            cap_end = game.allocations[name]
            marker = " <-- WINNER" if name == winner else (" <-- Worst" if name == loser else "")
            f.write(f"| {name:<16} | {pos:>8.1f}%  | ${cap_start:>10,.0f} | ${invested:>10,.0f} | ${dollar_ret:>+10,.0f} | ${cap_end:>10,.0f} |{marker}\n")
        
        f.write("+------------------+------------+--------------+--------------+--------------+--------------+\n")
        
        bench_invested = bench_before
        bench_dollar_ret = bench_after - bench_before
        marker = " <-- WINNER" if 'Buy-and-Hold' == winner else (" <-- Worst" if 'Buy-and-Hold' == loser else "")
        f.write(f"| {'Buy-and-Hold':<16} | {'100.0%':>10} | ${bench_before:>10,.0f} | ${bench_invested:>10,.0f} | ${bench_dollar_ret:>+10,.0f} | ${bench_after:>10,.0f} |{marker}\n")
        f.write("+------------------+------------+--------------+--------------+--------------+--------------+\n\n")
        
        f.write("Strategy Reasoning:\n")
        for name, dec in decisions.items():
            reasoning = dec["reasoning"][:77] + "..." if len(dec["reasoning"]) > 80 else dec["reasoning"]
            f.write(f"  * {name}: {reasoning}\n")
        f.write("\n" + "=" * 90 + "\n\n")
    
    def _write_readable_summary(self, f, game, benchmark_return, capital_history):
        summary = game.get_summary()
        f.write("\n" + "=" * 90 + "\n")
        f.write("                              FINAL RESULTS\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("FINAL CAPITAL STANDINGS:\n")
        f.write("+------------------+--------------+--------------+--------------+-------------+----------+\n")
        f.write("| Strategy         | Start Capital|  End Capital |    Change    | Return %    | Win Rate |\n")
        f.write("+------------------+--------------+--------------+--------------+-------------+----------+\n")
        
        initial_alloc = self.total_capital / len(self.strategy_names)
        results = []
        for name in self.strategy_names:
            end_cap = game.allocations[name]
            change = end_cap - initial_alloc
            ret_pct = summary['total_returns_pct'].get(name, 0)
            win_rate = summary['win_rates'].get(name, 0)
            results.append((name, initial_alloc, end_cap, change, ret_pct, win_rate))
        
        # results.append(('Buy-and-Hold', self.total_capital, capital_history['Buy-and-Hold'][-1], capital_history['Buy-and-Hold'][-1] - self.total_capital, benchmark_return, 0))
        benchmark_starting = self.total_capital / len(self.strategy_names)
        results.append(('Buy-and-Hold', benchmark_starting, capital_history['Buy-and-Hold'][-1], capital_history['Buy-and-Hold'][-1] - benchmark_starting, benchmark_return, 0))
        results.sort(key=lambda x: x[4], reverse=True)
        
        for name, start, end, change, ret_pct, win_rate in results:
            if name == 'Buy-and-Hold':
                f.write("+------------------+--------------+--------------+--------------+-------------+----------+\n")
                f.write(f"| {name:<16} | ${start:>10,.0f} | ${end:>10,.0f} | ${change:>+10,.0f} | {ret_pct:>+9.2f}% | {'N/A':>8} | <-- BENCHMARK\n")
            else:
                excess = ret_pct - benchmark_return
                beat = "[Y]" if excess > 0 else "[N]"
                f.write(f"| {name:<16} | ${start:>10,.0f} | ${end:>10,.0f} | ${change:>+10,.0f} | {ret_pct:>+9.2f}% | {win_rate:>7.1f}% | {beat} vs Benchmark\n")
        
        f.write("+------------------+--------------+--------------+--------------+-------------+----------+\n\n")
        
        f.write("PERFORMANCE VS BENCHMARK:\n")
        f.write("-" * 60 + "\n")
        for name in self.strategy_names:
            ret = summary['total_returns_pct'].get(name, 0)
            excess = ret - benchmark_return
            status = "BEAT" if excess > 0 else "LOST TO"
            symbol = "[Y]" if excess > 0 else "[N]"
            f.write(f"  {symbol} {name}: {status} benchmark by {excess:+.2f}%\n")
        
        f.write("\nGAME THEORY DYNAMICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Cooperation Rate: {summary['cooperation_rate']:.1%}\n")
        f.write(f"  Gini Coefficient: {summary['allocation_gini']:.3f}\n")
        f.write("\n" + "=" * 90 + "\n")
    
    def _create_round_detail(self, round_num, ctx, decisions, result, game, benchmark_ret):
        return convert_numpy_types({
            "round_num": round_num,
            "date": ctx.date,
            "regime": ctx.regime,
            "market": {"daily_return": round(float(ctx.daily_return), 6), "daily_return_pct": round(float(ctx.daily_return) * 100, 4)},
            "benchmark_return_pct": round(float(benchmark_ret) * 100, 4),
            "llm_signals": {
                "aggressive": {"stance": ctx.aggressive_stance, "position": round(float(ctx.aggressive_position) * 100, 2), "confidence": ctx.aggressive_confidence},
                "neutral": {"stance": ctx.neutral_stance, "position": round(float(ctx.neutral_position) * 100, 2), "confidence": ctx.neutral_confidence},
                "conservative": {"stance": ctx.conservative_stance, "position": round(float(ctx.conservative_position) * 100, 2), "confidence": ctx.conservative_confidence}
            },
            "strategy_decisions": {name: {"position_pct": round(float(dec["position"]), 2), "reasoning": dec["reasoning"]} for name, dec in decisions.items()},
            "round_results": {
                "winner": result.winner,
                "high_variance": bool(result.high_variance_round),
                "returns_pct": {k: round(float(v)*100, 4) for k, v in result.pct_returns.items()},
                "dollar_returns": {k: round(float(v), 2) for k, v in result.dollar_returns.items()},
                "allocations_after": {k: round(float(v), 2) for k, v in result.allocations_after.items()},
            }
        })
    
    def _print_ticker_summary(self, ticker, results):
        game, benchmark = results['game'], results['benchmark']
        summary = game.get_summary()
        print(f"\n  📊 Results for {ticker}:")
        print(f"     Benchmark: {benchmark['total_return_pct']:+.2f}%")
        for name, ret in sorted(summary['total_returns_pct'].items(), key=lambda x: -x[1]):
            excess = ret - benchmark['total_return_pct']
            status = "✓" if excess > 0 else "✗"
            print(f"     {status} {name}: {ret:+.2f}% (vs bench: {excess:+.2f}%)")
    
    def _print_final_summary(self):
        print(f"\n{'='*70}")
        print(f"  TOURNAMENT COMPLETE")
        print(f"{'='*70}")
        print(f"  Output Directory: {self.output_dir}")
        print(f"\n  Structure:")
        print(f"     by_ticker/     - Individual ticker results")
        print(f"     combined/      - Cross-ticker analysis")
        print(f"\n  Key Files:")
        print(f"     combined/results_table.txt")
        print(f"     combined/cross_ticker_dashboard.png")
        print(f"     combined/strategy_rankings.png")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    engine = GTEngine()
    tickers = engine.loader.get_available_tickers()
    if tickers:
        engine.run_all_tickers()
    else:
        print("No data available.")