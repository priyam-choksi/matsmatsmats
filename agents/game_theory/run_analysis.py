#!/usr/bin/env python
"""
run_analysis.py - CLI Entry Point for Game Theory Tournament

Location: agents/game_theory/run_analysis.py

UPDATED: 
- Buy-and-Hold is now EXTERNAL BENCHMARK (not tournament participant)
- Tournament has 4 strategies competing for capital
- Generates visualizations automatically

Usage:
    # Run single ticker
    python -m agents.game_theory.run_analysis --ticker AAPL
    
    # Run all tickers
    python -m agents.game_theory.run_analysis --all
    
    # List available tickers
    python -m agents.game_theory.run_analysis --list
    
    # Custom settings
    python -m agents.game_theory.run_analysis --ticker AAPL --capital 500000 --realloc 0.15
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add paths for imports
current_dir = Path(__file__).parent
agents_dir = current_dir.parent
project_root = agents_dir.parent

for path in [str(current_dir), str(agents_dir), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from game_theory.gt_engine import GTEngine
from game_theory.data_loader import DataLoader


def print_header():
    """Print banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GAME THEORY CAPITAL ALLOCATION TOURNAMENT                  ║
║                                                                              ║
║  4 Strategies compete for capital allocation based on relative performance   ║
║  Buy-and-Hold serves as EXTERNAL BENCHMARK for comparison                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_config(args):
    """Print configuration."""
    print("Configuration:")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Reallocation Rate: {args.realloc:.0%}")
    if args.ticker:
        print(f"  Ticker: {args.ticker}")
    else:
        print(f"  Mode: All tickers")
    if args.output_dir:
        print(f"  Output: {args.output_dir}")
    print()
    print("Tournament Strategies: Signal Follower, Cooperator, Defector, Tit-for-Tat")
    print("Benchmark: Buy-and-Hold (tracked separately)")
    print()


def list_tickers():
    """List available tickers."""
    print("\nSearching for available data...")
    
    try:
        loader = DataLoader()
        tickers = loader.get_available_tickers()
        
        if tickers:
            print(f"\nFound {len(tickers)} tickers with data:")
            for t in tickers:
                contexts = loader.load_ticker_data(t)
                print(f"  {t}: {len(contexts)} samples")
        else:
            print("\nNo tickers found.")
            print("Make sure you've run the data collector first.")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1


def run_tournament(args):
    """Run the tournament."""
    print_header()
    print_config(args)
    
    try:
        # Initialize engine
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        engine = GTEngine(
            total_capital=args.capital,
            reallocation_rate=args.realloc,
            output_dir=output_dir
        )
        
        # Run analysis
        if args.ticker:
            engine.run_ticker(args.ticker)
        else:
            engine.run_all_tickers()
        
        # Print completion
        print(f"\n{'=' * 70}")
        print("TOURNAMENT COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Results:  {engine.output_dir}")
        print()
        print("Output files:")
        print(f"  📝 Detailed logs: {engine.output_dir}/logs/")
        print(f"  📊 Summaries: {engine.output_dir}/summary/")
        print(f"  📈 Visualizations: {engine.output_dir}/visualizations/")
        print(f"{'=' * 70}\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you:")
        print("  1. Are running from the project root")
        print("  2. Have collected data using the data collector")
        return 1
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Game Theory Capital Allocation Tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    List available tickers
  %(prog)s --ticker AAPL             Run tournament for AAPL
  %(prog)s --all                     Run tournament for all tickers
  %(prog)s --ticker AAPL --capital 500000  Custom capital
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--ticker', '-t', type=str, help='Run for specific ticker')
    mode_group.add_argument('--all', '-a', action='store_true', help='Run for all tickers')
    mode_group.add_argument('--list', '-l', action='store_true', help='List available tickers')
    
    # Configuration
    parser.add_argument('--capital', '-c', type=float, default=1_000_000,
                       help='Total capital (default: 1,000,000)')
    parser.add_argument('--realloc', '-r', type=float, default=0.10,
                       help='Reallocation rate (default: 0.10)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Execute
    if args.list:
        return list_tickers()
    else:
        return run_tournament(args)


if __name__ == "__main__":
    sys.exit(main())