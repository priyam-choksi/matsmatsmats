#!/usr/bin/env python
"""
run_analysis.py - CLI Entry Point for Game Theory Tournament

Location: agents/game_theory/run_analysis.py

This is the command-line interface for running game theory tournaments.
Run this from your project root directory.

Usage Examples:
    # Run for single ticker
    python -m agents.game_theory.run_analysis --ticker AAPL
    
    # Run for all tickers
    python -m agents.game_theory.run_analysis --all
    
    # Quick mode (skip Monte Carlo and animated GIFs)
    python -m agents.game_theory.run_analysis --all --quick
    
    # Custom Monte Carlo iterations
    python -m agents.game_theory.run_analysis --ticker AAPL --mc-sims 5000
    
    # Skip only GIFs (keep Monte Carlo)
    python -m agents.game_theory.run_analysis --all --no-gifs
    
    # Custom output directory
    python -m agents.game_theory.run_analysis --ticker AAPL --output-dir my_results
    
    # Show available tickers
    python -m agents.game_theory.run_analysis --list

Alternative ways to run:
    # From project root
    python agents/game_theory/run_analysis.py --ticker AAPL
    
    # From agents folder
    cd agents
    python -m game_theory.run_analysis --ticker AAPL
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
# This allows running from various locations
current_dir = Path(__file__).parent
agents_dir = current_dir.parent
project_root = agents_dir.parent

for path in [str(current_dir), str(agents_dir), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Now import our modules
from game_theory.tournament_engine import TournamentEngine
from game_theory.monte_carlo_engine import MonteCarloEngine
from game_theory.data_loader import DataLoader


def print_header():
    """Print fancy header."""
    print("\n" + "=" * 70)
    print("   ██████╗  █████╗ ███╗   ███╗███████╗    ████████╗██╗  ██╗███████╗ ██████╗ ██████╗ ██╗   ██╗")
    print("  ██╔════╝ ██╔══██╗████╗ ████║██╔════╝    ╚══██╔══╝██║  ██║██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝")
    print("  ██║  ███╗███████║██╔████╔██║█████╗         ██║   ███████║█████╗  ██║   ██║██████╔╝ ╚████╔╝ ")
    print("  ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝         ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗  ╚██╔╝  ")
    print("  ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗       ██║   ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ")
    print("   ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ")
    print("=" * 70)
    print("  TRADING TOURNAMENT ANALYSIS")
    print("=" * 70)


def print_config(args):
    """Print configuration summary."""
    print(f"\n{'─' * 50}")
    print("CONFIGURATION")
    print(f"{'─' * 50}")
    print(f"  Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode:         {'Single ticker' if args.ticker else 'All tickers'}")
    if args.ticker:
        print(f"  Ticker:       {args.ticker}")
    print(f"  Monte Carlo:  {'Disabled' if args.quick else f'{args.mc_sims} simulations'}")
    print(f"  Animated GIFs: {'Disabled' if args.quick or args.no_gifs else 'Enabled'}")
    if args.output_dir:
        print(f"  Output Dir:   {args.output_dir}")
    print(f"{'─' * 50}\n")


def list_tickers():
    """List available tickers with sample counts."""
    print("\nSearching for available data...")
    
    try:
        loader = DataLoader()
        loader.print_data_summary()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you:")
        print("  1. Are running from the project root directory")
        print("  2. Have collected data using the parallel collector")
        return 1
    
    return 0


def run_tournament(args):
    """Run the tournament with given arguments."""
    print_header()
    print_config(args)
    
    try:
        # Initialize engine
        output_dir = Path(args.output_dir) if args.output_dir else None
        engine = TournamentEngine(output_dir=output_dir)
        
        # Update Monte Carlo simulations if specified
        if args.mc_sims != 1000:
            engine.monte_carlo = MonteCarloEngine(n_simulations=args.mc_sims)
            print(f"Monte Carlo set to {args.mc_sims} simulations")
        
        # Determine flags
        run_mc = not args.quick
        run_gifs = not args.quick and not args.no_gifs
        
        # Run analysis
        if args.ticker:
            # Single ticker
            engine.run_ticker(
                args.ticker,
                run_monte_carlo=run_mc,
                generate_gifs=run_gifs
            )
        else:
            # All tickers
            engine.run_all_tickers(
                run_monte_carlo=run_mc,
                generate_gifs=run_gifs
            )
        
        # Print completion
        print(f"\n{'=' * 70}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Finished:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Results:     {engine.output_dir}")
        print(f"{'=' * 70}\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you:")
        print("  1. Are running from the project root directory")
        print("  2. Have collected data using the parallel collector")
        print("  3. Have data in outputs/game_theory/TICKER/portfolio_100000/")
        return 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Game Theory Tournament Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ticker AAPL           Run for single ticker
  %(prog)s --all                   Run for all tickers
  %(prog)s --all --quick           Skip Monte Carlo and GIFs
  %(prog)s --ticker AAPL --mc-sims 5000  Custom MC iterations
  %(prog)s --list                  Show available tickers
        """
    )
    
    # Main mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--ticker', 
        type=str, 
        help='Single ticker to analyze (e.g., AAPL)'
    )
    mode_group.add_argument(
        '--all', 
        action='store_true', 
        help='Analyze all available tickers'
    )
    mode_group.add_argument(
        '--list', 
        action='store_true', 
        help='List available tickers and exit'
    )
    
    # Optional flags
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick mode: skip Monte Carlo and animated GIFs'
    )
    parser.add_argument(
        '--mc-sims', 
        type=int, 
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Custom output directory'
    )
    parser.add_argument(
        '--no-gifs', 
        action='store_true',
        help='Skip animated GIF generation (faster)'
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        return list_tickers()
    
    # Run tournament
    return run_tournament(args)


if __name__ == "__main__":
    sys.exit(main())