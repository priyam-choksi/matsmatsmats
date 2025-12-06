"""
run_complete_analysis.py - Run Complete Game Theory Analysis

One command to run everything:
1. Main tournament analysis (all tickers)
2. Monte Carlo simulations
3. Statistical significance tests
4. Generate report and dashboard

Usage:
    python run_complete_analysis.py
    python run_complete_analysis.py --quick          # Skip Monte Carlo
    python run_complete_analysis.py --ticker AAPL   # Single ticker only
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Run Complete Game Theory Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_complete_analysis.py                  # Full analysis
    python run_complete_analysis.py --quick          # Skip Monte Carlo (faster)
    python run_complete_analysis.py --ticker AAPL    # Single ticker
    python run_complete_analysis.py --mc-sims 500    # Custom MC iterations
        """
    )
    
    parser.add_argument('--ticker', type=str, default=None, help='Single ticker')
    parser.add_argument('--quick', action='store_true', help='Skip Monte Carlo')
    parser.add_argument('--mc-sims', type=int, default=500, help='Monte Carlo simulations')
    parser.add_argument('--portfolio', type=int, default=100000, help='Portfolio size')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPLETE GAME THEORY ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Single ticker' if args.ticker else 'All tickers'}")
    print(f"Monte Carlo: {'Skipped' if args.quick else f'{args.mc_sims} simulations'}")
    print("="*70 + "\n")
    
    # Import modules
    try:
        from game_theory_analysis import GameTheoryTournament
        from monte_carlo_analysis import MonteCarloSimulator
        from generate_report import ReportGenerator
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all files are in the same directory:")
        print("  - game_theory_analysis.py")
        print("  - monte_carlo_analysis.py") 
        print("  - generate_report.py")
        sys.exit(1)
    
    # Step 1: Main Tournament Analysis
    print("\n" + "="*50)
    print("STEP 1: MAIN TOURNAMENT ANALYSIS")
    print("="*50)
    
    tournament = GameTheoryTournament()
    
    if args.ticker:
        tournament.run_single_ticker(args.ticker, args.portfolio)
    else:
        tournament.run_all_tickers(args.portfolio)
    
    analysis_dir = tournament.output_dir
    
    # Step 2: Monte Carlo (unless skipped)
    if not args.quick:
        print("\n" + "="*50)
        print("STEP 2: MONTE CARLO SIMULATION")
        print("="*50)
        
        mc_output = analysis_dir / "monte_carlo"
        mc_output.mkdir(exist_ok=True)
        
        simulator = MonteCarloSimulator(output_dir=mc_output)
        
        if args.ticker:
            simulator.run_bootstrap_simulation(args.ticker, n_simulations=args.mc_sims)
            simulator.run_significance_test(args.ticker)
        else:
            simulator.run_all_tickers(n_simulations=args.mc_sims)
    else:
        print("\n[Skipping Monte Carlo - use without --quick for full analysis]")
    
    # Step 3: Generate Report
    print("\n" + "="*50)
    print("STEP 3: GENERATING REPORT")
    print("="*50)
    
    generator = ReportGenerator(analysis_dir=analysis_dir)
    generator.generate_markdown_report()
    generator.generate_summary_dashboard()
    
    # Final Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {analysis_dir}")
    print(f"\nKey outputs:")
    print(f"  📊 Visualizations: {analysis_dir / 'visualizations'}")
    print(f"  📈 By Ticker: {analysis_dir / 'by_ticker'}")
    print(f"  📋 Report: {analysis_dir / 'report' / 'analysis_report.md'}")
    print(f"  🖼️  Dashboard: {analysis_dir / 'report' / 'summary_dashboard.png'}")
    
    if not args.quick:
        print(f"  🎲 Monte Carlo: {analysis_dir / 'monte_carlo'}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()