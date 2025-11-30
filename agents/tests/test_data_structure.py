"""
test_data_structure.py - Quick check of available game theory data
Run this first to see what data is available

Usage: python test_data_structure.py
"""

import json
from pathlib import Path

def find_project_root():
    """Find project root by looking for outputs folder"""
    current = Path.cwd()
    
    # Check current and parent directories
    for _ in range(5):  # Go up max 5 levels
        if (current / "outputs" / "game_theory").exists():
            return current
        if (current / "outputs").exists():
            return current
        current = current.parent
    
    return Path.cwd()


def check_data_structure():
    """Check what data is available"""
    
    print("\n" + "="*70)
    print("CHECKING GAME THEORY DATA STRUCTURE")
    print("="*70 + "\n")
    
    # Find project root
    project_root = find_project_root()
    print(f"Project root: {project_root}")
    
    base_path = project_root / "outputs" / "game_theory"
    
    if not base_path.exists():
        print(f"\nERROR: {base_path} does not exist!")
        print("Run the parallel collector first.")
        print(f"\nSearched from: {Path.cwd()}")
        return
    
    print(f"Data path: {base_path}\n")
    
    # List all tickers
    tickers = [d.name for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(tickers)} ticker(s): {tickers}\n")
    
    for ticker in tickers:
        ticker_path = base_path / ticker
        print(f"\n{'='*60}")
        print(f"TICKER: {ticker}")
        print(f"{'='*60}")
        
        # Find portfolio folders
        portfolios = [d.name for d in ticker_path.iterdir() if d.is_dir() and d.name.startswith('portfolio_')]
        print(f"Portfolios found: {portfolios}")
        
        for portfolio in portfolios:
            portfolio_path = ticker_path / portfolio
            
            # Find sample folders (could be sample_N or day_N)
            samples = []
            for folder in portfolio_path.iterdir():
                if folder.is_dir():
                    if folder.name.startswith('sample_'):
                        try:
                            num = int(folder.name.replace('sample_', ''))
                            samples.append(('sample', num, folder))
                        except:
                            pass
                    elif folder.name.startswith('day_'):
                        try:
                            num = int(folder.name.replace('day_', ''))
                            samples.append(('day', num, folder))
                        except:
                            pass
            
            samples.sort(key=lambda x: x[1])
            
            print(f"\n  {portfolio}:")
            print(f"    Total folders: {len(samples)}")
            
            if samples:
                # Check first and last
                first_type, first_num, first_path = samples[0]
                last_type, last_num, last_path = samples[-1]
                
                print(f"    Format: {first_type}_N")
                print(f"    Range: {first_num} to {last_num}")
                
                # Check files in first sample
                files = list(first_path.glob("*.json"))
                print(f"    Files in {first_type}_{first_num}: {[f.name for f in files]}")
                
                # Validate data quality
                valid_count = 0
                for _, num, path in samples[:min(10, len(samples))]:
                    has_aggressive = (path / "aggressive_eval.json").exists()
                    has_neutral = (path / "neutral_eval.json").exists()
                    has_conservative = (path / "conservative_eval.json").exists()
                    has_date = (path / "date_info.json").exists()
                    
                    if has_aggressive and has_neutral and has_conservative:
                        valid_count += 1
                
                print(f"    Valid samples (first 10): {valid_count}/10")
                
                # Show sample data structure
                if files:
                    for f in files[:1]:  # Just show first file structure
                        try:
                            with open(f, 'r') as fp:
                                data = json.load(fp)
                                print(f"\n    Sample data from {f.name}:")
                                if isinstance(data, dict):
                                    for key in list(data.keys())[:5]:
                                        val = data[key]
                                        if isinstance(val, (str, int, float)):
                                            print(f"      {key}: {str(val)[:50]}")
                                        else:
                                            print(f"      {key}: {type(val).__name__}")
                        except Exception as e:
                            print(f"    Error reading {f.name}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTo run tournament on AAPL (if available):")
    print("  python run_gt_tournament.py AAPL")
    print("\nTo run with specific portfolio:")
    print("  python run_gt_tournament.py AAPL --portfolio 100000")
    print("\nTo limit samples:")
    print("  python run_gt_tournament.py AAPL --samples 20")


if __name__ == "__main__":
    check_data_structure()