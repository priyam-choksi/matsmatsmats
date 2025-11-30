"""
collect_game_theory_data.py - Collects data without touching main outputs folder
Uses temporary folder inside game_theory directory
Usage: python agents/orchestrators/collect_game_theory_data.py AAPL --days 5
"""

import os
import sys
import json
import shutil
import subprocess
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time

class GameTheoryDataCollector:
    def __init__(self, ticker: str, days: int = 5):
        self.ticker = ticker.upper()
        self.days = days
        self.portfolio_sizes = [10000, 100000, 1000000]
        
        # Setup paths - handle being run from orchestrators folder
        current = Path.cwd()
        if current.name == 'orchestrators':
            self.project_root = current.parent.parent
        elif current.name == 'agents':
            self.project_root = current.parent
        elif (current / 'agents').exists():
            self.project_root = current
        else:
            # Try to find project root
            temp = current
            while temp.parent != temp:
                if (temp / 'agents' / 'orchestrators').exists():
                    self.project_root = temp
                    break
                temp = temp.parent
            else:
                self.project_root = current
        
        # Main outputs path (we'll monitor but not touch)
        self.main_outputs_path = self.project_root / "outputs"
        
        # Game theory specific paths
        self.game_theory_path = self.main_outputs_path / "game_theory" / self.ticker
        
        # Temporary outputs path INSIDE game_theory folder
        self.temp_outputs_path = self.game_theory_path / "temp_workflow"
        
        # Files we need to save from Phase 4
        self.required_files = [
            "aggressive_eval.json",
            "neutral_eval.json",
            "conservative_eval.json"
        ]
        
        # Track collection stats
        self.stats = {
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Get historical prices first
        self.prices = self.fetch_historical_prices()
    
    def fetch_historical_prices(self):
        """Get historical price data for the ticker"""
        print(f"Fetching last {self.days} trading days for {self.ticker}...")
        
        try:
            stock = yf.Ticker(self.ticker)
            end_date = datetime.now()
            # Get extra days in case of weekends/holidays
            start_date = end_date - timedelta(days=self.days * 2 + 10)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                raise ValueError(f"No price data found for {self.ticker}")
            
            # Get last N trading days
            prices = hist.tail(self.days)
            
            # Process the price data
            price_data = []
            for date, row in prices.iterrows():
                price_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(row['Open'], 2),
                    'close': round(row['Close'], 2),
                    'high': round(row['High'], 2),
                    'low': round(row['Low'], 2),
                    'volume': int(row['Volume']),
                    'daily_return': round((row['Close'] - row['Open']) / row['Open'], 4)
                })
            
            print(f"  Found {len(price_data)} days of price data")
            print(f"  Date range: {price_data[0]['date']} to {price_data[-1]['date']}")
            return price_data
            
        except Exception as e:
            print(f"Error fetching price data: {e}")
            raise
    
    def setup_temp_folder(self):
        """Create temporary folder for workflow outputs"""
        # Create temp folder if it doesn't exist
        self.temp_outputs_path.mkdir(parents=True, exist_ok=True)
        print(f"  Using temp folder: {self.temp_outputs_path}")
    
    def run_workflow(self, portfolio_value: float, attempt: int = 1) -> bool:
        """Run the master orchestrator workflow"""
        max_attempts = 2
        
        # Setup temp folder for this run
        self.setup_temp_folder()
        
        # Find the master orchestrator script
        master_script = self.project_root / "agents" / "orchestrators" / "master_orchestrator.py"
        
        if not master_script.exists():
            print(f"    Error: Cannot find master_orchestrator.py at {master_script}")
            return False
        
        # Use the correct arguments for YOUR master orchestrator
        cmd = [
            sys.executable,
            str(master_script),
            self.ticker,
            "--portfolio-value", str(portfolio_value),
            "--research-mode", "shallow",
            "--research-rounds", "1"
            # NOT including --run-all to skip Phase 6 (game theory)
        ]
        
        print(f"  Running workflow (attempt {attempt}/{max_attempts})...")
        
        try:
            # Set environment for UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.project_root),
                encoding='utf-8',
                errors='replace',
                env=env
            )
            
            # Add a small delay to ensure files are written
            time.sleep(2)
            
            # The workflow creates files in main outputs, so we need to copy them to temp
            self.copy_to_temp()
            
            # Now check if all required files exist in temp folder
            missing_files = []
            for filename in self.required_files:
                filepath = self.temp_outputs_path / filename
                if not filepath.exists():
                    missing_files.append(filename)
            
            if missing_files:
                print(f"    Warning: Missing files: {missing_files}")
                if attempt < max_attempts:
                    print(f"    Retrying...")
                    time.sleep(5)
                    return self.run_workflow(portfolio_value, attempt + 1)
                return False  # Require ALL files for valid data
            
            print(f"    Success! All risk evaluation files generated")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"    Error: Workflow timed out after 300s")
            return False
        except Exception as e:
            print(f"    Error running workflow: {e}")
            return False
    
    def copy_to_temp(self):
        """Copy required files from main outputs to temp folder"""
        copied = 0
        for filename in self.required_files:
            src = self.main_outputs_path / filename
            dst = self.temp_outputs_path / filename
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
                # Delete from main outputs after copying
                src.unlink()
        
        if copied > 0:
            print(f"    Moved {copied} files to temp folder")
    
    def save_day_data(self, portfolio_size: int, day_num: int, price_info: dict) -> bool:
        """Save the required files for one day"""
        try:
            # Create directory for this day
            day_path = self.game_theory_path / f"portfolio_{portfolio_size}" / f"day_{day_num}"
            day_path.mkdir(parents=True, exist_ok=True)
            
            # Copy the 3 eval files from TEMP folder
            files_copied = 0
            for filename in self.required_files:
                src = self.temp_outputs_path / filename  # From temp folder
                dst = day_path / filename
                
                if src.exists():
                    shutil.copy2(src, dst)
                    files_copied += 1
                else:
                    print(f"    Warning: {filename} not found in temp!")
            
            # Add date and price info
            date_info = {
                "day": day_num,
                "date": price_info['date'],
                "ticker": self.ticker,
                "portfolio_size": portfolio_size,
                "market_data": {
                    "open": price_info['open'],
                    "close": price_info['close'],
                    "high": price_info['high'],
                    "low": price_info['low'],
                    "volume": price_info['volume'],
                    "daily_return": price_info['daily_return']
                },
                "files_saved": files_copied
            }
            
            with open(day_path / "date_info.json", 'w') as f:
                json.dump(date_info, f, indent=2)
            
            if files_copied > 0:
                print(f"    ✓ Day {day_num} saved ({files_copied}/3 files) - {price_info['date']} "
                      f"(Return: {price_info['daily_return']*100:+.2f}%)")
            
            return files_copied >= 2  # Accept if we have at least 2 out of 3 files
            
        except Exception as e:
            print(f"    Error saving data: {e}")
            return False
    
    def clean_temp_outputs(self):
        """Clean the TEMP outputs folder for next run"""
        try:
            # Delete entire temp folder contents
            if self.temp_outputs_path.exists():
                for file in self.temp_outputs_path.glob("*"):
                    if file.is_file():
                        file.unlink()
            print(f"    Cleaned temp folder")
        except Exception as e:
            print(f"    Warning: Could not clean temp outputs: {e}")
    
    def cleanup_main_outputs(self):
        """Clean any leftover files from main outputs (just in case)"""
        try:
            # Only remove our specific files if they exist
            for filename in self.required_files:
                filepath = self.main_outputs_path / filename
                if filepath.exists():
                    filepath.unlink()
                    print(f"    Cleaned {filename} from main outputs")
        except Exception as e:
            print(f"    Warning: Could not clean main outputs: {e}")
    
    def collect_all_data(self):
        """Main collection process"""
        self.stats['start_time'] = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"GAME THEORY DATA COLLECTION")
        print(f"{'='*70}")
        print(f"Project Root: {self.project_root}")
        print(f"Ticker: {self.ticker}")
        print(f"Days: {self.days}")
        print(f"Portfolios: {[f'${p:,.0f}' for p in self.portfolio_sizes]}")
        print(f"Output: {self.game_theory_path}")
        print(f"Temp Folder: {self.temp_outputs_path}")
        print(f"Total Runs: {self.days * len(self.portfolio_sizes)}")
        print(f"{'='*70}\n")
        
        # Show price data
        print("Historical Prices:")
        print("-" * 60)
        for i, price in enumerate(self.prices, 1):
            print(f"  Day {i}: {price['date']} | "
                  f"Open: ${price['open']:6.2f} | Close: ${price['close']:6.2f} | "
                  f"Return: {price['daily_return']*100:+5.2f}%")
        print("-" * 60)
        
        # Initial cleanup of any leftover files
        self.cleanup_main_outputs()
        
        # Collect data for each portfolio size
        for portfolio_idx, portfolio_size in enumerate(self.portfolio_sizes, 1):
            print(f"\n{'='*70}")
            print(f"PORTFOLIO {portfolio_idx}/{len(self.portfolio_sizes)}: ${portfolio_size:,.0f}")
            print(f"{'='*70}")
            
            portfolio_success = 0
            portfolio_failed = 0
            
            for day_num in range(1, self.days + 1):
                print(f"\nDay {day_num}/{self.days} - {self.prices[day_num-1]['date']}:")
                
                # Run workflow
                success = self.run_workflow(portfolio_size)
                
                if success:
                    # Save the files from temp folder
                    price_info = self.prices[day_num - 1]
                    saved = self.save_day_data(portfolio_size, day_num, price_info)
                    if saved:
                        self.stats['successful'] += 1
                        portfolio_success += 1
                    else:
                        self.stats['failed'] += 1
                        portfolio_failed += 1
                else:
                    print(f"  ✗ Failed to generate data")
                    self.stats['failed'] += 1
                    portfolio_failed += 1
                
                # Clean up temp folder for next run
                self.clean_temp_outputs()
                
                # Also cleanup main outputs just in case
                self.cleanup_main_outputs()
                
                # Progress indicator
                total_progress = ((portfolio_idx - 1) * self.days + day_num) / (len(self.portfolio_sizes) * self.days)
                print(f"  Overall Progress: {total_progress*100:.1f}%")
                
                # Small delay between runs
                if day_num < self.days:
                    time.sleep(3)
            
            # Portfolio summary
            print(f"\nPortfolio ${portfolio_size:,.0f} Complete: "
                  f"{portfolio_success}/{self.days} successful")
        
        # Final cleanup - remove temp folder entirely
        try:
            if self.temp_outputs_path.exists():
                shutil.rmtree(self.temp_outputs_path)
                print(f"\nRemoved temp folder: {self.temp_outputs_path}")
        except:
            pass
        
        # Create summary file
        self.stats['end_time'] = datetime.now()
        self.create_summary()
        
        # Final summary
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print(f"\n{'='*70}")
        print("DATA COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Time Elapsed: {elapsed/60:.1f} minutes")
        print(f"Successful: {self.stats['successful']}/{self.days * len(self.portfolio_sizes)}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Data Location: {self.game_theory_path}")
        print(f"Main outputs folder: Untouched ✓")
        print(f"{'='*70}\n")
    
    def create_summary(self):
        """Create a summary file for the experiment"""
        summary = {
            "ticker": self.ticker,
            "collection_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "days": self.days,
            "portfolio_sizes": self.portfolio_sizes,
            "collection_stats": {
                "successful": self.stats['successful'],
                "failed": self.stats['failed'],
                "duration_seconds": (self.stats['end_time'] - self.stats['start_time']).total_seconds() if self.stats['end_time'] else 0
            },
            "price_summary": {
                "start_date": self.prices[0]['date'],
                "end_date": self.prices[-1]['date'],
                "start_price": self.prices[0]['open'],
                "end_price": self.prices[-1]['close'],
                "total_return": round(
                    (self.prices[-1]['close'] - self.prices[0]['open']) / self.prices[0]['open'],
                    4
                )
            },
            "daily_returns": [p['daily_return'] for p in self.prices],
            "volatility": round(
                sum(abs(p['daily_return']) for p in self.prices) / len(self.prices),
                4
            ),
            "data_structure": {
                "portfolios": len(self.portfolio_sizes),
                "days_per_portfolio": self.days,
                "total_data_points": len(self.portfolio_sizes) * self.days,
                "files_per_day": len(self.required_files) + 1  # +1 for date_info.json
            }
        }
        
        summary_path = self.game_theory_path / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nExperiment Summary Saved:")
        print(f"  Period: {summary['price_summary']['start_date']} to {summary['price_summary']['end_date']}")
        print(f"  Market Return: {summary['price_summary']['total_return']*100:.2f}%")
        print(f"  Daily Volatility: {summary['volatility']*100:.2f}%")
        success_rate = self.stats['successful']/(self.stats['successful']+self.stats['failed'])*100 if (self.stats['successful']+self.stats['failed']) > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect risk evaluation data for game theory experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from project root:
  python agents/orchestrators/collect_game_theory_data.py NVDA --days 5
  python agents/orchestrators/collect_game_theory_data.py AAPL --days 30
  
  # Run from orchestrators folder:
  python collect_game_theory_data.py TSLA --days 10
  
Note: Uses temp folder inside game_theory directory to avoid
      interfering with main outputs folder.
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker (e.g., NVDA, AAPL)")
    parser.add_argument("--days", type=int, default=5, 
                       help="Number of days to collect (default: 5)")
    
    args = parser.parse_args()
    
    # Validate days
    if args.days < 1:
        print("Error: Days must be at least 1")
        sys.exit(1)
    if args.days > 252:  # Trading days in a year
        print("Warning: More than 252 days may include very old data")
    
    print(f"Starting data collection from: {os.getcwd()}")
    
    # Force UTF-8 for Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        os.system('chcp 65001 > nul')
    
    collector = GameTheoryDataCollector(
        ticker=args.ticker,
        days=args.days
    )
    
    try:
        collector.collect_all_data()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()