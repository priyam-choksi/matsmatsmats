"""
parallel_collector_sampled.py - With sampling support and better day labeling
Supports sampling every Nth day over longer periods for better regime coverage
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import tempfile
import uuid

# CRITICAL: Set up multiprocessing for Windows
if __name__ == '__main__' and sys.platform == 'win32':
    mp.set_start_method('spawn', force=True)

# WORKER FUNCTION - MUST BE AT MODULE LEVEL FOR WINDOWS
def process_single_workflow(task_data):
    """
    Process a single workflow task with ISOLATED workspace
    """
    ticker, portfolio_size, sample_num, actual_day, price_info, project_root = task_data
    
    # Create unique temporary directory for this workflow
    temp_id = f"{ticker}_{portfolio_size}_sample{sample_num}_{uuid.uuid4().hex[:8]}"
    temp_dir = Path(tempfile.gettempdir()) / "trading_workflows" / temp_id
    temp_outputs = temp_dir / "outputs"
    temp_outputs.mkdir(parents=True, exist_ok=True)
    
    try:
        # Setup final destination paths
        outputs_path = Path(project_root) / "outputs"
        
        # Create organized structure with both sample number and actual day
        date_str = price_info['date']
        sample_str = f"sample{sample_num:03d}_day{actual_day:03d}"
        
        ticker_path = outputs_path / "workflows" / ticker
        date_folder = f"{date_str}_{sample_str}"
        portfolio_folder = f"portfolio_{portfolio_size}"
        
        final_path = ticker_path / date_folder / portfolio_folder
        final_path.mkdir(parents=True, exist_ok=True)
        
        # Find and run master orchestrator with CUSTOM OUTPUT DIR
        master_script = Path(project_root) / "agents" / "orchestrators" / "master_orchestrator.py"
        
        if not master_script.exists():
            return {
                'success': False,
                'ticker': ticker,
                'portfolio': portfolio_size,
                'sample_num': sample_num,
                'actual_day': actual_day,
                'date': price_info['date'],
                'error': 'Master script not found',
                'path': None
            }
        
        # Set environment to use our temp directory
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['TEMP_OUTPUTS_DIR'] = str(temp_outputs)
        
        # Modified command to use temp outputs
        cmd = [
            sys.executable,
            "-c",  # Run as Python code to override output path
            f"""
import sys
import os
sys.path.insert(0, r'{project_root}')
os.chdir(r'{project_root}')

# Override the outputs directory
import agents.orchestrators.master_orchestrator as mo

# Monkey-patch the MasterOrchestrator to use our temp directory
original_init = mo.MasterOrchestrator.__init__

def new_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Override the output path
    self.outputs_path = Path(r'{temp_outputs}')
    self.outputs_path.mkdir(exist_ok=True)

mo.MasterOrchestrator.__init__ = new_init

# Now run the orchestrator
from pathlib import Path
orchestrator = mo.MasterOrchestrator(
    ticker='{ticker}',
    portfolio_value={portfolio_size},
    research_mode='shallow',
    research_rounds=1
)
orchestrator.run_complete_workflow()
"""
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root),
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        # Now collect files from the TEMP directory
        workflow_files = [
            "discussion_points.json",
            "bear_thesis.json", 
            "bull_thesis.json",
            "research_synthesis.json",
            "aggressive_eval.json",
            "neutral_eval.json",
            "conservative_eval.json",
            "risk_decision.json",
            "execution_log.json"
        ]
        
        files_saved = 0
        risk_files_copied = 0
        
        # Move files from temp to final destination
        for filename in workflow_files:
            src = temp_outputs / filename
            if src.exists():
                dst = final_path / filename
                shutil.move(str(src), str(dst))
                files_saved += 1
                
                # Count risk files
                if filename in ["aggressive_eval.json", "neutral_eval.json", "conservative_eval.json"]:
                    risk_files_copied += 1
        
        # Create summary with both sample and actual day info
        summary = {
            "ticker": ticker,
            "date": price_info['date'],
            "sample_number": sample_num,
            "actual_day_number": actual_day,
            "portfolio_size": portfolio_size,
            "market_data": price_info,
            "timestamp": datetime.now().isoformat(),
            "files_saved": files_saved,
            "path": str(final_path.relative_to(outputs_path))
        }
        
        # Extract decision if available
        risk_file = final_path / "risk_decision.json"
        if risk_file.exists():
            try:
                with open(risk_file, 'r') as f:
                    risk_data = json.load(f)
                    summary['decision'] = {
                        'verdict': risk_data.get('verdict', 'UNKNOWN'),
                        'position': risk_data.get('final_position_dollars', 0),
                        'confidence': risk_data.get('confidence', 'LOW')
                    }
            except:
                pass
        
        # Save summary
        with open(final_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also copy to game_theory folder for compatibility
        # Use sample_num for game_theory to maintain consistency
        game_theory_path = outputs_path / "game_theory" / ticker / f"portfolio_{portfolio_size}" / f"sample_{sample_num}"
        game_theory_path.mkdir(parents=True, exist_ok=True)
        
        for risk_file in ["aggressive_eval.json", "neutral_eval.json", "conservative_eval.json"]:
            src = final_path / risk_file
            if src.exists():
                dst = game_theory_path / risk_file
                shutil.copy2(src, dst)
        
        # Save date info with both sample and actual day
        with open(game_theory_path / "date_info.json", 'w') as f:
            json.dump({
                "sample_number": sample_num,
                "actual_day": actual_day,
                "date": price_info['date'],
                "ticker": ticker,
                "portfolio_size": portfolio_size,
                "market_data": price_info,
                "workflow_location": str(final_path.relative_to(outputs_path))
            }, f, indent=2)
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return {
            'success': risk_files_copied >= 2,
            'ticker': ticker,
            'portfolio': portfolio_size,
            'sample_num': sample_num,
            'actual_day': actual_day,
            'date': price_info['date'],
            'files_saved': files_saved,
            'risk_files': risk_files_copied,
            'path': str(final_path.relative_to(outputs_path)),
            'error': None if risk_files_copied >= 2 else f'Only {risk_files_copied} risk files'
        }
        
    except subprocess.TimeoutExpired:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        return {
            'success': False,
            'ticker': ticker,
            'portfolio': portfolio_size,
            'sample_num': sample_num,
            'actual_day': actual_day,
            'date': price_info.get('date', 'unknown'),
            'error': 'Timeout after 5 minutes',
            'path': None
        }
    except Exception as e:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        return {
            'success': False,
            'ticker': ticker,
            'portfolio': portfolio_size,
            'sample_num': sample_num,
            'actual_day': actual_day,
            'date': price_info.get('date', 'unknown'),
            'error': str(e)[:200],
            'path': None
        }


class ParallelCollector:
    def __init__(self, tickers, samples=90, sample_rate=1, max_workers=None):
        self.tickers = tickers
        self.samples = samples  # Number of samples to collect
        self.sample_rate = sample_rate  # 1 = every day, 3 = every 3rd day, etc.
        self.portfolio_sizes = [100000]
        
        # Calculate total days needed
        self.total_days_needed = (self.samples - 1) * self.sample_rate + 1
        
        # Set worker count
        if max_workers is None:
            self.max_workers = min(4, mp.cpu_count() // 2)
        else:
            self.max_workers = max_workers
        
        # Find project root
        self.project_root = self.find_project_root()
        
        # Setup directories
        self.outputs_path = self.project_root / "outputs"
        self.workflows_path = self.outputs_path / "workflows"
        self.game_theory_path = self.outputs_path / "game_theory"
        
        # Create directories
        self.workflows_path.mkdir(parents=True, exist_ok=True)
        self.game_theory_path.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for workflows
        temp_base = Path(tempfile.gettempdir()) / "trading_workflows"
        temp_base.mkdir(exist_ok=True)
        
        # Tracking
        self.start_time = datetime.now()
        self.completed = 0
        self.failed = 0
        self.results_log = []
    
    def find_project_root(self):
        """Find project root directory"""
        current = Path.cwd()
        if current.name == 'orchestrators':
            return current.parent.parent
        elif current.name == 'agents':
            return current.parent
        elif (current / 'agents').exists():
            return current
        else:
            temp = current
            while temp.parent != temp:
                if (temp / 'agents' / 'orchestrators').exists():
                    return temp
                temp = temp.parent
        return current
    
    def get_price_data(self, ticker):
        """Get historical prices for ticker with sampling"""
        try:
            print(f"Fetching {self.total_days_needed} days of prices for {ticker} (sampling every {self.sample_rate} days)...")
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            # Get extra days to ensure we have enough
            start_date = end_date - timedelta(days=self.total_days_needed * 2 + 30)
            
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                return None
            
            # Get enough days for sampling
            all_prices = hist.tail(self.total_days_needed + 10)
            
            # Sample the prices
            price_data = []
            sampled_prices = []
            
            # First, get all prices with day numbers
            all_price_list = []
            for idx, (date, row) in enumerate(all_prices.iterrows(), 1):
                all_price_list.append({
                    'actual_day': idx,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(row['Open'], 2),
                    'close': round(row['Close'], 2),
                    'high': round(row['High'], 2),
                    'low': round(row['Low'], 2),
                    'volume': int(row['Volume']),
                    'daily_return': round((row['Close'] - row['Open']) / row['Open'], 4)
                })
            
            # Now sample them
            for sample_idx in range(self.samples):
                actual_idx = sample_idx * self.sample_rate
                if actual_idx < len(all_price_list):
                    price_info = all_price_list[actual_idx].copy()
                    price_info['sample_num'] = sample_idx + 1
                    sampled_prices.append(price_info)
            
            print(f"  Got {len(sampled_prices)} samples for {ticker}")
            if sampled_prices:
                print(f"  Date range: {sampled_prices[0]['date']} to {sampled_prices[-1]['date']}")
                print(f"  Covering {sampled_prices[-1]['actual_day']} trading days")
            
            return sampled_prices
            
        except Exception as e:
            print(f"  Error getting prices for {ticker}: {e}")
            return None
    
    def check_exists(self, ticker, portfolio, sample_num, date):
        """Check if data already exists with file validation"""
        # Define ALL required files for a complete workflow
        required_files = [
            "discussion_points.json",
            "bear_thesis.json",
            "bull_thesis.json",
            "research_synthesis.json",
            "aggressive_eval.json",
            "neutral_eval.json",
            "conservative_eval.json",
            "summary.json"
            # Note: risk_decision.json often fails, so not required
            # execution_log.json is optional
        ]
        
        # Check new structure first
        # Find the actual folder (handles wildcards)
        workflow_pattern = f"{date}_sample{sample_num:03d}*"
        ticker_path = self.workflows_path / ticker
        
        if ticker_path.exists():
            for date_folder in ticker_path.iterdir():
                if date_folder.name.startswith(f"{date}_sample{sample_num:03d}"):
                    portfolio_path = date_folder / f"portfolio_{portfolio}"
                    if portfolio_path.exists():
                        # Validate files exist and are not empty
                        valid = True
                        for req_file in required_files:
                            file_path = portfolio_path / req_file
                            if not file_path.exists() or file_path.stat().st_size < 10:
                                valid = False
                                break
                        
                        if valid:
                            # Check if we have a valid decision in summary
                            try:
                                with open(portfolio_path / "summary.json", 'r') as f:
                                    summary = json.load(f)
                                    # Check ALL required files are saved (8 minimum)
                                    if summary.get('files_saved', 0) >= 8:
                                        return True
                            except:
                                pass
        
        # Also check game_theory folder as backup
        game_path = self.game_theory_path / ticker / f"portfolio_{portfolio}" / f"sample_{sample_num}"
        if game_path.exists():
            # Check if risk files exist and are valid
            risk_files = ["aggressive_eval.json", "neutral_eval.json", "conservative_eval.json"]
            valid_count = 0
            for risk_file in risk_files:
                file_path = game_path / risk_file
                if file_path.exists() and file_path.stat().st_size > 10:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if data.get('ticker') == ticker:  # Validate content
                                valid_count += 1
                    except:
                        pass
            
            if valid_count >= 2:  # At least 2 valid risk files
                return True
        
        return False
    
    def run_parallel_collection(self):
        """Main parallel collection process with resume capability"""
        print(f"\n{'='*80}")
        print(f"PARALLEL DATA COLLECTION (WITH SAMPLING)")
        print(f"{'='*80}")
        print(f"Tickers: {len(self.tickers)}")
        print(f"Samples per ticker: {self.samples}")
        print(f"Sample rate: Every {self.sample_rate} day(s)")
        print(f"Total days covered: ~{self.total_days_needed} trading days")
        print(f"Portfolio sizes: {self.portfolio_sizes}")
        print(f"Workers: {self.max_workers} parallel processes")
        print(f"CPU count: {mp.cpu_count()}")
        print(f"{'='*80}\n")
        
        # Check for previous incomplete runs
        self.check_previous_runs()
        
        # Build all tasks
        all_tasks = []
        skipped = 0
        invalid = 0
        
        for ticker in self.tickers:
            prices = self.get_price_data(ticker)
            
            if not prices or len(prices) < self.samples:
                print(f"  Skipping {ticker} - insufficient data")
                continue
            
            for portfolio_size in self.portfolio_sizes:
                for price_info in prices:
                    sample_num = price_info['sample_num']
                    actual_day = price_info['actual_day']
                    
                    if self.check_exists(ticker, portfolio_size, sample_num, price_info['date']):
                        skipped += 1
                    else:
                        # Check if partially complete (corrupted/incomplete files)
                        if self.is_partially_complete(ticker, portfolio_size, sample_num, price_info['date']):
                            invalid += 1
                            print(f"  Cleaning incomplete: {ticker} sample{sample_num} P${portfolio_size}")
                            self.clean_incomplete_workflow(ticker, portfolio_size, sample_num, price_info['date'])
                        
                        task = (ticker, portfolio_size, sample_num, actual_day, price_info, str(self.project_root))
                        all_tasks.append(task)
        
        total_tasks = len(all_tasks)
        print(f"Total tasks: {total_tasks} (skipped {skipped} valid, cleaned {invalid} incomplete)")
        
        if not all_tasks:
            print("No tasks to run! All workflows appear complete.")
            return
        
        # Save checkpoint file for recovery
        self.save_checkpoint(all_tasks)
        
        # Adjust time estimate for sampling
        workflows_per_minute = 12 if self.sample_rate == 1 else 10
        print(f"Estimated time: {total_tasks / self.max_workers / workflows_per_minute:.1f} hours")
        print(f"\nStarting parallel processing with isolated workspaces...\n")
        print(f"(You can safely Ctrl+C and resume later - progress is saved)\n")
        
        # Process with concurrent.futures
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_workflow, task): task 
                for task in all_tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    result = future.result(timeout=360)
                    
                    if result['success']:
                        self.completed += 1
                        print(f"✓ {result['ticker']} {result['date']} "
                              f"(sample {result['sample_num']}/day {result['actual_day']}) "
                              f"P${result['portfolio']:,}: {result['files_saved']} files")
                    else:
                        self.failed += 1
                        print(f"✗ {result['ticker']} {result['date']} "
                              f"(sample {result['sample_num']}/day {result['actual_day']}) "
                              f"P${result['portfolio']:,}: {result['error']}")
                    
                    self.results_log.append(result)
                    
                    # Progress update every 10 tasks
                    total_done = self.completed + self.failed
                    if total_done % 10 == 0:
                        self.print_progress(total_done, total_tasks)
                    
                    # Save progress checkpoint every 50 completions
                    if self.completed % 50 == 0:
                        self.save_progress_checkpoint()
                        
                except Exception as e:
                    self.failed += 1
                    task = future_to_task[future]
                    print(f"✗ Task failed: {task[0]} Sample{task[2]} P${task[1]} - {str(e)[:100]}")
        
        # Final summary
        self.print_summary(total_tasks)
        self.save_results_log()
        self.cleanup_checkpoint()
    
    def check_previous_runs(self):
        """Check for previous incomplete runs and offer to resume"""
        checkpoint_file = self.outputs_path / "collection_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"\n⚠️  Found previous run from {checkpoint['timestamp']}")
                print(f"   Completed: {checkpoint.get('completed', 0)} workflows")
                print(f"   Failed: {checkpoint.get('failed', 0)} workflows")
                response = input("   Resume from checkpoint? (y/n): ")
                if response.lower() == 'y':
                    self.completed = checkpoint.get('completed', 0)
                    self.failed = checkpoint.get('failed', 0)
                    self.results_log = checkpoint.get('results_log', [])
                    print("   Resuming from checkpoint...")
            except Exception as e:
                print(f"   Could not load checkpoint: {e}")
    
    def is_partially_complete(self, ticker, portfolio, sample_num, date):
        """Check if workflow is partially complete (has some files but not all)"""
        workflow_pattern = f"{date}_sample{sample_num:03d}*"
        ticker_path = self.workflows_path / ticker
        
        if ticker_path.exists():
            for date_folder in ticker_path.iterdir():
                if date_folder.name.startswith(f"{date}_sample{sample_num:03d}"):
                    portfolio_path = date_folder / f"portfolio_{portfolio}"
                    if portfolio_path.exists():
                        # Count valid JSON files
                        valid_files = 0
                        for json_file in portfolio_path.glob("*.json"):
                            if json_file.stat().st_size > 10:  # Not empty
                                try:
                                    with open(json_file, 'r') as f:
                                        json.load(f)  # Verify it's valid JSON
                                        valid_files += 1
                                except:
                                    pass
                        
                        # If has some files but less than 8 required, it's incomplete
                        if 0 < valid_files < 8:
                            return True
        return False
    
    def clean_incomplete_workflow(self, ticker, portfolio, sample_num, date):
        """Clean up incomplete workflow files"""
        # Clean workflow folder
        workflow_pattern = f"{date}_sample{sample_num:03d}*"
        ticker_path = self.workflows_path / ticker
        
        if ticker_path.exists():
            for date_folder in ticker_path.iterdir():
                if date_folder.name.startswith(f"{date}_sample{sample_num:03d}"):
                    portfolio_path = date_folder / f"portfolio_{portfolio}"
                    if portfolio_path.exists():
                        shutil.rmtree(portfolio_path)
        
        # Clean game_theory folder
        game_path = self.game_theory_path / ticker / f"portfolio_{portfolio}" / f"sample_{sample_num}"
        if game_path.exists():
            shutil.rmtree(game_path)
    
    def save_checkpoint(self, tasks):
        """Save checkpoint for recovery"""
        checkpoint_file = self.outputs_path / "collection_checkpoint.json"
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'tickers': self.tickers,
            'samples': self.samples,
            'sample_rate': self.sample_rate,
            'total_tasks': len(tasks),
            'completed': self.completed,
            'failed': self.failed,
            'results_log': self.results_log
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def save_progress_checkpoint(self):
        """Save progress during execution"""
        checkpoint_file = self.outputs_path / "collection_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                checkpoint['completed'] = self.completed
                checkpoint['failed'] = self.failed
                checkpoint['results_log'] = self.results_log
                checkpoint['last_update'] = datetime.now().isoformat()
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
            except:
                pass  # Don't fail on checkpoint save
    
    def cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion"""
        checkpoint_file = self.outputs_path / "collection_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("Checkpoint file cleaned up.")
    
    def print_progress(self, done, total):
        """Print progress update"""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        
        print(f"\n--- PROGRESS: {done}/{total} ({done*100/total:.1f}%) ---")
        print(f"    Elapsed: {elapsed:.1f} min | Rate: {rate:.1f}/min | Remaining: {remaining:.1f} min")
        print(f"    Success: {self.completed} | Failed: {self.failed}\n")
    
    def print_summary(self, total_tasks):
        """Print final summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        
        print(f"\n{'='*80}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed:.1f} minutes ({elapsed/60:.1f} hours)")
        print(f"Tasks completed: {self.completed + self.failed}/{total_tasks}")
        print(f"Successful: {self.completed}")
        print(f"Failed: {self.failed}")
        if self.completed > 0:
            print(f"Average time per task: {elapsed*60/self.completed:.1f} seconds")
        print(f"\nSampling configuration:")
        print(f"  Sample rate: Every {self.sample_rate} day(s)")
        print(f"  Samples collected: {self.samples} per ticker")
        print(f"  Days covered: ~{self.total_days_needed} trading days")
        print(f"\nData locations:")
        print(f"  Organized workflows: {self.workflows_path}")
        print(f"  Game theory data: {self.game_theory_path}")
        print(f"{'='*80}\n")
    
    def save_results_log(self):
        """Save detailed results log"""
        log_file = self.outputs_path / f"collection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'tickers': self.tickers,
            'samples': self.samples,
            'sample_rate': self.sample_rate,
            'total_days_covered': self.total_days_needed,
            'portfolio_sizes': self.portfolio_sizes,
            'workers': self.max_workers,
            'completed': self.completed,
            'failed': self.failed,
            'results': self.results_log
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Results log saved to: {log_file}")


# Your diverse 20 stocks (removed 10 for better coverage)
SELECTED_TICKERS = [
    # Tech (5)
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META',
    # Finance (3)
    'JPM', 'GS', 'V',
    # Healthcare (3)
    'LLY', 'JNJ', 'UNH',
    # Consumer (3)
    'AMZN', 'TSLA', 'WMT',
    # Energy (2)
    'XOM', 'CVX',
    # ETFs (2)
    'SPY', 'QQQ',
    # Defensive (2)
    'PG', 'KO'
]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parallel data collection with sampling support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 90 consecutive days
  python parallel_collector_sampled.py --samples 90 --sample-rate 1
  
  # Every 3rd day for 90 samples (covers ~270 days)
  python parallel_collector_sampled.py --samples 90 --sample-rate 3
  
  # Every 2nd day for 60 samples (covers ~120 days)
  python parallel_collector_sampled.py --samples 60 --sample-rate 2
  
  # Test mode
  python parallel_collector_sampled.py --test
        """
    )
    
    parser.add_argument('--tickers', nargs='+', default=None, help='Tickers to process')
    parser.add_argument('--samples', type=int, default=90, help='Number of samples to collect')
    parser.add_argument('--sample-rate', type=int, default=1, help='Sample every Nth day (1=daily, 3=every 3rd day)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--test', action='store_true', help='Test mode - 2 tickers, 5 samples')
    
    args = parser.parse_args()
    
    # Set environment variables for UTF-8
    if sys.platform == 'win32':
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
    
    # Determine parameters
    if args.test:
        tickers = SELECTED_TICKERS[:2]
        samples = 5
        sample_rate = 1
        print("TEST MODE: 2 tickers, 5 samples, daily")
    else:
        tickers = args.tickers if args.tickers else SELECTED_TICKERS
        samples = args.samples
        sample_rate = args.sample_rate
    
    # Calculate estimates
    total_tasks = len(tickers) * samples * 1  # 3 portfolio sizes
    total_days = (samples - 1) * sample_rate + 1
    workers = args.workers or min(8, mp.cpu_count() // 2)
    
    print(f"\n{'='*60}")
    print(f"Collection Plan:")
    print(f"  Tickers: {len(tickers)} stocks")
    print(f"  Samples: {samples} per ticker")
    print(f"  Sample rate: Every {sample_rate} day(s)")
    print(f"  Days covered: ~{total_days} trading days")
    print(f"  Total tasks: {total_tasks} workflows")
    print(f"  Workers: {workers} parallel processes")
    print(f"  Estimated time: {total_tasks/workers/12:.1f} hours")
    
    if sample_rate > 1:
        print(f"\nNOTE: Sampling every {sample_rate} days gives better regime")
        print(f"      coverage for game theory analysis!")
    
    print(f"{'='*60}")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    # Run collection
    collector = ParallelCollector(
        tickers=tickers, 
        samples=samples,
        sample_rate=sample_rate,
        max_workers=args.workers
    )
    collector.run_parallel_collection()


if __name__ == '__main__':
    # CRITICAL: Windows multiprocessing setup
    if sys.platform == 'win32':
        mp.freeze_support()
    
    main()