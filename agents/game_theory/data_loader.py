"""
data_loader.py - Load Collected Workflow Data for Game Theory Tournament

This module reads your collected data from the folder structure:
    outputs/game_theory/{TICKER}/portfolio_100000/sample_{N}/
    ├── date_info.json          # date, sample_number, market_data
    ├── aggressive_eval.json    # stance, position_size, confidence, reasoning
    ├── neutral_eval.json       # stance, position_size, confidence, reasoning
    └── conservative_eval.json  # stance, position_size, confidence, reasoning

And converts it into MarketContext objects that strategies can use.

Usage:
    from game_theory.data_loader import DataLoader
    
    loader = DataLoader()
    
    # Get available tickers
    tickers = loader.get_available_tickers()
    print(f"Found tickers: {tickers}")
    
    # Load data for a specific ticker
    contexts = loader.load_ticker_data("AAPL")
    print(f"Loaded {len(contexts)} samples for AAPL")
    
    # Access individual context
    ctx = contexts[0]
    print(f"Date: {ctx.date}, Return: {ctx.daily_return:.2%}")
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from .market_context import MarketContext
from .regime_detector import RegimeDetector


class DataLoader:
    """
    Load collected workflow data from your outputs folder structure.
    
    Automatically finds the project root by searching for the 
    outputs/game_theory folder, making it work from any subdirectory.
    
    Attributes:
        project_root: Path to project root directory
        data_path: Path to outputs/game_theory folder
        regime_detector: RegimeDetector instance for classifying market regimes
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            project_root: Optional explicit path to project root.
                         If not provided, will search for it automatically.
        """
        if project_root:
            self.project_root = Path(project_root)
        else:
            self.project_root = self._find_project_root()
        
        self.data_path = self.project_root / "outputs" / "game_theory"
        self.regime_detector = RegimeDetector()
        
        if not self.data_path.exists():
            print(f"Warning: Data path does not exist: {self.data_path}")
            print("Make sure you have collected data using the parallel collector.")
    
    def _find_project_root(self) -> Path:
        """
        Find project root by searching for outputs/game_theory folder.
        
        Searches current directory and up to 5 parent directories.
        Also handles being run from subdirectories like agents/orchestrators.
        
        Returns:
            Path to project root
            
        Raises:
            FileNotFoundError: If project root cannot be found
        """
        current = Path.cwd()
        
        # Check current and parent directories
        for _ in range(6):
            # Direct check
            if (current / "outputs" / "game_theory").exists():
                return current
            
            # Check if we're in a known subfolder structure
            if current.name in ['orchestrators', 'game_theory']:
                # Go up to agents/ then to project root
                parent = current.parent
                if parent.name == 'agents':
                    parent = parent.parent
                if (parent / "outputs" / "game_theory").exists():
                    return parent
            
            if current.name == 'agents':
                parent = current.parent
                if (parent / "outputs" / "game_theory").exists():
                    return parent
            
            # Move up one directory
            current = current.parent
        
        # If not found, check common relative paths from cwd
        cwd = Path.cwd()
        common_paths = [
            cwd / "outputs" / "game_theory",
            cwd.parent / "outputs" / "game_theory",
            cwd.parent.parent / "outputs" / "game_theory",
        ]
        
        for path in common_paths:
            if path.exists():
                return path.parent.parent
        
        raise FileNotFoundError(
            "Could not find outputs/game_theory folder.\n"
            "Make sure you're running from the project directory or a subdirectory.\n"
            f"Searched from: {Path.cwd()}"
        )
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of tickers that have collected data.
        
        Only returns tickers that have at least one sample in the
        portfolio_100000 folder (your primary portfolio size).
        
        Returns:
            Sorted list of ticker symbols (e.g., ["AAPL", "GOOGL", "MSFT"])
        """
        if not self.data_path.exists():
            print(f"Data path does not exist: {self.data_path}")
            return []
        
        tickers = []
        
        for ticker_dir in self.data_path.iterdir():
            # Skip hidden files/folders
            if not ticker_dir.is_dir() or ticker_dir.name.startswith('.'):
                continue
            
            # Check for portfolio_100000 folder with samples
            portfolio_dir = ticker_dir / "portfolio_100000"
            if portfolio_dir.exists():
                # Check if it has any sample folders
                sample_dirs = list(portfolio_dir.glob("sample_*"))
                if sample_dirs:
                    tickers.append(ticker_dir.name)
        
        return sorted(tickers)
    
    def get_sample_count(self, ticker: str, portfolio: int = 100000) -> int:
        """
        Get number of samples available for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            portfolio: Portfolio size (default 100000)
            
        Returns:
            Number of sample folders found
        """
        ticker_path = self.data_path / ticker / f"portfolio_{portfolio}"
        
        if not ticker_path.exists():
            return 0
        
        sample_dirs = [
            d for d in ticker_path.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        ]
        
        return len(sample_dirs)
    
    def load_ticker_data(
        self, 
        ticker: str, 
        portfolio: int = 100000,
        max_samples: Optional[int] = None
    ) -> List[MarketContext]:
        """
        Load all samples for a ticker and return as MarketContext objects.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            portfolio: Portfolio size to load (default 100000)
            max_samples: Optional limit on number of samples to load
            
        Returns:
            List of MarketContext objects, sorted by sample number
        """
        ticker_path = self.data_path / ticker / f"portfolio_{portfolio}"
        
        if not ticker_path.exists():
            print(f"Warning: No data for {ticker} at {ticker_path}")
            return []
        
        contexts: List[MarketContext] = []
        market_returns: List[float] = []  # For regime detection
        
        # Get sorted sample directories
        sample_dirs = self._get_sorted_sample_dirs(ticker_path)
        
        if max_samples:
            sample_dirs = sample_dirs[:max_samples]
        
        # Load each sample
        for sample_dir in sample_dirs:
            try:
                ctx = self._load_single_sample(
                    sample_dir, 
                    ticker, 
                    portfolio, 
                    market_returns
                )
                
                if ctx:
                    contexts.append(ctx)
                    market_returns.append(ctx.daily_return)
                    
            except Exception as e:
                print(f"Warning: Failed to load {sample_dir.name}: {e}")
                continue
        
        print(f"Loaded {len(contexts)} samples for {ticker}")
        return contexts
    
    def _get_sorted_sample_dirs(self, ticker_path: Path) -> List[Path]:
        """
        Get sample directories sorted by sample number.
        
        Args:
            ticker_path: Path to ticker's portfolio folder
            
        Returns:
            List of Path objects sorted by sample number
        """
        sample_dirs = [
            d for d in ticker_path.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        ]
        
        # Sort by sample number (extract number from "sample_N")
        def get_sample_num(path: Path) -> int:
            try:
                return int(path.name.split("_")[1])
            except (IndexError, ValueError):
                return 0
        
        return sorted(sample_dirs, key=get_sample_num)
    
    def _load_single_sample(
        self, 
        sample_dir: Path, 
        ticker: str, 
        portfolio: int,
        prior_returns: List[float]
    ) -> Optional[MarketContext]:
        """
        Load a single sample directory into a MarketContext.
        
        Args:
            sample_dir: Path to sample_N folder
            ticker: Stock symbol
            portfolio: Portfolio size
            prior_returns: List of prior daily returns (for regime detection)
            
        Returns:
            MarketContext object, or None if loading fails
        """
        # Define required files
        date_info_path = sample_dir / "date_info.json"
        aggressive_path = sample_dir / "aggressive_eval.json"
        neutral_path = sample_dir / "neutral_eval.json"
        conservative_path = sample_dir / "conservative_eval.json"
        
        # Check all required files exist
        required_files = [date_info_path, aggressive_path, neutral_path, conservative_path]
        for path in required_files:
            if not path.exists():
                print(f"  Missing: {path.name} in {sample_dir.name}")
                return None
        
        # Load JSON files with UTF-8 encoding
        date_info = self._load_json(date_info_path)
        aggressive = self._load_json(aggressive_path)
        neutral = self._load_json(neutral_path)
        conservative = self._load_json(conservative_path)
        
        if not all([date_info, aggressive, neutral, conservative]):
            return None
        
        # Extract market data
        market_data = date_info.get('market_data', {})
        daily_return = market_data.get('daily_return', 0.0)
        
        # Detect regime based on prior returns + current
        all_returns = prior_returns + [daily_return]
        regime = self.regime_detector.detect(all_returns)
        
        # Calculate volatility from recent returns
        recent_returns = all_returns[-10:] if len(all_returns) >= 10 else all_returns
        volatility = float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0
        
        # Extract sample number
        try:
            sample_num = int(sample_dir.name.split("_")[1])
        except (IndexError, ValueError):
            sample_num = date_info.get('sample_number', 0)
        
        # Create MarketContext
        return MarketContext(
            # Identifiers
            date=date_info.get('date', ''),
            ticker=ticker,
            sample_num=sample_num,
            portfolio_size=float(portfolio),
            
            # Price data
            open_price=market_data.get('open', 0.0),
            close_price=market_data.get('close', 0.0),
            high=market_data.get('high', 0.0),
            low=market_data.get('low', 0.0),
            volume=market_data.get('volume', 0),
            daily_return=daily_return,
            
            # Aggressive evaluation
            aggressive_stance=aggressive.get('stance', 'HOLD'),
            aggressive_position=float(aggressive.get('position_size', 0.0)),
            aggressive_confidence=aggressive.get('confidence', 'LOW'),
            aggressive_reasoning=aggressive.get('reasoning', ''),
            
            # Neutral evaluation
            neutral_stance=neutral.get('stance', 'HOLD'),
            neutral_position=float(neutral.get('position_size', 0.0)),
            neutral_confidence=neutral.get('confidence', 'LOW'),
            neutral_reasoning=neutral.get('reasoning', ''),
            
            # Conservative evaluation
            conservative_stance=conservative.get('stance', 'HOLD'),
            conservative_position=float(conservative.get('position_size', 0.0)),
            conservative_confidence=conservative.get('confidence', 'LOW'),
            conservative_reasoning=conservative.get('reasoning', ''),
            
            # Derived fields
            regime=regime,
            volatility=volatility
        )
    
    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a JSON file with proper encoding handling.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Parsed JSON as dictionary, or None if loading fails
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"  JSON decode error in {path.name}: {e}")
            return None
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Encoding error in {path.name}: {e}")
                return None
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
            return None
    
    def load_all_tickers(
        self, 
        portfolio: int = 100000,
        max_samples_per_ticker: Optional[int] = None
    ) -> Dict[str, List[MarketContext]]:
        """
        Load data for all available tickers.
        
        Args:
            portfolio: Portfolio size to load
            max_samples_per_ticker: Optional limit per ticker
            
        Returns:
            Dictionary mapping ticker -> list of MarketContext
        """
        tickers = self.get_available_tickers()
        all_data: Dict[str, List[MarketContext]] = {}
        
        print(f"Loading data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            contexts = self.load_ticker_data(
                ticker, 
                portfolio, 
                max_samples_per_ticker
            )
            if contexts:
                all_data[ticker] = contexts
        
        total_samples = sum(len(c) for c in all_data.values())
        print(f"Loaded {total_samples} total samples across {len(all_data)} tickers")
        
        return all_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about available data.
        
        Returns:
            Dictionary with data summary
        """
        tickers = self.get_available_tickers()
        
        summary = {
            "data_path": str(self.data_path),
            "total_tickers": len(tickers),
            "tickers": {},
        }
        
        total_samples = 0
        for ticker in tickers:
            count = self.get_sample_count(ticker)
            summary["tickers"][ticker] = count
            total_samples += count
        
        summary["total_samples"] = total_samples
        
        return summary
    
    def print_data_summary(self):
        """Print a formatted summary of available data."""
        summary = self.get_data_summary()
        
        print("\n" + "=" * 50)
        print("GAME THEORY DATA SUMMARY")
        print("=" * 50)
        print(f"Data path: {summary['data_path']}")
        print(f"Total tickers: {summary['total_tickers']}")
        print(f"Total samples: {summary['total_samples']}")
        print("-" * 50)
        
        if summary['tickers']:
            print(f"{'Ticker':<10} {'Samples':>10}")
            print("-" * 20)
            for ticker, count in sorted(summary['tickers'].items()):
                print(f"{ticker:<10} {count:>10}")
        else:
            print("No data found!")
        
        print("=" * 50 + "\n")


# === Convenience function for quick loading ===

def load_ticker(ticker: str, portfolio: int = 100000) -> List[MarketContext]:
    """
    Convenience function to quickly load data for a ticker.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        portfolio: Portfolio size (default 100000)
        
    Returns:
        List of MarketContext objects
        
    Example:
        from game_theory.data_loader import load_ticker
        
        contexts = load_ticker("AAPL")
        for ctx in contexts[:5]:
            print(ctx)
    """
    loader = DataLoader()
    return loader.load_ticker_data(ticker, portfolio)


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing DataLoader...")
    
    try:
        loader = DataLoader()
        loader.print_data_summary()
        
        tickers = loader.get_available_tickers()
        if tickers:
            # Load first ticker as test
            test_ticker = tickers[0]
            print(f"\nLoading sample data for {test_ticker}...")
            
            contexts = loader.load_ticker_data(test_ticker, max_samples=5)
            
            if contexts:
                print(f"\nFirst 5 samples:")
                for ctx in contexts[:5]:
                    print(f"  {ctx}")
            else:
                print("No contexts loaded!")
        else:
            print("No tickers found!")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")