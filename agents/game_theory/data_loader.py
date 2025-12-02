"""
data_loader.py - Load Collected Workflow Data for Game Theory Tournament

UPDATED: Now extracts rich signal data:
- risk_reward_ratio
- expected_value  
- bull_prob_pct / bear_prob_pct
- upside_pct / downside_pct
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from .market_context import MarketContext
from .regime_detector import RegimeDetector


class DataLoader:
    """Load collected workflow data from outputs folder."""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root:
            self.project_root = Path(project_root)
        else:
            self.project_root = self._find_project_root()
        
        self.data_path = self.project_root / "outputs" / "game_theory"
        self.regime_detector = RegimeDetector()
        
        if not self.data_path.exists():
            print(f"Warning: Data path does not exist: {self.data_path}")
    
    def _find_project_root(self) -> Path:
        current = Path.cwd()
        for _ in range(6):
            if (current / "outputs" / "game_theory").exists():
                return current
            if current.parent == current:
                break
            current = current.parent
        return Path.cwd()
    
    def get_available_tickers(self) -> List[str]:
        if not self.data_path.exists():
            return []
        
        tickers = []
        for item in self.data_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                portfolio_dir = item / "portfolio_100000"
                if portfolio_dir.exists():
                    samples = list(portfolio_dir.glob("sample_*"))
                    if samples:
                        tickers.append(item.name)
        return sorted(tickers)
    
    def get_sample_count(self, ticker: str, portfolio: int = 100000) -> int:
        ticker_path = self.data_path / ticker / f"portfolio_{portfolio}"
        if not ticker_path.exists():
            return 0
        return len(list(ticker_path.glob("sample_*")))
    
    def _extract_position(self, eval_data: dict) -> float:
        """Extract position as decimal (0.0-1.0)."""
        if 'position_pct' in eval_data:
            pos = float(eval_data.get('position_pct', 0) or 0)
            return pos / 100.0 if pos > 1.0 else pos
        if 'position_size' in eval_data:
            pos = float(eval_data.get('position_size', 0) or 0)
            return pos / 100.0 if pos > 1.0 else pos
        return 0.0
    
    def _extract_metrics(self, eval_data: dict) -> Dict[str, float]:
        """
        Extract rich metrics from evaluation data.
        Looks in both top-level and calculated_metrics.
        """
        metrics = eval_data.get('calculated_metrics', {})
        
        return {
            'risk_reward_ratio': float(
                eval_data.get('risk_reward_ratio') or 
                metrics.get('risk_reward_ratio') or 0.0
            ),
            'expected_value': float(
                eval_data.get('expected_value') or 
                metrics.get('expected_value') or 0.0
            ),
            'bull_prob_pct': float(
                metrics.get('bull_prob_pct') or 33.3
            ),
            'bear_prob_pct': float(
                metrics.get('bear_prob_pct') or 33.3
            ),
            'upside_pct': float(
                metrics.get('upside_pct') or 0.0
            ),
            'downside_pct': float(
                metrics.get('downside_pct') or 0.0
            ),
        }
    
    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    return json.load(f)
            except:
                return None
    
    def _load_sample(
        self, 
        sample_dir: Path, 
        ticker: str, 
        portfolio: int,
        prior_returns: List[float]
    ) -> Optional[MarketContext]:
        """Load a single sample with rich signal data."""
        
        date_info_path = sample_dir / "date_info.json"
        aggressive_path = sample_dir / "aggressive_eval.json"
        neutral_path = sample_dir / "neutral_eval.json"
        conservative_path = sample_dir / "conservative_eval.json"
        
        # Check files exist
        for path in [date_info_path, aggressive_path, neutral_path, conservative_path]:
            if not path.exists():
                return None
        
        # Load JSON files
        date_info = self._load_json(date_info_path)
        aggressive = self._load_json(aggressive_path)
        neutral = self._load_json(neutral_path)
        conservative = self._load_json(conservative_path)
        
        if not all([date_info, aggressive, neutral, conservative]):
            return None
        
        # Extract market data
        market_data = date_info.get('market_data', {})
        daily_return = market_data.get('daily_return', 0.0)
        
        # Detect regime
        all_returns = prior_returns + [daily_return]
        regime = self.regime_detector.detect(all_returns)
        
        # Calculate volatility
        recent_returns = all_returns[-10:] if len(all_returns) >= 10 else all_returns
        volatility = float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0
        
        # Extract sample number
        try:
            sample_num = int(sample_dir.name.split("_")[1])
        except:
            sample_num = date_info.get('sample_number', 0)
        
        # Extract positions
        aggressive_position = self._extract_position(aggressive)
        neutral_position = self._extract_position(neutral)
        conservative_position = self._extract_position(conservative)
        
        # Extract rich metrics (use aggressive eval as primary source)
        metrics = self._extract_metrics(aggressive)
        
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
            aggressive_position=aggressive_position,
            aggressive_confidence=aggressive.get('confidence', 'LOW'),
            aggressive_reasoning=aggressive.get('reasoning', ''),
            
            # Neutral evaluation
            neutral_stance=neutral.get('stance', 'HOLD'),
            neutral_position=neutral_position,
            neutral_confidence=neutral.get('confidence', 'LOW'),
            neutral_reasoning=neutral.get('reasoning', ''),
            
            # Conservative evaluation
            conservative_stance=conservative.get('stance', 'HOLD'),
            conservative_position=conservative_position,
            conservative_confidence=conservative.get('confidence', 'LOW'),
            conservative_reasoning=conservative.get('reasoning', ''),
            
            # Rich signal data
            risk_reward_ratio=metrics['risk_reward_ratio'],
            expected_value=metrics['expected_value'],
            bull_prob_pct=metrics['bull_prob_pct'],
            bear_prob_pct=metrics['bear_prob_pct'],
            upside_pct=metrics['upside_pct'],
            downside_pct=metrics['downside_pct'],
            
            # Derived
            regime=regime,
            volatility=volatility
        )
    
    def load_ticker_data(
        self, 
        ticker: str, 
        portfolio: int = 100000,
        max_samples: Optional[int] = None
    ) -> List[MarketContext]:
        """Load all samples for a ticker."""
        ticker_path = self.data_path / ticker / f"portfolio_{portfolio}"
        
        if not ticker_path.exists():
            print(f"No data found for {ticker}")
            return []
        
        sample_dirs = sorted(
            [d for d in ticker_path.iterdir() if d.is_dir() and d.name.startswith("sample_")],
            key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0
        )
        
        if max_samples:
            sample_dirs = sample_dirs[:max_samples]
        
        if not sample_dirs:
            return []
        
        print(f"Loading {len(sample_dirs)} samples for {ticker}...")
        
        contexts = []
        prior_returns = []
        
        for sample_dir in sample_dirs:
            ctx = self._load_sample(sample_dir, ticker, portfolio, prior_returns)
            if ctx:
                contexts.append(ctx)
                prior_returns.append(ctx.daily_return)
                if len(prior_returns) > 20:
                    prior_returns = prior_returns[-20:]
        
        print(f"  Loaded {len(contexts)}/{len(sample_dirs)} samples")
        
        # Debug: Show signal distribution
        if contexts:
            bullish = sum(1 for c in contexts if c.signal_consensus == "bullish")
            bearish = sum(1 for c in contexts if c.signal_consensus == "bearish")
            mixed = sum(1 for c in contexts if c.signal_consensus == "mixed")
            neutral = len(contexts) - bullish - bearish - mixed
            print(f"  Signals: {bullish} bullish, {bearish} bearish, {mixed} mixed, {neutral} neutral")
            
            avg_rr = np.mean([c.risk_reward_ratio for c in contexts])
            avg_ev = np.mean([c.expected_value for c in contexts])
            print(f"  Avg R/R: {avg_rr:.2f}, Avg EV: {avg_ev:+.2f}%")
        
        return contexts
    
    def load_all_tickers(
        self, 
        portfolio: int = 100000,
        max_samples_per_ticker: Optional[int] = None
    ) -> Dict[str, List[MarketContext]]:
        tickers = self.get_available_tickers()
        if not tickers:
            return {}
        
        all_data = {}
        for ticker in tickers:
            contexts = self.load_ticker_data(ticker, portfolio, max_samples_per_ticker)
            if contexts:
                all_data[ticker] = contexts
        return all_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        tickers = self.get_available_tickers()
        summary = {
            "data_path": str(self.data_path),
            "total_tickers": len(tickers),
            "tickers": {},
        }
        total = 0
        for ticker in tickers:
            count = self.get_sample_count(ticker)
            summary["tickers"][ticker] = count
            total += count
        summary["total_samples"] = total
        return summary
    
    def print_data_summary(self):
        print("\n" + "=" * 60)
        print("GAME THEORY DATA SUMMARY")
        print("=" * 60)
        
        tickers = self.get_available_tickers()
        if not tickers:
            print("No data found!")
            return
        
        print(f"Found {len(tickers)} tickers:")
        total = 0
        for ticker in tickers:
            count = self.get_sample_count(ticker)
            total += count
            print(f"  {ticker}: {count} samples")
        print(f"\nTotal: {total} samples")
        print("=" * 60)


# Convenience function for __init__.py
def load_ticker(ticker: str, portfolio: int = 100000) -> List[MarketContext]:
    loader = DataLoader()
    return loader.load_ticker_data(ticker, portfolio)


if __name__ == "__main__":
    loader = DataLoader()
    loader.print_data_summary()
    
    tickers = loader.get_available_tickers()
    if tickers:
        contexts = loader.load_ticker_data(tickers[0])
        if contexts:
            ctx = contexts[0]
            print(f"\nSample: {ctx}")
            print(f"  Signal: {ctx.signal_consensus}")
            print(f"  R/R: {ctx.risk_reward_ratio:.2f}")
            print(f"  EV: {ctx.expected_value:+.2f}%")