"""
game_theory_analysis.py - Comprehensive Game Theory Tournament Analysis

This is the MAIN analysis engine that:
1. Loads all collected data (20 tickers × 90 samples)
2. Runs 5 intelligent trading strategies
3. Calculates proper financial metrics (Sharpe, drawdown, etc.)
4. Performs regime-conditional analysis
5. Generates publication-ready visualizations

Usage:
    python game_theory_analysis.py                    # All tickers
    python game_theory_analysis.py --ticker AAPL     # Single ticker
    python game_theory_analysis.py --monte-carlo     # Include MC simulation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketContext:
    """Market information for a trading decision"""
    date: str
    ticker: str
    sample_num: int
    portfolio_size: float
    
    # Price data
    open_price: float
    close_price: float
    high: float
    low: float
    volume: int
    daily_return: float
    
    # Agent evaluations
    aggressive_stance: str
    aggressive_position: float
    aggressive_confidence: str
    
    neutral_stance: str
    neutral_position: float
    neutral_confidence: str
    
    conservative_stance: str
    conservative_position: float
    conservative_confidence: str
    
    # Derived
    regime: str = ""
    volatility: float = 0.0
    consensus: float = 0.0


@dataclass
class TradeResult:
    """Result of a single trade"""
    date: str
    position_pct: float
    position_dollars: float
    market_return: float
    trade_return: float
    cumulative_return: float
    reasoning: str


@dataclass 
class StrategyMetrics:
    """Complete metrics for a strategy"""
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_position: float
    volatility: float
    calmar_ratio: float
    trades: int
    
    # Regime performance
    bull_return: float = 0.0
    bear_return: float = 0.0
    sideways_return: float = 0.0


# ============================================================================
# STRATEGY BASE CLASS
# ============================================================================

class TradingStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
        # Track state
        self.trades: List[TradeResult] = []
        self.equity_curve = [1.0]
        self.positions: List[float] = []
        self.daily_returns: List[float] = []
        
        # For regime tracking
        self.regime_returns = {'bull': [], 'bear': [], 'sideways': []}
    
    @abstractmethod
    def decide_position(self, context: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        """
        Decide position size (0-100%) and reasoning.
        Returns: (position_percent, reasoning)
        """
        pass
    
    def execute_trade(self, context: MarketContext) -> TradeResult:
        """Execute a trade and update state"""
        position_pct, reasoning = self.decide_position(context, self.trades)
        position_pct = max(0, min(100, position_pct))
        
        position_dollars = position_pct / 100 * context.portfolio_size
        trade_return = context.daily_return * (position_pct / 100)
        
        new_equity = self.equity_curve[-1] * (1 + trade_return)
        self.equity_curve.append(new_equity)
        self.daily_returns.append(trade_return)
        self.positions.append(position_pct)
        
        # Track by regime
        if context.regime:
            self.regime_returns[context.regime].append(trade_return)
        
        result = TradeResult(
            date=context.date,
            position_pct=position_pct,
            position_dollars=position_dollars,
            market_return=context.daily_return,
            trade_return=trade_return,
            cumulative_return=(new_equity - 1) * 100,
            reasoning=reasoning
        )
        self.trades.append(result)
        return result
    
    def calculate_metrics(self, annualization_factor: float = 252/3) -> StrategyMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.daily_returns:
            return None
        
        returns = np.array(self.daily_returns)
        equity = np.array(self.equity_curve)
        
        # Basic returns
        total_return = (equity[-1] - 1) * 100
        annualized_return = ((equity[-1]) ** (annualization_factor / len(returns)) - 1) * 100
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(annualization_factor) * 100
        
        # Sharpe (assuming 0 risk-free rate for simplicity)
        if volatility > 0:
            sharpe = (annualized_return) / volatility
        else:
            sharpe = 0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(annualization_factor) * 100
            sortino = annualized_return / downside_std if downside_std > 0 else 0
        else:
            sortino = float('inf') if annualized_return > 0 else 0
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # Calmar Ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win/Loss stats
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) * 100 if len(returns) > 0 else 0
        avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0
        avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
        
        # Profit Factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average position
        avg_position = np.mean(self.positions) if self.positions else 0
        
        # Regime returns
        bull_ret = sum(self.regime_returns['bull']) * 100 if self.regime_returns['bull'] else 0
        bear_ret = sum(self.regime_returns['bear']) * 100 if self.regime_returns['bear'] else 0
        side_ret = sum(self.regime_returns['sideways']) * 100 if self.regime_returns['sideways'] else 0
        
        return StrategyMetrics(
            name=self.name,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_position=avg_position,
            volatility=volatility,
            calmar_ratio=calmar,
            trades=len(self.trades),
            bull_return=bull_ret,
            bear_return=bear_ret,
            sideways_return=side_ret
        )
    
    def reset(self):
        """Reset strategy state"""
        self.trades = []
        self.equity_curve = [1.0]
        self.positions = []
        self.daily_returns = []
        self.regime_returns = {'bull': [], 'bear': [], 'sideways': []}


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class CooperatorStrategy(TradingStrategy):
    """
    COOPERATOR - Adaptive Consensus Follower
    
    Philosophy: Trust the collective wisdom of agents, scale with confidence.
    - High consensus → Larger positions
    - Low consensus → Smaller positions  
    - Adapts position size based on recent performance
    """
    
    def __init__(self):
        super().__init__("Cooperator", "Adaptive Consensus Follower")
        self.confidence_multiplier = 1.0
    
    def decide_position(self, ctx: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        # Calculate consensus (how much agents agree)
        positions = [ctx.aggressive_position, ctx.neutral_position, ctx.conservative_position]
        avg_position = np.mean(positions) * 100
        position_std = np.std(positions) * 100
        
        # High agreement = low std
        consensus = max(0, 1 - position_std / 10)
        
        # Base position on average recommendation scaled by consensus
        base_position = avg_position * (0.5 + 0.5 * consensus)
        
        # Adjust based on recent performance
        if len(history) >= 5:
            recent_returns = [t.trade_return for t in history[-5:]]
            if sum(recent_returns) > 0.02:
                self.confidence_multiplier = min(1.3, self.confidence_multiplier + 0.1)
            elif sum(recent_returns) < -0.02:
                self.confidence_multiplier = max(0.7, self.confidence_multiplier - 0.1)
        
        position = base_position * self.confidence_multiplier
        
        # Scale up from conservative agent recommendations
        position = max(position, ctx.neutral_position * 100 * 1.5)
        
        reasoning = f"Consensus: {consensus:.0%}, Avg rec: {avg_position:.0f}%, Conf mult: {self.confidence_multiplier:.1f}"
        return position, reasoning


class DefectorStrategy(TradingStrategy):
    """
    DEFECTOR - Aggressive Contrarian
    
    Philosophy: The crowd is often wrong. Go against weak consensus.
    - Strong consensus → Follow it (crowd might be right)
    - Weak consensus → Take larger contrarian position
    - Always maintains significant exposure
    """
    
    def __init__(self):
        super().__init__("Defector", "Aggressive Contrarian")
    
    def decide_position(self, ctx: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        positions = [ctx.aggressive_position, ctx.neutral_position, ctx.conservative_position]
        avg_position = np.mean(positions) * 100
        position_std = np.std(positions) * 100
        
        # Check if aggressive recommends BUY
        agg_bullish = ctx.aggressive_stance.upper() in ['BUY', 'STRONG BUY']
        
        # Consensus measure
        consensus = max(0, 1 - position_std / 10)
        
        if consensus > 0.7:
            # Strong consensus - follow but be more aggressive
            position = avg_position * 1.5
            reasoning = f"Strong consensus ({consensus:.0%}), amplifying to {position:.0f}%"
        elif consensus < 0.3:
            # Weak consensus - be contrarian, go big
            if agg_bullish:
                position = 60 + (1 - consensus) * 20  # 60-80%
                reasoning = f"Weak consensus ({consensus:.0%}), contrarian bullish"
            else:
                position = 30  # Still maintain exposure
                reasoning = f"Weak consensus ({consensus:.0%}), cautious contrarian"
        else:
            # Medium consensus - aggressive baseline
            position = ctx.aggressive_position * 100 * 2
            reasoning = f"Medium consensus, 2x aggressive recommendation"
        
        # Defector never goes below 25%
        position = max(25, min(85, position))
        
        return position, reasoning


class TitForTatStrategy(TradingStrategy):
    """
    TIT-FOR-TAT - Momentum Follower
    
    Philosophy: Replicate what worked last time.
    - Winning positions → Repeat or increase
    - Losing positions → Reduce or reverse
    - Adapts to market regime
    """
    
    def __init__(self):
        super().__init__("Tit-for-Tat", "Momentum Follower")
        self.last_successful_position = 40
    
    def decide_position(self, ctx: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        if len(history) < 3:
            # Start moderate
            position = 35
            reasoning = "Initial period, starting moderate"
        else:
            # Analyze recent history
            recent = history[-5:]
            
            # Track what positions worked
            winning_trades = [t for t in recent if t.trade_return > 0]
            losing_trades = [t for t in recent if t.trade_return < 0]
            
            if winning_trades:
                avg_winning_pos = np.mean([t.position_pct for t in winning_trades])
                self.last_successful_position = avg_winning_pos
            
            # Recent momentum
            recent_pnl = sum(t.trade_return for t in recent)
            
            if recent_pnl > 0.03:
                # Winning streak - increase position
                position = self.last_successful_position * 1.2
                reasoning = f"Winning streak (+{recent_pnl*100:.1f}%), increasing"
            elif recent_pnl < -0.03:
                # Losing streak - reduce and try different size
                if self.last_successful_position > 40:
                    position = 25
                    reasoning = f"Losing streak ({recent_pnl*100:.1f}%), reducing"
                else:
                    position = 50
                    reasoning = f"Losing streak ({recent_pnl*100:.1f}%), trying larger"
            else:
                # Neutral - continue what worked
                position = self.last_successful_position
                reasoning = f"Neutral period, maintaining {position:.0f}%"
        
        position = max(15, min(70, position))
        return position, reasoning


class ConservativeBaselineStrategy(TradingStrategy):
    """
    CONSERVATIVE BASELINE - Risk-Averse Control
    
    Philosophy: Capital preservation above all.
    - Never exceeds 20% position
    - Only invests on strong consensus
    - Benchmark for risk-averse approach
    """
    
    def __init__(self):
        super().__init__("Conservative", "Risk-Averse Baseline")
    
    def decide_position(self, ctx: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        # Only follow conservative agent, slightly scaled
        base = ctx.conservative_position * 100
        
        # Check for strong bullish consensus
        all_bullish = all(s.upper() in ['BUY', 'STRONG BUY', 'SMALL BUY'] 
                        for s in [ctx.aggressive_stance, ctx.neutral_stance, ctx.conservative_stance])
        
        if all_bullish and ctx.aggressive_confidence == 'HIGH':
            position = min(20, base * 2)
            reasoning = "Strong consensus, max conservative position"
        elif ctx.conservative_stance.upper() in ['AVOID', 'SELL']:
            position = 0
            reasoning = "Conservative says avoid"
        else:
            position = min(15, base * 1.5)
            reasoning = f"Standard conservative: {position:.0f}%"
        
        return position, reasoning


class AggressiveBaselineStrategy(TradingStrategy):
    """
    AGGRESSIVE BASELINE - Maximum Exposure Control
    
    Philosophy: Market goes up over time, stay invested.
    - Minimum 50% position always
    - Scales higher on bullish signals
    - Benchmark for aggressive approach
    """
    
    def __init__(self):
        super().__init__("Aggressive", "Maximum Exposure Baseline")
    
    def decide_position(self, ctx: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        base = ctx.aggressive_position * 100
        
        if ctx.aggressive_confidence == 'HIGH':
            position = max(70, base * 2.5)
            reasoning = "High confidence, maximum aggression"
        elif ctx.aggressive_stance.upper() in ['BUY', 'STRONG BUY']:
            position = max(55, base * 2)
            reasoning = "Bullish signal, staying aggressive"
        else:
            position = 50
            reasoning = "Baseline aggressive minimum"
        
        position = min(90, position)
        return position, reasoning


class BuyAndHoldStrategy(TradingStrategy):
    """
    BUY AND HOLD - Market Benchmark
    
    100% invested at all times. The benchmark to beat.
    """
    
    def __init__(self):
        super().__init__("Buy-and-Hold", "Market Benchmark")
    
    def decide_position(self, ctx: MarketContext, history: List[TradeResult]) -> Tuple[float, str]:
        return 100, "Always 100% invested"


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """Detect market regime from price data"""
    
    @staticmethod
    def detect(returns: List[float], lookback: int = 10) -> str:
        """
        Classify market regime based on recent returns.
        
        Returns: 'bull', 'bear', or 'sideways'
        """
        if len(returns) < lookback:
            return 'sideways'
        
        recent = returns[-lookback:]
        cumulative = sum(recent)
        volatility = np.std(recent) if len(recent) > 1 else 0
        
        # Thresholds
        if cumulative > 0.03:  # >3% in lookback period
            return 'bull'
        elif cumulative < -0.03:  # <-3%
            return 'bear'
        else:
            return 'sideways'


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Load collected workflow data"""
    
    def __init__(self, project_root: Path = None):
        if project_root:
            self.project_root = project_root
        else:
            self.project_root = self._find_project_root()
        
        self.data_path = self.project_root / "outputs" / "game_theory"
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for outputs/game_theory folder"""
        current = Path.cwd()
        
        # First check if we're in orchestrators folder
        if current.name == 'orchestrators':
            project_root = current.parent.parent  # Go up to TradingAgent
            if (project_root / "outputs" / "game_theory").exists():
                return project_root
        
        # Check current and parent directories
        for _ in range(5):
            if (current / "outputs" / "game_theory").exists():
                return current
            current = current.parent
        
        # Fallback: look for TradingAgent folder
        current = Path.cwd()
        while current.parent != current:
            if current.name == "TradingAgent" or (current / "agents").exists():
                return current
            current = current.parent
        
        return Path.cwd()
    
    def get_available_tickers(self) -> List[str]:
        """Get list of tickers with data"""
        if not self.data_path.exists():
            return []
        return [d.name for d in self.data_path.iterdir() if d.is_dir()]
    
    def load_ticker_data(self, ticker: str, portfolio: int = 100000) -> List[MarketContext]:
        """Load all samples for a ticker"""
        ticker_path = self.data_path / ticker / f"portfolio_{portfolio}"
        
        if not ticker_path.exists():
            print(f"No data for {ticker} at {ticker_path}")
            return []
        
        contexts = []
        market_returns = []  # For regime detection
        
        # Get sorted samples
        sample_dirs = sorted([d for d in ticker_path.iterdir() if d.is_dir()])
        
        for sample_dir in sample_dirs:
            try:
                # Load files
                date_info = json.load(open(sample_dir / "date_info.json"))
                agg_eval = json.load(open(sample_dir / "aggressive_eval.json"))
                neu_eval = json.load(open(sample_dir / "neutral_eval.json"))
                con_eval = json.load(open(sample_dir / "conservative_eval.json"))
                
                market_data = date_info.get('market_data', {})
                daily_return = market_data.get('daily_return', 0)
                market_returns.append(daily_return)
                
                # Detect regime
                regime = RegimeDetector.detect(market_returns)
                
                ctx = MarketContext(
                    date=date_info.get('date', ''),
                    ticker=ticker,
                    sample_num=date_info.get('sample_number', 0),
                    portfolio_size=portfolio,
                    open_price=market_data.get('open', 0),
                    close_price=market_data.get('close', 0),
                    high=market_data.get('high', 0),
                    low=market_data.get('low', 0),
                    volume=market_data.get('volume', 0),
                    daily_return=daily_return,
                    aggressive_stance=agg_eval.get('stance', 'HOLD'),
                    aggressive_position=agg_eval.get('position_size', 0.1),
                    aggressive_confidence=agg_eval.get('confidence', 'LOW'),
                    neutral_stance=neu_eval.get('stance', 'HOLD'),
                    neutral_position=neu_eval.get('position_size', 0.05),
                    neutral_confidence=neu_eval.get('confidence', 'LOW'),
                    conservative_stance=con_eval.get('stance', 'HOLD'),
                    conservative_position=con_eval.get('position_size', 0.01),
                    conservative_confidence=con_eval.get('confidence', 'LOW'),
                    regime=regime,
                    volatility=np.std(market_returns[-10:]) if len(market_returns) >= 10 else 0
                )
                contexts.append(ctx)
                
            except Exception as e:
                continue
        
        return contexts


# ============================================================================
# MAIN TOURNAMENT ENGINE  
# ============================================================================

class GameTheoryTournament:
    """Main tournament engine"""
    
    def __init__(self, output_dir: Path = None):
        self.loader = DataLoader()
        
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.loader.project_root / "outputs" / "gt_analysis" / timestamp
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "by_ticker").mkdir(exist_ok=True)
        
        # Initialize strategies
        self.strategies = [
            CooperatorStrategy(),
            DefectorStrategy(),
            TitForTatStrategy(),
            ConservativeBaselineStrategy(),
            AggressiveBaselineStrategy(),
            BuyAndHoldStrategy()
        ]
        
        self.all_results = {}
        
        print(f"Game Theory Tournament initialized")
        print(f"Project root: {self.loader.project_root}")
        print(f"Data path: {self.loader.data_path}")
        print(f"Data path exists: {self.loader.data_path.exists()}")
        print(f"Output: {self.output_dir}")
    
    def run_single_ticker(self, ticker: str, portfolio: int = 100000) -> Dict[str, StrategyMetrics]:
        """Run tournament for a single ticker"""
        
        # Load data
        contexts = self.loader.load_ticker_data(ticker, portfolio)
        if not contexts:
            return {}
        
        print(f"\n{'='*60}")
        print(f"Running tournament: {ticker} ({len(contexts)} samples)")
        print(f"{'='*60}")
        
        # Reset strategies
        for strategy in self.strategies:
            strategy.reset()
        
        # Run each sample
        for ctx in contexts:
            for strategy in self.strategies:
                strategy.execute_trade(ctx)
        
        # Calculate metrics
        results = {}
        for strategy in self.strategies:
            metrics = strategy.calculate_metrics()
            if metrics:
                results[strategy.name] = metrics
                print(f"  {strategy.name:20} | Return: {metrics.total_return:+7.2f}% | "
                      f"Sharpe: {metrics.sharpe_ratio:+5.2f} | MaxDD: {metrics.max_drawdown:5.1f}%")
        
        # Save ticker results
        self._save_ticker_results(ticker, results, contexts)
        
        return results
    
    def run_all_tickers(self, portfolio: int = 100000) -> Dict[str, Dict[str, StrategyMetrics]]:
        """Run tournament for all available tickers"""
        
        tickers = self.loader.get_available_tickers()
        print(f"Found {len(tickers)} tickers: {tickers}")
        
        for ticker in tickers:
            results = self.run_single_ticker(ticker, portfolio)
            if results:
                self.all_results[ticker] = results
        
        # Generate combined analysis
        if self.all_results:
            self._generate_combined_analysis()
        
        return self.all_results
    
    def _save_ticker_results(self, ticker: str, results: Dict[str, StrategyMetrics], 
                            contexts: List[MarketContext]):
        """Save results for a single ticker"""
        ticker_dir = self.output_dir / "by_ticker" / ticker
        ticker_dir.mkdir(exist_ok=True)
        
        # Save metrics JSON
        metrics_dict = {name: vars(m) for name, m in results.items()}
        with open(ticker_dir / "metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        # Generate visualizations
        self._plot_equity_curves(ticker, ticker_dir)
        self._plot_position_sizes(ticker, ticker_dir)
        self._plot_regime_performance(ticker, results, ticker_dir)
    
    def _plot_equity_curves(self, ticker: str, output_dir: Path):
        """Plot equity curves for all strategies"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = {
            'Cooperator': '#2ecc71',
            'Defector': '#e74c3c', 
            'Tit-for-Tat': '#3498db',
            'Conservative': '#9b59b6',
            'Aggressive': '#f39c12',
            'Buy-and-Hold': '#2c3e50'
        }
        
        for strategy in self.strategies:
            equity_pct = [(e - 1) * 100 for e in strategy.equity_curve]
            style = '--' if strategy.name == 'Buy-and-Hold' else '-'
            width = 2.5 if strategy.name == 'Buy-and-Hold' else 1.8
            
            ax.plot(equity_pct, label=f"{strategy.name} ({equity_pct[-1]:+.1f}%)",
                   color=colors.get(strategy.name, '#888'), linestyle=style, linewidth=width)
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title(f'{ticker} - Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'equity_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_position_sizes(self, ticker: str, output_dir: Path):
        """Plot position sizes over time"""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        colors = {
            'Cooperator': '#2ecc71',
            'Defector': '#e74c3c',
            'Tit-for-Tat': '#3498db',
            'Conservative': '#9b59b6',
            'Aggressive': '#f39c12'
        }
        
        for strategy in self.strategies:
            if strategy.name != 'Buy-and-Hold':
                ax.plot(strategy.positions, label=strategy.name,
                       color=colors.get(strategy.name, '#888'), alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Position Size (%)', fontsize=12)
        ax.set_title(f'{ticker} - Position Sizes Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'position_sizes.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_performance(self, ticker: str, results: Dict[str, StrategyMetrics], 
                                 output_dir: Path):
        """Plot performance by market regime"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        strategies = [s for s in results.keys() if s != 'Buy-and-Hold']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
        
        for idx, regime in enumerate(['bull', 'bear', 'sideways']):
            ax = axes[idx]
            returns = [getattr(results[s], f'{regime}_return') for s in strategies]
            
            bars = ax.bar(range(len(strategies)), returns, color=colors[:len(strategies)])
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels([s.split('-')[0][:8] for s in strategies], rotation=45, ha='right')
            ax.set_ylabel('Return (%)')
            ax.set_title(f'{regime.upper()} Market', fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, returns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:+.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'{ticker} - Performance by Market Regime', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'regime_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_combined_analysis(self):
        """Generate analysis across all tickers"""
        print(f"\n{'='*60}")
        print("GENERATING COMBINED ANALYSIS")
        print(f"{'='*60}")
        
        # Aggregate metrics
        strategy_totals = {s.name: [] for s in self.strategies}
        
        for ticker, results in self.all_results.items():
            for name, metrics in results.items():
                strategy_totals[name].append(metrics)
        
        # Calculate averages
        avg_metrics = {}
        for name, metrics_list in strategy_totals.items():
            if metrics_list:
                avg_metrics[name] = {
                    'avg_return': np.mean([m.total_return for m in metrics_list]),
                    'avg_sharpe': np.mean([m.sharpe_ratio for m in metrics_list]),
                    'avg_max_dd': np.mean([m.max_drawdown for m in metrics_list]),
                    'avg_win_rate': np.mean([m.win_rate for m in metrics_list]),
                    'avg_position': np.mean([m.avg_position for m in metrics_list]),
                    'win_count': sum(1 for m in metrics_list if m.total_return > 0),
                    'beat_market': sum(1 for i, m in enumerate(metrics_list) 
                                      if m.total_return > strategy_totals['Buy-and-Hold'][i].total_return),
                    'total_tickers': len(metrics_list)
                }
        
        # Save combined metrics
        with open(self.output_dir / 'combined_metrics.json', 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        # Generate combined visualizations
        self._plot_strategy_comparison(avg_metrics)
        self._plot_returns_heatmap()
        self._plot_sharpe_comparison(avg_metrics)
        self._plot_regime_winners()
        
        # Print summary
        self._print_final_summary(avg_metrics)
    
    def _plot_strategy_comparison(self, avg_metrics: Dict):
        """Bar chart comparing strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        strategies = list(avg_metrics.keys())
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2c3e50']
        
        # Average Return
        ax = axes[0, 0]
        returns = [avg_metrics[s]['avg_return'] for s in strategies]
        bars = ax.bar(strategies, returns, color=colors[:len(strategies)])
        ax.set_ylabel('Average Return (%)')
        ax.set_title('Average Return Across All Tickers', fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        for bar, val in zip(bars, returns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:+.1f}%', ha='center', va='bottom', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Average Sharpe
        ax = axes[0, 1]
        sharpes = [avg_metrics[s]['avg_sharpe'] for s in strategies]
        bars = ax.bar(strategies, sharpes, color=colors[:len(strategies)])
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Average Sharpe Ratio', fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        for bar, val in zip(bars, sharpes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Win Rate
        ax = axes[1, 0]
        win_rates = [avg_metrics[s]['avg_win_rate'] for s in strategies]
        bars = ax.bar(strategies, win_rates, color=colors[:len(strategies)])
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Average Win Rate', fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Beat Market Count
        ax = axes[1, 1]
        beat_market = [avg_metrics[s].get('beat_market', 0) for s in strategies if s != 'Buy-and-Hold']
        strat_names = [s for s in strategies if s != 'Buy-and-Hold']
        bars = ax.bar(strat_names, beat_market, color=colors[:len(strat_names)])
        ax.set_ylabel('# Tickers Beat Market')
        ax.set_title('Times Strategy Beat Buy-and-Hold', fontweight='bold')
        total = avg_metrics[strategies[0]]['total_tickers']
        ax.axhline(y=total/2, color='gray', linestyle='--', alpha=0.5, label='50% line')
        for bar, val in zip(bars, beat_market):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val}/{total}', ha='center', va='bottom', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Strategy Comparison - All Tickers Combined', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'strategy_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_returns_heatmap(self):
        """Heatmap of returns by strategy and ticker"""
        # Build matrix
        tickers = list(self.all_results.keys())
        strategies = [s.name for s in self.strategies]
        
        matrix = np.zeros((len(tickers), len(strategies)))
        for i, ticker in enumerate(tickers):
            for j, strat in enumerate(strategies):
                if strat in self.all_results[ticker]:
                    matrix[i, j] = self.all_results[ticker][strat].total_return
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(tickers) * 0.4)))
        
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=[s[:10] for s in strategies], yticklabels=tickers,
                   ax=ax, cbar_kws={'label': 'Return (%)'})
        
        ax.set_title('Returns by Strategy and Ticker (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Ticker')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'returns_heatmap.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sharpe_comparison(self, avg_metrics: Dict):
        """Scatter plot of return vs risk"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {
            'Cooperator': '#2ecc71',
            'Defector': '#e74c3c',
            'Tit-for-Tat': '#3498db',
            'Conservative': '#9b59b6',
            'Aggressive': '#f39c12',
            'Buy-and-Hold': '#2c3e50'
        }
        
        for name, metrics in avg_metrics.items():
            ax.scatter(metrics['avg_max_dd'], metrics['avg_return'],
                      s=200, c=colors.get(name, '#888'), label=name,
                      edgecolors='white', linewidth=2)
            ax.annotate(name, (metrics['avg_max_dd'], metrics['avg_return']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Average Max Drawdown (%)', fontsize=12)
        ax.set_ylabel('Average Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile by Strategy', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'risk_return.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_winners(self):
        """Which strategy wins in each regime"""
        regime_wins = {'bull': {}, 'bear': {}, 'sideways': {}}
        
        for ticker, results in self.all_results.items():
            for regime in ['bull', 'bear', 'sideways']:
                best_strat = None
                best_return = -float('inf')
                
                for name, metrics in results.items():
                    if name == 'Buy-and-Hold':
                        continue
                    regime_ret = getattr(metrics, f'{regime}_return')
                    if regime_ret > best_return:
                        best_return = regime_ret
                        best_strat = name
                
                if best_strat:
                    regime_wins[regime][best_strat] = regime_wins[regime].get(best_strat, 0) + 1
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
        
        for idx, regime in enumerate(['bull', 'bear', 'sideways']):
            ax = axes[idx]
            if regime_wins[regime]:
                strategies = list(regime_wins[regime].keys())
                wins = list(regime_wins[regime].values())
                ax.pie(wins, labels=strategies, autopct='%1.0f%%',
                      colors=colors[:len(strategies)])
            ax.set_title(f'{regime.upper()} Market\nWinner Distribution', fontweight='bold')
        
        plt.suptitle('Which Strategy Wins in Each Market Regime?', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'regime_winners.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save regime analysis
        with open(self.output_dir / 'regime_analysis.json', 'w') as f:
            json.dump(regime_wins, f, indent=2)
    
    def _print_final_summary(self, avg_metrics: Dict):
        """Print final summary table"""
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        # Sort by average return
        sorted_strategies = sorted(avg_metrics.items(), 
                                  key=lambda x: x[1]['avg_return'], reverse=True)
        
        print(f"{'Strategy':<20} {'Avg Return':>12} {'Avg Sharpe':>12} {'Win Rate':>10} {'Beat Market':>12}")
        print("-" * 70)
        
        for name, m in sorted_strategies:
            beat = f"{m.get('beat_market', 0)}/{m['total_tickers']}"
            print(f"{name:<20} {m['avg_return']:>+11.2f}% {m['avg_sharpe']:>+11.2f} "
                  f"{m['avg_win_rate']:>9.1f}% {beat:>12}")
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Game Theory Tournament Analysis")
    parser.add_argument('--ticker', type=str, default=None, help='Single ticker to analyze')
    parser.add_argument('--portfolio', type=int, default=100000, help='Portfolio size')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    
    args = parser.parse_args()
    
    tournament = GameTheoryTournament()
    
    if args.ticker:
        tournament.run_single_ticker(args.ticker, args.portfolio)
    else:
        tournament.run_all_tickers(args.portfolio)
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()