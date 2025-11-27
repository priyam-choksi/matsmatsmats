"""
visualization_engine.py - Charts and Animated GIFs

Location: agents/game_theory/visualization_engine.py

This module creates visualizations for tournament results:
- Static charts: equity curves, strategy comparison, regime heatmap
- Animated GIFs: equity race, score evolution

These visualizations help communicate your research findings
and make the results more engaging for presentations.

Usage:
    from game_theory.visualization_engine import VisualizationEngine
    
    viz = VisualizationEngine(output_dir="outputs/gt_analysis/viz")
    
    # Static charts
    viz.create_equity_curves(strategies, ticker="AAPL")
    viz.create_strategy_comparison(metrics_dict)
    viz.create_regime_heatmap(metrics_dict)
    
    # Animated GIFs
    viz.create_equity_race_gif(strategies, ticker="AAPL")
    viz.create_score_evolution_gif(strategies, ticker="AAPL")

Requirements:
    pip install matplotlib seaborn pillow
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

# Use non-interactive backend for servers/scripts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

if TYPE_CHECKING:
    from .base_strategy import TradingStrategy
    from .metrics_calculator import StrategyMetrics
    from .monte_carlo_engine import MonteCarloResult


class VisualizationEngine:
    """
    Create static charts and animated GIFs for tournament results.
    
    All outputs are saved to the specified output directory.
    Uses a consistent color scheme for strategies across all charts.
    
    Attributes:
        output_dir: Directory to save visualizations
        COLORS: Color mapping for each strategy
    """
    
    # Consistent colors for each strategy
    COLORS = {
        'Actual Market': '#95a5a6',   # Gray (control)
        'Buy-and-Hold': '#2c3e50',    # Dark blue
        'Cooperator': '#27ae60',      # Green
        'Defector': '#e74c3c',        # Red
        'Tit-for-Tat': '#3498db',     # Blue
        'Conservative': '#9b59b6',    # Purple
        'Aggressive': '#f39c12',      # Orange
    }
    
    def __init__(self, output_dir: Path):
        """
        Initialize VisualizationEngine.
        
        Args:
            output_dir: Directory to save all visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    
    def _get_color(self, name: str) -> str:
        """Get color for a strategy, with fallback."""
        return self.COLORS.get(name, '#888888')
    
    # ========== ANIMATED GIFS ==========
    
    def create_equity_race_gif(
        self, 
        strategies: List['TradingStrategy'],
        ticker: str = "",
        fps: int = 10
    ) -> Path:
        """
        Create animated GIF showing equity curves building over time.
        Like watching a race unfold - great for presentations!
        
        Args:
            strategies: List of strategy instances with trade history
            ticker: Ticker symbol for title
            fps: Frames per second (default 10)
            
        Returns:
            Path to saved GIF
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Build equity data as percentages
        equity_data = {}
        max_len = 0
        
        for s in strategies:
            equity_pct = [(e - 1.0) * 100.0 for e in s.equity_curve]
            equity_data[s.name] = equity_pct
            max_len = max(max_len, len(equity_pct))
        
        # Calculate y-axis limits with padding
        all_vals = [v for vals in equity_data.values() for v in vals]
        y_min = min(all_vals) - 2
        y_max = max(all_vals) + 2
        
        # Initialize lines
        lines = {}
        for name in equity_data:
            color = self._get_color(name)
            line, = ax.plot([], [], label=name, color=color, linewidth=2)
            lines[name] = line
        
        # Setup axes
        ax.set_xlim(0, max_len - 1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Sample', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        
        title = f'{ticker} Strategy Performance Race' if ticker else 'Strategy Performance Race'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Leader text box
        leader_text = ax.text(
            0.98, 0.98, '', 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        def animate(frame):
            current_vals = {}
            
            for name, data in equity_data.items():
                if frame < len(data):
                    lines[name].set_data(range(frame + 1), data[:frame + 1])
                    current_vals[name] = data[frame]
            
            if current_vals:
                leader = max(current_vals, key=current_vals.get)
                leader_text.set_text(
                    f'Sample {frame}\n'
                    f'Leader: {leader}\n'
                    f'{current_vals[leader]:+.1f}%'
                )
            
            return list(lines.values()) + [leader_text]
        
        anim = FuncAnimation(
            fig, animate, 
            frames=max_len, 
            interval=1000//fps, 
            blit=True
        )
        
        # Save
        filename = f'{ticker}_equity_race.gif' if ticker else 'equity_race.gif'
        output_path = self.output_dir / filename
        anim.save(str(output_path), writer=PillowWriter(fps=fps), dpi=100)
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path
    
    def create_score_evolution_gif(
        self, 
        strategies: List['TradingStrategy'],
        ticker: str = "",
        fps: int = 8
    ) -> Path:
        """
        Animated bar chart showing scores evolving each round.
        
        Args:
            strategies: List of strategy instances
            ticker: Ticker symbol for title
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [s.name for s in strategies]
        score_data = {s.name: s.score_history for s in strategies}
        max_len = max(len(h) for h in score_data.values())
        colors = [self._get_color(n) for n in names]
        
        # Initialize bars
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, [0] * len(names), color=colors)
        
        # Setup axes
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_ylim(-12, 12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Score', fontsize=11)
        
        title = f'{ticker} Score Evolution' if ticker else 'Score Evolution'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Reference lines
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.4, label='High score zone')
        ax.axhline(y=-5, color='red', linestyle='--', alpha=0.4, label='Low score zone')
        
        # Round counter
        round_text = ax.text(
            0.02, 0.98, '', 
            transform=ax.transAxes,
            fontsize=11, 
            verticalalignment='top', 
            fontweight='bold'
        )
        
        def animate(frame):
            for i, name in enumerate(names):
                history = score_data[name]
                score = history[frame] if frame < len(history) else history[-1]
                bars[i].set_height(score)
                
                # Color by score
                if score >= 5:
                    bars[i].set_color('#27ae60')  # Green
                elif score <= -5:
                    bars[i].set_color('#e74c3c')  # Red
                else:
                    bars[i].set_color(self._get_color(name))
            
            round_text.set_text(f'Round {frame}')
            return list(bars) + [round_text]
        
        anim = FuncAnimation(
            fig, animate, 
            frames=max_len, 
            interval=1000//fps, 
            blit=True
        )
        
        # Save
        filename = f'{ticker}_score_evolution.gif' if ticker else 'score_evolution.gif'
        output_path = self.output_dir / filename
        anim.save(str(output_path), writer=PillowWriter(fps=fps), dpi=100)
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path
    
    # ========== STATIC CHARTS ==========
    
    def create_equity_curves(
        self, 
        strategies: List['TradingStrategy'],
        ticker: str = ""
    ) -> Path:
        """
        Create static equity curves chart.
        
        Args:
            strategies: List of strategy instances
            ticker: Ticker symbol for title
            
        Returns:
            Path to saved PNG
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for s in strategies:
            equity_pct = [(e - 1.0) * 100.0 for e in s.equity_curve]
            
            # Dashed line for baselines
            style = '--' if s.name in ['Buy-and-Hold', 'Actual Market'] else '-'
            
            ax.plot(
                equity_pct, 
                label=f"{s.name} ({equity_pct[-1]:+.1f}%)",
                color=self._get_color(s.name), 
                linestyle=style, 
                linewidth=2
            )
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Sample', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        
        title = f'{ticker} Strategy Performance' if ticker else 'Strategy Performance'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'{ticker}_equity_curves.png' if ticker else 'equity_curves.png'
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path
    
    def create_strategy_comparison(
        self, 
        metrics_dict: Dict[str, 'StrategyMetrics']
    ) -> Path:
        """
        Create 4-panel bar chart comparing strategies.
        
        Panels: Total Return, Sharpe Ratio, Win Rate, Max Drawdown
        
        Args:
            metrics_dict: Dict mapping strategy name to StrategyMetrics
            
        Returns:
            Path to saved PNG
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = list(metrics_dict.keys())
        colors = [self._get_color(n) for n in names]
        x = np.arange(len(names))
        
        # Panel 1: Total Return
        ax = axes[0, 0]
        values = [metrics_dict[n].total_return for n in names]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Total Return', fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height(),
                f'{val:+.1f}%', 
                ha='center', 
                va='bottom', 
                fontsize=8
            )
        
        # Panel 2: Sharpe Ratio
        ax = axes[0, 1]
        values = [metrics_dict[n].sharpe_ratio for n in names]
        # Handle infinity
        values = [v if not np.isinf(v) else 0 for v in values]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio', fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.4)
        
        # Panel 3: Win Rate
        ax = axes[1, 0]
        values = [metrics_dict[n].win_rate for n in names]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate', fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 4: Max Drawdown
        ax = axes[1, 1]
        values = [metrics_dict[n].max_drawdown for n in names]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Max Drawdown (lower is better)', fontweight='bold')
        
        plt.suptitle('Strategy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'strategy_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path
    
    def create_regime_heatmap(
        self, 
        metrics_dict: Dict[str, 'StrategyMetrics']
    ) -> Path:
        """
        Create heatmap of returns by strategy and regime.
        
        Great for answering: "Which strategy wins in which regime?"
        
        Args:
            metrics_dict: Dict mapping strategy name to StrategyMetrics
            
        Returns:
            Path to saved PNG
        """
        names = list(metrics_dict.keys())
        regimes = ['bull', 'bear', 'sideways']
        
        # Build data matrix
        data = np.zeros((len(names), len(regimes)))
        
        for i, name in enumerate(names):
            m = metrics_dict[name]
            data[i, 0] = m.bull_return
            data[i, 1] = m.bear_return
            data[i, 2] = m.sideways_return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            data, 
            annot=True, 
            fmt='.1f', 
            cmap='RdYlGn', 
            center=0,
            xticklabels=['Bull', 'Bear', 'Sideways'],
            yticklabels=names, 
            ax=ax, 
            cbar_kws={'label': 'Return (%)'}
        )
        
        ax.set_title('Performance by Market Regime', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'regime_heatmap.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path
    
    def create_monte_carlo_chart(
        self, 
        mc_results: Dict[str, 'MonteCarloResult']
    ) -> Path:
        """
        Create Monte Carlo return distribution summary.
        
        Shows mean return with 95% confidence interval error bars.
        
        Args:
            mc_results: Dict mapping strategy name to MonteCarloResult
            
        Returns:
            Path to saved PNG
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = list(mc_results.keys())
        x = np.arange(len(names))
        width = 0.6
        
        # Extract data
        means = [mc_results[n].mean_return for n in names]
        ci_lows = [mc_results[n].ci_lower_95 for n in names]
        ci_highs = [mc_results[n].ci_upper_95 for n in names]
        
        colors = [self._get_color(n) for n in names]
        
        # Create bars
        bars = ax.bar(x, means, width, color=colors, alpha=0.8)
        
        # Error bars for 95% CI
        errors = [
            [m - l for m, l in zip(means, ci_lows)],
            [h - m for m, h in zip(means, ci_highs)]
        ]
        ax.errorbar(x, means, yerr=errors, fmt='none', color='black', capsize=5)
        
        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Return (%) with 95% CI')
        ax.set_title('Monte Carlo Return Distributions', fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add P(>0) labels
        for i, name in enumerate(names):
            prob = mc_results[name].prob_positive * 100
            y_pos = means[i] + (ci_highs[i] - means[i]) + 1
            ax.text(i, y_pos, f'P(>0)={prob:.0f}%', ha='center', fontsize=8)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'monte_carlo_distributions.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path
    
    def create_score_history_chart(
        self, 
        strategies: List['TradingStrategy'],
        ticker: str = ""
    ) -> Path:
        """
        Create line chart of score history over time.
        
        Args:
            strategies: List of strategy instances
            ticker: Ticker symbol for title
            
        Returns:
            Path to saved PNG
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for s in strategies:
            ax.plot(
                s.score_history,
                label=f"{s.name} (final: {s.score:+.0f})",
                color=self._get_color(s.name),
                linewidth=2
            )
        
        # Reference zones
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=-5, color='red', linestyle='--', alpha=0.3)
        ax.axhspan(5, 10, alpha=0.1, color='green')
        ax.axhspan(-10, -5, alpha=0.1, color='red')
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(-11, 11)
        
        title = f'{ticker} Score History' if ticker else 'Score History'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'{ticker}_score_history.png' if ticker else 'score_history.png'
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
        return output_path


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing VisualizationEngine...")
    print("=" * 60)
    print("Note: This test creates mock data for visualization.")
    print("Run with actual strategy data for real charts.")
    
    # Create mock strategies for testing
    class MockStrategy:
        def __init__(self, name, returns):
            self.name = name
            self.daily_returns = returns
            self.equity_curve = [1.0]
            for r in returns:
                self.equity_curve.append(self.equity_curve[-1] * (1 + r))
            
            # Mock score history
            self.score = 0
            self.score_history = [0]
            for r in returns:
                if r > 0:
                    self.score = min(10, self.score + 1)
                else:
                    self.score = max(-10, self.score - 1)
                self.score_history.append(self.score)
    
    # Create test data
    np.random.seed(42)
    n_samples = 50
    
    strategies = [
        MockStrategy("Actual Market", list(np.random.normal(0.001, 0.015, n_samples))),
        MockStrategy("Cooperator", list(np.random.normal(0.002, 0.012, n_samples))),
        MockStrategy("Defector", list(np.random.normal(0.0005, 0.020, n_samples))),
        MockStrategy("Buy-and-Hold", list(np.random.normal(0.0015, 0.010, n_samples))),
        MockStrategy("Tit-for-Tat", list(np.random.normal(0.0018, 0.014, n_samples))),
    ]
    
    # Create output directory
    output_dir = Path("test_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    viz = VisualizationEngine(output_dir)
    
    # Test static charts
    print("\nCreating static charts...")
    viz.create_equity_curves(strategies, ticker="TEST")
    viz.create_score_history_chart(strategies, ticker="TEST")
    
    # Test animated GIFs (these take longer)
    print("\nCreating animated GIFs (this may take a moment)...")
    viz.create_equity_race_gif(strategies, ticker="TEST", fps=8)
    viz.create_score_evolution_gif(strategies, ticker="TEST", fps=6)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("VisualizationEngine test complete!")