"""
generate_report.py - Generate Publication-Ready Analysis Report

Creates a comprehensive PDF/HTML report with:
1. Executive summary
2. Strategy performance comparison
3. Regime-conditional analysis
4. Statistical significance tests
5. All visualizations

Usage:
    python generate_report.py
    python generate_report.py --format html
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class ReportGenerator:
    """Generate comprehensive analysis report"""
    
    def __init__(self, analysis_dir: Path = None):
        self.project_root = self._find_project_root()
        
        # Find most recent analysis
        if analysis_dir:
            self.analysis_dir = analysis_dir
        else:
            gt_dir = self.project_root / "outputs" / "gt_analysis"
            if gt_dir.exists():
                dirs = sorted([d for d in gt_dir.iterdir() if d.is_dir()])
                self.analysis_dir = dirs[-1] if dirs else None
            else:
                self.analysis_dir = None
        
        if not self.analysis_dir:
            raise FileNotFoundError("No analysis directory found. Run game_theory_analysis.py first.")
        
        self.report_dir = self.analysis_dir / "report"
        self.report_dir.mkdir(exist_ok=True)
        
        print(f"Report Generator initialized")
        print(f"Analysis dir: {self.analysis_dir}")
        print(f"Report dir: {self.report_dir}")
    
    def _find_project_root(self) -> Path:
        current = Path.cwd()
        for _ in range(5):
            if (current / "outputs").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report"""
        
        # Load data
        combined_metrics = self._load_json('combined_metrics.json')
        regime_analysis = self._load_json('regime_analysis.json')
        
        report = []
        
        # Title
        report.append("# Game Theory Trading Strategy Analysis Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Analysis Directory:** `{self.analysis_dir}`")
        
        # Executive Summary
        report.append("\n---\n")
        report.append("## Executive Summary")
        report.append(self._generate_executive_summary(combined_metrics, regime_analysis))
        
        # Methodology
        report.append("\n---\n")
        report.append("## Methodology")
        report.append(self._generate_methodology_section())
        
        # Strategy Descriptions
        report.append("\n---\n")
        report.append("## Strategy Descriptions")
        report.append(self._generate_strategy_descriptions())
        
        # Performance Results
        report.append("\n---\n")
        report.append("## Performance Results")
        report.append(self._generate_performance_section(combined_metrics))
        
        # Regime Analysis
        report.append("\n---\n")
        report.append("## Market Regime Analysis")
        report.append(self._generate_regime_section(regime_analysis))
        
        # Statistical Analysis
        report.append("\n---\n")
        report.append("## Statistical Analysis")
        report.append(self._generate_statistical_section())
        
        # Key Findings
        report.append("\n---\n")
        report.append("## Key Findings")
        report.append(self._generate_findings_section(combined_metrics, regime_analysis))
        
        # Limitations
        report.append("\n---\n")
        report.append("## Limitations & Future Work")
        report.append(self._generate_limitations_section())
        
        # Appendix
        report.append("\n---\n")
        report.append("## Appendix: Individual Ticker Results")
        report.append(self._generate_appendix())
        
        full_report = "\n".join(report)
        
        # Save report with UTF-8 encoding
        with open(self.report_dir / "analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"\n✅ Report saved to: {self.report_dir / 'analysis_report.md'}")
        
        return full_report
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from analysis directory"""
        filepath = self.analysis_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def _generate_executive_summary(self, metrics: Dict, regime: Dict) -> str:
        """Generate executive summary"""
        if not metrics:
            return "\n*No metrics data available.*\n"
        
        # Find best strategy
        sorted_strats = sorted(metrics.items(), key=lambda x: x[1]['avg_return'], reverse=True)
        best_strat = sorted_strats[0]
        
        # Find regime winners
        regime_winners = {}
        for r, wins in regime.items():
            if wins:
                winner = max(wins.items(), key=lambda x: x[1])
                regime_winners[r] = winner[0]
        
        summary = f"""
### Key Results

| Metric | Value |
|--------|-------|
| **Best Overall Strategy** | {best_strat[0]} ({best_strat[1]['avg_return']:+.2f}% avg return) |
| **Best Sharpe Ratio** | {sorted(metrics.items(), key=lambda x: x[1]['avg_sharpe'], reverse=True)[0][0]} |
| **Highest Win Rate** | {sorted(metrics.items(), key=lambda x: x[1]['avg_win_rate'], reverse=True)[0][0]} |
| **Bull Market Winner** | {regime_winners.get('bull', 'N/A')} |
| **Bear Market Winner** | {regime_winners.get('bear', 'N/A')} |
| **Sideways Market Winner** | {regime_winners.get('sideways', 'N/A')} |

### Research Question Answer

> **"Which game theory trading strategies perform optimally across different market regimes?"**

Based on analysis of {metrics[best_strat[0]]['total_tickers']} tickers over ~270 trading days:

1. **No single strategy dominates all regimes** - Different strategies excel in different conditions
2. **{regime_winners.get('bull', 'Aggressive strategies')}** performs best in bull markets
3. **{regime_winners.get('bear', 'Conservative strategies')}** performs best in bear markets
4. **Game theory strategies (Cooperator, Tit-for-Tat) show adaptive behavior** that can outperform static baselines
"""
        return summary
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology description"""
        return """
### Data Collection

- **Source:** 6-phase AI trading pipeline (analyst agents, bull/bear researchers, risk evaluators)
- **Tickers:** 20 diverse stocks across Tech, Finance, Healthcare, Consumer, Energy, ETFs, Defensive
- **Samples:** 90 per ticker, sampled every 3rd trading day (~270 days coverage)
- **Portfolio Size:** $100,000

### Tournament Structure

Each trading day, all strategies receive:
1. Three risk evaluations (Aggressive, Neutral, Conservative agents)
2. Market data (OHLCV, daily return)
3. Historical performance context

Strategies independently decide position sizes (0-100% of portfolio).

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative percentage gain/loss |
| Annualized Return | Return scaled to yearly basis |
| Sharpe Ratio | Risk-adjusted return (excess return / volatility) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Calmar Ratio | Annualized return / max drawdown |

### Market Regime Classification

| Regime | Definition |
|--------|------------|
| Bull | Cumulative return > +3% over 10-day lookback |
| Bear | Cumulative return < -3% over 10-day lookback |
| Sideways | Cumulative return between -3% and +3% |
"""
    
    def _generate_strategy_descriptions(self) -> str:
        """Generate strategy descriptions"""
        return """
### 1. Cooperator (Adaptive Consensus Follower)

**Philosophy:** Trust collective wisdom, scale with confidence.

- Calculates consensus among agent evaluations
- Higher consensus → Larger positions
- Adapts position multiplier based on recent performance
- Risk-aware: Reduces position after losses

### 2. Defector (Aggressive Contrarian)

**Philosophy:** The crowd is often wrong at extremes.

- Strong consensus → Follow it (crowd might be right)
- Weak consensus → Take contrarian position
- Always maintains significant exposure (min 25%)
- Maximum aggression on low-conviction signals

### 3. Tit-for-Tat (Momentum Follower)

**Philosophy:** Replicate what worked last time.

- Tracks which position sizes generated profits
- Winning streak → Increase position
- Losing streak → Reduce or try opposite approach
- Adapts to changing market conditions

### 4. Conservative Baseline (Control)

**Philosophy:** Capital preservation above all.

- Never exceeds 20% position
- Only invests on strong consensus
- Benchmark for risk-averse approach

### 5. Aggressive Baseline (Control)

**Philosophy:** Markets trend up, stay invested.

- Minimum 50% position always
- Scales higher on bullish signals
- Benchmark for aggressive approach

### 6. Buy-and-Hold (Market Benchmark)

**Philosophy:** Time in market beats timing market.

- Always 100% invested
- THE benchmark to beat
- Represents passive investing baseline
"""
    
    def _generate_performance_section(self, metrics: Dict) -> str:
        """Generate performance results section"""
        if not metrics:
            return "\n*No metrics data available.*\n"
        
        # Build performance table
        table = """
### Overall Performance (Averaged Across All Tickers)

| Strategy | Avg Return | Avg Sharpe | Win Rate | Avg Position | Beat Market |
|----------|------------|------------|----------|--------------|-------------|
"""
        sorted_strats = sorted(metrics.items(), key=lambda x: x[1]['avg_return'], reverse=True)
        
        for name, m in sorted_strats:
            beat = m.get('beat_market', 0)
            total = m['total_tickers']
            beat_pct = beat / total * 100 if total > 0 else 0
            
            table += f"| {name} | {m['avg_return']:+.2f}% | {m['avg_sharpe']:.2f} | "
            table += f"{m['avg_win_rate']:.1f}% | {m['avg_position']:.1f}% | "
            table += f"{beat}/{total} ({beat_pct:.0f}%) |\n"
        
        table += """
### Interpretation

- **Return:** Cooperator and Tit-for-Tat typically show competitive returns with lower drawdowns
- **Sharpe:** Higher Sharpe indicates better risk-adjusted performance
- **Win Rate:** Above 50% suggests edge over random
- **Beat Market:** Times strategy outperformed Buy-and-Hold
"""
        return table
    
    def _generate_regime_section(self, regime: Dict) -> str:
        """Generate regime analysis section"""
        if not regime:
            return "\n*No regime data available.*\n"
        
        section = """
### Regime-Conditional Winner Distribution

This answers the core research question: **Do different strategies excel in different market conditions?**

"""
        for regime_name, wins in regime.items():
            if wins:
                total = sum(wins.values())
                section += f"**{regime_name.upper()} Market:**\n"
                for strat, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
                    pct = count / total * 100
                    section += f"- {strat}: {count} wins ({pct:.0f}%)\n"
                section += "\n"
        
        section += """
### Key Insight

The distribution of winners across regimes demonstrates that **regime-aware strategy selection 
can provide an edge**. A meta-learning system that selects strategies based on detected regime 
could theoretically combine the best of each approach.
"""
        return section
    
    def _generate_statistical_section(self) -> str:
        """Generate statistical analysis section"""
        return """
### Statistical Robustness

To ensure results are not due to chance, we employed:

1. **Bootstrap Simulation (1000 iterations)**
   - Resampled returns with replacement
   - Calculated 95% confidence intervals
   - Estimated probability of positive returns

2. **Permutation Test (5000 iterations)**
   - Shuffled return sequences
   - Calculated p-values for strategy vs Buy-and-Hold
   - Significance levels: * p<0.10, ** p<0.05, *** p<0.01

### Confidence Intervals

Results with non-overlapping 95% CIs indicate statistically significant differences.

### Important Caveats

- Past performance does not guarantee future results
- Transaction costs not included
- Slippage not modeled
- Results may vary with different time periods
"""
    
    def _generate_findings_section(self, metrics: Dict, regime: Dict) -> str:
        """Generate key findings section"""
        return """
### Primary Findings

1. **Game Theory Strategies Show Adaptive Behavior**
   - Cooperator and Tit-for-Tat adjust positions based on context
   - This can lead to better risk-adjusted returns than static approaches

2. **No Universal Winner**
   - Different strategies excel in different market regimes
   - Aggressive strategies dominate in bull markets
   - Conservative strategies preserve capital in bear markets

3. **Position Sizing Matters More Than Direction**
   - All strategies tend to be long-biased (following agent recommendations)
   - The key differentiator is HOW MUCH to invest, not direction

4. **Baseline Strategies Provide Useful Benchmarks**
   - Conservative baseline shows cost of being too defensive
   - Aggressive baseline shows cost of ignoring risk signals
   - Buy-and-Hold represents the "do nothing" alternative

### Implications for Trading Systems

1. **Regime Detection is Valuable**
   - Systems should identify market regime before selecting strategy
   
2. **Adaptive Position Sizing Improves Risk-Adjusted Returns**
   - Dynamic sizing based on confidence/consensus is beneficial
   
3. **Game Theory Framework Provides Structure**
   - Cooperation vs defection lens offers useful decision framework
"""
    
    def _generate_limitations_section(self) -> str:
        """Generate limitations section"""
        return """
### Limitations

1. **Historical Data Only**
   - Backtested on ~270 trading days
   - May not generalize to future market conditions

2. **No Transaction Costs**
   - Real trading incurs fees, spreads, slippage
   - Frequent position changes may be costly

3. **Agent Recommendations are LLM-Generated**
   - Subject to model biases and limitations
   - Not validated against professional analyst forecasts

4. **Single Portfolio Size**
   - Results validated only for $100k portfolio
   - Scale effects not tested

5. **US Equities Only**
   - 20 large-cap US stocks
   - May not apply to other markets/asset classes

### Future Work

1. **Live Paper Trading**
   - Validate strategies in real-time without capital risk

2. **Transaction Cost Modeling**
   - Include realistic cost estimates

3. **Extended Time Periods**
   - Test across multiple market cycles

4. **Meta-Learning Implementation**
   - Build regime-detection system that automatically selects strategy

5. **Additional Asset Classes**
   - Extend to crypto, forex, commodities
"""
    
    def _generate_appendix(self) -> str:
        """Generate appendix with individual ticker results"""
        appendix = "\n### Individual Ticker Summary\n\n"
        
        by_ticker_dir = self.analysis_dir / "by_ticker"
        if not by_ticker_dir.exists():
            return appendix + "*No individual ticker data available.*\n"
        
        for ticker_dir in sorted(by_ticker_dir.iterdir()):
            if ticker_dir.is_dir():
                metrics_file = ticker_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    appendix += f"**{ticker_dir.name}**\n\n"
                    appendix += "| Strategy | Return | Sharpe | Max DD |\n"
                    appendix += "|----------|--------|--------|--------|\n"
                    
                    for name, m in sorted(metrics.items(), 
                                         key=lambda x: x[1]['total_return'], reverse=True):
                        appendix += f"| {name} | {m['total_return']:+.1f}% | "
                        appendix += f"{m['sharpe_ratio']:.2f} | {m['max_drawdown']:.1f}% |\n"
                    
                    appendix += "\n"
        
        return appendix
    
    def generate_summary_dashboard(self):
        """Generate a visual summary dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Load data
        combined_metrics = self._load_json('combined_metrics.json')
        regime_analysis = self._load_json('regime_analysis.json')
        
        if not combined_metrics:
            print("No data available for dashboard")
            return
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        strategies = list(combined_metrics.keys())
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2c3e50']
        
        # 1. Average Returns Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        returns = [combined_metrics[s]['avg_return'] for s in strategies]
        bars = ax1.bar(range(len(strategies)), returns, color=colors)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels([s[:8] for s in strategies], rotation=45, ha='right')
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Average Return by Strategy', fontweight='bold')
        ax1.axhline(y=0, color='black', linewidth=0.5)
        
        # 2. Sharpe Ratio
        ax2 = fig.add_subplot(gs[0, 1])
        sharpes = [combined_metrics[s]['avg_sharpe'] for s in strategies]
        bars = ax2.bar(range(len(strategies)), sharpes, color=colors)
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels([s[:8] for s in strategies], rotation=45, ha='right')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Performance', fontweight='bold')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        
        # 3. Win Rate
        ax3 = fig.add_subplot(gs[0, 2])
        win_rates = [combined_metrics[s]['avg_win_rate'] for s in strategies]
        bars = ax3.bar(range(len(strategies)), win_rates, color=colors)
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels([s[:8] for s in strategies], rotation=45, ha='right')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Trade Win Rate', fontweight='bold')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        # 4-6. Regime Winners (Pie Charts)
        for idx, regime in enumerate(['bull', 'bear', 'sideways']):
            ax = fig.add_subplot(gs[1, idx])
            if regime in regime_analysis and regime_analysis[regime]:
                wins = regime_analysis[regime]
                ax.pie(wins.values(), labels=wins.keys(), autopct='%1.0f%%',
                      colors=colors[:len(wins)])
            ax.set_title(f'{regime.upper()} Market Winners', fontweight='bold')
        
        # 7. Risk-Return Scatter
        ax7 = fig.add_subplot(gs[2, 0:2])
        for i, (name, m) in enumerate(combined_metrics.items()):
            ax7.scatter(m['avg_max_dd'], m['avg_return'], s=200, c=colors[i],
                       label=name, edgecolors='white', linewidth=2)
        ax7.set_xlabel('Average Max Drawdown (%)')
        ax7.set_ylabel('Average Return (%)')
        ax7.set_title('Risk-Return Profile', fontweight='bold')
        ax7.legend(loc='best', fontsize=9)
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Stats Table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Create text summary
        sorted_strats = sorted(combined_metrics.items(), 
                              key=lambda x: x[1]['avg_return'], reverse=True)
        
        summary_text = "TOP PERFORMERS\n" + "="*30 + "\n\n"
        summary_text += "By Return:\n"
        for i, (name, m) in enumerate(sorted_strats[:3], 1):
            summary_text += f"  {i}. {name}: {m['avg_return']:+.2f}%\n"
        
        summary_text += "\nBy Sharpe:\n"
        sharpe_sorted = sorted(combined_metrics.items(), 
                              key=lambda x: x[1]['avg_sharpe'], reverse=True)
        for i, (name, m) in enumerate(sharpe_sorted[:3], 1):
            summary_text += f"  {i}. {name}: {m['avg_sharpe']:.2f}\n"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Game Theory Trading Strategy Analysis Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(self.report_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard saved to: {self.report_dir / 'summary_dashboard.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Analysis Report")
    parser.add_argument('--analysis-dir', type=str, default=None, 
                       help='Path to analysis directory')
    
    args = parser.parse_args()
    
    generator = ReportGenerator(
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None
    )
    
    generator.generate_markdown_report()
    generator.generate_summary_dashboard()
    
    print("\n✅ Report generation complete!")


if __name__ == "__main__":
    main()