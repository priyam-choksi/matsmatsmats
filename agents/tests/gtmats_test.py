"""
smart_tournament_no_api.py - Smart Game Theory Tournament WITHOUT API Calls

Same logic as smart_tournament.py but uses rule-based simulation instead of GPT.
Use this to test while your other process is running.

Key difference from v3:
- Agents decide their OWN position (0-100%), not just follow evaluations
- Positions are based on agent personality + score + market context
- More realistic differentiation between strategies

Usage:
    python smart_tournament_no_api.py AAPL
    python smart_tournament_no_api.py AAPL --portfolio 100000
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketContext:
    """All information for a trading decision"""
    date: str
    ticker: str
    portfolio_size: int
    daily_return: float
    aggressive_eval: Dict
    neutral_eval: Dict
    conservative_eval: Dict


@dataclass
class AgentDecision:
    """Decision made by an agent"""
    position_percent: float  # 0-100
    position_dollars: float
    reasoning: str
    confidence: str


# ============================================================================
# BASE AGENT
# ============================================================================

class SmartAgent(ABC):
    """Base class for smart agents that decide their own positions"""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        
        # State
        self.score = 0  # -10 to +10
        self.total_return = 1.0
        self.history: List[Dict] = []
        self.daily_returns: List[float] = []
        self.positions: List[float] = []
        self.score_history = [0]
        
    @abstractmethod
    def decide_position(self, context: MarketContext) -> tuple:
        """
        Decide position percentage (0-100) and reasoning.
        Each agent implements this differently.
        Returns: (position_percent, reasoning, confidence)
        """
        pass
    
    def make_decision(self, context: MarketContext) -> AgentDecision:
        """Make a trading decision"""
        position_pct, reasoning, confidence = self.decide_position(context)
        
        # Clamp to valid range
        position_pct = max(0, min(100, position_pct))
        
        return AgentDecision(
            position_percent=position_pct,
            position_dollars=position_pct / 100 * context.portfolio_size,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def update_after_market(self, market_return: float, decision: AgentDecision) -> Dict:
        """Update state after seeing market outcome"""
        
        position_frac = decision.position_percent / 100
        trade_return = market_return * position_frac
        
        self.daily_returns.append(trade_return)
        self.positions.append(decision.position_percent)
        self.total_return *= (1 + trade_return)
        
        # Update score based on decision quality
        old_score = self.score
        
        if market_return > 0.002:  # Market up > 0.2%
            if decision.position_percent >= 40:
                self.score = min(10, self.score + 3)
                outcome = "✓ Aggressive in UP market (+3)"
            elif decision.position_percent >= 20:
                self.score = min(10, self.score + 1)
                outcome = "~ Moderate in UP market (+1)"
            else:
                self.score = max(-10, self.score - 1)
                outcome = "✗ Conservative in UP market (-1)"
                
        elif market_return < -0.002:  # Market down > 0.2%
            if decision.position_percent >= 40:
                self.score = max(-10, self.score - 3)
                outcome = "✗ Aggressive in DOWN market (-3)"
            elif decision.position_percent >= 20:
                self.score = max(-10, self.score - 1)
                outcome = "~ Moderate in DOWN market (-1)"
            else:
                self.score = min(10, self.score + 2)
                outcome = "✓ Conservative in DOWN market (+2)"
        else:
            outcome = "- Flat market"
        
        self.score_history.append(self.score)
        
        self.history.append({
            'position_percent': decision.position_percent,
            'market_return': market_return,
            'trade_return': trade_return,
            'trade_return_pct': trade_return * 100,
            'score_change': self.score - old_score,
            'outcome': outcome
        })
        
        return {
            'trade_return_pct': trade_return * 100,
            'total_return_pct': (self.total_return - 1) * 100,
            'score_change': self.score - old_score,
            'outcome': outcome
        }
    
    def get_summary(self) -> Dict:
        return {
            'name': self.name,
            'type': self.agent_type,
            'total_return_pct': (self.total_return - 1) * 100,
            'final_score': self.score,
            'avg_position': np.mean(self.positions) if self.positions else 0,
            'trades': len(self.history)
        }


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class CooperatorAgent(SmartAgent):
    """
    COOPERATOR - Adaptive Risk Manager
    
    Scales position based on confidence score:
    - High score → More aggressive (riding hot streak)
    - Low score → More defensive (protecting after losses)
    - Uses evaluations as guidance but makes own decision
    """
    
    def __init__(self):
        super().__init__("Cooperator", "Adaptive Risk Manager")
    
    def decide_position(self, context: MarketContext) -> tuple:
        # Get evaluation signals
        agg_pos = context.aggressive_eval.get('position_size', 0.15) * 100
        neu_pos = context.neutral_eval.get('position_size', 0.05) * 100
        con_pos = context.conservative_eval.get('position_size', 0.01) * 100
        
        agg_conf = context.aggressive_eval.get('confidence', 'LOW')
        
        # Base position on score
        if self.score >= 7:
            # Very confident - go aggressive
            base = 55 + (self.score - 7) * 5  # 55-70%
            reasoning = f"Score {self.score}: Hot streak! Going aggressive."
            confidence = 'HIGH'
        elif self.score >= 4:
            # Confident - above neutral
            base = 35 + (self.score - 4) * 5  # 35-50%
            reasoning = f"Score {self.score}: Building confidence, moderately aggressive."
            confidence = 'MEDIUM'
        elif self.score >= 0:
            # Neutral - follow neutral eval scaled up
            base = 20 + self.score * 3  # 20-32%
            reasoning = f"Score {self.score}: Neutral zone, balanced approach."
            confidence = 'MEDIUM'
        elif self.score >= -4:
            # Cautious
            base = 15 + (self.score + 4) * 2  # 15-23%
            reasoning = f"Score {self.score}: Recent losses, being cautious."
            confidence = 'LOW'
        else:
            # Defensive
            base = 5 + (self.score + 10) * 1.5  # 5-14%
            reasoning = f"Score {self.score}: Defensive mode, protecting capital."
            confidence = 'LOW'
        
        # Adjust based on aggressive eval confidence
        if agg_conf == 'HIGH' and self.score > 0:
            base *= 1.1
            reasoning += " Analysts are confident."
        
        # Add small randomness for realism
        position = base + random.uniform(-3, 3)
        
        return position, reasoning, confidence


class DefectorAgent(SmartAgent):
    """
    DEFECTOR - Aggressive Contrarian
    
    Always aggressive unless in survival mode:
    - Goes bigger than evaluations suggest
    - Only retreats at extreme negative score
    - Contrarian: more aggressive when others might be scared
    """
    
    def __init__(self):
        super().__init__("Defector", "Aggressive Contrarian")
    
    def decide_position(self, context: MarketContext) -> tuple:
        agg_pos = context.aggressive_eval.get('position_size', 0.15) * 100
        
        if self.score <= -8:
            # SURVIVAL MODE - only time Defector retreats
            position = 10 + random.uniform(0, 5)
            reasoning = f"SURVIVAL MODE: Score {self.score}. Temporary retreat."
            confidence = 'LOW'
        elif self.score <= -5:
            # Hurting but still aggressive
            position = 35 + random.uniform(0, 10)
            reasoning = f"Score {self.score}: Wounded but still fighting."
            confidence = 'MEDIUM'
        elif self.score >= 5:
            # Winning - press advantage
            position = 65 + (self.score - 5) * 3 + random.uniform(0, 5)  # 65-80%
            reasoning = f"Score {self.score}: Dominating! Maximum aggression."
            confidence = 'HIGH'
        else:
            # Normal aggressive
            position = 50 + self.score * 2 + random.uniform(0, 10)  # 40-60%
            reasoning = f"Score {self.score}: Fortune favors the bold."
            confidence = 'HIGH'
        
        # Defector always thinks analysts are too conservative
        if position < agg_pos * 2:
            position = max(position, agg_pos * 2.5)
            reasoning += " Analysts too timid."
        
        return position, reasoning, confidence


class TitForTatAgent(SmartAgent):
    """
    TIT-FOR-TAT - Momentum Follower
    
    Copies what's been working:
    - Starts moderate
    - If aggressive worked → go more aggressive
    - If conservative worked → stay conservative
    - Adapts to market regime
    """
    
    def __init__(self):
        super().__init__("Tit-for-Tat", "Momentum Follower")
        self.last_successful_position = 30  # Start moderate
    
    def decide_position(self, context: MarketContext) -> tuple:
        
        if len(self.history) < 2:
            # First trades - start moderate
            position = 30 + random.uniform(-5, 5)
            reasoning = "Starting moderate to gather data."
            confidence = 'MEDIUM'
        else:
            # Look at recent history
            recent = self.history[-3:]
            
            # What position size worked?
            winning_positions = [h['position_percent'] for h in recent if h['trade_return'] > 0]
            losing_positions = [h['position_percent'] for h in recent if h['trade_return'] < 0]
            
            if winning_positions:
                avg_winning = np.mean(winning_positions)
                self.last_successful_position = avg_winning
            
            # Score trend
            score_trend = self.score - self.score_history[-3] if len(self.score_history) >= 3 else 0
            
            if score_trend > 2:
                # Improving - do more of what's working
                position = self.last_successful_position * 1.15
                reasoning = f"Score improving (+{score_trend}). Increasing position."
                confidence = 'HIGH'
            elif score_trend < -2:
                # Declining - try opposite
                if self.last_successful_position > 40:
                    position = 25 + random.uniform(0, 10)
                    reasoning = f"Score declining ({score_trend}). Reducing exposure."
                else:
                    position = 45 + random.uniform(0, 10)
                    reasoning = f"Score declining ({score_trend}). Trying more aggressive."
                confidence = 'LOW'
            else:
                # Stable - continue current approach
                position = self.last_successful_position + random.uniform(-5, 5)
                reasoning = f"Score stable. Continuing current approach."
                confidence = 'MEDIUM'
        
        return position, reasoning, confidence


class ConservativeBaselineAgent(SmartAgent):
    """
    CONSERVATIVE BASELINE - Control Group
    
    Always takes small positions:
    - Never goes above 25%
    - Prioritizes capital preservation
    - Exists to show cost of being too defensive
    """
    
    def __init__(self):
        super().__init__("Conservative-Baseline", "Always Cautious (Control)")
    
    def decide_position(self, context: MarketContext) -> tuple:
        con_pos = context.conservative_eval.get('position_size', 0.01) * 100
        
        # Always conservative, but adjust slightly with score
        if self.score >= 5:
            position = 20 + random.uniform(0, 5)  # Max out at 25%
            reasoning = "Even with high score, staying cautious."
        elif self.score >= 0:
            position = 12 + random.uniform(0, 5)
            reasoning = "Baseline conservative approach."
        else:
            position = 5 + random.uniform(0, 5)
            reasoning = "Negative score reinforces caution."
        
        confidence = 'MEDIUM'
        return position, reasoning, confidence


class AggressiveBaselineAgent(SmartAgent):
    """
    AGGRESSIVE BASELINE - Control Group
    
    Always takes large positions:
    - Never goes below 40%
    - Maximizes market exposure
    - Exists to show pure aggressive play
    """
    
    def __init__(self):
        super().__init__("Aggressive-Baseline", "Always Bold (Control)")
    
    def decide_position(self, context: MarketContext) -> tuple:
        agg_pos = context.aggressive_eval.get('position_size', 0.15) * 100
        
        # Always aggressive
        if self.score >= 5:
            position = 70 + random.uniform(0, 10)  # 70-80%
            reasoning = "High score + aggressive = maximum position."
        elif self.score >= 0:
            position = 55 + random.uniform(0, 10)  # 55-65%
            reasoning = "Baseline aggressive approach."
        else:
            position = 40 + random.uniform(0, 10)  # 40-50% even when losing
            reasoning = "Staying aggressive despite losses."
        
        confidence = 'HIGH'
        return position, reasoning, confidence


class BuyAndHoldAgent(SmartAgent):
    """
    BUY AND HOLD - Market Benchmark
    
    Always 100% invested - the benchmark to beat.
    """
    
    def __init__(self):
        super().__init__("Buy-and-Hold", "Market Benchmark")
    
    def decide_position(self, context: MarketContext) -> tuple:
        return 100, "Always 100% invested.", 'HIGH'


# ============================================================================
# TOURNAMENT
# ============================================================================

class SmartTournament:
    """Tournament with smart position-deciding agents"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.project_root = self._find_project_root()
        self.data_path = self.project_root / "outputs" / "game_theory" / self.ticker
        self.output_path = self.project_root / "outputs" / "gtmats_result_smart_noapi" / self.ticker / f"smart_noapi_{self.timestamp}"
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"No data at: {self.data_path}")
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        
        # Initialize agents
        self.agents = [
            CooperatorAgent(),
            DefectorAgent(),
            TitForTatAgent(),
            ConservativeBaselineAgent(),
            AggressiveBaselineAgent(),
            BuyAndHoldAgent()
        ]
        
        self.results = []
        
        print(f"Smart Tournament (No API): {self.ticker}")
        print(f"Agents: {[a.name for a in self.agents]}")
        print(f"Output: {self.output_path}")
    
    def _find_project_root(self) -> Path:
        current = Path.cwd()
        for _ in range(5):
            if (current / "outputs").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _discover_samples(self, portfolio_size: int) -> List[int]:
        portfolio_path = self.data_path / f"portfolio_{portfolio_size}"
        if not portfolio_path.exists():
            return []
        
        samples = []
        for folder in portfolio_path.iterdir():
            if folder.is_dir() and folder.name.startswith("sample_"):
                try:
                    samples.append(int(folder.name.split("_")[1]))
                except:
                    pass
        return sorted(samples)
    
    def _load_sample(self, portfolio_size: int, sample_num: int) -> Optional[Dict]:
        sample_path = self.data_path / f"portfolio_{portfolio_size}" / f"sample_{sample_num}"
        if not sample_path.exists():
            return None
        
        try:
            data = {}
            for eval_type in ['aggressive', 'neutral', 'conservative']:
                with open(sample_path / f"{eval_type}_eval.json", 'r') as f:
                    data[eval_type] = json.load(f)
            with open(sample_path / "date_info.json", 'r') as f:
                data['date_info'] = json.load(f)
            return data
        except Exception as e:
            return None
    
    def run(self, portfolio_size: int = 100000, max_samples: Optional[int] = None):
        """Run tournament"""
        
        print(f"\n{'='*80}")
        print(f"SMART TOURNAMENT (NO API) - {self.ticker}")
        print(f"{'='*80}")
        print(f"Portfolio: ${portfolio_size:,}")
        
        samples = self._discover_samples(portfolio_size)
        if not samples:
            print("ERROR: No samples found")
            return
        
        samples_to_run = samples[:max_samples] if max_samples else samples
        print(f"Samples: {len(samples_to_run)}")
        print(f"{'='*80}\n")
        
        for sample_num in samples_to_run:
            raw_data = self._load_sample(portfolio_size, sample_num)
            if not raw_data:
                continue
            
            date_info = raw_data['date_info']
            market_data = date_info.get('market_data', {})
            
            context = MarketContext(
                date=date_info.get('date', ''),
                ticker=self.ticker,
                portfolio_size=portfolio_size,
                daily_return=market_data.get('daily_return', 0),
                aggressive_eval=raw_data['aggressive'],
                neutral_eval=raw_data['neutral'],
                conservative_eval=raw_data['conservative']
            )
            
            print(f"Sample {sample_num} | {context.date} | Market: {context.daily_return*100:+.2f}%")
            
            sample_results = {'sample': sample_num, 'date': context.date, 
                            'market_return': context.daily_return, 'agents': {}}
            
            for agent in self.agents:
                decision = agent.make_decision(context)
                result = agent.update_after_market(context.daily_return, decision)
                
                print(f"  {agent.name:24} | {decision.position_percent:5.1f}% | "
                      f"Ret: {result['trade_return_pct']:+6.2f}% | "
                      f"Total: {result['total_return_pct']:+7.2f}% | "
                      f"Score: {agent.score:+3d}")
                
                sample_results['agents'][agent.name] = {
                    'position': decision.position_percent,
                    'return_pct': result['trade_return_pct'],
                    'total_pct': result['total_return_pct'],
                    'score': agent.score
                }
            
            self.results.append(sample_results)
            print()
        
        self._generate_outputs()
        self._print_results()
    
    def _generate_outputs(self):
        """Generate visualizations and save results"""
        if not self.results:
            return
        
        colors = {
            'Cooperator': '#2ecc71',
            'Defector': '#e74c3c',
            'Tit-for-Tat': '#3498db',
            'Conservative-Baseline': '#9b59b6',
            'Aggressive-Baseline': '#f39c12',
            'Buy-and-Hold': '#000000'
        }
        
        # 1. Cumulative Returns
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for agent in self.agents:
            cumulative = [1.0]
            for ret in agent.daily_returns:
                cumulative.append(cumulative[-1] * (1 + ret))
            returns_pct = [(c - 1) * 100 for c in cumulative]
            
            style = '--' if agent.name == 'Buy-and-Hold' else '-'
            width = 3 if agent.name == 'Buy-and-Hold' else 2
            
            ax.plot(range(len(returns_pct)), returns_pct,
                   label=f"{agent.name} ({returns_pct[-1]:+.1f}%)",
                   color=colors.get(agent.name, '#888'),
                   linestyle=style, linewidth=width)
        
        ax.axhline(y=0, color='gray', alpha=0.3)
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Smart Tournament Results (No API)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'cumulative_returns.png', dpi=150)
        plt.close()
        
        # 2. Position Sizes
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for agent in self.agents:
            ax.plot(range(len(agent.positions)), agent.positions,
                   label=f"{agent.name} (avg: {np.mean(agent.positions):.0f}%)",
                   color=colors.get(agent.name), linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Position Size (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Position Sizes Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'position_sizes.png', dpi=150)
        plt.close()
        
        # 3. Score Evolution
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for agent in self.agents:
            if agent.name != 'Buy-and-Hold':
                ax.plot(range(len(agent.score_history)), agent.score_history,
                       label=agent.name, color=colors.get(agent.name), linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=-8, color='red', linestyle=':', alpha=0.5, label='Defector Survival')
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.ticker} - Score Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'score_evolution.png', dpi=150)
        plt.close()
        
        # 4. Final Ranking
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_agents = sorted(self.agents, key=lambda a: a.total_return, reverse=True)
        names = [a.name for a in sorted_agents]
        returns = [(a.total_return - 1) * 100 for a in sorted_agents]
        bar_colors = [colors.get(a.name, '#888') for a in sorted_agents]
        
        bars = ax.barh(names, returns, color=bar_colors)
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        for bar, val, agent in zip(bars, returns, sorted_agents):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{val:+.1f}% (avg pos: {np.mean(agent.positions):.0f}%)',
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Total Return (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Final Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'final_ranking.png', dpi=150)
        plt.close()
        
        # Save JSON
        summary = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'total_samples': len(self.results),
            'rankings': [a.get_summary() for a in sorted(self.agents, key=lambda a: a.total_return, reverse=True)]
        }
        
        with open(self.output_path / 'tournament_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(self.output_path / 'full_results.json', 'w') as f:
            json.dump({'summary': summary, 'samples': self.results}, f, indent=2)
        
        # CSV
        rows = []
        for r in self.results:
            row = {'Sample': r['sample'], 'Date': r['date'], 'Market_%': f"{r['market_return']*100:.2f}"}
            for name, data in r['agents'].items():
                prefix = name.replace('-', '_')
                row[f'{prefix}_Pos'] = f"{data['position']:.1f}"
                row[f'{prefix}_Ret'] = f"{data['return_pct']:.2f}"
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.output_path / 'results.csv', index=False)
        
        print(f"\nResults saved to: {self.output_path}")
    
    def _print_results(self):
        """Print final results"""
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}\n")
        
        sorted_agents = sorted(self.agents, key=lambda a: a.total_return, reverse=True)
        
        print(f"{'Rank':<5} {'Agent':<26} {'Return':>10} {'Score':>7} {'Avg Pos':>10}")
        print("-" * 65)
        
        for i, agent in enumerate(sorted_agents, 1):
            s = agent.get_summary()
            print(f"{i:<5} {agent.name:<26} {s['total_return_pct']:>+9.2f}% "
                  f"{s['final_score']:>+6d} {s['avg_position']:>9.1f}%")
        
        # Analysis
        bh = next((a for a in self.agents if a.name == 'Buy-and-Hold'), None)
        if bh:
            bh_ret = (bh.total_return - 1) * 100
            print(f"\n{'='*65}")
            print(f"BENCHMARK: {bh_ret:+.2f}%\n")
            print("Market Capture:")
            for agent in sorted_agents:
                if agent.name != 'Buy-and-Hold' and bh_ret != 0:
                    capture = (agent.total_return - 1) / (bh.total_return - 1) * 100
                    print(f"  {agent.name}: {capture:.0f}% of market (avg position: {np.mean(agent.positions):.0f}%)")
        
        print(f"\n{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Tournament - No API Required")
    parser.add_argument('ticker', help='Stock ticker')
    parser.add_argument('--portfolio', type=int, default=100000)
    parser.add_argument('--samples', type=int, default=None)
    
    args = parser.parse_args()
    
    try:
        tournament = SmartTournament(args.ticker)
        tournament.run(portfolio_size=args.portfolio, max_samples=args.samples)
        print("✅ Tournament completed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()