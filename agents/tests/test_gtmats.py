"""
run_gt_tournament_v3.py - CORRECT Game Theory Tournament

This version PROPERLY uses your LLM pipeline outputs:
- Reads the ACTUAL position sizes from aggressive/neutral/conservative_eval.json
- Uses the ACTUAL portfolio size from date_info.json
- Game theory agents CHOOSE which evaluation to follow (not modify positions)
- Compares against a 100% market exposure benchmark

Your LLM Pipeline (Phases 1-5) already calculated:
  - aggressive_eval.json: High-risk position recommendation
  - neutral_eval.json: Balanced position recommendation  
  - conservative_eval.json: Low-risk position recommendation

Game Theory Agents (Phase 6) CHOOSE which to follow based on their strategy.

Usage:
    python run_gt_tournament_v3.py AAPL
    python run_gt_tournament_v3.py AAPL --portfolio 100000
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# DATA STRUCTURES - Matching your actual JSON files
# ============================================================================

@dataclass
class RiskEvaluation:
    """
    Parsed from your aggressive/neutral/conservative_eval.json files
    These are the ACTUAL recommendations from your LLM risk debaters
    """
    profile: str           # "AGGRESSIVE", "NEUTRAL", "CONSERVATIVE"
    stance: str            # "BUY", "HOLD", "AVOID", "SMALL BUY"
    position_size: float   # Decimal (0.195 = 19.5%)
    position_dollars: float  # Actual dollar amount
    confidence: str        # "HIGH", "MEDIUM", "LOW"
    reasoning: str
    
    @classmethod
    def from_json(cls, data: Dict, portfolio_size: int) -> 'RiskEvaluation':
        """Parse from your actual JSON structure"""
        pos_size = data.get('position_size', 0)
        
        # Handle edge cases
        if pos_size is None:
            pos_size = 0
        if isinstance(pos_size, str):
            pos_size = float(pos_size.replace('%', '')) / 100
        
        return cls(
            profile=data.get('profile', 'UNKNOWN'),
            stance=data.get('stance', 'HOLD'),
            position_size=pos_size,
            position_dollars=pos_size * portfolio_size,
            confidence=data.get('confidence', 'LOW'),
            reasoning=data.get('reasoning', '')[:200]
        )


@dataclass
class MarketData:
    """Parsed from your date_info.json"""
    date: str
    daily_return: float
    open_price: float
    close_price: float
    portfolio_size: int
    sample_num: int
    actual_day: int


@dataclass
class AgentDecision:
    """What an agent decides to do"""
    evaluation_followed: str  # 'aggressive', 'neutral', 'conservative'
    position_size: float      # From the chosen evaluation
    position_dollars: float   # Actual dollars
    reasoning: str


# ============================================================================
# BASE AGENT
# ============================================================================

class BaseGameTheoryAgent(ABC):
    """
    Base class for game theory agents.
    
    Key concept: Agents CHOOSE which LLM evaluation to follow.
    They don't create their own positions - they select from:
      - aggressive_eval.json recommendations
      - neutral_eval.json recommendations
      - conservative_eval.json recommendations
    """
    
    def __init__(self, name: str, strategy_type: str, description: str):
        self.name = name
        self.strategy_type = strategy_type
        self.description = description
        
        # Score tracks how well the agent's choices have worked
        # Range: -10 to +10
        self.score = 0
        
        # Financial tracking
        self.total_return = 1.0  # Cumulative multiplier
        self.daily_returns: List[float] = []
        self.positions_taken: List[float] = []
        
        # History
        self.history: List[Dict] = []
        self.score_history: List[int] = [0]
        
        # Statistics
        self.evaluations_followed = {'aggressive': 0, 'neutral': 0, 'conservative': 0}
        self.correct_calls = 0
        self.wrong_calls = 0
    
    @abstractmethod
    def choose_evaluation(self, 
                         evals: Dict[str, RiskEvaluation],
                         market_data: MarketData,
                         other_agents: List['BaseGameTheoryAgent']) -> str:
        """
        Choose which evaluation to follow: 'aggressive', 'neutral', or 'conservative'
        This is the core decision each agent makes differently.
        """
        pass
    
    def make_decision(self,
                     evals: Dict[str, RiskEvaluation],
                     market_data: MarketData,
                     other_agents: List['BaseGameTheoryAgent']) -> AgentDecision:
        """Execute decision by choosing an evaluation to follow"""
        
        choice = self.choose_evaluation(evals, market_data, other_agents)
        chosen_eval = evals[choice]
        
        self.evaluations_followed[choice] += 1
        
        return AgentDecision(
            evaluation_followed=choice,
            position_size=chosen_eval.position_size,
            position_dollars=chosen_eval.position_dollars,
            reasoning=f"Following {choice} eval: {chosen_eval.stance} @ {chosen_eval.position_size*100:.1f}%"
        )
    
    def update_after_market(self, market_return: float, decision: AgentDecision) -> Dict:
        """Update agent state after seeing market outcome"""
        
        # Calculate return based on position
        trade_return = market_return * decision.position_size
        self.daily_returns.append(trade_return)
        self.positions_taken.append(decision.position_size)
        
        # Update cumulative return
        self.total_return *= (1 + trade_return)
        
        # Update score based on choice quality
        old_score = self.score
        choice = decision.evaluation_followed
        
        if market_return > 0.002:  # Market up > 0.2%
            if choice == 'aggressive':
                self.score = min(10, self.score + 3)
                self.correct_calls += 1
                outcome = "✓ Aggressive in UP market (+3)"
            elif choice == 'neutral':
                self.score = min(10, self.score + 1)
                outcome = "~ Neutral in UP market (+1)"
            else:
                self.score = max(-10, self.score - 1)
                self.wrong_calls += 1
                outcome = "✗ Conservative in UP market (-1)"
                
        elif market_return < -0.002:  # Market down > 0.2%
            if choice == 'aggressive':
                self.score = max(-10, self.score - 3)
                self.wrong_calls += 1
                outcome = "✗ Aggressive in DOWN market (-3)"
            elif choice == 'neutral':
                self.score = max(-10, self.score - 1)
                outcome = "~ Neutral in DOWN market (-1)"
            else:
                self.score = min(10, self.score + 2)
                self.correct_calls += 1
                outcome = "✓ Conservative in DOWN market (+2)"
        else:
            outcome = "- Flat market (no change)"
        
        self.score_history.append(self.score)
        
        # Record in history
        self.history.append({
            'choice': choice,
            'position_size': decision.position_size,
            'market_return': market_return,
            'trade_return': trade_return,
            'score_before': old_score,
            'score_after': self.score,
            'outcome': outcome
        })
        
        return {
            'trade_return_pct': trade_return * 100,
            'total_return_pct': (self.total_return - 1) * 100,
            'score_change': self.score - old_score,
            'outcome': outcome
        }
    
    def get_summary(self) -> Dict:
        """Get agent performance summary"""
        total_decisions = sum(self.evaluations_followed.values())
        
        return {
            'name': self.name,
            'strategy': self.strategy_type,
            'total_return_pct': (self.total_return - 1) * 100,
            'final_score': self.score,
            'total_decisions': total_decisions,
            'correct_calls': self.correct_calls,
            'wrong_calls': self.wrong_calls,
            'accuracy': self.correct_calls / (self.correct_calls + self.wrong_calls) if (self.correct_calls + self.wrong_calls) > 0 else 0,
            'avg_position': np.mean(self.positions_taken) if self.positions_taken else 0,
            'evaluation_distribution': {
                k: v / total_decisions if total_decisions > 0 else 0 
                for k, v in self.evaluations_followed.items()
            }
        }


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class CooperatorAgent(BaseGameTheoryAgent):
    """
    COOPERATOR - Adaptive Risk Manager
    
    Strategy: Adjusts which evaluation to follow based on confidence score
    - High score (≥5): Follow aggressive (riding hot streak)
    - Low score (≤-5): Follow conservative (protect after losses)
    - Middle: Follow neutral (balanced approach)
    
    This tests: Does adapting risk based on recent performance work?
    """
    
    def __init__(self):
        super().__init__(
            name="Cooperator",
            strategy_type="Adaptive Risk Manager",
            description="Follows aggressive when confident, conservative when scared"
        )
    
    def choose_evaluation(self, evals, market_data, other_agents) -> str:
        if self.score >= 5:
            return 'aggressive'
        elif self.score <= -5:
            return 'conservative'
        else:
            return 'neutral'


class DefectorAgent(BaseGameTheoryAgent):
    """
    DEFECTOR - Aggressive Risk Taker
    
    Strategy: Almost always follow aggressive evaluation
    - Only goes conservative at extreme negative score (≤-8)
    - Believes maximum aggression wins long-term
    
    This tests: Does consistent aggression beat adaptive strategies?
    """
    
    def __init__(self):
        super().__init__(
            name="Defector",
            strategy_type="Aggressive Risk Taker",
            description="Always aggressive unless in survival mode"
        )
    
    def choose_evaluation(self, evals, market_data, other_agents) -> str:
        if self.score <= -8:
            return 'conservative'  # Survival mode
        return 'aggressive'


class TitForTatAgent(BaseGameTheoryAgent):
    """
    TIT-FOR-TAT - Momentum Follower
    
    Strategy: Copy whatever the highest-scoring agent is doing
    - First decision: Start with neutral
    - After: Follow the leader's evaluation choice
    
    This tests: Does following winners work?
    """
    
    def __init__(self):
        super().__init__(
            name="Tit-for-Tat",
            strategy_type="Momentum Follower",
            description="Copies the winning agent's strategy"
        )
        self.last_choice = 'neutral'
    
    def choose_evaluation(self, evals, market_data, other_agents) -> str:
        if not self.history:
            self.last_choice = 'neutral'
            return 'neutral'
        
        # Find the leader (highest score, excluding self)
        others = [a for a in other_agents if a.name != self.name]
        if not others:
            return self.last_choice
        
        leader = max(others, key=lambda a: a.score)
        
        # If I'm beating everyone, keep doing what I'm doing
        if self.score >= leader.score:
            return self.last_choice
        
        # Copy leader's last choice
        if leader.history:
            self.last_choice = leader.history[-1]['choice']
        
        return self.last_choice


class StaticAggressiveAgent(BaseGameTheoryAgent):
    """
    STATIC AGGRESSIVE - Control (Always Aggressive)
    
    Strategy: Always follow aggressive evaluation, no matter what
    
    This is a CONTROL to compare adaptive strategies against.
    """
    
    def __init__(self):
        super().__init__(
            name="Static-Aggressive",
            strategy_type="Always Aggressive (Control)",
            description="Always follows aggressive evaluation"
        )
    
    def choose_evaluation(self, evals, market_data, other_agents) -> str:
        return 'aggressive'


class StaticConservativeAgent(BaseGameTheoryAgent):
    """
    STATIC CONSERVATIVE - Control (Always Conservative)
    
    Strategy: Always follow conservative evaluation, no matter what
    
    This is a CONTROL to show the cost of being too defensive.
    """
    
    def __init__(self):
        super().__init__(
            name="Static-Conservative",
            strategy_type="Always Conservative (Control)",
            description="Always follows conservative evaluation"
        )
    
    def choose_evaluation(self, evals, market_data, other_agents) -> str:
        return 'conservative'


class BuyAndHoldAgent(BaseGameTheoryAgent):
    """
    BUY AND HOLD - Market Benchmark
    
    Strategy: Always 100% invested (ignores all evaluations)
    
    This is THE BENCHMARK - if agents can't beat this, they add no value.
    """
    
    def __init__(self):
        super().__init__(
            name="Buy-and-Hold",
            strategy_type="Market Benchmark",
            description="Always 100% invested"
        )
        self.fixed_position = 1.0
    
    def choose_evaluation(self, evals, market_data, other_agents) -> str:
        return 'aggressive'  # For tracking purposes only
    
    def make_decision(self, evals, market_data, other_agents) -> AgentDecision:
        """Override to always use 100% position"""
        return AgentDecision(
            evaluation_followed='market',
            position_size=self.fixed_position,
            position_dollars=market_data.portfolio_size * self.fixed_position,
            reasoning="Buy & Hold: 100% market exposure"
        )


# ============================================================================
# TOURNAMENT
# ============================================================================

class GameTheoryTournament:
    """
    Runs the game theory tournament using your collected data
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find paths
        self.project_root = self._find_project_root()
        self.data_path = self.project_root / "outputs" / "game_theory" / self.ticker
        self.output_path = self.project_root / "outputs" / "gtmats_result" / self.ticker / self.timestamp
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"No data found at: {self.data_path}")
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        
        # Initialize agents
        self.agents = [
            CooperatorAgent(),
            DefectorAgent(),
            TitForTatAgent(),
            StaticAggressiveAgent(),
            StaticConservativeAgent(),
            BuyAndHoldAgent()
        ]
        
        # Results storage
        self.results: List[Dict] = []
        
        print(f"Tournament: {self.ticker}")
        print(f"Data: {self.data_path}")
        print(f"Output: {self.output_path}")
    
    def _find_project_root(self) -> Path:
        """Find project root"""
        current = Path.cwd()
        for _ in range(5):
            if (current / "outputs" / "game_theory").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _discover_samples(self, portfolio_size: int) -> List[int]:
        """Find available samples"""
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
        """Load all data for a sample"""
        sample_path = self.data_path / f"portfolio_{portfolio_size}" / f"sample_{sample_num}"
        
        if not sample_path.exists():
            return None
        
        try:
            data = {}
            
            # Load evaluations
            for eval_type in ['aggressive', 'neutral', 'conservative']:
                eval_file = sample_path / f"{eval_type}_eval.json"
                if eval_file.exists():
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        data[eval_type] = json.load(f)
            
            # Load date info
            date_file = sample_path / "date_info.json"
            if date_file.exists():
                with open(date_file, 'r', encoding='utf-8') as f:
                    data['date_info'] = json.load(f)
            
            # Validate
            if all(k in data for k in ['aggressive', 'neutral', 'conservative', 'date_info']):
                return data
            
            return None
            
        except Exception as e:
            print(f"Error loading sample {sample_num}: {e}")
            return None
    
    def run(self, portfolio_size: int = 100000, max_samples: Optional[int] = None):
        """Run the tournament"""
        
        print(f"\n{'='*80}")
        print(f"GAME THEORY TOURNAMENT - {self.ticker}")
        print(f"{'='*80}")
        print(f"Portfolio Size: ${portfolio_size:,}")
        
        # Discover samples
        samples = self._discover_samples(portfolio_size)
        if not samples:
            print(f"ERROR: No samples found for portfolio_{portfolio_size}")
            print(f"Available portfolios: {list(self.data_path.glob('portfolio_*'))}")
            return
        
        samples_to_run = samples[:max_samples] if max_samples else samples
        print(f"Samples: {len(samples_to_run)} of {len(samples)} available")
        print(f"{'='*80}\n")
        
        # Process each sample
        for sample_num in samples_to_run:
            raw_data = self._load_sample(portfolio_size, sample_num)
            if not raw_data:
                continue
            
            # Parse into proper structures
            date_info = raw_data['date_info']
            market_info = date_info.get('market_data', {})
            
            market_data = MarketData(
                date=date_info.get('date', ''),
                daily_return=market_info.get('daily_return', 0),
                open_price=market_info.get('open', 0),
                close_price=market_info.get('close', 0),
                portfolio_size=portfolio_size,
                sample_num=sample_num,
                actual_day=date_info.get('actual_day', sample_num)
            )
            
            evals = {
                'aggressive': RiskEvaluation.from_json(raw_data['aggressive'], portfolio_size),
                'neutral': RiskEvaluation.from_json(raw_data['neutral'], portfolio_size),
                'conservative': RiskEvaluation.from_json(raw_data['conservative'], portfolio_size)
            }
            
            # Print sample header
            print(f"Sample {sample_num} | {market_data.date} | Market: {market_data.daily_return*100:+.2f}%")
            print(f"  LLM Recommendations: Agg=${evals['aggressive'].position_dollars:,.0f} "
                  f"({evals['aggressive'].position_size*100:.1f}%) | "
                  f"Neu=${evals['neutral'].position_dollars:,.0f} "
                  f"({evals['neutral'].position_size*100:.1f}%) | "
                  f"Con=${evals['conservative'].position_dollars:,.0f} "
                  f"({evals['conservative'].position_size*100:.1f}%)")
            
            # Each agent makes decision
            sample_results = {
                'sample': sample_num,
                'date': market_data.date,
                'market_return': market_data.daily_return,
                'portfolio_size': portfolio_size,
                'evaluations': {
                    'aggressive': asdict(evals['aggressive']),
                    'neutral': asdict(evals['neutral']),
                    'conservative': asdict(evals['conservative'])
                },
                'agent_results': {}
            }
            
            for agent in self.agents:
                decision = agent.make_decision(evals, market_data, self.agents)
                result = agent.update_after_market(market_data.daily_return, decision)
                
                print(f"    {agent.name:20} → {decision.evaluation_followed:12} "
                      f"${decision.position_dollars:>8,.0f} ({decision.position_size*100:5.1f}%) "
                      f"| Ret: {result['trade_return_pct']:+5.2f}% "
                      f"| Total: {result['total_return_pct']:+6.2f}% "
                      f"| Score: {agent.score:+3d}")
                
                sample_results['agent_results'][agent.name] = {
                    'choice': decision.evaluation_followed,
                    'position_size': decision.position_size,
                    'position_dollars': decision.position_dollars,
                    'trade_return_pct': result['trade_return_pct'],
                    'total_return_pct': result['total_return_pct'],
                    'score': agent.score
                }
            
            self.results.append(sample_results)
            print()
        
        # Generate outputs
        self._generate_summary()
        self._generate_visualizations()
        self._save_results()
        self._print_final_results()
    
    def _generate_summary(self):
        """Generate tournament summary"""
        market_returns = [r['market_return'] for r in self.results]
        
        self.summary = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'total_samples': len(self.results),
            'portfolio_size': self.results[0]['portfolio_size'] if self.results else 0,
            'market': {
                'total_return_pct': (np.prod([1 + r for r in market_returns]) - 1) * 100,
                'avg_daily_pct': np.mean(market_returns) * 100,
                'volatility_pct': np.std(market_returns) * 100,
                'up_days': sum(1 for r in market_returns if r > 0),
                'down_days': sum(1 for r in market_returns if r < 0)
            },
            'rankings': sorted(
                [a.get_summary() for a in self.agents],
                key=lambda x: x['total_return_pct'],
                reverse=True
            )
        }
    
    def _generate_visualizations(self):
        """Generate charts"""
        if not self.results:
            return
        
        colors = {
            'Cooperator': '#2ecc71',
            'Defector': '#e74c3c',
            'Tit-for-Tat': '#3498db',
            'Static-Aggressive': '#f39c12',
            'Static-Conservative': '#9b59b6',
            'Buy-and-Hold': '#000000'
        }
        
        # 1. CUMULATIVE RETURNS
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
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Game Theory Tournament Results', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'cumulative_returns.png', dpi=150)
        plt.close()
        
        # 2. SCORE EVOLUTION
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for agent in self.agents:
            if agent.name != 'Buy-and-Hold':
                ax.plot(range(len(agent.score_history)), agent.score_history,
                       label=agent.name, color=colors.get(agent.name),
                       linewidth=2, marker='o', markersize=3)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=5, color='green', linestyle=':', alpha=0.5, label='Cooperator → Aggressive')
        ax.axhline(y=-5, color='orange', linestyle=':', alpha=0.5, label='Cooperator → Conservative')
        ax.axhline(y=-8, color='red', linestyle=':', alpha=0.5, label='Defector Survival Mode')
        
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.ticker} - Agent Score Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-12, 12)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'score_evolution.png', dpi=150)
        plt.close()
        
        # 3. EVALUATION CHOICES
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        eval_colors = {'aggressive': '#e74c3c', 'neutral': '#f1c40f', 'conservative': '#2ecc71', 'market': '#000000'}
        
        for i, agent in enumerate(self.agents):
            ax = axes[i]
            summary = agent.get_summary()
            dist = summary['evaluation_distribution']
            
            if agent.name == 'Buy-and-Hold':
                values = [100, 0, 0]
                labels = ['Market\n(100%)', 'N/A', 'N/A']
            else:
                values = [dist.get('aggressive', 0) * 100, 
                         dist.get('neutral', 0) * 100,
                         dist.get('conservative', 0) * 100]
                labels = ['Agg', 'Neu', 'Con']
            
            bars = ax.bar(labels, values, color=[eval_colors['aggressive'], eval_colors['neutral'], eval_colors['conservative']])
            ax.set_ylim(0, 105)
            ax.set_title(f"{agent.name}\n({summary['total_return_pct']:+.1f}%)", fontweight='bold')
            
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{val:.0f}%', ha='center', fontsize=10)
        
        fig.suptitle(f'{self.ticker} - Which Evaluation Each Agent Followed', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'evaluation_choices.png', dpi=150)
        plt.close()
        
        # 4. FINAL RANKING BAR CHART
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_agents = sorted(self.agents, key=lambda a: a.total_return, reverse=True)
        names = [a.name for a in sorted_agents]
        returns = [(a.total_return - 1) * 100 for a in sorted_agents]
        bar_colors = [colors.get(a.name, '#888') for a in sorted_agents]
        
        bars = ax.barh(names, returns, color=bar_colors)
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        for bar, val in zip(bars, returns):
            offset = 0.2 if val >= 0 else -0.2
            ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                   f'{val:+.2f}%', va='center', ha='left' if val >= 0 else 'right',
                   fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Total Return (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Final Performance Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'final_ranking.png', dpi=150)
        plt.close()
        
        print(f"Visualizations saved to: {self.output_path / 'visualizations'}")
    
    def _save_results(self):
        """Save all results"""
        # Summary JSON
        with open(self.output_path / 'tournament_summary.json', 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        # Full results
        with open(self.output_path / 'full_results.json', 'w') as f:
            json.dump({'summary': self.summary, 'samples': self.results}, f, indent=2)
        
        # CSV
        rows = []
        for r in self.results:
            row = {
                'Sample': r['sample'],
                'Date': r['date'],
                'Market_%': f"{r['market_return']*100:.2f}"
            }
            for agent_name, agent_result in r['agent_results'].items():
                prefix = agent_name.replace('-', '_').replace(' ', '_')
                row[f'{prefix}_Choice'] = agent_result['choice']
                row[f'{prefix}_Pos_%'] = f"{agent_result['position_size']*100:.1f}"
                row[f'{prefix}_Ret_%'] = f"{agent_result['trade_return_pct']:.2f}"
                row[f'{prefix}_Total_%'] = f"{agent_result['total_return_pct']:.2f}"
            rows.append(row)
        
        pd.DataFrame(rows).to_csv(self.output_path / 'results_matrix.csv', index=False)
        print(f"Results saved to: {self.output_path}")
    
    def _print_final_results(self):
        """Print final tournament results"""
        
        print(f"\n{'='*80}")
        print("FINAL TOURNAMENT RESULTS")
        print(f"{'='*80}\n")
        
        print(f"{'Rank':<5} {'Agent':<22} {'Return':>10} {'Score':>7} {'Accuracy':>10} {'Avg Pos':>10}")
        print("-" * 70)
        
        for i, r in enumerate(self.summary['rankings'], 1):
            print(f"{i:<5} {r['name']:<22} {r['total_return_pct']:>+9.2f}% "
                  f"{r['final_score']:>+6d} {r['accuracy']*100:>9.0f}% "
                  f"{r['avg_position']*100:>9.1f}%")
        
        print(f"\n{'='*70}")
        m = self.summary['market']
        print(f"MARKET: {m['total_return_pct']:+.2f}% | Up days: {m['up_days']} | Down days: {m['down_days']}")
        
        # Analysis
        print(f"\n{'='*70}")
        print("ANALYSIS:")
        
        buy_hold = next((a for a in self.agents if a.name == 'Buy-and-Hold'), None)
        if buy_hold:
            buy_hold_return = (buy_hold.total_return - 1) * 100
            
            for agent in self.agents:
                if agent.name != 'Buy-and-Hold':
                    agent_return = (agent.total_return - 1) * 100
                    capture_rate = (agent_return / buy_hold_return * 100) if buy_hold_return != 0 else 0
                    print(f"  {agent.name}: Captured {capture_rate:.0f}% of market return")
        
        print(f"\n{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Game Theory Tournament - Uses your LLM pipeline outputs correctly",
        epilog="""
This tournament uses the ACTUAL recommendations from your LLM risk evaluators.
Game theory agents CHOOSE which evaluation to follow based on their strategy.
        """
    )
    parser.add_argument('ticker', help='Stock ticker (e.g., AAPL)')
    parser.add_argument('--portfolio', type=int, default=100000,
                       help='Portfolio size (default: 100000)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Max samples to process (default: all)')
    
    args = parser.parse_args()
    
    try:
        tournament = GameTheoryTournament(args.ticker)
        tournament.run(portfolio_size=args.portfolio, max_samples=args.samples)
        print("✅ Tournament completed!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()