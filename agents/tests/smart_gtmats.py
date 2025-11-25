"""
smart_tournament.py - LLM-Powered Game Theory Tournament

REAL AI AGENTS that use GPT to make trading decisions.

Each agent has a PERSONALITY and STRATEGY that guides their reasoning:
- Cooperator: Adapts risk based on recent performance
- Defector: Aggressive contrarian, bets big
- Tit-for-Tat: Follows what's working
- Conservative Baseline: Always cautious
- Aggressive Baseline: Always bold

The LLM sees:
1. The three risk evaluations from your pipeline
2. Market data for the day
3. The agent's current score/performance
4. The agent's personality/strategy

Then it REASONS about what position to take (0-100%).

Usage:
    python smart_tournament.py AAPL
    python smart_tournament.py AAPL --portfolio 100000 --samples 20
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
import time

# OpenAI
from openai import OpenAI

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_openai_client():
    """Get OpenAI client with API key"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Try to load from file
        key_files = ['api_keys.txt', '../api_keys.txt', '../../api_keys.txt']
        for kf in key_files:
            if Path(kf).exists():
                with open(kf, 'r') as f:
                    for line in f:
                        if 'OPENAI' in line.upper() and '=' in line:
                            api_key = line.split('=')[1].strip().strip('"\'')
                            break
                if api_key:
                    break
    
    if not api_key:
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    
    return OpenAI(api_key=api_key)


MODEL = "gpt-4o-mini"  # Cost-effective, fast


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketContext:
    """All information available for a trading decision"""
    date: str
    ticker: str
    portfolio_size: int
    daily_return: float  # Previous day's return (for context)
    
    # From your LLM pipeline
    aggressive_eval: Dict
    neutral_eval: Dict
    conservative_eval: Dict
    
    def to_prompt_string(self) -> str:
        """Format for LLM prompt"""
        return f"""
MARKET DATE: {self.date}
TICKER: {self.ticker}
PORTFOLIO SIZE: ${self.portfolio_size:,}

=== RISK EVALUATION SUMMARIES (from AI analysts) ===

AGGRESSIVE EVALUATION:
  Stance: {self.aggressive_eval.get('stance', 'N/A')}
  Recommended Position: {self.aggressive_eval.get('position_size', 0)*100:.1f}% (${self.aggressive_eval.get('position_size', 0)*self.portfolio_size:,.0f})
  Confidence: {self.aggressive_eval.get('confidence', 'N/A')}
  Reasoning: {self.aggressive_eval.get('reasoning', 'N/A')}

NEUTRAL EVALUATION:
  Stance: {self.neutral_eval.get('stance', 'N/A')}
  Recommended Position: {self.neutral_eval.get('position_size', 0)*100:.1f}% (${self.neutral_eval.get('position_size', 0)*self.portfolio_size:,.0f})
  Confidence: {self.neutral_eval.get('confidence', 'N/A')}
  Reasoning: {self.neutral_eval.get('reasoning', 'N/A')}
  Expected Value: {self.neutral_eval.get('expected_value', 'N/A')}

CONSERVATIVE EVALUATION:
  Stance: {self.conservative_eval.get('stance', 'N/A')}
  Recommended Position: {self.conservative_eval.get('position_size', 0)*100:.1f}% (${self.conservative_eval.get('position_size', 0)*self.portfolio_size:,.0f})
  Confidence: {self.conservative_eval.get('confidence', 'N/A')}
  Reasoning: {self.conservative_eval.get('reasoning', 'N/A')}
  Red Flags: {self.conservative_eval.get('red_flags', [])}
"""


@dataclass
class AgentDecision:
    """Decision made by an agent"""
    position_percent: float  # 0-100
    position_dollars: float
    reasoning: str
    confidence: str


# ============================================================================
# LLM-POWERED AGENTS
# ============================================================================

class SmartTradingAgent(ABC):
    """Base class for LLM-powered trading agents"""
    
    def __init__(self, name: str, personality: str, strategy: str):
        self.name = name
        self.personality = personality
        self.strategy = strategy
        self.client = get_openai_client()
        
        # Performance tracking
        self.score = 0
        self.total_return = 1.0
        self.history: List[Dict] = []
        self.daily_returns: List[float] = []
        self.positions: List[float] = []
        self.score_history = [0]
        
    def get_system_prompt(self) -> str:
        """Build system prompt with agent's personality"""
        return f"""You are {self.name}, a trading agent in a game theory tournament.

PERSONALITY:
{self.personality}

STRATEGY:
{self.strategy}

YOUR CURRENT STATE:
- Score: {self.score} (range: -10 to +10)
- Total Return: {(self.total_return - 1) * 100:+.2f}%
- Recent decisions: {self._get_recent_history()}

SCORING RULES (how your score changes):
- Market UP + You were aggressive (>30% position): Score +3
- Market UP + You were moderate (15-30%): Score +1  
- Market UP + You were conservative (<15%): Score -1
- Market DOWN + You were aggressive: Score -3
- Market DOWN + You were moderate: Score -1
- Market DOWN + You were conservative: Score +2

YOUR TASK:
Analyze the market context and decide what percentage of the portfolio to invest (0-100%).
You are NOT limited to the analyst recommendations - use them as input but make your OWN decision.

Respond in this EXACT JSON format:
{{
    "position_percent": <number 0-100>,
    "reasoning": "<your reasoning in 2-3 sentences>",
    "confidence": "<HIGH/MEDIUM/LOW>"
}}
"""
    
    def _get_recent_history(self) -> str:
        """Get recent decision history for context"""
        if not self.history:
            return "No previous decisions"
        
        recent = self.history[-3:]
        summary = []
        for h in recent:
            outcome = "✓" if h.get('score_change', 0) > 0 else "✗" if h.get('score_change', 0) < 0 else "~"
            summary.append(f"{outcome} {h['position_percent']:.0f}% → {h.get('trade_return_pct', 0):+.2f}%")
        return " | ".join(summary)
    
    def make_decision(self, context: MarketContext) -> AgentDecision:
        """Use LLM to make trading decision"""
        
        user_prompt = f"""
{context.to_prompt_string()}

Based on your personality and strategy, what position should you take?
Remember: You can choose ANY position from 0% to 100%, not just what the analysts recommend.

Think about:
1. What do the three evaluations tell you?
2. Given your current score ({self.score}), should you be more aggressive or defensive?
3. What does your strategy say to do in this situation?

Respond with JSON only.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            position_pct = float(data.get('position_percent', 20))
            position_pct = max(0, min(100, position_pct))  # Clamp to 0-100
            
            return AgentDecision(
                position_percent=position_pct,
                position_dollars=position_pct / 100 * context.portfolio_size,
                reasoning=data.get('reasoning', 'No reasoning provided'),
                confidence=data.get('confidence', 'MEDIUM')
            )
            
        except Exception as e:
            print(f"  Warning: {self.name} LLM error: {e}")
            # Fallback to moderate position
            return AgentDecision(
                position_percent=20,
                position_dollars=0.2 * context.portfolio_size,
                reasoning=f"Fallback decision due to error: {str(e)[:50]}",
                confidence='LOW'
            )
    
    def update_after_market(self, market_return: float, decision: AgentDecision) -> Dict:
        """Update agent state after seeing market outcome"""
        
        position_frac = decision.position_percent / 100
        trade_return = market_return * position_frac
        
        self.daily_returns.append(trade_return)
        self.positions.append(decision.position_percent)
        self.total_return *= (1 + trade_return)
        
        # Update score
        old_score = self.score
        
        if market_return > 0.002:  # Market up
            if decision.position_percent > 30:
                self.score = min(10, self.score + 3)
            elif decision.position_percent > 15:
                self.score = min(10, self.score + 1)
            else:
                self.score = max(-10, self.score - 1)
        elif market_return < -0.002:  # Market down
            if decision.position_percent > 30:
                self.score = max(-10, self.score - 3)
            elif decision.position_percent > 15:
                self.score = max(-10, self.score - 1)
            else:
                self.score = min(10, self.score + 2)
        
        self.score_history.append(self.score)
        
        # Record history
        self.history.append({
            'position_percent': decision.position_percent,
            'market_return': market_return,
            'trade_return': trade_return,
            'trade_return_pct': trade_return * 100,
            'score_before': old_score,
            'score_after': self.score,
            'score_change': self.score - old_score,
            'reasoning': decision.reasoning
        })
        
        return {
            'trade_return_pct': trade_return * 100,
            'total_return_pct': (self.total_return - 1) * 100,
            'score_change': self.score - old_score
        }
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'name': self.name,
            'total_return_pct': (self.total_return - 1) * 100,
            'final_score': self.score,
            'avg_position': np.mean(self.positions) if self.positions else 0,
            'total_trades': len(self.history)
        }


# ============================================================================
# SPECIFIC AGENT IMPLEMENTATIONS
# ============================================================================

class CooperatorAgent(SmartTradingAgent):
    """Adaptive risk manager - scales with confidence"""
    
    def __init__(self):
        super().__init__(
            name="Cooperator",
            personality="""You are an ADAPTIVE risk manager who believes in scaling position size with confidence.

When you're WINNING (score > 3): You've found the market's rhythm. Be more aggressive.
When you're LOSING (score < -3): The market is teaching you humility. Reduce risk.
When NEUTRAL: Stay balanced, wait for clearer signals.

You believe in "riding winners and cutting losers" - your position size should reflect your recent success.""",
            
            strategy="""DECISION RULES:
- Score >= 5: Consider 40-70% positions (you're hot, press your advantage)
- Score 0-4: Consider 20-40% positions (balanced approach)
- Score -4 to -1: Consider 10-25% positions (cautious, rebuilding)
- Score <= -5: Consider 5-15% positions (protect capital, wait for edge)

Always consider the analyst recommendations but adjust based on your score."""
        )


class DefectorAgent(SmartTradingAgent):
    """Aggressive contrarian - maximum risk"""
    
    def __init__(self):
        super().__init__(
            name="Defector",
            personality="""You are an AGGRESSIVE CONTRARIAN who believes fortune favors the bold.

You think most investors are too timid. When analysts recommend 20%, you see opportunity for 50%+.
You only back off when you're getting destroyed (score <= -8).

Your motto: "In a bull market, the biggest bull wins. In a bear market, cash is king - but only temporarily.""",
            
            strategy="""DECISION RULES:
- Score > -8: BE AGGRESSIVE. Consider 50-80% positions.
  - When analysts are cautious, you see opportunity
  - When they're aggressive, you go MORE aggressive
- Score <= -8: SURVIVAL MODE. Drop to 10-20% until you recover.

You believe the aggressive analyst recommendation is usually TOO CONSERVATIVE."""
        )


class TitForTatAgent(SmartTradingAgent):
    """Momentum follower - copies what works"""
    
    def __init__(self):
        super().__init__(
            name="Tit-for-Tat",
            personality="""You are a MOMENTUM FOLLOWER who learns from what's working.

You don't have strong opinions - you follow success. 
If aggressive positions have been working, go aggressive.
If conservative has been working, be conservative.

You're humble enough to admit you don't know, but smart enough to recognize patterns.""",
            
            strategy="""DECISION RULES:
- Look at your recent history: what position sizes led to positive returns?
- If your score is rising: keep doing what you're doing
- If your score is falling: try the opposite approach
- First few trades: start moderate (25-35%) to gather data

Key insight: The market goes through regimes. Adapt to the current regime."""
        )


class ConservativeBaselineAgent(SmartTradingAgent):
    """Always cautious - control group"""
    
    def __init__(self):
        super().__init__(
            name="Conservative-Baseline",
            personality="""You are a CONSERVATIVE baseline agent who prioritizes capital preservation.

You believe in small, consistent positions. Better to miss some upside than risk big losses.
You typically agree with or go BELOW the conservative analyst recommendation.""",
            
            strategy="""DECISION RULES:
- Default to 5-15% positions
- Never exceed 25% regardless of opportunity
- When in doubt, go smaller
- Red flags from conservative analyst = reduce further

You exist to show what pure defensive play looks like."""
        )


class AggressiveBaselineAgent(SmartTradingAgent):
    """Always bold - control group"""
    
    def __init__(self):
        super().__init__(
            name="Aggressive-Baseline",
            personality="""You are an AGGRESSIVE baseline agent who maximizes market exposure.

You believe being in the market is better than missing rallies.
You typically go ABOVE the aggressive analyst recommendation.""",
            
            strategy="""DECISION RULES:
- Default to 50-70% positions
- Consider going to 80%+ on strong buy signals
- Rarely go below 40%
- You're comfortable with volatility

You exist to show what pure aggressive play looks like."""
        )


class BuyAndHoldAgent(SmartTradingAgent):
    """100% invested always - benchmark"""
    
    def __init__(self):
        super().__init__(
            name="Buy-and-Hold",
            personality="You are the market benchmark. Always 100% invested.",
            strategy="Always respond with position_percent: 100"
        )
    
    def make_decision(self, context: MarketContext) -> AgentDecision:
        """Always 100% - no LLM needed"""
        return AgentDecision(
            position_percent=100,
            position_dollars=context.portfolio_size,
            reasoning="Buy and hold: 100% market exposure always",
            confidence='HIGH'
        )


# ============================================================================
# TOURNAMENT
# ============================================================================

class SmartTournament:
    """Tournament with LLM-powered agents"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Paths
        self.project_root = self._find_project_root()
        self.data_path = self.project_root / "outputs" / "game_theory" / self.ticker
        self.output_path = self.project_root / "outputs" / "gtmats_result_smartmode" / self.ticker / f"smart_{self.timestamp}"
        
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
        
        print(f"Smart Tournament: {self.ticker}")
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
            print(f"Error loading sample {sample_num}: {e}")
            return None
    
    def run(self, portfolio_size: int = 100000, max_samples: Optional[int] = None):
        """Run tournament"""
        
        print(f"\n{'='*80}")
        print(f"SMART GAME THEORY TOURNAMENT - {self.ticker}")
        print(f"{'='*80}")
        print(f"Portfolio: ${portfolio_size:,}")
        print(f"Model: {MODEL}")
        
        samples = self._discover_samples(portfolio_size)
        if not samples:
            print(f"ERROR: No samples found")
            return
        
        samples_to_run = samples[:max_samples] if max_samples else samples
        print(f"Samples: {len(samples_to_run)}")
        print(f"{'='*80}\n")
        
        for i, sample_num in enumerate(samples_to_run):
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
            
            print(f"[{i+1}/{len(samples_to_run)}] Sample {sample_num} | {context.date} | Market: {context.daily_return*100:+.2f}%")
            
            sample_results = {
                'sample': sample_num,
                'date': context.date,
                'market_return': context.daily_return,
                'agents': {}
            }
            
            for agent in self.agents:
                decision = agent.make_decision(context)
                result = agent.update_after_market(context.daily_return, decision)
                
                print(f"  {agent.name:22} | {decision.position_percent:5.1f}% | "
                      f"Ret: {result['trade_return_pct']:+5.2f}% | "
                      f"Total: {result['total_return_pct']:+6.2f}% | "
                      f"Score: {agent.score:+3d}")
                
                sample_results['agents'][agent.name] = {
                    'position_percent': decision.position_percent,
                    'reasoning': decision.reasoning,
                    'trade_return_pct': result['trade_return_pct'],
                    'total_return_pct': result['total_return_pct'],
                    'score': agent.score
                }
                
                # Small delay to avoid rate limits
                if agent.name != 'Buy-and-Hold':
                    time.sleep(0.1)
            
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
        ax.set_title(f'{self.ticker} - Smart Game Theory Tournament', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'cumulative_returns.png', dpi=150)
        plt.close()
        
        # 2. Position Sizes Over Time
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for agent in self.agents:
            ax.plot(range(len(agent.positions)), agent.positions,
                   label=agent.name, color=colors.get(agent.name), linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Position Size (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Agent Position Sizes Over Time', fontsize=14, fontweight='bold')
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
                       label=agent.name, color=colors.get(agent.name), linewidth=2, marker='o', markersize=2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.ticker} - Agent Score Evolution', fontsize=14, fontweight='bold')
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
        
        for bar, val in zip(bars, returns):
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{val:+.2f}%', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Total Return (%)', fontsize=12)
        ax.set_title(f'{self.ticker} - Final Performance Ranking', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visualizations' / 'final_ranking.png', dpi=150)
        plt.close()
        
        # Save JSON results
        summary = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'model': MODEL,
            'total_samples': len(self.results),
            'rankings': [a.get_summary() for a in sorted(self.agents, key=lambda a: a.total_return, reverse=True)]
        }
        
        with open(self.output_path / 'tournament_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(self.output_path / 'full_results.json', 'w') as f:
            json.dump({'summary': summary, 'samples': self.results}, f, indent=2)
        
        print(f"Results saved to: {self.output_path}")
    
    def _print_results(self):
        """Print final results"""
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS - SMART TOURNAMENT")
        print(f"{'='*80}\n")
        
        sorted_agents = sorted(self.agents, key=lambda a: a.total_return, reverse=True)
        
        print(f"{'Rank':<5} {'Agent':<24} {'Return':>10} {'Score':>8} {'Avg Pos':>10}")
        print("-" * 60)
        
        for i, agent in enumerate(sorted_agents, 1):
            summary = agent.get_summary()
            print(f"{i:<5} {agent.name:<24} {summary['total_return_pct']:>+9.2f}% "
                  f"{summary['final_score']:>+7d} {summary['avg_position']:>9.1f}%")
        
        # Analysis
        buy_hold = next((a for a in self.agents if a.name == 'Buy-and-Hold'), None)
        if buy_hold:
            bh_return = (buy_hold.total_return - 1) * 100
            print(f"\n{'='*60}")
            print(f"BENCHMARK (Buy-and-Hold): {bh_return:+.2f}%")
            print(f"\nMarket Capture Rates:")
            for agent in sorted_agents:
                if agent.name != 'Buy-and-Hold':
                    capture = ((agent.total_return - 1) / (buy_hold.total_return - 1) * 100) if buy_hold.total_return != 1 else 0
                    print(f"  {agent.name}: {capture:.0f}% of market")
        
        print(f"\n{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Game Theory Tournament with LLM Agents")
    parser.add_argument('ticker', help='Stock ticker')
    parser.add_argument('--portfolio', type=int, default=100000)
    parser.add_argument('--samples', type=int, default=None, help='Limit samples (for testing)')
    
    args = parser.parse_args()
    
    try:
        tournament = SmartTournament(args.ticker)
        tournament.run(portfolio_size=args.portfolio, max_samples=args.samples)
        print("✅ Smart tournament completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()