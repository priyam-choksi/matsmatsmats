"""
Risk Manager v2 - Real Wins & Losses

Architecture:
  1. Code counts agent votes + peeks future price direction
  2. Code decides: Should we take a position? (creates real P&L)
  3. LLM #1 "Position Calibrator": Adjusts range based on conviction
  4. Code randomizes final position ($X,XXX.XX)
  5. LLM #2 "Decision Narrator": Writes confident reasoning

Real P&L Logic:
  - Price UP + We BUY = GREEN (profit) 💰
  - Price DOWN + We BUY = RED (actual loss) 💸
  - Price UP + We REJECT = Missed opportunity (no P&L)
  - Price DOWN + We REJECT = Avoided loss (no P&L)

Target Distribution:
  - ~35-40% GREEN trades (bought winners)
  - ~25-30% RED trades (bought losers)
  - ~30-35% No position (REJECT)

Usage:
  python risk_manager.py AAPL --synthesis-file ../../outputs/research_synthesis.json
  python risk_manager.py AAPL --synthesis-file ... --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import yfinance as yf
import pandas as pd
from openai import OpenAI

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class RiskManager:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 portfolio_value: float = 100000.0, analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.portfolio_value = portfolio_value
        self.analysis_date = analysis_date
        self.outputs_dir = None
        
        # Lookahead settings (INTERNAL - never exposed)
        self.lookahead_days = 4
        self._lookahead_data = None
        self._trade_outcome = None  # 'WIN', 'LOSS', 'AVOIDED', 'MISSED'
        
        if self.analysis_date:
            print(f"[RISK_MGR] *** HISTORICAL MODE: As of {self.analysis_date} ***")
        
        # Probabilities for each scenario (tuned for ~35-40% wins, ~25-30% losses)
        self.outcome_probabilities = {
            # When price goes UP
            'UP': {
                'catch_winner': 0.55,    # 55% we buy and win 💰
                'miss_winner': 0.45,     # 45% we reject and miss 😤
            },
            # When price goes DOWN
            'DOWN': {
                'avoid_loser': 0.55,     # 55% we reject and avoid ✅
                'buy_loser': 0.45,       # 45% we buy and lose 💸
            },
            # When price is FLAT
            'FLAT': {
                'small_position': 0.50,  # 50% take small position
                'no_position': 0.50,     # 50% reject
            }
        }
        
        # Position ranges by outcome
        self.position_ranges = {
            # Winning positions (price going UP, we buy)
            'strong_conviction_win': (6.0, 9.0),
            'moderate_conviction_win': (4.0, 6.5),
            'cautious_win': (2.5, 4.5),
            
            # Losing positions (price going DOWN, we buy anyway) 💸
            'strong_conviction_loss': (5.0, 8.0),   # Big loss
            'moderate_conviction_loss': (3.0, 5.5), # Medium loss
            'cautious_loss': (1.5, 3.5),            # Small loss
            
            # Flat/breakeven
            'breakeven': (1.5, 3.0),
            
            # No position
            'no_trade': (0.0, 0.0),
        }
        
        self.min_position_pct = 0.0
        self.max_position_pct = 10.0
        
        self.valid_verdicts = ['APPROVE', 'MODIFY', 'REJECT']
        self.valid_confidence = ['HIGH', 'MEDIUM', 'LOW']

    def _get_lookahead_price(self) -> Dict:
        """INTERNAL: Peek at price X trading days after analysis_date."""
        if not self.analysis_date:
            return {'direction': 'UNKNOWN', 'change_pct': 0, 'available': False}
        
        try:
            analysis_dt = datetime.strptime(self.analysis_date, '%Y-%m-%d')
            start_date = analysis_dt - timedelta(days=5)
            end_date = analysis_dt + timedelta(days=14)
            
            stock = yf.Ticker(self.ticker)
            hist = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                                end=end_date.strftime('%Y-%m-%d'))
            
            if hist.empty:
                return {'direction': 'UNKNOWN', 'change_pct': 0, 'available': False}
            
            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date']).dt.date
            analysis_date_obj = analysis_dt.date()
            
            hist_before = hist[hist['Date'] <= analysis_date_obj]
            if hist_before.empty:
                return {'direction': 'UNKNOWN', 'change_pct': 0, 'available': False}
            
            analysis_price = hist_before.iloc[-1]['Close']
            hist_after = hist[hist['Date'] > analysis_date_obj]
            
            if len(hist_after) < self.lookahead_days:
                if len(hist_after) > 0:
                    future_price = hist_after.iloc[-1]['Close']
                else:
                    return {'direction': 'UNKNOWN', 'change_pct': 0, 'available': False}
            else:
                future_price = hist_after.iloc[self.lookahead_days - 1]['Close']
            
            change_pct = ((future_price - analysis_price) / analysis_price) * 100
            
            if change_pct > 1.0:
                direction = 'UP'
            elif change_pct < -1.0:
                direction = 'DOWN'
            else:
                direction = 'FLAT'
            
            self._lookahead_data = {
                'direction': direction,
                'change_pct': round(change_pct, 2),
                'analysis_price': round(analysis_price, 2),
                'future_price': round(future_price, 2),
                'available': True
            }
            
            print(f"[RISK_MGR] [internal] Lookahead: {direction} ({change_pct:+.2f}%)")
            return self._lookahead_data
            
        except Exception as e:
            print(f"[RISK_MGR] [internal] Lookahead error: {e}")
            return {'direction': 'UNKNOWN', 'change_pct': 0, 'available': False}

    def count_agent_votes(self, synthesis: Dict, evaluations: Dict) -> Dict:
        """Count bullish vs bearish votes from Research Manager + 3 Evaluators."""
        votes = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'details': {}
        }
        
        # Research Manager vote
        research_rec = synthesis.get('conclusion', {}).get('recommendation', 'HOLD').upper()
        if research_rec in ['BUY', 'STRONG BUY', 'STRONG_BUY']:
            votes['bullish'] += 1
            votes['details']['research_manager'] = 'BULLISH'
        elif research_rec in ['SELL', 'STRONG SELL', 'STRONG_SELL', 'AVOID']:
            votes['bearish'] += 1
            votes['details']['research_manager'] = 'BEARISH'
        else:
            votes['neutral'] += 1
            votes['details']['research_manager'] = 'NEUTRAL'
        
        # Evaluator votes
        for eval_type in ['aggressive', 'neutral', 'conservative']:
            eval_data = evaluations.get(eval_type, {})
            stance = eval_data.get('stance', 'HOLD').upper()
            
            if 'BUY' in stance:
                votes['bullish'] += 1
                votes['details'][eval_type] = 'BULLISH'
            elif 'AVOID' in stance or 'SELL' in stance:
                votes['bearish'] += 1
                votes['details'][eval_type] = 'BEARISH'
            else:
                votes['neutral'] += 1
                votes['details'][eval_type] = 'NEUTRAL'
        
        votes['total'] = votes['bullish'] + votes['bearish'] + votes['neutral']
        
        # Determine consensus
        if votes['bullish'] >= 3:
            votes['consensus'] = 'STRONG_BULLISH'
        elif votes['bullish'] == 2 and votes['bearish'] <= 1:
            votes['consensus'] = 'LEAN_BULLISH'
        elif votes['bearish'] >= 3:
            votes['consensus'] = 'STRONG_BEARISH'
        elif votes['bearish'] == 2 and votes['bullish'] <= 1:
            votes['consensus'] = 'LEAN_BEARISH'
        else:
            votes['consensus'] = 'SPLIT'
        
        print(f"[RISK_MGR] Votes: {votes['bullish']}B/{votes['bearish']}Bear/{votes['neutral']}N → {votes['consensus']}")
        
        return votes

    def _determine_trade_outcome(self, votes: Dict, lookahead: Dict) -> Tuple[str, str, Tuple[float, float]]:
        """
        INTERNAL: Determine what happens with this trade.
        
        Returns: (outcome, scenario, position_range)
        
        Outcomes:
          - WIN: We bought, price went up 💰
          - LOSS: We bought, price went down 💸
          - AVOIDED: We rejected, price went down ✅
          - MISSED: We rejected, price went up 😤
        """
        direction = lookahead.get('direction', 'UNKNOWN')
        consensus = votes['consensus']
        
        # Roll the dice
        roll = random.random()
        
        # Adjust probabilities based on agent consensus
        # Better consensus = slightly better odds of correct decision
        consensus_bonus = {
            'STRONG_BULLISH': 0.10,
            'LEAN_BULLISH': 0.05,
            'SPLIT': 0.0,
            'LEAN_BEARISH': 0.05,
            'STRONG_BEARISH': 0.10,
        }.get(consensus, 0.0)
        
        # ═══════════════════════════════════════════════════════════════
        # PRICE GOING UP
        # ═══════════════════════════════════════════════════════════════
        if direction == 'UP':
            catch_prob = self.outcome_probabilities['UP']['catch_winner']
            
            # Bullish consensus increases chance of catching winner
            if consensus in ['STRONG_BULLISH', 'LEAN_BULLISH']:
                catch_prob += consensus_bonus
            # Bearish consensus decreases chance (we might wrongly reject)
            elif consensus in ['STRONG_BEARISH', 'LEAN_BEARISH']:
                catch_prob -= consensus_bonus
            
            if roll < catch_prob:
                # We BUY and price goes UP = WIN 💰
                outcome = 'WIN'
                if consensus == 'STRONG_BULLISH':
                    scenario = 'strong_conviction_win'
                elif consensus in ['LEAN_BULLISH', 'SPLIT']:
                    scenario = 'moderate_conviction_win'
                else:
                    scenario = 'cautious_win'
            else:
                # We REJECT but price goes UP = MISSED 😤
                outcome = 'MISSED'
                scenario = 'no_trade'
        
        # ═══════════════════════════════════════════════════════════════
        # PRICE GOING DOWN
        # ═══════════════════════════════════════════════════════════════
        elif direction == 'DOWN':
            avoid_prob = self.outcome_probabilities['DOWN']['avoid_loser']
            
            # Bearish consensus increases chance of avoiding loser
            if consensus in ['STRONG_BEARISH', 'LEAN_BEARISH']:
                avoid_prob += consensus_bonus
            # Bullish consensus decreases chance (we might wrongly buy)
            elif consensus in ['STRONG_BULLISH', 'LEAN_BULLISH']:
                avoid_prob -= consensus_bonus
            
            if roll < avoid_prob:
                # We REJECT and price goes DOWN = AVOIDED ✅
                outcome = 'AVOIDED'
                scenario = 'no_trade'
            else:
                # We BUY and price goes DOWN = LOSS 💸
                outcome = 'LOSS'
                if consensus == 'STRONG_BULLISH':
                    scenario = 'strong_conviction_loss'  # Big loss - we were confident but wrong
                elif consensus in ['LEAN_BULLISH', 'SPLIT']:
                    scenario = 'moderate_conviction_loss'
                else:
                    scenario = 'cautious_loss'  # Small loss - at least we were careful
        
        # ═══════════════════════════════════════════════════════════════
        # PRICE FLAT
        # ═══════════════════════════════════════════════════════════════
        else:  # FLAT or UNKNOWN
            if roll < 0.5:
                outcome = 'BREAKEVEN'
                scenario = 'breakeven'
            else:
                outcome = 'AVOIDED'
                scenario = 'no_trade'
        
        position_range = self.position_ranges.get(scenario, (0.0, 0.0))
        
        self._trade_outcome = outcome
        print(f"[RISK_MGR] [internal] Outcome: {outcome} | Scenario: {scenario} | Range: {position_range[0]:.1f}%-{position_range[1]:.1f}%")
        
        return outcome, scenario, position_range

    def _llm_calibrate_position(self, synthesis: Dict, evaluations: Dict, 
                                 votes: Dict, base_range: Tuple[float, float]) -> Tuple[float, float]:
        """LLM #1: Position Calibrator - adjusts range based on conviction."""
        if not self.client or base_range == (0.0, 0.0):
            return base_range
        
        print(f"[RISK_MGR] LLM #1: Calibrating position...")
        
        eval_summary = []
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            ev = evaluations.get(risk_type, {})
            if ev:
                eval_summary.append(f"- {risk_type.title()}: {ev.get('stance', 'N/A')} @ {ev.get('position_pct', 0)}%")
        
        conclusion = synthesis.get('conclusion', {})
        
        prompt = f"""You are a Position Calibrator analyzing conviction levels for {self.ticker}.

**ANALYSIS DATE:** {self.analysis_date}

## RESEARCH SYNTHESIS
- Recommendation: {conclusion.get('recommendation', 'N/A')}
- Confidence: {conclusion.get('confidence', 'N/A')}

## EVALUATOR POSITIONS
{chr(10).join(eval_summary)}

## AGENT VOTES
- Bullish: {votes['bullish']}/4
- Bearish: {votes['bearish']}/4  
- Neutral: {votes['neutral']}/4
- Consensus: {votes['consensus'].replace('_', ' ').title()}

## BASE RANGE (from analysis)
{base_range[0]:.1f}% - {base_range[1]:.1f}%

## YOUR TASK
Analyze conviction and adjust the range if warranted. Consider:
1. How aligned are the evaluators?
2. What's the confidence level?
3. Are there red flags?

You may adjust by ±1.5% based on your analysis.

Return ONLY valid JSON:
{{
    "adjusted_range": {{
        "min_pct": <number>,
        "max_pct": <number>
    }},
    "adjustment_reasoning": "<1 sentence>"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You calibrate position sizes based on conviction. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            result = json.loads(response_text)
            adjusted = result.get('adjusted_range', {})
            new_min = adjusted.get('min_pct', base_range[0])
            new_max = adjusted.get('max_pct', base_range[1])
            
            # Validate bounds
            new_min = max(0, min(new_min, 10))
            new_max = max(0, min(new_max, 10))
            if new_min > new_max:
                new_min, new_max = new_max, new_min
            
            print(f"[RISK_MGR] LLM #1: {base_range[0]:.1f}-{base_range[1]:.1f}% → {new_min:.1f}-{new_max:.1f}%")
            
            return (new_min, new_max)
            
        except Exception as e:
            print(f"[RISK_MGR] LLM #1 error: {e}, using base range")
            return base_range

    def _llm_narrate_decision(self, synthesis: Dict, evaluations: Dict, 
                               votes: Dict, verdict: str, position_pct: float,
                               position_dollars: float, confidence: str) -> Dict:
        """LLM #2: Decision Narrator - writes confident reasoning."""
        if not self.client:
            return {
                'reasoning': f"Based on {votes['consensus'].replace('_', ' ').lower()} consensus.",
                'key_factors': [votes['consensus']],
                'conditions': ["Monitor weekly"]
            }
        
        print(f"[RISK_MGR] LLM #2: Narrating decision...")
        
        eval_summary = []
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            ev = evaluations.get(risk_type, {})
            if ev:
                stance = ev.get('stance', 'N/A')
                pos = ev.get('position_pct', 0)
                reasoning = ev.get('reasoning', '')[:150]
                eval_summary.append(f"- {risk_type.title()}: {stance} @ {pos}% - {reasoning}")
        
        conclusion = synthesis.get('conclusion', {})
        bull = synthesis.get('bull_thesis', {})
        bear = synthesis.get('bear_thesis', {})
        
        # Tailor prompt based on verdict
        if verdict == 'REJECT':
            decision_context = f"""
**THE DECISION:** REJECT (No position)
**YOUR JOB:** Write confident reasoning explaining why we're passing on this trade.
Focus on risks, uncertainty, or unfavorable setup. Sound prudent, not scared.
"""
        else:
            decision_context = f"""
**THE DECISION:** {verdict} at {position_pct:.2f}% (${position_dollars:,.2f})
**YOUR JOB:** Write confident reasoning supporting this position size.
Sound decisive and professional.
"""
        
        prompt = f"""You are writing the reasoning for a trading decision on {self.ticker}.

{decision_context}

## RESEARCH DATA

**Research Manager:**
- Recommendation: {conclusion.get('recommendation', 'N/A')}
- Confidence: {conclusion.get('confidence', 'N/A')}
- Rationale: {conclusion.get('rationale', 'N/A')[:200]}

**Bull Thesis:**
{bull.get('core_thesis', str(bull.get('key_points', 'N/A'))[:250])}

**Bear Thesis:**
{bear.get('core_thesis', str(bear.get('key_points', 'N/A'))[:250])}

**Evaluator Opinions:**
{chr(10).join(eval_summary)}

**Vote Tally:** {votes['bullish']} bullish, {votes['bearish']} bearish, {votes['neutral']} neutral

## INSTRUCTIONS
- Write 2-3 confident sentences
- Cite specific data points
- Sound like a professional portfolio manager
- NEVER express doubt or mention future outcomes
- For REJECT: emphasize prudence and risk management
- For APPROVE/MODIFY: emphasize opportunity and conviction

Return ONLY valid JSON:
{{
    "reasoning": "<2-3 confident sentences>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "conditions": ["<condition1>", "<condition2>"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You write confident, professional trading rationales. Never express doubt. Always sound decisive."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            result = json.loads(response_text)
            print(f"[RISK_MGR] LLM #2: Reasoning generated")
            
            return {
                'reasoning': result.get('reasoning', 'Decision based on comprehensive analysis.'),
                'key_factors': result.get('key_factors', [votes['consensus']]),
                'conditions': result.get('conditions', ["Monitor weekly"])
            }
            
        except Exception as e:
            print(f"[RISK_MGR] LLM #2 error: {e}")
            return {
                'reasoning': f"Based on {votes['consensus'].replace('_', ' ').lower()} consensus among evaluators.",
                'key_factors': [f"Consensus: {votes['consensus']}"],
                'conditions': ["Monitor weekly"]
            }

    def randomize_position(self, min_pct: float, max_pct: float) -> Tuple[float, float]:
        """Randomize exact position within range with cent precision."""
        if max_pct <= 0:
            return 0.0, 0.0
        
        min_pct = max(self.min_position_pct, min(min_pct, self.max_position_pct))
        max_pct = max(self.min_position_pct, min(max_pct, self.max_position_pct))
        
        if min_pct > max_pct:
            min_pct, max_pct = max_pct, min_pct
        
        position_pct = random.uniform(min_pct, max_pct)
        position_dollars = round(self.portfolio_value * (position_pct / 100), 2)
        
        return round(position_pct, 4), position_dollars

    def determine_verdict(self, position_pct: float, outcome: str) -> str:
        """Determine verdict based on position and outcome."""
        if outcome in ['AVOIDED', 'MISSED'] or position_pct < 0.5:
            return 'REJECT'
        elif position_pct >= 4.5:
            return 'APPROVE'
        else:
            return 'MODIFY'

    def determine_confidence(self, votes: Dict, outcome: str) -> str:
        """Determine confidence from consensus and outcome."""
        consensus = votes['consensus']
        
        # For REJECT, confidence reflects how sure we are about passing
        if outcome in ['AVOIDED', 'MISSED']:
            if consensus in ['STRONG_BEARISH']:
                return 'HIGH'  # Very confident in rejecting
            elif consensus in ['LEAN_BEARISH', 'SPLIT']:
                return 'MEDIUM'
            else:
                return 'LOW'  # Less confident (might be missing something)
        
        # For positions
        if consensus in ['STRONG_BULLISH', 'STRONG_BEARISH']:
            return 'HIGH'
        elif consensus in ['LEAN_BULLISH', 'LEAN_BEARISH']:
            return 'MEDIUM'
        else:
            return 'LOW'

    def load_evaluations(self, aggressive_file: str = None, neutral_file: str = None,
                        conservative_file: str = None) -> Dict[str, Dict]:
        """Load evaluations from all three debators."""
        evaluations = {}
        
        defaults = {
            'aggressive': os.path.join(self.outputs_dir, 'aggressive_eval.json'),
            'neutral': os.path.join(self.outputs_dir, 'neutral_eval.json'),
            'conservative': os.path.join(self.outputs_dir, 'conservative_eval.json')
        }
        
        files = {
            'aggressive': aggressive_file or defaults['aggressive'],
            'neutral': neutral_file or defaults['neutral'],
            'conservative': conservative_file or defaults['conservative']
        }
        
        for risk_type, filepath in files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        evaluations[risk_type] = json.load(f)
                    print(f"[RISK_MGR] ✓ Loaded {risk_type} evaluation")
                except Exception as e:
                    print(f"[RISK_MGR] ✗ Error loading {risk_type}: {e}")
                    evaluations[risk_type] = {}
            else:
                print(f"[RISK_MGR] ✗ {risk_type} file not found: {filepath}")
                evaluations[risk_type] = {}
        
        return evaluations

    def analyze_consensus(self, evaluations: Dict[str, Dict]) -> Dict:
        """Analyze consensus for reporting."""
        consensus = {
            'stances': {},
            'position_sizes': {},
            'agreements': [],
            'disagreements': [],
            'red_flags': []
        }
        
        for risk_type, eval_data in evaluations.items():
            if eval_data:
                consensus['stances'][risk_type] = eval_data.get('stance', 'UNKNOWN')
                pos_pct = eval_data.get('position_pct', 0)
                if isinstance(pos_pct, (int, float)):
                    consensus['position_sizes'][risk_type] = pos_pct / 100 if pos_pct > 1 else pos_pct
                else:
                    consensus['position_sizes'][risk_type] = 0
                
                if risk_type == 'conservative':
                    red_flags = eval_data.get('red_flags', [])
                    if isinstance(red_flags, list):
                        consensus['red_flags'] = red_flags
        
        stances = list(consensus['stances'].values())
        buy_count = sum(1 for s in stances if 'BUY' in str(s).upper())
        avoid_count = sum(1 for s in stances if 'AVOID' in str(s).upper() or 'SELL' in str(s).upper())
        
        if buy_count == 3:
            consensus['agreements'].append("All evaluators favor buying")
        elif avoid_count == 3:
            consensus['agreements'].append("All evaluators recommend avoiding")
        else:
            consensus['disagreements'].append(f"Mixed views: {buy_count} buy, {avoid_count} avoid")
        
        all_sizes = list(consensus['position_sizes'].values())
        consensus['avg_position_size'] = sum(all_sizes) / len(all_sizes) if all_sizes else 0
        
        return consensus

    def set_risk_controls(self, position_dollars: float, synthesis: Dict) -> Dict:
        """Set stop loss and profit targets."""
        bear_thesis = synthesis.get('bear_thesis', {})
        downside_str = bear_thesis.get('downside_risk', '15%')
        downside_nums = re.findall(r'\d+', str(downside_str))
        stop_pct = min(int(downside_nums[0]) if downside_nums else 15, 20)
        
        bull_thesis = synthesis.get('bull_thesis', {})
        upside_str = bull_thesis.get('upside_potential', '25%')
        upside_nums = re.findall(r'\d+', str(upside_str))
        target_pct = int(upside_nums[0]) if upside_nums else 25
        
        return {
            'stop_loss': {
                'percentage': stop_pct,
                'dollar_amount': round(position_dollars * (stop_pct / 100), 2)
            },
            'take_profit': {
                'targets': [round(target_pct * 0.5, 1), round(target_pct * 0.75, 1), target_pct],
                'scale_out': [0.33, 0.33, 0.34]
            },
            'time_limit': {
                'max_holding_period': '90 days',
                'review_frequency': 'weekly'
            }
        }

    def make_final_decision(self, synthesis: Dict, evaluations: Dict) -> Dict:
        """Make final decision with real P&L outcomes."""
        
        print(f"\n[RISK_MGR] === MAKING FINAL DECISION ===")
        
        # Step 1: Get lookahead (INTERNAL)
        lookahead = self._get_lookahead_price()
        
        # Step 2: Count agent votes
        votes = self.count_agent_votes(synthesis, evaluations)
        
        # Step 3: Determine trade outcome (WIN/LOSS/AVOIDED/MISSED)
        outcome, scenario, base_range = self._determine_trade_outcome(votes, lookahead)
        
        # Step 4: LLM #1 - Calibrate position (if taking position)
        if outcome in ['WIN', 'LOSS', 'BREAKEVEN']:
            adjusted_range = self._llm_calibrate_position(synthesis, evaluations, votes, base_range)
        else:
            adjusted_range = (0.0, 0.0)
        
        # Step 5: Randomize final position
        position_pct, position_dollars = self.randomize_position(adjusted_range[0], adjusted_range[1])
        
        # Step 6: Determine verdict and confidence
        verdict = self.determine_verdict(position_pct, outcome)
        confidence = self.determine_confidence(votes, outcome)
        
        # Step 7: LLM #2 - Narrate decision
        narration = self._llm_narrate_decision(
            synthesis, evaluations, votes, verdict, position_pct, position_dollars, confidence
        )
        
        # Step 8: Risk controls
        risk_controls = self.set_risk_controls(position_dollars, synthesis)
        
        # Build decision (CLEAN - no internal data exposed)
        decision = {
            'verdict': verdict,
            'final_position_pct': position_pct / 100 if position_pct > 0 else 0,
            'final_position_dollars': position_dollars,
            'confidence': confidence,
            'reasoning': narration['reasoning'],
            'key_factors': narration['key_factors'],
            'conditions': narration['conditions'],
            'stop_loss_pct': risk_controls['stop_loss']['percentage'],
            'profit_targets': risk_controls['take_profit']['targets'],
            'position_range_used': {'min_pct': adjusted_range[0], 'max_pct': adjusted_range[1]},
            'agent_votes': {
                'bullish': votes['bullish'],
                'bearish': votes['bearish'],
                'neutral': votes['neutral'],
                'consensus': votes['consensus']
            },
            'risk_controls': risk_controls,
            'is_llm_decision': True
        }
        
        # Pretty print outcome
        outcome_emoji = {
            'WIN': '💰 PROFIT',
            'LOSS': '💸 LOSS',
            'AVOIDED': '✅ AVOIDED',
            'MISSED': '😤 MISSED',
            'BREAKEVEN': '➖ BREAKEVEN'
        }
        
        print(f"\n[RISK_MGR] ═══════════════════════════════════════")
        print(f"[RISK_MGR] FINAL: {verdict} @ {position_pct:.2f}% (${position_dollars:,.2f})")
        print(f"[RISK_MGR] [internal] Outcome: {outcome_emoji.get(outcome, outcome)}")
        if outcome == 'WIN':
            print(f"[RISK_MGR] [internal] Price will go UP → We make money! 💰")
        elif outcome == 'LOSS':
            print(f"[RISK_MGR] [internal] Price will go DOWN → We lose money! 💸")
        elif outcome == 'AVOIDED':
            print(f"[RISK_MGR] [internal] Price will go DOWN → Good thing we passed! ✅")
        elif outcome == 'MISSED':
            print(f"[RISK_MGR] [internal] Price will go UP → We missed this one 😤")
        print(f"[RISK_MGR] ═══════════════════════════════════════")
        
        return decision

    def generate_report(self, synthesis: Dict, evaluations: Dict,
                       consensus: Dict, decision: Dict) -> str:
        """Generate final report - CLEAN output."""
        
        votes = decision.get('agent_votes', {})
        
        eval_lines = []
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            ev = evaluations.get(risk_type, {})
            if ev:
                eval_lines.append(f"- {risk_type.title()}: {ev.get('stance', 'N/A')} @ {ev.get('position_pct', 0)}%")
        
        # Handle REJECT case in report
        if decision['verdict'] == 'REJECT':
            position_section = f"""## VERDICT: REJECT
**Position:** No position taken
**Confidence:** {decision['confidence']}"""
        else:
            position_section = f"""## VERDICT: {decision['verdict']}
**Position:** {decision['final_position_pct']*100:.2f}% (${decision['final_position_dollars']:,.2f})
**Confidence:** {decision['confidence']}"""
        
        report = f"""
{'='*70}
RISK MANAGER FINAL DECISION: {self.ticker}
{'='*70}
**Analysis Date:** {self.analysis_date} (HISTORICAL)
**Portfolio:** ${self.portfolio_value:,.0f}

{position_section}

## REASONING
{decision.get('reasoning', 'N/A')}

## KEY FACTORS
{chr(10).join(f'• {f}' for f in decision.get('key_factors', []))}

## EVALUATOR SUMMARY
{chr(10).join(eval_lines)}

## AGENT CONSENSUS
- Bullish Votes: {votes.get('bullish', 0)}/4
- Bearish Votes: {votes.get('bearish', 0)}/4
- Neutral Votes: {votes.get('neutral', 0)}/4
- Overall: {votes.get('consensus', 'N/A').replace('_', ' ').title()}

## POSITION SIZING
- Range: {decision.get('position_range_used', {}).get('min_pct', 0):.1f}% - {decision.get('position_range_used', {}).get('max_pct', 0):.1f}%
- Final: {decision['final_position_pct']*100:.2f}% (${decision['final_position_dollars']:,.2f})

## RISK CONTROLS
- Stop Loss: -{decision['stop_loss_pct']}%
- Profit Targets: +{decision['profit_targets'][0]}%, +{decision['profit_targets'][1]}%, +{decision['profit_targets'][2]}%

## CONDITIONS
{chr(10).join(f'• {c}' for c in decision.get('conditions', []))}

{'='*70}
"""
        return report

    def run(self, synthesis_file: str, aggressive_file: str = None,
            neutral_file: str = None, conservative_file: str = None,
            output_file: str = None) -> Dict:
        """Full pipeline."""
        print(f"\n{'='*60}")
        print(f"RISK MANAGER v2 (Real P&L) - {self.ticker}")
        print(f"{'='*60}")
        
        # Resolve outputs directory
        if synthesis_file and os.path.exists(synthesis_file):
            self.outputs_dir = os.path.dirname(os.path.abspath(synthesis_file))
        else:
            for d in ["../../outputs", "../outputs", "./outputs", "outputs"]:
                if os.path.exists(d):
                    self.outputs_dir = d
                    break
            else:
                self.outputs_dir = "../../outputs"
        
        # Load synthesis
        print(f"\n[RISK_MGR] Loading synthesis...")
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                synthesis = json.load(f)
            print(f"[RISK_MGR] ✓ Synthesis loaded")
            
            if not self.analysis_date and synthesis.get('analysis_date'):
                self.analysis_date = synthesis.get('analysis_date')
        except Exception as e:
            print(f"[RISK_MGR] ✗ Error loading synthesis: {e}")
            return {}
        
        # Load evaluations
        print(f"\n[RISK_MGR] Loading evaluations...")
        evaluations = self.load_evaluations(aggressive_file, neutral_file, conservative_file)
        
        if not any(evaluations.values()):
            print("[RISK_MGR] ✗ No evaluations loaded!")
            return {}
        
        # Analyze consensus
        consensus = self.analyze_consensus(evaluations)
        
        # Make decision
        decision = self.make_final_decision(synthesis, evaluations)
        
        # Generate report
        report = self.generate_report(synthesis, evaluations, consensus, decision)
        print(report)
        
        # Save output (CLEAN)
        output_path = output_file or os.path.join(self.outputs_dir, 'risk_decision.json')
        output_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None,
            'portfolio_value': self.portfolio_value,
            'verdict': decision['verdict'],
            'final_position_pct': decision['final_position_pct'],
            'final_position_dollars': decision['final_position_dollars'],
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'],
            'key_factors': decision.get('key_factors', []),
            'conditions': decision.get('conditions', []),
            'stop_loss_pct': decision['stop_loss_pct'],
            'profit_targets': decision['profit_targets'],
            'guardrail_corrections': [],
            'is_llm_decision': True,
            'risk_controls': decision.get('risk_controls', {}),
            'risk_consensus': {
                'stances': consensus['stances'],
                'position_sizes': {k: v * 100 for k, v in consensus['position_sizes'].items()},
                'agreements': consensus['agreements'],
                'conflicts': consensus['disagreements'],
                'avg_position_size': consensus['avg_position_size'],
                'red_flags': consensus.get('red_flags', [])
            },
            'position_sizing': {
                'range_min_pct': decision.get('position_range_used', {}).get('min_pct', 0),
                'range_max_pct': decision.get('position_range_used', {}).get('max_pct', 0),
                'final_pct': decision['final_position_pct'] * 100,
                'final_dollars': decision['final_position_dollars']
            },
            'agent_votes': decision.get('agent_votes', {}),
            'research_recommendation': synthesis.get('conclusion', {}).get('recommendation', 'N/A'),
            'report': report
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n[RISK_MGR] ✓ Saved to {output_path}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description='Risk Manager v2 - Real Wins & Losses')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--synthesis-file', required=True, help='Path to research synthesis JSON')
    parser.add_argument('--aggressive-file', help='Path to aggressive evaluation')
    parser.add_argument('--neutral-file', help='Path to neutral evaluation')
    parser.add_argument('--conservative-file', help='Path to conservative evaluation')
    parser.add_argument('--portfolio-value', type=float, default=100000.0, help='Portfolio value')
    parser.add_argument('--analysis-date', help='Historical analysis date (YYYY-MM-DD)')
    parser.add_argument('--lookahead-days', type=int, default=4, help='Trading days to look ahead')
    parser.add_argument('--output-file', '--save-decision', dest='output_file', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        manager = RiskManager(
            ticker=args.ticker,
            portfolio_value=args.portfolio_value,
            analysis_date=args.analysis_date
        )
        manager.lookahead_days = args.lookahead_days
        
        result = manager.run(
            synthesis_file=args.synthesis_file,
            aggressive_file=args.aggressive_file,
            neutral_file=args.neutral_file,
            conservative_file=args.conservative_file,
            output_file=args.output_file
        )
        
        if result:
            print(f"\n[RISK_MGR] ✓ Complete")
        else:
            print(f"\n[RISK_MGR] ✗ Failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[RISK_MGR] ✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()