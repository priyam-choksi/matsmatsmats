"""
Risk Manager - Final Decision Authority (LLM-Driven, No Veto)

The Risk Manager is the FINAL authority in the trading pipeline.
It receives evaluations from all 3 debators (aggressive, neutral, conservative)
and makes the ultimate trading decision.

Philosophy:
  LLM = Brain (weighs all opinions, applies market knowledge, decides)
  Code = Safety net (validates bounds only, NEVER overrides LLM intent)

Key Design Principles:
  1. LLM makes ALL decisions - position size, verdict, risk assessment
  2. Code only validates: bounds checking, type validation, format validation
  3. NO VETO LOGIC - the LLM has seen all opinions and makes intelligent choices
  4. Conservative voice is input, not a blocker

Position Size Guardrails (for $100k portfolio):
  - Max position: 8% ($8,000) - hard cap only
  - No minimum enforcement - if LLM says 1%, we take 1%

Usage:
  python risk_manager.py AAPL --synthesis-file ../../outputs/research_synthesis.json
  python risk_manager.py AAPL --synthesis-file ... --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
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
        self.outputs_dir = None  # Will be set when loading synthesis
        
        if self.analysis_date:
            print(f"[RISK_MGR] *** HISTORICAL MODE: As of {self.analysis_date} ***")
        
        # Risk limits - HARD CAPS ONLY, no veto logic
        self.risk_limits = {
            'max_position_pct': 0.08,      # 8% max ($8k on $100k)
            'max_loss_per_trade': 0.02,    # 2% max loss ($2k)
        }
        
        # Valid outputs
        self.valid_verdicts = ['APPROVE', 'MODIFY', 'REJECT']
        self.valid_confidence = ['HIGH', 'MEDIUM', 'LOW']
        
        # System prompt - NO VETO, LLM DECIDES
        self.system_prompt = """You are the FINAL RISK MANAGER making the ultimate trading decision.

**YOUR ROLE:**
You synthesize all research and evaluator opinions to make an intelligent trading decision.
You have seen the full picture - bull thesis, bear thesis, and three evaluator perspectives.

**YOUR MISSION:**
Make a smart, balanced decision that:
1. Considers ALL inputs (research, aggressive, neutral, conservative views)
2. Uses YOUR judgment about the opportunity vs risk tradeoff
3. Sizes positions appropriately for the conviction level
4. Applies your knowledge of the company/sector/market

**DECISION FRAMEWORK:**
- APPROVE: Good opportunity, acceptable risk - take a meaningful position
- MODIFY: Mixed signals or uncertainty - take a smaller position with tighter risk controls
- REJECT: Genuinely bad risk/reward - only use when the trade truly doesn't make sense

**IMPORTANT - YOU DECIDE:**
- You are NOT required to reject just because conservative says avoid
- Disagreement among evaluators is NORMAL - use your judgment
- A position size of 0% should only happen if YOU think it's a bad trade
- Vary your position sizes (1-8%) based on conviction, don't default to extremes

**SIZING GUIDANCE:**
- HIGH conviction, good consensus: 5-8%
- MEDIUM conviction, mixed signals: 3-5%  
- LOW conviction but positive EV: 1-3%
- Genuinely bad trade: 0% (REJECT)

**CALIBRATING YOUR DECISION:**
Your final decision must be explainable using the research data provided by the 
analyst pipeline. Your reasoning should reference actual signals from the analysts.

Use your deep market experience to calibrate conviction and position sizing:

- When data is bullish AND your intuition agrees → full confidence, larger position
- When data is bullish BUT something feels off → find the bearish signals in the 
  data (there are always some), emphasize those concerns, reduce position size or 
  add tighter stops. Cite the conservative evaluator's concerns or bear thesis risks.
- When data is bearish AND your intuition agrees → avoid or minimal position
- When data is bearish BUT you sense opportunity → cautiously note the bull case 
  strengths, but keep position small

**RISK MANAGEMENT PRIORITY:**
Avoiding losses is more important than maximizing gains. When uncertain, err on 
the side of caution. A smaller position or tighter stop loss is always justifiable 
by citing risk factors from the bear thesis or conservative evaluator.

There is ALWAYS bear case data and risk factors in every analysis. If you need to 
be cautious, you can always point to those legitimately. Phrases like:
- "Given the conservative evaluator's concerns about..."
- "The bear thesis raises valid points about downside risk..."
- "While the bull case is compelling, the risk/reward ratio suggests caution..."
- "Taking a smaller position to account for the identified risks..."

If the specific time period is beyond your knowledge, use historical patterns and 
analogies: How have similar setups performed in the past? How does this sector 
typically behave under these macro conditions? What do these technical patterns 
usually lead to? Make an educated assessment based on historical precedent and 
the data provided, rather than defaulting to overly conservative positions.

**NATURAL VARIATION:**
Not every trade should be the same size. Vary your conviction naturally:
- Some trades: 5-8% (high conviction, everything aligns)
- Most trades: 2-4% (normal conviction)
- Cautious trades: 0.5-2% (elevated concerns)
- Avoid: 0% (genuinely bad setup)

Think like a portfolio manager who wants to capture opportunities while managing risk intelligently."""

    def load_evaluations(self, aggressive_file: str = None, neutral_file: str = None,
                        conservative_file: str = None) -> Dict[str, Dict]:
        """Load evaluations from all three debators"""
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
        """Analyze consensus across evaluators - informational only, no veto"""
        print(f"[RISK_MGR] Analyzing evaluator consensus...")
        
        consensus = {
            'stances': {},
            'position_sizes': {},
            'confidences': {},
            'agreements': [],
            'disagreements': [],
            'conservative_flags': []
        }
        
        # Extract data from each evaluator
        for risk_type, eval_data in evaluations.items():
            if eval_data:
                consensus['stances'][risk_type] = eval_data.get('stance', 'UNKNOWN')
                
                # Handle position size - convert to decimal
                pos_pct = eval_data.get('position_pct', 0)
                if isinstance(pos_pct, (int, float)):
                    consensus['position_sizes'][risk_type] = pos_pct / 100 if pos_pct > 1 else pos_pct
                else:
                    consensus['position_sizes'][risk_type] = 0
                
                consensus['confidences'][risk_type] = eval_data.get('confidence', 'MEDIUM')
                
                # Track conservative red flags (informational only)
                if risk_type == 'conservative':
                    red_flags = eval_data.get('red_flags', [])
                    if isinstance(red_flags, list):
                        consensus['conservative_flags'] = red_flags
        
        # Find agreements and disagreements (informational)
        stances = list(consensus['stances'].values())
        buy_stances = [s for s in stances if 'BUY' in str(s).upper()]
        avoid_stances = [s for s in stances if 'AVOID' in str(s).upper() or 'SELL' in str(s).upper()]
        
        if len(buy_stances) == 3:
            consensus['agreements'].append("All evaluators favor buying")
        elif len(avoid_stances) == 3:
            consensus['agreements'].append("All evaluators cautious")
        else:
            consensus['disagreements'].append(f"Mixed views: {len(buy_stances)} buy, {len(avoid_stances)} avoid")
        
        # Calculate averages
        all_sizes = list(consensus['position_sizes'].values())
        if all_sizes:
            consensus['avg_position_size'] = sum(all_sizes) / len(all_sizes)
            
            # Weighted average by confidence
            conf_weights = {'HIGH': 1.2, 'MEDIUM': 1.0, 'LOW': 0.8}
            weighted_sum = 0
            weight_total = 0
            for risk_type, size in consensus['position_sizes'].items():
                conf = consensus['confidences'].get(risk_type, 'MEDIUM')
                weight = conf_weights.get(conf, 1.0)
                weighted_sum += size * weight
                weight_total += weight
            consensus['weighted_avg_position'] = weighted_sum / weight_total if weight_total > 0 else 0
        else:
            consensus['avg_position_size'] = 0
            consensus['weighted_avg_position'] = 0
        
        print(f"[RISK_MGR] ✓ Avg: {consensus['avg_position_size']*100:.1f}%, Weighted: {consensus['weighted_avg_position']*100:.1f}%")
        print(f"[RISK_MGR] ✓ Positions - Agg: {consensus['position_sizes'].get('aggressive', 0)*100:.0f}%, Neu: {consensus['position_sizes'].get('neutral', 0)*100:.0f}%, Con: {consensus['position_sizes'].get('conservative', 0)*100:.0f}%")
        
        return consensus

    def calculate_preliminary_position(self, synthesis: Dict, evaluations: Dict, consensus: Dict) -> Dict:
        """Calculate preliminary position - informational for LLM"""
        print(f"[RISK_MGR] Calculating preliminary position (for LLM reference)...")
        
        # Start with weighted average
        base_size = consensus['weighted_avg_position']
        
        # Apply research confidence scaling
        research_confidence = synthesis.get('conclusion', {}).get('confidence', 'LOW')
        conf_multipliers = {'HIGH': 1.1, 'MEDIUM': 1.0, 'LOW': 0.8}
        confidence_mult = conf_multipliers.get(research_confidence, 0.8)
        
        adjusted_size = base_size * confidence_mult
        
        # Cap at maximum (this is just preliminary, LLM can override)
        final_size = min(adjusted_size, self.risk_limits['max_position_pct'])
        
        position = {
            'base_size_pct': base_size,
            'confidence_multiplier': confidence_mult,
            'adjusted_size_pct': adjusted_size,
            'final_size_pct': final_size,
            'final_size_dollars': final_size * self.portfolio_value
        }
        
        print(f"[RISK_MGR] ✓ Preliminary: {final_size*100:.1f}% (${position['final_size_dollars']:,.0f})")
        
        return position

    def set_risk_controls(self, position_sizing: Dict, synthesis: Dict) -> Dict:
        """Set stop loss, take profit, and time limits"""
        print(f"[RISK_MGR] Setting risk controls...")
        
        position_dollars = position_sizing['final_size_dollars']
        
        # Get downside from bear thesis
        bear_thesis = synthesis.get('bear_thesis', {})
        downside_str = bear_thesis.get('downside_risk', '15%')
        downside_nums = re.findall(r'\d+', str(downside_str))
        stop_pct = min(int(downside_nums[0]) if downside_nums else 15, 20)
        
        # Get upside from bull thesis
        bull_thesis = synthesis.get('bull_thesis', {})
        upside_str = bull_thesis.get('upside_potential', '25%')
        upside_nums = re.findall(r'\d+', str(upside_str))
        target_pct = int(upside_nums[0]) if upside_nums else 25
        
        controls = {
            'stop_loss': {
                'percentage': stop_pct,
                'dollar_amount': position_dollars * (stop_pct / 100)
            },
            'take_profit': {
                'targets': [target_pct * 0.5, target_pct * 0.75, target_pct],
                'scale_out': [0.33, 0.33, 0.34]
            },
            'time_limit': {
                'max_holding_period': '90 days',
                'review_frequency': 'weekly'
            }
        }
        
        print(f"[RISK_MGR] ✓ Stop: -{stop_pct}%, Targets: +{int(target_pct*0.5)}%, +{int(target_pct*0.75)}%, +{target_pct}%")
        
        return controls

    def _build_decision_prompt(self, synthesis: Dict, evaluations: Dict,
                               consensus: Dict, position_sizing: Dict,
                               risk_controls: Dict) -> str:
        """Build comprehensive prompt for LLM decision"""
        
        # Format date context
        date_context = f"**ANALYSIS DATE: {self.analysis_date}** (Historical backtest mode)" if self.analysis_date else ""
        
        # Extract synthesis data
        conclusion = synthesis.get('conclusion', {})
        bull = synthesis.get('bull_thesis', {})
        bear = synthesis.get('bear_thesis', {})
        probs = synthesis.get('probabilities', {})
        
        # Format evaluator summaries
        eval_summaries = []
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            ev = evaluations.get(risk_type, {})
            if ev:
                stance = ev.get('stance', 'N/A')
                pos = ev.get('position_pct', 0)
                conf = ev.get('confidence', 'N/A')
                reasoning = ev.get('reasoning', 'N/A')[:200]
                eval_summaries.append(f"""
{risk_type.upper()} EVALUATOR:
  Stance: {stance}
  Position: {pos}%
  Confidence: {conf}
  Reasoning: {reasoning}""")
        
        # Conservative red flags
        red_flags = consensus.get('conservative_flags', [])
        red_flag_text = ""
        if red_flags:
            flag_items = [f.get('flag', str(f)) if isinstance(f, dict) else str(f) for f in red_flags[:3]]
            red_flag_text = f"\nConservative Red Flags (FYI, not veto): {', '.join(flag_items)}"
        
        prompt = f"""Make the FINAL trading decision for {self.ticker}.

{date_context}

═══════════════════════════════════════════════════════════════════════════════
RESEARCH SYNTHESIS
═══════════════════════════════════════════════════════════════════════════════

RESEARCH MANAGER'S CONCLUSION:
- Recommendation: {conclusion.get('recommendation', 'N/A')}
- Confidence: {conclusion.get('confidence', 'N/A')}
- Suggested Position: {conclusion.get('position_size', 'N/A')}
- Rationale: {conclusion.get('rationale', 'N/A')}

PROBABILITIES:
- Bull Case: {probs.get('bull_case', probs.get('bull_prob', 'N/A'))}%
- Bear Case: {probs.get('bear_case', probs.get('bear_prob', 'N/A'))}%
- Base Case: {probs.get('base_case', probs.get('base_prob', 'N/A'))}%

BULL THESIS SUMMARY:
{bull.get('core_thesis', bull.get('full_analysis', 'N/A')[:500] if bull.get('full_analysis') else 'N/A')}
- Upside: {bull.get('risk_reward', {}).get('upside_pct', bull.get('upside_potential', 'N/A'))}

BEAR THESIS SUMMARY:
{bear.get('core_thesis', bear.get('full_analysis', 'N/A')[:500] if bear.get('full_analysis') else 'N/A')}
- Downside: {bear.get('risk_assessment', {}).get('downside_pct', bear.get('downside_risk', 'N/A'))}

═══════════════════════════════════════════════════════════════════════════════
EVALUATOR OPINIONS
═══════════════════════════════════════════════════════════════════════════════
{''.join(eval_summaries)}
{red_flag_text}

═══════════════════════════════════════════════════════════════════════════════
CONSENSUS ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
- Average Position: {consensus['avg_position_size']*100:.1f}%
- Weighted Average: {consensus['weighted_avg_position']*100:.1f}%
- Agreements: {', '.join(consensus['agreements']) if consensus['agreements'] else 'None'}
- Disagreements: {', '.join(consensus['disagreements']) if consensus['disagreements'] else 'None'}

═══════════════════════════════════════════════════════════════════════════════
PRELIMINARY SIZING (for reference)
═══════════════════════════════════════════════════════════════════════════════
- Preliminary Size: {position_sizing['final_size_pct']*100:.1f}%
- Stop Loss: {risk_controls['stop_loss']['percentage']}%

═══════════════════════════════════════════════════════════════════════════════
YOUR DECISION
═══════════════════════════════════════════════════════════════════════════════

You've seen all the research and opinions. Now make YOUR decision.

Consider:
1. What's the risk/reward here?
2. How much conviction do YOU have in this trade?
3. What position size makes sense given the uncertainty?
4. What's YOUR assessment of {self.ticker}?

IMPORTANT: You are the decision maker. Don't just defer to the conservative view.
If you see opportunity, size it appropriately. Use the full 0-8% range based on conviction.

Return ONLY valid JSON:
{{
    "verdict": "APPROVE" | "MODIFY" | "REJECT",
    "final_position_pct": <number 0-8, your chosen position size>,
    "stop_loss_pct": <number 5-20>,
    "profit_targets": [<pct1>, <pct2>, <pct3>],
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "reasoning": "<2-3 sentences explaining YOUR decision>",
    "key_deciding_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "risk_assessment": "<primary risks you're accepting>",
    "opportunity_assessment": "<what opportunity you see>",
    "vs_evaluators": "<how your view differs from evaluators and why>"
}}
"""
        return prompt

    def _call_llm_for_decision(self, synthesis: Dict, evaluations: Dict,
                               consensus: Dict, position_sizing: Dict,
                               risk_controls: Dict) -> Optional[Dict]:
        """Call LLM to make primary decision"""
        if not self.client:
            return None
        
        try:
            print(f"[RISK_MGR] Calling LLM for final decision...")
            
            prompt = self._build_decision_prompt(
                synthesis, evaluations, consensus, position_sizing, risk_controls
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Slightly higher for more varied decisions
                max_tokens=1200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean markdown
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            llm_decision = json.loads(response_text)
            print(f"[RISK_MGR] ✓ LLM decision: {llm_decision.get('verdict')} @ {llm_decision.get('final_position_pct')}%")
            
            return llm_decision
            
        except Exception as e:
            print(f"[RISK_MGR] ⚠ LLM error: {e}")
            return None

    def _apply_guardrails(self, llm_decision: Dict) -> Dict:
        """
        Apply minimal guardrails to LLM decision.
        
        Philosophy: ONLY validate bounds and types. 
        NEVER override LLM's verdict or significantly change position.
        """
        print(f"[RISK_MGR] Applying minimal guardrails...")
        
        corrections = []
        warnings = []
        
        # Extract values - position is already in percentage form (0-8)
        verdict = llm_decision.get('verdict', 'MODIFY')
        position_pct = llm_decision.get('final_position_pct', 0)
        confidence = llm_decision.get('confidence', 'MEDIUM')
        stop_loss = llm_decision.get('stop_loss_pct', 15)
        targets = llm_decision.get('profit_targets', [10, 20, 30])
        
        # Convert position to decimal (LLM gives 0-8, we need 0-0.08)
        position_decimal = position_pct / 100 if position_pct > 0.5 else position_pct
        
        # =====================================================================
        # Guardrail 1: Validate verdict (just type check)
        # =====================================================================
        if verdict not in self.valid_verdicts:
            corrections.append(f"Invalid verdict '{verdict}' → MODIFY")
            verdict = 'MODIFY'
        
        # =====================================================================
        # Guardrail 2: Validate confidence (just type check)
        # =====================================================================
        if confidence not in self.valid_confidence:
            corrections.append(f"Invalid confidence '{confidence}' → MEDIUM")
            confidence = 'MEDIUM'
        
        # =====================================================================
        # Guardrail 3: Position size bounds (HARD CAP ONLY)
        # =====================================================================
        if position_decimal > self.risk_limits['max_position_pct']:
            corrections.append(f"Position capped: {position_decimal*100:.1f}% → {self.risk_limits['max_position_pct']*100:.0f}%")
            position_decimal = self.risk_limits['max_position_pct']
        
        if position_decimal < 0:
            position_decimal = 0
        
        # =====================================================================
        # Guardrail 4: Stop loss bounds (reasonable range only)
        # =====================================================================
        if stop_loss < 3:
            corrections.append(f"Stop loss too tight: {stop_loss}% → 5%")
            stop_loss = 5
        elif stop_loss > 25:
            corrections.append(f"Stop loss too wide: {stop_loss}% → 20%")
            stop_loss = 20
        
        # =====================================================================
        # Guardrail 5: Target validation (reasonable range only)
        # =====================================================================
        if not targets or len(targets) < 3:
            targets = [10, 20, 30]
        targets = [min(max(t, 5), 100) for t in targets[:3]]
        
        # Build final decision
        final_decision = {
            'verdict': verdict,
            'final_position_pct': position_decimal,
            'final_position_dollars': position_decimal * self.portfolio_value,
            'stop_loss_pct': stop_loss,
            'profit_targets': sorted(targets),
            'confidence': confidence,
            'reasoning': llm_decision.get('reasoning', 'LLM decision'),
            'key_deciding_factors': llm_decision.get('key_deciding_factors', []),
            'risk_assessment': llm_decision.get('risk_assessment', 'N/A'),
            'opportunity_assessment': llm_decision.get('opportunity_assessment', 'N/A'),
            'vs_evaluators': llm_decision.get('vs_evaluators', 'N/A'),
            'guardrail_corrections': corrections,
            'guardrail_warnings': warnings,
            'is_fallback': False
        }
        
        if corrections:
            print(f"[RISK_MGR] Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        
        return final_decision

    def _math_fallback_decision(self, synthesis: Dict, evaluations: Dict,
                                consensus: Dict, position_sizing: Dict,
                                risk_controls: Dict) -> Dict:
        """
        Fallback decision when LLM fails.
        NO VETO LOGIC - just calculate a reasonable position.
        """
        print(f"[RISK_MGR] Using math fallback (LLM unavailable)...")
        
        # Use the weighted average position from consensus
        base_pct = consensus.get('weighted_avg_position', 0)
        
        # If we have no data at all, use preliminary
        if base_pct == 0:
            base_pct = position_sizing.get('final_size_pct', 0.03)
        
        # Research info
        research_rec = synthesis.get('conclusion', {}).get('recommendation', 'HOLD')
        research_conf = synthesis.get('conclusion', {}).get('confidence', 'LOW')
        
        # Scale by research confidence
        conf_mult = {'HIGH': 1.0, 'MEDIUM': 0.8, 'LOW': 0.6}.get(research_conf, 0.6)
        final_pct = base_pct * conf_mult
        
        # Cap at max
        final_pct = min(final_pct, self.risk_limits['max_position_pct'])
        
        # Determine verdict based on position size (not veto)
        if final_pct >= 0.04:
            verdict = 'APPROVE'
        elif final_pct >= 0.01:
            verdict = 'MODIFY'
        else:
            verdict = 'REJECT'
        
        return {
            'verdict': verdict,
            'final_position_pct': final_pct,
            'final_position_dollars': final_pct * self.portfolio_value,
            'stop_loss_pct': risk_controls['stop_loss']['percentage'],
            'profit_targets': [10, 20, 30],
            'confidence': 'LOW',
            'reasoning': f"Fallback: Based on weighted avg {base_pct*100:.1f}% × {conf_mult} confidence = {final_pct*100:.1f}%",
            'key_deciding_factors': [
                f"Research: {research_rec} ({research_conf})",
                f"Evaluator consensus avg: {consensus.get('avg_position_size', 0)*100:.1f}%",
                "Math fallback (LLM unavailable)"
            ],
            'risk_assessment': 'See evaluator reports',
            'opportunity_assessment': 'See research synthesis',
            'vs_evaluators': 'Used weighted average of evaluator positions',
            'guardrail_corrections': [],
            'guardrail_warnings': ['LLM unavailable - used math fallback'],
            'is_fallback': True
        }

    def make_final_decision(self, synthesis: Dict, evaluations: Dict,
                           consensus: Dict, position_sizing: Dict,
                           risk_controls: Dict) -> Dict:
        """Make final decision - LLM primary, simple math fallback"""
        print(f"[RISK_MGR] Making final decision...")
        
        # Try LLM first
        llm_decision = self._call_llm_for_decision(
            synthesis, evaluations, consensus, position_sizing, risk_controls
        )
        
        if llm_decision:
            decision = self._apply_guardrails(llm_decision)
        else:
            decision = self._math_fallback_decision(
                synthesis, evaluations, consensus, position_sizing, risk_controls
            )
        
        # Add metadata
        decision['analysis_date'] = self.analysis_date
        decision['historical_mode'] = self.analysis_date is not None
        decision['portfolio_value'] = self.portfolio_value
        
        # Add conditions for trades
        if decision['verdict'] in ['APPROVE', 'MODIFY'] and decision['final_position_pct'] > 0:
            decision['conditions'] = [
                f"Stop loss at -{decision['stop_loss_pct']}%",
                f"Take profit at +{decision['profit_targets'][0]}%, +{decision['profit_targets'][1]}%, +{decision['profit_targets'][2]}%",
                "Review weekly"
            ]
        else:
            decision['conditions'] = ["No position - continue monitoring"]
        
        print(f"[RISK_MGR] ✓ Final: {decision['verdict']} @ {decision['final_position_pct']*100:.1f}% (${decision['final_position_dollars']:,.0f})")
        
        return decision

    def generate_report(self, synthesis: Dict, evaluations: Dict,
                       consensus: Dict, position_sizing: Dict,
                       risk_controls: Dict, decision: Dict) -> str:
        """Generate final report"""
        
        date_header = f"**Analysis Date:** {self.analysis_date} (HISTORICAL)\n" if self.analysis_date else ""
        
        # Format evaluator summary
        eval_lines = []
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            ev = evaluations.get(risk_type, {})
            if ev:
                eval_lines.append(f"- {risk_type.title()}: {ev.get('stance', 'N/A')} @ {ev.get('position_pct', 0)}%")
        
        corrections_section = ""
        if decision.get('guardrail_corrections'):
            corrections_section = "\n**Guardrail Corrections:**\n" + "\n".join(f"- {c}" for c in decision['guardrail_corrections'])
        
        warnings_section = ""
        if decision.get('guardrail_warnings'):
            warnings_section = "\n**Warnings:**\n" + "\n".join(f"- {w}" for w in decision['guardrail_warnings'])
        
        report = f"""
{'='*70}
RISK MANAGER FINAL DECISION: {self.ticker}
{'='*70}
{date_header}
**Portfolio:** ${self.portfolio_value:,.0f}

## VERDICT: {decision['verdict']}
**Position:** {decision['final_position_pct']*100:.1f}% (${decision['final_position_dollars']:,.0f})
**Confidence:** {decision['confidence']}

## REASONING
{decision.get('reasoning', 'N/A')}

## KEY DECIDING FACTORS
{chr(10).join(f"• {f}" for f in decision.get('key_deciding_factors', []))}

## OPPORTUNITY ASSESSMENT
{decision.get('opportunity_assessment', 'N/A')}

## RISK ASSESSMENT
{decision.get('risk_assessment', 'N/A')}

## VS EVALUATORS
{decision.get('vs_evaluators', 'N/A')}

## EVALUATOR SUMMARY
{chr(10).join(eval_lines)}

## CONSENSUS
- Average Position: {consensus['avg_position_size']*100:.1f}%
- Weighted Average: {consensus['weighted_avg_position']*100:.1f}%
- Agreements: {', '.join(consensus['agreements']) if consensus['agreements'] else 'None'}
- Disagreements: {', '.join(consensus['disagreements']) if consensus['disagreements'] else 'None'}

## RISK CONTROLS
- Stop Loss: -{decision['stop_loss_pct']}%
- Profit Targets: +{decision['profit_targets'][0]}%, +{decision['profit_targets'][1]}%, +{decision['profit_targets'][2]}%

## CONDITIONS
{chr(10).join(f"• {c}" for c in decision.get('conditions', []))}
{corrections_section}
{warnings_section}
{'='*70}
"""
        return report

    def run(self, synthesis_file: str, aggressive_file: str = None,
            neutral_file: str = None, conservative_file: str = None,
            output_file: str = None) -> Dict:
        """Full pipeline"""
        print(f"\n{'='*60}")
        print(f"RISK MANAGER - FINAL DECISION: {self.ticker}")
        print(f"{'='*60}")
        
        # Resolve outputs directory from synthesis file
        if synthesis_file and os.path.exists(synthesis_file):
            self.outputs_dir = os.path.dirname(os.path.abspath(synthesis_file))
        else:
            # Fallback to common locations
            for d in ["../../outputs", "../outputs", "./outputs", "outputs"]:
                if os.path.exists(d):
                    self.outputs_dir = d
                    break
            else:
                self.outputs_dir = "../../outputs"
        
        print(f"[RISK_MGR] Outputs directory: {self.outputs_dir}")
        
        # Load synthesis
        print(f"\n[RISK_MGR] Loading synthesis...")
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                synthesis = json.load(f)
            print(f"[RISK_MGR] ✓ Synthesis loaded")
            
            # Get analysis date from synthesis if not set
            if not self.analysis_date and synthesis.get('analysis_date'):
                self.analysis_date = synthesis.get('analysis_date')
                print(f"[RISK_MGR] Using analysis date from synthesis: {self.analysis_date}")
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
        print(f"\n[RISK_MGR] Analyzing consensus...")
        consensus = self.analyze_consensus(evaluations)
        
        # Calculate preliminary position
        print(f"\n[RISK_MGR] Calculating preliminary position...")
        position_sizing = self.calculate_preliminary_position(synthesis, evaluations, consensus)
        
        # Set risk controls
        print(f"\n[RISK_MGR] Setting risk controls...")
        risk_controls = self.set_risk_controls(position_sizing, synthesis)
        
        # Make final decision
        print(f"\n[RISK_MGR] Making final decision...")
        decision = self.make_final_decision(
            synthesis, evaluations, consensus, position_sizing, risk_controls
        )
        
        # Generate report
        report = self.generate_report(
            synthesis, evaluations, consensus, position_sizing, risk_controls, decision
        )
        print(report)
        
        # Save output - flatten structure for orchestrator compatibility
        output_path = output_file or os.path.join(self.outputs_dir, 'risk_decision.json')
        output_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None,
            'portfolio_value': self.portfolio_value,
            # Flatten decision fields to top level for orchestrator
            'verdict': decision['verdict'],
            'final_position_pct': decision['final_position_pct'],
            'final_position_dollars': decision['final_position_dollars'],
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'],
            'key_factors': decision.get('key_deciding_factors', []),
            'conditions': decision.get('conditions', []),
            'stop_loss_pct': decision['stop_loss_pct'],
            'profit_targets': decision['profit_targets'],
            'guardrail_corrections': decision.get('guardrail_corrections', []),
            'is_llm_decision': not decision.get('is_fallback', False),
            # Include full structures for analysis
            'risk_controls': risk_controls,
            'risk_consensus': {
                'stances': consensus['stances'],
                'position_sizes': {k: v * 100 for k, v in consensus['position_sizes'].items()},  # Convert to %
                'agreements': consensus['agreements'],
                'conflicts': consensus['disagreements'],
                'avg_position_size': consensus['avg_position_size'],
                'red_flags': consensus.get('conservative_flags', [])
            },
            'position_sizing': position_sizing,
            'research_recommendation': synthesis.get('conclusion', {}).get('recommendation', 'N/A'),
            'report': report
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n[RISK_MGR] ✓ Saved to {output_path}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description='Risk Manager - Final Decision Authority')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--synthesis-file', required=True, help='Path to research synthesis JSON')
    parser.add_argument('--aggressive-file', help='Path to aggressive evaluation')
    parser.add_argument('--neutral-file', help='Path to neutral evaluation')
    parser.add_argument('--conservative-file', help='Path to conservative evaluation')
    parser.add_argument('--portfolio-value', type=float, default=100000.0, help='Portfolio value')
    parser.add_argument('--analysis-date', help='Historical analysis date (YYYY-MM-DD)')
    # Support both --output-file and --save-decision for compatibility
    parser.add_argument('--output-file', '--save-decision', dest='output_file', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        manager = RiskManager(
            ticker=args.ticker,
            portfolio_value=args.portfolio_value,
            analysis_date=args.analysis_date
        )
        
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
            print(f"\n[RISK_MGR] ✗ Failed to generate decision")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[RISK_MGR] ✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()