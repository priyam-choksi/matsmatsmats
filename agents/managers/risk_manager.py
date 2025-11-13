"""
Risk Manager - Final Decision Maker
Synthesizes aggressive, neutral, conservative evaluations and makes final call

Usage: python risk_manager.py AAPL --synthesis-file ../../outputs/research_synthesis.json --portfolio-value 100000
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class RiskManager:
    def __init__(self, ticker: str, portfolio_value: float = 100000, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.portfolio_value = portfolio_value
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Enhanced system prompt - the final decision maker
        self.system_prompt = """You are the Risk Manager with FINAL DECISION AUTHORITY and VETO POWER.

**YOUR CRITICAL ROLE:**
You are the last line of defense. After hearing from Research Manager and three Risk Analysts (Aggressive, Neutral, Conservative), YOU make the final trading decision.

**YOUR RESPONSIBILITIES:**
1. **Evaluate All Perspectives:** Weigh aggressive, neutral, conservative viewpoints objectively
2. **Make Final Decision:** APPROVE, MODIFY, or REJECT the trade
3. **Set Exact Position Size:** Based on risk tolerance and portfolio constraints
4. **Define Risk Controls:** Specific stop loss, take profit, time limits
5. **Veto Bad Trades:** Use veto power to protect capital when needed

**DECISION FRAMEWORK:**

**APPROVE (Execute as recommended):**
- Multiple risk analysts agree (2+ out of 3)
- Research Manager recommendation is sound
- Risk/reward is favorable (>2:1)
- Position sizing is appropriate for risk
- No critical red flags
- Clear exit strategy defined

**MODIFY (Adjust before executing):**
- Good opportunity but sizing too aggressive
- Risk controls need tightening
- Entry timing should be adjusted
- Partial position makes more sense
- Conditions not quite met for full position

**REJECT (Veto the trade):**
- Excessive risk relative to reward
- Critical red flags present
- Violates portfolio risk limits
- No clear exit strategy
- Better opportunities elsewhere
- When in doubt, protect capital

**PORTFOLIO RISK LIMITS:**
- Max Single Position: 15% of portfolio
- Max Portfolio Risk: 20% total exposure
- Max Loss Per Trade: 2% of portfolio
- Min Reward/Risk: 2:1 required
- Max Correlated Exposure: 30%

**YOUR VETO POWER:**
You have ABSOLUTE authority to reject any trade that:
- Exceeds risk limits
- Lacks clear thesis
- Has inadequate risk controls
- Would over-concentrate portfolio
- Presents unclear or excessive risk

**CRITICAL OUTPUT REQUIREMENTS:**

## Risk Manager Decision

### Summary of Debate
**Aggressive:** [Their stance and key argument]
**Neutral:** [Their stance and key argument]  
**Conservative:** [Their stance and key argument]
**Research Manager:** [Their recommendation]

### Your Analysis
**Areas of Agreement:** [Where do 2+ analysts agree?]
**Key Conflicts:** [Where do they disagree?]
**Deciding Factors:** [What tipped your decision?]

### Final Decision
**VERDICT:** APPROVE / MODIFY / REJECT

**Position Size:** $X (Y% of portfolio)
**Rationale:** [Why this sizing?]

**Risk Controls:**
- Stop Loss: $X (-Y%)
- Take Profit 1: $X (+Y%)
- Take Profit 2: $X (+Y%)
- Time Limit: [Maximum holding period]

**Conditions for Approval:**
[List specific conditions that must be met]

**Monitoring Requirements:**
[What to watch, how often]

RISK DECISION: APPROVE/MODIFY/REJECT - Position: $X (Y%) - Confidence: High/Medium/Low

**Be decisive. Use your veto power when necessary. Protect capital first, seek returns second.**"""
        
        # Portfolio risk limits
        self.risk_limits = {
            'max_position_pct': 0.15,
            'max_portfolio_risk': 0.20,
            'max_loss_per_trade': 0.02,
            'min_reward_ratio': 2.0,
            'max_correlated_exposure': 0.30,
            'concentration_limit': 0.25
        }
        
        # Storage
        self.decision = {}
    
    def load_research_synthesis(self, synthesis_file: str) -> Optional[Dict]:
        """Load Research Manager's synthesis"""
        print(f"[RISK_MGR] Loading research synthesis...")
        
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                synthesis = json.load(f)
            print(f"[RISK_MGR] ✓ Research synthesis loaded")
            return synthesis
        except FileNotFoundError:
            print(f"[RISK_MGR] ⚠️  Synthesis not found: {synthesis_file}")
            return None
        except Exception as e:
            print(f"[RISK_MGR] ⚠️  Load error: {e}")
            return None
    
    def load_risk_evaluations(self) -> Dict[str, Dict]:
        """Load all 3 risk evaluations"""
        print(f"[RISK_MGR] Loading risk evaluations...")
        
        evaluations = {
            'aggressive': None,
            'neutral': None,
            'conservative': None
        }
        
        files = {
            'aggressive': f"../../outputs/aggressive_eval.json",
            'neutral': f"../../outputs/neutral_eval.json",
            'conservative': f"../../outputs/conservative_eval.json"
        }
        
        loaded_count = 0
        
        for risk_type, filepath in files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        evaluations[risk_type] = json.load(f)
                    loaded_count += 1
                    print(f"[RISK_MGR]   ✓ {risk_type}")
                except Exception as e:
                    print(f"[RISK_MGR]   ⚠️  {risk_type}: {e}")
            else:
                print(f"[RISK_MGR]   ⚠️  {risk_type}: not found")
        
        print(f"[RISK_MGR] ✓ Loaded {loaded_count}/3 evaluations")
        
        return evaluations
    
    def analyze_risk_consensus(self, evaluations: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze agreement/disagreement between risk analysts"""
        print(f"[RISK_MGR] Analyzing risk consensus...")
        
        consensus = {
            'stances': {},
            'position_sizes': {},
            'agreements': [],
            'conflicts': [],
            'avg_position_size': 0
        }
        
        # Collect stances
        for risk_type, eval_data in evaluations.items():
            if eval_data:
                consensus['stances'][risk_type] = eval_data.get('stance', 'UNKNOWN')
                consensus['position_sizes'][risk_type] = eval_data.get('position_size', 0)
        
        # Check for agreement
        stances = list(consensus['stances'].values())
        if stances:
            # Count BUY vs SELL vs HOLD/AVOID
            buy_count = sum(1 for s in stances if 'BUY' in s)
            sell_count = sum(1 for s in stances if 'SELL' in s or 'AVOID' in s)
            hold_count = sum(1 for s in stances if 'HOLD' in s)
            
            if buy_count >= 2:
                consensus['agreements'].append(f"{buy_count}/3 risk analysts favor buying")
            elif sell_count >= 2:
                consensus['agreements'].append(f"{sell_count}/3 risk analysts favor avoiding/selling")
            elif hold_count >= 2:
                consensus['agreements'].append(f"{hold_count}/3 risk analysts recommend holding")
            else:
                consensus['conflicts'].append("No clear consensus - all 3 disagree")
        
        # Check for extreme disagreement
        if buy_count > 0 and sell_count > 0:
            consensus['conflicts'].append("Direct conflict: Some say BUY, others say SELL/AVOID")
        
        # Calculate average position size
        sizes = [s for s in consensus['position_sizes'].values() if s > 0]
        consensus['avg_position_size'] = sum(sizes) / len(sizes) if sizes else 0
        
        print(f"[RISK_MGR] ✓ Avg position: {consensus['avg_position_size']*100:.1f}%")
        
        return consensus
    
    def calculate_final_position_size(self, synthesis: Dict, evaluations: Dict, consensus: Dict) -> Dict[str, Any]:
        """Calculate risk-adjusted position size"""
        print(f"[RISK_MGR] Calculating final position size...")
        
        # Start with average of risk analysts
        base_size = consensus['avg_position_size']
        
        # Adjust based on Research Manager confidence
        research_confidence = synthesis.get('conclusion', {}).get('confidence', 'LOW')
        if research_confidence == 'HIGH':
            confidence_multiplier = 1.2
        elif research_confidence == 'MEDIUM':
            confidence_multiplier = 1.0
        else:
            confidence_multiplier = 0.7
        
        # Adjust based on probabilities
        probs = synthesis.get('probabilities', {})
        bull_prob = probs.get('bull_case', 50)
        bear_prob = probs.get('bear_case', 30)
        
        if bear_prob > bull_prob * 1.5:
            prob_multiplier = 0.5  # Halve if bear case much stronger
        elif bull_prob > bear_prob * 1.5:
            prob_multiplier = 1.2  # Increase if bull case much stronger
        else:
            prob_multiplier = 1.0
        
        # Calculate final size
        final_size = base_size * confidence_multiplier * prob_multiplier
        
        # Apply hard limits
        final_size = min(final_size, self.risk_limits['max_position_pct'])
        final_size = max(final_size, 0)
        
        position_sizing = {
            'base_size': base_size,
            'confidence_multiplier': confidence_multiplier,
            'probability_multiplier': prob_multiplier,
            'final_size_pct': final_size,
            'final_size_dollars': final_size * self.portfolio_value,
            'reasoning': []
        }
        
        position_sizing['reasoning'].append(f"Base: {base_size*100:.1f}% (avg of risk analysts)")
        position_sizing['reasoning'].append(f"Confidence adj: ×{confidence_multiplier:.1f} ({research_confidence})")
        position_sizing['reasoning'].append(f"Probability adj: ×{prob_multiplier:.1f}")
        
        if final_size < base_size:
            position_sizing['reasoning'].append(f"Reduced to {final_size*100:.1f}% after adjustments")
        
        print(f"[RISK_MGR] ✓ Final position: {final_size*100:.1f}% (${final_size*self.portfolio_value:,.0f})")
        
        return position_sizing
    
    def set_risk_controls(self, position_sizing: Dict, synthesis: Dict) -> Dict[str, Any]:
        """Define specific risk controls"""
        print(f"[RISK_MGR] Setting risk controls...")
        
        position_pct = position_sizing['final_size_pct']
        position_dollars = position_sizing['final_size_dollars']
        
        controls = {
            'stop_loss': {},
            'take_profit': [],
            'time_limit': {},
            'monitoring': {},
            'exit_triggers': []
        }
        
        # Stop loss based on position size
        if position_pct > 0.10:  # Large position (>10%)
            stop_pct = 0.05  # Tight 5% stop
        elif position_pct > 0.05:  # Medium position
            stop_pct = 0.07  # 7% stop
        else:  # Small position
            stop_pct = 0.10  # 10% stop
        
        max_loss_dollars = position_dollars * stop_pct
        
        controls['stop_loss'] = {
            'percentage': stop_pct * 100,
            'max_dollar_loss': max_loss_dollars,
            'type': 'TRAILING' if position_pct > 0.10 else 'FIXED',
            'reasoning': f"Stop at -{stop_pct*100:.0f}% limits loss to ${max_loss_dollars:,.0f}"
        }
        
        # Take profit levels
        controls['take_profit'] = [
            {
                'level': 1,
                'target_pct': 10,
                'exit_pct': 33,
                'reasoning': 'Take 1/3 at +10% (lock in quick gain)'
            },
            {
                'level': 2,
                'target_pct': 20,
                'exit_pct': 33,
                'reasoning': 'Take 1/3 at +20% (solid profit)'
            },
            {
                'level': 3,
                'target_pct': 30,
                'exit_pct': 34,
                'reasoning': 'Final 1/3 at +30% (let winners run)'
            }
        ]
        
        # Time limits
        controls['time_limit'] = {
            'max_holding_period': '6 months',
            'review_frequency': 'Weekly' if position_pct > 0.10 else 'Bi-weekly',
            'mandatory_review': '30 days',
            'reasoning': 'Prevent indefinite holding, force re-evaluation'
        }
        
        # Monitoring
        controls['monitoring'] = {
            'daily_checks': ['Price vs stop/target', 'Volume', 'News alerts'],
            'weekly_checks': ['Technical levels', 'Sentiment shifts', 'Thesis validity'],
            'monthly_checks': ['Fundamental review', 'Position sizing', 'Risk/reward update']
        }
        
        # Exit triggers from bear thesis
        bear_thesis = synthesis.get('bear_thesis', {})
        for trigger in bear_thesis.get('downside_triggers', [])[:3]:
            controls['exit_triggers'].append({
                'condition': trigger.get('description', ''),
                'action': 'IMMEDIATE_REVIEW',
                'severity': 'HIGH' if trigger.get('impact', '') == 'HIGH' else 'MEDIUM'
            })
        
        print(f"[RISK_MGR] ✓ Controls set: -{stop_pct*100:.0f}% stop, +10/20/30% targets")
        
        return controls
    
    def make_final_decision(
        self,
        synthesis: Dict,
        evaluations: Dict,
        consensus: Dict,
        position_sizing: Dict,
        risk_controls: Dict
    ) -> Dict[str, Any]:
        """Make the final APPROVE/MODIFY/REJECT decision with veto power"""
        print(f"[RISK_MGR] Making final decision...")
        
        decision = {
            'verdict': 'APPROVE',
            'reasoning': [],
            'conditions': [],
            'final_position_dollars': position_sizing['final_size_dollars'],
            'final_position_pct': position_sizing['final_size_pct'],
            'confidence': 'MEDIUM'
        }
        
        # Check for veto conditions
        veto_triggered = False
        
        # Veto 1: Research Manager says SELL/AVOID
        research_rec = synthesis.get('conclusion', {}).get('recommendation', '')
        if research_rec in ['SELL', 'AVOID']:
            # But check if it's justified
            bear_prob = synthesis.get('probabilities', {}).get('bear_case', 0)
            
            if bear_prob > 60:
                decision['verdict'] = 'REJECT'
                decision['reasoning'].append(f"Research Manager: {research_rec} with {bear_prob:.0f}% bear probability")
                veto_triggered = True
            else:
                decision['verdict'] = 'MODIFY'
                decision['reasoning'].append(f"Research says {research_rec} but weak conviction - downgrading to HOLD")
                decision['final_position_dollars'] = position_sizing['final_size_dollars'] * 0.5
                decision['final_position_pct'] = position_sizing['final_size_pct'] * 0.5
        
        # Veto 2: Conservative found 2+ red flags
        conservative_eval = evaluations.get('conservative', {})
        red_flags = conservative_eval.get('red_flags', [])
        if len(red_flags) >= 2 and not veto_triggered:
            decision['verdict'] = 'REJECT'
            decision['reasoning'].append(f"Multiple red flags detected: {', '.join(red_flags[:2])}")
            veto_triggered = True
        
        # Veto 3: Position size exceeds limits
        if position_sizing['final_size_pct'] > self.risk_limits['max_position_pct'] and not veto_triggered:
            decision['verdict'] = 'MODIFY'
            decision['reasoning'].append(f"Position {position_sizing['final_size_pct']*100:.0f}% exceeds {self.risk_limits['max_position_pct']*100:.0f}% limit")
            decision['final_position_pct'] = self.risk_limits['max_position_pct']
            decision['final_position_dollars'] = self.risk_limits['max_position_pct'] * self.portfolio_value
        
        # Veto 4: All 3 risk analysts say AVOID
        if all(ev.get('stance', '') in ['AVOID', 'SELL/AVOID'] for ev in evaluations.values() if ev):
            decision['verdict'] = 'REJECT'
            decision['reasoning'].append("All risk analysts recommend avoiding - unanimous rejection")
            veto_triggered = True
        
        # Set final confidence
        if decision['verdict'] == 'APPROVE':
            # Check agreement level
            if len(consensus.get('agreements', [])) > 0:
                decision['confidence'] = 'MEDIUM'
            else:
                decision['confidence'] = 'LOW'
        elif decision['verdict'] == 'MODIFY':
            decision['confidence'] = 'LOW'
        else:  # REJECT
            decision['confidence'] = 'HIGH'  # High confidence in rejection
        
        # Add conditions for approved/modified trades
        if decision['verdict'] in ['APPROVE', 'MODIFY']:
            decision['conditions'] = [
                f"Strict stop loss at -{risk_controls['stop_loss']['percentage']:.0f}%",
                f"Maximum holding: {risk_controls['time_limit']['max_holding_period']}",
                f"Review frequency: {risk_controls['time_limit']['review_frequency']}",
                "Exit immediately if new red flags emerge"
            ]
        
        print(f"[RISK_MGR] ✓ Decision: {decision['verdict']}")
        
        return decision
    
    def synthesize_with_llm(
        self,
        synthesis: Dict,
        evaluations: Dict,
        consensus: Dict,
        position_sizing: Dict,
        risk_controls: Dict,
        decision: Dict
    ) -> str:
        """Generate final risk management decision with LLM"""
        
        if not self.client:
            print("[RISK_MGR] ⚠️  No API key - using fallback")
            return self._create_fallback_report(synthesis, evaluations, consensus, position_sizing, risk_controls, decision)
        
        try:
            print(f"[RISK_MGR] Generating final decision with {self.model}...")
            
            # Format debate history
            debate_summary = "## Risk Analyst Debate\n\n"
            
            for risk_type in ['aggressive', 'neutral', 'conservative']:
                eval_data = evaluations.get(risk_type, {})
                if eval_data:
                    debate_summary += f"### {risk_type.title()} Risk Analyst\n"
                    debate_summary += f"**Stance:** {eval_data.get('stance', 'N/A')}\n"
                    debate_summary += f"**Position Size:** {eval_data.get('position_size', 0)*100:.1f}%\n"
                    debate_summary += f"**Reasoning:** {eval_data.get('reasoning', 'N/A')}\n"
                    
                    if 'debate_argument' in eval_data:
                        debate_summary += f"**Argument:** {eval_data['debate_argument'][:300]}...\n"
                    
                    debate_summary += "\n"
            
            context = f"""# Final Risk Management Decision for {self.ticker}

## Research Manager's Recommendation
{synthesis.get('conclusion', {}).get('rationale', 'Not available')}

Recommendation: {synthesis.get('conclusion', {}).get('recommendation', 'N/A')}
Confidence: {synthesis.get('conclusion', {}).get('confidence', 'N/A')}

{debate_summary}

## Consensus Analysis
{json.dumps(consensus, indent=2)}

## Position Sizing Calculation
{json.dumps(position_sizing, indent=2)}

## Risk Controls
{json.dumps(risk_controls, indent=2)}

## Preliminary Decision
{json.dumps(decision, indent=2)}

---

**Your Task as Risk Manager:**

Make the FINAL decision with your veto power:
1. APPROVE, MODIFY, or REJECT this trade
2. Set exact position size and risk controls
3. List specific conditions for execution
4. Provide clear rationale for your decision
5. Use veto power if necessary to protect capital

Be decisive and specific. Reference the debate arguments in your reasoning."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.5,  # Moderate temp for balanced final decision
                max_tokens=3000
            )
            
            final_report = response.choices[0].message.content
            
            # Validate decision format
            if "RISK DECISION:" not in final_report:
                final_report += f"\n\nRISK DECISION: {decision['verdict']} - Position: ${decision['final_position_dollars']:,.0f} ({decision['final_position_pct']*100:.1f}%) - Confidence: {decision['confidence']}"
            
            print(f"[RISK_MGR] ✓ Final decision generated ({len(final_report)} chars)")
            
            return final_report
            
        except Exception as e:
            print(f"[RISK_MGR] ❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_report(synthesis, evaluations, consensus, position_sizing, risk_controls, decision)
    
    def _create_fallback_report(
        self,
        synthesis: Dict,
        evaluations: Dict,
        consensus: Dict,
        position_sizing: Dict,
        risk_controls: Dict,
        decision: Dict
    ) -> str:
        """Fallback report without LLM"""
        report = f"""
# RISK MANAGER FINAL DECISION: {self.ticker}
{'='*70}
**Portfolio Value:** ${self.portfolio_value:,.0f}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## VERDICT: {decision['verdict']}
{'='*70}

## Risk Analyst Summary
"""
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            eval_data = evaluations.get(risk_type, {})
            if eval_data:
                report += f"\n**{risk_type.title()}:** {eval_data.get('stance', 'N/A')} "
                report += f"({eval_data.get('position_size', 0)*100:.0f}% position)\n"
        
        report += f"""
## Position Sizing
**Final Decision:** {position_sizing['final_size_pct']*100:.1f}% (${position_sizing['final_size_dollars']:,.0f})

**Calculation:**
"""
        for reason in position_sizing['reasoning']:
            report += f"  - {reason}\n"
        
        report += f"""
## Risk Controls
**Stop Loss:** -{risk_controls['stop_loss']['percentage']:.0f}% ({risk_controls['stop_loss']['type']})
  Max Loss: ${risk_controls['stop_loss']['max_dollar_loss']:,.0f}

**Take Profit Levels:**
"""
        for tp in risk_controls['take_profit']:
            report += f"  - Level {tp['level']}: +{tp['target_pct']}% (exit {tp['exit_pct']}%)\n"
        
        report += f"""
**Time Limits:**
  - Max Holding: {risk_controls['time_limit']['max_holding_period']}
  - Review: {risk_controls['time_limit']['review_frequency']}

## Decision Rationale
"""
        for reason in decision['reasoning']:
            report += f"  - {reason}\n"
        
        if decision['conditions']:
            report += "\n**Conditions for Execution:**\n"
            for condition in decision['conditions']:
                report += f"  - {condition}\n"
        
        report += f"""
{'='*70}
RISK DECISION: {decision['verdict']} - Position: ${decision['final_position_dollars']:,.0f} ({decision['final_position_pct']*100:.1f}%) - Confidence: {decision['confidence']}
"""
        return report
    
    def execute(
        self,
        synthesis_file: Optional[str] = None,
        synthesis: Optional[Dict] = None
    ) -> tuple:
        """
        Main execution workflow
        Returns (report, decision_data)
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"RISK MANAGER FINAL DECISION: {self.ticker}")
        print(f"Portfolio: ${self.portfolio_value:,.0f}")
        print(f"{'='*70}\n")
        
        # Load research synthesis
        if not synthesis:
            if not synthesis_file:
                synthesis_file = "../../outputs/research_synthesis.json"
            synthesis = self.load_research_synthesis(synthesis_file)
        
        if not synthesis:
            print("[RISK_MGR] ❌ Cannot proceed without research synthesis")
            return "Error: No synthesis available", {}
        
        # Load risk evaluations
        evaluations = self.load_risk_evaluations()
        
        if not any(evaluations.values()):
            print("[RISK_MGR] ⚠️  No risk evaluations found - will use research synthesis only")
        
        # Analyze consensus
        consensus = self.analyze_risk_consensus(evaluations)
        
        # Calculate position size
        position_sizing = self.calculate_final_position_size(synthesis, evaluations, consensus)
        
        # Set risk controls
        risk_controls = self.set_risk_controls(position_sizing, synthesis)
        
        # Make final decision
        decision = self.make_final_decision(synthesis, evaluations, consensus, position_sizing, risk_controls)
        
        # Generate final report
        print(f"\n[RISK_MGR] Generating final decision report...\n")
        report = self.synthesize_with_llm(synthesis, evaluations, consensus, position_sizing, risk_controls, decision)
        
        # Compile decision data
        decision_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'verdict': decision['verdict'],
            'final_position_dollars': decision['final_position_dollars'],
            'final_position_pct': decision['final_position_pct'],
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'],
            'conditions': decision.get('conditions', []),
            'risk_controls': risk_controls,
            'position_sizing_breakdown': position_sizing,
            'risk_consensus': consensus,
            'research_recommendation': synthesis.get('conclusion', {}).get('recommendation', 'N/A')
        }
        
        elapsed = time.time() - start_time
        print(f"\n[RISK_MGR] ✓ Final decision complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, decision_data
    
    def save_decision(self, filepath: str, decision_data: Dict):
        """Save final decision"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(decision_data, f, indent=2)
            print(f"[RISK_MGR] ✓ Decision saved to {filepath}")
        except Exception as e:
            print(f"[RISK_MGR] ⚠️  Save error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Risk Manager - Final decision maker with veto power",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python risk_manager.py AAPL --synthesis-file ../../outputs/research_synthesis.json
  python risk_manager.py AAPL --portfolio-value 250000
  python risk_manager.py AAPL --save-decision ../../outputs/risk_decision.json
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("--synthesis-file", default="../../outputs/research_synthesis.json",
                       help="Research synthesis JSON")
    parser.add_argument("--portfolio-value", type=float, default=100000,
                       help="Portfolio value (default: $100,000)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--save-decision", help="Save decision JSON")
    
    args = parser.parse_args()
    
    try:
        manager = RiskManager(
            ticker=args.ticker,
            portfolio_value=args.portfolio_value,
            api_key=args.api_key,
            model=args.model
        )
        
        # Execute
        report, decision_data = manager.execute(synthesis_file=args.synthesis_file)
        
        print(report)
        
        # Save
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to {args.output}")
        
        if args.save_decision:
            manager.save_decision(args.save_decision, decision_data)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()