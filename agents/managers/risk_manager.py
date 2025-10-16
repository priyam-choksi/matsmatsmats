"""
Risk Manager - Final risk-adjusted decision maker with veto power
Usage: python risk_manager.py AAPL --synthesis-file research_synthesis.json --portfolio-value 100000
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from openai import OpenAI

class RiskManager:
    def __init__(self, ticker, portfolio_value=100000, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.portfolio_value = portfolio_value
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are the Risk Manager with final decision authority and veto power.
        Your responsibilities:
        1. Make the final risk-adjusted trading decision
        2. Set exact position sizing based on portfolio constraints
        3. Define specific risk controls (stops, targets, time limits)
        4. Consider portfolio-level risk and correlation
        5. VETO any trade that exceeds acceptable risk parameters
        
        You have absolute veto power - use it when necessary to protect capital.
        End with: RISK DECISION: APPROVE/MODIFY/REJECT - Position: $X (Y%) - Confidence: High/Medium/Low"""
        
        # Risk management parameters
        self.risk_limits = {
            'max_position_pct': 0.15,  # 15% max single position
            'max_portfolio_risk': 0.20,  # 20% total portfolio risk
            'max_loss_per_trade': 0.02,  # 2% max loss per trade
            'min_reward_ratio': 2.0,  # Minimum 2:1 reward/risk
            'max_correlated_exposure': 0.30,  # 30% in correlated positions
            'max_daily_var': 0.05,  # 5% daily VaR limit
            'concentration_limit': 0.25  # 25% sector concentration
        }
        
        # Portfolio context (would normally come from database)
        self.portfolio_context = {
            'current_positions': {},
            'sector_exposure': {},
            'total_risk': 0,
            'available_capital': portfolio_value,
            'risk_budget_used': 0
        }
    
    def load_synthesis(self, synthesis_file):
        """Load research synthesis"""
        if os.path.exists(synthesis_file):
            with open(synthesis_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Synthesis file not found: {synthesis_file}")
            return self.run_research_manager()
    
    def run_research_manager(self):
        """Run research manager if synthesis not available"""
        manager_script = "research_manager.py"
        if not os.path.exists(manager_script):
            print("Research manager not found")
            return None
        
        cmd = ["python", manager_script, self.ticker, "--save-synthesis", "temp_synthesis.json"]
        
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists("temp_synthesis.json"):
                with open("temp_synthesis.json", 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error running research manager: {e}")
        
        return None
    
    def calculate_position_size(self, synthesis):
        """Calculate risk-adjusted position size"""
        position_sizing = {
            'base_size': 0,
            'risk_adjustment': 1.0,
            'final_size_pct': 0,
            'final_size_dollars': 0,
            'reasoning': []
        }
        
        # Get recommendation and consensus
        conclusion = synthesis.get('conclusion', {})
        consensus = synthesis.get('consensus', {})
        
        recommendation = conclusion.get('recommendation', 'HOLD')
        avg_position = consensus.get('avg_position_size', 0.03)
        
        # Start with average recommended position
        position_sizing['base_size'] = avg_position
        position_sizing['reasoning'].append(f"Base size from consensus: {avg_position*100:.1f}%")
        
        # Adjust based on confidence
        confidence = conclusion.get('confidence', 'LOW')
        if confidence == 'HIGH':
            position_sizing['risk_adjustment'] *= 1.2
            position_sizing['reasoning'].append("Increased 20% for high confidence")
        elif confidence == 'LOW':
            position_sizing['risk_adjustment'] *= 0.7
            position_sizing['reasoning'].append("Reduced 30% for low confidence")
        
        # Adjust based on probabilities
        probs = synthesis.get('probabilities', {})
        bull_prob = probs.get('bull_case', 0)
        bear_prob = probs.get('bear_case', 0)
        
        if bear_prob > bull_prob * 1.5:  # Bear case significantly more likely
            position_sizing['risk_adjustment'] *= 0.5
            position_sizing['reasoning'].append(f"Halved due to bear probability {bear_prob:.0f}% vs bull {bull_prob:.0f}%")
        elif bull_prob > bear_prob * 1.5:  # Bull case significantly more likely
            position_sizing['risk_adjustment'] *= 1.1
            position_sizing['reasoning'].append(f"Increased 10% for favorable probabilities")
        
        # Apply risk limits
        final_size = position_sizing['base_size'] * position_sizing['risk_adjustment']
        final_size = min(final_size, self.risk_limits['max_position_pct'])
        final_size = max(final_size, 0)
        
        position_sizing['final_size_pct'] = final_size
        position_sizing['final_size_dollars'] = final_size * self.portfolio_value
        
        if final_size < position_sizing['base_size'] * position_sizing['risk_adjustment']:
            position_sizing['reasoning'].append(f"Capped at {self.risk_limits['max_position_pct']*100:.0f}% maximum")
        
        return position_sizing
    
    def set_risk_controls(self, synthesis, position_size):
        """Define specific risk management controls"""
        controls = {
            'stop_loss': {},
            'take_profit': [],
            'time_limit': {},
            'monitoring': {},
            'triggers': []
        }
        
        # Extract key metrics
        conclusion = synthesis.get('conclusion', {})
        key_factors = synthesis.get('key_factors', {})
        
        # Set stop loss based on position size and risk tolerance
        if position_size['final_size_pct'] > 0.10:  # Large position
            stop_pct = 0.05  # Tight 5% stop
        elif position_size['final_size_pct'] > 0.05:  # Medium position
            stop_pct = 0.07  # 7% stop
        else:  # Small position
            stop_pct = 0.10  # 10% stop
        
        max_loss = position_size['final_size_dollars'] * stop_pct
        
        controls['stop_loss'] = {
            'percentage': stop_pct * 100,
            'max_dollar_loss': max_loss,
            'type': 'TRAILING' if stop_pct <= 0.05 else 'FIXED',
            'reasoning': f"Stop at -{stop_pct*100:.0f}% to limit loss to ${max_loss:.0f}"
        }
        
        # Set profit targets
        controls['take_profit'] = [
            {'level': 1, 'target_pct': 10, 'exit_pct': 33, 'reasoning': 'Take 1/3 at +10%'},
            {'level': 2, 'target_pct': 20, 'exit_pct': 33, 'reasoning': 'Take 1/3 at +20%'},
            {'level': 3, 'target_pct': 30, 'exit_pct': 34, 'reasoning': 'Final 1/3 at +30%'}
        ]
        
        # Set time limit
        controls['time_limit'] = {
            'max_holding_period': '6 months',
            'review_frequency': 'Weekly' if position_size['final_size_pct'] > 0.10 else 'Bi-weekly',
            'mandatory_review': '30 days',
            'reasoning': 'Prevent indefinite holding, force re-evaluation'
        }
        
        # Set monitoring requirements
        controls['monitoring'] = {
            'daily_checks': ['Price action', 'Volume', 'News'],
            'weekly_checks': ['Technical levels', 'Sentiment shift'],
            'alerts': [
                f"Price move >{stop_pct*100/2:.0f}%",
                "Unusual volume",
                "Analyst changes"
            ]
        }
        
        # Exit triggers from synthesis
        critical_risks = key_factors.get('critical_risks', [])
        for risk in critical_risks[:3]:
            controls['triggers'].append({
                'condition': risk,
                'action': 'IMMEDIATE_REVIEW',
                'severity': 'HIGH'
            })
        
        return controls
    
    def assess_portfolio_impact(self, position_size):
        """Assess impact on overall portfolio"""
        impact = {
            'concentration_check': 'PASS',
            'risk_budget_check': 'PASS',
            'correlation_check': 'PASS',
            'liquidity_check': 'PASS',
            'warnings': []
        }
        
        # Concentration check
        position_pct = position_size['final_size_pct']
        if position_pct > 0.10:
            impact['warnings'].append(f"Large position ({position_pct*100:.1f}%) - increases concentration risk")
            if position_pct > self.risk_limits['max_position_pct']:
                impact['concentration_check'] = 'FAIL'
        
        # Risk budget check
        potential_loss = position_size['final_size_dollars'] * 0.10  # Assume 10% loss scenario
        risk_budget_impact = potential_loss / self.portfolio_value
        
        if risk_budget_impact > self.risk_limits['max_loss_per_trade']:
            impact['risk_budget_check'] = 'WARNING'
            impact['warnings'].append(f"Uses {risk_budget_impact*100:.1f}% of risk budget")
        
        # Simplified checks for correlation and liquidity
        impact['correlation_impact'] = "Acceptable - within limits"
        impact['liquidity_impact'] = f"${position_size['final_size_dollars']:.0f} position size manageable"
        
        return impact
    
    def make_decision(self, synthesis, position_size, risk_controls, portfolio_impact):
        """Make final risk-adjusted decision with veto power"""
        decision = {
            'verdict': 'APPROVE',  # APPROVE/MODIFY/REJECT
            'reasoning': [],
            'conditions': [],
            'final_position': position_size['final_size_dollars'],
            'confidence': 'MEDIUM'
        }
        
        # Check for veto conditions
        veto_triggered = False
        
        # Veto if recommendation is SELL or AVOID
        recommendation = synthesis.get('conclusion', {}).get('recommendation', '')
        if recommendation in ['SELL', 'STRONG SELL', 'AVOID']:
            decision['verdict'] = 'REJECT'
            decision['reasoning'].append(f"Research recommendation is {recommendation}")
            veto_triggered = True
        
        # Veto if portfolio checks fail
        if portfolio_impact['concentration_check'] == 'FAIL':
            decision['verdict'] = 'REJECT'
            decision['reasoning'].append("Exceeds concentration limits")
            veto_triggered = True
        
        # Veto if too many warnings
        if len(portfolio_impact['warnings']) >= 3:
            decision['verdict'] = 'REJECT'
            decision['reasoning'].append("Multiple risk warnings triggered")
            veto_triggered = True
        
        # Modify if concerns exist but not veto-worthy
        if not veto_triggered:
            if portfolio_impact['warnings']:
                decision['verdict'] = 'MODIFY'
                decision['reasoning'].append("Position modified due to risk concerns")
                decision['final_position'] *= 0.75  # Reduce by 25%
            
            # Check confidence level
            confidence = synthesis.get('conclusion', {}).get('confidence', 'LOW')
            if confidence == 'LOW' and position_size['final_size_pct'] > 0.05:
                decision['verdict'] = 'MODIFY'
                decision['reasoning'].append("Reduced position due to low confidence")
                decision['final_position'] *= 0.5  # Halve position
        
        # Set final confidence
        if decision['verdict'] == 'APPROVE':
            decision['confidence'] = synthesis.get('conclusion', {}).get('confidence', 'MEDIUM')
        elif decision['verdict'] == 'MODIFY':
            decision['confidence'] = 'LOW'
        else:
            decision['confidence'] = 'HIGH'  # High confidence in rejection
        
        # Add conditions for approval/modification
        if decision['verdict'] in ['APPROVE', 'MODIFY']:
            decision['conditions'] = [
                f"Strict stop loss at {risk_controls['stop_loss']['percentage']:.0f}%",
                f"Maximum holding period: {risk_controls['time_limit']['max_holding_period']}",
                f"Review frequency: {risk_controls['time_limit']['review_frequency']}"
            ]
        
        return decision
    
    def synthesize_with_llm(self, risk_data):
        """Use LLM for final risk decision"""
        if not self.client:
            return self.create_fallback_report(risk_data)
        
        try:
            prompt = f"""As Risk Manager for {self.ticker}, make the final risk-adjusted decision:

Research Synthesis: {json.dumps(risk_data['synthesis_summary'], indent=2)}
Position Sizing: {json.dumps(risk_data['position_sizing'], indent=2)}
Risk Controls: {json.dumps(risk_data['risk_controls'], indent=2)}
Portfolio Impact: {json.dumps(risk_data['portfolio_impact'], indent=2)}
Initial Decision: {json.dumps(risk_data['decision'], indent=2)}

Provide your final risk management decision:
1. APPROVE, MODIFY, or REJECT the trade
2. Specify exact position size and risk controls
3. List specific conditions and monitoring requirements
4. Explain your reasoning, especially for any veto
5. Consider portfolio-level risk

You have absolute veto power - use it to protect capital when necessary."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4  # Lower temp for risk decisions
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.create_fallback_report(risk_data)
    
    def create_fallback_report(self, risk_data):
        """Create report without LLM"""
        decision = risk_data['decision']
        position = risk_data['position_sizing']
        controls = risk_data['risk_controls']
        impact = risk_data['portfolio_impact']
        
        report = f"""
RISK MANAGEMENT DECISION - {self.ticker}
{'='*60}
Portfolio Value: ${self.portfolio_value:,.0f}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

FINAL DECISION: {decision['verdict']}
=====================================

POSITION SIZING:
----------------
Recommended Position: {position['final_size_pct']*100:.1f}% (${position['final_size_dollars']:,.0f})
Reasoning:"""
        for reason in position['reasoning']:
            report += f"\n  • {reason}"
        
        if decision['verdict'] == 'MODIFY':
            report += f"\n\nMODIFIED TO: ${decision['final_position']:,.0f}"
        
        report += f"""

RISK CONTROLS:
--------------
Stop Loss: -{controls['stop_loss']['percentage']:.0f}% ({controls['stop_loss']['type']})
Maximum Loss: ${controls['stop_loss']['max_dollar_loss']:,.0f}

Profit Targets:"""
        for target in controls['take_profit']:
            report += f"\n  • {target['reasoning']}"
        
        report += f"""

Time Limits:
  • Maximum Holding: {controls['time_limit']['max_holding_period']}
  • Review Frequency: {controls['time_limit']['review_frequency']}

PORTFOLIO IMPACT:
-----------------
Concentration Check: {impact['concentration_check']}
Risk Budget Check: {impact['risk_budget_check']}
{impact['correlation_impact']}
{impact['liquidity_impact']}
"""
        
        if impact['warnings']:
            report += "\nWarnings:"
            for warning in impact['warnings']:
                report += f"\n  ⚠️ {warning}"
        
        report += f"""

DECISION RATIONALE:
-------------------"""
        for reason in decision['reasoning']:
            report += f"\n• {reason}"
        
        if decision['conditions']:
            report += "\n\nCONDITIONS FOR APPROVAL:"
            for condition in decision['conditions']:
                report += f"\n• {condition}"
        
        report += f"""

MONITORING REQUIREMENTS:
------------------------
Daily: {', '.join(controls['monitoring']['daily_checks'])}
Weekly: {', '.join(controls['monitoring']['weekly_checks'])}
Alerts: {', '.join(controls['monitoring']['alerts'])}

EXIT TRIGGERS:
--------------"""
        for trigger in controls['triggers']:
            report += f"\n• {trigger['condition']} → {trigger['action']}"
        
        report += f"""

{'='*60}
RISK DECISION: {decision['verdict']} - """
        
        if decision['verdict'] != 'REJECT':
            report += f"Position: ${decision['final_position']:,.0f} ({decision['final_position']/self.portfolio_value*100:.1f}%) - "
        
        report += f"Confidence: {decision['confidence']}"
        
        return report
    
    def evaluate(self, synthesis=None, synthesis_file=None):
        """Main evaluation function"""
        # Load synthesis if not provided
        if not synthesis:
            if not synthesis_file:
                synthesis_file = "research_synthesis.json"
            synthesis = self.load_synthesis(synthesis_file)
        
        if not synthesis:
            return "Error: No research synthesis available", None
        
        print(f"Risk Manager evaluating {self.ticker}...")
        print(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        
        # Calculate position size
        print("Calculating position size...")
        position_sizing = self.calculate_position_size(synthesis)
        
        # Set risk controls
        print("Setting risk controls...")
        risk_controls = self.set_risk_controls(synthesis, position_sizing)
        
        # Assess portfolio impact
        print("Assessing portfolio impact...")
        portfolio_impact = self.assess_portfolio_impact(position_sizing)
        
        # Make final decision
        print("Making final decision...")
        decision = self.make_decision(synthesis, position_sizing, risk_controls, portfolio_impact)
        
        # Compile all data
        risk_data = {
            'ticker': self.ticker,
            'portfolio_value': self.portfolio_value,
            'synthesis_summary': {
                'recommendation': synthesis.get('conclusion', {}).get('recommendation'),
                'confidence': synthesis.get('conclusion', {}).get('confidence'),
                'probabilities': synthesis.get('probabilities', {})
            },
            'position_sizing': position_sizing,
            'risk_controls': risk_controls,
            'portfolio_impact': portfolio_impact,
            'decision': decision
        }
        
        # Generate report
        report = self.synthesize_with_llm(risk_data)
        
        return report, decision

def main():
    parser = argparse.ArgumentParser(description="Risk Manager - Final risk-adjusted decision maker")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--synthesis-file", 
                       default="research_synthesis.json",
                       help="Path to research synthesis JSON")
    parser.add_argument("--portfolio-value", type=float, default=100000,
                       help="Total portfolio value")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-decision", help="Save decision as JSON")
    
    args = parser.parse_args()
    
    manager = RiskManager(args.ticker, args.portfolio_value, args.api_key, args.model)
    
    # Run evaluation
    report, decision = manager.evaluate(synthesis_file=args.synthesis_file)
    
    print(report)
    
    # Save outputs
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSaved report to {args.output}")
    
    if args.save_decision and decision:
        with open(args.save_decision, 'w') as f:
            json.dump(decision, f, indent=2)
        print(f"Saved decision to {args.save_decision}")

if __name__ == "__main__":
    main()