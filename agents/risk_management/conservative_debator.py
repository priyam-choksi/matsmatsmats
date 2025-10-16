"""
Conservative Debator - Risk-averse evaluation of bull/bear research
Usage: python conservative_debator.py AAPL --bull-file ../researcher/bull_thesis.json --bear-file ../researcher/bear_thesis.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any
from openai import OpenAI

class ConservativeDebator:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.risk_profile = "CONSERVATIVE"
        
        self.system_prompt = """You are a conservative, risk-averse trader focused on capital preservation.
        You believe in protecting downside above chasing upside, with safety as the priority.
        Your philosophy:
        - Return OF capital is more important than return ON capital
        - When in doubt, stay out
        - Small positions in only the highest conviction ideas
        - Maximum position size: 5% of portfolio
        - Require 4:1 reward/risk minimum
        - Avoid volatility and uncertain situations
        - Preservation of capital is paramount
        
        Evaluate opportunities with extreme caution and recommend only when risk is minimal.
        End with: CONSERVATIVE STANCE: BUY/SELL/HOLD - Position Size: X% - Confidence: High/Medium/Low"""
        
        self.risk_parameters = {
            'max_position_size': 0.05,  # 5% of portfolio maximum
            'min_position_size': 0.01,  # 1% minimum
            'max_drawdown_tolerance': 0.03,  # 3% drawdown limit
            'min_reward_ratio': 4.0,  # Want 4:1 reward/risk minimum
            'volatility_preference': 'LOW',  # Avoid volatile stocks
            'time_horizon': 'LONG',  # 6-12 months minimum
            'conviction_threshold': 0.80  # 80% conviction required
        }
    
    def load_research(self, bull_file, bear_file):
        """Load bull and bear research reports"""
        bull_thesis = {}
        bear_thesis = {}
        
        # Load bull thesis
        if os.path.exists(bull_file):
            with open(bull_file, 'r') as f:
                bull_thesis = json.load(f)
        else:
            print(f"Bull thesis file not found: {bull_file}")
            bull_thesis = self.create_default_thesis('bull')
        
        # Load bear thesis
        if os.path.exists(bear_file):
            with open(bear_file, 'r') as f:
                bear_thesis = json.load(f)
        else:
            print(f"Bear thesis file not found: {bear_file}")
            bear_thesis = self.create_default_thesis('bear')
        
        return bull_thesis, bear_thesis
    
    def create_default_thesis(self, thesis_type):
        """Create default thesis if files not found"""
        if thesis_type == 'bull':
            return {
                'risk_reward': {'upside_potential': '10-15%', 'downside_risk': '15%', 'reward_risk_ratio': 1.0},
                'catalysts': [{'description': 'Uncertain catalysts', 'timeline': 'unclear'}],
                'core_thesis': 'Limited conviction bull case'
            }
        else:
            return {
                'risk_assessment': {'downside_risk': '20%', 'risk_level': 'HIGH'},
                'downside_triggers': [{'description': 'Multiple risks present', 'timeline': 'ongoing'}],
                'core_thesis': 'Significant downside risks'
            }
    
    def evaluate_opportunity(self, bull_thesis, bear_thesis):
        """Evaluate from conservative perspective - focus on risk"""
        evaluation = {
            'profile': self.risk_profile,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract key metrics
        bull_rr = bull_thesis.get('risk_reward', {})
        bear_ra = bear_thesis.get('risk_assessment', {})
        
        # Parse reward/risk ratio
        rr_ratio = bull_rr.get('reward_risk_ratio', 1.0)
        if isinstance(rr_ratio, str):
            rr_ratio = 1.0
        
        # Conservative scoring - heavily weight downside
        upside = bull_rr.get('upside_potential', '10%')
        downside = bear_ra.get('downside_risk', '20%')
        risk_level = bear_ra.get('risk_level', 'HIGH')
        
        # Extract percentages (conservative = use worst case)
        upside_pct = self.extract_percentage(upside, conservative=True)
        downside_pct = self.extract_percentage(downside, conservative=False)  # Use higher downside
        
        # Conservative decision logic - very strict criteria
        if risk_level == 'HIGH':
            evaluation['stance'] = 'AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = "Risk level too high for conservative approach"
        elif rr_ratio >= 4.0 and downside_pct <= 10 and risk_level == 'LOW':
            evaluation['stance'] = 'SMALL BUY'
            evaluation['position_size'] = 0.03  # 3% position
            evaluation['reasoning'] = "Exceptional risk/reward with limited downside"
        elif rr_ratio >= 3.0 and downside_pct <= 15 and risk_level in ['LOW', 'MEDIUM']:
            evaluation['stance'] = 'MINIMAL BUY'
            evaluation['position_size'] = 0.02  # 2% position
            evaluation['reasoning'] = "Acceptable risk/reward for minimal exposure"
        elif downside_pct >= 20:
            evaluation['stance'] = 'SELL/AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = "Downside risk exceeds conservative tolerance"
        else:
            evaluation['stance'] = 'HOLD/WAIT'
            evaluation['position_size'] = 0.01  # 1% watching position
            evaluation['reasoning'] = "Insufficient risk/reward for conservative entry"
        
        # Further reduce position for any uncertainty
        confidence_levels = bull_thesis.get('discussion_summary', {}).get('confidence_levels', {})
        low_confidence_count = sum(1 for conf in confidence_levels.values() if conf == 'Low')
        if low_confidence_count >= 2:
            evaluation['position_size'] *= 0.5
            evaluation['reasoning'] += " (further reduced due to low analyst confidence)"
        
        # Check for red flags
        red_flags = self.identify_red_flags(bull_thesis, bear_thesis)
        if red_flags:
            evaluation['red_flags'] = red_flags
            if len(red_flags) >= 2:
                evaluation['stance'] = 'AVOID'
                evaluation['position_size'] = 0.0
                evaluation['reasoning'] = f"Multiple red flags detected: {', '.join(red_flags[:2])}"
        
        evaluation['confidence'] = self.calculate_confidence(bull_thesis, bear_thesis, risk_level)
        
        return evaluation
    
    def extract_percentage(self, value_str, conservative=True):
        """Extract percentage - conservative takes worst case"""
        if isinstance(value_str, (int, float)):
            return value_str
        if isinstance(value_str, str):
            import re
            numbers = re.findall(r'\d+', value_str)
            if numbers:
                if conservative:
                    return float(min(numbers))  # Take lower number for upside
                else:
                    return float(max(numbers))  # Take higher number for downside
        return 15.0  # Default to cautious
    
    def identify_red_flags(self, bull_thesis, bear_thesis):
        """Identify serious warning signs"""
        red_flags = []
        
        # Check bear thesis for high-risk indicators
        triggers = bear_thesis.get('downside_triggers', [])
        for trigger in triggers:
            if 'imminent' in str(trigger.get('timeline', '')).lower():
                red_flags.append("Imminent risk trigger")
            if 'HIGH' in str(trigger.get('impact', '')):
                red_flags.append("High impact risk event")
        
        # Check risk assessment
        risk_assessment = bear_thesis.get('risk_assessment', {})
        if risk_assessment.get('risk_score', 0) > 70:
            red_flags.append(f"Risk score too high: {risk_assessment.get('risk_score', 0):.0f}")
        
        # Check for lack of consensus
        summary = bull_thesis.get('discussion_summary', {})
        recommendations = summary.get('recommendations', {})
        if recommendations:
            sell_count = sum(1 for rec in recommendations.values() if rec == 'SELL')
            if sell_count >= 2:
                red_flags.append(f"{sell_count} analysts recommend SELL")
        
        return red_flags
    
    def calculate_confidence(self, bull_thesis, bear_thesis, risk_level):
        """Calculate confidence - conservatives need high certainty"""
        confidence_score = 30  # Start low
        
        # Need strong bull case for any confidence
        if bull_thesis.get('risk_reward', {}).get('conviction_level') == 'HIGH':
            confidence_score += 30
        
        # Heavily penalize high risk
        if risk_level == 'HIGH':
            confidence_score -= 40
        elif risk_level == 'MEDIUM':
            confidence_score -= 20
        elif risk_level == 'LOW':
            confidence_score += 20
        
        # Conservative investors need very high confidence
        if confidence_score >= 70:
            return 'HIGH'
        elif confidence_score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_conservative_plan(self, evaluation, bull_thesis, bear_thesis):
        """Generate conservative trading plan"""
        plan = {
            'entry_strategy': '',
            'position_building': '',
            'risk_management': '',
            'profit_targets': [],
            'exit_triggers': [],
            'monitoring': ''
        }
        
        if evaluation['stance'] in ['SMALL BUY', 'MINIMAL BUY']:
            plan['entry_strategy'] = "Wait for pullback to support, enter in small tranches"
            plan['position_building'] = f"Maximum {evaluation['position_size']*100:.1f}%, no averaging down"
            plan['risk_management'] = "Tight stop at -3%, no exceptions"
            plan['profit_targets'] = [
                "First target (50%): +8%",
                "Second target (30%): +12%",
                "Final target (20%): +15%"
            ]
            plan['exit_triggers'] = [
                "Any break below entry support",
                "Fundamental deterioration",
                "Risk level increase"
            ]
            plan['monitoring'] = "Daily monitoring with strict stop discipline"
        elif evaluation['stance'] in ['SELL/AVOID', 'AVOID']:
            plan['entry_strategy'] = "No entry - avoid completely"
            plan['position_building'] = "Zero allocation"
            plan['risk_management'] = "Stay in cash or safe assets"
            plan['profit_targets'] = ["N/A - Risk avoidance mode"]
            plan['exit_triggers'] = ["Exit any existing positions"]
            plan['monitoring'] = "Quarterly review only"
        else:  # HOLD/WAIT
            plan['entry_strategy'] = "Wait for 4:1 risk/reward with clear catalyst"
            plan['position_building'] = "1% pilot position maximum"
            plan['risk_management'] = "Stop at -2%"
            plan['profit_targets'] = ["Single target: +8%"]
            plan['exit_triggers'] = ["Any negative surprise"]
            plan['monitoring'] = "Weekly review for setup improvement"
        
        # Add safety measures
        plan['safety_rules'] = [
            "Never add to losing positions",
            "Exit immediately on earnings warnings",
            "Reduce all positions if market volatility spikes"
        ]
        
        return plan
    
    def synthesize_with_llm(self, evaluation_data):
        """Use LLM for final synthesis"""
        if not self.client:
            return self.create_fallback_report(evaluation_data)
        
        try:
            prompt = f"""As a conservative, risk-averse trader, evaluate this opportunity for {self.ticker}:

Bull Case Summary: {evaluation_data['bull_summary']}
Bear Case Summary: {evaluation_data['bear_summary']}
Red Flags: {evaluation_data['evaluation'].get('red_flags', [])}
Initial Evaluation: {evaluation_data['evaluation']}

Provide a conservative trading recommendation that:
1. Prioritizes capital preservation above returns
2. Identifies all significant risks and red flags
3. Recommends minimal position sizing (max 5%)
4. Sets strict risk controls and exit triggers
5. Only suggests entry if risk/reward is exceptional

Focus on protecting capital. When in doubt, recommend staying out."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Low temp for conservative consistency
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.create_fallback_report(evaluation_data)
    
    def create_fallback_report(self, evaluation_data):
        """Create report without LLM"""
        eval = evaluation_data['evaluation']
        plan = evaluation_data['trading_plan']
        
        report = f"""
CONSERVATIVE RISK EVALUATION - {self.ticker}
{'='*60}
Risk Profile: CONSERVATIVE (Capital Preservation Focus)
Evaluation Date: {eval['timestamp']}

DECISION: {eval['stance']}
Position Size: {eval['position_size']*100:.1f}% of portfolio
Reasoning: {eval['reasoning']}
"""
        
        if 'red_flags' in eval:
            report += "\nRED FLAGS DETECTED:\n"
            for flag in eval['red_flags']:
                report += f"  ⚠️ {flag}\n"
        
        report += f"""
CONSERVATIVE TRADING PLAN:
Entry Strategy: {plan['entry_strategy']}
Position Building: {plan['position_building']}
Risk Management: {plan['risk_management']}

Profit Targets:
"""
        for target in plan['profit_targets']:
            report += f"  • {target}\n"
        
        report += "\nExit Triggers:\n"
        for trigger in plan['exit_triggers']:
            report += f"  • {trigger}\n"
        
        report += "\nSafety Rules:\n"
        for rule in plan['safety_rules']:
            report += f"  • {rule}\n"
        
        report += f"""
Monitoring: {plan['monitoring']}

CONSERVATIVE PERSPECTIVE:
Capital preservation is the primary objective.
Only the highest quality setups with minimal risk are considered.
Maximum position: {self.risk_parameters['max_position_size']*100:.0f}%
Required reward/risk: {self.risk_parameters['min_reward_ratio']:.1f}x
Maximum acceptable drawdown: {self.risk_parameters['max_drawdown_tolerance']*100:.0f}%

CONSERVATIVE STANCE: {eval['stance']} - Position Size: {eval['position_size']*100:.0f}% - Confidence: {eval['confidence']}
"""
        return report
    
    def evaluate(self, bull_thesis=None, bear_thesis=None, bull_file=None, bear_file=None):
        """Main evaluation function"""
        # Load research if not provided
        if not bull_thesis or not bear_thesis:
            if not bull_file:
                bull_file = f"../researcher/bull_thesis_{self.ticker}.json"
            if not bear_file:
                bear_file = f"../researcher/bear_thesis_{self.ticker}.json"
            bull_thesis, bear_thesis = self.load_research(bull_file, bear_file)
        
        print(f"Conservative evaluation for {self.ticker}...")
        
        # Evaluate opportunity
        evaluation = self.evaluate_opportunity(bull_thesis, bear_thesis)
        
        # Generate trading plan
        trading_plan = self.generate_conservative_plan(evaluation, bull_thesis, bear_thesis)
        
        # Compile all data
        evaluation_data = {
            'evaluation': evaluation,
            'trading_plan': trading_plan,
            'bull_summary': bull_thesis.get('core_thesis', 'No bull thesis'),
            'bear_summary': bear_thesis.get('core_thesis', 'No bear thesis'),
            'risk_parameters': self.risk_parameters
        }
        
        # Generate report
        report = self.synthesize_with_llm(evaluation_data)
        
        return report, evaluation

def main():
    parser = argparse.ArgumentParser(description="Conservative Risk Debator - Risk-averse evaluation")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--bull-file", help="Path to bull thesis JSON")
    parser.add_argument("--bear-file", help="Path to bear thesis JSON")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-evaluation", help="Save evaluation as JSON")
    
    args = parser.parse_args()
    
    debator = ConservativeDebator(args.ticker, args.api_key, args.model)
    
    # Run evaluation
    report, evaluation = debator.evaluate(bull_file=args.bull_file, bear_file=args.bear_file)
    
    print(report)
    
    # Save outputs
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSaved report to {args.output}")
    
    if args.save_evaluation:
        with open(args.save_evaluation, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Saved evaluation to {args.save_evaluation}")

if __name__ == "__main__":
    main()