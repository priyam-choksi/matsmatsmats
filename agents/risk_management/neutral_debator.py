"""
Neutral Debator - Balanced risk evaluation of bull/bear research
Usage: python neutral_debator.py AAPL --bull-file ../researcher/bull_thesis.json --bear-file ../researcher/bear_thesis.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any
from openai import OpenAI

class NeutralDebator:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.risk_profile = "NEUTRAL"
        
        self.system_prompt = """You are a balanced, objective trader who weighs risk and reward equally.
        You believe in measured positions based on probability-weighted outcomes.
        Your philosophy:
        - Balance is key - neither too aggressive nor too conservative
        - Position size should match conviction and risk/reward
        - Both upside and downside deserve equal consideration
        - Maximum position size: 10% of portfolio
        - Target 3:1 reward/risk ratio for new positions
        - Diversification matters
        
        Evaluate both bull and bear cases objectively and make a balanced recommendation.
        End with: NEUTRAL STANCE: BUY/SELL/HOLD - Position Size: X% - Confidence: High/Medium/Low"""
        
        self.risk_parameters = {
            'max_position_size': 0.10,  # 10% of portfolio
            'min_position_size': 0.02,  # 2% minimum
            'max_drawdown_tolerance': 0.07,  # 7% drawdown limit
            'min_reward_ratio': 3.0,  # Want 3:1 reward/risk
            'volatility_preference': 'MEDIUM',  # Moderate volatility
            'time_horizon': 'MEDIUM',  # 3-6 months
            'conviction_threshold': 0.65  # 65% conviction to act
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
                'risk_reward': {'upside_potential': '15-20%', 'downside_risk': '10%', 'reward_risk_ratio': 1.75},
                'catalysts': [{'description': 'Moderate momentum', 'timeline': 'medium-term'}],
                'core_thesis': 'Balanced bullish outlook'
            }
        else:
            return {
                'risk_assessment': {'downside_risk': '15%', 'risk_level': 'MEDIUM'},
                'downside_triggers': [{'description': 'Standard market risks', 'timeline': 'ongoing'}],
                'core_thesis': 'Manageable risk levels'
            }
    
    def evaluate_opportunity(self, bull_thesis, bear_thesis):
        """Evaluate from neutral/balanced perspective"""
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
        
        # Neutral scoring - balanced approach
        upside = bull_rr.get('upside_potential', '10%')
        downside = bear_ra.get('downside_risk', '20%')
        
        # Extract percentages
        upside_pct = self.extract_percentage(upside)
        downside_pct = self.extract_percentage(downside)
        
        # Calculate probability-weighted outcome
        bull_prob = bull_rr.get('bull_percentage', 50) / 100
        bear_prob = 1 - bull_prob
        
        expected_value = (upside_pct * bull_prob) - (downside_pct * bear_prob)
        
        # Neutral decision logic - based on expected value and ratio
        if expected_value > 10 and rr_ratio >= 3.0:
            evaluation['stance'] = 'BUY'
            evaluation['position_size'] = 0.08  # 8% position
            evaluation['reasoning'] = "Positive expected value with favorable risk/reward"
        elif expected_value > 5 and rr_ratio >= 2.0:
            evaluation['stance'] = 'SMALL BUY'
            evaluation['position_size'] = 0.05  # 5% position
            evaluation['reasoning'] = "Moderately positive setup warrants small position"
        elif expected_value < -10:
            evaluation['stance'] = 'SELL/AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = "Negative expected value suggests avoiding position"
        else:
            evaluation['stance'] = 'HOLD'
            evaluation['position_size'] = 0.03  # 3% watching position
            evaluation['reasoning'] = "Neutral expected value - wait for better setup"
        
        # Adjust for balanced risk management
        risk_level = bear_ra.get('risk_level', 'MEDIUM')
        if risk_level == 'HIGH':
            evaluation['position_size'] *= 0.5  # Halve position for high risk
            evaluation['reasoning'] += " (position reduced due to elevated risk)"
        
        evaluation['confidence'] = self.calculate_confidence(bull_thesis, bear_thesis, expected_value)
        evaluation['expected_value'] = expected_value
        
        return evaluation
    
    def extract_percentage(self, value_str):
        """Extract percentage from string like '20-30%' """
        if isinstance(value_str, (int, float)):
            return value_str
        if isinstance(value_str, str):
            # Extract numbers and average them for ranges
            import re
            numbers = re.findall(r'\d+', value_str)
            if numbers:
                return sum(float(n) for n in numbers) / len(numbers)
        return 10.0  # Default
    
    def calculate_confidence(self, bull_thesis, bear_thesis, expected_value):
        """Calculate confidence level based on agreement between factors"""
        confidence_score = 50  # Start neutral
        
        # Add points for positive expected value
        if expected_value > 10:
            confidence_score += 20
        elif expected_value > 5:
            confidence_score += 10
        elif expected_value < -5:
            confidence_score -= 20
        
        # Consider conviction levels from both sides
        bull_conviction = bull_thesis.get('risk_reward', {}).get('conviction_level', 'MEDIUM')
        bear_conviction = bear_thesis.get('risk_assessment', {}).get('conviction_level', 'MEDIUM')
        
        if bull_conviction == 'HIGH' and bear_conviction == 'LOW':
            confidence_score += 15
        elif bear_conviction == 'HIGH' and bull_conviction == 'LOW':
            confidence_score -= 15
        
        # Neutral traders need higher agreement for high confidence
        if confidence_score >= 75:
            return 'HIGH'
        elif confidence_score >= 50:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_balanced_plan(self, evaluation, bull_thesis, bear_thesis):
        """Generate balanced trading plan"""
        plan = {
            'entry_strategy': '',
            'position_building': '',
            'risk_management': '',
            'profit_targets': [],
            'rebalancing': '',
            'monitoring': ''
        }
        
        if evaluation['stance'] in ['BUY', 'SMALL BUY']:
            plan['entry_strategy'] = f"Scale in gradually over 2-3 entries"
            plan['position_building'] = f"Start with {evaluation['position_size']*100/3:.1f}%, add on confirmation"
            plan['risk_management'] = f"Stop loss at -7%, trail after +10%"
            plan['profit_targets'] = [
                "First target (33%): +15%", 
                "Second target (33%): +25%", 
                "Final target (34%): +35%"
            ]
            plan['rebalancing'] = "Trim if position exceeds 12% of portfolio"
            plan['monitoring'] = "Daily review, weekly reassessment"
        elif evaluation['stance'] == 'SELL/AVOID':
            plan['entry_strategy'] = "No new positions, exit existing"
            plan['position_building'] = "Systematic exit over 2-3 days"
            plan['risk_management'] = "Immediate exit if -5% from here"
            plan['profit_targets'] = ["N/A - Risk management mode"]
            plan['rebalancing'] = "Reallocate to safer assets"
            plan['monitoring'] = "Watch for reversal signals"
        else:  # HOLD
            plan['entry_strategy'] = "Wait for 3:1 risk/reward setup"
            plan['position_building'] = "Small pilot position only"
            plan['risk_management'] = "Tight stop at -3%"
            plan['profit_targets'] = ["Quick exit at +10%"]
            plan['rebalancing'] = "Maintain current allocation"
            plan['monitoring'] = "Wait for catalyst or setup improvement"
        
        # Add expected value to plan
        plan['expected_value'] = f"{evaluation.get('expected_value', 0):.1f}%"
        
        return plan
    
    def synthesize_with_llm(self, evaluation_data):
        """Use LLM for final synthesis"""
        if not self.client:
            return self.create_fallback_report(evaluation_data)
        
        try:
            prompt = f"""As a neutral, balanced trader, evaluate this opportunity for {self.ticker}:

Bull Case Summary: {evaluation_data['bull_summary']}
Bear Case Summary: {evaluation_data['bear_summary']}
Expected Value: {evaluation_data['evaluation']['expected_value']:.1f}%
Initial Evaluation: {evaluation_data['evaluation']}

Provide a balanced trading recommendation that:
1. Weighs both bull and bear cases equally
2. Calculates probability-weighted outcomes
3. Specifies appropriate position sizing (max 10%)
4. Defines clear risk management rules
5. Sets realistic profit targets

Focus on expected value and risk-adjusted returns."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5  # Moderate temp for balanced view
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
NEUTRAL RISK EVALUATION - {self.ticker}
{'='*60}
Risk Profile: NEUTRAL (Balanced Risk Approach)
Evaluation Date: {eval['timestamp']}

DECISION: {eval['stance']}
Position Size: {eval['position_size']*100:.1f}% of portfolio
Expected Value: {eval['expected_value']:.1f}%
Reasoning: {eval['reasoning']}

BALANCED TRADING PLAN:
Entry Strategy: {plan['entry_strategy']}
Position Building: {plan['position_building']}
Risk Management: {plan['risk_management']}
Rebalancing: {plan['rebalancing']}

Profit Targets:
"""
        for target in plan['profit_targets']:
            report += f"  • {target}\n"
        
        report += f"""
Monitoring: {plan['monitoring']}

NEUTRAL PERSPECTIVE:
This analysis weighs upside potential and downside risk equally.
Position sizing reflects probability-weighted expected outcomes.
Maximum position: {self.risk_parameters['max_position_size']*100:.0f}%
Required reward/risk: {self.risk_parameters['min_reward_ratio']:.1f}x

NEUTRAL STANCE: {eval['stance']} - Position Size: {eval['position_size']*100:.0f}% - Confidence: {eval['confidence']}
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
        
        print(f"Neutral evaluation for {self.ticker}...")
        
        # Evaluate opportunity
        evaluation = self.evaluate_opportunity(bull_thesis, bear_thesis)
        
        # Generate trading plan
        trading_plan = self.generate_balanced_plan(evaluation, bull_thesis, bear_thesis)
        
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
    parser = argparse.ArgumentParser(description="Neutral Risk Debator - Balanced risk evaluation")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--bull-file", help="Path to bull thesis JSON")
    parser.add_argument("--bear-file", help="Path to bear thesis JSON")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-evaluation", help="Save evaluation as JSON")
    
    args = parser.parse_args()
    
    debator = NeutralDebator(args.ticker, args.api_key, args.model)
    
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