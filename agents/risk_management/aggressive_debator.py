"""
Aggressive Debator - High risk tolerance evaluation of bull/bear research
Usage: python aggressive_debator.py AAPL --bull-file ../researcher/bull_thesis.json --bear-file ../researcher/bear_thesis.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any
from openai import OpenAI

class AggressiveDebator:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.risk_profile = "AGGRESSIVE"
        
        self.system_prompt = """You are an aggressive risk-taking trader who favors high-conviction, high-reward trades.
        You believe in taking larger positions when the opportunity is right, accepting volatility for returns.
        Your philosophy:
        - Fortune favors the bold - bigger bets on strong convictions
        - Volatility is opportunity, not just risk
        - Missing upside is worse than temporary drawdowns
        - Maximum position size: 25% of portfolio
        - Accept 10-15% drawdowns for 30%+ upside potential
        
        Evaluate the bull and bear cases and make an aggressive but reasoned recommendation.
        End with: AGGRESSIVE STANCE: BUY/SELL/HOLD - Position Size: X% - Confidence: High/Medium/Low"""
        
        self.risk_parameters = {
            'max_position_size': 0.25,  # 25% of portfolio
            'min_position_size': 0.05,  # 5% minimum
            'max_drawdown_tolerance': 0.15,  # 15% drawdown acceptable
            'min_reward_ratio': 2.0,  # Want 2:1 reward/risk minimum
            'volatility_preference': 'HIGH',  # Actually prefer volatility
            'time_horizon': 'MEDIUM',  # 3-6 months
            'conviction_threshold': 0.6  # 60% conviction to act
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
                'risk_reward': {'upside_potential': '20-30%', 'downside_risk': '10%', 'reward_risk_ratio': 2.5},
                'catalysts': [{'description': 'Strong momentum', 'timeline': 'near-term'}],
                'core_thesis': 'Bullish momentum intact'
            }
        else:
            return {
                'risk_assessment': {'downside_risk': '20%', 'risk_level': 'MEDIUM'},
                'downside_triggers': [{'description': 'Market weakness', 'timeline': 'ongoing'}],
                'core_thesis': 'Risks are manageable'
            }
    
    def evaluate_opportunity(self, bull_thesis, bear_thesis):
        """Evaluate from aggressive perspective"""
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
        
        # Aggressive scoring - we favor upside potential
        upside = bull_rr.get('upside_potential', '10%')
        downside = bear_ra.get('downside_risk', '20%')
        
        # Extract percentages
        upside_pct = self.extract_percentage(upside)
        downside_pct = self.extract_percentage(downside)
        
        # Aggressive decision logic
        if rr_ratio >= 2.0 and upside_pct >= 20:
            evaluation['stance'] = 'STRONG BUY'
            evaluation['position_size'] = 0.20  # 20% position
            evaluation['reasoning'] = "Excellent risk/reward with significant upside"
        elif rr_ratio >= 1.5 and upside_pct >= 15:
            evaluation['stance'] = 'BUY'
            evaluation['position_size'] = 0.15  # 15% position
            evaluation['reasoning'] = "Good opportunity worth aggressive positioning"
        elif downside_pct >= 30:
            evaluation['stance'] = 'SELL'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = "Even for aggressive traders, downside risk too severe"
        else:
            evaluation['stance'] = 'HOLD/WAIT'
            evaluation['position_size'] = 0.05  # Small position
            evaluation['reasoning'] = "Insufficient conviction for aggressive bet"
        
        # Check catalysts for timing
        catalysts = bull_thesis.get('catalysts', [])
        if catalysts and 'imminent' in str(catalysts[0].get('timeline', '')):
            evaluation['position_size'] *= 1.5  # Increase size for imminent catalysts
            evaluation['position_size'] = min(evaluation['position_size'], self.risk_parameters['max_position_size'])
        
        evaluation['confidence'] = self.calculate_confidence(bull_thesis, bear_thesis)
        
        return evaluation
    
    def extract_percentage(self, value_str):
        """Extract percentage from string like '20-30%' """
        if isinstance(value_str, (int, float)):
            return value_str
        if isinstance(value_str, str):
            # Extract first number
            import re
            numbers = re.findall(r'\d+', value_str)
            if numbers:
                return float(numbers[0])
        return 10.0  # Default
    
    def calculate_confidence(self, bull_thesis, bear_thesis):
        """Calculate confidence level"""
        # Aggressive traders have higher baseline confidence
        confidence_score = 70  # Start at 70%
        
        # Boost confidence for strong bull case
        if bull_thesis.get('risk_reward', {}).get('conviction_level') == 'HIGH':
            confidence_score += 20
        
        # Only slightly reduce for bear risks (aggressive = risk tolerant)
        if bear_thesis.get('risk_assessment', {}).get('risk_level') == 'HIGH':
            confidence_score -= 10  # Only -10 instead of more
        
        if confidence_score >= 80:
            return 'HIGH'
        elif confidence_score >= 60:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_aggressive_plan(self, evaluation, bull_thesis, bear_thesis):
        """Generate aggressive trading plan"""
        plan = {
            'entry_strategy': '',
            'position_building': '',
            'risk_management': '',
            'profit_targets': [],
            'time_frame': ''
        }
        
        if evaluation['stance'] in ['BUY', 'STRONG BUY']:
            plan['entry_strategy'] = "Scale in aggressively on any weakness"
            plan['position_building'] = f"Start with {evaluation['position_size']*100/2:.0f}%, double on confirmation"
            plan['risk_management'] = "Wide stop at -15% to avoid shakeouts"
            plan['profit_targets'] = ["First target: +20%", "Second target: +35%", "Let remainder run"]
            plan['time_frame'] = "3-6 months for full position"
        elif evaluation['stance'] == 'SELL':
            plan['entry_strategy'] = "Exit immediately or short aggressively"
            plan['position_building'] = "Full exit, consider inverse position"
            plan['risk_management'] = "No averaging down - cut losses"
            plan['profit_targets'] = ["Target: Exit or -20% on short"]
            plan['time_frame'] = "Immediate action"
        else:
            plan['entry_strategy'] = "Wait for better setup"
            plan['position_building'] = "Minimal pilot position only"
            plan['risk_management'] = "Tight stop at -5%"
            plan['profit_targets'] = ["Quick profit at +10%"]
            plan['time_frame'] = "Re-evaluate in 1-2 weeks"
        
        return plan
    
    def synthesize_with_llm(self, evaluation_data):
        """Use LLM for final synthesis"""
        if not self.client:
            return self.create_fallback_report(evaluation_data)
        
        try:
            prompt = f"""As an aggressive trader, evaluate this opportunity for {self.ticker}:

Bull Case Summary: {evaluation_data['bull_summary']}
Bear Case Summary: {evaluation_data['bear_summary']}
Initial Evaluation: {evaluation_data['evaluation']}

Provide an aggressive but intelligent trading recommendation that:
1. Explains why this is or isn't an aggressive opportunity
2. Specifies exact position sizing (max 25%)
3. Defines entry points and scaling strategy
4. Sets profit targets (be ambitious)
5. Acknowledges risks but explains why they're acceptable

Be bold but not reckless. Focus on asymmetric opportunities."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8  # Higher temp for aggressive stance
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
AGGRESSIVE RISK EVALUATION - {self.ticker}
{'='*60}
Risk Profile: AGGRESSIVE (High Risk Tolerance)
Evaluation Date: {eval['timestamp']}

DECISION: {eval['stance']}
Position Size: {eval['position_size']*100:.1f}% of portfolio
Reasoning: {eval['reasoning']}

TRADING PLAN:
Entry Strategy: {plan['entry_strategy']}
Position Building: {plan['position_building']}
Risk Management: {plan['risk_management']}

Profit Targets:
"""
        for target in plan['profit_targets']:
            report += f"  • {target}\n"
        
        report += f"""
Time Frame: {plan['time_frame']}

AGGRESSIVE PERSPECTIVE:
This analysis favors upside potential over downside protection.
We accept higher volatility and drawdowns for outsized returns.
Maximum drawdown tolerance: {self.risk_parameters['max_drawdown_tolerance']*100:.0f}%
Minimum reward/risk ratio: {self.risk_parameters['min_reward_ratio']:.1f}x

AGGRESSIVE STANCE: {eval['stance']} - Position Size: {eval['position_size']*100:.0f}% - Confidence: {eval['confidence']}
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
        
        print(f"Aggressive evaluation for {self.ticker}...")
        
        # Evaluate opportunity
        evaluation = self.evaluate_opportunity(bull_thesis, bear_thesis)
        
        # Generate trading plan
        trading_plan = self.generate_aggressive_plan(evaluation, bull_thesis, bear_thesis)
        
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
    parser = argparse.ArgumentParser(description="Aggressive Risk Debator - High risk tolerance evaluation")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--bull-file", help="Path to bull thesis JSON")
    parser.add_argument("--bear-file", help="Path to bear thesis JSON")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-evaluation", help="Save evaluation as JSON")
    
    args = parser.parse_args()
    
    debator = AggressiveDebator(args.ticker, args.api_key, args.model)
    
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