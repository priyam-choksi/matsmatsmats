"""
Research Manager - Synthesizes all research and risk evaluations
Usage: python research_manager.py AAPL --bull-file ../researcher/bull_thesis.json --bear-file ../researcher/bear_thesis.json --output synthesis.json
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any
from openai import OpenAI

class ResearchManager:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are a Research Manager synthesizing multiple analyst perspectives.
        Your role is to:
        1. Objectively weigh all research (bull, bear, and risk evaluations)
        2. Identify consensus views and critical disagreements
        3. Calculate probability-weighted outcomes
        4. Determine the most likely scenarios
        5. Provide clear, actionable intelligence for final decision-making
        
        Be objective and data-driven. Don't favor any particular stance.
        End with: RESEARCH CONCLUSION: Strong Buy/Buy/Hold/Sell/Strong Sell - Confidence: High/Medium/Low"""
        
        # Storage for all inputs
        self.research_inputs = {
            'bull_thesis': {},
            'bear_thesis': {},
            'risk_evaluations': {
                'aggressive': {},
                'neutral': {},
                'conservative': {}
            }
        }
    
    def load_research_files(self, bull_file, bear_file):
        """Load bull and bear research"""
        # Load bull thesis
        if os.path.exists(bull_file):
            with open(bull_file, 'r') as f:
                self.research_inputs['bull_thesis'] = json.load(f)
        else:
            print(f"Warning: Bull thesis not found at {bull_file}")
            
        # Load bear thesis
        if os.path.exists(bear_file):
            with open(bear_file, 'r') as f:
                self.research_inputs['bear_thesis'] = json.load(f)
        else:
            print(f"Warning: Bear thesis not found at {bear_file}")
    
    def run_risk_evaluations(self):
        """Run or load risk debator evaluations"""
        risk_agents = {
            'aggressive': '../risk_management/aggressive_debator.py',
            'neutral': '../risk_management/neutral_debator.py',
            'conservative': '../risk_management/conservative_debator.py'
        }
        
        bull_file = '../researcher/bull_thesis.json'
        bear_file = '../researcher/bear_thesis.json'
        
        for risk_type, script_path in risk_agents.items():
            print(f"  Getting {risk_type} evaluation...")
            
            # Check for saved evaluation first
            saved_file = f"../risk_management/{risk_type}_evaluation_{self.ticker}.json"
            if os.path.exists(saved_file):
                with open(saved_file, 'r') as f:
                    self.research_inputs['risk_evaluations'][risk_type] = json.load(f)
            elif os.path.exists(script_path):
                # Run the debator
                cmd = [
                    "python", script_path, self.ticker,
                    "--bull-file", bull_file,
                    "--bear-file", bear_file,
                    "--save-evaluation", saved_file
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and os.path.exists(saved_file):
                        with open(saved_file, 'r') as f:
                            self.research_inputs['risk_evaluations'][risk_type] = json.load(f)
                except Exception as e:
                    print(f"    Error running {risk_type}: {e}")
            else:
                # Create default evaluation
                self.research_inputs['risk_evaluations'][risk_type] = {
                    'stance': 'HOLD',
                    'position_size': 0.05 if risk_type == 'aggressive' else 0.03 if risk_type == 'neutral' else 0.01,
                    'confidence': 'LOW'
                }
    
    def analyze_consensus(self):
        """Analyze consensus across all inputs"""
        consensus = {
            'recommendations': {},
            'position_sizes': {},
            'confidence_levels': {},
            'key_agreements': [],
            'key_conflicts': []
        }
        
        # Collect recommendations
        if self.research_inputs['bull_thesis']:
            bull_rr = self.research_inputs['bull_thesis'].get('risk_reward', {})
            consensus['recommendations']['bull'] = 'BUY' if bull_rr.get('reward_risk_ratio', 0) > 2 else 'HOLD'
            
        if self.research_inputs['bear_thesis']:
            bear_ra = self.research_inputs['bear_thesis'].get('risk_assessment', {})
            consensus['recommendations']['bear'] = 'SELL' if bear_ra.get('risk_level') == 'HIGH' else 'HOLD'
        
        # Collect risk evaluations
        for risk_type, evaluation in self.research_inputs['risk_evaluations'].items():
            if evaluation:
                stance = evaluation.get('stance', 'HOLD')
                consensus['recommendations'][risk_type] = stance
                consensus['position_sizes'][risk_type] = evaluation.get('position_size', 0)
                consensus['confidence_levels'][risk_type] = evaluation.get('confidence', 'LOW')
        
        # Find agreements
        all_recs = list(consensus['recommendations'].values())
        if all_recs:
            most_common = max(set(all_recs), key=all_recs.count)
            agreement_count = all_recs.count(most_common)
            if agreement_count >= 3:
                consensus['key_agreements'].append(f"{agreement_count}/5 agree on {most_common}")
        
        # Find conflicts
        if 'BUY' in all_recs and 'SELL' in all_recs:
            consensus['key_conflicts'].append("Direct conflict: Some say BUY, others say SELL")
        
        # Average position size recommendation
        position_sizes = [ps for ps in consensus['position_sizes'].values() if ps > 0]
        consensus['avg_position_size'] = sum(position_sizes) / len(position_sizes) if position_sizes else 0
        
        return consensus
    
    def calculate_probabilities(self):
        """Calculate outcome probabilities based on all inputs"""
        probabilities = {
            'bull_case': 0,
            'bear_case': 0,
            'base_case': 0,
            'scenarios': []
        }
        
        # Weight different inputs
        weights = {
            'bull_research': 0.20,
            'bear_research': 0.20,
            'aggressive': 0.15,
            'neutral': 0.30,  # Highest weight to balanced view
            'conservative': 0.15
        }
        
        # Calculate bull probability
        bull_score = 0
        
        # Bull thesis contribution
        if self.research_inputs['bull_thesis']:
            bull_rr = self.research_inputs['bull_thesis'].get('risk_reward', {})
            if bull_rr.get('conviction_level') == 'HIGH':
                bull_score += weights['bull_research']
            elif bull_rr.get('conviction_level') == 'MEDIUM':
                bull_score += weights['bull_research'] * 0.5
        
        # Risk evaluations contribution
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            evaluation = self.research_inputs['risk_evaluations'].get(risk_type, {})
            stance = evaluation.get('stance', '')
            if 'BUY' in stance:
                bull_score += weights[risk_type]
            elif 'HOLD' in stance:
                bull_score += weights[risk_type] * 0.3
        
        probabilities['bull_case'] = min(bull_score * 100, 90)  # Cap at 90%
        
        # Calculate bear probability
        bear_score = 0
        
        # Bear thesis contribution
        if self.research_inputs['bear_thesis']:
            bear_ra = self.research_inputs['bear_thesis'].get('risk_assessment', {})
            if bear_ra.get('risk_level') == 'HIGH':
                bear_score += weights['bear_research']
            elif bear_ra.get('risk_level') == 'MEDIUM':
                bear_score += weights['bear_research'] * 0.5
        
        # Risk evaluations contribution
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            evaluation = self.research_inputs['risk_evaluations'].get(risk_type, {})
            stance = evaluation.get('stance', '')
            if 'SELL' in stance or 'AVOID' in stance:
                bear_score += weights[risk_type]
        
        probabilities['bear_case'] = min(bear_score * 100, 90)  # Cap at 90%
        
        # Base case is remainder
        probabilities['base_case'] = max(100 - probabilities['bull_case'] - probabilities['bear_case'], 10)
        
        # Normalize to 100%
        total = probabilities['bull_case'] + probabilities['bear_case'] + probabilities['base_case']
        if total > 0:
            probabilities['bull_case'] = (probabilities['bull_case'] / total) * 100
            probabilities['bear_case'] = (probabilities['bear_case'] / total) * 100
            probabilities['base_case'] = (probabilities['base_case'] / total) * 100
        
        # Define scenarios
        probabilities['scenarios'] = [
            {
                'name': 'Bull Case',
                'probability': f"{probabilities['bull_case']:.0f}%",
                'outcome': self.research_inputs['bull_thesis'].get('risk_reward', {}).get('upside_potential', '15-20%'),
                'description': 'Positive catalysts materialize, momentum continues'
            },
            {
                'name': 'Bear Case',
                'probability': f"{probabilities['bear_case']:.0f}%",
                'outcome': self.research_inputs['bear_thesis'].get('risk_assessment', {}).get('downside_risk', '15-20%'),
                'description': 'Risk factors dominate, downside triggers activate'
            },
            {
                'name': 'Base Case',
                'probability': f"{probabilities['base_case']:.0f}%",
                'outcome': 'Sideways ±5%',
                'description': 'Mixed signals, range-bound trading'
            }
        ]
        
        return probabilities
    
    def identify_key_factors(self):
        """Identify the most important decision factors"""
        key_factors = {
            'critical_positives': [],
            'critical_risks': [],
            'decision_drivers': [],
            'watch_points': []
        }
        
        # Extract from bull thesis
        if self.research_inputs['bull_thesis']:
            catalysts = self.research_inputs['bull_thesis'].get('catalysts', [])[:2]
            for catalyst in catalysts:
                key_factors['critical_positives'].append(catalyst.get('description', ''))
        
        # Extract from bear thesis
        if self.research_inputs['bear_thesis']:
            triggers = self.research_inputs['bear_thesis'].get('downside_triggers', [])[:2]
            for trigger in triggers:
                key_factors['critical_risks'].append(trigger.get('description', ''))
        
        # Extract red flags from conservative
        conservative_eval = self.research_inputs['risk_evaluations'].get('conservative', {})
        if 'red_flags' in conservative_eval:
            key_factors['critical_risks'].extend(conservative_eval['red_flags'][:2])
        
        # Decision drivers based on consensus
        neutral_eval = self.research_inputs['risk_evaluations'].get('neutral', {})
        if neutral_eval:
            expected_value = neutral_eval.get('expected_value', 0)
            key_factors['decision_drivers'].append(f"Expected Value: {expected_value:.1f}%")
        
        # Watch points
        key_factors['watch_points'] = [
            "Monitor for catalyst confirmation",
            "Watch for risk escalation",
            "Track analyst consensus changes"
        ]
        
        return key_factors
    
    def form_conclusion(self, consensus, probabilities, key_factors):
        """Form final research conclusion"""
        conclusion = {
            'recommendation': '',
            'confidence': '',
            'rationale': '',
            'position_size_range': '',
            'time_horizon': '',
            'key_risks': [],
            'key_catalysts': []
        }
        
        # Determine recommendation based on probabilities and consensus
        bull_prob = probabilities['bull_case']
        bear_prob = probabilities['bear_case']
        avg_position = consensus['avg_position_size']
        
        if bull_prob > 60 and avg_position > 0.05:
            conclusion['recommendation'] = 'BUY'
            conclusion['confidence'] = 'MEDIUM' if bull_prob > 70 else 'LOW'
        elif bear_prob > 60:
            conclusion['recommendation'] = 'SELL'
            conclusion['confidence'] = 'MEDIUM' if bear_prob > 70 else 'LOW'
        elif avg_position > 0.02:
            conclusion['recommendation'] = 'HOLD'
            conclusion['confidence'] = 'LOW'
        else:
            conclusion['recommendation'] = 'AVOID'
            conclusion['confidence'] = 'MEDIUM'
        
        # Position size range based on risk evaluations
        positions = consensus['position_sizes']
        if positions:
            min_pos = min(p for p in positions.values() if p > 0) if any(p > 0 for p in positions.values()) else 0
            max_pos = max(positions.values())
            conclusion['position_size_range'] = f"{min_pos*100:.0f}%-{max_pos*100:.0f}%"
        else:
            conclusion['position_size_range'] = "0%-5%"
        
        # Rationale
        conclusion['rationale'] = f"Bull probability: {bull_prob:.0f}%, Bear probability: {bear_prob:.0f}%. "
        conclusion['rationale'] += f"Risk evaluations suggest {conclusion['position_size_range']} position. "
        conclusion['rationale'] += consensus['key_agreements'][0] if consensus['key_agreements'] else "Mixed signals across evaluations."
        
        # Time horizon
        conclusion['time_horizon'] = "3-6 months"
        
        # Key risks and catalysts
        conclusion['key_risks'] = key_factors['critical_risks'][:3]
        conclusion['key_catalysts'] = key_factors['critical_positives'][:3]
        
        return conclusion
    
    def synthesize_with_llm(self, synthesis_data):
        """Use LLM for final synthesis"""
        if not self.client:
            return self.create_fallback_report(synthesis_data)
        
        try:
            prompt = f"""As Research Manager, synthesize all research for {self.ticker}:

Consensus Analysis: {json.dumps(synthesis_data['consensus'], indent=2)}
Probability Assessment: {json.dumps(synthesis_data['probabilities'], indent=2)}
Key Factors: {json.dumps(synthesis_data['key_factors'], indent=2)}
Initial Conclusion: {json.dumps(synthesis_data['conclusion'], indent=2)}

Create a comprehensive synthesis that:
1. Weighs all perspectives objectively
2. Identifies the most probable outcome
3. Highlights critical decision factors
4. Provides clear recommendation with rationale
5. Specifies actionable intelligence for risk management

Be objective and data-driven in your assessment."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.create_fallback_report(synthesis_data)
    
    def create_fallback_report(self, synthesis_data):
        """Create report without LLM"""
        consensus = synthesis_data['consensus']
        probs = synthesis_data['probabilities']
        factors = synthesis_data['key_factors']
        conclusion = synthesis_data['conclusion']
        
        report = f"""
RESEARCH SYNTHESIS - {self.ticker}
{'='*60}
Research Manager: Objective Analysis of All Inputs
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

CONSENSUS ANALYSIS:
-------------------
Recommendations Summary:
"""
        for source, rec in consensus['recommendations'].items():
            report += f"  • {source:12}: {rec}\n"
        
        if consensus['key_agreements']:
            report += f"\nKey Agreement: {consensus['key_agreements'][0]}\n"
        if consensus['key_conflicts']:
            report += f"Key Conflict: {consensus['key_conflicts'][0]}\n"
        
        report += f"""
Average Position Size: {consensus['avg_position_size']*100:.1f}%

PROBABILITY ASSESSMENT:
-----------------------"""
        for scenario in probs['scenarios']:
            report += f"""
{scenario['name']}: {scenario['probability']}
  Outcome: {scenario['outcome']}
  {scenario['description']}"""
        
        report += f"""

KEY DECISION FACTORS:
---------------------
Critical Positives:"""
        for positive in factors['critical_positives'][:3]:
            report += f"\n  ✓ {positive}"
        
        report += "\n\nCritical Risks:"
        for risk in factors['critical_risks'][:3]:
            report += f"\n  ⚠ {risk}"
        
        report += f"""

RESEARCH CONCLUSION:
--------------------
Recommendation: {conclusion['recommendation']}
Confidence: {conclusion['confidence']}
Position Size Range: {conclusion['position_size_range']}
Time Horizon: {conclusion['time_horizon']}

Rationale: {conclusion['rationale']}

ACTIONABLE INTELLIGENCE:
------------------------
1. {conclusion['recommendation']} recommendation with {conclusion['confidence']} confidence
2. Position sizing should be {conclusion['position_size_range']} based on risk tolerance
3. Key risks to monitor: {', '.join(conclusion['key_risks'][:2]) if conclusion['key_risks'] else 'Various'}
4. Potential catalysts: {', '.join(conclusion['key_catalysts'][:2]) if conclusion['key_catalysts'] else 'Limited'}

RESEARCH CONCLUSION: {conclusion['recommendation']} - Confidence: {conclusion['confidence']}
"""
        return report
    
    def synthesize(self, load_evaluations=True):
        """Main synthesis function"""
        print(f"Research Manager synthesizing data for {self.ticker}...")
        
        # Load or run risk evaluations if needed
        if load_evaluations:
            print("Loading risk evaluations...")
            self.run_risk_evaluations()
        
        # Analyze consensus
        print("Analyzing consensus...")
        consensus = self.analyze_consensus()
        
        # Calculate probabilities
        print("Calculating probabilities...")
        probabilities = self.calculate_probabilities()
        
        # Identify key factors
        print("Identifying key factors...")
        key_factors = self.identify_key_factors()
        
        # Form conclusion
        print("Forming conclusion...")
        conclusion = self.form_conclusion(consensus, probabilities, key_factors)
        
        # Compile synthesis data
        synthesis_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'consensus': consensus,
            'probabilities': probabilities,
            'key_factors': key_factors,
            'conclusion': conclusion,
            'research_inputs_summary': {
                'bull_thesis_loaded': bool(self.research_inputs['bull_thesis']),
                'bear_thesis_loaded': bool(self.research_inputs['bear_thesis']),
                'risk_evaluations_loaded': sum(1 for e in self.research_inputs['risk_evaluations'].values() if e)
            }
        }
        
        # Generate report
        report = self.synthesize_with_llm(synthesis_data)
        
        return report, synthesis_data

def main():
    parser = argparse.ArgumentParser(description="Research Manager - Synthesizes all research")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--bull-file", 
                       default="../researcher/bull_thesis.json",
                       help="Path to bull thesis JSON")
    parser.add_argument("--bear-file", 
                       default="../researcher/bear_thesis.json",
                       help="Path to bear thesis JSON")
    parser.add_argument("--skip-evaluations", action="store_true",
                       help="Skip running risk evaluations")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-synthesis", help="Save synthesis data as JSON")
    
    args = parser.parse_args()
    
    manager = ResearchManager(args.ticker, args.api_key, args.model)
    
    # Load research files
    manager.load_research_files(args.bull_file, args.bear_file)
    
    # Run synthesis
    report, synthesis_data = manager.synthesize(load_evaluations=not args.skip_evaluations)
    
    print(report)
    
    # Save outputs
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSaved report to {args.output}")
    
    if args.save_synthesis:
        with open(args.save_synthesis, 'w') as f:
            json.dump(synthesis_data, f, indent=2)
        print(f"Saved synthesis to {args.save_synthesis}")

if __name__ == "__main__":
    main()