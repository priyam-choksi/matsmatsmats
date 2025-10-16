"""
Bear Researcher - Builds comprehensive bearish case from discussion points
Usage: python bear_researcher.py AAPL --discussion-file ../orchestrators/discussion_points.json --output bear_thesis.txt
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any
from openai import OpenAI

class BearResearcher:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are a bearish equity researcher identifying risks and building the SELL/AVOID case.
        Your job is to:
        1. Synthesize all negative signals and risk factors
        2. Counter bullish arguments with logical concerns
        3. Identify triggers that could drive price decline
        4. Quantify downside risks with specific targets
        5. Suggest hedging strategies or exit points
        
        Be thorough and critical, but maintain intellectual honesty.
        End with: BEAR CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low"""
        
        self.bear_thesis = {
            'core_thesis': '',
            'risk_factors': [],
            'bull_rebuttals': [],
            'downside_triggers': [],
            'technical_concerns': {},
            'fundamental_risks': {},
            'risk_assessment': {},
            'hedging_strategies': [],
            'confidence': 0
        }
    
    def load_discussion_points(self, filepath):
        """Load discussion points from hub"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Discussion file not found: {filepath}")
            print("Running discussion hub first...")
            return self.run_discussion_hub()
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filepath}")
            return None
    
    def run_discussion_hub(self):
        """Run discussion hub if needed"""
        hub_script = "../orchestrators/discussion_hub.py"
        if not os.path.exists(hub_script):
            print(f"Discussion hub not found at {hub_script}")
            return None
        
        cmd = ["python", hub_script, self.ticker, "--run-analysts", "--format", "json"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"Error running discussion hub: {result.stderr}")
                return None
        except Exception as e:
            print(f"Failed to run discussion hub: {e}")
            return None
    
    def analyze_bear_signals(self, discussion_points):
        """Extract and strengthen bearish signals"""
        bear_evidence = discussion_points.get('bear_evidence', [])
        
        # Group signals by source
        signals_by_source = {}
        for evidence in bear_evidence:
            source = evidence.get('source', 'unknown')
            signal = evidence.get('signal', '')
            
            if source not in signals_by_source:
                signals_by_source[source] = []
            signals_by_source[source].append(signal)
        
        # Build risk categories
        risk_categories = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'macro': []
        }
        
        for source, signals in signals_by_source.items():
            if 'technical' in source:
                risk_categories['technical'].extend(signals)
            elif 'fundamental' in source:
                risk_categories['fundamental'].extend(signals)
            elif 'news' in source:
                risk_categories['sentiment'].extend(signals)
            elif 'macro' in source:
                risk_categories['macro'].extend(signals)
        
        # Add additional risk factors based on missing bullish signals
        bull_evidence = discussion_points.get('bull_evidence', [])
        if len(bear_evidence) > len(bull_evidence):
            risk_categories['sentiment'].append("Negative sentiment outweighs positive signals")
        
        return risk_categories
    
    def counter_bull_arguments(self, discussion_points):
        """Develop rebuttals to bullish arguments"""
        bull_evidence = discussion_points.get('bull_evidence', [])
        rebuttals = []
        
        for evidence in bull_evidence[:5]:  # Top 5 bull points
            signal = evidence.get('signal', '')
            source = evidence.get('source', '')
            
            # Create rebuttal structure
            rebuttal = {
                'bull_claim': signal,
                'source': source,
                'counter_argument': self.generate_rebuttal(signal, source)
            }
            rebuttals.append(rebuttal)
        
        return rebuttals
    
    def generate_rebuttal(self, bull_signal, source):
        """Generate counter-argument to bullish signal"""
        # Risk-focused rebuttals
        rebuttals_map = {
            'oversold': 'Oversold can get more oversold in downtrends; catching falling knives is dangerous',
            'support': 'Support levels often break in weak markets',
            'growth': 'Growth is slowing compared to historical rates',
            'beat': 'Beat expectations were already lowered; actual growth disappointing',
            'bullish': 'Bullish sentiment often marks tops',
            'upgrade': 'Analyst upgrades lag price action',
            'strong': 'Strength may be temporary or sector-specific',
            'buy': 'Buy ratings from analysts have poor track record at tops'
        }
        
        signal_lower = bull_signal.lower()
        for key, rebuttal in rebuttals_map.items():
            if key in signal_lower:
                return rebuttal
        
        return "This positive factor may already be priced in at current levels"
    
    def identify_downside_triggers(self, discussion_points):
        """Identify potential negative catalysts"""
        triggers = []
        
        # Check for immediate risks in priorities
        priorities = discussion_points.get('research_priorities', [])
        for priority in priorities:
            if priority.get('priority') == 'URGENT':
                triggers.append({
                    'type': 'immediate',
                    'description': f"Time-sensitive risk: {priority.get('description', '')}",
                    'impact': 'HIGH',
                    'timeline': 'imminent'
                })
        
        # Standard risk triggers
        risk_triggers = [
            {'type': 'earnings_miss', 'description': 'Potential earnings disappointment', 'timeline': '1-3 months'},
            {'type': 'technical_breakdown', 'description': 'Break below key support levels', 'timeline': '1-2 weeks'},
            {'type': 'macro_headwinds', 'description': 'Rising rates or recession risk', 'timeline': 'ongoing'},
            {'type': 'competitive_threats', 'description': 'Market share loss to competitors', 'timeline': '6-12 months'},
            {'type': 'regulatory', 'description': 'Regulatory scrutiny or changes', 'timeline': '3-6 months'}
        ]
        
        # Add relevant triggers based on signals
        summary = discussion_points.get('summary', {})
        if summary.get('bear_signal_count', 0) > summary.get('bull_signal_count', 0):
            triggers.extend(risk_triggers[:3])  # Add top risk triggers
        else:
            triggers.extend(risk_triggers[2:4])  # Add macro and competitive risks
        
        return triggers
    
    def calculate_risk_assessment(self, discussion_points):
        """Calculate comprehensive risk metrics"""
        summary = discussion_points.get('summary', {})
        recommendations = summary.get('recommendations', {})
        
        # Count bearish vs bullish recommendations
        bear_count = sum(1 for rec in recommendations.values() if rec == 'SELL')
        hold_count = sum(1 for rec in recommendations.values() if rec == 'HOLD')
        bull_count = sum(1 for rec in recommendations.values() if rec == 'BUY')
        total = len(recommendations)
        
        # Calculate risk score
        if total > 0:
            bear_percentage = ((bear_count + hold_count * 0.5) / total) * 100
        else:
            bear_percentage = 50
        
        # Estimate downside risks
        if bear_percentage >= 75:
            downside = "25-35%"
            upside = "5-10%"
            risk_level = "HIGH"
        elif bear_percentage >= 50:
            downside = "15-25%"
            upside = "10-15%"
            risk_level = "MEDIUM"
        else:
            downside = "10-15%"
            upside = "15-25%"
            risk_level = "LOW"
        
        return {
            'downside_risk': downside,
            'limited_upside': upside,
            'risk_level': risk_level,
            'bear_percentage': bear_percentage,
            'risk_score': min(100, bear_percentage * 1.2),  # Amplified risk score
            'conviction_level': 'HIGH' if bear_percentage >= 75 else 'MEDIUM' if bear_percentage >= 50 else 'LOW'
        }
    
    def suggest_hedging_strategies(self, risk_assessment):
        """Suggest risk management strategies"""
        strategies = []
        
        risk_level = risk_assessment['risk_level']
        
        if risk_level == 'HIGH':
            strategies.extend([
                {'strategy': 'EXIT', 'description': 'Consider full exit from position', 'urgency': 'IMMEDIATE'},
                {'strategy': 'PUT_OPTIONS', 'description': 'Buy protective puts 10% out of money', 'urgency': 'HIGH'},
                {'strategy': 'REDUCE', 'description': 'Reduce position by 75%', 'urgency': 'HIGH'}
            ])
        elif risk_level == 'MEDIUM':
            strategies.extend([
                {'strategy': 'STOP_LOSS', 'description': 'Set tight stop loss at -5%', 'urgency': 'MEDIUM'},
                {'strategy': 'REDUCE', 'description': 'Reduce position by 50%', 'urgency': 'MEDIUM'},
                {'strategy': 'COLLAR', 'description': 'Implement collar strategy', 'urgency': 'MEDIUM'}
            ])
        else:
            strategies.extend([
                {'strategy': 'MONITOR', 'description': 'Increase monitoring frequency', 'urgency': 'LOW'},
                {'strategy': 'STOP_LOSS', 'description': 'Set stop loss at -10%', 'urgency': 'LOW'},
                {'strategy': 'COVERED_CALLS', 'description': 'Sell covered calls for income', 'urgency': 'LOW'}
            ])
        
        return strategies
    
    def build_bear_thesis(self, risk_categories, risk_assessment):
        """Construct the core bear thesis"""
        thesis_components = []
        
        if risk_categories['technical']:
            thesis_components.append("Technical indicators show weakness and breakdown risk")
        if risk_categories['fundamental']:
            thesis_components.append("Fundamental deterioration threatens valuation")
        if risk_categories['sentiment']:
            thesis_components.append("Negative sentiment shift could accelerate selling")
        if risk_categories['macro']:
            thesis_components.append("Macro headwinds create systematic risk")
        
        if not thesis_components:
            thesis_components.append("Multiple risk factors suggest caution is warranted")
        
        core_thesis = f"""
The bear case for {self.ticker} is based on {len(thesis_components)} critical concerns:
{'. '.join(thesis_components)}.

With {risk_assessment['downside_risk']} downside risk versus only {risk_assessment['limited_upside']} upside potential,
the risk/reward is unfavorable for long positions. Risk level: {risk_assessment['risk_level']}.
Risk Score: {risk_assessment['risk_score']:.0f}/100 ({risk_assessment['bear_percentage']:.0f}% of signals are bearish).
"""
        return core_thesis
    
    def synthesize_with_llm(self, bear_data):
        """Use LLM to create comprehensive bear case"""
        if not self.client:
            return self.create_fallback_report(bear_data)
        
        try:
            prompt = f"""Analyze this data and create a compelling bear case for {self.ticker}:

{json.dumps(bear_data, indent=2)}

Create a comprehensive bearish research report that:
1. Presents critical risks and concerns
2. Counters bullish arguments effectively
3. Identifies specific downside catalysts
4. Quantifies potential losses
5. Recommends risk management strategies

Be critical but maintain analytical objectivity."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.create_fallback_report(bear_data)
    
    def create_fallback_report(self, bear_data):
        """Create report without LLM"""
        report = f"""
BEARISH RESEARCH REPORT - {self.ticker}
{'='*60}

CORE BEAR THESIS:
{bear_data['core_thesis']}

RISK FACTORS:
"""
        # Add technical risks
        if bear_data['risks']['technical']:
            report += "\nTechnical Concerns:\n"
            for signal in bear_data['risks']['technical'][:3]:
                report += f"  • {signal}\n"
        
        # Add fundamental risks
        if bear_data['risks']['fundamental']:
            report += "\nFundamental Risks:\n"
            for signal in bear_data['risks']['fundamental'][:3]:
                report += f"  • {signal}\n"
        
        # Add downside triggers
        report += "\nDOWNSIDE TRIGGERS:\n"
        for trigger in bear_data['downside_triggers'][:3]:
            report += f"  • {trigger['description']} (Timeline: {trigger.get('timeline', 'near-term')})\n"
        
        # Add risk assessment
        ra = bear_data['risk_assessment']
        report += f"""
RISK ASSESSMENT:
  Downside Risk: {ra['downside_risk']}
  Limited Upside: {ra['limited_upside']}
  Risk Level: {ra['risk_level']}
  Risk Score: {ra['risk_score']:.0f}/100
  
HEDGING STRATEGIES:
"""
        for strategy in bear_data['hedging_strategies'][:3]:
            report += f"  • [{strategy['urgency']}] {strategy['description']}\n"
        
        report += f"\nBEAR CASE STRENGTH: {ra['conviction_level']} - Confidence: {ra['conviction_level']}"
        
        return report
    
    def research(self, discussion_points):
        """Main research function"""
        if not discussion_points:
            return "Error: No discussion points available"
        
        print(f"Building bear case for {self.ticker}...")
        
        # Analyze components
        risk_categories = self.analyze_bear_signals(discussion_points)
        rebuttals = self.counter_bull_arguments(discussion_points)
        triggers = self.identify_downside_triggers(discussion_points)
        risk_assessment = self.calculate_risk_assessment(discussion_points)
        hedging = self.suggest_hedging_strategies(risk_assessment)
        
        # Build thesis
        core_thesis = self.build_bear_thesis(risk_categories, risk_assessment)
        
        # Compile bear data
        bear_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'core_thesis': core_thesis,
            'risks': risk_categories,
            'bull_rebuttals': rebuttals,
            'downside_triggers': triggers,
            'risk_assessment': risk_assessment,
            'hedging_strategies': hedging,
            'discussion_summary': discussion_points.get('summary', {})
        }
        
        # Store for later use
        self.bear_thesis = bear_data
        
        # Generate report with LLM
        report = self.synthesize_with_llm(bear_data)
        
        return report
    
    def save_thesis(self, filepath):
        """Save bear thesis data for other agents"""
        with open(filepath, 'w') as f:
            json.dump(self.bear_thesis, f, indent=2)
        print(f"Saved bear thesis data to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Bear Researcher - Builds bearish investment case")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--discussion-file", 
                       default="../orchestrators/discussion_points.json",
                       help="Path to discussion points JSON file")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-data", help="Save thesis data as JSON")
    
    args = parser.parse_args()
    
    researcher = BearResearcher(args.ticker, args.api_key, args.model)
    
    # Load discussion points
    discussion_points = researcher.load_discussion_points(args.discussion_file)
    
    # Run research
    report = researcher.research(discussion_points)
    
    print(report)
    
    # Save outputs
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSaved report to {args.output}")
    
    if args.save_data:
        researcher.save_thesis(args.save_data)

if __name__ == "__main__":
    main()