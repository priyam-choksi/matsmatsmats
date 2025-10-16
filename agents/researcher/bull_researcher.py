"""
Bull Researcher - Builds comprehensive bullish case from discussion points
Usage: python bull_researcher.py AAPL --discussion-file ../orchestrators/discussion_points.json --output bull_thesis.txt
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any
from openai import OpenAI

class BullResearcher:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are a bullish equity researcher building the strongest possible BUY case.
        Your job is to:
        1. Synthesize all positive signals into a coherent bull thesis
        2. Counter bearish arguments with logical rebuttals
        3. Identify catalysts that could drive price appreciation
        4. Quantify upside potential with specific targets
        5. Acknowledge risks but show why reward outweighs them
        
        Be thorough and data-driven, but maintain intellectual honesty.
        End with: BULL CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low"""
        
        self.bull_thesis = {
            'core_thesis': '',
            'supporting_evidence': [],
            'bear_rebuttals': [],
            'catalysts': [],
            'technical_setup': {},
            'fundamental_drivers': {},
            'risk_reward': {},
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
                # Parse JSON from stdout
                return json.loads(result.stdout)
            else:
                print(f"Error running discussion hub: {result.stderr}")
                return None
        except Exception as e:
            print(f"Failed to run discussion hub: {e}")
            return None
    
    def analyze_bull_signals(self, discussion_points):
        """Extract and strengthen bullish signals"""
        bull_evidence = discussion_points.get('bull_evidence', [])
        
        # Group signals by source
        signals_by_source = {}
        for evidence in bull_evidence:
            source = evidence.get('source', 'unknown')
            signal = evidence.get('signal', '')
            
            if source not in signals_by_source:
                signals_by_source[source] = []
            signals_by_source[source].append(signal)
        
        # Build evidence categories
        evidence_categories = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'macro': []
        }
        
        for source, signals in signals_by_source.items():
            if 'technical' in source:
                evidence_categories['technical'].extend(signals)
            elif 'fundamental' in source:
                evidence_categories['fundamental'].extend(signals)
            elif 'news' in source:
                evidence_categories['sentiment'].extend(signals)
            elif 'macro' in source:
                evidence_categories['macro'].extend(signals)
        
        return evidence_categories
    
    def counter_bear_arguments(self, discussion_points):
        """Develop rebuttals to bearish concerns"""
        bear_evidence = discussion_points.get('bear_evidence', [])
        rebuttals = []
        
        for evidence in bear_evidence[:5]:  # Top 5 bear concerns
            signal = evidence.get('signal', '')
            source = evidence.get('source', '')
            
            # Create rebuttal structure
            rebuttal = {
                'bear_concern': signal,
                'source': source,
                'counter_argument': self.generate_rebuttal(signal, source)
            }
            rebuttals.append(rebuttal)
        
        return rebuttals
    
    def generate_rebuttal(self, bear_signal, source):
        """Generate counter-argument to bearish signal"""
        # Simple rule-based rebuttals
        rebuttals_map = {
            'overbought': 'Overbought can persist in strong trends; momentum often continues',
            'resistance': 'Previous resistance becomes support after breakout',
            'overvalued': 'Growth companies often trade at premium multiples',
            'debt': 'Debt is manageable with strong cash flow generation',
            'competition': 'Market leader position provides competitive moat',
            'downgrade': 'Analyst downgrades often mark bottoms',
            'risk': 'Risk is already priced in at current levels'
        }
        
        signal_lower = bear_signal.lower()
        for key, rebuttal in rebuttals_map.items():
            if key in signal_lower:
                return rebuttal
        
        return "This concern appears overblown given the positive momentum"
    
    def identify_catalysts(self, discussion_points):
        """Identify potential positive catalysts"""
        catalysts = []
        
        # Look for time-sensitive events in research priorities
        priorities = discussion_points.get('research_priorities', [])
        for priority in priorities:
            if priority.get('focus') == 'time_sensitive':
                catalysts.append({
                    'type': 'immediate',
                    'description': priority.get('description', ''),
                    'impact': 'HIGH'
                })
        
        # Standard catalyst categories
        catalyst_types = [
            {'type': 'earnings', 'description': 'Next earnings beat potential', 'timeline': '1-3 months'},
            {'type': 'technical', 'description': 'Breakout from consolidation pattern', 'timeline': '1-2 weeks'},
            {'type': 'macro', 'description': 'Favorable sector rotation', 'timeline': 'ongoing'},
            {'type': 'fundamental', 'description': 'Improving margins and growth', 'timeline': '6-12 months'}
        ]
        
        # Add relevant catalysts based on signals
        summary = discussion_points.get('summary', {})
        if summary.get('bull_signal_count', 0) > summary.get('bear_signal_count', 0):
            catalysts.extend(catalyst_types[:2])  # Add near-term catalysts
        
        return catalysts
    
    def calculate_risk_reward(self, discussion_points):
        """Calculate risk/reward metrics"""
        summary = discussion_points.get('summary', {})
        recommendations = summary.get('recommendations', {})
        
        # Count bullish vs bearish recommendations
        bull_count = sum(1 for rec in recommendations.values() if rec == 'BUY')
        bear_count = sum(1 for rec in recommendations.values() if rec == 'SELL')
        total = len(recommendations)
        
        # Calculate conviction score
        if total > 0:
            bull_percentage = (bull_count / total) * 100
        else:
            bull_percentage = 50
        
        # Estimate risk/reward
        if bull_percentage >= 75:
            upside = "20-30%"
            downside = "5-10%"
            ratio = 3.0
        elif bull_percentage >= 50:
            upside = "15-20%"
            downside = "10-15%"
            ratio = 1.5
        else:
            upside = "10-15%"
            downside = "10-20%"
            ratio = 0.75
        
        return {
            'upside_potential': upside,
            'downside_risk': downside,
            'reward_risk_ratio': ratio,
            'bull_percentage': bull_percentage,
            'conviction_level': 'HIGH' if bull_percentage >= 75 else 'MEDIUM' if bull_percentage >= 50 else 'LOW'
        }
    
    def build_bull_thesis(self, evidence_categories, risk_reward):
        """Construct the core bull thesis"""
        thesis_components = []
        
        if evidence_categories['technical']:
            thesis_components.append("Technical momentum is strongly bullish")
        if evidence_categories['fundamental']:
            thesis_components.append("Fundamentals support higher valuation")
        if evidence_categories['sentiment']:
            thesis_components.append("Sentiment shift turning positive")
        if evidence_categories['macro']:
            thesis_components.append("Macro environment favors this sector")
        
        if not thesis_components:
            thesis_components.append("Multiple factors align for potential upside")
        
        core_thesis = f"""
The bull case for {self.ticker} rests on {len(thesis_components)} key pillars:
{'. '.join(thesis_components)}.

With {risk_reward['upside_potential']} upside potential versus {risk_reward['downside_risk']} downside risk,
the risk/reward ratio of {risk_reward['reward_risk_ratio']:.1f}x is attractive for long positions.
Conviction level: {risk_reward['conviction_level']} ({risk_reward['bull_percentage']:.0f}% of analysts lean bullish).
"""
        return core_thesis
    
    def synthesize_with_llm(self, bull_data):
        """Use LLM to create comprehensive bull case"""
        if not self.client:
            return self.create_fallback_report(bull_data)
        
        try:
            prompt = f"""Analyze this data and create a compelling bull case for {self.ticker}:

{json.dumps(bull_data, indent=2)}

Create a comprehensive bullish research report that:
1. Presents the strongest bull thesis
2. Addresses and counters bear concerns
3. Identifies specific catalysts with timelines
4. Quantifies upside potential
5. Acknowledges but minimizes risks

Be persuasive but maintain analytical rigor."""
            
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
            return self.create_fallback_report(bull_data)
    
    def create_fallback_report(self, bull_data):
        """Create report without LLM"""
        report = f"""
BULLISH RESEARCH REPORT - {self.ticker}
{'='*60}

CORE BULL THESIS:
{bull_data['core_thesis']}

SUPPORTING EVIDENCE:
"""
        # Add technical evidence
        if bull_data['evidence']['technical']:
            report += "\nTechnical Factors:\n"
            for signal in bull_data['evidence']['technical'][:3]:
                report += f"  • {signal}\n"
        
        # Add fundamental evidence
        if bull_data['evidence']['fundamental']:
            report += "\nFundamental Factors:\n"
            for signal in bull_data['evidence']['fundamental'][:3]:
                report += f"  • {signal}\n"
        
        # Add catalysts
        report += "\nKEY CATALYSTS:\n"
        for catalyst in bull_data['catalysts'][:3]:
            report += f"  • {catalyst['description']} (Timeline: {catalyst.get('timeline', 'near-term')})\n"
        
        # Add risk/reward
        rr = bull_data['risk_reward']
        report += f"""
RISK/REWARD ANALYSIS:
  Upside Potential: {rr['upside_potential']}
  Downside Risk: {rr['downside_risk']}
  Reward/Risk Ratio: {rr['reward_risk_ratio']:.1f}x
  
BULL CASE STRENGTH: {rr['conviction_level']} - Confidence: {rr['conviction_level']}
"""
        return report
    
    def research(self, discussion_points):
        """Main research function"""
        if not discussion_points:
            return "Error: No discussion points available"
        
        print(f"Building bull case for {self.ticker}...")
        
        # Analyze components
        evidence_categories = self.analyze_bull_signals(discussion_points)
        rebuttals = self.counter_bear_arguments(discussion_points)
        catalysts = self.identify_catalysts(discussion_points)
        risk_reward = self.calculate_risk_reward(discussion_points)
        
        # Build thesis
        core_thesis = self.build_bull_thesis(evidence_categories, risk_reward)
        
        # Compile bull data
        bull_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'core_thesis': core_thesis,
            'evidence': evidence_categories,
            'bear_rebuttals': rebuttals,
            'catalysts': catalysts,
            'risk_reward': risk_reward,
            'discussion_summary': discussion_points.get('summary', {})
        }
        
        # Store for later use
        self.bull_thesis = bull_data
        
        # Generate report with LLM
        report = self.synthesize_with_llm(bull_data)
        
        return report
    
    def save_thesis(self, filepath):
        """Save bull thesis data for other agents"""
        with open(filepath, 'w') as f:
            json.dump(self.bull_thesis, f, indent=2)
        print(f"Saved bull thesis data to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Bull Researcher - Builds bullish investment case")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--discussion-file", 
                       default="../orchestrators/discussion_points.json",
                       help="Path to discussion points JSON file")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-data", help="Save thesis data as JSON")
    
    args = parser.parse_args()
    
    researcher = BullResearcher(args.ticker, args.api_key, args.model)
    
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