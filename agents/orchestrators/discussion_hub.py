"""
Discussion Hub - Aggregates and structures analyst reports for research team
Usage: python discussion_hub.py AAPL --run-analysts --output discussion_points.json
"""

import os
import sys
import json
import re
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from openai import OpenAI

class DiscussionHub:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are a discussion facilitator that synthesizes analyst reports.
        Extract key bullish and bearish points, identify conflicts between analysts, and find consensus.
        Structure the information for debate between bull and bear researchers.
        Be objective and balanced in your assessment."""
        
        # Store analyst reports
        self.analyst_reports = {}
        self.recommendations = {}
        self.confidence_levels = {}
        
        # Auto-detect paths
        self.setup_paths()
    
    def setup_paths(self):
        """Set up paths for the fixed folder structure"""
        # Fixed structure: discussion_hub.py is in orchestrators folder
        # Analyst agents are in sibling analyst folder
        self.analyst_configs = {
            'technical': '../analyst/technical_agent.py',
            'news': '../analyst/news_agent.py',
            'fundamental': '../analyst/fundamental_agent.py',
            'macro': '../analyst/macro_agent.py'
        }
        
        # Quick check if paths exist
        if os.path.exists(self.analyst_configs['technical']):
            print(f"Found analyst agents in ../analyst/")
        else:
            print(f"Warning: Could not find agents at ../analyst/")
            print(f"Current directory: {os.getcwd()}")
            print(f"Looking for: {os.path.abspath(self.analyst_configs['technical'])}")
    
    def run_analyst(self, agent_name, agent_script):
        """Run an individual analyst agent"""
        print(f"  Running {agent_name} analyst...")
        
        cmd = ["python", agent_script]
        if agent_name != "macro":  # Macro doesn't need ticker
            cmd.append(self.ticker)
        
        # Add default parameters for consistency
        if agent_name == "technical":
            cmd.extend(["--days", "7"])
        elif agent_name == "news":
            cmd.extend(["--sources", "yahoo", "--days", "7"])
        elif agent_name == "macro":
            cmd.extend(["--days", "7"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"    Warning: {agent_name} returned error: {result.stderr}")
                return f"Error running {agent_name}: {result.stderr}"
        except subprocess.TimeoutExpired:
            return f"{agent_name} analyst timed out"
        except FileNotFoundError:
            return f"{agent_script} not found"
        except Exception as e:
            return f"Error running {agent_name}: {str(e)}"
    
    def extract_recommendation(self, report):
        """Extract recommendation and confidence from report"""
        recommendation = "HOLD"
        confidence = "Low"
        
        # Look for RECOMMENDATION: pattern
        rec_pattern = r"RECOMMENDATION:\s*(\w+)"
        conf_pattern = r"Confidence:\s*(\w+)"
        
        rec_match = re.search(rec_pattern, report, re.IGNORECASE)
        if rec_match:
            rec = rec_match.group(1).upper()
            if "BUY" in rec:
                recommendation = "BUY"
            elif "SELL" in rec:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
        
        conf_match = re.search(conf_pattern, report, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).capitalize()
        
        return recommendation, confidence
    
    def extract_key_points(self, report, analyst_type):
        """Extract key bullish and bearish points from a report"""
        bullish_signals = []
        bearish_signals = []
        
        # Keywords to identify sentiment
        bullish_keywords = [
            'bullish', 'buy', 'upside', 'growth', 'strong', 'positive',
            'outperform', 'upgrade', 'momentum', 'breakout', 'support',
            'oversold', 'undervalued', 'beat', 'exceed'
        ]
        
        bearish_keywords = [
            'bearish', 'sell', 'downside', 'risk', 'weak', 'negative',
            'underperform', 'downgrade', 'resistance', 'overbought',
            'overvalued', 'miss', 'concern', 'deteriorat'
        ]
        
        # Split report into sentences
        sentences = report.replace('\n', '. ').split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for bullish signals
            if any(keyword in sentence_lower for keyword in bullish_keywords):
                # Clean and add if meaningful
                cleaned = sentence.strip()
                if len(cleaned) > 20 and len(cleaned) < 200:
                    bullish_signals.append({
                        'source': analyst_type,
                        'signal': cleaned
                    })
            
            # Check for bearish signals
            if any(keyword in sentence_lower for keyword in bearish_keywords):
                cleaned = sentence.strip()
                if len(cleaned) > 20 and len(cleaned) < 200:
                    bearish_signals.append({
                        'source': analyst_type,
                        'signal': cleaned
                    })
        
        return bullish_signals, bearish_signals
    
    def identify_conflicts(self):
        """Identify conflicts between analyst recommendations"""
        conflicts = []
        
        # Check for disagreements in recommendations
        rec_list = list(self.recommendations.values())
        if len(set(rec_list)) > 1:  # Not all same recommendation
            # Find which analysts disagree
            for analyst1, rec1 in self.recommendations.items():
                for analyst2, rec2 in self.recommendations.items():
                    if analyst1 < analyst2 and rec1 != rec2:
                        conflicts.append({
                            'type': 'recommendation_conflict',
                            'analysts': [analyst1, analyst2],
                            'positions': {analyst1: rec1, analyst2: rec2},
                            'description': f"{analyst1} says {rec1} while {analyst2} says {rec2}"
                        })
        
        return conflicts
    
    def find_consensus(self):
        """Find points where analysts agree"""
        consensus_points = []
        
        # Check if majority agree on direction
        rec_counts = {}
        for rec in self.recommendations.values():
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        if rec_counts:
            majority_rec = max(rec_counts, key=rec_counts.get)
            if rec_counts[majority_rec] >= 3:  # At least 3 out of 4 agree
                consensus_points.append({
                    'type': 'recommendation_consensus',
                    'value': majority_rec,
                    'strength': f"{rec_counts[majority_rec]}/4 analysts agree",
                    'description': f"Majority recommendation is {majority_rec}"
                })
        
        # Check for high confidence agreements
        high_conf_analysts = [a for a, c in self.confidence_levels.items() if c == "High"]
        if len(high_conf_analysts) >= 2:
            consensus_points.append({
                'type': 'confidence_consensus',
                'analysts': high_conf_analysts,
                'description': f"Multiple analysts show high confidence: {', '.join(high_conf_analysts)}"
            })
        
        return consensus_points
    
    def synthesize_with_llm(self, discussion_data):
        """Use LLM to create structured discussion points"""
        if not self.client:
            return discussion_data  # Return raw data if no LLM
        
        try:
            prompt = f"""Analyze these discussion points for {self.ticker} and create a structured debate framework.
            
Discussion Data:
{json.dumps(discussion_data, indent=2)}

Create a balanced assessment that:
1. Summarizes the strongest bull case
2. Summarizes the strongest bear case  
3. Identifies key decision factors
4. Highlights areas needing deeper research
5. Notes any critical time-sensitive factors"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Add LLM synthesis to discussion data
            discussion_data['llm_synthesis'] = response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM synthesis error: {e}")
            discussion_data['llm_synthesis'] = "LLM synthesis unavailable"
        
        return discussion_data
    
    def aggregate_reports(self, reports=None, run_analysts=False):
        """Main aggregation function"""
        
        # Step 1: Get analyst reports
        if run_analysts:
            print(f"\nRunning analysts for {self.ticker}...")
            
            # Use auto-detected paths
            for name, script in self.analyst_configs.items():
                if os.path.exists(script):
                    report = self.run_analyst(name, script)
                    self.analyst_reports[name] = report
                    
                    # Extract recommendation and confidence
                    rec, conf = self.extract_recommendation(report)
                    self.recommendations[name] = rec
                    self.confidence_levels[name] = conf
                    print(f"    {name}: {rec} (Confidence: {conf})")
                else:
                    print(f"    Warning: {script} not found")
        
        elif reports:
            # Use provided reports
            self.analyst_reports = reports
            for name, report in reports.items():
                rec, conf = self.extract_recommendation(report)
                self.recommendations[name] = rec
                self.confidence_levels[name] = conf
        
        # Step 2: Extract signals from each report
        print("\nExtracting signals...")
        all_bullish = []
        all_bearish = []
        
        for analyst_type, report in self.analyst_reports.items():
            bullish, bearish = self.extract_key_points(report, analyst_type)
            all_bullish.extend(bullish)
            all_bearish.extend(bearish)
        
        # Step 3: Identify conflicts and consensus
        conflicts = self.identify_conflicts()
        consensus = self.find_consensus()
        
        # Step 4: Structure discussion points
        discussion_points = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'recommendations': self.recommendations,
                'confidence_levels': self.confidence_levels,
                'bull_signal_count': len(all_bullish),
                'bear_signal_count': len(all_bearish)
            },
            'bull_evidence': all_bullish[:10],  # Top 10 bull points
            'bear_evidence': all_bearish[:10],  # Top 10 bear points
            'key_conflicts': conflicts,
            'consensus_points': consensus,
            'analyst_reports_summary': {
                name: {
                    'recommendation': self.recommendations.get(name, 'N/A'),
                    'confidence': self.confidence_levels.get(name, 'N/A'),
                    'report_length': len(report),
                    'has_data': len(report) > 100
                }
                for name, report in self.analyst_reports.items()
            },
            'research_priorities': self.identify_research_priorities(all_bullish, all_bearish, conflicts)
        }
        
        # Step 5: Optional LLM synthesis
        discussion_points = self.synthesize_with_llm(discussion_points)
        
        return discussion_points
    
    def identify_research_priorities(self, bullish_signals, bearish_signals, conflicts):
        """Identify what researchers should focus on"""
        priorities = []
        
        # Priority 1: Resolve major conflicts
        if conflicts:
            priorities.append({
                'priority': 'HIGH',
                'focus': 'conflict_resolution',
                'description': f"Resolve {len(conflicts)} analyst conflicts",
                'details': conflicts[0] if conflicts else None
            })
        
        # Priority 2: Investigate dominant signals
        if len(bullish_signals) > len(bearish_signals) * 2:
            priorities.append({
                'priority': 'MEDIUM',
                'focus': 'validate_bull_thesis',
                'description': 'Strong bullish bias needs validation',
                'signal_ratio': f"{len(bullish_signals)}:{len(bearish_signals)}"
            })
        elif len(bearish_signals) > len(bullish_signals) * 2:
            priorities.append({
                'priority': 'MEDIUM', 
                'focus': 'validate_bear_thesis',
                'description': 'Strong bearish bias needs validation',
                'signal_ratio': f"{len(bullish_signals)}:{len(bearish_signals)}"
            })
        
        # Priority 3: Time-sensitive factors
        for signal in bullish_signals + bearish_signals:
            if any(word in signal.get('signal', '').lower() 
                   for word in ['earnings', 'announcement', 'tomorrow', 'today', 'imminent']):
                priorities.append({
                    'priority': 'URGENT',
                    'focus': 'time_sensitive',
                    'description': 'Time-sensitive event detected',
                    'signal': signal
                })
                break
        
        return priorities[:3]  # Return top 3 priorities
    
    def format_report(self, discussion_points):
        """Format discussion points as readable report"""
        report = f"""
================================================================================
                        DISCUSSION HUB ANALYSIS
================================================================================
Ticker: {discussion_points['ticker']}
Time: {discussion_points['timestamp']}

ANALYST CONSENSUS
--------------------------------------------------------------------------------
"""
        # Add recommendations
        for analyst, rec in discussion_points['summary']['recommendations'].items():
            conf = discussion_points['summary']['confidence_levels'].get(analyst, 'N/A')
            report += f"  {analyst:12} -> {rec:5} (Confidence: {conf})\n"
        
        report += f"""
SIGNAL SUMMARY
--------------------------------------------------------------------------------
  Bullish Signals: {discussion_points['summary']['bull_signal_count']}
  Bearish Signals: {discussion_points['summary']['bear_signal_count']}

KEY CONFLICTS
--------------------------------------------------------------------------------
"""
        if discussion_points['key_conflicts']:
            for conflict in discussion_points['key_conflicts']:
                report += f"  • {conflict['description']}\n"
        else:
            report += "  • No major conflicts identified\n"
        
        report += """
CONSENSUS POINTS
--------------------------------------------------------------------------------
"""
        if discussion_points['consensus_points']:
            for consensus in discussion_points['consensus_points']:
                report += f"  • {consensus['description']}\n"
        else:
            report += "  • No strong consensus found\n"
        
        report += """
RESEARCH PRIORITIES
--------------------------------------------------------------------------------
"""
        for priority in discussion_points['research_priorities']:
            report += f"  [{priority['priority']}] {priority['description']}\n"
        
        # Add LLM synthesis if available
        if 'llm_synthesis' in discussion_points:
            report += f"""
AI SYNTHESIS
--------------------------------------------------------------------------------
{discussion_points['llm_synthesis']}
"""
        
        report += """
================================================================================
"""
        return report

def main():
    parser = argparse.ArgumentParser(description="Discussion Hub - Aggregates analyst reports")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--run-analysts", action="store_true",
                       help="Run analyst agents before aggregating")
    parser.add_argument("--api-key", help="OpenAI API key for synthesis")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file (JSON format)")
    parser.add_argument("--format", choices=['json', 'text'], default='text',
                       help="Output format")
    
    args = parser.parse_args()
    
    hub = DiscussionHub(args.ticker, args.api_key, args.model)
    
    # Run aggregation
    discussion_points = hub.aggregate_reports(run_analysts=args.run_analysts)
    
    # Output results
    if args.format == 'json':
        output = json.dumps(discussion_points, indent=2)
        print(output)
    else:
        output = hub.format_report(discussion_points)
        print(output)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.output.endswith('.json'):
                json.dump(discussion_points, f, indent=2)
            else:
                f.write(output if args.format == 'text' else json.dumps(discussion_points, indent=2))
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()