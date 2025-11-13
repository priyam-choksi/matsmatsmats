"""
Discussion Hub - Enhanced with Full Report Preservation
Aggregates analyst reports while preserving complete context for downstream agents

Usage: python discussion_hub.py AAPL --run-analysts --output discussion_points.json
"""

import os
import sys
import json
import re
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from openai import OpenAI

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class DiscussionHub:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Enhanced system prompt for synthesis
        self.system_prompt = """You are a research coordinator synthesizing multiple analyst perspectives.

**YOUR ROLE:**
You have access to complete reports from 4 analysts (technical, fundamental, news, macro). Create a synthesis that:

1. **Executive Summary (1-2 paragraphs):**
   - What's the overall picture? Bullish, bearish, or mixed?
   - What are the 2-3 most important factors driving the decision?

2. **Bull Case (3-5 key points):**
   - Strongest reasons to buy from ALL analysts
   - Reference which analyst provided each point

3. **Bear Case (3-5 key points):**
   - Biggest risks or reasons to sell from ALL analysts
   - Reference which analyst flagged each concern

4. **Key Conflicts:**
   - Where do analysts disagree and why?
   - Which conflict is most important to resolve?

5. **Research Priorities:**
   - What needs deeper investigation?
   - Any time-sensitive catalysts?

Be balanced and objective. Your synthesis will guide the bull and bear researchers."""
        
        # Storage
        self.analyst_reports = {}
        self.recommendations = {}
        self.confidence_levels = {}
        
        # Setup paths
        self.setup_paths()
    
    def setup_paths(self):
        """Detect analyst agent locations"""
        self.analyst_configs = {
            'technical': '../analyst/technical_agent.py',
            'news': '../analyst/news_agent.py',
            'fundamental': '../analyst/fundamental_agent.py',
            'macro': '../analyst/macro_agent.py'
        }
        
        # Validate
        found = sum(1 for path in self.analyst_configs.values() if os.path.exists(path))
        print(f"[HUB] Found {found}/4 analyst agents")
    
    def run_analyst(self, agent_name: str, agent_script: str) -> str:
        """Run individual analyst via subprocess"""
        print(f"[HUB] 🔧 Running {agent_name}...")
        
        cmd = ["python", agent_script]
        
        if agent_name != "macro":
            cmd.append(self.ticker)
        
        # Standard parameters
        if agent_name == "technical":
            cmd.extend(["--days", "7"])
        elif agent_name == "news":
            cmd.extend(["--sources", "yahoo", "--days", "7"])
        elif agent_name == "macro":
            cmd.extend(["--days", "7"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
                cwd=Path(agent_script).parent,
                encoding='utf-8'

            )
            
            if result.returncode == 0:
                print(f"[HUB] ✓ {agent_name} completed ({len(result.stdout)} chars)")
                return result.stdout
            else:
                print(f"[HUB] ⚠️  {agent_name} error: {result.stderr[:100]}")
                return f"# {agent_name.upper()} ANALYSIS ERROR\n\n{result.stderr}\n\nRECOMMENDATION: HOLD - Confidence: Low"
                
        except subprocess.TimeoutExpired:
            print(f"[HUB] ⚠️  {agent_name} timeout")
            return f"# {agent_name.upper()} ANALYSIS TIMEOUT\n\nRECOMMENDATION: HOLD - Confidence: Low"
        except FileNotFoundError:
            print(f"[HUB] ❌ {agent_script} not found")
            return f"# {agent_name.upper()} NOT FOUND\n\nRECOMMENDATION: HOLD - Confidence: Low"
        except Exception as e:
            print(f"[HUB] ❌ {agent_name} error: {str(e)}")
            return f"# {agent_name.upper()} ERROR\n\n{str(e)}\n\nRECOMMENDATION: HOLD - Confidence: Low"
    
    def extract_recommendation(self, report: str) -> Tuple[str, str]:
        """Extract recommendation and confidence"""
        recommendation = "HOLD"
        confidence = "Low"
        
        rec_pattern = r"RECOMMENDATION:\s*(\w+(?:\s+\w+)?)"
        conf_pattern = r"Confidence:\s*(\w+)"
        
        rec_match = re.search(rec_pattern, report, re.IGNORECASE)
        if rec_match:
            rec = rec_match.group(1).upper()
            if "BUY" in rec:
                recommendation = "BUY"
            elif "SELL" in rec:
                recommendation = "SELL"
            elif "RISK-ON" in rec or "RISK_ON" in rec:
                recommendation = "BUY"
            elif "RISK-OFF" in rec or "RISK_OFF" in rec:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
        
        conf_match = re.search(conf_pattern, report, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).capitalize()
        
        return recommendation, confidence
    
    def create_analyst_summary(self, report: str, analyst_type: str) -> str:
        """
        Create concise 1-2 sentence summary of analyst report
        For quick scanning without reading full report
        """
        rec, conf = self.extract_recommendation(report)
        
        # Extract key metrics/findings based on analyst type
        key_finding = ""
        
        if analyst_type == "technical":
            # Look for trend and RSI
            if "uptrend" in report.lower():
                key_finding = "Uptrend confirmed"
            elif "downtrend" in report.lower():
                key_finding = "Downtrend confirmed"
            else:
                key_finding = "Sideways action"
            
            rsi_match = re.search(r"RSI.*?(\d+\.?\d*)", report)
            if rsi_match:
                rsi = float(rsi_match.group(1))
                if rsi < 30:
                    key_finding += ", oversold"
                elif rsi > 70:
                    key_finding += ", overbought"
        
        elif analyst_type == "fundamental":
            # Look for growth and valuation
            if "growth" in report.lower():
                key_finding = "Growth present"
            if "undervalued" in report.lower():
                key_finding += ", undervalued"
            elif "overvalued" in report.lower():
                key_finding += ", overvalued"
        
        elif analyst_type == "news":
            # Look for sentiment
            if "bullish" in report.lower():
                key_finding = "Bullish sentiment"
            elif "bearish" in report.lower():
                key_finding = "Bearish sentiment"
            else:
                key_finding = "Neutral sentiment"
        
        elif analyst_type == "macro":
            # Look for risk environment
            if "risk-on" in report.lower():
                key_finding = "Risk-on environment"
            elif "risk-off" in report.lower():
                key_finding = "Risk-off environment"
            else:
                key_finding = "Neutral macro"
        
        if not key_finding:
            key_finding = "Analysis complete"
        
        return f"{key_finding}. {rec} - {conf} confidence"
    
    def extract_key_points(self, report: str, analyst_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract bullish and bearish signals with scoring"""
        bullish_signals = []
        bearish_signals = []
        
        bullish_keywords = [
            'bullish', 'buy', 'upside', 'growth', 'strong', 'positive',
            'outperform', 'upgrade', 'momentum', 'breakout', 'support',
            'oversold', 'undervalued', 'beat', 'exceed', 'improving',
            'expansion', 'accelerating', 'strength'
        ]
        
        bearish_keywords = [
            'bearish', 'sell', 'downside', 'risk', 'weak', 'negative',
            'underperform', 'downgrade', 'resistance', 'overbought',
            'overvalued', 'miss', 'concern', 'deteriorat', 'decline',
            'contraction', 'slowing', 'weakness'
        ]
        
        # Split into sentences
        sentences = report.replace('\n', '. ').split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            cleaned = sentence.strip()
            
            # Filter length
            if len(cleaned) < 20 or len(cleaned) > 300:
                continue
            
            # Count keyword matches (for scoring)
            bullish_count = sum(1 for kw in bullish_keywords if kw in sentence_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in sentence_lower)
            
            # Classify signal
            if bullish_count > bearish_count and bullish_count >= 1:
                bullish_signals.append({
                    'source': analyst_type,
                    'signal': cleaned,
                    'strength': bullish_count,
                    'keywords_matched': bullish_count
                })
            elif bearish_count > bullish_count and bearish_count >= 1:
                bearish_signals.append({
                    'source': analyst_type,
                    'signal': cleaned,
                    'strength': bearish_count,
                    'keywords_matched': bearish_count
                })
        
        # Sort by strength (strongest signals first)
        bullish_signals.sort(key=lambda x: x['strength'], reverse=True)
        bearish_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        return bullish_signals[:20], bearish_signals[:20]
    
    def identify_conflicts(self) -> List[Dict]:
        """Identify and categorize conflicts"""
        conflicts = []
        
        rec_list = list(self.recommendations.values())
        
        if len(set(rec_list)) > 1:
            for analyst1, rec1 in self.recommendations.items():
                for analyst2, rec2 in self.recommendations.items():
                    if analyst1 < analyst2 and rec1 != rec2:
                        # Categorize conflict severity
                        if (rec1 == 'BUY' and rec2 == 'SELL') or (rec1 == 'SELL' and rec2 == 'BUY'):
                            severity = 'CRITICAL'
                        elif 'HOLD' in [rec1, rec2]:
                            severity = 'MINOR'
                        else:
                            severity = 'MODERATE'
                        
                        conflicts.append({
                            'type': 'recommendation_conflict',
                            'severity': severity,
                            'analysts': [analyst1, analyst2],
                            'positions': {analyst1: rec1, analyst2: rec2},
                            'description': f"{analyst1} says {rec1} while {analyst2} says {rec2}"
                        })
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'MODERATE': 1, 'MINOR': 2}
        conflicts.sort(key=lambda x: severity_order.get(x['severity'], 99))
        
        return conflicts
    
    def find_consensus(self) -> List[Dict]:
        """Find agreement points"""
        consensus_points = []
        
        rec_counts = {}
        for rec in self.recommendations.values():
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        if rec_counts:
            majority_rec = max(rec_counts, key=rec_counts.get)
            agreement_count = rec_counts[majority_rec]
            
            if agreement_count >= 3:
                consensus_points.append({
                    'type': 'strong_consensus',
                    'value': majority_rec,
                    'strength': f"{agreement_count}/4",
                    'description': f"Strong consensus: {majority_rec} ({agreement_count}/4 analysts)"
                })
            elif agreement_count == 2 and len(rec_counts) == 2:
                consensus_points.append({
                    'type': 'split_decision',
                    'value': 'MIXED',
                    'strength': '2/4 vs 2/4',
                    'description': f"Split: {' vs '.join(rec_counts.keys())}"
                })
        
        # High confidence agreements
        high_conf = [a for a, c in self.confidence_levels.items() if c == "High"]
        if len(high_conf) >= 2:
            high_recs = [self.recommendations[a] for a in high_conf]
            if len(set(high_recs)) == 1:
                consensus_points.append({
                    'type': 'high_confidence_agreement',
                    'analysts': high_conf,
                    'value': high_recs[0],
                    'description': f"High-confidence agreement: {', '.join(high_conf)} → {high_recs[0]}"
                })
        
        return consensus_points
    
    def identify_research_priorities(self, bullish: List, bearish: List, conflicts: List) -> List[Dict]:
        """Determine research priorities"""
        priorities = []
        
        # Critical conflicts
        critical_conflicts = [c for c in conflicts if c.get('severity') == 'CRITICAL']
        if critical_conflicts:
            priorities.append({
                'priority': 'CRITICAL',
                'focus': 'resolve_major_conflict',
                'description': f"BUY vs SELL conflict - {critical_conflicts[0]['description']}",
                'action': 'Deep dive needed to determine which analyst is correct'
            })
        
        # Strong directional bias
        bull_count = len(bullish)
        bear_count = len(bearish)
        
        if bull_count > bear_count * 2.5:
            priorities.append({
                'priority': 'HIGH',
                'focus': 'validate_bull_thesis',
                'description': f'Strong bullish bias ({bull_count} vs {bear_count} signals)',
                'action': 'Validate assumptions - could be herd mentality or overlooked risks'
            })
        elif bear_count > bull_count * 2.5:
            priorities.append({
                'priority': 'HIGH',
                'focus': 'validate_bear_thesis',
                'description': f'Strong bearish bias ({bear_count} vs {bull_count} signals)',
                'action': 'Validate assumptions - justified fear or oversold opportunity?'
            })
        elif abs(bull_count - bear_count) <= 5:
            priorities.append({
                'priority': 'MEDIUM',
                'focus': 'break_tie',
                'description': f'Balanced signals ({bull_count} vs {bear_count})',
                'action': 'Need deeper analysis to determine edge'
            })
        
        # Time-sensitive events
        time_keywords = ['earnings', 'announcement', 'tomorrow', 'today', 'imminent', 'upcoming', 'next week']
        
        for signal in bullish + bearish:
            signal_text = signal.get('signal', '').lower()
            if any(kw in signal_text for kw in time_keywords):
                priorities.append({
                    'priority': 'URGENT',
                    'focus': 'time_sensitive_event',
                    'description': 'Upcoming catalyst detected',
                    'signal': signal.get('signal', ''),
                    'source': signal.get('source', ''),
                    'action': 'Research before event occurs'
                })
                break
        
        return priorities[:5]
    
    def synthesize_with_llm(self, discussion_data: Dict) -> Dict:
        """
        Create intelligent synthesis using FULL analyst reports
        This is the key improvement - LLM sees complete context
        """
        if not self.client:
            print("[HUB] ⚠️  No API key - skipping synthesis")
            discussion_data['llm_synthesis'] = "LLM synthesis unavailable (no API key)"
            return discussion_data
        
        try:
            print(f"[HUB] Generating synthesis from full reports...")
            
            # Prepare FULL context for LLM (this is the key change!)
            full_context = f"""# Complete Analyst Reports for {self.ticker}

## Quick Summary
- Recommendations: {discussion_data['summary']['recommendations']}
- Bullish signals: {discussion_data['summary']['bull_signal_count']}
- Bearish signals: {discussion_data['summary']['bear_signal_count']}

## Full Analyst Reports

### Technical Analysis
{discussion_data['full_analyst_reports'].get('technical', 'Not available')}

---

### Fundamental Analysis
{discussion_data['full_analyst_reports'].get('fundamental', 'Not available')}

---

### News & Sentiment
{discussion_data['full_analyst_reports'].get('news', 'Not available')}

---

### Macro Environment
{discussion_data['full_analyst_reports'].get('macro', 'Not available')}

---

## Identified Conflicts
{json.dumps(discussion_data['key_conflicts'], indent=2)}

## Consensus Points
{json.dumps(discussion_data['consensus_points'], indent=2)}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Synthesize these complete analyst reports:\n\n{full_context}"}
                ],
                temperature=0.7,
                max_tokens=2500  # Increased for comprehensive synthesis
            )
            
            discussion_data['llm_synthesis'] = response.choices[0].message.content
            print(f"[HUB] ✓ Synthesis complete ({len(response.choices[0].message.content)} chars)")
            
        except Exception as e:
            print(f"[HUB] ⚠️  Synthesis error: {e}")
            discussion_data['llm_synthesis'] = f"Synthesis error: {str(e)}"
        
        return discussion_data
    
    def aggregate_reports(self, reports: Optional[Dict] = None, run_analysts: bool = False) -> Dict:
        """
        Main aggregation workflow - NOW PRESERVES FULL REPORTS
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"DISCUSSION HUB: {self.ticker}")
        print(f"{'='*70}\n")
        
        # Step 1: Get analyst reports
        if run_analysts:
            print(f"[HUB] Running all analysts...\n")
            
            for name, script in self.analyst_configs.items():
                if os.path.exists(script):
                    report = self.run_analyst(name, script)
                    self.analyst_reports[name] = report
                    
                    rec, conf = self.extract_recommendation(report)
                    self.recommendations[name] = rec
                    self.confidence_levels[name] = conf
                else:
                    print(f"[HUB] ⚠️  Skipping {name} (not found)")
            
            print()
            
        elif reports:
            print(f"[HUB] Using provided reports...")
            self.analyst_reports = reports
            for name, report in reports.items():
                rec, conf = self.extract_recommendation(report)
                self.recommendations[name] = rec
                self.confidence_levels[name] = conf
        else:
            print(f"[HUB] ❌ No reports available")
            return {
                'ticker': self.ticker,
                'error': 'No analyst reports',
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: Create concise summaries for quick reference
        print(f"[HUB] Creating analyst summaries...")
        analyst_summaries = {}
        for name, report in self.analyst_reports.items():
            analyst_summaries[name] = self.create_analyst_summary(report, name)
        print(f"[HUB] ✓ Summaries created\n")
        
        # Step 3: Extract signals
        print(f"[HUB] Extracting signals...")
        all_bullish = []
        all_bearish = []
        
        for analyst_type, report in self.analyst_reports.items():
            bullish, bearish = self.extract_key_points(report, analyst_type)
            all_bullish.extend(bullish)
            all_bearish.extend(bearish)
            print(f"[HUB]   {analyst_type}: {len(bullish)} bull, {len(bearish)} bear")
        
        print(f"[HUB] ✓ Total: {len(all_bullish)} bullish, {len(all_bearish)} bearish\n")
        
        # Step 4: Conflicts and consensus
        print(f"[HUB] Analyzing consensus...")
        conflicts = self.identify_conflicts()
        consensus = self.find_consensus()
        print(f"[HUB] ✓ {len(conflicts)} conflicts, {len(consensus)} consensus points\n")
        
        # Step 5: Research priorities
        print(f"[HUB] Identifying priorities...")
        priorities = self.identify_research_priorities(all_bullish, all_bearish, conflicts)
        print(f"[HUB] ✓ {len(priorities)} priorities identified\n")
        
        # Step 6: Structure comprehensive output
        discussion_points = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            
            # Quick reference summary
            'summary': {
                'recommendations': self.recommendations,
                'confidence_levels': self.confidence_levels,
                'bull_signal_count': len(all_bullish),
                'bear_signal_count': len(all_bearish),
                'analyst_count': len(self.analyst_reports),
                'net_sentiment': 'BULLISH' if len(all_bullish) > len(all_bearish) else 'BEARISH' if len(all_bearish) > len(all_bullish) else 'NEUTRAL'
            },
            
            # ONE-LINE SUMMARIES for quick scanning
            'analyst_summaries': analyst_summaries,
            
            # FULL REPORTS for deep analysis (THIS IS NEW!)
            'full_analyst_reports': self.analyst_reports,
            
            # Extracted signals (organized)
            'bull_evidence': all_bullish[:20],
            'bear_evidence': all_bearish[:20],
            
            # Analysis
            'key_conflicts': conflicts,
            'consensus_points': consensus,
            'research_priorities': priorities,
            
            # Metadata
            'analyst_reports_summary': {
                name: {
                    'recommendation': self.recommendations.get(name, 'N/A'),
                    'confidence': self.confidence_levels.get(name, 'N/A'),
                    'report_length': len(report),
                    'has_data': len(report) > 100,
                    'completed_successfully': 'Error' not in report[:100] and 'RECOMMENDATION:' in report
                }
                for name, report in self.analyst_reports.items()
            }
        }
        
        # Step 7: LLM synthesis (uses full reports!)
        print(f"[HUB] Generating comprehensive synthesis...")
        discussion_points = self.synthesize_with_llm(discussion_points)
        
        elapsed = time.time() - start_time
        print(f"\n[HUB] ✓ Aggregation complete in {elapsed:.2f}s")
        
        # Report what we captured
        total_chars = sum(len(r) for r in self.analyst_reports.values())
        print(f"[HUB] 📊 Captured {total_chars:,} total characters from analysts")
        print(f"{'='*70}\n")
        
        return discussion_points
    
    def format_report(self, discussion_points: Dict) -> str:
        """Format as readable text report"""
        dp = discussion_points
        
        report = f"""
{'='*80}
                        DISCUSSION HUB ANALYSIS
{'='*80}
Ticker: {dp['ticker']}
Timestamp: {dp['timestamp']}
Net Sentiment: {dp['summary']['net_sentiment']}

ANALYST QUICK SUMMARIES
{'-'*80}
"""
        for analyst, summary in dp.get('analyst_summaries', {}).items():
            rec = dp['summary']['recommendations'].get(analyst, 'N/A')
            icon = "✓" if rec in ['BUY', 'SELL'] else "○"
            report += f"  {icon} {analyst:12} | {summary}\n"
        
        report += f"""
CONSENSUS & CONFLICTS
{'-'*80}
"""
        if dp['consensus_points']:
            report += "Consensus:\n"
            for consensus in dp['consensus_points']:
                report += f"  ✓ {consensus['description']}\n"
        
        if dp['key_conflicts']:
            report += "\nConflicts:\n"
            for conflict in dp['key_conflicts']:
                severity_icon = "🔴" if conflict['severity'] == 'CRITICAL' else "🟡" if conflict['severity'] == 'MODERATE' else "⚪"
                report += f"  {severity_icon} {conflict['description']}\n"
        
        report += f"""
SIGNAL BREAKDOWN
{'-'*80}
  Bullish Signals: {dp['summary']['bull_signal_count']}
  Bearish Signals: {dp['summary']['bear_signal_count']}

RESEARCH PRIORITIES
{'-'*80}
"""
        for priority in dp.get('research_priorities', []):
            priority_icon = "🔥" if priority['priority'] == 'URGENT' else "⚠️" if priority['priority'] == 'CRITICAL' else "📊"
            report += f"  {priority_icon} [{priority['priority']}] {priority['description']}\n"
            if 'action' in priority:
                report += f"      → {priority['action']}\n"
        
        if 'llm_synthesis' in dp and 'unavailable' not in dp['llm_synthesis']:
            report += f"""
AI SYNTHESIS (Based on Full Reports)
{'-'*80}
{dp['llm_synthesis']}
"""
        
        report += f"""
DATA CAPTURED
{'-'*80}
  Full Reports: {len(dp.get('full_analyst_reports', {}))} analysts
  Total Size: {sum(len(r) for r in dp.get('full_analyst_reports', {}).values()):,} characters
  Extracted Signals: {len(dp.get('bull_evidence', []))} bull + {len(dp.get('bear_evidence', []))} bear

{'='*80}
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Discussion Hub - Aggregates analyst reports with full context preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discussion_hub.py AAPL --run-analysts
  python discussion_hub.py MSFT --run-analysts --output discussion.json
  python discussion_hub.py GOOGL --run-analysts --format text --output report.txt

Output Format:
  The JSON output contains:
  - analyst_summaries: One-line summaries for quick scanning
  - full_analyst_reports: Complete original reports (NEW!)
  - bull_evidence/bear_evidence: Extracted signals
  - llm_synthesis: AI synthesis using full context (NEW!)
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--run-analysts", action="store_true",
                       help="Run all 4 analyst agents")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--format", choices=['json', 'text'], default='text',
                       help="Output format")
    
    args = parser.parse_args()
    
    try:
        hub = DiscussionHub(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
        # Run aggregation
        discussion_points = hub.aggregate_reports(run_analysts=args.run_analysts)
        
        # Format output
        if args.format == 'json' or (args.output and args.output.endswith('.json')):
            output = json.dumps(discussion_points, indent=2)
        else:
            output = hub.format_report(discussion_points)
        
        print(output)
        
        # Save
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.output.endswith('.json'):
                    json.dump(discussion_points, f, indent=2)
                else:
                    f.write(output)
            print(f"\n✓ Saved to {args.output}")
        
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