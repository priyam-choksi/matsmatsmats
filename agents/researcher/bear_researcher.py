"""
Bear Researcher - Clean Single-Pass Analysis
Builds comprehensive bearish case from analyst discussion points
Debate orchestration handled by research_manager.py

MODIFIED: Now supports historical backtesting via analysis_date parameter

Usage: 
  python bear_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json
  python bear_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json --analysis-date 2024-06-15
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


class BearResearcher:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):  # <-- NEW PARAMETER
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # === HISTORICAL BACKTESTING SUPPORT ===
        self.analysis_date = analysis_date  # Format: 'YYYY-MM-DD' or None for current
        
        if self.analysis_date:
            print(f"[BEAR] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[BEAR] Running in LIVE mode (current data)")
        
        self.system_prompt = """You are a Bear Analyst building the strongest possible case AGAINST investing in this stock.

**YOUR MISSION:**
Create a compelling, data-driven risk assessment emphasizing dangers, weaknesses, and downside catalysts.

**ANALYSIS REQUIREMENTS:**

1. **Technical Analysis:**
   - Overbought/overextended conditions
   - Key resistance levels and breakdown risks
   - Negative momentum divergences
   - Volume concerns

2. **Fundamental Analysis:**
   - Overvaluation metrics vs peers/history
   - Growth deceleration signals
   - Margin pressure or deterioration
   - Balance sheet concerns (debt, cash burn)

3. **Risk Factors & Triggers:**
   - Near-term downside catalysts
   - Competitive threats
   - Macro headwinds

4. **Risk/Reward Assessment:**
   - Downside target with rationale
   - Limited upside explanation
   - Risk management recommendations

5. **Counter Bull Arguments:**
   - Address bullish points
   - Explain why optimism is misplaced

**OUTPUT FORMAT:**
Be specific with numbers, price levels, and risk metrics.
End with: BEAR CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low"""

        self.bear_thesis = {}
    
    def load_discussion_points(self, filepath: str) -> Optional[Dict]:
        """Load discussion points from analysts"""
        print(f"[BEAR] Loading discussion points from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[BEAR] ✓ Loaded discussion for {data.get('ticker', 'unknown')}")
            
            # NEW: Check if discussion points have historical date info
            if data.get('analysis_date'):
                print(f"[BEAR] Discussion data is from historical date: {data.get('analysis_date')}")
            
            return data
        except FileNotFoundError:
            print(f"[BEAR] ❌ File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"[BEAR] ❌ Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"[BEAR] ❌ Load error: {e}")
            return None
    
    def extract_risks(self, discussion_points: Dict) -> Dict[str, List[str]]:
        """Extract bearish signals from full analyst reports"""
        full_reports = discussion_points.get('full_analyst_reports', {})
        risks = {'technical': [], 'fundamental': [], 'sentiment': [], 'macro': []}
        
        keywords = {
            'technical': ['overbought', 'resistance', 'downtrend', 'breakdown', 'bearish', 'weakness', 'divergence', 'below'],
            'fundamental': ['overvalued', 'debt', 'declining', 'expensive', 'concern', 'pressure', 'deterioration', 'slowing'],
            'sentiment': ['negative', 'selling', 'downgrade', 'caution', 'bearish', 'pessimistic', 'distribution'],
            'macro': ['risk-off', 'defensive', 'recession', 'headwind', 'rising rates', 'volatility', 'uncertainty']
        }
        
        for report_type, kw_list in keywords.items():
            report_text = full_reports.get(report_type, '')
            for kw in kw_list:
                if kw in report_text.lower():
                    sentences = report_text.split('.')
                    for sent in sentences:
                        if kw in sent.lower() and 20 < len(sent.strip()) < 300:
                            risks[report_type].append(sent.strip())
                            break
        
        # Add extracted bear evidence
        for evidence in discussion_points.get('bear_evidence', [])[:10]:
            source = evidence.get('source', 'unknown')
            signal = evidence.get('signal', '')
            if source in risks and signal:
                risks[source].append(signal)
        
        # Deduplicate and limit
        for category in risks:
            risks[category] = list(set(risks[category]))[:10]
        
        total = sum(len(v) for v in risks.values())
        print(f"[BEAR] ✓ Extracted {total} risk signals")
        return risks
    
    def calculate_risk_assessment(self, discussion_points: Dict) -> Dict[str, Any]:
        """Calculate risk metrics from analyst consensus"""
        summary = discussion_points.get('summary', {})
        recs = summary.get('recommendations', {})
        
        if not recs:
            return {
                'downside_risk': '15-20%',
                'limited_upside': '10-15%',
                'risk_level': 'MEDIUM',
                'bear_percentage': 50,
                'risk_score': 50,
                'conviction_level': 'MEDIUM',
                'analyst_breakdown': {'sell': 0, 'hold': 0, 'buy': 0}
            }
        
        bear_count = sum(1 for r in recs.values() if r == 'SELL')
        hold_count = sum(1 for r in recs.values() if r == 'HOLD')
        bull_count = sum(1 for r in recs.values() if r == 'BUY')
        total = len(recs)
        
        # Calculate bear percentage (HOLD counts as 0.5 bearish)
        bear_pct = ((bear_count + hold_count * 0.5) / total) * 100 if total > 0 else 50
        
        # Determine risk level and targets
        if bear_pct >= 75:
            downside, upside, risk, conviction = "25-35%", "5-10%", "HIGH", "HIGH"
        elif bear_pct >= 50:
            downside, upside, risk, conviction = "15-25%", "10-15%", "MEDIUM", "MEDIUM"
        else:
            downside, upside, risk, conviction = "10-15%", "15-25%", "LOW", "LOW"
        
        return {
            'downside_risk': downside,
            'limited_upside': upside,
            'risk_level': risk,
            'bear_percentage': bear_pct,
            'risk_score': min(100, bear_pct * 1.3),
            'conviction_level': conviction,
            'analyst_breakdown': {'sell': bear_count, 'hold': hold_count, 'buy': bull_count}
        }
    
    def identify_triggers(self, discussion_points: Dict) -> List[Dict]:
        """Identify potential downside triggers"""
        triggers = []
        
        # Check for urgent/critical risks
        for priority in discussion_points.get('research_priorities', []):
            if priority.get('priority') in ['URGENT', 'CRITICAL']:
                triggers.append({
                    'type': 'immediate',
                    'description': priority.get('description', ''),
                    'impact': 'HIGH',
                    'timeline': 'Imminent',
                    'probability': 'Medium-High'
                })
        
        # Standard downside triggers
        standard_triggers = [
            {'type': 'earnings', 'description': 'Earnings miss or guidance cut', 'impact': 'HIGH', 'timeline': '1-3 months', 'probability': 'Medium'},
            {'type': 'technical', 'description': 'Breakdown below key support level', 'impact': 'MEDIUM', 'timeline': '1-4 weeks', 'probability': 'Medium'},
            {'type': 'macro', 'description': 'Rising rates or recession fears escalate', 'impact': 'HIGH', 'timeline': 'Ongoing', 'probability': 'Medium'},
            {'type': 'competitive', 'description': 'Market share loss to competitors', 'impact': 'MEDIUM', 'timeline': '6-12 months', 'probability': 'Low-Medium'},
            {'type': 'regulatory', 'description': 'Adverse regulatory action or policy change', 'impact': 'MEDIUM', 'timeline': '3-6 months', 'probability': 'Low'}
        ]
        
        # Add triggers based on bear strength
        bear_count = discussion_points.get('summary', {}).get('bear_signal_count', 0)
        bull_count = discussion_points.get('summary', {}).get('bull_signal_count', 0)
        
        num_to_add = 4 if bear_count > bull_count else 2
        triggers.extend(standard_triggers[:num_to_add])
        
        return triggers
    
    def suggest_hedging(self, risk_assessment: Dict) -> List[Dict]:
        """Generate hedging/risk management recommendations"""
        risk_level = risk_assessment['risk_level']
        
        strategies = {
            'HIGH': [
                {'strategy': 'EXIT_POSITION', 'description': 'Full position exit recommended', 'urgency': 'Immediate', 'rationale': 'Risk/reward strongly unfavorable'},
                {'strategy': 'PROTECTIVE_PUTS', 'description': 'Buy puts 5-10% OTM', 'urgency': 'High', 'rationale': 'Hedge against sharp decline'},
                {'strategy': 'REDUCE_SIZE', 'description': 'Reduce position by 75-100%', 'urgency': 'High', 'rationale': 'Minimize exposure to downside'}
            ],
            'MEDIUM': [
                {'strategy': 'TIGHT_STOP', 'description': 'Set stop loss at -5% to -7%', 'urgency': 'Medium', 'rationale': 'Limit losses if support breaks'},
                {'strategy': 'PARTIAL_EXIT', 'description': 'Reduce position by 50%', 'urgency': 'Medium', 'rationale': 'De-risk while maintaining exposure'},
                {'strategy': 'COLLAR', 'description': 'Sell calls and buy puts', 'urgency': 'Medium', 'rationale': 'Cap upside to fund downside protection'}
            ],
            'LOW': [
                {'strategy': 'MONITOR', 'description': 'Daily monitoring of key levels', 'urgency': 'Low', 'rationale': 'Watch for deterioration signals'},
                {'strategy': 'STANDARD_STOP', 'description': 'Set stop loss at -10%', 'urgency': 'Low', 'rationale': 'Standard risk management'},
                {'strategy': 'COVERED_CALLS', 'description': 'Sell calls for income', 'urgency': 'Low', 'rationale': 'Generate income while waiting'}
            ]
        }
        
        return strategies.get(risk_level, strategies['MEDIUM'])
    
    def build_core_thesis(self, risks: Dict, risk_assessment: Dict) -> str:
        """Build core thesis statement"""
        pillars = []
        
        if risks.get('technical'):
            pillars.append("Technical indicators signal overbought conditions and breakdown risk")
        if risks.get('fundamental'):
            pillars.append("Fundamental deterioration threatens current valuation")
        if risks.get('sentiment'):
            pillars.append("Negative sentiment shift could accelerate selling pressure")
        if risks.get('macro'):
            pillars.append("Macroeconomic headwinds create systematic downside risk")
        
        if not pillars:
            pillars.append("Multiple risk factors warrant defensive positioning")
        
        thesis = f"""The bear case for {self.ticker} rests on {len(pillars)} critical pillars:

{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(pillars))}

**Risk Assessment Summary:**
- Downside Risk: {risk_assessment['downside_risk']}
- Limited Upside: {risk_assessment['limited_upside']}
- Risk Level: {risk_assessment['risk_level']}
- Conviction Level: {risk_assessment['conviction_level']} ({risk_assessment['bear_percentage']:.0f}% bearish consensus)
- Risk Score: {risk_assessment['risk_score']:.0f}/100
"""
        return thesis
    
    def generate_analysis(self, discussion_points: Dict) -> str:
        """Generate comprehensive bear analysis using LLM"""
        full_reports = discussion_points.get('full_analyst_reports', {})
        risks = self.extract_risks(discussion_points)
        risk_assessment = self.calculate_risk_assessment(discussion_points)
        core_thesis = self.build_core_thesis(risks, risk_assessment)
        
        if not self.client:
            print("[BEAR] ⚠️ No API client, using extracted data only")
            return core_thesis
        
        # === ADD HISTORICAL DATE CONTEXT ===
        date_context = ""
        if self.analysis_date:
            date_context = f"""
**⚠️ HISTORICAL ANALYSIS MODE ⚠️**
You are analyzing data AS OF {self.analysis_date}.
All analyst reports are from this historical date.
Build your bear case as if you were making a decision ON {self.analysis_date}.
Do NOT reference any events or data after {self.analysis_date}.

"""
        
        # Build comprehensive context for LLM
        context = f"""{date_context}# Build Comprehensive Bear Case for {self.ticker}

## Full Analyst Reports

### Technical Analysis:
{full_reports.get('technical', 'Not available')[:2000]}

### Fundamental Analysis:
{full_reports.get('fundamental', 'Not available')[:2000]}

### News & Sentiment:
{full_reports.get('news', 'Not available')[:1000]}

### Macro Environment:
{full_reports.get('macro', 'Not available')[:1000]}

## Extracted Risk Signals:
{json.dumps(risks, indent=2)}

## Preliminary Risk Assessment:
{json.dumps(risk_assessment, indent=2)}

---

Using the above data, create a compelling and comprehensive bear case.
- Be SPECIFIC with numbers, price levels, risk metrics
- Reference actual data from the reports
- Address potential bull arguments and explain why they're misguided
- Provide clear risk management recommendations

End with: BEAR CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low"""

        try:
            print(f"[BEAR] Generating analysis with {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_completion_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[BEAR] ❌ LLM error: {e}")
            return core_thesis
    
    def research(self, discussion_points: Dict) -> str:
        """Main research method - generates complete bear thesis"""
        print(f"\n{'='*60}")
        print(f"BEAR RESEARCHER: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Extract all components
        risks = self.extract_risks(discussion_points)
        risk_assessment = self.calculate_risk_assessment(discussion_points)
        triggers = self.identify_triggers(discussion_points)
        hedging = self.suggest_hedging(risk_assessment)
        core_thesis = self.build_core_thesis(risks, risk_assessment)
        
        # Generate LLM analysis
        full_analysis = self.generate_analysis(discussion_points)
        
        # Store complete thesis data
        self.bear_thesis = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,  # NEW: Include in output
            'historical_mode': self.analysis_date is not None,  # NEW
            'core_thesis': core_thesis,
            'risks': risks,
            'downside_triggers': triggers,
            'risk_assessment': risk_assessment,
            'hedging_strategies': hedging,
            'full_analysis': full_analysis
        }
        
        elapsed = time.time() - start_time
        print(f"\n[BEAR] ✓ Analysis complete in {elapsed:.2f}s")
        
        return full_analysis
    
    def save_thesis(self, filepath: str):
        """Save thesis data to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.bear_thesis, f, indent=2, ensure_ascii=False)
            print(f"[BEAR] ✓ Thesis saved to {filepath}")
        except Exception as e:
            print(f"[BEAR] ❌ Save error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Bear Researcher - Builds comprehensive bearish investment case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bear_researcher.py AAPL
  python bear_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json
  python bear_researcher.py AAPL --save-data ../../outputs/bear_thesis.json
  
  # HISTORICAL BACKTESTING:
  python bear_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json --analysis-date 2024-06-15
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--discussion-file", default="../../outputs/discussion_points.json",
                       help="Path to discussion points JSON from analysts")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", help="Save report to text file")
    parser.add_argument("--save-data", help="Save thesis data to JSON file")
    
    # ============================================================
    # NEW: Add analysis-date argument for historical backtesting
    # ============================================================
    parser.add_argument("--analysis-date",
                       type=str,
                       default=None,
                       help="Historical analysis date (YYYY-MM-DD format)")
    
    # Keep these for backward compatibility with master_orchestrator
    parser.add_argument("--mode", default="shallow", help="[Deprecated] Kept for compatibility")
    parser.add_argument("--rounds", type=int, default=1, help="[Deprecated] Kept for compatibility")
    
    args = parser.parse_args()
    
    try:
        researcher = BearResearcher(
            ticker=args.ticker,
            api_key=args.api_key,
            model=args.model,
            analysis_date=args.analysis_date  # NEW: Pass to researcher
        )
        
        # Load discussion points
        discussion_points = researcher.load_discussion_points(args.discussion_file)
        if not discussion_points:
            print("\n❌ Failed to load discussion points. Run analysts first.")
            sys.exit(1)
        
        # Run research
        report = researcher.research(discussion_points)
        
        # Print report
        print("\n" + "="*60)
        print(report)
        print("="*60 + "\n")
        
        # Save outputs
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ Report saved to {args.output}")
        
        if args.save_data:
            researcher.save_thesis(args.save_data)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()