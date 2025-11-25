"""
Bull Researcher - Clean Single-Pass Analysis
Builds comprehensive bullish case from analyst discussion points
Debate orchestration handled by research_manager.py

Usage: python bull_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json
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


class BullResearcher:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-5-nano"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are a Bull Analyst building the strongest possible case FOR investing in this stock.

**YOUR MISSION:**
Create a compelling, data-driven investment thesis emphasizing opportunities, strengths, and catalysts.

**ANALYSIS REQUIREMENTS:**

1. **Technical Analysis:**
   - Trend direction and strength
   - Key support/resistance levels
   - Momentum indicators (RSI, MACD)
   - Volume patterns

2. **Fundamental Analysis:**
   - Valuation metrics vs peers/history
   - Growth trajectory (revenue, earnings)
   - Margin trends and profitability
   - Balance sheet strength

3. **Catalysts & Opportunities:**
   - Near-term positive triggers
   - Product/market expansion
   - Macro tailwinds

4. **Risk/Reward Assessment:**
   - Upside target with rationale
   - Downside risk acknowledgment
   - Position sizing recommendation

5. **Counter Bear Arguments:**
   - Address major concerns
   - Explain why risks are overblown

**OUTPUT FORMAT:**
Be specific with numbers, price levels, and timeframes.
End with: BULL CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low"""

        self.bull_thesis = {}
    
    def load_discussion_points(self, filepath: str) -> Optional[Dict]:
        """Load discussion points from analysts"""
        print(f"[BULL] Loading discussion points from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[BULL] ✓ Loaded discussion for {data.get('ticker', 'unknown')}")
            return data
        except FileNotFoundError:
            print(f"[BULL] ❌ File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"[BULL] ❌ Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"[BULL] ❌ Load error: {e}")
            return None
    
    def extract_opportunities(self, discussion_points: Dict) -> Dict[str, List[str]]:
        """Extract bullish signals from full analyst reports"""
        full_reports = discussion_points.get('full_analyst_reports', {})
        opportunities = {'technical': [], 'fundamental': [], 'sentiment': [], 'macro': []}
        
        keywords = {
            'technical': ['uptrend', 'support', 'bullish', 'momentum', 'breakout', 'oversold', 'golden cross', 'above'],
            'fundamental': ['growth', 'undervalued', 'strong', 'beat', 'expanding', 'improving', 'positive', 'solid'],
            'sentiment': ['bullish', 'upgrade', 'positive', 'buy rating', 'optimistic', 'favorable', 'accumulation'],
            'macro': ['risk-on', 'tailwind', 'favorable', 'supportive', 'expansion', 'growth']
        }
        
        for report_type, kw_list in keywords.items():
            report_text = full_reports.get(report_type, '')
            for kw in kw_list:
                if kw in report_text.lower():
                    sentences = report_text.split('.')
                    for sent in sentences:
                        if kw in sent.lower() and 20 < len(sent.strip()) < 300:
                            opportunities[report_type].append(sent.strip())
                            break
        
        # Add extracted bull evidence
        for evidence in discussion_points.get('bull_evidence', [])[:10]:
            source = evidence.get('source', 'unknown')
            signal = evidence.get('signal', '')
            if source in opportunities and signal:
                opportunities[source].append(signal)
        
        # Deduplicate and limit
        for category in opportunities:
            opportunities[category] = list(set(opportunities[category]))[:10]
        
        total = sum(len(v) for v in opportunities.values())
        print(f"[BULL] ✓ Extracted {total} opportunity signals")
        return opportunities
    
    def calculate_risk_reward(self, discussion_points: Dict) -> Dict[str, Any]:
        """Calculate risk/reward metrics from analyst consensus"""
        summary = discussion_points.get('summary', {})
        recs = summary.get('recommendations', {})
        
        if not recs:
            return {
                'upside_potential': '15-25%',
                'downside_risk': '10-15%',
                'reward_risk_ratio': 1.5,
                'bull_percentage': 50,
                'conviction_level': 'MEDIUM',
                'analyst_breakdown': {'buy': 0, 'hold': 0, 'sell': 0}
            }
        
        bull_count = sum(1 for r in recs.values() if r == 'BUY')
        hold_count = sum(1 for r in recs.values() if r == 'HOLD')
        bear_count = sum(1 for r in recs.values() if r == 'SELL')
        total = len(recs)
        
        # Calculate bull percentage (HOLD counts as 0.5 bullish)
        bull_pct = ((bull_count + hold_count * 0.5) / total) * 100 if total > 0 else 50
        
        # Determine conviction and targets based on consensus
        if bull_pct >= 75:
            upside, downside, rr_ratio, conviction = "30-50%", "5-10%", 4.0, "HIGH"
        elif bull_pct >= 50:
            upside, downside, rr_ratio, conviction = "20-30%", "10-15%", 2.0, "MEDIUM"
        else:
            upside, downside, rr_ratio, conviction = "10-20%", "15-20%", 1.0, "LOW"
        
        return {
            'upside_potential': upside,
            'downside_risk': downside,
            'reward_risk_ratio': rr_ratio,
            'bull_percentage': bull_pct,
            'conviction_level': conviction,
            'analyst_breakdown': {'buy': bull_count, 'hold': hold_count, 'sell': bear_count}
        }
    
    def identify_catalysts(self, discussion_points: Dict) -> List[Dict]:
        """Identify potential upside catalysts"""
        catalysts = []
        
        # Check for urgent/time-sensitive items
        for priority in discussion_points.get('research_priorities', []):
            if priority.get('priority') == 'URGENT':
                catalysts.append({
                    'type': 'immediate',
                    'description': priority.get('description', ''),
                    'impact': 'HIGH',
                    'timeline': 'Imminent',
                    'probability': 'Medium-High'
                })
        
        # Standard catalysts
        standard_catalysts = [
            {'type': 'earnings', 'description': 'Quarterly earnings beat and guidance raise', 'impact': 'HIGH', 'timeline': '1-3 months', 'probability': 'Medium'},
            {'type': 'technical', 'description': 'Breakout above key resistance level', 'impact': 'MEDIUM', 'timeline': '1-4 weeks', 'probability': 'Medium'},
            {'type': 'macro', 'description': 'Favorable sector rotation or rate environment', 'impact': 'MEDIUM', 'timeline': 'Ongoing', 'probability': 'Medium'},
            {'type': 'product', 'description': 'New product launch or market expansion', 'impact': 'HIGH', 'timeline': '3-6 months', 'probability': 'Medium'},
            {'type': 'institutional', 'description': 'Increased institutional buying or upgrades', 'impact': 'MEDIUM', 'timeline': '1-3 months', 'probability': 'Low-Medium'}
        ]
        
        # Add catalysts based on bull strength
        bull_count = discussion_points.get('summary', {}).get('bull_signal_count', 0)
        bear_count = discussion_points.get('summary', {}).get('bear_signal_count', 0)
        
        num_to_add = 4 if bull_count > bear_count else 2
        catalysts.extend(standard_catalysts[:num_to_add])
        
        return catalysts
    
    def suggest_entry_strategies(self, risk_reward: Dict) -> List[Dict]:
        """Generate entry strategy recommendations"""
        conviction = risk_reward['conviction_level']
        
        strategies = {
            'HIGH': [
                {'strategy': 'FULL_POSITION', 'description': 'Enter full position at current levels', 'timing': 'Immediate', 'size': '10-15% of portfolio'},
                {'strategy': 'BUY_DIPS', 'description': 'Add on any pullbacks to support', 'timing': 'Opportunistic', 'size': 'Scale up to 15%'}
            ],
            'MEDIUM': [
                {'strategy': 'SCALE_IN', 'description': 'Build position over 2-3 entries', 'timing': 'Gradual', 'size': '5-10% of portfolio'},
                {'strategy': 'WAIT_PULLBACK', 'description': 'Wait for 3-5% pullback to support', 'timing': 'Patient', 'size': '5-8% of portfolio'}
            ],
            'LOW': [
                {'strategy': 'STARTER_POSITION', 'description': 'Small pilot position only', 'timing': 'Wait & Watch', 'size': '2-3% of portfolio'},
                {'strategy': 'WAIT_CATALYST', 'description': 'Wait for confirming catalyst', 'timing': 'Patient', 'size': '3-5% after confirmation'}
            ]
        }
        
        return strategies.get(conviction, strategies['MEDIUM'])
    
    def build_core_thesis(self, opportunities: Dict, risk_reward: Dict) -> str:
        """Build core thesis statement"""
        pillars = []
        
        if opportunities.get('technical'):
            pillars.append("Technical momentum and trend structure favor continued upside")
        if opportunities.get('fundamental'):
            pillars.append("Fundamental strength supports current and higher valuation levels")
        if opportunities.get('sentiment'):
            pillars.append("Positive sentiment shift creating sustained buying pressure")
        if opportunities.get('macro'):
            pillars.append("Macroeconomic tailwinds provide systematic support")
        
        if not pillars:
            pillars.append("Multiple factors align for potential appreciation")
        
        thesis = f"""The bull case for {self.ticker} rests on {len(pillars)} key pillars:

{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(pillars))}

**Risk/Reward Summary:**
- Upside Potential: {risk_reward['upside_potential']}
- Downside Risk: {risk_reward['downside_risk']}
- Reward/Risk Ratio: {risk_reward['reward_risk_ratio']:.1f}:1
- Conviction Level: {risk_reward['conviction_level']} ({risk_reward['bull_percentage']:.0f}% bullish consensus)
"""
        return thesis
    
    def generate_analysis(self, discussion_points: Dict) -> str:
        """Generate comprehensive bull analysis using LLM"""
        full_reports = discussion_points.get('full_analyst_reports', {})
        opportunities = self.extract_opportunities(discussion_points)
        risk_reward = self.calculate_risk_reward(discussion_points)
        core_thesis = self.build_core_thesis(opportunities, risk_reward)
        
        if not self.client:
            print("[BULL] ⚠️ No API client, using extracted data only")
            return core_thesis
        
        # Build comprehensive context for LLM
        context = f"""# Build Comprehensive Bull Case for {self.ticker}

## Full Analyst Reports

### Technical Analysis:
{full_reports.get('technical', 'Not available')[:2000]}

### Fundamental Analysis:
{full_reports.get('fundamental', 'Not available')[:2000]}

### News & Sentiment:
{full_reports.get('news', 'Not available')[:1000]}

### Macro Environment:
{full_reports.get('macro', 'Not available')[:1000]}

## Extracted Opportunity Signals:
{json.dumps(opportunities, indent=2)}

## Preliminary Risk/Reward Assessment:
{json.dumps(risk_reward, indent=2)}

---

Using the above data, create a compelling and comprehensive bull case.
- Be SPECIFIC with numbers, price levels, percentages
- Reference actual data from the reports
- Address potential bear concerns and explain why they're overblown
- Provide clear entry strategy and price targets

End with: BULL CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low"""

        try:
            print(f"[BULL] Generating analysis with {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[BULL] ❌ LLM error: {e}")
            return core_thesis
    
    def research(self, discussion_points: Dict) -> str:
        """Main research method - generates complete bull thesis"""
        print(f"\n{'='*60}")
        print(f"BULL RESEARCHER: {self.ticker}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Extract all components
        opportunities = self.extract_opportunities(discussion_points)
        risk_reward = self.calculate_risk_reward(discussion_points)
        catalysts = self.identify_catalysts(discussion_points)
        entry_strategies = self.suggest_entry_strategies(risk_reward)
        core_thesis = self.build_core_thesis(opportunities, risk_reward)
        
        # Generate LLM analysis
        full_analysis = self.generate_analysis(discussion_points)
        
        # Store complete thesis data
        self.bull_thesis = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'core_thesis': core_thesis,
            'opportunities': opportunities,
            'catalysts': catalysts,
            'risk_reward': risk_reward,
            'entry_strategies': entry_strategies,
            'full_analysis': full_analysis
        }
        
        elapsed = time.time() - start_time
        print(f"\n[BULL] ✓ Analysis complete in {elapsed:.2f}s")
        
        return full_analysis
    
    def save_thesis(self, filepath: str):
        """Save thesis data to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.bull_thesis, f, indent=2, ensure_ascii=False)
            print(f"[BULL] ✓ Thesis saved to {filepath}")
        except Exception as e:
            print(f"[BULL] ❌ Save error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Bull Researcher - Builds comprehensive bullish investment case",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bull_researcher.py AAPL
  python bull_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json
  python bull_researcher.py AAPL --save-data ../../outputs/bull_thesis.json
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--discussion-file", default="../../outputs/discussion_points.json",
                       help="Path to discussion points JSON from analysts")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", help="Save report to text file")
    parser.add_argument("--save-data", help="Save thesis data to JSON file")
    
    # Keep these for backward compatibility with master_orchestrator
    parser.add_argument("--mode", default="shallow", help="[Deprecated] Kept for compatibility")
    parser.add_argument("--rounds", type=int, default=1, help="[Deprecated] Kept for compatibility")
    
    args = parser.parse_args()
    
    try:
        researcher = BullResearcher(
            ticker=args.ticker,
            api_key=args.api_key,
            model=args.model
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