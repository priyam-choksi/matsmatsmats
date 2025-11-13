"""
Bear Researcher - Enhanced with Multi-Round Debate Support
Builds comprehensive bearish case with shallow/deep/research modes

Usage: 
  python bear_researcher.py AAPL --mode shallow
  python bear_researcher.py AAPL --mode deep --rounds 3
  python bear_researcher.py AAPL --mode research --rounds 5
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class BearResearcher:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # System prompts for different modes
        self.base_system_prompt = """You are a Bear Analyst making the case AGAINST investing in this stock.

**YOUR ROLE:**
Present a well-reasoned argument emphasizing risks, challenges, and negative indicators. You are in a debate with a Bull Analyst, so your arguments must be:
- Data-driven and specific (reference actual numbers from reports)
- Logically sound (critical but intellectually honest)
- Comprehensive (technical, fundamental, sentiment, macro risks)
- Actionable (specific price targets, catalysts, risk management)

**ANALYSIS FRAMEWORK:**

1. **Risk Identification:**
   - Technical: Overbought, resistance, bearish patterns, breakdown risk
   - Fundamental: Overvaluation, declining growth, margin pressure, debt concerns
   - Sentiment: Excessive optimism (contrarian signal), negative catalysts
   - Macro: Interest rate headwinds, recession risk, sector rotation

2. **Counter Bull Arguments:**
   - Address each major bullish point with data-driven rebuttal
   - Expose over-optimistic assumptions
   - Show what bulls are missing or underweighting

3. **Downside Catalysts:**
   - Specific events that could trigger price decline
   - Timeline and probability for each
   - Quantify potential impact

4. **Risk/Reward:**
   - Downside target (% and price level)
   - Limited upside potential
   - Unfavorable risk/reward ratio

5. **Action Plan:**
   - Exit strategy (full/partial/hedge)
   - Specific levels for action
   - Risk management approach
"""
        
        # Debate-specific prompts
        self.debate_prompt_template = """
{base_prompt}

**DEBATE CONTEXT:**
Round {round_num} of {total_rounds} - {mode} mode

**Previous Debate History:**
{debate_history}

**Bull's Latest Argument:**
{bull_argument}

**YOUR TASK FOR THIS ROUND:**
{round_instructions}

{output_format}
"""
        
        # Storage
        self.bear_thesis = {}
        self.debate_history = []
    
    def load_discussion_points(self, filepath: str) -> Optional[Dict]:
        """Load discussion points with validation"""
        print(f"[BEAR] Loading discussion points from {filepath}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            required_keys = ['ticker', 'summary', 'full_analyst_reports']
            missing = [k for k in required_keys if k not in data]
            
            if missing:
                print(f"[BEAR] ⚠️  Missing keys: {missing}")
                return None
            
            print(f"[BEAR] ✓ Loaded discussion for {data['ticker']}")
            print(f"[BEAR] ✓ Full reports: {len(data.get('full_analyst_reports', {}))}")
            
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
    
    def extract_bear_signals_from_full_reports(self, discussion_points: Dict) -> Dict[str, List[str]]:
        """Extract bearish signals from full analyst reports"""
        print(f"[BEAR] Analyzing full analyst reports...")
        
        full_reports = discussion_points.get('full_analyst_reports', {})
        
        risk_categories = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'macro': []
        }
        
        # Keyword sets for each category
        keywords = {
            'technical': ['overbought', 'resistance', 'downtrend', 'breakdown', 'bearish', 'weakness', 'poor risk/reward'],
            'fundamental': ['overvalued', 'debt', 'declining', 'margin compression', 'liquidity concern', 'expensive', 'high P/E'],
            'sentiment': ['negative', 'bearish sentiment', 'selling pressure', 'downgrade', 'concern', 'caution'],
            'macro': ['risk-off', 'defensive', 'recession', 'headwind', 'rising rates', 'volatility', 'uncertainty']
        }
        
        # Extract from each report
        for report_type, keyword_list in keywords.items():
            report_text = full_reports.get(report_type, '')
            
            for keyword in keyword_list:
                if keyword in report_text.lower():
                    # Find sentences containing keyword
                    sentences = report_text.split('.')
                    for sent in sentences:
                        if keyword in sent.lower() and len(sent.strip()) > 20 and len(sent.strip()) < 300:
                            risk_categories[report_type].append(sent.strip())
                            break  # One per keyword
        
        # Also add extracted bear evidence
        for evidence in discussion_points.get('bear_evidence', [])[:10]:
            source = evidence.get('source', 'unknown')
            signal = evidence.get('signal', '')
            
            if source in risk_categories:
                risk_categories[source].append(signal)
        
        # Deduplicate and limit
        for category in risk_categories:
            risk_categories[category] = list(set(risk_categories[category]))[:10]
        
        total_risks = sum(len(v) for v in risk_categories.values())
        print(f"[BEAR] ✓ Extracted {total_risks} risk factors")
        
        return risk_categories
    
    def counter_bull_arguments(self, discussion_points: Dict) -> List[Dict]:
        """Build rebuttals to bullish arguments"""
        print(f"[BEAR] Developing bull rebuttals...")
        
        bull_evidence = discussion_points.get('bull_evidence', [])
        rebuttals = []
        
        for evidence in bull_evidence[:5]:
            signal = evidence.get('signal', '')
            source = evidence.get('source', '')
            
            rebuttal = {
                'bull_claim': signal,
                'source': source,
                'counter_argument': self._generate_smart_rebuttal(signal)
            }
            rebuttals.append(rebuttal)
        
        print(f"[BEAR] ✓ Created {len(rebuttals)} rebuttals")
        return rebuttals
    
    def _generate_smart_rebuttal(self, bull_signal: str) -> str:
        """Generate context-aware rebuttals"""
        signal_lower = bull_signal.lower()
        
        rebuttals = {
            'oversold': 'Oversold can persist in downtrends - catching falling knives is dangerous',
            'support': 'Support levels often break in weak markets',
            'growth': 'Growth is decelerating from historical peaks',
            'beat': 'Beating lowered estimates is not impressive - actual growth disappointing',
            'bullish': 'Excessive bullish sentiment often marks tops (contrarian signal)',
            'upgrade': 'Analyst upgrades lag price action and miss turning points',
            'strong': 'Temporary strength may mask underlying deterioration',
            'momentum': 'Momentum is mean-reverting and often reverses sharply',
            'undervalued': 'Value traps exist - cheap can get cheaper',
            'uptrend': 'Uptrends eventually end - exhaustion signs present'
        }
        
        for keyword, rebuttal in rebuttals.items():
            if keyword in signal_lower:
                return rebuttal
        
        return "This positive factor appears already priced in at current levels"
    
    def identify_downside_triggers(self, discussion_points: Dict) -> List[Dict]:
        """Identify downside catalysts"""
        print(f"[BEAR] Identifying downside triggers...")
        
        triggers = []
        
        # Urgent risks from priorities
        for priority in discussion_points.get('research_priorities', []):
            if priority.get('priority') in ['URGENT', 'CRITICAL']:
                triggers.append({
                    'type': 'immediate_risk',
                    'description': priority.get('description', ''),
                    'impact': 'HIGH',
                    'timeline': 'Imminent',
                    'probability': 'Medium to High'
                })
        
        # Standard triggers
        standard_triggers = [
            {'type': 'earnings_miss', 'description': 'Quarterly earnings disappointment or guidance cut', 'impact': 'HIGH', 'timeline': '1-3 months', 'probability': 'Medium'},
            {'type': 'technical_breakdown', 'description': 'Break below key support triggering stop losses', 'impact': 'MEDIUM', 'timeline': '1-2 weeks', 'probability': 'Medium'},
            {'type': 'macro_deterioration', 'description': 'Rising rates or recession fears escalating', 'impact': 'HIGH', 'timeline': 'Ongoing', 'probability': 'Medium to High'},
            {'type': 'competitive_threat', 'description': 'Market share loss to competitors', 'impact': 'MEDIUM', 'timeline': '6-12 months', 'probability': 'Low to Medium'},
            {'type': 'regulatory_action', 'description': 'Regulatory scrutiny or adverse policy changes', 'impact': 'MEDIUM', 'timeline': '3-6 months', 'probability': 'Low to Medium'}
        ]
        
        # Add based on bear strength
        bear_pct = discussion_points.get('summary', {}).get('bear_signal_count', 0)
        bull_pct = discussion_points.get('summary', {}).get('bull_signal_count', 0)
        
        if bear_pct > bull_pct:
            triggers.extend(standard_triggers[:4])
        else:
            triggers.extend(standard_triggers[2:4])
        
        print(f"[BEAR] ✓ Identified {len(triggers)} triggers")
        return triggers
    
    def calculate_risk_assessment(self, discussion_points: Dict) -> Dict[str, Any]:
        """Calculate risk metrics"""
        summary = discussion_points.get('summary', {})
        recs = summary.get('recommendations', {})
        
        bear_count = sum(1 for r in recs.values() if r == 'SELL')
        hold_count = sum(1 for r in recs.values() if r == 'HOLD')
        bull_count = sum(1 for r in recs.values() if r == 'BUY')
        total = len(recs)
        
        bear_pct = ((bear_count + hold_count * 0.5) / total) * 100 if total > 0 else 50
        
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
    
    def suggest_hedging_strategies(self, risk_assessment: Dict) -> List[Dict]:
        """Generate hedging strategies"""
        risk_level = risk_assessment['risk_level']
        
        strategies = {
            'HIGH': [
                {'strategy': 'IMMEDIATE_EXIT', 'description': 'Full position exit - risk too high', 'urgency': 'IMMEDIATE', 'rationale': 'Downside significantly outweighs upside'},
                {'strategy': 'PROTECTIVE_PUTS', 'description': 'Buy puts 5-10% OTM', 'urgency': 'HIGH', 'rationale': 'Hedge sharp downside'},
                {'strategy': 'POSITION_REDUCTION', 'description': 'Reduce 75-100%', 'urgency': 'HIGH', 'rationale': 'De-risk before catalyst'}
            ],
            'MEDIUM': [
                {'strategy': 'TIGHT_STOP', 'description': 'Stop loss at -5% to -7%', 'urgency': 'MEDIUM', 'rationale': 'Limit losses if support breaks'},
                {'strategy': 'PARTIAL_EXIT', 'description': 'Reduce 50%', 'urgency': 'MEDIUM', 'rationale': 'Take profits, reduce exposure'},
                {'strategy': 'COLLAR', 'description': 'Sell calls, buy puts', 'urgency': 'MEDIUM', 'rationale': 'Cap upside, protect downside'}
            ],
            'LOW': [
                {'strategy': 'MONITORING', 'description': 'Daily monitoring', 'urgency': 'LOW', 'rationale': 'Watch for deterioration'},
                {'strategy': 'STANDARD_STOP', 'description': 'Stop at -10%', 'urgency': 'LOW', 'rationale': 'Standard risk management'},
                {'strategy': 'COVERED_CALLS', 'description': 'Sell calls for income', 'urgency': 'LOW', 'rationale': 'Generate income while waiting'}
            ]
        }
        
        return strategies.get(risk_level, strategies['LOW'])
    
    def build_bear_thesis(self, risk_categories: Dict, risk_assessment: Dict) -> str:
        """Build core thesis statement"""
        components = []
        
        if risk_categories['technical']:
            components.append("Technical indicators signal overbought conditions and breakdown risk")
        if risk_categories['fundamental']:
            components.append("Fundamental deterioration threatens current valuation")
        if risk_categories['sentiment']:
            components.append("Negative sentiment shift could accelerate selling")
        if risk_categories['macro']:
            components.append("Macro headwinds create systematic downside risk")
        
        if not components:
            components.append("Multiple risk factors warrant defensive positioning")
        
        thesis = f"""The bear case for {self.ticker} rests on {len(components)} critical pillars:

{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(components))}

**Risk/Reward Analysis:**
- Downside Risk: {risk_assessment['downside_risk']}
- Limited Upside: {risk_assessment['limited_upside']}
- Risk Level: {risk_assessment['risk_level']}
- Conviction: {risk_assessment['conviction_level']} ({risk_assessment['bear_percentage']:.0f}% bearish/neutral)
- Risk Score: {risk_assessment['risk_score']:.0f}/100
"""
        return thesis
    
    def research_shallow(self, discussion_points: Dict) -> str:
        """
        SHALLOW MODE: Quick single-pass bear case
        Fast analysis, minimal LLM calls
        """
        print(f"\n[BEAR] 📊 SHALLOW MODE - Quick bear case\n")
        
        start_time = time.time()
        
        # Quick extraction
        risk_categories = self.extract_bear_signals_from_full_reports(discussion_points)
        risk_assessment = self.calculate_risk_assessment(discussion_points)
        triggers = self.identify_downside_triggers(discussion_points)[:3]  # Top 3 only
        hedging = self.suggest_hedging_strategies(risk_assessment)[:2]  # Top 2 only
        
        # Build basic thesis
        core_thesis = self.build_bear_thesis(risk_categories, risk_assessment)
        
        bear_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': 'SHALLOW',
            'core_thesis': core_thesis,
            'risks': risk_categories,
            'downside_triggers': triggers,
            'risk_assessment': risk_assessment,
            'hedging_strategies': hedging,
            'discussion_summary': discussion_points.get('summary', {})
        }
        
        self.bear_thesis = bear_data
        
        # Quick LLM synthesis (no debate)
        if self.client:
            report = self._quick_synthesis(bear_data, discussion_points)
        else:
            report = self._create_fallback_report(bear_data)
        
        elapsed = time.time() - start_time
        print(f"\n[BEAR] ✓ Shallow analysis in {elapsed:.2f}s\n")
        
        return report
    
    def research_deep(self, discussion_points: Dict, rounds: int = 3, bull_thesis: Optional[str] = None) -> str:
        """
        DEEP MODE: Multi-round debate with bull analyst
        Iterative refinement through argumentation
        """
        print(f"\n[BEAR] 🎯 DEEP MODE - {rounds}-round debate\n")
        
        start_time = time.time()
        
        # Extract comprehensive data
        risk_categories = self.extract_bear_signals_from_full_reports(discussion_points)
        rebuttals = self.counter_bull_arguments(discussion_points)
        triggers = self.identify_downside_triggers(discussion_points)
        risk_assessment = self.calculate_risk_assessment(discussion_points)
        hedging = self.suggest_hedging_strategies(risk_assessment)
        
        # Build initial thesis
        core_thesis = self.build_bear_thesis(risk_categories, risk_assessment)
        
        # Debate rounds
        debate_history = []
        current_argument = ""
        
        for round_num in range(1, rounds + 1):
            print(f"[BEAR] 🔄 Debate Round {round_num}/{rounds}")
            
            if round_num == 1:
                # Initial bear argument
                current_argument = self._generate_initial_argument(
                    core_thesis, risk_categories, triggers, 
                    risk_assessment, discussion_points
                )
            else:
                # Respond to bull's counter
                current_argument = self._generate_debate_response(
                    round_num, rounds, debate_history, 
                    bull_thesis, risk_categories, discussion_points
                )
            
            debate_history.append({
                'round': round_num,
                'speaker': 'bear',
                'argument': current_argument
            })
            
            print(f"[BEAR] ✓ Round {round_num} complete ({len(current_argument)} chars)")
        
        # Compile final thesis
        bear_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': 'DEEP',
            'rounds': rounds,
            'core_thesis': core_thesis,
            'risks': risk_categories,
            'bull_rebuttals': rebuttals,
            'downside_triggers': triggers,
            'risk_assessment': risk_assessment,
            'hedging_strategies': hedging,
            'debate_history': debate_history,
            'final_argument': current_argument
        }
        
        self.bear_thesis = bear_data
        
        # Create final report
        report = self._create_debate_report(bear_data)
        
        elapsed = time.time() - start_time
        print(f"\n[BEAR] ✓ Deep analysis in {elapsed:.2f}s ({rounds} rounds)\n")
        
        return report
    
    def research_comprehensive(self, discussion_points: Dict, rounds: int = 5, bull_thesis: Optional[str] = None) -> str:
        """
        RESEARCH MODE: Maximum depth with extended debate
        Most thorough analysis, highest LLM usage
        """
        print(f"\n[BEAR] 🔬 RESEARCH MODE - Comprehensive {rounds}-round analysis\n")
        
        # Research mode is like deep mode but with:
        # 1. More rounds
        # 2. Deeper data extraction
        # 3. More detailed synthesis
        
        return self.research_deep(discussion_points, rounds=rounds, bull_thesis=bull_thesis)
    
    def _generate_initial_argument(
        self, 
        core_thesis: str,
        risk_categories: Dict,
        triggers: List[Dict],
        risk_assessment: Dict,
        discussion_points: Dict
    ) -> str:
        """Generate Round 1 bear argument"""
        
        if not self.client:
            return core_thesis
        
        full_reports = discussion_points.get('full_analyst_reports', {})
        
        context = f"""# Initial Bear Argument for {self.ticker}

## Core Thesis
{core_thesis}

## Full Analyst Reports
{json.dumps(full_reports, indent=2)[:5000]}  # First 5k chars

## Risk Factors Identified
{json.dumps(risk_categories, indent=2)}

## Downside Triggers
{json.dumps(triggers, indent=2)}

Build your opening bear argument. Be compelling and specific."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.base_system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[BEAR] ⚠️  LLM error in round 1: {e}")
            return core_thesis
    
    def _generate_debate_response(
        self,
        round_num: int,
        total_rounds: int,
        debate_history: List[Dict],
        bull_argument: Optional[str],
        risk_categories: Dict,
        discussion_points: Dict
    ) -> str:
        """Generate debate response in later rounds"""
        
        if not self.client or not bull_argument:
            return "Bear maintains position based on identified risks."
        
        # Format debate history
        history_str = "\n\n".join([
            f"Round {d['round']} ({d['speaker']}): {d['argument'][:500]}..."
            for d in debate_history[-2:]  # Last 2 rounds
        ])
        
        instructions = f"""Counter the bull's latest argument while strengthening your bear case.

Focus on:
1. Refuting bull's specific claims with data
2. Highlighting risks they're ignoring
3. Reinforcing your strongest bear points
4. Providing new evidence from analyst reports

Keep your argument sharp and data-driven."""
        
        context = f"""# Debate Round {round_num}/{total_rounds}

## Previous Debate
{history_str}

## Bull's Latest Argument
{bull_argument[:1000]}

## Your Risk Analysis
{json.dumps(risk_categories, indent=2)[:2000]}

{instructions}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.base_system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[BEAR] ⚠️  LLM error in round {round_num}: {e}")
            return f"Bear maintains risk-focused position (Round {round_num} error)"
    
    def _quick_synthesis(self, bear_data: Dict, discussion_points: Dict) -> str:
        """Quick synthesis for shallow mode"""
        
        context = f"""Create a concise bear case for {self.ticker}:

Core Thesis: {bear_data['core_thesis']}
Risk Assessment: {json.dumps(bear_data['risk_assessment'], indent=2)}
Key Triggers: {json.dumps(bear_data['downside_triggers'], indent=2)}

Provide a brief but compelling bear case (500-800 words)."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.base_system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[BEAR] ⚠️  Synthesis error: {e}")
            return self._create_fallback_report(bear_data)
    
    def _create_debate_report(self, bear_data: Dict) -> str:
        """Format debate-style report"""
        report = f"""
# BEAR RESEARCH REPORT: {self.ticker}
{'='*70}
**Mode:** {bear_data['mode']} ({bear_data['rounds']} rounds)
**Generated:** {bear_data['timestamp']}

## Core Bear Thesis
{bear_data['core_thesis']}

## Debate Evolution
"""
        for entry in bear_data['debate_history']:
            report += f"\n### Round {entry['round']} - Bear Argument\n"
            report += f"{entry['argument'][:800]}...\n"
        
        report += f"""
## Risk Assessment Summary
- Downside: {bear_data['risk_assessment']['downside_risk']}
- Upside: {bear_data['risk_assessment']['limited_upside']}
- Risk Level: **{bear_data['risk_assessment']['risk_level']}**
- Conviction: {bear_data['risk_assessment']['conviction_level']}

## Recommended Actions
"""
        for strat in bear_data['hedging_strategies']:
            report += f"\n**{strat['strategy']}** [{strat['urgency']}]\n"
            report += f"- {strat['description']}\n"
            report += f"- {strat['rationale']}\n"
        
        report += f"\n{'='*70}\n"
        report += f"BEAR CASE STRENGTH: {bear_data['risk_assessment']['conviction_level']} - Confidence: {bear_data['risk_assessment']['conviction_level']}\n"
        
        return report
    
    def _create_fallback_report(self, bear_data: Dict) -> str:
        """Fallback report without LLM"""
        report = f"""
# BEAR RESEARCH REPORT: {self.ticker}
{'='*70}
*Fallback Mode - LLM Unavailable*

## Core Thesis
{bear_data['core_thesis']}

## Risk Factors
"""
        for category, risks in bear_data.get('risks', {}).items():
            if risks:
                report += f"\n### {category.title()} Risks:\n"
                for risk in risks[:5]:
                    report += f"- {risk}\n"
        
        report += f"""
## Risk Assessment
- Downside: {bear_data['risk_assessment']['downside_risk']}
- Risk Level: {bear_data['risk_assessment']['risk_level']}
- Conviction: {bear_data['risk_assessment']['conviction_level']}

BEAR CASE STRENGTH: {bear_data['risk_assessment']['conviction_level']} - Confidence: {bear_data['risk_assessment']['conviction_level']}
"""
        return report
    
    def research(self, discussion_points: Dict, mode: str = 'shallow', rounds: int = 1, bull_thesis: Optional[str] = None) -> str:
        """
        Main research entry point - dispatches to appropriate mode
        """
        if mode == 'shallow':
            return self.research_shallow(discussion_points)
        elif mode == 'deep':
            return self.research_deep(discussion_points, rounds=rounds, bull_thesis=bull_thesis)
        elif mode == 'research':
            return self.research_comprehensive(discussion_points, rounds=rounds, bull_thesis=bull_thesis)
        else:
            print(f"[BEAR] ⚠️  Unknown mode '{mode}', using shallow")
            return self.research_shallow(discussion_points)
    
    def save_thesis(self, filepath: str):
        """Save thesis data"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.bear_thesis, f, indent=2)
            print(f"[BEAR] ✓ Saved to {filepath}")
        except Exception as e:
            print(f"[BEAR] ⚠️  Save error: {e}")
    
    def extract_bear_signals_from_full_reports(self, discussion_points: Dict) -> Dict[str, List[str]]:
        """Extract from full reports (implementation from above)"""
        print(f"[BEAR] Analyzing full reports...")
        
        full_reports = discussion_points.get('full_analyst_reports', {})
        risk_categories = {'technical': [], 'fundamental': [], 'sentiment': [], 'macro': []}
        
        keywords = {
            'technical': ['overbought', 'resistance', 'downtrend', 'bearish', 'weakness'],
            'fundamental': ['overvalued', 'debt', 'declining', 'expensive', 'concern'],
            'sentiment': ['negative', 'selling', 'downgrade', 'caution'],
            'macro': ['risk-off', 'defensive', 'recession', 'headwind']
        }
        
        for report_type, kw_list in keywords.items():
            report = full_reports.get(report_type, '')
            for kw in kw_list:
                if kw in report.lower():
                    for sent in report.split('.'):
                        if kw in sent.lower() and 20 < len(sent.strip()) < 300:
                            risk_categories[report_type].append(sent.strip())
                            break
        
        # Add extracted evidence
        for ev in discussion_points.get('bear_evidence', [])[:10]:
            source = ev.get('source', 'unknown')
            if source in risk_categories:
                risk_categories[source].append(ev.get('signal', ''))
        
        # Deduplicate
        for cat in risk_categories:
            risk_categories[cat] = list(set(risk_categories[cat]))[:10]
        
        print(f"[BEAR] ✓ Extracted {sum(len(v) for v in risk_categories.values())} risks")
        return risk_categories


def main():
    parser = argparse.ArgumentParser(
        description="Bear Researcher - Multi-mode bearish analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Research Modes:
  shallow  - Quick single-pass (1 LLM call, ~30s)
  deep     - Multi-round debate (3-5 rounds, ~2-3 min)
  research - Comprehensive analysis (5+ rounds, ~5 min)

Examples:
  python bear_researcher.py AAPL --mode shallow
  python bear_researcher.py AAPL --mode deep --rounds 3
  python bear_researcher.py AAPL --mode research --rounds 5 --output bear_thesis.txt
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("--discussion-file", default="../../outputs/discussion_points.json",
                       help="Discussion points JSON")
    parser.add_argument("--mode", choices=['shallow', 'deep', 'research'], default='shallow',
                       help="Analysis depth (default: shallow)")
    parser.add_argument("--rounds", type=int, default=3,
                       help="Debate rounds for deep/research mode (default: 3)")
    parser.add_argument("--bull-thesis", help="Bull thesis for debate (optional)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-data", help="Save JSON")
    
    args = parser.parse_args()
    
    try:
        researcher = BearResearcher(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
        # Load discussion
        discussion_points = researcher.load_discussion_points(args.discussion_file)
        if not discussion_points:
            sys.exit(1)
        
        # Load bull thesis if in debate mode
        bull_thesis = None
        if args.bull_thesis and os.path.exists(args.bull_thesis):
            with open(args.bull_thesis, 'r', encoding='utf-8') as f:
                bull_thesis = f.read()
        
        # Run research
        report = researcher.research(
            discussion_points,
            mode=args.mode,
            rounds=args.rounds,
            bull_thesis=bull_thesis
        )
        
        print(report)
        
        # Save
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Saved to {args.output}")
        
        if args.save_data:
            researcher.save_thesis(args.save_data)
        
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