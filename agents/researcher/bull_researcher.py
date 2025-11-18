"""
Bull Researcher - Enhanced with Multi-Round Debate Support
Builds comprehensive bullish case with shallow/deep/research modes

Usage: 
  python bull_researcher.py AAPL --mode shallow
  python bull_researcher.py AAPL --mode deep --rounds 3
  python bull_researcher.py AAPL --mode research --rounds 5
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


class BullResearcher:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Enhanced system prompt combining reference paper's debate style
        self.base_system_prompt = """You are a Bull Analyst making the strongest possible case FOR investing in this stock.

**YOUR MISSION:**
Build a compelling, well-reasoned argument emphasizing opportunities, strengths, and positive catalysts. You are building a thesis that will be debated against the Bear Analyst, so your arguments must be:
1. Data-driven and specific (reference actual numbers)
2. Logically sound (optimistic but not naive)
3. Comprehensive (technical, fundamental, sentiment, macro tailwinds)
4. Actionable (provide specific price targets and catalysts)

**ANALYSIS FRAMEWORK:**

1. **Opportunity Identification:**
   - Technical: Uptrends, support levels, bullish patterns, momentum
   - Fundamental: Undervaluation, accelerating growth, margin expansion, strong balance sheet
   - Sentiment: Positive shifts, upgrades, institutional accumulation
   - Macro: Favorable rates, sector rotation into, economic tailwinds

2. **Counter Bear Arguments:**
   - For each major bearish concern, provide logical rebuttal
   - Use data to show why fears are overblown or temporary
   - Identify what bears are missing or overweighting

3. **Upside Catalysts & Drivers:**
   - Specific events that could drive price appreciation
   - Timeline and probability for each
   - Quantify potential upside impact

4. **Risk/Reward Assessment:**
   - Quantify upside potential (% and $ targets)
   - Acknowledge but minimize downside risk
   - Calculate favorable risk/reward ratios (aim for 3:1+)

5. **Actionable Recommendations:**
   - Entry strategies (immediate, scale-in, wait for pullback)
   - Specific price levels for entry
   - Position sizing recommendations

**OUTPUT REQUIREMENTS:**

## Bull Case Summary
[2-3 sentence thesis of why this is a BUY]

## Critical Opportunity Factors

### Technical Catalysts
[List 3-5 technical positives with specific levels]

### Fundamental Strengths
[List 3-5 fundamental drivers with metrics]

### Sentiment & News Catalysts
[Positive shifts, upcoming events]

### Macro Tailwinds
[Favorable economic/market conditions]

## Counter to Bear Arguments
[For top 3 bear concerns, provide data-driven rebuttals]

## Upside Catalysts & Timeline
[List 5+ specific events with timeline and probability]

## Risk/Reward Assessment
- **Upside Target:** $X (+Y%)
- **Downside Risk:** $X (-Y%)
- **Risk/Reward Ratio:** Favorable (X:1)
- **Conviction Level:** High/Medium/Low

## Recommended Entry Strategy
[Specific trading recommendations: buy now, scale-in, wait]

BULL CASE STRENGTH: Strong/Moderate/Weak - Confidence: High/Medium/Low

**CRITICAL:** Be intellectually honest. If opportunities are limited, say so. Your job is rigorous analysis, not blind optimism."""
        
        # Storage
        self.bull_thesis = {}
        self.debate_history = []
    
    def load_discussion_points(self, filepath: str) -> Optional[Dict]:
        """Load discussion points with validation"""
        print(f"[BULL] Loading discussion points from {filepath}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            required_keys = ['ticker', 'summary', 'full_analyst_reports']
            missing = [k for k in required_keys if k not in data]
            
            if missing:
                print(f"[BULL] ⚠️  Missing keys: {missing}")
                return None
            
            print(f"[BULL] ✓ Loaded discussion for {data['ticker']}")
            print(f"[BULL] ✓ Full reports: {len(data.get('full_analyst_reports', {}))}")
            
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
    
    def extract_bull_signals_from_full_reports(self, discussion_points: Dict) -> Dict[str, List[str]]:
        """Extract bullish signals from full analyst reports"""
        print(f"[BULL] Analyzing full analyst reports...")
        
        full_reports = discussion_points.get('full_analyst_reports', {})
        
        opportunity_categories = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'macro': []
        }
        
        # Keyword sets for bullish signals
        keywords = {
            'technical': ['uptrend', 'support', 'bullish', 'momentum', 'breakout', 'oversold', 'golden cross'],
            'fundamental': ['growth', 'undervalued', 'strong', 'beat', 'expanding', 'improving', 'positive cash flow'],
            'sentiment': ['bullish', 'upgrade', 'positive', 'buy rating', 'optimistic', 'favorable'],
            'macro': ['risk-on', 'tailwind', 'favorable', 'growth', 'supportive', 'cyclical strength']
        }
        
        # Extract from each report
        for report_type, kw_list in keywords.items():
            report_text = full_reports.get(report_type, '')
            
            for keyword in kw_list:
                if keyword in report_text.lower():
                    sentences = report_text.split('.')
                    for sent in sentences:
                        if keyword in sent.lower() and 20 < len(sent.strip()) < 300:
                            opportunity_categories[report_type].append(sent.strip())
                            break
        
        # Add extracted bull evidence
        for evidence in discussion_points.get('bull_evidence', [])[:10]:
            source = evidence.get('source', 'unknown')
            signal = evidence.get('signal', '')
            
            if source in opportunity_categories:
                opportunity_categories[source].append(signal)
        
        # Deduplicate
        for category in opportunity_categories:
            opportunity_categories[category] = list(set(opportunity_categories[category]))[:10]
        
        total_opportunities = sum(len(v) for v in opportunity_categories.values())
        print(f"[BULL] ✓ Extracted {total_opportunities} opportunity factors")
        
        return opportunity_categories
    
    def counter_bear_arguments(self, discussion_points: Dict) -> List[Dict]:
        """Build rebuttals to bearish arguments"""
        print(f"[BULL] Developing rebuttals to bear arguments...")
        
        bear_evidence = discussion_points.get('bear_evidence', [])
        rebuttals = []
        
        for evidence in bear_evidence[:5]:
            signal = evidence.get('signal', '')
            source = evidence.get('source', '')
            
            rebuttal = {
                'bear_concern': signal,
                'source': source,
                'counter_argument': self._generate_smart_rebuttal(signal)
            }
            rebuttals.append(rebuttal)
        
        print(f"[BULL] ✓ Created {len(rebuttals)} rebuttals")
        return rebuttals
    
    def _generate_smart_rebuttal(self, bear_signal: str) -> str:
        """Generate context-aware rebuttals to bear concerns"""
        signal_lower = bear_signal.lower()
        
        rebuttals = {
            'overbought': 'Overbought can persist in strong uptrends - momentum often continues further than expected',
            'resistance': 'Previous resistance becomes support after breakout - levels are meant to be broken',
            'overvalued': 'Growth companies often trade at premium multiples - justified by future potential',
            'debt': 'Debt is manageable with strong cash flow generation and low rates',
            'risk': 'Risk is already priced in - market looks forward not backward',
            'concern': 'Temporary concerns often create buying opportunities',
            'weakness': 'Short-term weakness in strong trends offers better entry points',
            'bearish': 'Excessive bearish sentiment is often a contrarian buy signal',
            'downgrade': 'Analyst downgrades frequently mark bottoms - they lag price action',
            'expensive': 'Quality deserves premium - you get what you pay for',
            'high': 'High metrics reflect quality and growth potential',
            'declining': 'Cyclical declines are temporary in secular growth stories'
        }
        
        for keyword, rebuttal in rebuttals.items():
            if keyword in signal_lower:
                return rebuttal
        
        return "This concern appears overblown given the positive fundamental momentum"
    
    def identify_upside_catalysts(self, discussion_points: Dict) -> List[Dict]:
        """Identify catalysts that could drive price higher"""
        print(f"[BULL] Identifying upside catalysts...")
        
        catalysts = []
        
        # Urgent positive catalysts from priorities
        for priority in discussion_points.get('research_priorities', []):
            if priority.get('priority') == 'URGENT' and 'time_sensitive' in priority.get('focus', ''):
                catalysts.append({
                    'type': 'immediate_catalyst',
                    'description': priority.get('description', ''),
                    'impact': 'HIGH',
                    'timeline': 'Imminent',
                    'probability': 'Medium to High'
                })
        
        # Standard positive catalysts
        standard_catalysts = [
            {
                'type': 'earnings_beat',
                'description': 'Quarterly earnings beat and guidance raise',
                'impact': 'HIGH',
                'timeline': '1-3 months',
                'probability': 'Medium to High'
            },
            {
                'type': 'technical_breakout',
                'description': 'Breakout above resistance triggering momentum buying',
                'impact': 'MEDIUM',
                'timeline': '1-2 weeks',
                'probability': 'Medium'
            },
            {
                'type': 'macro_tailwind',
                'description': 'Favorable sector rotation or rate environment',
                'impact': 'MEDIUM',
                'timeline': 'Ongoing',
                'probability': 'Medium'
            },
            {
                'type': 'product_innovation',
                'description': 'New product launch or market expansion',
                'impact': 'MEDIUM to HIGH',
                'timeline': '3-6 months',
                'probability': 'Medium'
            },
            {
                'type': 'institutional_accumulation',
                'description': 'Increased institutional buying or analyst upgrades',
                'impact': 'MEDIUM',
                'timeline': '1-3 months',
                'probability': 'Low to Medium'
            }
        ]
        
        # Add based on bull strength
        bull_count = discussion_points.get('summary', {}).get('bull_signal_count', 0)
        bear_count = discussion_points.get('summary', {}).get('bear_signal_count', 0)
        
        if bull_count > bear_count:
            catalysts.extend(standard_catalysts[:4])
        else:
            catalysts.extend(standard_catalysts[:2])
        
        print(f"[BULL] ✓ Identified {len(catalysts)} catalysts")
        return catalysts
    
    def calculate_risk_reward(self, discussion_points: Dict) -> Dict[str, Any]:
        """Calculate risk/reward metrics from bull perspective"""
        print(f"[BULL] Calculating risk/reward...")
        
        summary = discussion_points.get('summary', {})
        recs = summary.get('recommendations', {})
        
        bull_count = sum(1 for r in recs.values() if r == 'BUY')
        hold_count = sum(1 for r in recs.values() if r == 'HOLD')
        bear_count = sum(1 for r in recs.values() if r == 'SELL')
        total = len(recs)
        
        # Bull percentage (HOLD counts as 0.5 bullish)
        bull_pct = ((bull_count + hold_count * 0.5) / total) * 100 if total > 0 else 50
        
        # Estimate upside/downside from bull perspective
        if bull_pct >= 75:
            upside = "30-50%"
            downside = "5-10%"
            rr_ratio = 4.0
            conviction = "HIGH"
        elif bull_pct >= 50:
            upside = "20-30%"
            downside = "10-15%"
            rr_ratio = 2.0
            conviction = "MEDIUM"
        else:
            upside = "10-20%"
            downside = "15-20%"
            rr_ratio = 1.0
            conviction = "LOW"
        
        assessment = {
            'upside_potential': upside,
            'downside_risk': downside,
            'reward_risk_ratio': rr_ratio,
            'bull_percentage': bull_pct,
            'conviction_level': conviction,
            'analyst_breakdown': {'buy': bull_count, 'hold': hold_count, 'sell': bear_count}
        }
        
        print(f"[BULL] ✓ R/R Ratio: {rr_ratio:.1f}:1, Bull%: {bull_pct:.0f}%")
        return assessment
    
    def suggest_entry_strategies(self, risk_reward: Dict) -> List[Dict]:
        """Generate entry and position building strategies"""
        conviction = risk_reward['conviction_level']
        
        strategies = {
            'HIGH': [
                {
                    'strategy': 'AGGRESSIVE_ENTRY',
                    'description': 'Enter full position immediately',
                    'timing': 'IMMEDIATE',
                    'rationale': 'Strong upside with limited downside - high conviction setup'
                },
                {
                    'strategy': 'BUY_WEAKNESS',
                    'description': 'Add on any pullbacks to support',
                    'timing': 'OPPORTUNISTIC',
                    'rationale': 'Scale up on temporary weakness in strong trend'
                },
                {
                    'strategy': 'LARGE_POSITION',
                    'description': 'Position size 10-15% of portfolio',
                    'timing': 'IMMEDIATE',
                    'rationale': 'High conviction warrants larger allocation'
                }
            ],
            'MEDIUM': [
                {
                    'strategy': 'SCALE_IN',
                    'description': 'Build position over 2-3 entries',
                    'timing': 'GRADUAL',
                    'rationale': 'Reduce timing risk while building exposure'
                },
                {
                    'strategy': 'WAIT_FOR_PULLBACK',
                    'description': 'Wait for 3-5% pullback to support',
                    'timing': 'PATIENT',
                    'rationale': 'Better risk/reward by waiting for dip'
                },
                {
                    'strategy': 'MODERATE_POSITION',
                    'description': 'Position size 5-8% of portfolio',
                    'timing': 'GRADUAL',
                    'rationale': 'Moderate conviction = moderate sizing'
                }
            ],
            'LOW': [
                {
                    'strategy': 'PILOT_POSITION',
                    'description': 'Small starter position only',
                    'timing': 'WATCH',
                    'rationale': 'Low conviction - wait for more evidence'
                },
                {
                    'strategy': 'WAIT_FOR_CATALYST',
                    'description': 'Wait for clear catalyst before entry',
                    'timing': 'PATIENT',
                    'rationale': 'Need better setup or confirmation'
                },
                {
                    'strategy': 'SMALL_POSITION',
                    'description': 'Position size 2-3% maximum',
                    'timing': 'GRADUAL',
                    'rationale': 'Limited conviction = limited exposure'
                }
            ]
        }
        
        return strategies.get(conviction, strategies['LOW'])
    
    def build_bull_thesis(self, opportunity_categories: Dict, risk_reward: Dict) -> str:
        """Build core bull thesis statement"""
        components = []
        
        if opportunity_categories['technical']:
            components.append("Technical momentum and trend structure favor further upside")
        if opportunity_categories['fundamental']:
            components.append("Fundamental strength supports higher valuation levels")
        if opportunity_categories['sentiment']:
            components.append("Positive sentiment shift creating buying pressure")
        if opportunity_categories['macro']:
            components.append("Macroeconomic tailwinds provide systematic support")
        
        if not components:
            components.append("Multiple positive factors align for potential appreciation")
        
        thesis = f"""The bull case for {self.ticker} rests on {len(components)} key pillars:

{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(components))}

**Risk/Reward Analysis:**
- Upside Potential: {risk_reward['upside_potential']}
- Downside Risk: {risk_reward['downside_risk']}
- Reward/Risk Ratio: {risk_reward['reward_risk_ratio']:.1f}:1
- Conviction: {risk_reward['conviction_level']} ({risk_reward['bull_percentage']:.0f}% bullish/neutral)
"""
        return thesis
    
    def research_shallow(self, discussion_points: Dict) -> str:
        """
        SHALLOW MODE: Quick single-pass bull case
        Fast analysis, minimal LLM calls
        """
        print(f"\n[BULL] 📊 SHALLOW MODE - Quick bull case\n")
        
        start_time = time.time()
        
        # Quick extraction
        opportunity_categories = self.extract_bull_signals_from_full_reports(discussion_points)
        risk_reward = self.calculate_risk_reward(discussion_points)
        catalysts = self.identify_upside_catalysts(discussion_points)[:3]
        entry_strategies = self.suggest_entry_strategies(risk_reward)[:2]
        
        # Build thesis
        core_thesis = self.build_bull_thesis(opportunity_categories, risk_reward)
        
        bull_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': 'SHALLOW',
            'core_thesis': core_thesis,
            'opportunities': opportunity_categories,
            'catalysts': catalysts,
            'risk_reward': risk_reward,
            'entry_strategies': entry_strategies,
            'discussion_summary': discussion_points.get('summary', {})
        }
        
        self.bull_thesis = bull_data
        
        # Quick synthesis
        if self.client:
            report = self._quick_synthesis(bull_data, discussion_points)
        else:
            report = self._create_fallback_report(bull_data)
        
        elapsed = time.time() - start_time
        print(f"\n[BULL] ✓ Shallow analysis in {elapsed:.2f}s\n")
        
        return report
    
    def research_deep(self, discussion_points: Dict, rounds: int = 3, bear_thesis: Optional[str] = None) -> str:
        """
        DEEP MODE: Multi-round debate with bear analyst
        Iterative refinement through argumentation
        """
        print(f"\n[BULL] 🎯 DEEP MODE - {rounds}-round debate\n")
        
        start_time = time.time()
        
        # Extract comprehensive data
        opportunity_categories = self.extract_bull_signals_from_full_reports(discussion_points)
        rebuttals = self.counter_bear_arguments(discussion_points)
        catalysts = self.identify_upside_catalysts(discussion_points)
        risk_reward = self.calculate_risk_reward(discussion_points)
        entry_strategies = self.suggest_entry_strategies(risk_reward)
        
        # Build initial thesis
        core_thesis = self.build_bull_thesis(opportunity_categories, risk_reward)
        
        # Debate rounds
        debate_history = []
        current_argument = ""
        
        for round_num in range(1, rounds + 1):
            print(f"[BULL] 🔄 Debate Round {round_num}/{rounds}")
            
            if round_num == 1:
                # Initial bull argument
                current_argument = self._generate_initial_argument(
                    core_thesis, opportunity_categories, catalysts,
                    risk_reward, discussion_points
                )
            else:
                # Respond to bear's counter
                current_argument = self._generate_debate_response(
                    round_num, rounds, debate_history,
                    bear_thesis, opportunity_categories, discussion_points
                )
            
            debate_history.append({
                'round': round_num,
                'speaker': 'bull',
                'argument': current_argument
            })
            
            print(f"[BULL] ✓ Round {round_num} complete ({len(current_argument)} chars)")
        
        # Compile final thesis
        bull_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': 'DEEP',
            'rounds': rounds,
            'core_thesis': core_thesis,
            'opportunities': opportunity_categories,
            'bear_rebuttals': rebuttals,
            'catalysts': catalysts,
            'risk_reward': risk_reward,
            'entry_strategies': entry_strategies,
            'debate_history': debate_history,
            'final_argument': current_argument
        }
        
        self.bull_thesis = bull_data
        
        # Create final report
        report = self._create_debate_report(bull_data)
        
        elapsed = time.time() - start_time
        print(f"\n[BULL] ✓ Deep analysis in {elapsed:.2f}s ({rounds} rounds)\n")
        
        return report
    
    def research_comprehensive(self, discussion_points: Dict, rounds: int = 5, bear_thesis: Optional[str] = None) -> str:
        """
        RESEARCH MODE: Maximum depth with extended debate
        Most thorough analysis, highest LLM usage
        """
        print(f"\n[BULL] 🔬 RESEARCH MODE - Comprehensive {rounds}-round analysis\n")
        
        return self.research_deep(discussion_points, rounds=rounds, bear_thesis=bear_thesis)
    
    def _generate_initial_argument(
        self,
        core_thesis: str,
        opportunity_categories: Dict,
        catalysts: List[Dict],
        risk_reward: Dict,
        discussion_points: Dict
    ) -> str:
        """Generate Round 1 bull argument"""
        
        if not self.client:
            return core_thesis
        
        full_reports = discussion_points.get('full_analyst_reports', {})
        
        context = f"""# Initial Bull Argument for {self.ticker}

## Core Thesis
{core_thesis}

## Full Analyst Reports (Excerpts)
Technical: {full_reports.get('technical', '')[:1000]}...
Fundamental: {full_reports.get('fundamental', '')[:1000]}...
News: {full_reports.get('news', '')[:800]}...
Macro: {full_reports.get('macro', '')[:800]}...

## Opportunity Factors
{json.dumps(opportunity_categories, indent=2)}

## Upside Catalysts
{json.dumps(catalysts, indent=2)}

## Risk/Reward
{json.dumps(risk_reward, indent=2)}

Build your opening bull argument. Be compelling, specific, and data-driven."""
        
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
            print(f"[BULL] ⚠️  LLM error in round 1: {e}")
            return core_thesis
    
    def _generate_debate_response(
        self,
        round_num: int,
        total_rounds: int,
        debate_history: List[Dict],
        bear_argument: Optional[str],
        opportunity_categories: Dict,
        discussion_points: Dict
    ) -> str:
        """Generate debate response in later rounds"""
        
        if not self.client or not bear_argument:
            return "Bull maintains position based on identified opportunities."
        
        # Format history
        history_str = "\n\n".join([
            f"Round {d['round']} ({d['speaker']}): {d['argument'][:500]}..."
            for d in debate_history[-2:]
        ])
        
        instructions = f"""Counter the bear's latest argument while strengthening your bull case.

Focus on:
1. Refuting bear's specific concerns with data
2. Highlighting opportunities they're missing
3. Reinforcing your strongest bull points
4. Providing new positive evidence from reports

Be persuasive but data-driven."""
        
        context = f"""# Debate Round {round_num}/{total_rounds}

## Previous Debate
{history_str}

## Bear's Latest Argument
{bear_argument[:1000]}

## Your Opportunity Analysis
{json.dumps(opportunity_categories, indent=2)[:2000]}

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
            print(f"[BULL] ⚠️  LLM error in round {round_num}: {e}")
            return f"Bull maintains opportunity-focused position (Round {round_num} error)"
    
    def _quick_synthesis(self, bull_data: Dict, discussion_points: Dict) -> str:
        """Quick synthesis for shallow mode"""
        
        full_reports = discussion_points.get('full_analyst_reports', {})
        
        context = f"""Create a compelling bull case for {self.ticker}:

## Core Thesis
{bull_data['core_thesis']}

## Full Analyst Reports (Key Excerpts)
Technical: {full_reports.get('technical', '')[:800]}
Fundamental: {full_reports.get('fundamental', '')[:800]}

## Risk/Reward
{json.dumps(bull_data['risk_reward'], indent=2)}

## Catalysts
{json.dumps(bull_data['catalysts'], indent=2)}

Provide a compelling but concise bull case (500-1000 words)."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.base_system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=1800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[BULL] ⚠️  Synthesis error: {e}")
            return self._create_fallback_report(bull_data)
    
    def _create_debate_report(self, bull_data: Dict) -> str:
        """Format debate-style report"""
        report = f"""
# BULL RESEARCH REPORT: {self.ticker}
{'='*70}
**Mode:** {bull_data['mode']} ({bull_data['rounds']} rounds)
**Generated:** {bull_data['timestamp']}

## Core Bull Thesis
{bull_data['core_thesis']}

## Debate Evolution
"""
        for entry in bull_data['debate_history']:
            report += f"\n### Round {entry['round']} - Bull Argument\n"
            report += f"{entry['argument'][:800]}...\n"
        
        report += f"""
## Risk/Reward Summary
- Upside: {bull_data['risk_reward']['upside_potential']}
- Downside: {bull_data['risk_reward']['downside_risk']}
- R/R Ratio: **{bull_data['risk_reward']['reward_risk_ratio']:.1f}:1**
- Conviction: {bull_data['risk_reward']['conviction_level']}

## Recommended Entry Strategy
"""
        for strat in bull_data['entry_strategies']:
            report += f"\n**{strat['strategy']}** [{strat['timing']}]\n"
            report += f"- {strat['description']}\n"
            report += f"- {strat['rationale']}\n"
        
        report += f"\n{'='*70}\n"
        report += f"BULL CASE STRENGTH: {bull_data['risk_reward']['conviction_level']} - Confidence: {bull_data['risk_reward']['conviction_level']}\n"
        
        return report
    
    def _create_fallback_report(self, bull_data: Dict) -> str:
        """Fallback without LLM"""
        report = f"""
# BULL RESEARCH REPORT: {self.ticker}
{'='*70}
*Fallback Mode - LLM Unavailable*

## Core Thesis
{bull_data['core_thesis']}

## Opportunity Factors
"""
        for category, opps in bull_data.get('opportunities', {}).items():
            if opps:
                report += f"\n### {category.title()} Opportunities:\n"
                for opp in opps[:5]:
                    report += f"- {opp}\n"
        
        report += f"""
## Risk/Reward
- Upside: {bull_data['risk_reward']['upside_potential']}
- R/R Ratio: {bull_data['risk_reward']['reward_risk_ratio']:.1f}:1
- Conviction: {bull_data['risk_reward']['conviction_level']}

BULL CASE STRENGTH: {bull_data['risk_reward']['conviction_level']} - Confidence: {bull_data['risk_reward']['conviction_level']}
"""
        return report
    
    def research(self, discussion_points: Dict, mode: str = 'shallow', rounds: int = 1, bear_thesis: Optional[str] = None) -> str:
        """
        Main research entry point - dispatches to appropriate mode
        """
        if mode == 'shallow':
            return self.research_shallow(discussion_points)
        elif mode == 'deep':
            return self.research_deep(discussion_points, rounds=rounds, bear_thesis=bear_thesis)
        elif mode == 'research':
            return self.research_comprehensive(discussion_points, rounds=rounds, bear_thesis=bear_thesis)
        else:
            print(f"[BULL] ⚠️  Unknown mode '{mode}', using shallow")
            return self.research_shallow(discussion_points)
    
    def save_thesis(self, filepath: str):
        """Save thesis data"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.bull_thesis, f, indent=2)
            print(f"[BULL] ✓ Saved to {filepath}")
        except Exception as e:
            print(f"[BULL] ⚠️  Save error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Bull Researcher - Multi-mode bullish analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Research Modes:
  shallow  - Quick single-pass (1 LLM call, ~30s)
  deep     - Multi-round debate (3-5 rounds, ~2-3 min)
  research - Comprehensive analysis (5+ rounds, ~5 min)

Examples:
  python bull_researcher.py AAPL --mode shallow
  python bull_researcher.py AAPL --mode deep --rounds 3
  python bull_researcher.py AAPL --mode research --rounds 5 --output bull_thesis.txt
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("--discussion-file", default="../../outputs/discussion_points.json",
                       help="Discussion points JSON")
    parser.add_argument("--mode", choices=['shallow', 'deep', 'research'], default='shallow',
                       help="Analysis depth (default: shallow)")
    parser.add_argument("--rounds", type=int, default=3,
                       help="Debate rounds for deep/research (default: 3)")
    parser.add_argument("--bear-thesis", help="Bear thesis file for debate")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-data", help="Save JSON")
    
    args = parser.parse_args()
    
    try:
        researcher = BullResearcher(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
        # Load discussion
        discussion_points = researcher.load_discussion_points(args.discussion_file)
        if not discussion_points:
            sys.exit(1)
        
        # Load bear thesis if provided
        bear_thesis = None
        if args.bear_thesis and os.path.exists(args.bear_thesis):
            with open(args.bear_thesis, 'r', encoding='utf-8') as f:
                bear_thesis = f.read()
        
        # Run research
        report = researcher.research(
            discussion_points,
            mode=args.mode,
            rounds=args.rounds,
            bear_thesis=bear_thesis
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