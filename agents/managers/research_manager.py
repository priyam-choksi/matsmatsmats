"""
Research Manager - With Internal Debate Orchestration
Loads bull/bear theses, runs debate rounds internally, synthesizes final decision

MODIFIED: Now supports historical backtesting via analysis_date parameter

Usage: 
  python research_manager.py AAPL --bull-file ... --bear-file ...
  python research_manager.py AAPL --bull-file ... --bear-file ... --debate-rounds 3
  python research_manager.py AAPL --bull-file ... --bear-file ... --analysis-date 2024-06-15
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


class ResearchManager:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):  # <-- NEW PARAMETER
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # === HISTORICAL BACKTESTING SUPPORT ===
        self.analysis_date = analysis_date  # Format: 'YYYY-MM-DD' or None for current
        
        if self.analysis_date:
            print(f"[RESEARCH_MGR] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[RESEARCH_MGR] Running in LIVE mode (current data)")
        
        # Debate prompts
        self.bull_debate_prompt = """You are a Bull Analyst in Round {round_num} of an investment debate for {ticker}.

Your core thesis: {thesis}

**CRITICAL DEBATE RULES:**
1. You MUST directly address the Bear's specific arguments
2. Quote or reference their exact claims before countering
3. Use data and numbers to refute their points
4. Acknowledge valid concerns but explain why upside still dominates
5. Bring NEW evidence, don't just repeat yourself

Respond in 400-600 words. Be specific and persuasive.
End with your strongest conviction point."""

        self.bear_debate_prompt = """You are a Bear Analyst in Round {round_num} of an investment debate for {ticker}.

Your core thesis: {thesis}

**CRITICAL DEBATE RULES:**
1. You MUST directly address the Bull's specific arguments
2. Quote or reference their exact claims before countering
3. Use data and numbers to refute their points
4. Acknowledge valid points but explain why risks still dominate
5. Bring NEW evidence, don't just repeat yourself

Respond in 400-600 words. Be specific and persuasive.
End with your key risk concern."""

        self.moderator_prompt = """You are the Research Manager - an objective debate moderator and final decision maker.

**YOUR CRITICAL ROLE:**
1. Evaluate the debate objectively - the strongest arguments win
2. Make a DECISIVE recommendation (avoid defaulting to HOLD)
3. Weigh probability-adjusted outcomes
4. Provide clear, actionable investment guidance

**DECISION FRAMEWORK:**

**BUY when:**
- Bull arguments are significantly stronger
- Risk/reward ratio is favorable (>2:1)
- Catalysts outweigh concerns

**SELL when:**
- Bear arguments are significantly stronger
- Risk/reward is unfavorable
- Downside triggers are imminent

**HOLD only when:**
- Arguments are genuinely balanced (rare)
- Need specific catalyst for clarity

**Be decisive. The best evidence wins.**

End with: RESEARCH CONCLUSION: Strong Buy/Buy/Hold/Sell/Strong Sell - Confidence: High/Medium/Low"""

        # Storage
        self.research_inputs = {
            'bull_thesis': {},
            'bear_thesis': {},
            'debate_history': [],
            'risk_evaluations': {}
        }
    
    def load_research_files(self, bull_file: str, bear_file: str):
        """Load bull and bear thesis files"""
        print(f"[RESEARCH_MGR] Loading research files...")
        
        if os.path.exists(bull_file):
            with open(bull_file, 'r', encoding='utf-8') as f:
                self.research_inputs['bull_thesis'] = json.load(f)
            print(f"[RESEARCH_MGR] ✓ Bull thesis loaded")
            
            # NEW: Check for historical date in loaded data
            if self.research_inputs['bull_thesis'].get('analysis_date'):
                print(f"[RESEARCH_MGR]   Bull thesis from: {self.research_inputs['bull_thesis'].get('analysis_date')}")
        else:
            print(f"[RESEARCH_MGR] ⚠️ Bull thesis not found: {bull_file}")
        
        if os.path.exists(bear_file):
            with open(bear_file, 'r', encoding='utf-8') as f:
                self.research_inputs['bear_thesis'] = json.load(f)
            print(f"[RESEARCH_MGR] ✓ Bear thesis loaded")
            
            # NEW: Check for historical date in loaded data
            if self.research_inputs['bear_thesis'].get('analysis_date'):
                print(f"[RESEARCH_MGR]   Bear thesis from: {self.research_inputs['bear_thesis'].get('analysis_date')}")
        else:
            print(f"[RESEARCH_MGR] ⚠️ Bear thesis not found: {bear_file}")
    
    def load_risk_evaluations(self, outputs_dir: str = "../../outputs"):
        """Load risk team evaluations if available"""
        print(f"[RESEARCH_MGR] Loading risk evaluations...")
        
        risk_files = {
            'aggressive': os.path.join(outputs_dir, "aggressive_eval.json"),
            'neutral': os.path.join(outputs_dir, "neutral_eval.json"),
            'conservative': os.path.join(outputs_dir, "conservative_eval.json")
        }
        
        loaded = 0
        for risk_type, filepath in risk_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.research_inputs['risk_evaluations'][risk_type] = json.load(f)
                    loaded += 1
                except Exception as e:
                    print(f"[RESEARCH_MGR] ⚠️ Error loading {risk_type}: {e}")
        
        print(f"[RESEARCH_MGR] ✓ Loaded {loaded}/3 risk evaluations")

    # ==================== INTERNAL DEBATE ENGINE ====================
    
    def run_debate(self, rounds: int = 3) -> List[Dict]:
        """
        Run internal debate between bull and bear positions.
        Each round: Bull argues → Bear responds → Bull responds → etc.
        """
        if not self.client:
            print("[RESEARCH_MGR] ⚠️ No API client available, skipping debate")
            return []
        
        bull_thesis = self.research_inputs.get('bull_thesis', {})
        bear_thesis = self.research_inputs.get('bear_thesis', {})
        
        if not bull_thesis or not bear_thesis:
            print("[RESEARCH_MGR] ⚠️ Missing thesis data, skipping debate")
            return []
        
        # Get core theses for context
        bull_core = bull_thesis.get('core_thesis', '') or bull_thesis.get('full_analysis', '')[:1500]
        bear_core = bear_thesis.get('core_thesis', '') or bear_thesis.get('full_analysis', '')[:1500]
        bull_full = bull_thesis.get('full_analysis', bull_core)
        bear_full = bear_thesis.get('full_analysis', bear_core)
        
        print(f"\n{'='*70}")
        print(f"INVESTMENT DEBATE: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*70}")
        print(f"Rounds: {rounds}")
        print(f"Model: {self.model}")
        print(f"{'='*70}\n")
        
        debate_history = []
        
        for round_num in range(1, rounds + 1):
            print(f"\n--- ROUND {round_num}/{rounds} ---\n")
            
            # === BULL'S TURN ===
            print(f"[🐂 BULL] Generating argument...")
            
            if round_num == 1:
                # Opening argument
                bull_arg = self._generate_opening('bull', bull_full, bear_core)
            else:
                # Respond to bear's last argument
                bear_last = self._get_last_argument(debate_history, 'bear')
                bull_arg = self._generate_rebuttal('bull', bull_core, bear_last, debate_history, round_num)
            
            debate_history.append({
                'round': round_num,
                'side': 'bull',
                'argument': bull_arg,
                'timestamp': datetime.now().isoformat()
            })
            print(f"[🐂 BULL] ✓ Complete ({len(bull_arg)} chars)")
            
            # === BEAR'S TURN ===
            print(f"[🐻 BEAR] Generating response...")
            
            if round_num == 1:
                # Opening response to bull
                bear_arg = self._generate_opening('bear', bear_full, bull_arg)
            else:
                # Respond to bull's argument from this round
                bear_arg = self._generate_rebuttal('bear', bear_core, bull_arg, debate_history, round_num)
            
            debate_history.append({
                'round': round_num,
                'side': 'bear',
                'argument': bear_arg,
                'timestamp': datetime.now().isoformat()
            })
            print(f"[🐻 BEAR] ✓ Complete ({len(bear_arg)} chars)")
            
            # Small delay for rate limiting
            time.sleep(0.3)
        
        self.research_inputs['debate_history'] = debate_history
        
        print(f"\n{'='*70}")
        print(f"DEBATE COMPLETE: {len(debate_history)} arguments over {rounds} rounds")
        print(f"{'='*70}\n")
        
        return debate_history
    
    def _generate_opening(self, side: str, full_analysis: str, opponent_thesis: str) -> str:
        """Generate opening argument for a side"""
        
        # === ADD HISTORICAL DATE CONTEXT ===
        date_context = ""
        if self.analysis_date:
            date_context = f"""
**⚠️ HISTORICAL ANALYSIS MODE: {self.analysis_date} ⚠️**
All data is from this date. Do NOT reference events after {self.analysis_date}.

"""
        
        if side == 'bull':
            system = f"""You are the Bull Analyst presenting your opening argument for {self.ticker}.
Be compelling, specific, and data-driven. Address potential bear concerns preemptively."""
            
            context = f"""{date_context}## Your Complete Bull Analysis:
{full_analysis[:2500]}

## Bear's Thesis (you'll be debating against):
{opponent_thesis[:1000]}

---
Present your OPENING bull case. Reference specific data points, price levels, and catalysts.
Preemptively address the bear's main concerns.
End with your strongest conviction point."""

        else:  # bear
            system = f"""You are the Bear Analyst responding to the Bull's opening for {self.ticker}.
Be compelling, specific, and data-driven. Directly counter the bull's key points."""
            
            context = f"""{date_context}## Your Complete Bear Analysis:
{full_analysis[:2500]}

## Bull's Opening Argument (you must respond to this):
{opponent_thesis[:1500]}

---
DIRECTLY counter the Bull's specific claims, then present your bear case.
Reference specific data points, risk levels, and downside triggers.
End with your key risk concern."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_completion_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[RESEARCH_MGR] ⚠️ LLM error ({side} opening): {e}")
            return f"{side.title()}'s opening argument (LLM error: {e})"
    
    def _generate_rebuttal(self, side: str, my_thesis: str, opponent_arg: str, 
                           history: List[Dict], round_num: int) -> str:
        """Generate rebuttal to opponent's argument"""
        
        # Format recent debate history
        recent = history[-4:] if len(history) > 4 else history
        history_str = "\n\n".join([
            f"**Round {h['round']} - {h['side'].upper()}:**\n{h['argument'][:400]}..."
            for h in recent
        ])
        
        # === ADD HISTORICAL DATE CONTEXT ===
        date_note = ""
        if self.analysis_date:
            date_note = f"\n**HISTORICAL MODE: {self.analysis_date}** - Do not reference future events.\n"
        
        if side == 'bull':
            system = self.bull_debate_prompt.format(
                round_num=round_num,
                ticker=self.ticker,
                thesis=my_thesis[:800]
            ) + date_note
        else:
            system = self.bear_debate_prompt.format(
                round_num=round_num,
                ticker=self.ticker,
                thesis=my_thesis[:800]
            ) + date_note
        
        context = f"""## Debate History:
{history_str}

## OPPONENT'S LATEST ARGUMENT (YOU MUST RESPOND TO THIS):
{opponent_arg}

---
Directly address their specific claims first, then reinforce your position with new evidence."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_completion_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[RESEARCH_MGR] ⚠️ LLM error ({side} round {round_num}): {e}")
            return f"{side.title()}'s rebuttal for round {round_num} (LLM error)"
    
    def _get_last_argument(self, history: List[Dict], side: str) -> str:
        """Get the last argument from specified side"""
        for entry in reversed(history):
            if entry['side'] == side:
                return entry['argument']
        return ""

    # ==================== SYNTHESIS ====================
    
    def format_debate_for_synthesis(self) -> str:
        """Format the complete debate for final synthesis"""
        debate = self.research_inputs.get('debate_history', [])
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        # === ADD HISTORICAL DATE HEADER ===
        date_header = ""
        if self.analysis_date:
            date_header = f"""
**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date} ⚠️**
All data and arguments are based on information available on this date.

"""
        
        formatted = f"""# Investment Debate Transcript: {self.ticker}
{'='*60}
{date_header}
"""
        
        if debate:
            # We have actual debate history
            for entry in debate:
                icon = "🐂 BULL" if entry['side'] == 'bull' else "🐻 BEAR"
                formatted += f"## Round {entry['round']} - {icon}\n\n{entry['argument']}\n\n{'─'*40}\n\n"
        else:
            # No debate - use original theses
            formatted += f"""## BULL POSITION (No Debate)
{bull.get('full_analysis', bull.get('core_thesis', 'Not available'))[:2000]}

{'─'*40}

## BEAR POSITION (No Debate)
{bear.get('full_analysis', bear.get('core_thesis', 'Not available'))[:2000]}
"""
        
        # Add quantitative data
        formatted += f"""
{'='*60}
# QUANTITATIVE ASSESSMENTS

## Bull Risk/Reward:
{json.dumps(bull.get('risk_reward', {}), indent=2)}

## Bear Risk Assessment:
{json.dumps(bear.get('risk_assessment', {}), indent=2)}

## Key Bull Catalysts:
{json.dumps(bull.get('catalysts', [])[:3], indent=2)}

## Key Bear Triggers:
{json.dumps(bear.get('downside_triggers', [])[:3], indent=2)}
"""
        return formatted
    
    def calculate_probabilities(self) -> Dict[str, Any]:
        """Calculate scenario probabilities based on conviction levels"""
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        # Get conviction levels
        bull_conv = bull.get('risk_reward', {}).get('conviction_level', 'MEDIUM')
        bear_conv = bear.get('risk_assessment', {}).get('conviction_level', 'MEDIUM')
        
        # Score based on conviction
        conv_scores = {'HIGH': 0.75, 'MEDIUM': 0.50, 'LOW': 0.30}
        bull_score = conv_scores.get(bull_conv, 0.5)
        bear_score = conv_scores.get(bear_conv, 0.5)
        
        # Factor in risk evaluations if available
        for risk_type, eval_data in self.research_inputs.get('risk_evaluations', {}).items():
            stance = eval_data.get('stance', '')
            weight = 0.1  # Each risk eval adds 10% weight
            if 'BUY' in stance:
                bull_score += weight
            elif 'SELL' in stance or 'AVOID' in stance:
                bear_score += weight
        
        # Normalize to probabilities (leaving room for base case)
        total = bull_score + bear_score
        if total > 0:
            bull_prob = (bull_score / total) * 75  # Max 75% to leave room for base
            bear_prob = (bear_score / total) * 75
        else:
            bull_prob = bear_prob = 37.5
        
        base_prob = 100 - bull_prob - bear_prob
        
        return {
            'bull_case': round(bull_prob, 1),
            'bear_case': round(bear_prob, 1),
            'base_case': round(base_prob, 1),
            'scenarios': [
                {
                    'name': 'Bull Case',
                    'probability': f"{bull_prob:.0f}%",
                    'outcome': bull.get('risk_reward', {}).get('upside_potential', '20-30%')
                },
                {
                    'name': 'Bear Case', 
                    'probability': f"{bear_prob:.0f}%",
                    'outcome': bear.get('risk_assessment', {}).get('downside_risk', '15-20%')
                },
                {
                    'name': 'Base Case',
                    'probability': f"{base_prob:.0f}%",
                    'outcome': 'Sideways (±5%)'
                }
            ]
        }
    
    def analyze_consensus(self) -> Dict[str, Any]:
        """Analyze consensus between bull and bear positions with improved extraction"""
        consensus = {
            'recommendations': {},
            'conviction_levels': {},
            'key_agreements': [],
            'key_conflicts': []
        }
               
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        # Bull recommendation
        rr = bull.get('risk_reward', {})
        if rr.get('reward_risk_ratio', 0) >= 2:
            consensus['recommendations']['bull'] = 'BUY'
        else:
            consensus['recommendations']['bull'] = 'HOLD'
        consensus['conviction_levels']['bull'] = rr.get('conviction_level', 'MEDIUM')
        
        # Bear recommendation
        ra = bear.get('risk_assessment', {})
        if ra.get('risk_level') == 'HIGH':
            consensus['recommendations']['bear'] = 'SELL'
        else:
            consensus['recommendations']['bear'] = 'HOLD'
        consensus['conviction_levels']['bear'] = ra.get('conviction_level', 'MEDIUM')
        
        # Risk evaluations
        for risk_type, eval_data in self.research_inputs.get('risk_evaluations', {}).items():
            consensus['recommendations'][risk_type] = eval_data.get('stance', 'HOLD')
            consensus['conviction_levels'][risk_type] = eval_data.get('confidence', 'MEDIUM')
        
        # =====================================================================
        # NEW: Extract key agreements from debate history
        # =====================================================================
        debate_history = self.research_inputs.get('debate_history', [])
        
        # Common themes that both sides might acknowledge
        bull_points = []
        bear_points = []
        
        for entry in debate_history:
            text = entry.get('argument', '').lower()
            side = entry.get('side', '')
            
            if side == 'bull':
                bull_points.append(text)
            elif side == 'bear':
                bear_points.append(text)
        
        # Find agreements - themes mentioned positively by both sides
        agreement_keywords = {
            'strong fundamentals': ['strong fundamental', 'solid fundamental', 'robust fundamental'],
            'growth potential': ['growth potential', 'growth trajectory', 'revenue growth'],
            'market leader': ['market leader', 'leading position', 'dominant position'],
            'cash flow': ['cash flow', 'free cash flow', 'cash generation'],
            'valuation concerns': ['valuation', 'overvalued', 'premium valuation', 'high p/e'],
            'competitive pressure': ['competition', 'competitive', 'competitors'],
            'macro risks': ['macro', 'interest rate', 'economic', 'recession'],
            'technical overbought': ['overbought', 'rsi above', 'pullback']
        }
        
        bull_text = ' '.join(bull_points)
        bear_text = ' '.join(bear_points)
        
        for theme, keywords in agreement_keywords.items():
            bull_mentions = any(kw in bull_text for kw in keywords)
            bear_mentions = any(kw in bear_text for kw in keywords)
            
            if bull_mentions and bear_mentions:
                consensus['key_agreements'].append({
                    'topic': theme,
                    'description': f"Both sides acknowledge {theme}",
                    'bull_acknowledges': True,
                    'bear_acknowledges': True
                })
        
        # =====================================================================
        # NEW: Extract key conflicts from debate
        # =====================================================================
        
        # Direct recommendation conflict
        recs = list(consensus['recommendations'].values())
        if 'BUY' in recs and ('SELL' in recs or 'AVOID' in recs):
            consensus['key_conflicts'].append({
                'type': 'recommendation',
                'severity': 'HIGH',
                'description': "Direct BUY vs SELL/AVOID conflict between analysts",
                'resolution_needed': True
            })
        
        # Valuation interpretation conflict
        bull_sees_value = any(term in bull_text for term in ['undervalued', 'fair value', 'justified', 'reasonable'])
        bear_sees_overvalue = any(term in bear_text for term in ['overvalued', 'expensive', 'premium', 'stretched'])
        
        if bull_sees_value and bear_sees_overvalue:
            consensus['key_conflicts'].append({
                'type': 'valuation',
                'severity': 'MEDIUM',
                'description': "Disagreement on valuation: Bull sees fair value, Bear sees overvaluation",
                'resolution_needed': True
            })
        
        # Technical interpretation conflict
        bull_bullish_tech = any(term in bull_text for term in ['uptrend', 'bullish momentum', 'breakout'])
        bear_bearish_tech = any(term in bear_text for term in ['pullback', 'correction', 'breakdown', 'overbought'])
        
        if bull_bullish_tech and bear_bearish_tech:
            consensus['key_conflicts'].append({
                'type': 'technical',
                'severity': 'MEDIUM',
                'description': "Disagreement on technicals: Bull sees continuation, Bear sees reversal risk",
                'resolution_needed': True
            })
        
        # Risk tolerance conflict
        bull_risk_ok = any(term in bull_text for term in ['manageable risk', 'acceptable', 'limited downside'])
        bear_risk_high = any(term in bear_text for term in ['excessive risk', 'significant downside', 'major risk'])
        
        if bull_risk_ok and bear_risk_high:
            consensus['key_conflicts'].append({
                'type': 'risk_assessment',
                'severity': 'HIGH',
                'description': "Disagreement on risk: Bull sees manageable, Bear sees excessive",
                'resolution_needed': True
            })
        
        # Fallback if no conflicts found through keyword analysis
        if not consensus['key_conflicts'] and 'BUY' in recs and 'HOLD' in recs:
            consensus['key_conflicts'].append({
                'type': 'conviction',
                'severity': 'LOW',
                'description': "Mixed conviction levels between analysts",
                'resolution_needed': False
            })
        
        # Fallback if no agreements found
        if not consensus['key_agreements']:
            consensus['key_agreements'].append({
                'topic': 'data_quality',
                'description': "Both sides using same underlying data sources",
                'bull_acknowledges': True,
                'bear_acknowledges': True
            })
        
        return consensus
        
    def form_conclusion(self, probabilities: Dict, consensus: Dict) -> Dict[str, Any]:
        """Form final investment conclusion"""
        bull_prob = probabilities['bull_case']
        bear_prob = probabilities['bear_case']
        
        # Decision logic - be decisive
        if bull_prob >= 55:
            rec = 'BUY'
            conf = 'HIGH' if bull_prob >= 65 else 'MEDIUM'
            rationale = f"Bull case dominates ({bull_prob:.0f}% vs {bear_prob:.0f}%)"
        elif bear_prob >= 55:
            rec = 'SELL'
            conf = 'HIGH' if bear_prob >= 65 else 'MEDIUM'
            rationale = f"Bear case dominates ({bear_prob:.0f}% vs {bull_prob:.0f}%)"
        elif bull_prob > bear_prob + 5:
            rec = 'BUY'
            conf = 'LOW'
            rationale = f"Slight bull edge ({bull_prob:.0f}% vs {bear_prob:.0f}%)"
        elif bear_prob > bull_prob + 5:
            rec = 'SELL'
            conf = 'LOW'
            rationale = f"Slight bear edge ({bear_prob:.0f}% vs {bull_prob:.0f}%)"
        else:
            rec = 'HOLD'
            conf = 'LOW'
            rationale = f"Balanced probabilities ({bull_prob:.0f}% vs {bear_prob:.0f}%)"
        
        # Position sizing
        if rec == 'BUY':
            position = "10-15%" if conf == 'HIGH' else "5-10%" if conf == 'MEDIUM' else "3-5%"
        elif rec == 'SELL':
            position = "0% (Exit/Avoid)"
        else:
            position = "2-5% (Maintain if held)"
        
        return {
            'recommendation': rec,
            'confidence': conf,
            'rationale': rationale,
            'position_size': position,
            'time_horizon': '3-6 months'
        }
    
    def generate_synthesis(self, debate_formatted: str, probabilities: Dict, 
                           consensus: Dict, conclusion: Dict) -> str:
        """Generate final synthesis report with LLM"""
        if not self.client:
            return self._fallback_report(probabilities, consensus, conclusion)
        
        # === ADD HISTORICAL DATE CONTEXT ===
        date_context = ""
        if self.analysis_date:
            date_context = f"""
**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date} ⚠️**
All debate arguments and data are from this date.
Make your decision as if you were deciding ON {self.analysis_date}.
Do NOT reference any events after this date.

"""
        
        context = f"""{date_context}{debate_formatted}

{'='*60}
# PRELIMINARY ANALYSIS

## Probability Assessment:
{json.dumps(probabilities, indent=2)}

## Consensus Analysis:
{json.dumps(consensus, indent=2)}

## Initial Conclusion:
{json.dumps(conclusion, indent=2)}

{'='*60}

As Research Manager, evaluate this debate and provide your FINAL DECISION.
- Who presented stronger arguments?
- Which side had better data support?
- What is your definitive recommendation?

Be thorough but decisive."""

        try:
            print(f"[RESEARCH_MGR] Generating final synthesis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.moderator_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.5,  # Lower temp for more decisive output
                max_completion_tokens=2500
            )
            
            result = response.choices[0].message.content
            
            # Ensure conclusion is present
            if "RESEARCH CONCLUSION:" not in result:
                result += f"\n\nRESEARCH CONCLUSION: {conclusion['recommendation']} - Confidence: {conclusion['confidence']}"
            
            return result
            
        except Exception as e:
            print(f"[RESEARCH_MGR] ❌ Synthesis error: {e}")
            return self._fallback_report(probabilities, consensus, conclusion)
    
    def _fallback_report(self, probabilities: Dict, consensus: Dict, conclusion: Dict) -> str:
        """Fallback report when LLM unavailable"""
        debate_conducted = len(self.research_inputs.get('debate_history', [])) > 0
        
        # === ADD HISTORICAL DATE INFO ===
        date_info = ""
        if self.analysis_date:
            date_info = f"*Historical analysis as of {self.analysis_date}*\n"
        
        return f"""
# RESEARCH SYNTHESIS: {self.ticker}
{'='*70}
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
{date_info}*Debate Conducted: {'Yes' if debate_conducted else 'No'}*

## Probability Assessment

| Scenario | Probability | Expected Outcome |
|----------|-------------|------------------|
| Bull Case | {probabilities['bull_case']:.0f}% | {probabilities['scenarios'][0]['outcome']} |
| Bear Case | {probabilities['bear_case']:.0f}% | {probabilities['scenarios'][1]['outcome']} |
| Base Case | {probabilities['base_case']:.0f}% | {probabilities['scenarios'][2]['outcome']} |

## Consensus Analysis

**Recommendations:**
{chr(10).join(f"- {k}: {v}" for k, v in consensus['recommendations'].items())}

**Conflicts:** {consensus['key_conflicts'][0] if consensus['key_conflicts'] else 'None identified'}

## Final Decision

**Recommendation:** {conclusion['recommendation']}
**Confidence:** {conclusion['confidence']}
**Position Size:** {conclusion['position_size']}
**Time Horizon:** {conclusion['time_horizon']}

**Rationale:** {conclusion['rationale']}

{'='*70}
RESEARCH CONCLUSION: {conclusion['recommendation']} - Confidence: {conclusion['confidence']}
"""

    def synthesize(self, debate_rounds: int = 0, skip_risk_evals: bool = False, 
                   outputs_dir: str = "../../outputs") -> tuple:
        """
        Main synthesis workflow.
        
        Args:
            debate_rounds: Number of debate rounds (0 = no debate, just synthesis)
            skip_risk_evals: Skip loading risk team evaluations
            outputs_dir: Directory for risk evaluation files
        
        Returns:
            (report_text, synthesis_data_dict)
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"RESEARCH MANAGER: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*70}")
        print(f"Debate Rounds: {debate_rounds if debate_rounds > 0 else 'None (direct synthesis)'}")
        print(f"{'='*70}\n")
        
        # Run internal debate if rounds > 0
        if debate_rounds > 0:
            self.run_debate(rounds=debate_rounds)
        
        # Load risk evaluations
        if not skip_risk_evals:
            self.load_risk_evaluations(outputs_dir)
        
        # Format debate/theses
        debate_formatted = self.format_debate_for_synthesis()
        
        # Calculate metrics
        probabilities = self.calculate_probabilities()
        consensus = self.analyze_consensus()
        conclusion = self.form_conclusion(probabilities, consensus)
        
        # Generate final synthesis
        report = self.generate_synthesis(debate_formatted, probabilities, consensus, conclusion)
        
        # Compile synthesis data
        synthesis_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,  # NEW: Include in output
            'historical_mode': self.analysis_date is not None,  # NEW
            'debate_rounds': debate_rounds,
            'debate_conducted': debate_rounds > 0,
            'probabilities': probabilities,
            'consensus': consensus,
            'conclusion': conclusion,
            'debate_history': self.research_inputs.get('debate_history', []),
            'inputs_summary': {
                'bull_thesis': bool(self.research_inputs.get('bull_thesis')),
                'bear_thesis': bool(self.research_inputs.get('bear_thesis')),
                'risk_evaluations': len(self.research_inputs.get('risk_evaluations', {}))
            }
        }
        
        elapsed = time.time() - start_time
        print(f"\n[RESEARCH_MGR] ✓ Synthesis complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, synthesis_data


def main():
    parser = argparse.ArgumentParser(
        description="Research Manager - Orchestrates debate and synthesizes final investment decision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct synthesis (no debate)
  python research_manager.py AAPL --bull-file ../../outputs/bull_thesis.json --bear-file ../../outputs/bear_thesis.json
  
  # With 3 rounds of debate
  python research_manager.py AAPL --bull-file ... --bear-file ... --debate-rounds 3
  
  # Deep debate (5 rounds)
  python research_manager.py AAPL --bull-file ... --bear-file ... --debate-rounds 5
  
  # HISTORICAL BACKTESTING:
  python research_manager.py AAPL --bull-file ... --bear-file ... --analysis-date 2024-06-15
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--bull-file", default="../../outputs/bull_thesis.json",
                       help="Path to bull thesis JSON")
    parser.add_argument("--bear-file", default="../../outputs/bear_thesis.json",
                       help="Path to bear thesis JSON")
    parser.add_argument("--debate-rounds", type=int, default=0,
                       help="Number of debate rounds (0=no debate, 3=standard, 5=deep)")
    parser.add_argument("--skip-evaluations", action="store_true",
                       help="Skip loading risk team evaluations")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", help="Save report to text file")
    parser.add_argument("--save-synthesis", help="Save synthesis data to JSON")
    
    # ============================================================
    # NEW: Add analysis-date argument for historical backtesting
    # ============================================================
    parser.add_argument("--analysis-date",
                       type=str,
                       default=None,
                       help="Historical analysis date (YYYY-MM-DD format)")
    
    args = parser.parse_args()
    
    try:
        manager = ResearchManager(
            ticker=args.ticker,
            api_key=args.api_key,
            model=args.model,
            analysis_date=args.analysis_date  # NEW: Pass to manager
        )
        
        # Load thesis files
        manager.load_research_files(args.bull_file, args.bear_file)
        
        # Validate we have required data
        if not manager.research_inputs['bull_thesis'] or not manager.research_inputs['bear_thesis']:
            print("\n❌ Error: Both bull and bear thesis files are required")
            print("Run bull_researcher.py and bear_researcher.py first!")
            sys.exit(1)
        
        # Run synthesis
        report, synthesis_data = manager.synthesize(
            debate_rounds=args.debate_rounds,
            skip_risk_evals=args.skip_evaluations
        )
        
        # Print report
        print(report)
        
        # Save outputs
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to {args.output}")
        
        if args.save_synthesis:
            with open(args.save_synthesis, 'w', encoding='utf-8') as f:
                json.dump(synthesis_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Synthesis data saved to {args.save_synthesis}")
        
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