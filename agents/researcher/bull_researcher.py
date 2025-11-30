"""
Bull Researcher - LLM-Driven Analysis
LLM does the heavy lifting: reads reports, extracts insights, decides metrics
Code only provides guardrails to catch unrealistic outputs

Philosophy:
  LLM = Brain (analyzes, decides, reasons)
  Code = Safety net (validates bounds, catches errors)

Usage: 
  python bull_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json
  python bull_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class BullResearcher:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[BULL] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[BULL] Running in LIVE mode (current data)")
        
        # Guardrails - comprehensive validation to catch fabrication and errors
        self.guardrails = {
            # Percentage bounds
            'max_upside_pct': 100,      # Flag if LLM says >100% upside
            'min_upside_pct': 1,        # Flag if <1%
            'max_downside_pct': 50,     # Flag if >50% downside
            'min_downside_pct': 1,      # Flag if <1%
            'max_rr_ratio': 10,         # Flag if R/R > 10:1
            'min_rr_ratio': 0.1,        # Flag if R/R < 0.1:1
            
            # Valid enums
            'valid_convictions': ['HIGH', 'MEDIUM', 'LOW'],
            'valid_recommendations': ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'],
            'valid_signal_strengths': ['strong', 'moderate', 'weak'],
            'valid_impact_levels': ['high', 'medium', 'low'],
            'valid_data_quality': ['strong', 'moderate', 'weak'],
            
            # Consistency rules
            'high_conviction_requires_strong_data': True,
            'strong_buy_requires_min_rr': 1.5,
            'max_signals_without_source': 0,  # All signals must have source
            
            # Required fields
            'required_fields': ['core_thesis', 'risk_reward', 'conviction', 'recommendation'],
            'required_rr_fields': ['upside_pct', 'downside_pct', 'rationale'],
            'required_conviction_fields': ['level', 'data_quality'],
        }
        
        self.bull_thesis = {}
        self.validation_warnings = []  # Track all warnings
    
    def load_discussion_points(self, filepath: str) -> Optional[Dict]:
        """Load discussion points from analysts"""
        print(f"[BULL] Loading discussion points from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[BULL] ✓ Loaded discussion for {data.get('ticker', 'unknown')}")
            
            if data.get('analysis_date'):
                print(f"[BULL] Discussion data is from historical date: {data.get('analysis_date')}")
            
            return data
        except FileNotFoundError:
            print(f"[BULL] ✗ File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"[BULL] ✗ Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"[BULL] ✗ Load error: {e}")
            return None

    def _build_analysis_prompt(self, discussion_points: Dict) -> str:
        """Build the prompt for LLM to analyze and decide everything"""
        full_reports = discussion_points.get('full_analyst_reports', {})
        summary = discussion_points.get('summary', {})
        
        date_context = ""
        if self.analysis_date:
            date_context = f"""
⚠️ HISTORICAL ANALYSIS MODE ⚠️
You are analyzing as of {self.analysis_date}. 
Only use information that would have been available on this date.
Do NOT reference any events after {self.analysis_date}.
"""

        prompt = f"""You are a Senior Bull Analyst building the investment case FOR {self.ticker}.

{date_context}

═══════════════════════════════════════════════════════════════════════
ANALYST REPORTS TO ANALYZE
═══════════════════════════════════════════════════════════════════════

TECHNICAL ANALYSIS:
{full_reports.get('technical', 'No technical report available')}

FUNDAMENTAL ANALYSIS:
{full_reports.get('fundamental', 'No fundamental report available')}

NEWS & SENTIMENT:
{full_reports.get('news', 'No news report available')}

MACRO ENVIRONMENT:
{full_reports.get('macro', 'No macro report available')}

ANALYST SUMMARY:
{json.dumps(summary, indent=2) if summary else 'No summary available'}

═══════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════

Analyze ALL the reports above and build a comprehensive BULL CASE.

You must:
1. READ the reports carefully and identify bullish signals
2. EXTRACT specific data points that support the bull case (prices, levels, metrics)
3. DETERMINE realistic upside potential based on what the data shows
4. ASSESS the quality of evidence (is there strong data or just vague signals?)
5. DECIDE your conviction level based on evidence strength

CRITICAL RULES:
- Base your analysis ONLY on what's in the reports above
- If a report doesn't mention specific price targets, estimate based on technical levels or valuation
- Be realistic - mega-cap stocks don't move 50% in 3 months
- If data is weak/missing, say so and lower your conviction
- Cite specific evidence from the reports

Return your analysis as JSON with this structure:
{{
    "core_thesis": "2-3 sentence summary of why this stock should go up",
    
    "key_bullish_signals": [
        {{"source": "technical/fundamental/news/macro", "signal": "specific finding", "strength": "strong/moderate/weak"}}
    ],
    
    "catalysts": [
        {{"catalyst": "description", "timeline": "when", "impact": "high/medium/low"}}
    ],
    
    "risk_reward": {{
        "current_price": <number or null if not found>,
        "upside_target": <number or null>,
        "downside_support": <number or null>,
        "upside_pct": <your estimate based on data>,
        "downside_pct": <your estimate based on data>,
        "reward_risk_ratio": <calculated or estimated>,
        "rationale": "explain how you arrived at these numbers"
    }},
    
    "conviction": {{
        "level": "HIGH/MEDIUM/LOW",
        "reasoning": "why this conviction level",
        "data_quality": "strong/moderate/weak - how much hard data vs speculation"
    }},
    
    "recommendation": {{
        "action": "STRONG BUY/BUY/HOLD",
        "position_size": "suggested % of portfolio",
        "entry_strategy": "how to enter",
        "time_horizon": "expected holding period"
    }},
    
    "counter_bear_arguments": [
        {{"bear_concern": "what bears might say", "bull_response": "why it's overblown"}}
    ],
    
    "full_analysis": "Your complete written bull case (3-5 paragraphs)"
}}

Return ONLY valid JSON. No markdown, no explanation outside JSON."""

        return prompt

    def _validate_and_fix(self, analysis: Dict, discussion_points: Dict) -> Dict:
        """
        Comprehensive guardrails to catch fabrication, errors, and inconsistencies.
        Fixes what can be fixed, flags what can't.
        """
        print("[BULL] Applying comprehensive guardrails...")
        corrections = []
        warnings = []
        
        # ═══════════════════════════════════════════════════════════════
        # 1. CHECK REQUIRED FIELDS EXIST
        # ═══════════════════════════════════════════════════════════════
        for field in self.guardrails['required_fields']:
            if field not in analysis or not analysis[field]:
                warnings.append(f"MISSING REQUIRED: '{field}' not provided by LLM")
                if field == 'risk_reward':
                    analysis['risk_reward'] = {}
                elif field == 'conviction':
                    analysis['conviction'] = {'level': 'LOW', 'data_quality': 'weak'}
                elif field == 'recommendation':
                    analysis['recommendation'] = {'action': 'HOLD'}
        
        rr = analysis.get('risk_reward', {})
        conviction = analysis.get('conviction', {})
        rec = analysis.get('recommendation', {})
        
        # ═══════════════════════════════════════════════════════════════
        # 2. VALIDATE PERCENTAGE BOUNDS
        # ═══════════════════════════════════════════════════════════════
        upside = rr.get('upside_pct')
        if upside is not None:
            if not isinstance(upside, (int, float)):
                warnings.append(f"INVALID TYPE: upside_pct is {type(upside).__name__}, not number")
                rr['upside_pct'] = None
            elif upside > self.guardrails['max_upside_pct']:
                corrections.append(f"Upside {upside}% capped to {self.guardrails['max_upside_pct']}%")
                rr['upside_pct'] = self.guardrails['max_upside_pct']
                rr['upside_capped'] = True
            elif upside < self.guardrails['min_upside_pct']:
                warnings.append(f"Upside {upside}% suspiciously low for bull case")
        else:
            warnings.append("MISSING: upside_pct not provided")
        
        downside = rr.get('downside_pct')
        if downside is not None:
            if not isinstance(downside, (int, float)):
                warnings.append(f"INVALID TYPE: downside_pct is {type(downside).__name__}, not number")
                rr['downside_pct'] = None
            elif downside > self.guardrails['max_downside_pct']:
                corrections.append(f"Downside {downside}% capped to {self.guardrails['max_downside_pct']}%")
                rr['downside_pct'] = self.guardrails['max_downside_pct']
                rr['downside_capped'] = True
            elif downside < 0:
                corrections.append(f"Downside {downside}% cannot be negative - setting to 0")
                rr['downside_pct'] = 0
        else:
            warnings.append("MISSING: downside_pct not provided")
        
        # ═══════════════════════════════════════════════════════════════
        # 3. VALIDATE R/R RATIO MATH
        # ═══════════════════════════════════════════════════════════════
        rr_ratio = rr.get('reward_risk_ratio')
        upside_val = rr.get('upside_pct')
        downside_val = rr.get('downside_pct')
        
        if upside_val and downside_val and downside_val > 0:
            calculated_rr = round(upside_val / downside_val, 2)
            if rr_ratio is not None:
                # Check if LLM's ratio matches the math
                if abs(calculated_rr - rr_ratio) > 0.5:
                    corrections.append(f"R/R ratio {rr_ratio} doesn't match math ({upside_val}/{downside_val}={calculated_rr}) - fixing")
                    rr['reward_risk_ratio'] = calculated_rr
                    rr['rr_ratio_corrected'] = True
            else:
                # LLM didn't provide, calculate it
                rr['reward_risk_ratio'] = calculated_rr
                rr['rr_ratio_calculated'] = True
        
        # Cap R/R ratio
        if rr.get('reward_risk_ratio'):
            if rr['reward_risk_ratio'] > self.guardrails['max_rr_ratio']:
                corrections.append(f"R/R {rr['reward_risk_ratio']} capped to {self.guardrails['max_rr_ratio']}")
                rr['reward_risk_ratio'] = self.guardrails['max_rr_ratio']
        
        # ═══════════════════════════════════════════════════════════════
        # 4. VALIDATE PRICE CONSISTENCY
        # ═══════════════════════════════════════════════════════════════
        current = rr.get('current_price')
        target = rr.get('upside_target')
        support = rr.get('downside_support')
        
        if current and target:
            if target <= current:
                warnings.append(f"INCONSISTENT: Upside target ${target} <= current ${current}")
        
        if current and support:
            if support >= current:
                warnings.append(f"INCONSISTENT: Downside support ${support} >= current ${current}")
        
        # ═══════════════════════════════════════════════════════════════
        # 5. VALIDATE ENUMS
        # ═══════════════════════════════════════════════════════════════
        if conviction.get('level') not in self.guardrails['valid_convictions']:
            corrections.append(f"Invalid conviction '{conviction.get('level')}' → MEDIUM")
            conviction['level'] = 'MEDIUM'
        
        if conviction.get('data_quality') not in self.guardrails['valid_data_quality']:
            corrections.append(f"Invalid data_quality '{conviction.get('data_quality')}' → moderate")
            conviction['data_quality'] = 'moderate'
        
        if rec.get('action') not in self.guardrails['valid_recommendations']:
            corrections.append(f"Invalid recommendation '{rec.get('action')}' → HOLD")
            rec['action'] = 'HOLD'
        
        # ═══════════════════════════════════════════════════════════════
        # 6. VALIDATE CONSISTENCY (conviction vs data quality)
        # ═══════════════════════════════════════════════════════════════
        if conviction.get('level') == 'HIGH' and conviction.get('data_quality') == 'weak':
            warnings.append("INCONSISTENT: HIGH conviction with weak data quality - suspicious")
            # Downgrade conviction
            corrections.append("HIGH conviction + weak data → downgraded to MEDIUM")
            conviction['level'] = 'MEDIUM'
            conviction['downgraded_reason'] = 'weak data quality'
        
        # ═══════════════════════════════════════════════════════════════
        # 7. VALIDATE RECOMMENDATION vs R/R
        # ═══════════════════════════════════════════════════════════════
        if rec.get('action') == 'STRONG BUY':
            rr_val = rr.get('reward_risk_ratio', 0)
            if rr_val < self.guardrails['strong_buy_requires_min_rr']:
                warnings.append(f"INCONSISTENT: STRONG BUY but R/R only {rr_val}:1 (min {self.guardrails['strong_buy_requires_min_rr']})")
                corrections.append(f"STRONG BUY → BUY (R/R {rr_val} < {self.guardrails['strong_buy_requires_min_rr']})")
                rec['action'] = 'BUY'
                rec['downgraded_reason'] = 'insufficient R/R ratio'
        
        # ═══════════════════════════════════════════════════════════════
        # 8. VALIDATE SIGNALS HAVE SOURCES
        # ═══════════════════════════════════════════════════════════════
        valid_sources = ['technical', 'fundamental', 'news', 'macro', 'sentiment']
        signals = analysis.get('key_bullish_signals', [])
        
        for i, signal in enumerate(signals):
            if not signal.get('source'):
                warnings.append(f"Signal {i+1} missing source - could be fabricated")
            elif signal.get('source').lower() not in valid_sources:
                warnings.append(f"Signal {i+1} has invalid source '{signal.get('source')}'")
            
            if not signal.get('signal'):
                warnings.append(f"Signal {i+1} has no actual signal text")
            
            if signal.get('strength') not in self.guardrails['valid_signal_strengths']:
                signal['strength'] = 'moderate'  # Default
        
        # ═══════════════════════════════════════════════════════════════
        # 9. CHECK FOR RATIONALE (anti-fabrication)
        # ═══════════════════════════════════════════════════════════════
        rationale = rr.get('rationale', '')
        if not rationale or len(rationale) < 20:
            warnings.append("WEAK RATIONALE: No explanation for risk/reward numbers - possibly fabricated")
            rr['rationale_warning'] = True
        
        # ═══════════════════════════════════════════════════════════════
        # 10. STORE RESULTS
        # ═══════════════════════════════════════════════════════════════
        analysis['guardrail_corrections'] = corrections
        analysis['guardrail_warnings'] = warnings
        analysis['validation_passed'] = len(warnings) == 0
        analysis['validation_score'] = max(0, 100 - len(warnings) * 10 - len(corrections) * 5)
        
        # Print summary
        print(f"[BULL] Validation score: {analysis['validation_score']}/100")
        if corrections:
            print(f"[BULL] ⚠ Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        if warnings:
            print(f"[BULL] ⚠ {len(warnings)} warning(s):")
            for w in warnings[:5]:  # Show first 5
                print(f"    ⚠ {w}")
            if len(warnings) > 5:
                print(f"    ... and {len(warnings) - 5} more")
        
        if not corrections and not warnings:
            print("[BULL] ✓ All validations passed")
        
        return analysis

    def _fallback_analysis(self, discussion_points: Dict) -> Dict:
        """Minimal fallback if LLM fails completely"""
        print("[BULL] ⚠ Using fallback analysis")
        
        return {
            'core_thesis': f"Unable to generate full analysis for {self.ticker}. LLM analysis failed.",
            'key_bullish_signals': [],
            'catalysts': [],
            'risk_reward': {
                'upside_pct': None,
                'downside_pct': None,
                'reward_risk_ratio': None,
                'rationale': 'Analysis failed - no data available'
            },
            'conviction': {
                'level': 'LOW',
                'reasoning': 'Fallback due to analysis failure',
                'data_quality': 'weak'
            },
            'recommendation': {
                'action': 'HOLD',
                'position_size': '0%',
                'entry_strategy': 'Wait for proper analysis',
                'time_horizon': 'N/A'
            },
            'counter_bear_arguments': [],
            'full_analysis': 'Analysis could not be completed. Please retry or check input data.',
            'is_fallback': True
        }

    def analyze(self, discussion_points: Dict) -> Dict:
        """
        Main analysis method - LLM does all the heavy lifting.
        """
        if not self.client:
            print("[BULL] ✗ No API client available")
            return self._fallback_analysis(discussion_points)
        
        prompt = self._build_analysis_prompt(discussion_points)
        
        try:
            print(f"[BULL] LLM analyzing reports...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior equity analyst. Analyze the provided reports and return structured JSON. Be specific, cite evidence, and be realistic in your estimates."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Slightly lower for more consistent structured output
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean markdown if present
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            analysis = json.loads(response_text)
            print(f"[BULL] ✓ LLM analysis complete")
            
            # Apply guardrails with discussion points for cross-reference
            analysis = self._validate_and_fix(analysis, discussion_points)
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"[BULL] ✗ JSON parse error: {e}")
            print(f"[BULL] Raw response: {response_text[:500]}...")
            return self._fallback_analysis(discussion_points)
        except Exception as e:
            print(f"[BULL] ✗ Analysis error: {e}")
            return self._fallback_analysis(discussion_points)

    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable report from analysis"""
        rr = analysis.get('risk_reward', {})
        conviction = analysis.get('conviction', {})
        rec = analysis.get('recommendation', {})
        
        # Format bullish signals
        signals_text = ""
        for signal in analysis.get('key_bullish_signals', [])[:5]:
            signals_text += f"  • [{signal.get('source', 'N/A').upper()}] {signal.get('signal', 'N/A')} (Strength: {signal.get('strength', 'N/A')})\n"
        
        # Format catalysts
        catalysts_text = ""
        for cat in analysis.get('catalysts', [])[:3]:
            catalysts_text += f"  • {cat.get('catalyst', 'N/A')} ({cat.get('timeline', 'N/A')}, Impact: {cat.get('impact', 'N/A')})\n"
        
        # Format counter-arguments
        counter_text = ""
        for counter in analysis.get('counter_bear_arguments', [])[:2]:
            counter_text += f"  • Bear: {counter.get('bear_concern', 'N/A')}\n    Bull Response: {counter.get('bull_response', 'N/A')}\n"
        
        guardrails_text = ""
        if analysis.get('guardrail_corrections'):
            guardrails_text = "\n**Guardrail Corrections Applied:**\n" + "\n".join(f"  ⚠ {c}" for c in analysis['guardrail_corrections'])
        
        report = f"""
# BULL THESIS: {self.ticker}
{'='*70}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Analysis Date:** {self.analysis_date or 'Current'}
**Model:** {self.model}

## CORE THESIS
{analysis.get('core_thesis', 'N/A')}

## KEY BULLISH SIGNALS
{signals_text or '  No signals identified'}

## CATALYSTS
{catalysts_text or '  No catalysts identified'}

## RISK/REWARD ASSESSMENT
- **Current Price:** {rr.get('current_price') or 'Not specified'}
- **Upside Target:** {rr.get('upside_target') or 'Not specified'}
- **Downside Support:** {rr.get('downside_support') or 'Not specified'}
- **Upside Potential:** {rr.get('upside_pct')}%{' (CAPPED)' if rr.get('upside_capped') else ''}
- **Downside Risk:** {rr.get('downside_pct')}%{' (CAPPED)' if rr.get('downside_capped') else ''}
- **Reward/Risk Ratio:** {rr.get('reward_risk_ratio')}:1
- **Rationale:** {rr.get('rationale', 'N/A')}

## CONVICTION
- **Level:** {conviction.get('level', 'N/A')}
- **Data Quality:** {conviction.get('data_quality', 'N/A')}
- **Reasoning:** {conviction.get('reasoning', 'N/A')}

## RECOMMENDATION
- **Action:** {rec.get('action', 'N/A')}
- **Position Size:** {rec.get('position_size', 'N/A')}
- **Entry Strategy:** {rec.get('entry_strategy', 'N/A')}
- **Time Horizon:** {rec.get('time_horizon', 'N/A')}

## COUNTER BEAR ARGUMENTS
{counter_text or '  None provided'}

## FULL ANALYSIS
{analysis.get('full_analysis', 'N/A')}
{guardrails_text}

{'='*70}
BULL CASE: {rec.get('action', 'N/A')} | Conviction: {conviction.get('level', 'N/A')} | Data Quality: {conviction.get('data_quality', 'N/A')}
{'='*70}
"""
        return report

    def research(self, discussion_points: Dict) -> str:
        """Main entry point - analyze and generate report"""
        print(f"\n{'='*60}")
        print(f"BULL RESEARCHER: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # LLM does all the analysis
        analysis = self.analyze(discussion_points)
        
        # Generate report
        report = self.generate_report(analysis)
        
        # Store thesis data
        self.bull_thesis = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None,
            'model': self.model,
            
            # LLM's analysis
            'core_thesis': analysis.get('core_thesis'),
            'key_bullish_signals': analysis.get('key_bullish_signals', []),
            'catalysts': analysis.get('catalysts', []),
            'risk_reward': analysis.get('risk_reward', {}),
            'conviction': analysis.get('conviction', {}),
            'recommendation': analysis.get('recommendation', {}),
            'counter_bear_arguments': analysis.get('counter_bear_arguments', []),
            'full_analysis': analysis.get('full_analysis'),
            
            # Metadata
            'guardrail_corrections': analysis.get('guardrail_corrections', []),
            'is_fallback': analysis.get('is_fallback', False)
        }
        
        elapsed = time.time() - start_time
        print(f"\n[BULL] ✓ Research complete in {elapsed:.2f}s")
        print(f"[BULL] Conviction: {analysis.get('conviction', {}).get('level', 'N/A')}")
        print(f"[BULL] Recommendation: {analysis.get('recommendation', {}).get('action', 'N/A')}")
        
        return report
    
    def save_thesis(self, filepath: str):
        """Save thesis data to JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.bull_thesis, f, indent=2, ensure_ascii=False)
            print(f"[BULL] ✓ Thesis saved to {filepath}")
        except Exception as e:
            print(f"[BULL] ✗ Save error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bull Researcher - LLM-Driven Analysis")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--discussion-file", help="Path to discussion_points.json")
    parser.add_argument("--save-data", help="Path to save thesis JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--analysis-date", help="Historical analysis date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    researcher = BullResearcher(
        ticker=args.ticker,
        model=args.model,
        analysis_date=args.analysis_date
    )
    
    if args.discussion_file:
        discussion_points = researcher.load_discussion_points(args.discussion_file)
        if discussion_points:
            report = researcher.research(discussion_points)
            print(report)
            
            if args.save_data:
                researcher.save_thesis(args.save_data)
    else:
        print("[BULL] No discussion file provided. Use --discussion-file")


if __name__ == "__main__":
    main()