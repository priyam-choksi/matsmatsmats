"""
Bear Researcher - LLM-Driven Analysis
LLM does the heavy lifting: reads reports, extracts insights, decides metrics
Code only provides guardrails to catch unrealistic outputs

Philosophy:
  LLM = Brain (analyzes, decides, reasons)
  Code = Safety net (validates bounds, catches errors)

Usage: 
  python bear_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json
  python bear_researcher.py AAPL --discussion-file ../../outputs/discussion_points.json --analysis-date 2024-06-15
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


class BearResearcher:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[BEAR] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[BEAR] Running in LIVE mode (current data)")
        
        # Guardrails - comprehensive validation to catch fabrication and errors
        self.guardrails = {
            # Percentage bounds
            'max_downside_pct': 80,     # Flag if LLM says >80% downside
            'min_downside_pct': 1,      # Flag if <1%
            'max_upside_pct': 100,      # Flag if >100% (limited) upside
            'min_upside_pct': 0,        # Can be 0 for bear case
            'max_risk_score': 100,      # Risk score 0-100
            'min_risk_score': 0,
            
            # Valid enums
            'valid_risk_levels': ['HIGH', 'MEDIUM', 'LOW'],
            'valid_convictions': ['HIGH', 'MEDIUM', 'LOW'],
            'valid_recommendations': ['STRONG SELL', 'SELL', 'HOLD', 'AVOID', 'REDUCE'],
            'valid_signal_severity': ['high', 'medium', 'low'],
            'valid_probability': ['high', 'medium', 'low'],
            'valid_data_quality': ['strong', 'moderate', 'weak'],
            
            # Consistency rules
            'high_conviction_requires_strong_data': True,
            'strong_sell_requires_min_risk_score': 60,
            'max_signals_without_source': 0,
            
            # Required fields
            'required_fields': ['core_thesis', 'risk_assessment', 'conviction', 'recommendation'],
            'required_ra_fields': ['downside_pct', 'risk_score', 'rationale'],
            'required_conviction_fields': ['level', 'data_quality'],
        }
        
        self.bear_thesis = {}
        self.validation_warnings = []
    
    def load_discussion_points(self, filepath: str) -> Optional[Dict]:
        """Load discussion points from analysts"""
        print(f"[BEAR] Loading discussion points from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[BEAR] ✓ Loaded discussion for {data.get('ticker', 'unknown')}")
            
            if data.get('analysis_date'):
                print(f"[BEAR] Discussion data is from historical date: {data.get('analysis_date')}")
            
            return data
        except FileNotFoundError:
            print(f"[BEAR] ✗ File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"[BEAR] ✗ Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"[BEAR] ✗ Load error: {e}")
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

        prompt = f"""You are a Senior Bear Analyst building the investment case AGAINST {self.ticker}.

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

Analyze ALL the reports above and build a comprehensive BEAR CASE.

You must:
1. READ the reports carefully and identify bearish signals, risks, and red flags
2. EXTRACT specific data points that support the bear case (overbought indicators, valuation concerns, debt levels, etc.)
3. DETERMINE realistic downside potential based on what the data shows
4. ASSESS the quality of evidence (is there strong data or just vague concerns?)
5. DECIDE your conviction level based on evidence strength

CRITICAL RULES:
- Base your analysis ONLY on what's in the reports above
- If a report doesn't mention specific support levels, estimate based on technical analysis or historical ranges
- Be realistic - even bear cases need to be grounded in data
- If data is weak/missing, say so and lower your conviction
- Cite specific evidence from the reports

Return your analysis as JSON with this structure:
{{
    "core_thesis": "2-3 sentence summary of why this stock has downside risk",
    
    "key_risk_signals": [
        {{"source": "technical/fundamental/news/macro", "signal": "specific risk finding", "severity": "high/medium/low"}}
    ],
    
    "downside_triggers": [
        {{"trigger": "what could cause decline", "timeline": "when", "probability": "high/medium/low", "impact": "description"}}
    ],
    
    "risk_assessment": {{
        "current_price": <number or null if not found>,
        "downside_target": <number or null>,
        "upside_resistance": <number or null>,
        "downside_pct": <your estimate based on data>,
        "limited_upside_pct": <your estimate - how much upside is capped>,
        "risk_score": <0-100, your assessment of overall risk>,
        "rationale": "explain how you arrived at these numbers"
    }},
    
    "conviction": {{
        "level": "HIGH/MEDIUM/LOW",
        "reasoning": "why this conviction level",
        "data_quality": "strong/moderate/weak - how much hard data vs speculation"
    }},
    
    "recommendation": {{
        "action": "STRONG SELL/SELL/AVOID/REDUCE/HOLD",
        "risk_level": "HIGH/MEDIUM/LOW",
        "hedging_strategy": "suggested protection",
        "time_horizon": "when risks most likely to materialize"
    }},
    
    "counter_bull_arguments": [
        {{"bull_argument": "what bulls might say", "bear_response": "why it's wrong or risky"}}
    ],
    
    "full_analysis": "Your complete written bear case (3-5 paragraphs)"
}}

Return ONLY valid JSON. No markdown, no explanation outside JSON."""

        return prompt

    def _validate_and_fix(self, analysis: Dict, discussion_points: Dict) -> Dict:
        """
        Comprehensive guardrails to catch fabrication, errors, and inconsistencies.
        Fixes what can be fixed, flags what can't.
        """
        print("[BEAR] Applying comprehensive guardrails...")
        corrections = []
        warnings = []
        
        # ═══════════════════════════════════════════════════════════════
        # 1. CHECK REQUIRED FIELDS EXIST
        # ═══════════════════════════════════════════════════════════════
        for field in self.guardrails['required_fields']:
            if field not in analysis or not analysis[field]:
                warnings.append(f"MISSING REQUIRED: '{field}' not provided by LLM")
                if field == 'risk_assessment':
                    analysis['risk_assessment'] = {}
                elif field == 'conviction':
                    analysis['conviction'] = {'level': 'LOW', 'data_quality': 'weak'}
                elif field == 'recommendation':
                    analysis['recommendation'] = {'action': 'HOLD', 'risk_level': 'MEDIUM'}
        
        ra = analysis.get('risk_assessment', {})
        conviction = analysis.get('conviction', {})
        rec = analysis.get('recommendation', {})
        
        # ═══════════════════════════════════════════════════════════════
        # 2. VALIDATE PERCENTAGE BOUNDS
        # ═══════════════════════════════════════════════════════════════
        downside = ra.get('downside_pct')
        if downside is not None:
            if not isinstance(downside, (int, float)):
                warnings.append(f"INVALID TYPE: downside_pct is {type(downside).__name__}, not number")
                ra['downside_pct'] = None
            elif downside > self.guardrails['max_downside_pct']:
                corrections.append(f"Downside {downside}% capped to {self.guardrails['max_downside_pct']}%")
                ra['downside_pct'] = self.guardrails['max_downside_pct']
                ra['downside_capped'] = True
            elif downside < 0:
                corrections.append(f"Downside {downside}% cannot be negative - setting to 0")
                ra['downside_pct'] = 0
        else:
            warnings.append("MISSING: downside_pct not provided")
        
        limited_upside = ra.get('limited_upside_pct')
        if limited_upside is not None:
            if not isinstance(limited_upside, (int, float)):
                warnings.append(f"INVALID TYPE: limited_upside_pct is {type(limited_upside).__name__}, not number")
                ra['limited_upside_pct'] = None
            elif limited_upside > self.guardrails['max_upside_pct']:
                corrections.append(f"Limited upside {limited_upside}% capped to {self.guardrails['max_upside_pct']}%")
                ra['limited_upside_pct'] = self.guardrails['max_upside_pct']
            elif limited_upside < 0:
                corrections.append(f"Limited upside {limited_upside}% cannot be negative - setting to 0")
                ra['limited_upside_pct'] = 0
        
        # ═══════════════════════════════════════════════════════════════
        # 3. VALIDATE RISK SCORE
        # ═══════════════════════════════════════════════════════════════
        risk_score = ra.get('risk_score')
        if risk_score is not None:
            if not isinstance(risk_score, (int, float)):
                warnings.append(f"INVALID TYPE: risk_score is {type(risk_score).__name__}, not number")
                ra['risk_score'] = 50  # Default to medium
            elif risk_score > self.guardrails['max_risk_score']:
                corrections.append(f"Risk score {risk_score} capped to 100")
                ra['risk_score'] = 100
            elif risk_score < self.guardrails['min_risk_score']:
                corrections.append(f"Risk score {risk_score} floored to 0")
                ra['risk_score'] = 0
        else:
            warnings.append("MISSING: risk_score not provided")
        
        # ═══════════════════════════════════════════════════════════════
        # 4. VALIDATE PRICE CONSISTENCY
        # ═══════════════════════════════════════════════════════════════
        current = ra.get('current_price')
        downside_target = ra.get('downside_target')
        resistance = ra.get('upside_resistance')
        
        if current and downside_target:
            if downside_target >= current:
                warnings.append(f"INCONSISTENT: Downside target ${downside_target} >= current ${current}")
        
        if current and resistance:
            if resistance <= current:
                warnings.append(f"INCONSISTENT: Resistance ${resistance} <= current ${current}")
        
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
        
        if rec.get('risk_level') not in self.guardrails['valid_risk_levels']:
            corrections.append(f"Invalid risk_level '{rec.get('risk_level')}' → MEDIUM")
            rec['risk_level'] = 'MEDIUM'
        
        # ═══════════════════════════════════════════════════════════════
        # 6. VALIDATE CONSISTENCY (conviction vs data quality)
        # ═══════════════════════════════════════════════════════════════
        if conviction.get('level') == 'HIGH' and conviction.get('data_quality') == 'weak':
            warnings.append("INCONSISTENT: HIGH conviction with weak data quality - suspicious")
            corrections.append("HIGH conviction + weak data → downgraded to MEDIUM")
            conviction['level'] = 'MEDIUM'
            conviction['downgraded_reason'] = 'weak data quality'
        
        # ═══════════════════════════════════════════════════════════════
        # 7. VALIDATE RECOMMENDATION vs RISK SCORE
        # ═══════════════════════════════════════════════════════════════
        if rec.get('action') == 'STRONG SELL':
            score = ra.get('risk_score', 0)
            if score < self.guardrails['strong_sell_requires_min_risk_score']:
                warnings.append(f"INCONSISTENT: STRONG SELL but risk score only {score} (min {self.guardrails['strong_sell_requires_min_risk_score']})")
                corrections.append(f"STRONG SELL → SELL (risk score {score} < {self.guardrails['strong_sell_requires_min_risk_score']})")
                rec['action'] = 'SELL'
                rec['downgraded_reason'] = 'insufficient risk score'
        
        # ═══════════════════════════════════════════════════════════════
        # 8. VALIDATE RISK LEVEL vs RISK SCORE CONSISTENCY
        # ═══════════════════════════════════════════════════════════════
        score = ra.get('risk_score')
        level = rec.get('risk_level')
        if score is not None and level:
            if score >= 70 and level == 'LOW':
                warnings.append(f"INCONSISTENT: Risk score {score} but risk level is LOW")
                corrections.append(f"Risk level LOW → HIGH (score {score})")
                rec['risk_level'] = 'HIGH'
            elif score <= 30 and level == 'HIGH':
                warnings.append(f"INCONSISTENT: Risk score {score} but risk level is HIGH")
                corrections.append(f"Risk level HIGH → LOW (score {score})")
                rec['risk_level'] = 'LOW'
        
        # ═══════════════════════════════════════════════════════════════
        # 9. VALIDATE SIGNALS HAVE SOURCES
        # ═══════════════════════════════════════════════════════════════
        valid_sources = ['technical', 'fundamental', 'news', 'macro', 'sentiment']
        signals = analysis.get('key_risk_signals', [])
        
        for i, signal in enumerate(signals):
            if not signal.get('source'):
                warnings.append(f"Risk signal {i+1} missing source - could be fabricated")
            elif signal.get('source').lower() not in valid_sources:
                warnings.append(f"Risk signal {i+1} has invalid source '{signal.get('source')}'")
            
            if not signal.get('signal'):
                warnings.append(f"Risk signal {i+1} has no actual signal text")
            
            if signal.get('severity') not in self.guardrails['valid_signal_severity']:
                signal['severity'] = 'medium'
        
        # Validate triggers
        triggers = analysis.get('downside_triggers', [])
        for i, trigger in enumerate(triggers):
            if not trigger.get('trigger'):
                warnings.append(f"Trigger {i+1} has no description")
            if trigger.get('probability') not in self.guardrails['valid_probability']:
                trigger['probability'] = 'medium'
        
        # ═══════════════════════════════════════════════════════════════
        # 10. CHECK FOR RATIONALE (anti-fabrication)
        # ═══════════════════════════════════════════════════════════════
        rationale = ra.get('rationale', '')
        if not rationale or len(rationale) < 20:
            warnings.append("WEAK RATIONALE: No explanation for risk numbers - possibly fabricated")
            ra['rationale_warning'] = True
        
        # ═══════════════════════════════════════════════════════════════
        # 11. STORE RESULTS
        # ═══════════════════════════════════════════════════════════════
        analysis['guardrail_corrections'] = corrections
        analysis['guardrail_warnings'] = warnings
        analysis['validation_passed'] = len(warnings) == 0
        analysis['validation_score'] = max(0, 100 - len(warnings) * 10 - len(corrections) * 5)
        
        # Print summary
        print(f"[BEAR] Validation score: {analysis['validation_score']}/100")
        if corrections:
            print(f"[BEAR] ⚠ Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        if warnings:
            print(f"[BEAR] ⚠ {len(warnings)} warning(s):")
            for w in warnings[:5]:
                print(f"    ⚠ {w}")
            if len(warnings) > 5:
                print(f"    ... and {len(warnings) - 5} more")
        
        if not corrections and not warnings:
            print("[BEAR] ✓ All validations passed")
        
        return analysis

    def _fallback_analysis(self, discussion_points: Dict) -> Dict:
        """Minimal fallback if LLM fails completely"""
        print("[BEAR] ⚠ Using fallback analysis")
        
        return {
            'core_thesis': f"Unable to generate full analysis for {self.ticker}. LLM analysis failed.",
            'key_risk_signals': [],
            'downside_triggers': [],
            'risk_assessment': {
                'downside_pct': None,
                'limited_upside_pct': None,
                'risk_score': None,
                'rationale': 'Analysis failed - no data available'
            },
            'conviction': {
                'level': 'LOW',
                'reasoning': 'Fallback due to analysis failure',
                'data_quality': 'weak'
            },
            'recommendation': {
                'action': 'HOLD',
                'risk_level': 'MEDIUM',
                'hedging_strategy': 'Wait for proper analysis',
                'time_horizon': 'N/A'
            },
            'counter_bull_arguments': [],
            'full_analysis': 'Analysis could not be completed. Please retry or check input data.',
            'is_fallback': True
        }

    def analyze(self, discussion_points: Dict) -> Dict:
        """
        Main analysis method - LLM does all the heavy lifting.
        """
        if not self.client:
            print("[BEAR] ✗ No API client available")
            return self._fallback_analysis(discussion_points)
        
        prompt = self._build_analysis_prompt(discussion_points)
        
        try:
            print(f"[BEAR] LLM analyzing reports...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior risk analyst. Analyze the provided reports and return structured JSON. Be specific about risks, cite evidence, and be realistic in your estimates."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean markdown if present
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            analysis = json.loads(response_text)
            print(f"[BEAR] ✓ LLM analysis complete")
            
            # Apply guardrails with discussion points for cross-reference
            analysis = self._validate_and_fix(analysis, discussion_points)
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"[BEAR] ✗ JSON parse error: {e}")
            print(f"[BEAR] Raw response: {response_text[:500]}...")
            return self._fallback_analysis(discussion_points)
        except Exception as e:
            print(f"[BEAR] ✗ Analysis error: {e}")
            return self._fallback_analysis(discussion_points)

    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable report from analysis"""
        ra = analysis.get('risk_assessment', {})
        conviction = analysis.get('conviction', {})
        rec = analysis.get('recommendation', {})
        
        # Format risk signals
        signals_text = ""
        for signal in analysis.get('key_risk_signals', [])[:5]:
            signals_text += f"  • [{signal.get('source', 'N/A').upper()}] {signal.get('signal', 'N/A')} (Severity: {signal.get('severity', 'N/A')})\n"
        
        # Format triggers
        triggers_text = ""
        for trig in analysis.get('downside_triggers', [])[:3]:
            triggers_text += f"  • {trig.get('trigger', 'N/A')} ({trig.get('timeline', 'N/A')}, Prob: {trig.get('probability', 'N/A')})\n"
        
        # Format counter-arguments
        counter_text = ""
        for counter in analysis.get('counter_bull_arguments', [])[:2]:
            counter_text += f"  • Bull: {counter.get('bull_argument', 'N/A')}\n    Bear Response: {counter.get('bear_response', 'N/A')}\n"
        
        guardrails_text = ""
        if analysis.get('guardrail_corrections'):
            guardrails_text = "\n**Guardrail Corrections Applied:**\n" + "\n".join(f"  ⚠ {c}" for c in analysis['guardrail_corrections'])
        
        report = f"""
# BEAR THESIS: {self.ticker}
{'='*70}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Analysis Date:** {self.analysis_date or 'Current'}
**Model:** {self.model}

## CORE THESIS
{analysis.get('core_thesis', 'N/A')}

## KEY RISK SIGNALS
{signals_text or '  No risk signals identified'}

## DOWNSIDE TRIGGERS
{triggers_text or '  No triggers identified'}

## RISK ASSESSMENT
- **Current Price:** {ra.get('current_price') or 'Not specified'}
- **Downside Target:** {ra.get('downside_target') or 'Not specified'}
- **Upside Resistance:** {ra.get('upside_resistance') or 'Not specified'}
- **Downside Potential:** {ra.get('downside_pct')}%{' (CAPPED)' if ra.get('downside_capped') else ''}
- **Limited Upside:** {ra.get('limited_upside_pct')}%
- **Risk Score:** {ra.get('risk_score')}/100
- **Rationale:** {ra.get('rationale', 'N/A')}

## CONVICTION
- **Level:** {conviction.get('level', 'N/A')}
- **Data Quality:** {conviction.get('data_quality', 'N/A')}
- **Reasoning:** {conviction.get('reasoning', 'N/A')}

## RECOMMENDATION
- **Action:** {rec.get('action', 'N/A')}
- **Risk Level:** {rec.get('risk_level', 'N/A')}
- **Hedging Strategy:** {rec.get('hedging_strategy', 'N/A')}
- **Time Horizon:** {rec.get('time_horizon', 'N/A')}

## COUNTER BULL ARGUMENTS
{counter_text or '  None provided'}

## FULL ANALYSIS
{analysis.get('full_analysis', 'N/A')}
{guardrails_text}

{'='*70}
BEAR CASE: {rec.get('action', 'N/A')} | Risk Level: {rec.get('risk_level', 'N/A')} | Conviction: {conviction.get('level', 'N/A')}
{'='*70}
"""
        return report

    def research(self, discussion_points: Dict) -> str:
        """Main entry point - analyze and generate report"""
        print(f"\n{'='*60}")
        print(f"BEAR RESEARCHER: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # LLM does all the analysis
        analysis = self.analyze(discussion_points)
        
        # Generate report
        report = self.generate_report(analysis)
        
        # Store thesis data
        self.bear_thesis = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None,
            'model': self.model,
            
            # LLM's analysis
            'core_thesis': analysis.get('core_thesis'),
            'key_risk_signals': analysis.get('key_risk_signals', []),
            'downside_triggers': analysis.get('downside_triggers', []),
            'risk_assessment': analysis.get('risk_assessment', {}),
            'conviction': analysis.get('conviction', {}),
            'recommendation': analysis.get('recommendation', {}),
            'counter_bull_arguments': analysis.get('counter_bull_arguments', []),
            'full_analysis': analysis.get('full_analysis'),
            
            # Metadata
            'guardrail_corrections': analysis.get('guardrail_corrections', []),
            'is_fallback': analysis.get('is_fallback', False)
        }
        
        elapsed = time.time() - start_time
        print(f"\n[BEAR] ✓ Research complete in {elapsed:.2f}s")
        print(f"[BEAR] Risk Level: {analysis.get('recommendation', {}).get('risk_level', 'N/A')}")
        print(f"[BEAR] Recommendation: {analysis.get('recommendation', {}).get('action', 'N/A')}")
        
        return report
    
    def save_thesis(self, filepath: str):
        """Save thesis data to JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.bear_thesis, f, indent=2, ensure_ascii=False)
            print(f"[BEAR] ✓ Thesis saved to {filepath}")
        except Exception as e:
            print(f"[BEAR] ✗ Save error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bear Researcher - LLM-Driven Analysis")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--discussion-file", help="Path to discussion_points.json")
    parser.add_argument("--save-data", help="Path to save thesis JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--analysis-date", help="Historical analysis date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    researcher = BearResearcher(
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
        print("[BEAR] No discussion file provided. Use --discussion-file")


if __name__ == "__main__":
    main()