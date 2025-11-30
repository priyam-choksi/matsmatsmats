"""
Neutral Risk Debator - Math by Code, Judgment by LLM

Philosophy:
  Code = Calculator (EV, R/R ratio, position sizing formulas)
  LLM = Judge (interprets what the numbers mean, makes final call)

This ensures:
  - No math errors (code is precise)
  - LLM focuses on nuanced judgment
  - Results are reproducible and auditable

Usage: python neutral_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
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


class NeutralDebator:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.analysis_date = analysis_date
        self.risk_profile = "NEUTRAL"
        
        if self.analysis_date:
            print(f"[NEUTRAL] *** HISTORICAL MODE: As of {self.analysis_date} ***")
        
        # Guardrails
        self.guardrails = {
            'max_position_pct': 12,
            'min_position_pct': 0,
            'max_stop_loss_pct': 15,
            'min_stop_loss_pct': 5,
            'max_single_target_pct': 50,
            'min_target_pct': 5,
            'min_ev_for_buy': 2.0,      # Need 2%+ EV for BUY
            'min_ev_for_small_buy': 0,   # Any positive EV for SMALL BUY
            'min_rr_for_buy': 1.5,       # Need 1.5:1 R/R for BUY
            'valid_stances': ['BUY', 'SMALL BUY', 'HOLD', 'AVOID'],
            'valid_confidence': ['HIGH', 'MEDIUM', 'LOW'],
        }
        
        self.evaluation = {}
        self.calculated_metrics = {}

    def load_synthesis(self, synthesis_file: str) -> Optional[Dict]:
        """Load research synthesis"""
        if not synthesis_file or not os.path.exists(synthesis_file):
            print(f"[NEUTRAL] ✗ Synthesis file not found")
            return None
        
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[NEUTRAL] ✓ Synthesis loaded")
            
            if data.get('analysis_date') and not self.analysis_date:
                self.analysis_date = data.get('analysis_date')
            
            return data
        except Exception as e:
            print(f"[NEUTRAL] ✗ Load error: {e}")
            return None

    def load_thesis_files(self, bull_file: Optional[str], bear_file: Optional[str]) -> tuple:
        """Load bull/bear thesis files"""
        bull_data, bear_data = None, None
        
        if bull_file and os.path.exists(bull_file):
            try:
                with open(bull_file, 'r', encoding='utf-8') as f:
                    bull_data = json.load(f)
                print(f"[NEUTRAL] ✓ Bull thesis loaded")
            except Exception as e:
                print(f"[NEUTRAL] ⚠ Bull file error: {e}")
        
        if bear_file and os.path.exists(bear_file):
            try:
                with open(bear_file, 'r', encoding='utf-8') as f:
                    bear_data = json.load(f)
                print(f"[NEUTRAL] ✓ Bear thesis loaded")
            except Exception as e:
                print(f"[NEUTRAL] ⚠ Bear file error: {e}")
        
        return bull_data, bear_data

    def _extract_number(self, value: Any, default: float = 0) -> float:
        """Safely extract a number from various formats"""
        if value is None or value == 'N/A':
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            nums = re.findall(r'[-+]?\d*\.?\d+', value)
            return float(nums[0]) if nums else default
        return default

    def calculate_metrics(self, synthesis: Dict, bull_data: Optional[Dict], 
                         bear_data: Optional[Dict]) -> Dict:
        """
        CORE FUNCTION: Calculate all metrics with CODE (no LLM math)
        
        This ensures accuracy - LLM will only interpret these numbers.
        """
        print(f"[NEUTRAL] Calculating metrics with code...")
        
        bull = bull_data or synthesis.get('bull_thesis', {})
        bear = bear_data or synthesis.get('bear_thesis', {})
        probs = synthesis.get('probabilities', {})
        
        # Extract probabilities (ensure they're decimals for calculation)
        bull_prob_raw = self._extract_number(probs.get('bull_case', probs.get('bull_prob', 40)), 40)
        bear_prob_raw = self._extract_number(probs.get('bear_case', probs.get('bear_prob', 30)), 30)
        base_prob_raw = self._extract_number(probs.get('base_case', probs.get('base_prob', 30)), 30)
        
        # Normalize to sum to 100 if needed
        total_prob = bull_prob_raw + bear_prob_raw + base_prob_raw
        if total_prob > 0:
            bull_prob = bull_prob_raw / total_prob
            bear_prob = bear_prob_raw / total_prob
            base_prob = base_prob_raw / total_prob
        else:
            bull_prob, bear_prob, base_prob = 0.4, 0.3, 0.3
        
        # Extract upside/downside percentages
        bull_rr = bull.get('risk_reward', {})
        bear_ra = bear.get('risk_assessment', {})
        
        upside_pct = self._extract_number(bull_rr.get('upside_pct'), 10)
        downside_pct = self._extract_number(
            bear_ra.get('downside_pct', bull_rr.get('downside_pct')), 
            10
        )
        
        # Calculate Expected Value: (Upside × Bull Prob) - (Downside × Bear Prob)
        # Base case assumed to be ~0% return
        expected_value = (upside_pct * bull_prob) - (downside_pct * bear_prob)
        
        # Calculate Risk/Reward Ratio
        risk_reward_ratio = upside_pct / downside_pct if downside_pct > 0 else 0
        
        # Calculate suggested position size based on EV and Kelly-inspired formula
        # Simplified: position = EV / downside (capped)
        if expected_value > 0 and downside_pct > 0:
            kelly_fraction = expected_value / downside_pct
            suggested_position = min(kelly_fraction * 100, self.guardrails['max_position_pct'])
            suggested_position = max(suggested_position, 0)
        else:
            suggested_position = 0
        
        # Determine stance based on pure math
        if expected_value >= self.guardrails['min_ev_for_buy'] and risk_reward_ratio >= self.guardrails['min_rr_for_buy']:
            math_stance = 'BUY'
        elif expected_value > self.guardrails['min_ev_for_small_buy']:
            math_stance = 'SMALL BUY'
        elif expected_value < -2:
            math_stance = 'AVOID'
        else:
            math_stance = 'HOLD'
        
        metrics = {
            # Probabilities (as percentages for display)
            'bull_prob_pct': round(bull_prob * 100, 1),
            'bear_prob_pct': round(bear_prob * 100, 1),
            'base_prob_pct': round(base_prob * 100, 1),
            
            # Price targets
            'upside_pct': round(upside_pct, 2),
            'downside_pct': round(downside_pct, 2),
            
            # Calculated metrics
            'expected_value': round(expected_value, 2),
            'ev_calculation': f"({upside_pct:.1f}% × {bull_prob*100:.1f}%) - ({downside_pct:.1f}% × {bear_prob*100:.1f}%)",
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            
            # Suggested outputs from math
            'math_suggested_position': round(suggested_position, 1),
            'math_suggested_stance': math_stance,
            
            # Thresholds for LLM context
            'ev_threshold_buy': self.guardrails['min_ev_for_buy'],
            'rr_threshold_buy': self.guardrails['min_rr_for_buy'],
        }
        
        self.calculated_metrics = metrics
        
        print(f"[NEUTRAL] ✓ EV: {metrics['expected_value']:+.2f}%")
        print(f"[NEUTRAL] ✓ R/R: {metrics['risk_reward_ratio']:.2f}:1")
        print(f"[NEUTRAL] ✓ Math suggests: {math_stance} @ {suggested_position:.1f}%")
        
        return metrics

    def _build_evaluation_prompt(self, synthesis: Dict, metrics: Dict,
                                  bull_data: Optional[Dict] = None,
                                  bear_data: Optional[Dict] = None) -> str:
        """Build prompt with PRE-CALCULATED metrics for LLM to judge"""
        
        bull = bull_data or synthesis.get('bull_thesis', {})
        bear = bear_data or synthesis.get('bear_thesis', {})
        conclusion = synthesis.get('conclusion', {})
        
        date_context = ""
        if self.analysis_date:
            date_context = f"⚠️ HISTORICAL MODE: Analyze as of {self.analysis_date}\n"

        prompt = f"""You are a NEUTRAL risk evaluator for {self.ticker}.
Your style: Probability-weighted, expected value focused. No bias toward bull or bear.

{date_context}

═══════════════════════════════════════════════════════════════════════════════
PRE-CALCULATED METRICS (verified by code - use these exact numbers)
═══════════════════════════════════════════════════════════════════════════════

**EXPECTED VALUE: {metrics['expected_value']:+.2f}%**
Calculation: {metrics['ev_calculation']} = {metrics['expected_value']:+.2f}%

**RISK/REWARD RATIO: {metrics['risk_reward_ratio']:.2f}:1**
(Upside {metrics['upside_pct']}% ÷ Downside {metrics['downside_pct']}%)

**PROBABILITIES:**
- Bull Case: {metrics['bull_prob_pct']}%
- Bear Case: {metrics['bear_prob_pct']}%
- Base Case: {metrics['base_prob_pct']}%

**MATH-BASED SUGGESTION:**
- Stance: {metrics['math_suggested_stance']}
- Position: {metrics['math_suggested_position']}%

**THRESHOLDS:**
- Need EV ≥ {metrics['ev_threshold_buy']}% for BUY
- Need R/R ≥ {metrics['rr_threshold_buy']}:1 for BUY

═══════════════════════════════════════════════════════════════════════════════
CONTEXT FOR YOUR JUDGMENT
═══════════════════════════════════════════════════════════════════════════════

RESEARCH MANAGER'S CONCLUSION:
- Recommendation: {conclusion.get('recommendation', 'N/A')}
- Confidence: {conclusion.get('confidence', 'N/A')}
- Rationale: {conclusion.get('rationale', 'N/A')[:300] if conclusion.get('rationale') else 'N/A'}

BULL THESIS SUMMARY:
{bull.get('core_thesis', 'N/A')[:400] if bull.get('core_thesis') else 'N/A'}

BEAR THESIS SUMMARY:
{bear.get('core_thesis', 'N/A')[:400] if bear.get('core_thesis') else 'N/A'}

KEY RISKS:
{json.dumps(synthesis.get('risks_to_monitor', [])[:3], indent=2)}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK - INTERPRET THE NUMBERS
═══════════════════════════════════════════════════════════════════════════════

The math has been done for you. Your job is to INTERPRET what it means:

1. Is the EV of {metrics['expected_value']:+.2f}% enough to justify a position?
2. Is the R/R of {metrics['risk_reward_ratio']:.2f}:1 acceptable?
3. Are the probability estimates realistic? Should you adjust your view?
4. What qualitative factors might the math be missing?

You can AGREE with the math suggestion ({metrics['math_suggested_stance']} @ {metrics['math_suggested_position']}%)
or OVERRIDE it with reasoning.

Return JSON:
{{
    "stance": "BUY/SMALL BUY/HOLD/AVOID",
    "position_pct": <0-12>,
    "confidence": "HIGH/MEDIUM/LOW",
    
    "agrees_with_math": true/false,
    "override_reasoning": "if you disagree with math, explain why",
    
    "probability_assessment": {{
        "bull_prob_realistic": true/false,
        "bear_prob_realistic": true/false,
        "your_adjusted_view": "any adjustments you'd make"
    }},
    
    "ev_interpretation": "is {metrics['expected_value']:+.2f}% EV enough? why/why not?",
    "rr_interpretation": "is {metrics['risk_reward_ratio']:.2f}:1 R/R acceptable? why/why not?",
    
    "qualitative_factors": ["factors math might miss"],
    "key_concerns": ["main concerns"],
    
    "stop_loss_pct": <5-15>,
    "profit_targets": [<target1>, <target2>],
    
    "reasoning": "2-3 sentences on your final judgment",
    
    "vs_research_manager": "how your view differs"
}}

Return ONLY valid JSON."""

        return prompt

    def _validate_evaluation(self, evaluation: Dict, metrics: Dict) -> Dict:
        """Validate and apply guardrails"""
        print("[NEUTRAL] Applying guardrails...")
        corrections = []
        warnings = []
        
        # Inject calculated metrics (code values override any LLM math)
        evaluation['expected_value'] = metrics['expected_value']
        evaluation['ev_calculation'] = metrics['ev_calculation']
        evaluation['risk_reward_ratio'] = metrics['risk_reward_ratio']
        evaluation['calculated_metrics'] = metrics
        
        # Validate stance
        stance = evaluation.get('stance', 'HOLD')
        if stance not in self.guardrails['valid_stances']:
            corrections.append(f"Invalid stance '{stance}' → HOLD")
            evaluation['stance'] = 'HOLD'
        
        # Validate position size
        pos = evaluation.get('position_pct', 0)
        if not isinstance(pos, (int, float)):
            pos = 0
        if pos > self.guardrails['max_position_pct']:
            corrections.append(f"Position {pos}% capped to {self.guardrails['max_position_pct']}%")
            pos = self.guardrails['max_position_pct']
        if pos < 0:
            pos = 0
        evaluation['position_pct'] = pos
        
        # Validate stop loss
        stop = evaluation.get('stop_loss_pct', 10)
        if not isinstance(stop, (int, float)):
            stop = 10
        stop = max(self.guardrails['min_stop_loss_pct'], min(stop, self.guardrails['max_stop_loss_pct']))
        evaluation['stop_loss_pct'] = stop
        
        # Validate profit targets
        targets = evaluation.get('profit_targets', [10, 20])
        if not isinstance(targets, list) or len(targets) < 2:
            targets = [10, 20]
        evaluation['profit_targets'] = sorted([
            max(self.guardrails['min_target_pct'], min(t, self.guardrails['max_single_target_pct']))
            for t in targets[:2] if isinstance(t, (int, float))
        ]) or [10, 20]
        
        # Validate confidence
        conf = evaluation.get('confidence', 'MEDIUM')
        if conf not in self.guardrails['valid_confidence']:
            evaluation['confidence'] = 'MEDIUM'
        
        # CRITICAL: Enforce EV-stance consistency
        ev = metrics['expected_value']
        stance = evaluation.get('stance')
        
        if stance in ['BUY', 'SMALL BUY'] and ev < -1:
            corrections.append(f"{stance} with negative EV ({ev:.1f}%) → AVOID")
            evaluation['stance'] = 'AVOID'
            evaluation['position_pct'] = 0
        
        if stance == 'BUY' and ev < self.guardrails['min_ev_for_buy']:
            corrections.append(f"BUY but EV {ev:.1f}% < {self.guardrails['min_ev_for_buy']}% → SMALL BUY")
            evaluation['stance'] = 'SMALL BUY'
            evaluation['position_pct'] = min(evaluation['position_pct'], 5)
        
        # Enforce stance-position consistency
        stance = evaluation.get('stance')
        pos = evaluation.get('position_pct', 0)
        
        if stance == 'AVOID' and pos > 0:
            corrections.append(f"AVOID but position {pos}% → 0%")
            evaluation['position_pct'] = 0
        
        if stance == 'HOLD' and pos > 5:
            corrections.append(f"HOLD but position {pos}% → max 5%")
            evaluation['position_pct'] = 5
        
        evaluation['guardrail_corrections'] = corrections
        evaluation['guardrail_warnings'] = warnings
        evaluation['validation_passed'] = len(corrections) == 0
        evaluation['validation_score'] = max(0, 100 - len(corrections) * 10 - len(warnings) * 5)
        
        if corrections:
            print(f"[NEUTRAL] Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        
        return evaluation

    def _fallback_evaluation(self, metrics: Dict) -> Dict:
        """Fallback using pure math when LLM fails"""
        return {
            'stance': metrics.get('math_suggested_stance', 'HOLD'),
            'position_pct': metrics.get('math_suggested_position', 0),
            'confidence': 'LOW',
            'expected_value': metrics.get('expected_value', 0),
            'ev_calculation': metrics.get('ev_calculation', 'N/A'),
            'risk_reward_ratio': metrics.get('risk_reward_ratio', 0),
            'stop_loss_pct': 10,
            'profit_targets': [10, 20],
            'reasoning': 'Fallback to math-based decision (LLM unavailable)',
            'agrees_with_math': True,
            'is_fallback': True,
            'calculated_metrics': metrics
        }

    def evaluate(self, synthesis: Dict, bull_data: Optional[Dict] = None,
                 bear_data: Optional[Dict] = None) -> Dict:
        """Main evaluation - Math first, then LLM judges"""
        
        # Step 1: Calculate metrics with CODE
        metrics = self.calculate_metrics(synthesis, bull_data, bear_data)
        
        if not self.client:
            print("[NEUTRAL] ✗ No API client - using math fallback")
            return self._fallback_evaluation(metrics)
        
        # Step 2: LLM interprets the metrics
        prompt = self._build_evaluation_prompt(synthesis, metrics, bull_data, bear_data)
        
        try:
            print(f"[NEUTRAL] LLM interpreting metrics...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a neutral risk evaluator. Interpret pre-calculated metrics. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            evaluation = json.loads(response_text)
            
            agrees = "agrees" if evaluation.get('agrees_with_math', True) else "overrides"
            print(f"[NEUTRAL] ✓ LLM {agrees}: {evaluation.get('stance')} @ {evaluation.get('position_pct')}%")
            
            # Step 3: Validate (enforce constraints)
            evaluation = self._validate_evaluation(evaluation, metrics)
            
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"[NEUTRAL] ✗ JSON error: {e}")
            return self._fallback_evaluation(metrics)
        except Exception as e:
            print(f"[NEUTRAL] ✗ Error: {e}")
            return self._fallback_evaluation(metrics)

    def generate_report(self, evaluation: Dict) -> str:
        """Generate human-readable report"""
        metrics = evaluation.get('calculated_metrics', {})
        
        report = f"""
# NEUTRAL RISK EVALUATION: {self.ticker}
{'='*70}
**Analysis Date:** {self.analysis_date or 'Current'}
**Validation Score:** {evaluation.get('validation_score', 'N/A')}/100

## CODE-CALCULATED METRICS
- **Expected Value:** {metrics.get('expected_value', 0):+.2f}%
- **Calculation:** {metrics.get('ev_calculation', 'N/A')}
- **Risk/Reward:** {metrics.get('risk_reward_ratio', 0):.2f}:1
- **Math Suggested:** {metrics.get('math_suggested_stance', 'N/A')} @ {metrics.get('math_suggested_position', 0)}%

## LLM JUDGMENT
- **Final Stance:** {evaluation.get('stance', 'N/A')}
- **Position Size:** {evaluation.get('position_pct', 0)}%
- **Confidence:** {evaluation.get('confidence', 'N/A')}
- **Agrees with Math:** {evaluation.get('agrees_with_math', 'N/A')}

## EV INTERPRETATION
{evaluation.get('ev_interpretation', 'N/A')}

## R/R INTERPRETATION
{evaluation.get('rr_interpretation', 'N/A')}

## REASONING
{evaluation.get('reasoning', 'N/A')}

## RISK CONTROLS
- **Stop Loss:** {evaluation.get('stop_loss_pct', 10)}%
- **Profit Targets:** {', '.join(f"+{t}%" for t in evaluation.get('profit_targets', []))}

{'='*70}
"""
        return report

    def run(self, synthesis_file: str, bull_file: Optional[str] = None,
            bear_file: Optional[str] = None, save_path: Optional[str] = None) -> tuple:
        """Main entry point"""
        print(f"\n{'='*70}")
        print(f"NEUTRAL DEBATOR: {self.ticker}")
        print(f"{'='*70}\n")
        
        synthesis = self.load_synthesis(synthesis_file)
        if not synthesis:
            return "Error: No synthesis data", {}
        
        bull_data, bear_data = self.load_thesis_files(bull_file, bear_file)
        
        evaluation = self.evaluate(synthesis, bull_data, bear_data)
        
        result = {
            'profile': self.risk_profile,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            **evaluation
        }
        
        report = self.generate_report(evaluation)
        print(report)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"[NEUTRAL] ✓ Saved to {save_path}")
        
        return report, result


def main():
    parser = argparse.ArgumentParser(description='Neutral Risk Debator')
    parser.add_argument('ticker', help='Stock ticker')
    parser.add_argument('--synthesis-file', required=True, help='Research synthesis JSON')
    parser.add_argument('--bull-file', help='Bull thesis JSON')
    parser.add_argument('--bear-file', help='Bear thesis JSON')
    parser.add_argument('--analysis-date', help='Historical date (YYYY-MM-DD)')
    parser.add_argument('--save-evaluation', help='Output path')
    
    args = parser.parse_args()
    
    debator = NeutralDebator(
        ticker=args.ticker,
        analysis_date=args.analysis_date
    )
    
    debator.run(
        synthesis_file=args.synthesis_file,
        bull_file=args.bull_file,
        bear_file=args.bear_file,
        save_path=args.save_evaluation
    )


if __name__ == "__main__":
    main()