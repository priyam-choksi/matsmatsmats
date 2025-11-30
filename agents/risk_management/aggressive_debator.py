"""
Aggressive Risk Debator - Math by Code, Judgment by LLM

Philosophy:
  Code = Calculator (R/R ratio, position sizing, upside potential)
  LLM = Judge (interprets opportunities, assesses catalysts, makes bold calls)

Aggressive Style:
  - Focuses on upside potential and asymmetric opportunities
  - Willing to take larger positions when conviction is high
  - Still respects risk limits (not reckless)

Usage: python aggressive_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
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


class AggressiveDebator:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.analysis_date = analysis_date
        self.risk_profile = "AGGRESSIVE"
        
        if self.analysis_date:
            print(f"[AGGRESSIVE] *** HISTORICAL MODE: As of {self.analysis_date} ***")
        
        # Guardrails - aggressive but not reckless
        self.guardrails = {
            'max_position_pct': 20,
            'min_position_pct': 0,
            'max_stop_loss_pct': 20,
            'min_stop_loss_pct': 5,
            'max_single_target_pct': 100,
            'min_target_pct': 5,
            'min_rr_for_strong_buy': 2.0,
            'min_rr_for_buy': 1.5,
            'valid_stances': ['STRONG BUY', 'BUY', 'HOLD', 'AVOID'],
            'valid_confidence': ['HIGH', 'MEDIUM', 'LOW'],
        }
        
        self.evaluation = {}
        self.calculated_metrics = {}

    def load_synthesis(self, synthesis_file: str) -> Optional[Dict]:
        """Load research synthesis"""
        if not synthesis_file or not os.path.exists(synthesis_file):
            print(f"[AGGRESSIVE] ✗ Synthesis file not found")
            return None
        
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[AGGRESSIVE] ✓ Synthesis loaded")
            
            if data.get('analysis_date') and not self.analysis_date:
                self.analysis_date = data.get('analysis_date')
            
            return data
        except Exception as e:
            print(f"[AGGRESSIVE] ✗ Load error: {e}")
            return None

    def load_thesis_files(self, bull_file: Optional[str], bear_file: Optional[str]) -> tuple:
        """Load bull/bear thesis files"""
        bull_data, bear_data = None, None
        
        if bull_file and os.path.exists(bull_file):
            try:
                with open(bull_file, 'r', encoding='utf-8') as f:
                    bull_data = json.load(f)
                print(f"[AGGRESSIVE] ✓ Bull thesis loaded")
            except Exception as e:
                print(f"[AGGRESSIVE] ⚠ Bull file error: {e}")
        
        if bear_file and os.path.exists(bear_file):
            try:
                with open(bear_file, 'r', encoding='utf-8') as f:
                    bear_data = json.load(f)
                print(f"[AGGRESSIVE] ✓ Bear thesis loaded")
            except Exception as e:
                print(f"[AGGRESSIVE] ⚠ Bear file error: {e}")
        
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
        Calculate all metrics with CODE - aggressive focuses on upside
        """
        print(f"[AGGRESSIVE] Calculating metrics with code...")
        
        bull = bull_data or synthesis.get('bull_thesis', {})
        bear = bear_data or synthesis.get('bear_thesis', {})
        probs = synthesis.get('probabilities', {})
        conclusion = synthesis.get('conclusion', {})
        
        # Extract probabilities
        bull_prob = self._extract_number(probs.get('bull_case', probs.get('bull_prob', 40)), 40) / 100
        bear_prob = self._extract_number(probs.get('bear_case', probs.get('bear_prob', 30)), 30) / 100
        
        # Extract price targets
        bull_rr = bull.get('risk_reward', {})
        bear_ra = bear.get('risk_assessment', {})
        
        upside_pct = self._extract_number(bull_rr.get('upside_pct'), 15)
        downside_pct = self._extract_number(
            bear_ra.get('downside_pct', bull_rr.get('downside_pct')), 
            10
        )
        
        # Risk/Reward ratio (aggressive cares most about this)
        risk_reward_ratio = upside_pct / downside_pct if downside_pct > 0 else 0
        
        # Expected Value
        expected_value = (upside_pct * bull_prob) - (downside_pct * bear_prob)
        
        # Asymmetry score (aggressive loves asymmetric bets)
        # Higher when upside >> downside
        asymmetry_score = (upside_pct - downside_pct) / max(downside_pct, 1)
        
        # Bull thesis conviction
        bull_conviction = bull.get('conviction', {}).get('level', 'MEDIUM')
        conviction_multiplier = {'HIGH': 1.3, 'MEDIUM': 1.0, 'LOW': 0.7}.get(bull_conviction, 1.0)
        
        # Suggested position (aggressive sizes up when R/R is good)
        if risk_reward_ratio >= self.guardrails['min_rr_for_strong_buy']:
            base_position = 15
            math_stance = 'STRONG BUY'
        elif risk_reward_ratio >= self.guardrails['min_rr_for_buy']:
            base_position = 10
            math_stance = 'BUY'
        elif expected_value > 0:
            base_position = 5
            math_stance = 'HOLD'
        else:
            base_position = 0
            math_stance = 'AVOID'
        
        suggested_position = min(base_position * conviction_multiplier, self.guardrails['max_position_pct'])
        
        # Extract catalysts count
        catalysts = bull.get('catalysts', [])
        catalyst_count = len(catalysts) if isinstance(catalysts, list) else 0
        
        metrics = {
            # Core metrics
            'upside_pct': round(upside_pct, 2),
            'downside_pct': round(downside_pct, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'expected_value': round(expected_value, 2),
            
            # Aggressive-specific
            'asymmetry_score': round(asymmetry_score, 2),
            'bull_conviction': bull_conviction,
            'catalyst_count': catalyst_count,
            
            # Probabilities
            'bull_prob_pct': round(bull_prob * 100, 1),
            'bear_prob_pct': round(bear_prob * 100, 1),
            
            # Math suggestions
            'math_suggested_stance': math_stance,
            'math_suggested_position': round(suggested_position, 1),
            
            # Thresholds
            'rr_threshold_strong_buy': self.guardrails['min_rr_for_strong_buy'],
            'rr_threshold_buy': self.guardrails['min_rr_for_buy'],
        }
        
        self.calculated_metrics = metrics
        
        print(f"[AGGRESSIVE] ✓ R/R: {metrics['risk_reward_ratio']:.2f}:1")
        print(f"[AGGRESSIVE] ✓ Upside: {metrics['upside_pct']}% | Downside: {metrics['downside_pct']}%")
        print(f"[AGGRESSIVE] ✓ Math suggests: {math_stance} @ {suggested_position:.1f}%")
        
        return metrics

    def _build_evaluation_prompt(self, synthesis: Dict, metrics: Dict,
                                  bull_data: Optional[Dict] = None,
                                  bear_data: Optional[Dict] = None) -> str:
        """Build prompt with PRE-CALCULATED metrics"""
        
        bull = bull_data or synthesis.get('bull_thesis', {})
        bear = bear_data or synthesis.get('bear_thesis', {})
        conclusion = synthesis.get('conclusion', {})
        
        # Format catalysts
        catalysts = bull.get('catalysts', [])
        catalyst_text = ""
        if catalysts:
            for c in catalysts[:3]:
                if isinstance(c, dict):
                    catalyst_text += f"  - {c.get('catalyst', 'N/A')} ({c.get('timeline', 'N/A')})\n"
        
        date_context = ""
        if self.analysis_date:
            date_context = f"⚠️ HISTORICAL MODE: Analyze as of {self.analysis_date}\n"

        prompt = f"""You are an AGGRESSIVE risk evaluator for {self.ticker}.
Your style: "Fortune favors the bold" - look for asymmetric upside opportunities.

{date_context}

═══════════════════════════════════════════════════════════════════════════════
PRE-CALCULATED METRICS (verified by code - use these exact numbers)
═══════════════════════════════════════════════════════════════════════════════

**RISK/REWARD RATIO: {metrics['risk_reward_ratio']:.2f}:1**
- Upside Potential: +{metrics['upside_pct']}%
- Downside Risk: -{metrics['downside_pct']}%

**EXPECTED VALUE: {metrics['expected_value']:+.2f}%**

**ASYMMETRY SCORE: {metrics['asymmetry_score']:.2f}**
(Positive = upside exceeds downside, aggressive likes this)

**PROBABILITIES:**
- Bull Case: {metrics['bull_prob_pct']}%
- Bear Case: {metrics['bear_prob_pct']}%

**BULL CONVICTION: {metrics['bull_conviction']}**
**CATALYST COUNT: {metrics['catalyst_count']}**

**MATH-BASED SUGGESTION:**
- Stance: {metrics['math_suggested_stance']}
- Position: {metrics['math_suggested_position']}%

**THRESHOLDS:**
- Need R/R ≥ {metrics['rr_threshold_strong_buy']}:1 for STRONG BUY
- Need R/R ≥ {metrics['rr_threshold_buy']}:1 for BUY

═══════════════════════════════════════════════════════════════════════════════
OPPORTUNITY CONTEXT
═══════════════════════════════════════════════════════════════════════════════

BULL THESIS:
{bull.get('core_thesis', 'N/A')[:500] if bull.get('core_thesis') else 'N/A'}

CATALYSTS:
{catalyst_text or '  None identified'}

BEAR RISKS (what you're accepting):
{bear.get('core_thesis', 'N/A')[:300] if bear.get('core_thesis') else 'N/A'}

RESEARCH MANAGER SAYS: {conclusion.get('recommendation', 'N/A')} ({conclusion.get('confidence', 'N/A')})

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK - AGGRESSIVE INTERPRETATION
═══════════════════════════════════════════════════════════════════════════════

As an aggressive evaluator, consider:
1. Is the R/R of {metrics['risk_reward_ratio']:.2f}:1 good enough to bet big?
2. Are there catalysts that could drive outsized gains?
3. Is the market underpricing this opportunity?
4. What asymmetric upside might others be missing?

You can be MORE aggressive than the math suggests if you see opportunity.
You can also pull back if risks seem underappreciated.

Return JSON:
{{
    "stance": "STRONG BUY/BUY/HOLD/AVOID",
    "position_pct": <0-20>,
    "confidence": "HIGH/MEDIUM/LOW",
    
    "agrees_with_math": true/false,
    "more_aggressive_than_math": true/false,
    "aggression_reasoning": "why you're being more/less aggressive",
    
    "opportunity_assessment": "what upside others might miss",
    "catalyst_assessment": "which catalysts are most compelling",
    "risks_accepting": "what risks you're consciously taking",
    
    "stop_loss_pct": <5-20>,
    "profit_targets": [<target1>, <target2>, <target3>],
    
    "reasoning": "2-3 sentences on your aggressive take",
    "key_factors": ["factor1", "factor2", "factor3"],
    
    "trading_plan": {{
        "entry": "how to enter",
        "scaling": "how to build position",
        "exit_strategy": "when to take profits"
    }},
    
    "vs_research_manager": "how your view differs"
}}

Return ONLY valid JSON."""

        return prompt

    def _validate_evaluation(self, evaluation: Dict, metrics: Dict) -> Dict:
        """Validate and apply guardrails"""
        print("[AGGRESSIVE] Applying guardrails...")
        corrections = []
        warnings = []
        
        # Inject calculated metrics
        evaluation['calculated_metrics'] = metrics
        evaluation['risk_reward_ratio'] = metrics['risk_reward_ratio']
        evaluation['expected_value'] = metrics['expected_value']
        
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
        targets = evaluation.get('profit_targets', [15, 25, 40])
        if not isinstance(targets, list) or len(targets) < 3:
            targets = [15, 25, 40]
        evaluation['profit_targets'] = sorted([
            max(self.guardrails['min_target_pct'], min(t, self.guardrails['max_single_target_pct']))
            for t in targets[:3] if isinstance(t, (int, float))
        ]) or [15, 25, 40]
        
        # Validate confidence
        conf = evaluation.get('confidence', 'MEDIUM')
        if conf not in self.guardrails['valid_confidence']:
            evaluation['confidence'] = 'MEDIUM'
        
        # R/R consistency check
        rr = metrics['risk_reward_ratio']
        stance = evaluation.get('stance')
        
        if stance == 'STRONG BUY' and rr < self.guardrails['min_rr_for_strong_buy']:
            corrections.append(f"STRONG BUY but R/R {rr:.2f} < {self.guardrails['min_rr_for_strong_buy']} → BUY")
            evaluation['stance'] = 'BUY'
        
        if stance == 'BUY' and rr < self.guardrails['min_rr_for_buy']:
            corrections.append(f"BUY but R/R {rr:.2f} < {self.guardrails['min_rr_for_buy']} → HOLD")
            evaluation['stance'] = 'HOLD'
            evaluation['position_pct'] = min(evaluation['position_pct'], 5)
        
        # Stance-position consistency
        stance = evaluation.get('stance')
        pos = evaluation.get('position_pct', 0)
        
        if stance == 'AVOID' and pos > 0:
            corrections.append(f"AVOID but position {pos}% → 0%")
            evaluation['position_pct'] = 0
        
        if stance in ['STRONG BUY', 'BUY'] and pos == 0:
            corrections.append(f"{stance} but 0% → minimum 5%")
            evaluation['position_pct'] = 5
        
        evaluation['guardrail_corrections'] = corrections
        evaluation['guardrail_warnings'] = warnings
        evaluation['validation_passed'] = len(corrections) == 0
        evaluation['validation_score'] = max(0, 100 - len(corrections) * 10 - len(warnings) * 5)
        
        if corrections:
            print(f"[AGGRESSIVE] Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        
        return evaluation

    def _fallback_evaluation(self, metrics: Dict) -> Dict:
        """Fallback using pure math"""
        return {
            'stance': metrics.get('math_suggested_stance', 'HOLD'),
            'position_pct': metrics.get('math_suggested_position', 0),
            'confidence': 'LOW',
            'risk_reward_ratio': metrics.get('risk_reward_ratio', 0),
            'expected_value': metrics.get('expected_value', 0),
            'stop_loss_pct': 10,
            'profit_targets': [15, 25, 40],
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
            print("[AGGRESSIVE] ✗ No API client - using math fallback")
            return self._fallback_evaluation(metrics)
        
        # Step 2: LLM interprets the metrics
        prompt = self._build_evaluation_prompt(synthesis, metrics, bull_data, bear_data)
        
        try:
            print(f"[AGGRESSIVE] LLM interpreting metrics...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an aggressive risk evaluator seeking asymmetric opportunities. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            evaluation = json.loads(response_text)
            
            more_agg = "more aggressive" if evaluation.get('more_aggressive_than_math') else "aligned"
            print(f"[AGGRESSIVE] ✓ LLM ({more_agg}): {evaluation.get('stance')} @ {evaluation.get('position_pct')}%")
            
            # Step 3: Validate
            evaluation = self._validate_evaluation(evaluation, metrics)
            
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"[AGGRESSIVE] ✗ JSON error: {e}")
            return self._fallback_evaluation(metrics)
        except Exception as e:
            print(f"[AGGRESSIVE] ✗ Error: {e}")
            return self._fallback_evaluation(metrics)

    def generate_report(self, evaluation: Dict) -> str:
        """Generate human-readable report"""
        metrics = evaluation.get('calculated_metrics', {})
        trading = evaluation.get('trading_plan', {})
        
        report = f"""
# AGGRESSIVE RISK EVALUATION: {self.ticker}
{'='*70}
**Analysis Date:** {self.analysis_date or 'Current'}
**Validation Score:** {evaluation.get('validation_score', 'N/A')}/100

## CODE-CALCULATED METRICS
- **Risk/Reward:** {metrics.get('risk_reward_ratio', 0):.2f}:1
- **Upside:** +{metrics.get('upside_pct', 0)}% | **Downside:** -{metrics.get('downside_pct', 0)}%
- **Expected Value:** {metrics.get('expected_value', 0):+.2f}%
- **Asymmetry Score:** {metrics.get('asymmetry_score', 0):.2f}
- **Math Suggested:** {metrics.get('math_suggested_stance', 'N/A')} @ {metrics.get('math_suggested_position', 0)}%

## LLM JUDGMENT
- **Final Stance:** {evaluation.get('stance', 'N/A')}
- **Position Size:** {evaluation.get('position_pct', 0)}%
- **Confidence:** {evaluation.get('confidence', 'N/A')}
- **More Aggressive Than Math:** {evaluation.get('more_aggressive_than_math', 'N/A')}

## OPPORTUNITY ASSESSMENT
{evaluation.get('opportunity_assessment', 'N/A')}

## RISKS ACCEPTING
{evaluation.get('risks_accepting', 'N/A')}

## REASONING
{evaluation.get('reasoning', 'N/A')}

## TRADING PLAN
- **Entry:** {trading.get('entry', 'N/A')}
- **Scaling:** {trading.get('scaling', 'N/A')}
- **Exit:** {trading.get('exit_strategy', 'N/A')}

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
        print(f"AGGRESSIVE DEBATOR: {self.ticker}")
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
            print(f"[AGGRESSIVE] ✓ Saved to {save_path}")
        
        return report, result


def main():
    parser = argparse.ArgumentParser(description='Aggressive Risk Debator')
    parser.add_argument('ticker', help='Stock ticker')
    parser.add_argument('--synthesis-file', required=True, help='Research synthesis JSON')
    parser.add_argument('--bull-file', help='Bull thesis JSON')
    parser.add_argument('--bear-file', help='Bear thesis JSON')
    parser.add_argument('--analysis-date', help='Historical date (YYYY-MM-DD)')
    parser.add_argument('--save-evaluation', help='Output path')
    
    args = parser.parse_args()
    
    debator = AggressiveDebator(
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