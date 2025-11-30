"""
Conservative Risk Debator - Math by Code, Judgment by LLM

Philosophy:
  Code = Calculator (downside risk, red flag severity, position limits)
  LLM = Judge (interprets risks, identifies concerns, makes cautious calls)

Conservative Style:
  - "First rule: don't lose money"
  - Focuses on downside protection
  - Requires high R/R (>3:1) for any position
  - Default stance is AVOID unless compellingly safe

Usage: python conservative_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
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


class ConservativeDebator:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.analysis_date = analysis_date
        self.risk_profile = "CONSERVATIVE"
        
        if self.analysis_date:
            print(f"[CONSERVATIVE] *** HISTORICAL MODE: As of {self.analysis_date} ***")
        
        # Guardrails - very strict
        self.guardrails = {
            'max_position_pct': 6,           # Conservative max is 6%
            'min_position_pct': 0,
            'max_stop_loss_pct': 10,         # Tight stops
            'min_stop_loss_pct': 3,
            'max_single_target_pct': 30,     # Realistic targets
            'min_target_pct': 5,
            'min_rr_for_any_position': 3.0,  # Need 3:1 R/R minimum
            'max_acceptable_downside': 15,   # Won't accept >15% downside
            'max_red_flags_for_position': 1, # >1 red flag = AVOID
            'valid_stances': ['SMALL BUY', 'MINIMAL BUY', 'HOLD', 'AVOID'],
            'valid_confidence': ['HIGH', 'MEDIUM', 'LOW'],
            'valid_red_flag_severity': ['HIGH', 'MEDIUM', 'LOW', 'NONE'],
        }
        
        self.evaluation = {}
        self.calculated_metrics = {}

    def load_synthesis(self, synthesis_file: str) -> Optional[Dict]:
        """Load research synthesis"""
        if not synthesis_file or not os.path.exists(synthesis_file):
            print(f"[CONSERVATIVE] ✗ Synthesis file not found")
            return None
        
        try:
            with open(synthesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[CONSERVATIVE] ✓ Synthesis loaded")
            
            if data.get('analysis_date') and not self.analysis_date:
                self.analysis_date = data.get('analysis_date')
            
            return data
        except Exception as e:
            print(f"[CONSERVATIVE] ✗ Load error: {e}")
            return None

    def load_thesis_files(self, bull_file: Optional[str], bear_file: Optional[str]) -> tuple:
        """Load bull/bear thesis files"""
        bull_data, bear_data = None, None
        
        if bull_file and os.path.exists(bull_file):
            try:
                with open(bull_file, 'r', encoding='utf-8') as f:
                    bull_data = json.load(f)
                print(f"[CONSERVATIVE] ✓ Bull thesis loaded")
            except Exception as e:
                print(f"[CONSERVATIVE] ⚠ Bull file error: {e}")
        
        if bear_file and os.path.exists(bear_file):
            try:
                with open(bear_file, 'r', encoding='utf-8') as f:
                    bear_data = json.load(f)
                print(f"[CONSERVATIVE] ✓ Bear thesis loaded")
            except Exception as e:
                print(f"[CONSERVATIVE] ⚠ Bear file error: {e}")
        
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
        Calculate all metrics with CODE - conservative focuses on DOWNSIDE
        """
        print(f"[CONSERVATIVE] Calculating metrics with code...")
        
        bull = bull_data or synthesis.get('bull_thesis', {})
        bear = bear_data or synthesis.get('bear_thesis', {})
        probs = synthesis.get('probabilities', {})
        
        # Extract probabilities
        bull_prob = self._extract_number(probs.get('bull_case', probs.get('bull_prob', 40)), 40) / 100
        bear_prob = self._extract_number(probs.get('bear_case', probs.get('bear_prob', 30)), 30) / 100
        
        # Extract price targets (conservative focuses on downside)
        bull_rr = bull.get('risk_reward', {})
        bear_ra = bear.get('risk_assessment', {})
        
        upside_pct = self._extract_number(bull_rr.get('upside_pct'), 10)
        downside_pct = self._extract_number(
            bear_ra.get('downside_pct', bull_rr.get('downside_pct')), 
            15
        )
        
        # Risk/Reward ratio
        risk_reward_ratio = upside_pct / downside_pct if downside_pct > 0 else 0
        
        # Expected Value (conservative weighs downside more heavily)
        # Using asymmetric weighting: downside counts 1.5x
        conservative_ev = (upside_pct * bull_prob) - (downside_pct * bear_prob * 1.5)
        standard_ev = (upside_pct * bull_prob) - (downside_pct * bear_prob)
        
        # Risk score from bear thesis
        bear_risk_score = self._extract_number(bear_ra.get('risk_score'), 50)
        
        # Count risk signals from bear thesis
        risk_signals = bear.get('key_risk_signals', [])
        high_severity_count = sum(1 for r in risk_signals if isinstance(r, dict) and r.get('severity') == 'high')
        total_risk_signals = len(risk_signals) if isinstance(risk_signals, list) else 0
        
        # Downside triggers
        downside_triggers = bear.get('downside_triggers', [])
        high_prob_triggers = sum(1 for t in downside_triggers if isinstance(t, dict) and t.get('probability') == 'high')
        
        # Data quality
        bull_data_quality = bull.get('conviction', {}).get('data_quality', 'moderate')
        bear_data_quality = bear.get('conviction', {}).get('data_quality', 'moderate')
        
        # Calculate red flag count (conservative's key metric)
        red_flag_count = high_severity_count + high_prob_triggers
        if downside_pct > self.guardrails['max_acceptable_downside']:
            red_flag_count += 1
        if bear_risk_score > 70:
            red_flag_count += 1
        if bull_data_quality == 'weak' or bear_data_quality == 'weak':
            red_flag_count += 1
        
        # Determine red flag severity
        if red_flag_count >= 3:
            red_flag_severity = 'HIGH'
        elif red_flag_count >= 2:
            red_flag_severity = 'MEDIUM'
        elif red_flag_count >= 1:
            red_flag_severity = 'LOW'
        else:
            red_flag_severity = 'NONE'
        
        # Math-based stance (conservative defaults to AVOID)
        if red_flag_count > self.guardrails['max_red_flags_for_position']:
            math_stance = 'AVOID'
            math_position = 0
        elif downside_pct > self.guardrails['max_acceptable_downside']:
            math_stance = 'AVOID'
            math_position = 0
        elif risk_reward_ratio >= self.guardrails['min_rr_for_any_position'] and conservative_ev > 2:
            math_stance = 'SMALL BUY'
            math_position = min(4, self.guardrails['max_position_pct'])
        elif risk_reward_ratio >= 2.0 and conservative_ev > 0:
            math_stance = 'MINIMAL BUY'
            math_position = 2
        else:
            math_stance = 'HOLD'
            math_position = 0
        
        metrics = {
            # Core metrics
            'upside_pct': round(upside_pct, 2),
            'downside_pct': round(downside_pct, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'standard_ev': round(standard_ev, 2),
            'conservative_ev': round(conservative_ev, 2),
            
            # Risk-focused metrics
            'bear_risk_score': round(bear_risk_score, 0),
            'red_flag_count': red_flag_count,
            'red_flag_severity': red_flag_severity,
            'high_severity_risks': high_severity_count,
            'high_prob_triggers': high_prob_triggers,
            
            # Data quality
            'bull_data_quality': bull_data_quality,
            'bear_data_quality': bear_data_quality,
            
            # Probabilities
            'bull_prob_pct': round(bull_prob * 100, 1),
            'bear_prob_pct': round(bear_prob * 100, 1),
            
            # Math suggestions
            'math_suggested_stance': math_stance,
            'math_suggested_position': math_position,
            
            # Thresholds
            'rr_threshold': self.guardrails['min_rr_for_any_position'],
            'max_acceptable_downside': self.guardrails['max_acceptable_downside'],
            'max_red_flags': self.guardrails['max_red_flags_for_position'],
            
            # Flags for prompt
            'downside_exceeds_threshold': downside_pct > self.guardrails['max_acceptable_downside'],
            'rr_below_threshold': risk_reward_ratio < self.guardrails['min_rr_for_any_position'],
        }
        
        self.calculated_metrics = metrics
        
        print(f"[CONSERVATIVE] ✓ Downside: {metrics['downside_pct']}% (max acceptable: {self.guardrails['max_acceptable_downside']}%)")
        print(f"[CONSERVATIVE] ✓ Red Flags: {red_flag_count} ({red_flag_severity})")
        print(f"[CONSERVATIVE] ✓ R/R: {metrics['risk_reward_ratio']:.2f}:1 (need ≥{self.guardrails['min_rr_for_any_position']}:1)")
        print(f"[CONSERVATIVE] ✓ Math suggests: {math_stance} @ {math_position}%")
        
        return metrics

    def _build_evaluation_prompt(self, synthesis: Dict, metrics: Dict,
                                  bull_data: Optional[Dict] = None,
                                  bear_data: Optional[Dict] = None) -> str:
        """Build prompt with PRE-CALCULATED metrics"""
        
        bull = bull_data or synthesis.get('bull_thesis', {})
        bear = bear_data or synthesis.get('bear_thesis', {})
        conclusion = synthesis.get('conclusion', {})
        
        # Format risk signals
        risk_signals = bear.get('key_risk_signals', [])
        risk_text = ""
        for r in risk_signals[:4]:
            if isinstance(r, dict):
                risk_text += f"  - [{r.get('severity', '?').upper()}] {r.get('signal', 'N/A')}\n"
        
        # Format downside triggers
        triggers = bear.get('downside_triggers', [])
        trigger_text = ""
        for t in triggers[:3]:
            if isinstance(t, dict):
                trigger_text += f"  - {t.get('trigger', 'N/A')} (prob: {t.get('probability', '?')})\n"
        
        risks_to_monitor = synthesis.get('risks_to_monitor', [])
        
        date_context = ""
        if self.analysis_date:
            date_context = f"⚠️ HISTORICAL MODE: Analyze as of {self.analysis_date}\n"

        prompt = f"""You are a CONSERVATIVE risk evaluator for {self.ticker}.
Your style: "First rule of investing: don't lose money." Capital preservation above all.

{date_context}

═══════════════════════════════════════════════════════════════════════════════
PRE-CALCULATED RISK METRICS (verified by code)
═══════════════════════════════════════════════════════════════════════════════

**🚨 RED FLAG COUNT: {metrics['red_flag_count']}** (Severity: {metrics['red_flag_severity']})
- Max allowed for any position: {metrics['max_red_flags']}

**DOWNSIDE RISK: {metrics['downside_pct']}%**
- Max acceptable: {metrics['max_acceptable_downside']}%
- EXCEEDS THRESHOLD: {'⚠️ YES' if metrics['downside_exceeds_threshold'] else '✓ No'}

**RISK/REWARD RATIO: {metrics['risk_reward_ratio']:.2f}:1**
- Conservative minimum: {metrics['rr_threshold']}:1
- BELOW THRESHOLD: {'⚠️ YES' if metrics['rr_below_threshold'] else '✓ No'}

**CONSERVATIVE EV: {metrics['conservative_ev']:+.2f}%**
(Downside weighted 1.5x in calculation)

**BEAR RISK SCORE: {metrics['bear_risk_score']}/100**

**DATA QUALITY:**
- Bull thesis: {metrics['bull_data_quality']}
- Bear thesis: {metrics['bear_data_quality']}

**MATH-BASED SUGGESTION:**
- Stance: {metrics['math_suggested_stance']}
- Position: {metrics['math_suggested_position']}%

═══════════════════════════════════════════════════════════════════════════════
RISK DETAILS (focus here)
═══════════════════════════════════════════════════════════════════════════════

BEAR THESIS:
{bear.get('core_thesis', 'N/A')[:500] if bear.get('core_thesis') else 'N/A'}

KEY RISK SIGNALS:
{risk_text or '  None identified'}

DOWNSIDE TRIGGERS:
{trigger_text or '  None identified'}

RISKS TO MONITOR:
{json.dumps(risks_to_monitor[:3], indent=2) if risks_to_monitor else '  None'}

═══════════════════════════════════════════════════════════════════════════════
BULL CASE (view skeptically)
═══════════════════════════════════════════════════════════════════════════════

{bull.get('core_thesis', 'N/A')[:300] if bull.get('core_thesis') else 'N/A'}

RESEARCH MANAGER SAYS: {conclusion.get('recommendation', 'N/A')} ({conclusion.get('confidence', 'N/A')})

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK - CONSERVATIVE RISK ASSESSMENT
═══════════════════════════════════════════════════════════════════════════════

As a conservative evaluator, your default is AVOID. Only recommend a position if:
1. Red flags ≤ {metrics['max_red_flags']} ✓/✗
2. Downside ≤ {metrics['max_acceptable_downside']}% ✓/✗
3. R/R ≥ {metrics['rr_threshold']}:1 ✓/✗
4. Data quality is not weak

Ask yourself: "What could go wrong?" and "Can I afford to be wrong?"

Return JSON:
{{
    "stance": "SMALL BUY/MINIMAL BUY/HOLD/AVOID",
    "position_pct": <0-6>,
    "confidence": "HIGH/MEDIUM/LOW",
    
    "agrees_with_math": true/false,
    "more_conservative_than_math": true/false,
    "caution_reasoning": "why you're being more/less cautious",
    
    "red_flags": [
        {{"flag": "description", "severity": "HIGH/MEDIUM/LOW", "source": "where identified"}}
    ],
    "red_flag_severity": "HIGH/MEDIUM/LOW/NONE",
    
    "what_could_go_wrong": "worst case scenario description",
    "downside_assessment": {{
        "stated_downside": {metrics['downside_pct']},
        "your_estimate": <your estimate>,
        "within_tolerance": true/false,
        "reasoning": "why you agree/disagree with stated downside"
    }},
    
    "bull_case_problems": ["problem 1", "problem 2"],
    "conditions_to_reconsider": ["condition 1", "condition 2"],
    
    "stop_loss_pct": <3-10, tight stops>,
    "profit_targets": [<target1>, <target2>],
    
    "reasoning": "2-3 sentences on your conservative assessment",
    "key_concerns": ["concern 1", "concern 2", "concern 3"],
    
    "vs_research_manager": "how your view differs"
}}

Return ONLY valid JSON."""

        return prompt

    def _validate_evaluation(self, evaluation: Dict, metrics: Dict) -> Dict:
        """Validate and apply strict conservative guardrails"""
        print("[CONSERVATIVE] Applying guardrails...")
        corrections = []
        warnings = []
        
        # Inject calculated metrics
        evaluation['calculated_metrics'] = metrics
        evaluation['risk_reward_ratio'] = metrics['risk_reward_ratio']
        evaluation['conservative_ev'] = metrics['conservative_ev']
        
        # Validate stance (conservative-only options)
        stance = evaluation.get('stance', 'AVOID')
        if stance not in self.guardrails['valid_stances']:
            if stance in ['STRONG BUY', 'BUY']:
                corrections.append(f"'{stance}' not allowed for conservative → SMALL BUY")
                evaluation['stance'] = 'SMALL BUY'
            else:
                corrections.append(f"Invalid stance '{stance}' → AVOID")
                evaluation['stance'] = 'AVOID'
        
        # Validate position size (strict cap)
        pos = evaluation.get('position_pct', 0)
        if not isinstance(pos, (int, float)):
            pos = 0
        if pos > self.guardrails['max_position_pct']:
            corrections.append(f"Position {pos}% capped to {self.guardrails['max_position_pct']}% (conservative max)")
            pos = self.guardrails['max_position_pct']
        if pos < 0:
            pos = 0
        evaluation['position_pct'] = pos
        
        # Validate stop loss (tight stops required)
        stop = evaluation.get('stop_loss_pct', 5)
        if not isinstance(stop, (int, float)):
            stop = 5
        stop = max(self.guardrails['min_stop_loss_pct'], min(stop, self.guardrails['max_stop_loss_pct']))
        evaluation['stop_loss_pct'] = stop
        
        # Validate profit targets (realistic only)
        targets = evaluation.get('profit_targets', [8, 15])
        if not isinstance(targets, list) or len(targets) < 2:
            targets = [8, 15]
        evaluation['profit_targets'] = sorted([
            max(self.guardrails['min_target_pct'], min(t, self.guardrails['max_single_target_pct']))
            for t in targets[:2] if isinstance(t, (int, float))
        ]) or [8, 15]
        
        # Validate confidence
        conf = evaluation.get('confidence', 'LOW')
        if conf not in self.guardrails['valid_confidence']:
            evaluation['confidence'] = 'LOW'
        
        # Validate red flag severity
        severity = evaluation.get('red_flag_severity', metrics['red_flag_severity'])
        if severity not in self.guardrails['valid_red_flag_severity']:
            evaluation['red_flag_severity'] = metrics['red_flag_severity']
        
        # CRITICAL: Enforce conservative rules
        red_flag_count = metrics['red_flag_count']
        stance = evaluation.get('stance')
        pos = evaluation.get('position_pct', 0)
        
        # Rule 1: Too many red flags = AVOID
        if red_flag_count > self.guardrails['max_red_flags_for_position'] and pos > 0:
            corrections.append(f"{red_flag_count} red flags > {self.guardrails['max_red_flags_for_position']} → AVOID")
            evaluation['stance'] = 'AVOID'
            evaluation['position_pct'] = 0
        
        # Rule 2: Downside too high = AVOID
        if metrics['downside_exceeds_threshold'] and pos > 0:
            corrections.append(f"Downside {metrics['downside_pct']}% > {self.guardrails['max_acceptable_downside']}% → AVOID")
            evaluation['stance'] = 'AVOID'
            evaluation['position_pct'] = 0
        
        # Rule 3: R/R too low = max HOLD
        if metrics['rr_below_threshold'] and stance in ['SMALL BUY', 'MINIMAL BUY']:
            corrections.append(f"R/R {metrics['risk_reward_ratio']:.2f} < {self.guardrails['min_rr_for_any_position']} → HOLD")
            evaluation['stance'] = 'HOLD'
            evaluation['position_pct'] = 0
        
        # Stance-position consistency
        stance = evaluation.get('stance')
        pos = evaluation.get('position_pct', 0)
        
        if stance == 'AVOID' and pos > 0:
            corrections.append(f"AVOID but position {pos}% → 0%")
            evaluation['position_pct'] = 0
        
        if stance == 'HOLD' and pos > 2:
            corrections.append(f"HOLD but position {pos}% → max 2%")
            evaluation['position_pct'] = 2
        
        if stance == 'MINIMAL BUY' and pos > 3:
            corrections.append(f"MINIMAL BUY but position {pos}% → max 3%")
            evaluation['position_pct'] = 3
        
        # Ensure red_flag_count is in output
        evaluation['red_flag_count'] = red_flag_count
        
        evaluation['guardrail_corrections'] = corrections
        evaluation['guardrail_warnings'] = warnings
        evaluation['validation_passed'] = len(corrections) == 0
        evaluation['validation_score'] = max(0, 100 - len(corrections) * 10 - len(warnings) * 5)
        
        if corrections:
            print(f"[CONSERVATIVE] Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        
        return evaluation

    def _fallback_evaluation(self, metrics: Dict) -> Dict:
        """Fallback - conservative defaults to AVOID"""
        return {
            'stance': metrics.get('math_suggested_stance', 'AVOID'),
            'position_pct': metrics.get('math_suggested_position', 0),
            'confidence': 'LOW',
            'risk_reward_ratio': metrics.get('risk_reward_ratio', 0),
            'conservative_ev': metrics.get('conservative_ev', 0),
            'red_flags': [{'flag': 'Analysis failed', 'severity': 'HIGH', 'source': 'System'}],
            'red_flag_count': metrics.get('red_flag_count', 1),
            'red_flag_severity': metrics.get('red_flag_severity', 'HIGH'),
            'what_could_go_wrong': 'Unable to complete risk analysis - conservative approach requires AVOID',
            'stop_loss_pct': 5,
            'profit_targets': [8, 15],
            'reasoning': 'Fallback to AVOID (LLM unavailable)',
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
            print("[CONSERVATIVE] ✗ No API client - using math fallback")
            return self._fallback_evaluation(metrics)
        
        # Step 2: LLM interprets the metrics
        prompt = self._build_evaluation_prompt(synthesis, metrics, bull_data, bear_data)
        
        try:
            print(f"[CONSERVATIVE] LLM interpreting risks...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a conservative risk evaluator focused on capital preservation. Default to caution. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            evaluation = json.loads(response_text)
            
            red_flags = len(evaluation.get('red_flags', []))
            print(f"[CONSERVATIVE] ✓ LLM: {evaluation.get('stance')} @ {evaluation.get('position_pct')}% ({red_flags} red flags)")
            
            # Step 3: Validate (enforce strict conservative rules)
            evaluation = self._validate_evaluation(evaluation, metrics)
            
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"[CONSERVATIVE] ✗ JSON error: {e}")
            return self._fallback_evaluation(metrics)
        except Exception as e:
            print(f"[CONSERVATIVE] ✗ Error: {e}")
            return self._fallback_evaluation(metrics)

    def generate_report(self, evaluation: Dict) -> str:
        """Generate human-readable report"""
        metrics = evaluation.get('calculated_metrics', {})
        downside = evaluation.get('downside_assessment', {})
        
        # Format red flags
        flags_text = ""
        for flag in evaluation.get('red_flags', [])[:5]:
            if isinstance(flag, dict):
                flags_text += f"  🚩 [{flag.get('severity', '?')}] {flag.get('flag', 'N/A')}\n"
        
        report = f"""
# CONSERVATIVE RISK EVALUATION: {self.ticker}
{'='*70}
**Analysis Date:** {self.analysis_date or 'Current'}
**Validation Score:** {evaluation.get('validation_score', 'N/A')}/100

## CODE-CALCULATED RISK METRICS
- **Red Flags:** {metrics.get('red_flag_count', 0)} ({metrics.get('red_flag_severity', 'N/A')})
- **Downside Risk:** {metrics.get('downside_pct', 0)}% (max acceptable: {metrics.get('max_acceptable_downside', 15)}%)
- **Risk/Reward:** {metrics.get('risk_reward_ratio', 0):.2f}:1 (need ≥{metrics.get('rr_threshold', 3)}:1)
- **Conservative EV:** {metrics.get('conservative_ev', 0):+.2f}%
- **Bear Risk Score:** {metrics.get('bear_risk_score', 0)}/100
- **Math Suggested:** {metrics.get('math_suggested_stance', 'N/A')} @ {metrics.get('math_suggested_position', 0)}%

## RED FLAGS IDENTIFIED
{flags_text or '  ✓ No red flags identified'}

## LLM JUDGMENT
- **Final Stance:** {evaluation.get('stance', 'N/A')}
- **Position Size:** {evaluation.get('position_pct', 0)}% (max 6%)
- **Confidence:** {evaluation.get('confidence', 'N/A')}

## WHAT COULD GO WRONG
{evaluation.get('what_could_go_wrong', 'N/A')}

## DOWNSIDE ASSESSMENT
- Stated: {downside.get('stated_downside', 'N/A')}%
- My Estimate: {downside.get('your_estimate', 'N/A')}%
- Within Tolerance: {downside.get('within_tolerance', 'N/A')}

## KEY CONCERNS
{chr(10).join(f"• {c}" for c in evaluation.get('key_concerns', [])) or 'None listed'}

## BULL CASE PROBLEMS
{chr(10).join(f"• {p}" for p in evaluation.get('bull_case_problems', [])) or 'None listed'}

## CONDITIONS TO RECONSIDER
{chr(10).join(f"• {c}" for c in evaluation.get('conditions_to_reconsider', [])) or 'None listed'}

## REASONING
{evaluation.get('reasoning', 'N/A')}

## RISK CONTROLS
- **Stop Loss:** {evaluation.get('stop_loss_pct', 5)}% (tight)
- **Profit Targets:** {', '.join(f"+{t}%" for t in evaluation.get('profit_targets', []))}

{'='*70}
"""
        return report

    def run(self, synthesis_file: str, bull_file: Optional[str] = None,
            bear_file: Optional[str] = None, save_path: Optional[str] = None) -> tuple:
        """Main entry point"""
        print(f"\n{'='*70}")
        print(f"CONSERVATIVE DEBATOR: {self.ticker}")
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
            print(f"[CONSERVATIVE] ✓ Saved to {save_path}")
        
        return report, result


def main():
    parser = argparse.ArgumentParser(description='Conservative Risk Debator')
    parser.add_argument('ticker', help='Stock ticker')
    parser.add_argument('--synthesis-file', required=True, help='Research synthesis JSON')
    parser.add_argument('--bull-file', help='Bull thesis JSON')
    parser.add_argument('--bear-file', help='Bear thesis JSON')
    parser.add_argument('--analysis-date', help='Historical date (YYYY-MM-DD)')
    parser.add_argument('--save-evaluation', help='Output path')
    
    args = parser.parse_args()
    
    debator = ConservativeDebator(
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