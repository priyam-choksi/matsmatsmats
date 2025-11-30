"""
Research Manager - LLM-Driven Synthesis
LLM does the heavy lifting: reads theses, runs debate, synthesizes decision
Code only provides guardrails to catch unrealistic outputs

Philosophy:
  LLM = Brain (analyzes, debates, decides)
  Code = Safety net (validates bounds, catches errors)

Usage: 
  python research_manager.py AAPL --bull-file ... --bear-file ...
  python research_manager.py AAPL --bull-file ... --bear-file ... --debate-rounds 3
  python research_manager.py AAPL --bull-file ... --bear-file ... --analysis-date 2024-06-15
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


class ResearchManager:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[RESEARCH_MGR] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[RESEARCH_MGR] Running in LIVE mode")
        
        # Research inputs container
        self.research_inputs = {
            'bull_thesis': {},
            'bear_thesis': {},
            'risk_evaluations': {},
            'debate_history': []
        }
        
        # Guardrails for synthesis validation
        self.guardrails = {
            # Probability bounds
            'max_probability': 95,
            'min_probability': 5,
            'probabilities_must_sum_to_100': True,
            
            # Position size bounds
            'max_position_pct': 20,
            'min_position_pct': 0,
            
            # Valid enums
            'valid_recommendations': ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'],
            'valid_confidence': ['HIGH', 'MEDIUM', 'LOW'],
            'valid_time_horizons': ['1-2 weeks', '1 month', '1-3 months', '3-6 months', '6-12 months', '1+ year'],
            
            # Consistency rules
            'strong_buy_requires_bull_majority': True,
            'strong_sell_requires_bear_majority': True,
            'high_confidence_requires_data_quality': True,
            
            # Required fields
            'required_conclusion_fields': ['recommendation', 'confidence', 'position_size', 'rationale'],
            'required_probability_fields': ['bull_case', 'bear_case', 'base_case'],
        }
        
        # Debate prompts
        self.bull_debate_prompt = """You are a Bull Analyst in Round {round_num} of an investment debate for {ticker}.

Your thesis summary:
{thesis}

{previous_argument}

**DEBATE RULES:**
1. Directly address the opponent's specific points
2. Use data and evidence from your thesis
3. Don't fabricate new data - stick to what's in your thesis
4. Be persuasive but factual
5. End with your strongest point

Respond in 300-500 words."""

        self.bear_debate_prompt = """You are a Bear Analyst in Round {round_num} of an investment debate for {ticker}.

Your thesis summary:
{thesis}

{previous_argument}

**DEBATE RULES:**
1. Directly address the opponent's specific points
2. Use data and evidence from your thesis
3. Don't fabricate new data - stick to what's in your thesis
4. Be persuasive but factual
5. End with your key risk concern

Respond in 300-500 words."""

    def load_research_files(self, bull_file: str, bear_file: str) -> bool:
        """Load bull and bear thesis files"""
        loaded = 0
        
        if os.path.exists(bull_file):
            try:
                with open(bull_file, 'r', encoding='utf-8') as f:
                    self.research_inputs['bull_thesis'] = json.load(f)
                bull_validation = self.research_inputs['bull_thesis'].get('validation_score', 'N/A')
                print(f"[RESEARCH_MGR] ✓ Bull thesis loaded (validation: {bull_validation})")
                loaded += 1
            except Exception as e:
                print(f"[RESEARCH_MGR] ⚠ Bull thesis error: {e}")
        
        if os.path.exists(bear_file):
            try:
                with open(bear_file, 'r', encoding='utf-8') as f:
                    self.research_inputs['bear_thesis'] = json.load(f)
                bear_validation = self.research_inputs['bear_thesis'].get('validation_score', 'N/A')
                print(f"[RESEARCH_MGR] ✓ Bear thesis loaded (validation: {bear_validation})")
                loaded += 1
            except Exception as e:
                print(f"[RESEARCH_MGR] ⚠ Bear thesis error: {e}")
        
        return loaded == 2

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
                    print(f"[RESEARCH_MGR] ⚠ Error loading {risk_type}: {e}")
        
        print(f"[RESEARCH_MGR] ✓ Loaded {loaded}/3 risk evaluations")

    def run_debate(self, rounds: int = 3) -> List[Dict]:
        """Run internal debate between bull and bear positions"""
        if not self.client:
            print("[RESEARCH_MGR] ⚠ No API client, skipping debate")
            return []
        
        bull_thesis = self.research_inputs.get('bull_thesis', {})
        bear_thesis = self.research_inputs.get('bear_thesis', {})
        
        if not bull_thesis or not bear_thesis:
            print("[RESEARCH_MGR] ⚠ Missing thesis data, skipping debate")
            return []
        
        # Build thesis summaries for debate
        bull_summary = self._build_thesis_summary(bull_thesis, 'bull')
        bear_summary = self._build_thesis_summary(bear_thesis, 'bear')
        
        print(f"\n{'='*70}")
        print(f"INVESTMENT DEBATE: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*70}")
        print(f"Rounds: {rounds}")
        print(f"{'='*70}\n")
        
        debate_history = []
        
        for round_num in range(1, rounds + 1):
            print(f"--- Round {round_num} ---")
            
            # Bull argues
            prev_arg = ""
            if round_num > 1:
                prev_arg = f"**Opponent's last argument (address this):**\n{debate_history[-1]['argument'][:800]}"
            
            bull_prompt = self.bull_debate_prompt.format(
                round_num=round_num,
                ticker=self.ticker,
                thesis=bull_summary,
                previous_argument=prev_arg
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": bull_prompt}],
                    temperature=0.7,
                    max_tokens=700
                )
                bull_arg = response.choices[0].message.content
                debate_history.append({'round': round_num, 'side': 'bull', 'argument': bull_arg})
                print(f"  🐂 Bull: {len(bull_arg)} chars")
            except Exception as e:
                print(f"  ⚠ Bull debate error: {e}")
                continue
            
            # Bear responds
            bear_prompt = self.bear_debate_prompt.format(
                round_num=round_num,
                ticker=self.ticker,
                thesis=bear_summary,
                previous_argument=f"**Opponent's argument (address this):**\n{bull_arg[:800]}"
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": bear_prompt}],
                    temperature=0.7,
                    max_tokens=700
                )
                bear_arg = response.choices[0].message.content
                debate_history.append({'round': round_num, 'side': 'bear', 'argument': bear_arg})
                print(f"  🐻 Bear: {len(bear_arg)} chars")
            except Exception as e:
                print(f"  ⚠ Bear debate error: {e}")
        
        self.research_inputs['debate_history'] = debate_history
        print(f"\n[RESEARCH_MGR] ✓ Debate complete: {len(debate_history)} arguments")
        
        return debate_history

    def _build_thesis_summary(self, thesis: Dict, side: str) -> str:
        """Build a summary of thesis for debate context"""
        if side == 'bull':
            rr = thesis.get('risk_reward', {})
            conviction = thesis.get('conviction', {})
            return f"""
CORE THESIS: {thesis.get('core_thesis', 'N/A')}

KEY POINTS:
- Upside potential: {rr.get('upside_pct', 'N/A')}%
- Downside risk: {rr.get('downside_pct', 'N/A')}%
- R/R Ratio: {rr.get('reward_risk_ratio', 'N/A')}:1
- Conviction: {conviction.get('level', 'N/A')}
- Data quality: {conviction.get('data_quality', 'N/A')}

TOP BULLISH SIGNALS:
{self._format_signals(thesis.get('key_bullish_signals', [])[:3])}

CATALYSTS:
{self._format_catalysts(thesis.get('catalysts', [])[:2])}
"""
        else:  # bear
            ra = thesis.get('risk_assessment', {})
            conviction = thesis.get('conviction', {})
            return f"""
CORE THESIS: {thesis.get('core_thesis', 'N/A')}

KEY POINTS:
- Downside risk: {ra.get('downside_pct', 'N/A')}%
- Limited upside: {ra.get('limited_upside_pct', 'N/A')}%
- Risk score: {ra.get('risk_score', 'N/A')}/100
- Conviction: {conviction.get('level', 'N/A')}
- Data quality: {conviction.get('data_quality', 'N/A')}

TOP RISK SIGNALS:
{self._format_signals(thesis.get('key_risk_signals', [])[:3])}

DOWNSIDE TRIGGERS:
{self._format_triggers(thesis.get('downside_triggers', [])[:2])}
"""

    def _format_signals(self, signals: List[Dict]) -> str:
        if not signals:
            return "  (none provided)"
        return "\n".join(f"  - [{s.get('source', '?')}] {s.get('signal', 'N/A')}" for s in signals)

    def _format_catalysts(self, catalysts: List[Dict]) -> str:
        if not catalysts:
            return "  (none provided)"
        return "\n".join(f"  - {c.get('catalyst', 'N/A')} ({c.get('timeline', '?')})" for c in catalysts)

    def _format_triggers(self, triggers: List[Dict]) -> str:
        if not triggers:
            return "  (none provided)"
        return "\n".join(f"  - {t.get('trigger', 'N/A')} ({t.get('probability', '?')} prob)" for t in triggers)

    def _build_synthesis_prompt(self) -> str:
        """Build prompt for LLM to synthesize everything and decide"""
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        debate = self.research_inputs.get('debate_history', [])
        risk_evals = self.research_inputs.get('risk_evaluations', {})
        
        date_context = ""
        if self.analysis_date:
            date_context = f"""
⚠️ HISTORICAL ANALYSIS MODE ⚠️
You are making this decision as of {self.analysis_date}.
Do NOT use any information from after this date.
"""

        # Format debate
        debate_text = ""
        if debate:
            debate_text = "\n\n## DEBATE TRANSCRIPT\n"
            for entry in debate:
                icon = "🐂 BULL" if entry['side'] == 'bull' else "🐻 BEAR"
                debate_text += f"\n### Round {entry['round']} - {icon}\n{entry['argument']}\n"
        else:
            debate_text = "\n\n## NO DEBATE CONDUCTED\n"

        # Format risk evaluations
        risk_text = "\n## RISK TEAM EVALUATIONS\n"
        if risk_evals:
            for risk_type, eval_data in risk_evals.items():
                risk_text += f"\n### {risk_type.upper()}\n"
                risk_text += f"- Stance: {eval_data.get('stance', 'N/A')}\n"
                risk_text += f"- Position: {eval_data.get('position_pct', 'N/A')}%\n"
                risk_text += f"- Confidence: {eval_data.get('confidence', 'N/A')}\n"
        else:
            risk_text += "(No risk evaluations available)\n"

        prompt = f"""You are the Research Manager for {self.ticker}. Your job is to synthesize all research and make the FINAL investment decision.

{date_context}

═══════════════════════════════════════════════════════════════════════
BULL THESIS
═══════════════════════════════════════════════════════════════════════
{self._build_thesis_summary(bull, 'bull')}

**Bull Validation Score:** {bull.get('validation_score', 'N/A')}/100
**Bull Recommendation:** {bull.get('recommendation', {}).get('action', 'N/A')}

═══════════════════════════════════════════════════════════════════════
BEAR THESIS
═══════════════════════════════════════════════════════════════════════
{self._build_thesis_summary(bear, 'bear')}

**Bear Validation Score:** {bear.get('validation_score', 'N/A')}/100
**Bear Recommendation:** {bear.get('recommendation', {}).get('action', 'N/A')}
{debate_text}
{risk_text}

═══════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════

Analyze everything above and make your FINAL decision.

Consider:
1. Which thesis has stronger evidence? (check validation scores and data quality)
2. Who won the debate? Which arguments were more convincing?
3. What do the risk evaluators say?
4. What's the overall risk/reward picture?

CRITICAL RULES:
- Base your decision ONLY on the information provided above
- Don't fabricate new data or analysis
- If both theses have weak data, be conservative
- Explain your reasoning clearly
- Cite specific evidence from the theses/debate

Return your synthesis as JSON:
{{
    "probabilities": {{
        "bull_case": <0-100, your assessment>,
        "bear_case": <0-100>,
        "base_case": <0-100>,
        "rationale": "why these probabilities"
    }},
    
    "debate_winner": {{
        "winner": "bull/bear/tie",
        "reasoning": "who had better arguments and why",
        "key_points": ["winning argument 1", "winning argument 2"]
    }},
    
    "thesis_quality": {{
        "bull_quality": "strong/moderate/weak",
        "bear_quality": "strong/moderate/weak",
        "better_thesis": "bull/bear/equal",
        "reasoning": "which thesis was better supported"
    }},
    
    "conclusion": {{
        "recommendation": "STRONG BUY/BUY/HOLD/SELL/STRONG SELL",
        "confidence": "HIGH/MEDIUM/LOW",
        "position_size": "<percentage>%",
        "stop_loss": "<percentage>%",
        "target": "<percentage>% upside",
        "time_horizon": "expected holding period",
        "rationale": "detailed explanation of your decision (2-4 sentences)"
    }},
    
    "key_factors": [
        {{"factor": "description", "impact": "bullish/bearish/neutral", "weight": "high/medium/low"}}
    ],
    
    "risks_to_monitor": [
        "risk 1 to watch",
        "risk 2 to watch"
    ],
    
    "full_synthesis": "Your complete written synthesis (2-3 paragraphs summarizing the decision)"
}}

Return ONLY valid JSON. No markdown, no explanation outside JSON."""

        return prompt

    def _validate_synthesis(self, synthesis: Dict) -> Dict:
        """Comprehensive guardrails for synthesis output"""
        print("[RESEARCH_MGR] Applying synthesis guardrails...")
        corrections = []
        warnings = []
        
        # ═══════════════════════════════════════════════════════════════
        # 1. VALIDATE PROBABILITIES
        # ═══════════════════════════════════════════════════════════════
        probs = synthesis.get('probabilities', {})
        
        bull_prob = probs.get('bull_case')
        bear_prob = probs.get('bear_case')
        base_prob = probs.get('base_case')
        
        # Check types
        for name, val in [('bull_case', bull_prob), ('bear_case', bear_prob), ('base_case', base_prob)]:
            if val is not None and not isinstance(val, (int, float)):
                warnings.append(f"INVALID TYPE: {name} is {type(val).__name__}")
                probs[name] = 33  # Default
        
        # Check bounds
        for name, val in [('bull_case', probs.get('bull_case')), 
                          ('bear_case', probs.get('bear_case')), 
                          ('base_case', probs.get('base_case'))]:
            if val is not None:
                if val > self.guardrails['max_probability']:
                    corrections.append(f"{name} {val}% capped to {self.guardrails['max_probability']}%")
                    probs[name] = self.guardrails['max_probability']
                elif val < self.guardrails['min_probability']:
                    corrections.append(f"{name} {val}% floored to {self.guardrails['min_probability']}%")
                    probs[name] = self.guardrails['min_probability']
        
        # Check sum
        total = (probs.get('bull_case', 0) or 0) + (probs.get('bear_case', 0) or 0) + (probs.get('base_case', 0) or 0)
        if abs(total - 100) > 5:
            warnings.append(f"Probabilities sum to {total}%, not 100%")
            # Normalize
            if total > 0:
                probs['bull_case'] = round((probs.get('bull_case', 0) / total) * 100, 1)
                probs['bear_case'] = round((probs.get('bear_case', 0) / total) * 100, 1)
                probs['base_case'] = round(100 - probs['bull_case'] - probs['bear_case'], 1)
                corrections.append(f"Probabilities normalized to sum to 100%")
        
        # ═══════════════════════════════════════════════════════════════
        # 2. VALIDATE CONCLUSION
        # ═══════════════════════════════════════════════════════════════
        conclusion = synthesis.get('conclusion', {})
        
        # Check required fields
        for field in self.guardrails['required_conclusion_fields']:
            if field not in conclusion or not conclusion[field]:
                warnings.append(f"MISSING: conclusion.{field}")
        
        # Check recommendation enum
        rec = conclusion.get('recommendation')
        if rec not in self.guardrails['valid_recommendations']:
            corrections.append(f"Invalid recommendation '{rec}' → HOLD")
            conclusion['recommendation'] = 'HOLD'
        
        # Check confidence enum
        conf = conclusion.get('confidence')
        if conf not in self.guardrails['valid_confidence']:
            corrections.append(f"Invalid confidence '{conf}' → MEDIUM")
            conclusion['confidence'] = 'MEDIUM'
        
        # ═══════════════════════════════════════════════════════════════
        # 3. VALIDATE POSITION SIZE
        # ═══════════════════════════════════════════════════════════════
        pos_size = conclusion.get('position_size', '')
        if pos_size:
            # Extract number from string like "10%" or "5-10%"
            nums = re.findall(r'\d+', str(pos_size))
            if nums:
                max_pos = max(int(n) for n in nums)
                if max_pos > self.guardrails['max_position_pct']:
                    corrections.append(f"Position {max_pos}% capped to {self.guardrails['max_position_pct']}%")
                    conclusion['position_size'] = f"{self.guardrails['max_position_pct']}%"
                    conclusion['position_capped'] = True
        
        # ═══════════════════════════════════════════════════════════════
        # 4. VALIDATE CONSISTENCY
        # ═══════════════════════════════════════════════════════════════
        bull_prob = probs.get('bull_case', 50)
        bear_prob = probs.get('bear_case', 50)
        rec = conclusion.get('recommendation', 'HOLD')
        
        # STRONG BUY should have bull majority
        if rec == 'STRONG BUY' and bull_prob < bear_prob:
            warnings.append(f"INCONSISTENT: STRONG BUY but bull_prob ({bull_prob}%) < bear_prob ({bear_prob}%)")
            corrections.append("STRONG BUY → BUY (bear probability higher)")
            conclusion['recommendation'] = 'BUY'
            conclusion['downgraded_reason'] = 'probabilities favor bear'
        
        # STRONG SELL should have bear majority
        if rec == 'STRONG SELL' and bear_prob < bull_prob:
            warnings.append(f"INCONSISTENT: STRONG SELL but bear_prob ({bear_prob}%) < bull_prob ({bull_prob}%)")
            corrections.append("STRONG SELL → SELL (bull probability higher)")
            conclusion['recommendation'] = 'SELL'
            conclusion['downgraded_reason'] = 'probabilities favor bull'
        
        # HIGH confidence needs good thesis quality
        thesis_quality = synthesis.get('thesis_quality', {})
        if conclusion.get('confidence') == 'HIGH':
            bull_q = thesis_quality.get('bull_quality', 'weak')
            bear_q = thesis_quality.get('bear_quality', 'weak')
            if bull_q == 'weak' and bear_q == 'weak':
                warnings.append("INCONSISTENT: HIGH confidence but both theses are weak quality")
                corrections.append("HIGH confidence → MEDIUM (weak thesis quality)")
                conclusion['confidence'] = 'MEDIUM'
                conclusion['confidence_downgraded'] = True
        
        # ═══════════════════════════════════════════════════════════════
        # 5. VALIDATE DEBATE WINNER
        # ═══════════════════════════════════════════════════════════════
        debate_winner = synthesis.get('debate_winner', {})
        winner = debate_winner.get('winner', '').lower()
        
        if winner not in ['bull', 'bear', 'tie']:
            corrections.append(f"Invalid debate winner '{winner}' → tie")
            debate_winner['winner'] = 'tie'
        
        # Check consistency with recommendation
        if winner == 'bull' and rec in ['SELL', 'STRONG SELL']:
            warnings.append(f"INCONSISTENT: Bull won debate but recommendation is {rec}")
        elif winner == 'bear' and rec in ['BUY', 'STRONG BUY']:
            warnings.append(f"INCONSISTENT: Bear won debate but recommendation is {rec}")
        
        # ═══════════════════════════════════════════════════════════════
        # 6. VALIDATE RATIONALE EXISTS
        # ═══════════════════════════════════════════════════════════════
        rationale = conclusion.get('rationale', '')
        if not rationale or len(rationale) < 30:
            warnings.append("WEAK RATIONALE: Conclusion has no proper explanation")
            conclusion['rationale_warning'] = True
        
        prob_rationale = probs.get('rationale', '')
        if not prob_rationale or len(prob_rationale) < 20:
            warnings.append("WEAK RATIONALE: Probabilities have no explanation")
        
        # ═══════════════════════════════════════════════════════════════
        # 7. STORE VALIDATION RESULTS
        # ═══════════════════════════════════════════════════════════════
        synthesis['guardrail_corrections'] = corrections
        synthesis['guardrail_warnings'] = warnings
        synthesis['validation_passed'] = len(warnings) == 0
        synthesis['validation_score'] = max(0, 100 - len(warnings) * 10 - len(corrections) * 5)
        
        # Print summary
        print(f"[RESEARCH_MGR] Validation score: {synthesis['validation_score']}/100")
        if corrections:
            print(f"[RESEARCH_MGR] ⚠ Applied {len(corrections)} correction(s):")
            for c in corrections:
                print(f"    → {c}")
        if warnings:
            print(f"[RESEARCH_MGR] ⚠ {len(warnings)} warning(s):")
            for w in warnings[:5]:
                print(f"    ⚠ {w}")
            if len(warnings) > 5:
                print(f"    ... and {len(warnings) - 5} more")
        
        if not corrections and not warnings:
            print("[RESEARCH_MGR] ✓ All validations passed")
        
        return synthesis

    def _fallback_synthesis(self) -> Dict:
        """Fallback when LLM fails"""
        print("[RESEARCH_MGR] ⚠ Using fallback synthesis")
        return {
            'probabilities': {
                'bull_case': 33,
                'bear_case': 33,
                'base_case': 34,
                'rationale': 'Fallback - equal weighting due to synthesis failure'
            },
            'debate_winner': {
                'winner': 'tie',
                'reasoning': 'Unable to determine - synthesis failed'
            },
            'thesis_quality': {
                'bull_quality': 'unknown',
                'bear_quality': 'unknown',
                'better_thesis': 'equal'
            },
            'conclusion': {
                'recommendation': 'HOLD',
                'confidence': 'LOW',
                'position_size': '0%',
                'rationale': 'Synthesis failed - defaulting to HOLD'
            },
            'key_factors': [],
            'risks_to_monitor': ['Synthesis failed - manual review required'],
            'full_synthesis': 'Unable to complete synthesis. Please retry.',
            'is_fallback': True
        }

    def synthesize_decision(self) -> Dict:
        """Main synthesis - LLM decides everything"""
        if not self.client:
            print("[RESEARCH_MGR] ✗ No API client")
            return self._fallback_synthesis()
        
        prompt = self._build_synthesis_prompt()
        
        try:
            print(f"[RESEARCH_MGR] LLM synthesizing decision...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior research manager making investment decisions. Analyze all inputs and return structured JSON. Be decisive but explain your reasoning."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean markdown
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            synthesis = json.loads(response_text)
            print(f"[RESEARCH_MGR] ✓ LLM synthesis complete")
            
            # Apply guardrails
            synthesis = self._validate_synthesis(synthesis)
            
            return synthesis
            
        except json.JSONDecodeError as e:
            print(f"[RESEARCH_MGR] ✗ JSON parse error: {e}")
            return self._fallback_synthesis()
        except Exception as e:
            print(f"[RESEARCH_MGR] ✗ Synthesis error: {e}")
            return self._fallback_synthesis()

    def generate_report(self, synthesis: Dict) -> str:
        """Generate human-readable report"""
        probs = synthesis.get('probabilities', {})
        conclusion = synthesis.get('conclusion', {})
        debate = synthesis.get('debate_winner', {})
        quality = synthesis.get('thesis_quality', {})
        
        # Format key factors
        factors_text = ""
        for f in synthesis.get('key_factors', [])[:5]:
            factors_text += f"  • {f.get('factor', 'N/A')} ({f.get('impact', '?')}, weight: {f.get('weight', '?')})\n"
        
        # Format risks
        risks_text = "\n".join(f"  • {r}" for r in synthesis.get('risks_to_monitor', [])[:5])
        
        guardrails_text = ""
        if synthesis.get('guardrail_corrections') or synthesis.get('guardrail_warnings'):
            guardrails_text = "\n**Validation Issues:**\n"
            for c in synthesis.get('guardrail_corrections', []):
                guardrails_text += f"  ⚠ CORRECTED: {c}\n"
            for w in synthesis.get('guardrail_warnings', [])[:3]:
                guardrails_text += f"  ⚠ WARNING: {w}\n"
        
        report = f"""
# RESEARCH SYNTHESIS: {self.ticker}
{'='*70}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Analysis Date:** {self.analysis_date or 'Current'}
**Validation Score:** {synthesis.get('validation_score', 'N/A')}/100

## PROBABILITY ASSESSMENT
| Scenario | Probability |
|----------|-------------|
| Bull Case | {probs.get('bull_case', 'N/A')}% |
| Bear Case | {probs.get('bear_case', 'N/A')}% |
| Base Case | {probs.get('base_case', 'N/A')}% |

**Rationale:** {probs.get('rationale', 'N/A')}

## DEBATE ANALYSIS
- **Winner:** {debate.get('winner', 'N/A').upper()}
- **Reasoning:** {debate.get('reasoning', 'N/A')}

## THESIS QUALITY
- **Bull Thesis:** {quality.get('bull_quality', 'N/A')}
- **Bear Thesis:** {quality.get('bear_quality', 'N/A')}
- **Better Thesis:** {quality.get('better_thesis', 'N/A')}

## FINAL DECISION
- **Recommendation:** {conclusion.get('recommendation', 'N/A')}
- **Confidence:** {conclusion.get('confidence', 'N/A')}
- **Position Size:** {conclusion.get('position_size', 'N/A')}
- **Stop Loss:** {conclusion.get('stop_loss', 'N/A')}
- **Target:** {conclusion.get('target', 'N/A')}
- **Time Horizon:** {conclusion.get('time_horizon', 'N/A')}

**Rationale:** {conclusion.get('rationale', 'N/A')}

## KEY FACTORS
{factors_text or '  (none provided)'}

## RISKS TO MONITOR
{risks_text or '  (none provided)'}

## FULL SYNTHESIS
{synthesis.get('full_synthesis', 'N/A')}
{guardrails_text}

{'='*70}
RESEARCH CONCLUSION: {conclusion.get('recommendation', 'N/A')} | Confidence: {conclusion.get('confidence', 'N/A')}
{'='*70}
"""
        return report

    def synthesize(self, debate_rounds: int = 0, skip_risk_evals: bool = False,
                   outputs_dir: str = "../../outputs") -> tuple:
        """Main workflow"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"RESEARCH MANAGER: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*70}")
        print(f"Debate Rounds: {debate_rounds if debate_rounds > 0 else 'None'}")
        print(f"{'='*70}\n")
        
        # Run debate if requested
        if debate_rounds > 0:
            self.run_debate(rounds=debate_rounds)
        
        # Load risk evaluations
        if not skip_risk_evals:
            self.load_risk_evaluations(outputs_dir)
        
        # LLM synthesizes everything
        synthesis = self.synthesize_decision()
        
        # Generate report
        report = self.generate_report(synthesis)
        
        # Build output data
        synthesis_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None,
            'debate_rounds': debate_rounds,
            'debate_conducted': debate_rounds > 0,
            
            # LLM's synthesis
            'probabilities': synthesis.get('probabilities', {}),
            'debate_winner': synthesis.get('debate_winner', {}),
            'thesis_quality': synthesis.get('thesis_quality', {}),
            'conclusion': synthesis.get('conclusion', {}),
            'key_factors': synthesis.get('key_factors', []),
            'risks_to_monitor': synthesis.get('risks_to_monitor', []),
            'full_synthesis': synthesis.get('full_synthesis', ''),
            
            # Include input theses for downstream
            'bull_thesis': self.research_inputs.get('bull_thesis', {}),
            'bear_thesis': self.research_inputs.get('bear_thesis', {}),
            'debate_history': self.research_inputs.get('debate_history', []),
            
            # Validation
            'guardrail_corrections': synthesis.get('guardrail_corrections', []),
            'guardrail_warnings': synthesis.get('guardrail_warnings', []),
            'validation_score': synthesis.get('validation_score', 0),
            'is_fallback': synthesis.get('is_fallback', False)
        }
        
        elapsed = time.time() - start_time
        print(f"\n[RESEARCH_MGR] ✓ Complete in {elapsed:.2f}s")
        print(f"[RESEARCH_MGR] Recommendation: {synthesis.get('conclusion', {}).get('recommendation', 'N/A')}")
        print(f"[RESEARCH_MGR] Confidence: {synthesis.get('conclusion', {}).get('confidence', 'N/A')}")
        
        return report, synthesis_data


def main():
    parser = argparse.ArgumentParser(description="Research Manager - LLM-Driven Synthesis")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--bull-file", default="../../outputs/bull_thesis.json")
    parser.add_argument("--bear-file", default="../../outputs/bear_thesis.json")
    parser.add_argument("--debate-rounds", type=int, default=0)
    parser.add_argument("--skip-evaluations", action="store_true")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", help="Save report to file")
    parser.add_argument("--save-synthesis", help="Save synthesis JSON")
    parser.add_argument("--analysis-date", type=str, default=None)
    
    args = parser.parse_args()
    
    try:
        manager = ResearchManager(
            ticker=args.ticker,
            api_key=args.api_key,
            model=args.model,
            analysis_date=args.analysis_date
        )
        
        manager.load_research_files(args.bull_file, args.bear_file)
        
        if not manager.research_inputs['bull_thesis'] or not manager.research_inputs['bear_thesis']:
            print("\n✗ Error: Both bull and bear thesis files required")
            sys.exit(1)
        
        report, synthesis_data = manager.synthesize(
            debate_rounds=args.debate_rounds,
            skip_risk_evals=args.skip_evaluations
        )
        
        print(report)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to {args.output}")
        
        if args.save_synthesis:
            with open(args.save_synthesis, 'w', encoding='utf-8') as f:
                json.dump(synthesis_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Synthesis saved to {args.save_synthesis}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()