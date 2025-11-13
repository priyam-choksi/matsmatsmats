"""
Aggressive Risk Debator - Token-Efficient Version
High risk tolerance evaluation with smart data loading

Usage: python aggressive_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
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


class AggressiveDebator:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.risk_profile = "AGGRESSIVE"
        
        self.system_prompt = """You are the Aggressive Risk Analyst - a high-reward champion who seeks bold opportunities.

**YOUR PERSONALITY:**
"Fortune favors the bold" - You believe missing upside is worse than temporary drawdowns.

**DECISION CRITERIA:**
- STRONG BUY (20-25%): R/R >3:1, upside >30%
- BUY (10-15%): R/R >2:1, upside >20%
- HOLD/SMALL (5%): R/R ~1.5:1, need more confirmation
- AVOID (0%): R/R <1:1 or downside >25%

**OUTPUT:** 
Provide aggressive analysis with specific position sizing, entry strategy, and profit targets.

AGGRESSIVE STANCE: [STRONG BUY/BUY/HOLD/AVOID] - Position Size: X% - Confidence: [High/Medium/Low]"""
        
        self.risk_parameters = {
            'max_position_size': 0.25,
            'min_position_size': 0.05,
            'max_drawdown_tolerance': 0.15,
            'min_reward_ratio': 2.0
        }
        
        self.evaluation = {}
    
    def load_all_data(
        self,
        synthesis_file: Optional[str] = None,
        bull_file: Optional[str] = None,
        bear_file: Optional[str] = None
    ) -> Dict:
        """
        Smart data loading - token efficient
        Tries synthesis first, falls back to separate files
        """
        print(f"[AGGRESSIVE] Loading data...")
        
        # Try loading synthesis
        if synthesis_file and os.path.exists(synthesis_file):
            try:
                with open(synthesis_file, 'r', encoding='utf-8') as f:
                    synthesis = json.load(f)
                print(f"[AGGRESSIVE] ✓ Synthesis loaded")
                
                # Check if bull/bear are embedded
                if 'bull_thesis' in synthesis and 'bear_thesis' in synthesis:
                    print(f"[AGGRESSIVE] ✓ Bull/bear found in synthesis")
                    return synthesis
                else:
                    # Synthesis exists but doesn't have bull/bear - load separately
                    print(f"[AGGRESSIVE] → Synthesis missing bull/bear, loading separately...")
                    
                    # Load bull
                    if not bull_file:
                        bull_file = "../../outputs/bull_thesis.json"
                    if not bear_file:
                        bear_file = "../../outputs/bear_thesis.json"
                    
                    if os.path.exists(bull_file):
                        with open(bull_file, 'r', encoding='utf-8') as f:
                            synthesis['bull_thesis'] = json.load(f)
                        print(f"[AGGRESSIVE] ✓ Bull thesis loaded")
                    
                    if os.path.exists(bear_file):
                        with open(bear_file, 'r', encoding='utf-8') as f:
                            synthesis['bear_thesis'] = json.load(f)
                        print(f"[AGGRESSIVE] ✓ Bear thesis loaded")
                    
                    return synthesis
                    
            except Exception as e:
                print(f"[AGGRESSIVE] ⚠️  Synthesis load error: {e}")
        
        # Fallback: Load bull/bear directly
        print(f"[AGGRESSIVE] → Loading bull/bear directly (no synthesis)...")
        
        if not bull_file:
            bull_file = "../../outputs/bull_thesis.json"
        if not bear_file:
            bear_file = "../../outputs/bear_thesis.json"
        
        bull_thesis = {}
        bear_thesis = {}
        
        if os.path.exists(bull_file):
            with open(bull_file, 'r', encoding='utf-8') as f:
                bull_thesis = json.load(f)
            print(f"[AGGRESSIVE] ✓ Bull loaded")
        
        if os.path.exists(bear_file):
            with open(bear_file, 'r', encoding='utf-8') as f:
                bear_thesis = json.load(f)
            print(f"[AGGRESSIVE] ✓ Bear loaded")
        
        return {
            'ticker': self.ticker,
            'bull_thesis': bull_thesis,
            'bear_thesis': bear_thesis,
            'source': 'direct_load'
        }
    
    def evaluate_opportunity(self, synthesis: Dict) -> Dict[str, Any]:
        """Evaluate from aggressive perspective"""
        print(f"[AGGRESSIVE] Evaluating opportunity...")
        
        bull_thesis = synthesis.get('bull_thesis', {})
        bear_thesis = synthesis.get('bear_thesis', {})
        
        # Get metrics
        bull_rr = bull_thesis.get('risk_reward', {})
        bear_ra = bear_thesis.get('risk_assessment', {})
        
        rr_ratio = bull_rr.get('reward_risk_ratio', 1.0)
        upside = bull_rr.get('upside_potential', '20%')
        downside = bear_ra.get('downside_risk', '15%')
        
        # Parse percentages
        import re
        upside_nums = re.findall(r'\d+', str(upside))
        downside_nums = re.findall(r'\d+', str(downside))
        
        upside_pct = sum(int(n) for n in upside_nums) / len(upside_nums) if upside_nums else 20
        downside_pct = sum(int(n) for n in downside_nums) / len(downside_nums) if downside_nums else 15
        
        evaluation = {
            'profile': self.risk_profile,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat()
        }
        
        # Aggressive logic
        if rr_ratio >= 3.0 and upside_pct >= 30:
            evaluation['stance'] = 'STRONG BUY'
            evaluation['position_size'] = 0.25
            evaluation['reasoning'] = f"Exceptional R/R ({rr_ratio:.1f}:1) with {upside_pct:.0f}% upside"
        elif rr_ratio >= 2.0 and upside_pct >= 20:
            evaluation['stance'] = 'BUY'
            evaluation['position_size'] = 0.15
            evaluation['reasoning'] = f"Strong R/R ({rr_ratio:.1f}:1) with {upside_pct:.0f}% upside"
        elif rr_ratio >= 1.5 and upside_pct >= 15:
            evaluation['stance'] = 'BUY'
            evaluation['position_size'] = 0.10
            evaluation['reasoning'] = f"Decent R/R ({rr_ratio:.1f}:1) worth aggressive bet"
        elif downside_pct >= 30:
            evaluation['stance'] = 'AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = f"Even aggressive traders avoid {downside_pct:.0f}% downside"
        else:
            evaluation['stance'] = 'HOLD/SMALL'
            evaluation['position_size'] = 0.05
            evaluation['reasoning'] = f"Insufficient conviction"
        
        # Catalyst boost
        for catalyst in bull_thesis.get('catalysts', []):
            if 'imminent' in str(catalyst.get('timeline', '')).lower():
                evaluation['position_size'] = min(evaluation['position_size'] * 1.3, 0.25)
                evaluation['reasoning'] += " (BOOSTED for imminent catalyst)"
                break
        
        # Confidence
        if evaluation['stance'] == 'STRONG BUY':
            evaluation['confidence'] = 'HIGH'
        elif evaluation['stance'] == 'BUY' and evaluation['position_size'] >= 0.15:
            evaluation['confidence'] = 'HIGH'
        elif evaluation['stance'] == 'BUY':
            evaluation['confidence'] = 'MEDIUM'
        else:
            evaluation['confidence'] = 'LOW'
        
        print(f"[AGGRESSIVE] ✓ {evaluation['stance']}, Position: {evaluation['position_size']*100:.0f}%")
        
        return evaluation
    
    def generate_trading_plan(self, evaluation: Dict, synthesis: Dict) -> Dict[str, Any]:
        """Generate trading plan"""
        plan = {
            'entry_strategy': '',
            'position_building': '',
            'profit_targets': [],
            'stop_loss': '',
            'time_frame': ''
        }
        
        if evaluation['stance'] in ['STRONG BUY', 'BUY']:
            plan['entry_strategy'] = "Aggressive entry on weakness or at market"
            plan['position_building'] = f"{evaluation['position_size']*50:.0f}% initial, double on confirmation"
            plan['stop_loss'] = "Wide stop -12 to -15%"
            plan['profit_targets'] = ["+25%", "+40%", "+50%+"]
            plan['time_frame'] = "3-6 months"
        elif evaluation['stance'] == 'AVOID':
            plan['entry_strategy'] = "No entry"
            plan['position_building'] = "Zero"
            plan['stop_loss'] = "N/A"
            plan['profit_targets'] = ["N/A"]
            plan['time_frame'] = "No position"
        else:
            plan['entry_strategy'] = "Small pilot"
            plan['position_building'] = "5% max"
            plan['stop_loss'] = "-8%"
            plan['profit_targets'] = ["+15%"]
            plan['time_frame'] = "1-2 months"
        
        return plan
    
    def synthesize_with_llm(self, evaluation: Dict, trading_plan: Dict, synthesis: Dict) -> str:
        """Generate report - ONLY sends relevant excerpts to LLM"""
        
        if not self.client:
            return self._create_fallback_report(evaluation, trading_plan)
        
        try:
            print(f"[AGGRESSIVE] Generating report...")
            
            # TOKEN OPTIMIZATION: Only send key excerpts, not full theses!
            bull_summary = synthesis.get('bull_thesis', {}).get('core_thesis', 'Not available')[:500]
            bear_summary = synthesis.get('bear_thesis', {}).get('core_thesis', 'Not available')[:500]
            research_rec = synthesis.get('conclusion', {}).get('recommendation', 'N/A')
            
            # Small, focused context
            context = f"""Aggressive evaluation for {self.ticker}:

Research Recommendation: {research_rec}
Bull Case: {bull_summary}
Bear Case: {bear_summary}

Your Assessment: {json.dumps(evaluation, indent=2)}
Trading Plan: {json.dumps(trading_plan, indent=2)}

Explain why aggressive positioning is warranted (or not). Be specific."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.75,
                max_tokens=1500  # Reduced - don't need massive report
            )
            
            report = response.choices[0].message.content
            
            if "AGGRESSIVE STANCE:" not in report:
                report += f"\n\nAGGRESSIVE STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.0f}% - Confidence: {evaluation['confidence']}"
            
            print(f"[AGGRESSIVE] ✓ Report complete")
            
            return report
            
        except Exception as e:
            print(f"[AGGRESSIVE] ❌ LLM error: {e}")
            return self._create_fallback_report(evaluation, trading_plan)
    
    def _create_fallback_report(self, evaluation: Dict, trading_plan: Dict) -> str:
        """Fallback report"""
        report = f"""
# AGGRESSIVE RISK EVALUATION: {self.ticker}
{'='*70}

**Stance:** {evaluation['stance']}
**Position:** {evaluation['position_size']*100:.0f}%
**Reasoning:** {evaluation['reasoning']}

**Entry:** {trading_plan['entry_strategy']}
**Stop:** {trading_plan['stop_loss']}
**Targets:** {', '.join(trading_plan['profit_targets'])}

AGGRESSIVE STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.0f}% - Confidence: {evaluation['confidence']}
"""
        return report
    
    def evaluate(
        self,
        synthesis_file: Optional[str] = None,
        bull_file: Optional[str] = None,
        bear_file: Optional[str] = None
    ) -> tuple:
        """Main evaluation"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"AGGRESSIVE RISK EVALUATION: {self.ticker}")
        print(f"{'='*70}\n")
        
        # Smart loading - token efficient!
        synthesis = self.load_all_data(synthesis_file, bull_file, bear_file)
        
        # Validate we have minimum data
        if not synthesis:
            print("[AGGRESSIVE] ❌ No data loaded")
            return "Error: No data", {}
        
        bull_thesis = synthesis.get('bull_thesis', {})
        bear_thesis = synthesis.get('bear_thesis', {})
        
        if not bull_thesis or not bear_thesis:
            print("[AGGRESSIVE] ❌ Missing bull or bear thesis")
            return "Error: Incomplete data", {}
        
        # Evaluate
        evaluation = self.evaluate_opportunity(synthesis)
        trading_plan = self.generate_trading_plan(evaluation, synthesis)
        
        # Generate report (token-efficient!)
        report = self.synthesize_with_llm(evaluation, trading_plan, synthesis)
        
        # Store
        self.evaluation = {
            **evaluation,
            'trading_plan': trading_plan,
            'risk_parameters': self.risk_parameters
        }
        
        elapsed = time.time() - start_time
        print(f"\n[AGGRESSIVE] ✓ Complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, self.evaluation
    
    def save_evaluation(self, filepath: str):
        """Save evaluation"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation, f, indent=2)
            print(f"[AGGRESSIVE] ✓ Saved to {filepath}")
        except Exception as e:
            print(f"[AGGRESSIVE] ⚠️  Save error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Aggressive Risk Debator")
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("--synthesis-file", default="../../outputs/research_synthesis.json")
    parser.add_argument("--bull-file", default="../../outputs/bull_thesis.json")
    parser.add_argument("--bear-file", default="../../outputs/bear_thesis.json")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-evaluation", help="Save JSON")
    
    args = parser.parse_args()
    
    try:
        debator = AggressiveDebator(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
        report, evaluation = debator.evaluate(
            synthesis_file=args.synthesis_file,
            bull_file=args.bull_file,
            bear_file=args.bear_file
        )
        
        print(report)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Saved to {args.output}")
        
        if args.save_evaluation:
            debator.save_evaluation(args.save_evaluation)
        
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