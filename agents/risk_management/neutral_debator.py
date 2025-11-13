"""
Neutral Risk Debator - Token-Efficient Version
Balanced risk evaluation with expected value focus

Usage: python neutral_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
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


class NeutralDebator:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.risk_profile = "NEUTRAL"
        
        self.system_prompt = """You are the Neutral Risk Analyst - the balanced voice of reason.

**YOUR PHILOSOPHY:**
"Let the data decide" - Expected value (probability × outcome) is what matters.

**DECISION CRITERIA:**
- BUY (8-10%): EV >10%, R/R >3:1
- SMALL BUY (4-6%): EV >5%, R/R >2:1
- HOLD (2-3%): EV neutral
- SELL (0%): EV <-5%

**OUTPUT:**
Show expected value calculation, balanced assessment, probability-weighted sizing.

NEUTRAL STANCE: [BUY/SMALL BUY/HOLD/SELL] - Position Size: X% - Confidence: [High/Medium/Low]"""
        
        self.risk_parameters = {
            'max_position_size': 0.10,
            'min_reward_ratio': 3.0,
            'conviction_threshold': 0.65
        }
        
        self.evaluation = {}
    
    def load_all_data(
        self,
        synthesis_file: Optional[str] = None,
        bull_file: Optional[str] = None,
        bear_file: Optional[str] = None
    ) -> Dict:
        """Smart data loading"""
        print(f"[NEUTRAL] Loading data...")
        
        if synthesis_file and os.path.exists(synthesis_file):
            try:
                with open(synthesis_file, 'r', encoding='utf-8') as f:
                    synthesis = json.load(f)
                print(f"[NEUTRAL] ✓ Synthesis loaded")
                
                if 'bull_thesis' in synthesis and 'bear_thesis' in synthesis:
                    return synthesis
                else:
                    print(f"[NEUTRAL] → Loading bull/bear separately...")
                    
                    if not bull_file:
                        bull_file = "../../outputs/bull_thesis.json"
                    if not bear_file:
                        bear_file = "../../outputs/bear_thesis.json"
                    
                    if os.path.exists(bull_file):
                        with open(bull_file, 'r', encoding='utf-8') as f:
                            synthesis['bull_thesis'] = json.load(f)
                        print(f"[NEUTRAL] ✓ Bull loaded")
                    
                    if os.path.exists(bear_file):
                        with open(bear_file, 'r', encoding='utf-8') as f:
                            synthesis['bear_thesis'] = json.load(f)
                        print(f"[NEUTRAL] ✓ Bear loaded")
                    
                    return synthesis
            except Exception as e:
                print(f"[NEUTRAL] ⚠️  Error: {e}")
        
        # Fallback
        if not bull_file:
            bull_file = "../../outputs/bull_thesis.json"
        if not bear_file:
            bear_file = "../../outputs/bear_thesis.json"
        
        bull_thesis = {}
        bear_thesis = {}
        
        if os.path.exists(bull_file):
            with open(bull_file, 'r', encoding='utf-8') as f:
                bull_thesis = json.load(f)
        
        if os.path.exists(bear_file):
            with open(bear_file, 'r', encoding='utf-8') as f:
                bear_thesis = json.load(f)
        
        return {'ticker': self.ticker, 'bull_thesis': bull_thesis, 'bear_thesis': bear_thesis}
    
    def calculate_expected_value(self, synthesis: Dict) -> Dict[str, float]:
        """Calculate EV"""
        print(f"[NEUTRAL] Calculating expected value...")
        
        probs = synthesis.get('probabilities', {})
        bull_prob = probs.get('bull_case', 50) / 100
        bear_prob = probs.get('bear_case', 30) / 100
        
        bull_thesis = synthesis.get('bull_thesis', {})
        bear_thesis = synthesis.get('bear_thesis', {})
        
        upside = bull_thesis.get('risk_reward', {}).get('upside_potential', '20%')
        downside = bear_thesis.get('risk_assessment', {}).get('downside_risk', '15%')
        
        import re
        upside_nums = re.findall(r'\d+', str(upside))
        downside_nums = re.findall(r'\d+', str(downside))
        
        upside_pct = sum(int(n) for n in upside_nums) / len(upside_nums) if upside_nums else 20
        downside_pct = sum(int(n) for n in downside_nums) / len(downside_nums) if downside_nums else 15
        
        expected_value = (upside_pct * bull_prob) - (downside_pct * bear_prob)
        
        ev_calc = {
            'bull_probability': bull_prob,
            'bear_probability': bear_prob,
            'upside_pct': upside_pct,
            'downside_pct': downside_pct,
            'expected_value': expected_value,
            'calculation': f"({upside_pct:.0f}% × {bull_prob:.0%}) - ({downside_pct:.0f}% × {bear_prob:.0%}) = {expected_value:+.1f}%"
        }
        
        print(f"[NEUTRAL] ✓ EV: {expected_value:+.1f}%")
        
        return ev_calc
    
    def evaluate_opportunity(self, synthesis: Dict, ev_calc: Dict) -> Dict[str, Any]:
        """Evaluate based on expected value"""
        print(f"[NEUTRAL] Evaluating...")
        
        expected_value = ev_calc['expected_value']
        rr_ratio = synthesis.get('bull_thesis', {}).get('risk_reward', {}).get('reward_risk_ratio', 1.0)
        
        evaluation = {
            'profile': self.risk_profile,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'expected_value': expected_value
        }
        
        if expected_value > 10 and rr_ratio >= 3.0:
            evaluation['stance'] = 'BUY'
            evaluation['position_size'] = 0.10
            evaluation['reasoning'] = f"Positive EV {expected_value:+.1f}% with R/R {rr_ratio:.1f}:1"
        elif expected_value > 5 and rr_ratio >= 2.5:
            evaluation['stance'] = 'SMALL BUY'
            evaluation['position_size'] = 0.06
            evaluation['reasoning'] = f"Moderate EV {expected_value:+.1f}%"
        elif expected_value > 2:
            evaluation['stance'] = 'SMALL BUY'
            evaluation['position_size'] = 0.04
            evaluation['reasoning'] = f"Modest EV {expected_value:+.1f}%"
        elif expected_value < -5:
            evaluation['stance'] = 'SELL/AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = f"Negative EV {expected_value:+.1f}%"
        else:
            evaluation['stance'] = 'HOLD'
            evaluation['position_size'] = 0.03
            evaluation['reasoning'] = f"Neutral EV {expected_value:+.1f}%"
        
        evaluation['confidence'] = 'MEDIUM' if abs(expected_value) > 8 else 'LOW'
        
        print(f"[NEUTRAL] ✓ {evaluation['stance']}, Position: {evaluation['position_size']*100:.0f}%")
        
        return evaluation
    
    def generate_trading_plan(self, evaluation: Dict) -> Dict:
        """Generate plan"""
        if evaluation['stance'] in ['BUY', 'SMALL BUY']:
            return {
                'entry_strategy': "Scale in 2-3 entries",
                'stop_loss': "-7%",
                'profit_targets': ["+15%", "+25%", "+35%"],
                'time_frame': "3-6 months"
            }
        elif evaluation['stance'] == 'SELL/AVOID':
            return {'entry_strategy': "Exit", 'stop_loss': "N/A", 'profit_targets': ["N/A"], 'time_frame': "Exit"}
        else:
            return {'entry_strategy': "Wait", 'stop_loss': "-5%", 'profit_targets': ["+10%"], 'time_frame': "Watch"}
    
    def synthesize_with_llm(self, evaluation: Dict, trading_plan: Dict, ev_calc: Dict, synthesis: Dict) -> str:
        """Generate report - token efficient"""
        
        if not self.client:
            return self._create_fallback_report(evaluation, trading_plan, ev_calc)
        
        try:
            print(f"[NEUTRAL] Generating report...")
            
            # Minimal context
            context = f"""Neutral evaluation for {self.ticker}:

Expected Value: {ev_calc['calculation']}
Result: {evaluation['expected_value']:+.1f}%

Assessment: {json.dumps(evaluation, indent=2)}
Plan: {json.dumps(trading_plan, indent=2)}

Explain your balanced, probability-weighted approach."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.6,
                max_tokens=1500
            )
            
            report = response.choices[0].message.content
            
            if "NEUTRAL STANCE:" not in report:
                report += f"\n\nNEUTRAL STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.0f}% - Confidence: {evaluation['confidence']}"
            
            return report
            
        except Exception as e:
            print(f"[NEUTRAL] ❌ Error: {e}")
            return self._create_fallback_report(evaluation, trading_plan, ev_calc)
    
    def _create_fallback_report(self, evaluation: Dict, trading_plan: Dict, ev_calc: Dict) -> str:
        """Fallback"""
        return f"""
# NEUTRAL RISK EVALUATION: {self.ticker}
{'='*70}

**EV Calculation:** {ev_calc['calculation']}
**Result:** {evaluation['expected_value']:+.1f}%

**Stance:** {evaluation['stance']}
**Position:** {evaluation['position_size']*100:.0f}%

NEUTRAL STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.0f}% - Confidence: {evaluation['confidence']}
"""
    
    def evaluate(
        self,
        synthesis_file: Optional[str] = None,
        bull_file: Optional[str] = None,
        bear_file: Optional[str] = None
    ) -> tuple:
        """Main evaluation"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"NEUTRAL RISK EVALUATION: {self.ticker}")
        print(f"{'='*70}\n")
        
        synthesis = self.load_all_data(synthesis_file, bull_file, bear_file)
        
        if not synthesis or not synthesis.get('bull_thesis') or not synthesis.get('bear_thesis'):
            print("[NEUTRAL] ❌ Missing data")
            return "Error: No data", {}
        
        ev_calc = self.calculate_expected_value(synthesis)
        evaluation = self.evaluate_opportunity(synthesis, ev_calc)
        trading_plan = self.generate_trading_plan(evaluation)
        
        report = self.synthesize_with_llm(evaluation, trading_plan, ev_calc, synthesis)
        
        self.evaluation = {
            **evaluation,
            'trading_plan': trading_plan,
            'expected_value_calc': ev_calc,
            'risk_parameters': self.risk_parameters
        }
        
        elapsed = time.time() - start_time
        print(f"\n[NEUTRAL] ✓ Complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, self.evaluation
    
    def save_evaluation(self, filepath: str):
        """Save"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation, f, indent=2)
            print(f"[NEUTRAL] ✓ Saved to {filepath}")
        except Exception as e:
            print(f"[NEUTRAL] ⚠️  Save error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Neutral Risk Debator")
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
        debator = NeutralDebator(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
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