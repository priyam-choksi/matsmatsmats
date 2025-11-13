"""
Conservative Risk Debator - Token-Efficient Version
Risk-averse evaluation with red flag detection

Usage: python conservative_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
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


class ConservativeDebator:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.risk_profile = "CONSERVATIVE"
        
        self.system_prompt = """You are the Conservative Risk Analyst - capital preservation champion.

**YOUR PHILOSOPHY:**
"Return OF capital before return ON capital" - Protecting capital is paramount.

**DECISION CRITERIA:**
- SMALL BUY (3-5%): R/R >4:1, downside <10%, LOW risk only
- MINIMAL BUY (1-2%): R/R >3:1, downside <15%
- HOLD (0-1%): Insufficient safety margin
- AVOID (0%): 2+ red flags, HIGH risk, downside >20%

**OUTPUT:**
Identify red flags, explain why caution is warranted, provide safety-first approach.

CONSERVATIVE STANCE: [SMALL BUY/MINIMAL BUY/HOLD/AVOID] - Position Size: X% - Confidence: [High/Medium/Low]"""
        
        self.risk_parameters = {
            'max_position_size': 0.05,
            'min_reward_ratio': 4.0,
            'max_drawdown_tolerance': 0.03
        }
        
        self.evaluation = {}
    
    def load_all_data(
        self,
        synthesis_file: Optional[str] = None,
        bull_file: Optional[str] = None,
        bear_file: Optional[str] = None
    ) -> Dict:
        """Smart data loading"""
        print(f"[CONSERVATIVE] Loading data...")
        
        if synthesis_file and os.path.exists(synthesis_file):
            try:
                with open(synthesis_file, 'r', encoding='utf-8') as f:
                    synthesis = json.load(f)
                print(f"[CONSERVATIVE] ✓ Synthesis loaded")
                
                if 'bull_thesis' in synthesis and 'bear_thesis' in synthesis:
                    return synthesis
                else:
                    if not bull_file:
                        bull_file = "../../outputs/bull_thesis.json"
                    if not bear_file:
                        bear_file = "../../outputs/bear_thesis.json"
                    
                    if os.path.exists(bull_file):
                        with open(bull_file, 'r', encoding='utf-8') as f:
                            synthesis['bull_thesis'] = json.load(f)
                        print(f"[CONSERVATIVE] ✓ Bull loaded")
                    
                    if os.path.exists(bear_file):
                        with open(bear_file, 'r', encoding='utf-8') as f:
                            synthesis['bear_thesis'] = json.load(f)
                        print(f"[CONSERVATIVE] ✓ Bear loaded")
                    
                    return synthesis
            except Exception as e:
                print(f"[CONSERVATIVE] ⚠️  Error: {e}")
        
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
    
    def identify_red_flags(self, synthesis: Dict) -> List[str]:
        """Identify warning signs"""
        print(f"[CONSERVATIVE] Scanning for red flags...")
        
        red_flags = []
        
        bear_thesis = synthesis.get('bear_thesis', {})
        bear_ra = bear_thesis.get('risk_assessment', {})
        
        if bear_ra.get('risk_level') == 'HIGH':
            red_flags.append("HIGH risk level")
        
        if bear_ra.get('conviction_level') == 'HIGH':
            red_flags.append("Bear has HIGH conviction")
        
        if bear_ra.get('risk_score', 0) > 70:
            red_flags.append(f"Risk score {bear_ra.get('risk_score', 0):.0f}/100")
        
        for trigger in bear_thesis.get('downside_triggers', []):
            if trigger.get('impact') == 'HIGH' and trigger.get('timeline') == 'Imminent':
                red_flags.append(f"Imminent high-impact risk")
                break
        
        print(f"[CONSERVATIVE] ✓ Found {len(red_flags)} red flags")
        
        return red_flags
    
    def evaluate_opportunity(self, synthesis: Dict, red_flags: List[str]) -> Dict[str, Any]:
        """Evaluate conservatively"""
        print(f"[CONSERVATIVE] Evaluating...")
        
        bull_rr = synthesis.get('bull_thesis', {}).get('risk_reward', {})
        bear_ra = synthesis.get('bear_thesis', {}).get('risk_assessment', {})
        
        rr_ratio = bull_rr.get('reward_risk_ratio', 1.0)
        risk_level = bear_ra.get('risk_level', 'MEDIUM')
        
        import re
        downside = bear_ra.get('downside_risk', '15%')
        downside_nums = re.findall(r'\d+', str(downside))
        downside_pct = max(int(n) for n in downside_nums) if downside_nums else 20
        
        evaluation = {
            'profile': self.risk_profile,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'red_flags': red_flags
        }
        
        if len(red_flags) >= 2:
            evaluation['stance'] = 'AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = f"{len(red_flags)} red flags - too risky"
        elif risk_level == 'HIGH':
            evaluation['stance'] = 'AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = "HIGH risk exceeds tolerance"
        elif rr_ratio >= 4.0 and downside_pct <= 10 and risk_level == 'LOW':
            evaluation['stance'] = 'SMALL BUY'
            evaluation['position_size'] = 0.05
            evaluation['reasoning'] = f"Exceptional R/R ({rr_ratio:.1f}:1), limited downside"
        elif rr_ratio >= 3.0 and downside_pct <= 15 and risk_level == 'LOW':
            evaluation['stance'] = 'MINIMAL BUY'
            evaluation['position_size'] = 0.02
            evaluation['reasoning'] = f"Good R/R ({rr_ratio:.1f}:1), acceptable risk"
        elif downside_pct >= 20:
            evaluation['stance'] = 'AVOID'
            evaluation['position_size'] = 0.0
            evaluation['reasoning'] = f"Downside {downside_pct}% exceeds 20% limit"
        else:
            evaluation['stance'] = 'HOLD'
            evaluation['position_size'] = 0.01
            evaluation['reasoning'] = "Insufficient safety margin"
        
        evaluation['confidence'] = 'HIGH' if evaluation['stance'] == 'AVOID' and len(red_flags) >= 2 else 'MEDIUM' if evaluation['stance'] in ['SMALL BUY', 'MINIMAL BUY'] else 'LOW'
        
        print(f"[CONSERVATIVE] ✓ {evaluation['stance']}, Position: {evaluation['position_size']*100:.1f}%")
        
        return evaluation
    
    def generate_trading_plan(self, evaluation: Dict) -> Dict:
        """Generate plan"""
        if evaluation['stance'] in ['SMALL BUY', 'MINIMAL BUY']:
            return {
                'entry_strategy': "Wait for pullback, small tranches",
                'stop_loss': "-3%",
                'profit_targets': ["+8%", "+12%", "+15%"],
                'exit_triggers': ["Any support break", "New red flags"],
                'safety_rules': ["Never add to losers", "Exit on warnings"]
            }
        elif evaluation['stance'] == 'AVOID':
            return {'entry_strategy': "No entry", 'stop_loss': "N/A", 'profit_targets': ["N/A"]}
        else:
            return {'entry_strategy': "Wait for 4:1 setup", 'stop_loss': "-2%", 'profit_targets': ["+8%"]}
    
    def synthesize_with_llm(self, evaluation: Dict, trading_plan: Dict, synthesis: Dict, red_flags: List[str]) -> str:
        """Generate report - token efficient"""
        
        if not self.client:
            return self._create_fallback_report(evaluation, trading_plan, red_flags)
        
        try:
            context = f"""Conservative evaluation for {self.ticker}:

Red Flags: {len(red_flags)} detected
{chr(10).join(f'- {flag}' for flag in red_flags)}

Assessment: {json.dumps(evaluation, indent=2)}
Plan: {json.dumps(trading_plan, indent=2)}

Explain safety-first approach."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            report = response.choices[0].message.content
            
            if "CONSERVATIVE STANCE:" not in report:
                report += f"\n\nCONSERVATIVE STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.1f}% - Confidence: {evaluation['confidence']}"
            
            return report
            
        except Exception as e:
            return self._create_fallback_report(evaluation, trading_plan, red_flags)
    
    def _create_fallback_report(self, evaluation: Dict, trading_plan: Dict, red_flags: List[str]) -> str:
        """Fallback"""
        report = f"""
# CONSERVATIVE RISK EVALUATION: {self.ticker}
{'='*70}

**Red Flags:** {len(red_flags)}
"""
        for flag in red_flags:
            report += f"  ⚠️ {flag}\n"
        
        report += f"""
**Stance:** {evaluation['stance']}
**Position:** {evaluation['position_size']*100:.1f}%

CONSERVATIVE STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.1f}% - Confidence: {evaluation['confidence']}
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
        print(f"CONSERVATIVE RISK EVALUATION: {self.ticker}")
        print(f"{'='*70}\n")
        
        synthesis = self.load_all_data(synthesis_file, bull_file, bear_file)
        
        if not synthesis or not synthesis.get('bull_thesis') or not synthesis.get('bear_thesis'):
            print("[CONSERVATIVE] ❌ Missing data")
            return "Error: No data", {}
        
        red_flags = self.identify_red_flags(synthesis)
        evaluation = self.evaluate_opportunity(synthesis, red_flags)
        trading_plan = self.generate_trading_plan(evaluation)
        
        report = self.synthesize_with_llm(evaluation, trading_plan, synthesis, red_flags)
        
        self.evaluation = {
            **evaluation,
            'trading_plan': trading_plan,
            'risk_parameters': self.risk_parameters
        }
        
        elapsed = time.time() - start_time
        print(f"\n[CONSERVATIVE] ✓ Complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, self.evaluation
    
    def save_evaluation(self, filepath: str):
        """Save"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation, f, indent=2)
            print(f"[CONSERVATIVE] ✓ Saved to {filepath}")
        except Exception as e:
            print(f"[CONSERVATIVE] ⚠️  Save error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Conservative Risk Debator")
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
        debator = ConservativeDebator(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
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