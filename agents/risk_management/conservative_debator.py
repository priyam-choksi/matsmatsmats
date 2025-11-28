"""
Conservative Risk Debator - Token-Efficient Version
Risk-averse evaluation with red flag detection

MODIFIED: Now supports historical backtesting via analysis_date parameter

Usage: python conservative_debator.py AAPL --synthesis-file ../../outputs/research_synthesis.json
       python conservative_debator.py AAPL --synthesis-file ... --analysis-date 2024-06-15
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
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Historical backtesting support
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[CONSERVATIVE] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[CONSERVATIVE] Running in LIVE mode (current date)")
        
        self.risk_profile = "CONSERVATIVE"
        
        self.system_prompt = """You are a conservative financial analyst conducting disciplined risk assessments.
Your task is to evaluate investment opportunities using a conservative focused risk management framework that emphasizes capital preservation while identifying asymmetric opportunities.

**REQUIRED INPUTS:**
Before analysis, ensure you have:
- Asset name and ticker symbol
- Current price and relevant financial metrics
- Identified price targets (upside scenario)
- Key support levels (downside scenario)
- Analysis timeframe (e.g., 3-month, 12-month outlook)
- Relevant fundamental or technical catalysts

**EVALUATION FRAMEWORK:**

Calculate the risk-reward ratio (R/R) as: (Target Price - Entry Price) / (Entry Price - Stop Loss)

Apply the following allocation guidelines based on opportunity assessment:

1. **SMALL POSITION (3-5% allocation)**
   - R/R ratio exceeds 4:1
   - Maximum downside risk below 10%
   - Low volatility profile with strong fundamental support
   - Multiple protective factors identified (liquidity, diversification, technical support)

2. **MINIMAL POSITION (1-2% allocation)**
   - R/R ratio exceeds 3:1
   - Maximum downside risk below 15%
   - Moderate risk with identifiable catalysts
   - At least one strong protective factor present

3. **HOLD/WATCH (0-1% allocation)**
   - R/R ratio below 3:1
   - Maximum downside risk exceeds 15%
   - Insufficient margin of safety for commitment
   - Requires additional confirmation or risk reduction

4. **AVOID (0% allocation)**
   - R/R ratio below 2:1
   - Maximum downside risk exceeds 20%
   - Two or more critical red flags present
   - High volatility or fundamental deterioration

**RED FLAGS CHECKLIST:**
Document if any of the following are present:
- Deteriorating earnings or revenue trends
- High financial leverage or liquidity concerns
- Regulatory, legal, or governance uncertainties
- Technical breakdown below critical support levels
- Unfavorable macroeconomic or sector headwinds
- Valuation disconnect from fundamentals

**OUTPUT STRUCTURE:**

Provide your analysis in the following format:

**RECOMMENDATION:** [SMALL POSITION / MINIMAL POSITION / HOLD / AVOID]  
**Position Size:** X%  
**Confidence Level:** [High / Medium / Low]

**Entry Strategy:**
- Recommended entry price or price range
- Scaling approach (if applicable)

**Price Targets:**
- Conservative target: $X (X% upside)
- Optimistic target: $X (X% upside)

**Risk Management:**
- Stop-loss level: $X (X% downside)
- Risk-reward ratio: X:1
- Maximum portfolio impact at stop-loss: X%

**Risk Assessment:**
[Identify and explain any red flags, protective factors, and key risk considerations that inform the conservative position sizing]

**Rationale:**
[2-3 lines explaining the investment thesis, why the risk-reward profile justifies the allocation, and how this position aligns with capital preservation principles while capturing asymmetric opportunities]
"""
        
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
                
                # Check for historical date in loaded data
                if synthesis.get('analysis_date') and not self.analysis_date:
                    self.analysis_date = synthesis.get('analysis_date')
                    print(f"[CONSERVATIVE] → Using historical date from synthesis: {self.analysis_date}")
                
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
            
            # Check for historical date
            if bull_thesis.get('analysis_date') and not self.analysis_date:
                self.analysis_date = bull_thesis.get('analysis_date')
                print(f"[CONSERVATIVE] → Using historical date from bull thesis: {self.analysis_date}")
        
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
            'red_flags': red_flags,
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None
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
    
    def generate_trading_plan(self, evaluation: Dict, synthesis: Dict = None) -> Dict:
        """
        Generate ultra-conservative trading plan with tight risk controls.
        Conservative evaluator prioritizes capital preservation over returns.
        Targets are scaled down to reflect safety-first approach.
        """
        
        # =====================================================================
        # NEW: Extract stock characteristics, then apply conservative discount
        # =====================================================================
        avg_upside = 20  # Default
        avg_downside = 15  # Default
        
        if synthesis:
            bull_thesis = synthesis.get('bull_thesis', {})
            bear_thesis = synthesis.get('bear_thesis', {})
            
            import re
            upside_str = bull_thesis.get('risk_reward', {}).get('upside_potential', '20%')
            downside_str = bear_thesis.get('risk_assessment', {}).get('downside_risk', '15%')
            
            upside_nums = re.findall(r'\d+', str(upside_str))
            downside_nums = re.findall(r'\d+', str(downside_str))
            
            if upside_nums:
                avg_upside = sum(int(n) for n in upside_nums) / len(upside_nums)
            if downside_nums:
                avg_downside = sum(int(n) for n in downside_nums) / len(downside_nums)
        
        # Conservative approach: Heavily discount upside, strict on downside
        # Apply 50% haircut to expected upside
        conservative_upside = avg_upside * 0.5
        
        if evaluation['stance'] in ['SMALL BUY', 'MINIMAL BUY']:
            # Very conservative targets
            target1 = max(5, int(conservative_upside * 0.5))   # 25% of bull's upside
            target2 = max(8, int(conservative_upside * 0.75))  # 37.5% of bull's upside
            target3 = max(10, int(conservative_upside * 1.0))  # 50% of bull's upside (max)
            
            # Tight stops - never risk more than the expected gain
            stop_pct = min(target1, max(3, int(avg_downside * 0.25)))  # 25% of expected downside
            
            return {
                'entry_strategy': "Wait for 5%+ pullback, scale in small tranches",
                'stop_loss': f"-{stop_pct}%",
                'profit_targets': [f"+{target1}%", f"+{target2}%", f"+{target3}%"],
                'exit_triggers': [
                    "Any support break",
                    "New red flags emerge",
                    "Sector weakness",
                    f"Stop at -{stop_pct}% is non-negotiable"
                ],
                'safety_rules': [
                    "Never add to losing positions",
                    "Exit immediately on warning signs",
                    "Take profits early rather than late",
                    "Position size: max 5% of portfolio"
                ],
                'capital_preservation': True,
                'target_rationale': f"Conservative targets: 50% haircut applied to {avg_upside:.0f}% expected upside"
            }
            
        elif evaluation['stance'] == 'AVOID':
            return {
                'entry_strategy': "No entry - capital preservation priority",
                'stop_loss': "N/A",
                'profit_targets': ["N/A"],
                'exit_triggers': ["N/A - no position"],
                'safety_rules': ["Avoid this trade entirely"],
                'capital_preservation': True,
                'target_rationale': f"Risk too high: {avg_downside:.0f}% potential downside exceeds tolerance"
            }
            
        else:  # HOLD
            # Minimal targets for holding
            hold_target = max(4, int(conservative_upside * 0.3))
            
            return {
                'entry_strategy': "Wait for 4:1 risk/reward setup minimum",
                'stop_loss': "-2%",
                'profit_targets': [f"+{hold_target}%"],
                'exit_triggers': ["Any adverse news", "Technical breakdown"],
                'safety_rules': ["No new positions", "Reduce on any rally"],
                'capital_preservation': True,
                'target_rationale': "Minimal exposure until better setup emerges"
            }
    
    def synthesize_with_llm(self, evaluation: Dict, trading_plan: Dict, synthesis: Dict, red_flags: List[str]) -> str:
        """Generate report - token efficient"""
        
        if not self.client:
            return self._create_fallback_report(evaluation, trading_plan, red_flags)
        
        try:
            # Historical date context for LLM
            date_context = ""
            if self.analysis_date:
                date_context = f"""
**⚠️ HISTORICAL ANALYSIS MODE ⚠️**
You are analyzing data AS OF {self.analysis_date}.
All research and analyst reports are from this historical date.
Make your conservative evaluation as if you were deciding ON {self.analysis_date}.
Do NOT reference any events or data after {self.analysis_date}.

"""
            
            context = f"""{date_context}Conservative evaluation for {self.ticker}:

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
                max_completion_tokens=1500
            )
            
            report = response.choices[0].message.content
            
            if "CONSERVATIVE STANCE:" not in report:
                report += f"\n\nCONSERVATIVE STANCE: {evaluation['stance']} - Position Size: {evaluation['position_size']*100:.1f}% - Confidence: {evaluation['confidence']}"
            
            return report
            
        except Exception as e:
            return self._create_fallback_report(evaluation, trading_plan, red_flags)
    
    def _create_fallback_report(self, evaluation: Dict, trading_plan: Dict, red_flags: List[str]) -> str:
        """Fallback"""
        date_header = f"\n**Analysis Date:** {self.analysis_date} (HISTORICAL)\n" if self.analysis_date else ""
        
        report = f"""
# CONSERVATIVE RISK EVALUATION: {self.ticker}
{'='*70}
{date_header}
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
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*70}\n")
        
        synthesis = self.load_all_data(synthesis_file, bull_file, bear_file)
        
        if not synthesis or not synthesis.get('bull_thesis') or not synthesis.get('bear_thesis'):
            print("[CONSERVATIVE] ✗ Missing data")
            return "Error: No data", {}
        
        red_flags = self.identify_red_flags(synthesis)
        evaluation = self.evaluate_opportunity(synthesis, red_flags)
        trading_plan = self.generate_trading_plan(evaluation, synthesis) 
               
        report = self.synthesize_with_llm(evaluation, trading_plan, synthesis, red_flags)
        
        self.evaluation = {
            **evaluation,
            'trading_plan': trading_plan,
            'risk_parameters': self.risk_parameters,
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None
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
    parser.add_argument("--analysis-date", help="Historical analysis date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        debator = ConservativeDebator(
            ticker=args.ticker, 
            api_key=args.api_key, 
            model=args.model,
            analysis_date=args.analysis_date
        )
        
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
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()