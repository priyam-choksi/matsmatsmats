"""
Research Manager - Enhanced with Debate Synthesis
Synthesizes bull/bear research and coordinates risk evaluations

Usage: python research_manager.py AAPL --bull-file ../../outputs/bull_thesis.json --bear-file ../../outputs/bear_thesis.json
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ResearchManager:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Enhanced system prompt inspired by reference paper's moderator role
        self.system_prompt = """You are the Research Manager acting as an objective debate moderator and portfolio manager.

**YOUR CRITICAL ROLE:**
You've received comprehensive research from both Bull and Bear analysts. Your job is to:
1. Critically evaluate BOTH sides of the debate
2. Make a definitive decision (not default to HOLD unless strongly justified)
3. Weigh evidence objectively - strongest arguments win
4. Calculate probability-weighted outcomes
5. Provide clear, actionable investment recommendation

**DECISION FRAMEWORK:**

**When to BUY:**
- Bull case significantly stronger than bear case
- High-probability positive catalysts identified
- Risk/reward ratio favorable (>2:1)
- Multiple analysts aligned on upside
- Bear concerns are minor or temporary

**When to SELL:**
- Bear case significantly stronger than bull case
- High-probability downside triggers identified
- Risk/reward unfavorable
- Multiple red flags across analysts
- Bull optimism ignoring critical risks

**When to HOLD (Only if justified):**
- Arguments genuinely balanced with no edge
- Need more data/time before catalyst clarity
- Fair valuation with no strong directional catalyst
- DO NOT default to HOLD just because both have points

**ANALYSIS REQUIREMENTS:**

## Executive Summary
[2-3 sentences: What's the definitive conclusion and why?]

## Debate Evaluation

### Bull Case Strengths
[List 3 strongest bull arguments with supporting data]

### Bear Case Strengths  
[List 3 strongest bear arguments with supporting data]

### Critical Conflicts
[Where do they disagree? Who has better evidence?]

## Probability-Weighted Analysis
- Bull Case Probability: X%
- Bear Case Probability: Y%
- Base Case Probability: Z%
- Expected Value: [Calculate weighted outcome]

## Decision Rationale
[Why you're choosing BUY/SELL/HOLD - reference specific evidence that tipped the scales]

## Investment Plan
**Recommendation:** BUY/SELL/HOLD
**Position Size:** X% of portfolio
**Entry Strategy:** [Specific approach]
**Risk Management:** [Stop loss, position limits]
**Time Horizon:** [Expected holding period]
**Key Catalysts to Monitor:** [What could change thesis]

RESEARCH CONCLUSION: Strong Buy/Buy/Hold/Sell/Strong Sell - Confidence: High/Medium/Low

**Be decisive.** Commit to the stance supported by strongest evidence. Avoid fence-sitting."""
        
        # Storage
        self.research_inputs = {
            'bull_thesis': {},
            'bear_thesis': {},
            'risk_evaluations': {
                'aggressive': {},
                'neutral': {},
                'conservative': {}
            }
        }
    
    def load_research_files(self, bull_file: str, bear_file: str):
        """Load bull and bear theses"""
        print(f"[RESEARCH_MGR] Loading research files...")
        
        # Load bull thesis
        if os.path.exists(bull_file):
            with open(bull_file, 'r', encoding='utf-8') as f:
                self.research_inputs['bull_thesis'] = json.load(f)
            print(f"[RESEARCH_MGR] ✓ Bull thesis loaded ({bull_file})")
        else:
            print(f"[RESEARCH_MGR] ⚠️  Bull thesis not found: {bull_file}")
        
        # Load bear thesis
        if os.path.exists(bear_file):
            with open(bear_file, 'r', encoding='utf-8') as f:
                self.research_inputs['bear_thesis'] = json.load(f)
            print(f"[RESEARCH_MGR] ✓ Bear thesis loaded ({bear_file})")
        else:
            print(f"[RESEARCH_MGR] ⚠️  Bear thesis not found: {bear_file}")
    
    def load_risk_evaluations(self):
        """Load risk debator evaluations (optional)"""
        print(f"[RESEARCH_MGR] Loading risk evaluations...")
        
        risk_files = {
            'aggressive': f"../../outputs/aggressive_evaluation_{self.ticker}.json",
            'neutral': f"../../outputs/neutral_evaluation_{self.ticker}.json",
            'conservative': f"../../outputs/conservative_evaluation_{self.ticker}.json"
        }
        
        loaded_count = 0
        
        for risk_type, filepath in risk_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.research_inputs['risk_evaluations'][risk_type] = json.load(f)
                    loaded_count += 1
                except Exception as e:
                    print(f"[RESEARCH_MGR] ⚠️  Error loading {risk_type}: {e}")
        
        print(f"[RESEARCH_MGR] ✓ Loaded {loaded_count}/3 risk evaluations")
    
    def format_debate_for_analysis(self) -> str:
        """Format bull/bear debate for LLM analysis"""
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        debate = f"""# Investment Debate for {self.ticker}

## BULL ANALYST POSITION

**Core Thesis:**
{bull.get('core_thesis', 'Not available')}

**Key Opportunities:**
"""
        # Add bull opportunities
        for category, opps in bull.get('opportunities', {}).items():
            if opps:
                debate += f"\n### {category.title()}:\n"
                for opp in opps[:3]:
                    debate += f"- {opp}\n"
        
        # Add bull catalysts
        debate += "\n**Upside Catalysts:**\n"
        for catalyst in bull.get('catalysts', [])[:5]:
            debate += f"- {catalyst.get('description', '')} ({catalyst.get('timeline', 'TBD')})\n"
        
        # Add bull R/R
        rr = bull.get('risk_reward', {})
        debate += f"\n**Risk/Reward:** {rr.get('upside_potential', 'N/A')} upside, {rr.get('downside_risk', 'N/A')} downside"
        debate += f" (Ratio: {rr.get('reward_risk_ratio', 0):.1f}:1)\n"
        
        # If debate history exists (from deep/research mode)
        if 'debate_history' in bull:
            debate += "\n**Debate Arguments:**\n"
            for entry in bull['debate_history']:
                debate += f"\nRound {entry['round']}: {entry['argument'][:400]}...\n"
        
        debate += "\n" + "="*70 + "\n\n"
        
        debate += "## BEAR ANALYST POSITION\n\n"
        
        debate += f"**Core Thesis:**\n{bear.get('core_thesis', 'Not available')}\n\n"
        
        # Add bear risks
        debate += "**Risk Factors:**\n"
        for category, risks in bear.get('risks', {}).items():
            if risks:
                debate += f"\n### {category.title()}:\n"
                for risk in risks[:3]:
                    debate += f"- {risk}\n"
        
        # Add bear triggers
        debate += "\n**Downside Triggers:**\n"
        for trigger in bear.get('downside_triggers', [])[:5]:
            debate += f"- {trigger.get('description', '')} ({trigger.get('timeline', 'TBD')})\n"
        
        # Add bear risk assessment
        ra = bear.get('risk_assessment', {})
        debate += f"\n**Risk Assessment:** {ra.get('downside_risk', 'N/A')} downside, {ra.get('limited_upside', 'N/A')} upside"
        debate += f" (Risk Level: {ra.get('risk_level', 'N/A')})\n"
        
        # If debate history exists
        if 'debate_history' in bear:
            debate += "\n**Debate Arguments:**\n"
            for entry in bear['debate_history']:
                debate += f"\nRound {entry['round']}: {entry['argument'][:400]}...\n"
        
        return debate
    
    def calculate_probabilities(self) -> Dict[str, Any]:
        """Calculate scenario probabilities"""
        print(f"[RESEARCH_MGR] Calculating probabilities...")
        
        weights = {
            'bull_research': 0.25,
            'bear_research': 0.25,
            'aggressive': 0.15,
            'neutral': 0.25,
            'conservative': 0.10
        }
        
        bull_score = 0
        bear_score = 0
        
        # Bull thesis
        bull_conviction = self.research_inputs['bull_thesis'].get('risk_reward', {}).get('conviction_level', 'LOW')
        if bull_conviction == 'HIGH':
            bull_score += weights['bull_research']
        elif bull_conviction == 'MEDIUM':
            bull_score += weights['bull_research'] * 0.6
        
        # Bear thesis
        bear_risk = self.research_inputs['bear_thesis'].get('risk_assessment', {}).get('risk_level', 'LOW')
        if bear_risk == 'HIGH':
            bear_score += weights['bear_research']
        elif bear_risk == 'MEDIUM':
            bear_score += weights['bear_research'] * 0.6
        
        # Risk evaluations
        for risk_type in ['aggressive', 'neutral', 'conservative']:
            eval_data = self.research_inputs['risk_evaluations'].get(risk_type, {})
            stance = eval_data.get('stance', '')
            
            if 'BUY' in stance:
                bull_score += weights[risk_type]
            elif 'SELL' in stance or 'AVOID' in stance:
                bear_score += weights[risk_type]
            elif 'HOLD' in stance:
                # Distribute hold votes proportionally
                bull_score += weights[risk_type] * 0.3
                bear_score += weights[risk_type] * 0.2
        
        # Normalize
        bull_prob = min(bull_score * 100, 85)
        bear_prob = min(bear_score * 100, 85)
        base_prob = max(100 - bull_prob - bear_prob, 10)
        
        # Renormalize to 100%
        total = bull_prob + bear_prob + base_prob
        if total > 0:
            bull_prob = (bull_prob / total) * 100
            bear_prob = (bear_prob / total) * 100
            base_prob = (base_prob / total) * 100
        
        bull_rr = self.research_inputs['bull_thesis'].get('risk_reward', {})
        bear_ra = self.research_inputs['bear_thesis'].get('risk_assessment', {})
        
        probabilities = {
            'bull_case': bull_prob,
            'bear_case': bear_prob,
            'base_case': base_prob,
            'scenarios': [
                {
                    'name': 'Bull Case',
                    'probability': f"{bull_prob:.0f}%",
                    'outcome': bull_rr.get('upside_potential', '20-30%'),
                    'description': 'Positive catalysts materialize, upside targets reached'
                },
                {
                    'name': 'Bear Case',
                    'probability': f"{bear_prob:.0f}%",
                    'outcome': bear_ra.get('downside_risk', '15-20%'),
                    'description': 'Risk triggers activate, downside scenario plays out'
                },
                {
                    'name': 'Base Case',
                    'probability': f"{base_prob:.0f}%",
                    'outcome': 'Sideways ±5%',
                    'description': 'Mixed signals, range-bound action'
                }
            ]
        }
        
        print(f"[RESEARCH_MGR] ✓ Probabilities: Bull {bull_prob:.0f}%, Bear {bear_prob:.0f}%, Base {base_prob:.0f}%")
        
        return probabilities
    
    def analyze_consensus(self) -> Dict[str, Any]:
        """Analyze consensus across all inputs"""
        print(f"[RESEARCH_MGR] Analyzing consensus...")
        
        consensus = {
            'recommendations': {},
            'position_sizes': {},
            'conviction_levels': {},
            'key_agreements': [],
            'key_conflicts': []
        }
        
        # Bull recommendation
        if self.research_inputs['bull_thesis']:
            bull_rr = self.research_inputs['bull_thesis'].get('risk_reward', {})
            rr_ratio = bull_rr.get('reward_risk_ratio', 0)
            consensus['recommendations']['bull'] = 'BUY' if rr_ratio > 2 else 'HOLD'
            consensus['conviction_levels']['bull'] = bull_rr.get('conviction_level', 'LOW')
        
        # Bear recommendation
        if self.research_inputs['bear_thesis']:
            bear_ra = self.research_inputs['bear_thesis'].get('risk_assessment', {})
            consensus['recommendations']['bear'] = 'SELL' if bear_ra.get('risk_level') == 'HIGH' else 'HOLD'
            consensus['conviction_levels']['bear'] = bear_ra.get('conviction_level', 'LOW')
        
        # Risk evaluations
        for risk_type, evaluation in self.research_inputs['risk_evaluations'].items():
            if evaluation:
                consensus['recommendations'][risk_type] = evaluation.get('stance', 'HOLD')
                consensus['position_sizes'][risk_type] = evaluation.get('position_size', 0)
                consensus['conviction_levels'][risk_type] = evaluation.get('confidence', 'LOW')
        
        # Find agreements
        all_recs = list(consensus['recommendations'].values())
        if all_recs:
            from collections import Counter
            rec_counts = Counter(all_recs)
            most_common = rec_counts.most_common(1)[0]
            
            if most_common[1] >= 3:
                consensus['key_agreements'].append(
                    f"{most_common[1]}/{len(all_recs)} agree on {most_common[0]}"
                )
        
        # Find conflicts
        if 'BUY' in all_recs and 'SELL' in all_recs:
            consensus['key_conflicts'].append("Direct BUY vs SELL conflict - requires resolution")
        
        # Average position size
        positions = [p for p in consensus['position_sizes'].values() if p > 0]
        consensus['avg_position_size'] = sum(positions) / len(positions) if positions else 0
        
        print(f"[RESEARCH_MGR] ✓ Consensus analyzed")
        
        return consensus
    
    def form_conclusion(self, consensus: Dict, probabilities: Dict) -> Dict[str, Any]:
        """Form final investment conclusion"""
        print(f"[RESEARCH_MGR] Forming conclusion...")
        
        bull_prob = probabilities['bull_case']
        bear_prob = probabilities['bear_case']
        avg_position = consensus['avg_position_size']
        
        # Decision logic (decisive, not fence-sitting)
        if bull_prob > 60 and avg_position > 0.05:
            recommendation = 'BUY'
            confidence = 'HIGH' if bull_prob > 75 else 'MEDIUM'
            rationale = f"Bull case dominates ({bull_prob:.0f}% probability) with favorable risk/reward"
        elif bull_prob > 50 and avg_position > 0.03:
            recommendation = 'BUY'
            confidence = 'MEDIUM'
            rationale = f"Moderate bull edge ({bull_prob:.0f}% vs {bear_prob:.0f}%)"
        elif bear_prob > 60:
            recommendation = 'SELL'
            confidence = 'HIGH' if bear_prob > 75 else 'MEDIUM'
            rationale = f"Bear case dominates ({bear_prob:.0f}% probability) - risk too high"
        elif bear_prob > 50:
            recommendation = 'SELL'
            confidence = 'MEDIUM'
            rationale = f"Moderate bear edge ({bear_prob:.0f}% vs {bull_prob:.0f}%)"
        elif avg_position > 0.02:
            recommendation = 'HOLD'
            confidence = 'LOW'
            rationale = f"Balanced probabilities ({bull_prob:.0f}% vs {bear_prob:.0f}%) - wait for clarity"
        else:
            recommendation = 'AVOID'
            confidence = 'MEDIUM'
            rationale = "Insufficient edge and low conviction across all analyses"
        
        # Position sizing
        if recommendation == 'BUY':
            if confidence == 'HIGH':
                position_range = "8-12%"
            elif confidence == 'MEDIUM':
                position_range = "4-8%"
            else:
                position_range = "2-4%"
        elif recommendation == 'SELL':
            position_range = "0% (Exit)"
        elif recommendation == 'HOLD':
            position_range = "2-5% (Maintain or small pilot)"
        else:  # AVOID
            position_range = "0%"
        
        conclusion = {
            'recommendation': recommendation,
            'confidence': confidence,
            'rationale': rationale,
            'position_size_range': position_range,
            'time_horizon': '3-6 months',
            'key_catalysts': self._extract_key_catalysts(),
            'key_risks': self._extract_key_risks(),
            'expected_value': self._calculate_expected_value(probabilities)
        }
        
        print(f"[RESEARCH_MGR] ✓ Conclusion: {recommendation} - {confidence} confidence")
        
        return conclusion
    
    def _extract_key_catalysts(self) -> List[str]:
        """Extract top catalysts from bull thesis"""
        catalysts = self.research_inputs['bull_thesis'].get('catalysts', [])
        return [c.get('description', '') for c in catalysts[:3]]
    
    def _extract_key_risks(self) -> List[str]:
        """Extract top risks from bear thesis"""
        triggers = self.research_inputs['bear_thesis'].get('downside_triggers', [])
        return [t.get('description', '') for t in triggers[:3]]
    
    def _calculate_expected_value(self, probabilities: Dict) -> float:
        """Calculate probability-weighted expected value"""
        # Extract upside/downside percentages
        bull_outcome = self.research_inputs['bull_thesis'].get('risk_reward', {}).get('upside_potential', '20%')
        bear_outcome = self.research_inputs['bear_thesis'].get('risk_assessment', {}).get('downside_risk', '15%')
        
        # Parse percentages (take midpoint of range)
        import re
        
        bull_nums = re.findall(r'\d+', bull_outcome)
        bear_nums = re.findall(r'\d+', bear_outcome)
        
        bull_pct = sum(int(n) for n in bull_nums) / len(bull_nums) if bull_nums else 20
        bear_pct = sum(int(n) for n in bear_nums) / len(bear_nums) if bear_nums else 15
        
        # Weighted calculation
        bull_prob = probabilities['bull_case'] / 100
        bear_prob = probabilities['bear_case'] / 100
        
        expected_value = (bull_pct * bull_prob) - (bear_pct * bear_prob)
        
        return expected_value
    
    def synthesize_with_llm(self, debate_formatted: str, consensus: Dict, probabilities: Dict, conclusion: Dict) -> str:
        """
        Generate comprehensive synthesis using LLM
        Like the reference paper's moderator decision
        """
        if not self.client:
            print("[RESEARCH_MGR] ⚠️  No API key - using fallback")
            return self._create_fallback_report(consensus, probabilities, conclusion)
        
        try:
            print(f"[RESEARCH_MGR] Generating synthesis with {self.model}...")
            
            context = f"""# Research Synthesis for {self.ticker}

{debate_formatted}

{'='*70}

## Preliminary Analysis

**Probability Assessment:**
{json.dumps(probabilities, indent=2)}

**Consensus Summary:**
{json.dumps(consensus, indent=2)}

**Initial Conclusion:**
{json.dumps(conclusion, indent=2)}

{'='*70}

As Research Manager, provide your definitive investment decision. Evaluate both sides critically and make a clear recommendation supported by the strongest evidence."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.6,  # Moderate temp for balanced decision
                max_tokens=3000
            )
            
            synthesis = response.choices[0].message.content
            
            # Validate conclusion
            if "RESEARCH CONCLUSION:" not in synthesis:
                synthesis += f"\n\nRESEARCH CONCLUSION: {conclusion['recommendation']} - Confidence: {conclusion['confidence']}"
            
            print(f"[RESEARCH_MGR] ✓ Synthesis complete ({len(synthesis)} chars)")
            
            return synthesis
            
        except Exception as e:
            print(f"[RESEARCH_MGR] ❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_report(consensus, probabilities, conclusion)
    
    def _create_fallback_report(self, consensus: Dict, probabilities: Dict, conclusion: Dict) -> str:
        """Fallback report without LLM"""
        report = f"""
# RESEARCH SYNTHESIS: {self.ticker}
{'='*70}
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Mode: Fallback (LLM unavailable)*

## Probability Assessment
"""
        for scenario in probabilities['scenarios']:
            report += f"\n**{scenario['name']}:** {scenario['probability']}\n"
            report += f"  Outcome: {scenario['outcome']}\n"
            report += f"  {scenario['description']}\n"
        
        report += f"""
## Consensus Analysis

**Recommendations:**
"""
        for source, rec in consensus['recommendations'].items():
            report += f"  - {source}: {rec}\n"
        
        if consensus['key_agreements']:
            report += f"\n**Agreement:** {consensus['key_agreements'][0]}\n"
        
        if consensus['key_conflicts']:
            report += f"**Conflict:** {consensus['key_conflicts'][0]}\n"
        
        report += f"""
## Final Conclusion

**Recommendation:** {conclusion['recommendation']}
**Confidence:** {conclusion['confidence']}
**Position Size:** {conclusion['position_size_range']}
**Time Horizon:** {conclusion['time_horizon']}

**Rationale:** {conclusion['rationale']}

**Expected Value:** {conclusion['expected_value']:+.1f}%

RESEARCH CONCLUSION: {conclusion['recommendation']} - Confidence: {conclusion['confidence']}
"""
        return report
    
    def synthesize(self, load_risk_evals: bool = True) -> tuple:
        """
        Main synthesis workflow
        Returns (report, synthesis_data)
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"RESEARCH MANAGER: {self.ticker}")
        print(f"{'='*70}\n")
        
        # Load risk evaluations (optional)
        if load_risk_evals:
            self.load_risk_evaluations()
        
        # Format debate
        print(f"[RESEARCH_MGR] Formatting debate...")
        debate_formatted = self.format_debate_for_analysis()
        
        # Calculate probabilities
        probabilities = self.calculate_probabilities()
        
        # Analyze consensus
        consensus = self.analyze_consensus()
        
        # Form conclusion
        conclusion = self.form_conclusion(consensus, probabilities)
        
        # Generate LLM synthesis
        print(f"\n[RESEARCH_MGR] Synthesizing final decision...\n")
        report = self.synthesize_with_llm(debate_formatted, consensus, probabilities, conclusion)
        
        # Compile synthesis data
        synthesis_data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'consensus': consensus,
            'probabilities': probabilities,
            'conclusion': conclusion,
            'research_inputs_summary': {
                'bull_thesis': bool(self.research_inputs['bull_thesis']),
                'bear_thesis': bool(self.research_inputs['bear_thesis']),
                'bull_mode': self.research_inputs['bull_thesis'].get('mode', 'unknown'),
                'bear_mode': self.research_inputs['bear_thesis'].get('mode', 'unknown'),
                'risk_evals': sum(1 for e in self.research_inputs['risk_evaluations'].values() if e)
            }
        }
        
        elapsed = time.time() - start_time
        print(f"\n[RESEARCH_MGR] ✓ Synthesis complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, synthesis_data


def main():
    parser = argparse.ArgumentParser(
        description="Research Manager - Synthesizes bull/bear debate and makes final decision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_manager.py AAPL
  python research_manager.py AAPL --bull-file ../../outputs/bull_thesis.json
  python research_manager.py AAPL --skip-evaluations --output synthesis.txt
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("--bull-file", default="../../outputs/bull_thesis.json",
                       help="Bull thesis JSON file")
    parser.add_argument("--bear-file", default="../../outputs/bear_thesis.json",
                       help="Bear thesis JSON file")
    parser.add_argument("--skip-evaluations", action="store_true",
                       help="Skip loading risk evaluations")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-synthesis", help="Save synthesis JSON")
    
    args = parser.parse_args()
    
    try:
        manager = ResearchManager(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
        # Load research files
        manager.load_research_files(args.bull_file, args.bear_file)
        
        # Check we have minimum data
        if not manager.research_inputs['bull_thesis'] or not manager.research_inputs['bear_thesis']:
            print("\n❌ Error: Both bull and bear theses required")
            print("Run bull_researcher.py and bear_researcher.py first!")
            sys.exit(1)
        
        # Run synthesis
        report, synthesis_data = manager.synthesize(load_risk_evals=not args.skip_evaluations)
        
        print(report)
        
        # Save outputs
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to {args.output}")
        
        if args.save_synthesis:
            with open(args.save_synthesis, 'w', encoding='utf-8') as f:
                json.dump(synthesis_data, f, indent=2)
            print(f"✓ Synthesis data saved to {args.save_synthesis}")
        
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