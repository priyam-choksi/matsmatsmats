"""
Research Manager - Enhanced with Live Debate Orchestration
Coordinates bull/bear debate and synthesizes results

Usage: 
  # Old way (static files)
  python research_manager.py AAPL --bull-file bull.json --bear-file bear.json
  
  # New way (orchestrated debate)
  python research_manager.py AAPL --mode debate --rounds 3
"""

import os
from pathlib import Path
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

# Import the researchers (you'll need to adjust paths)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researcher.bull_researcher import BullResearcher
from researcher.bear_researcher import BearResearcher


class ResearchManager:
    def _init_(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Initialize researchers for debate mode
        self.bull_researcher = None
        self.bear_researcher = None
        
        # Enhanced system prompt (keeping original)
        self.system_prompt = """You are the Research Manager acting as an objective debate moderator and portfolio manager.

*YOUR CRITICAL ROLE:*
You've received comprehensive research from both Bull and Bear analysts. Your job is to:
1. Critically evaluate BOTH sides of the debate
2. Make a definitive decision (not default to HOLD unless strongly justified)
3. Weigh evidence objectively - strongest arguments win
4. Calculate probability-weighted outcomes
5. Provide clear, actionable investment recommendation

*DECISION FRAMEWORK:*

*When to BUY:*
- Bull case significantly stronger than bear case
- High-probability positive catalysts identified
- Risk/reward ratio favorable (>2:1)
- Multiple analysts aligned on upside
- Bear concerns are minor or temporary

*When to SELL:*
- Bear case significantly stronger than bull case
- High-probability downside triggers identified
- Risk/reward unfavorable
- Multiple red flags across analysts
- Bull optimism ignoring critical risks

*When to HOLD (Only if justified):*
- Arguments genuinely balanced with no edge
- Need more data/time before catalyst clarity
- Fair valuation with no strong directional catalyst
- DO NOT default to HOLD just because both have points

RESEARCH CONCLUSION: Strong Buy/Buy/Hold/Sell/Strong Sell - Confidence: High/Medium/Low

*Be decisive.* Commit to the stance supported by strongest evidence. Avoid fence-sitting."""
        
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
        
        # Debate history for orchestration
        self.debate_history = []
    
    def orchestrate_debate(self, discussion_points: Dict, rounds: int = 3, mode: str = 'deep') -> Dict:
        """
        NEW METHOD: Orchestrate live debate between bull and bear
        This replaces static file loading with dynamic interaction
        """
        print(f"\n[RESEARCH_MGR] 🎯 ORCHESTRATING {rounds}-ROUND DEBATE\n")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Initialize researchers
        self.bull_researcher = BullResearcher(self.ticker, api_key=self.api_key, model=self.model)
        self.bear_researcher = BearResearcher(self.ticker, api_key=self.api_key, model=self.model)
        
        # Store arguments for each round
        bull_arguments = []
        bear_arguments = []
        
        for round_num in range(1, rounds + 1):
            print(f"[RESEARCH_MGR] 🔄 Round {round_num}/{rounds}")
            print(f"{'-'*40}")
            
            if round_num == 1:
                # Round 1: Initial arguments based on discussion points
                print(f"[RESEARCH_MGR] Bull generating opening argument...")
                bull_arg = self.bull_researcher._generate_initial_argument(
                    self.bull_researcher.build_bull_thesis(
                        self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                        self.bull_researcher.calculate_risk_reward(discussion_points)
                    ),
                    self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                    self.bull_researcher.identify_upside_catalysts(discussion_points),
                    self.bull_researcher.calculate_risk_reward(discussion_points),
                    discussion_points
                )
                
                print(f"[RESEARCH_MGR] Bear generating opening argument...")
                bear_arg = self.bear_researcher._generate_initial_argument(
                    self.bear_researcher.build_bear_thesis(
                        self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                        self.bear_researcher.calculate_risk_assessment(discussion_points)
                    ),
                    self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                    self.bear_researcher.identify_downside_triggers(discussion_points),
                    self.bear_researcher.calculate_risk_assessment(discussion_points),
                    discussion_points
                )
            else:
                # Subsequent rounds: Respond to each other's arguments
                print(f"[RESEARCH_MGR] Bull responding to bear...")
                bull_arg = self.bull_researcher._generate_debate_response(
                    round_num,
                    rounds,
                    self.debate_history,
                    bear_arguments[-1],  # Respond to bear's last argument
                    self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                    discussion_points
                )
                
                print(f"[RESEARCH_MGR] Bear responding to bull...")
                bear_arg = self.bear_researcher._generate_debate_response(
                    round_num,
                    rounds,
                    self.debate_history,
                    bull_arguments[-1],  # Respond to bull's last argument
                    self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                    discussion_points
                )
            
            # Store arguments
            bull_arguments.append(bull_arg)
            bear_arguments.append(bear_arg)
            
            # Update debate history
            self.debate_history.append({
                'round': round_num,
                'speaker': 'bull',
                'argument': bull_arg
            })
            self.debate_history.append({
                'round': round_num,
                'speaker': 'bear',
                'argument': bear_arg
            })
            
            print(f"[RESEARCH_MGR] ✓ Round {round_num} complete")
            print(f"  Bull: {len(bull_arg)} chars")
            print(f"  Bear: {len(bear_arg)} chars")
            print()
        
        # Compile final theses with debate history
        self.research_inputs['bull_thesis'] = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': f'DEBATE_{mode.upper()}',
            'rounds': rounds,
            'core_thesis': self.bull_researcher.build_bull_thesis(
                self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                self.bull_researcher.calculate_risk_reward(discussion_points)
            ),
            'opportunities': self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
            'catalysts': self.bull_researcher.identify_upside_catalysts(discussion_points),
            'risk_reward': self.bull_researcher.calculate_risk_reward(discussion_points),
            'entry_strategies': self.bull_researcher.suggest_entry_strategies(
                self.bull_researcher.calculate_risk_reward(discussion_points)
            ),
            'debate_history': [h for h in self.debate_history if h['speaker'] == 'bull'],
            'final_argument': bull_arguments[-1]
        }
        
        self.research_inputs['bear_thesis'] = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': f'DEBATE_{mode.upper()}',
            'rounds': rounds,
            'core_thesis': self.bear_researcher.build_bear_thesis(
                self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                self.bear_researcher.calculate_risk_assessment(discussion_points)
            ),
            'risks': self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
            'downside_triggers': self.bear_researcher.identify_downside_triggers(discussion_points),
            'risk_assessment': self.bear_researcher.calculate_risk_assessment(discussion_points),
            'hedging_strategies': self.bear_researcher.suggest_hedging_strategies(
                self.bear_researcher.calculate_risk_assessment(discussion_points)
            ),
            'debate_history': [h for h in self.debate_history if h['speaker'] == 'bear'],
            'final_argument': bear_arguments[-1]
        }
        
        elapsed = time.time() - start_time
        print(f"[RESEARCH_MGR] ✓ Debate complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return {
            'debate_complete': True,
            'rounds': rounds,
            'duration': elapsed,
            'bull_thesis': self.research_inputs['bull_thesis'],
            'bear_thesis': self.research_inputs['bear_thesis']
        }
    
    def enhance_static_with_debate(self) -> bool:
        """
        NEW: Enhance existing bull/bear theses with debate rounds
        This is called when we have static files but want to add debate
        """
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        # Check if these were generated with rounds > 1
        bull_rounds = bull.get('rounds', 1)
        bear_rounds = bear.get('rounds', 1)
        
        if bull_rounds > 1 or bear_rounds > 1:
            rounds = max(bull_rounds, bear_rounds)
            print(f"[RESEARCH_MGR] Enhancing with {rounds}-round debate...")
            
            # Load discussion points
            discussion_file = self.outputs_path / "discussion_points.json"
            if not discussion_file.exists():
                discussion_file = Path("../../outputs/discussion_points.json")
            
            if discussion_file.exists():
                with open(discussion_file, 'r') as f:
                    discussion_points = json.load(f)
                
                # Initialize researchers
                self.bull_researcher = BullResearcher(self.ticker, api_key=self.api_key, model=self.model)
                self.bear_researcher = BearResearcher(self.ticker, api_key=self.api_key, model=self.model)
                
                # Generate debate based on existing theses
                debate_history = []
                
                # Use existing core theses as round 1
                bull_arg = bull.get('final_argument') or bull.get('core_thesis', '')
                bear_arg = bear.get('final_argument') or bear.get('core_thesis', '')
                
                debate_history.append({'round': 1, 'speaker': 'bull', 'argument': bull_arg})
                debate_history.append({'round': 1, 'speaker': 'bear', 'argument': bear_arg})
                
                # Generate additional rounds if needed
                for round_num in range(2, rounds + 1):
                    print(f"[RESEARCH_MGR] Generating debate round {round_num}...")
                    
                    # Bull responds to bear
                    bull_arg = self.bull_researcher._generate_debate_response(
                        round_num, rounds, debate_history, bear_arg,
                        bull.get('opportunities', {}), discussion_points
                    )
                    
                    # Bear responds to bull  
                    bear_arg = self.bear_researcher._generate_debate_response(
                        round_num, rounds, debate_history, bull_arg,
                        bear.get('risks', {}), discussion_points
                    )
                    
                    debate_history.append({'round': round_num, 'speaker': 'bull', 'argument': bull_arg})
                    debate_history.append({'round': round_num, 'speaker': 'bear', 'argument': bear_arg})
                
                # Update theses with debate history
                self.research_inputs['bull_thesis']['debate_history'] = [h for h in debate_history if h['speaker'] == 'bull']
                self.research_inputs['bear_thesis']['debate_history'] = [h for h in debate_history if h['speaker'] == 'bear']
                self.research_inputs['bull_thesis']['final_argument'] = bull_arg
                self.research_inputs['bear_thesis']['final_argument'] = bear_arg
                
                print(f"[RESEARCH_MGR] ✓ Enhanced with {rounds}-round debate")
                return True
            else:
                print(f"[RESEARCH_MGR] No discussion points found for debate enhancement")
                return False
        else:
            print(f"[RESEARCH_MGR] Single-round theses, no debate enhancement needed")
            return False
        """
        NEW: Reconstruct debate from bull/bear thesis files
        This runs when we have thesis files that were generated with deep/research mode
        """
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        # Check if these were generated with debate capability
        bull_mode = bull.get('mode', 'SHALLOW')
        bear_mode = bear.get('mode', 'SHALLOW')
        bull_rounds = bull.get('rounds', 1)
        bear_rounds = bear.get('rounds', 1)
        
        if ('DEEP' in bull_mode or 'RESEARCH' in bull_mode) and bull_rounds > 1:
            print(f"[RESEARCH_MGR] Detected {bull_rounds}-round capable theses, orchestrating debate...")
            
            # Load discussion points if available
            discussion_file = self.outputs_path / "discussion_points.json"
            if discussion_file.exists():
                with open(discussion_file, 'r') as f:
                    discussion_points = json.load(f)
                
                # Initialize researchers
                self.bull_researcher = BullResearcher(self.ticker, api_key=self.api_key, model=self.model)
                self.bear_researcher = BearResearcher(self.ticker, api_key=self.api_key, model=self.model)
                
                # Reconstruct debate using the existing analysis
                debate_history = []
                
                for round_num in range(1, min(bull_rounds, bear_rounds) + 1):
                    print(f"[RESEARCH_MGR] Generating debate round {round_num}...")
                    
                    if round_num == 1:
                        # Use initial arguments from the thesis data
                        bull_arg = bull.get('core_thesis', '')
                        bear_arg = bear.get('core_thesis', '')
                    else:
                        # Generate responses based on previous round
                        bull_arg = self.bull_researcher._generate_debate_response(
                            round_num, bull_rounds, debate_history,
                            debate_history[-1]['argument'] if debate_history and debate_history[-1]['speaker'] == 'bear' else None,
                            bull.get('opportunities', {}), discussion_points
                        )
                        
                        bear_arg = self.bear_researcher._generate_debate_response(
                            round_num, bear_rounds, debate_history,
                            debate_history[-1]['argument'] if debate_history and debate_history[-1]['speaker'] == 'bull' else None,
                            bear.get('risks', {}), discussion_points
                        )
                    
                    debate_history.append({'round': round_num, 'speaker': 'bull', 'argument': bull_arg})
                    debate_history.append({'round': round_num, 'speaker': 'bear', 'argument': bear_arg})
                
                # Update the theses with debate history
                self.research_inputs['bull_thesis']['debate_history'] = [h for h in debate_history if h['speaker'] == 'bull']
                self.research_inputs['bear_thesis']['debate_history'] = [h for h in debate_history if h['speaker'] == 'bear']
                
                print(f"[RESEARCH_MGR] ✓ Debate reconstruction complete")
                return True
            else:
                print(f"[RESEARCH_MGR] No discussion points found, using static theses")
                return False
        else:
            print(f"[RESEARCH_MGR] Theses are shallow mode, no debate needed")
            return False
        """
        NEW METHOD: Orchestrate live debate between bull and bear
        This replaces static file loading with dynamic interaction
        """
        print(f"\n[RESEARCH_MGR] 🎯 ORCHESTRATING {rounds}-ROUND DEBATE\n")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Initialize researchers
        self.bull_researcher = BullResearcher(self.ticker, api_key=self.api_key, model=self.model)
        self.bear_researcher = BearResearcher(self.ticker, api_key=self.api_key, model=self.model)
        
        # Store arguments for each round
        bull_arguments = []
        bear_arguments = []
        
        for round_num in range(1, rounds + 1):
            print(f"[RESEARCH_MGR] 🔄 Round {round_num}/{rounds}")
            print(f"{'-'*40}")
            
            if round_num == 1:
                # Round 1: Initial arguments based on discussion points
                print(f"[RESEARCH_MGR] Bull generating opening argument...")
                bull_arg = self.bull_researcher._generate_initial_argument(
                    self.bull_researcher.build_bull_thesis(
                        self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                        self.bull_researcher.calculate_risk_reward(discussion_points)
                    ),
                    self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                    self.bull_researcher.identify_upside_catalysts(discussion_points),
                    self.bull_researcher.calculate_risk_reward(discussion_points),
                    discussion_points
                )
                
                print(f"[RESEARCH_MGR] Bear generating opening argument...")
                bear_arg = self.bear_researcher._generate_initial_argument(
                    self.bear_researcher.build_bear_thesis(
                        self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                        self.bear_researcher.calculate_risk_assessment(discussion_points)
                    ),
                    self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                    self.bear_researcher.identify_downside_triggers(discussion_points),
                    self.bear_researcher.calculate_risk_assessment(discussion_points),
                    discussion_points
                )
            else:
                # Subsequent rounds: Respond to each other's arguments
                print(f"[RESEARCH_MGR] Bull responding to bear...")
                bull_arg = self.bull_researcher._generate_debate_response(
                    round_num,
                    rounds,
                    self.debate_history,
                    bear_arguments[-1],  # Respond to bear's last argument
                    self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                    discussion_points
                )
                
                print(f"[RESEARCH_MGR] Bear responding to bull...")
                bear_arg = self.bear_researcher._generate_debate_response(
                    round_num,
                    rounds,
                    self.debate_history,
                    bull_arguments[-1],  # Respond to bull's last argument
                    self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                    discussion_points
                )
            
            # Store arguments
            bull_arguments.append(bull_arg)
            bear_arguments.append(bear_arg)
            
            # Update debate history
            self.debate_history.append({
                'round': round_num,
                'speaker': 'bull',
                'argument': bull_arg
            })
            self.debate_history.append({
                'round': round_num,
                'speaker': 'bear',
                'argument': bear_arg
            })
            
            print(f"[RESEARCH_MGR] ✓ Round {round_num} complete")
            print(f"  Bull: {len(bull_arg)} chars")
            print(f"  Bear: {len(bear_arg)} chars")
            print()
        
        # Compile final theses with debate history
        self.research_inputs['bull_thesis'] = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': f'DEBATE_{mode.upper()}',
            'rounds': rounds,
            'core_thesis': self.bull_researcher.build_bull_thesis(
                self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
                self.bull_researcher.calculate_risk_reward(discussion_points)
            ),
            'opportunities': self.bull_researcher.extract_bull_signals_from_full_reports(discussion_points),
            'catalysts': self.bull_researcher.identify_upside_catalysts(discussion_points),
            'risk_reward': self.bull_researcher.calculate_risk_reward(discussion_points),
            'entry_strategies': self.bull_researcher.suggest_entry_strategies(
                self.bull_researcher.calculate_risk_reward(discussion_points)
            ),
            'debate_history': [h for h in self.debate_history if h['speaker'] == 'bull'],
            'final_argument': bull_arguments[-1]
        }
        
        self.research_inputs['bear_thesis'] = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'mode': f'DEBATE_{mode.upper()}',
            'rounds': rounds,
            'core_thesis': self.bear_researcher.build_bear_thesis(
                self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
                self.bear_researcher.calculate_risk_assessment(discussion_points)
            ),
            'risks': self.bear_researcher.extract_bear_signals_from_full_reports(discussion_points),
            'downside_triggers': self.bear_researcher.identify_downside_triggers(discussion_points),
            'risk_assessment': self.bear_researcher.calculate_risk_assessment(discussion_points),
            'hedging_strategies': self.bear_researcher.suggest_hedging_strategies(
                self.bear_researcher.calculate_risk_assessment(discussion_points)
            ),
            'debate_history': [h for h in self.debate_history if h['speaker'] == 'bear'],
            'final_argument': bear_arguments[-1]
        }
        
        elapsed = time.time() - start_time
        print(f"[RESEARCH_MGR] ✓ Debate complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return {
            'debate_complete': True,
            'rounds': rounds,
            'duration': elapsed,
            'bull_thesis': self.research_inputs['bull_thesis'],
            'bear_thesis': self.research_inputs['bear_thesis']
        }
    
    def load_research_files(self, bull_file: str, bear_file: str):
        """Original method - load from static files"""
        print(f"[RESEARCH_MGR] Loading research files...")
        
        # Load bull thesis
        if os.path.exists(bull_file):
            with open(bull_file, 'r', encoding='utf-8') as f:
                self.research_inputs['bull_thesis'] = json.load(f)
            print(f"[RESEARCH_MGR] ✓ Bull thesis loaded ({bull_file})")
        else:
            print(f"[RESEARCH_MGR] ⚠  Bull thesis not found: {bull_file}")
        
        # Load bear thesis
        if os.path.exists(bear_file):
            with open(bear_file, 'r', encoding='utf-8') as f:
                self.research_inputs['bear_thesis'] = json.load(f)
            print(f"[RESEARCH_MGR] ✓ Bear thesis loaded ({bear_file})")
        else:
            print(f"[RESEARCH_MGR] ⚠  Bear thesis not found: {bear_file}")
    
    def synthesize(self, load_risk_evals: bool = True) -> tuple:
        """
        Main synthesis workflow - works with both static and debate modes
        Returns (report, synthesis_data)
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"RESEARCH SYNTHESIS: {self.ticker}")
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
        
        # Save synthesis.json for downstream components
        synthesis_output = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'bull_thesis': self.research_inputs['bull_thesis'].get('final_argument', 
                          self.research_inputs['bull_thesis'].get('core_thesis', '')),
            'bear_thesis': self.research_inputs['bear_thesis'].get('final_argument',
                          self.research_inputs['bear_thesis'].get('core_thesis', '')),
            'consensus': consensus,
            'probabilities': probabilities,
            'conclusion': conclusion,
            'recommendation': conclusion['recommendation'],
            'confidence': conclusion['confidence']
        }
        
        elapsed = time.time() - start_time
        print(f"\n[RESEARCH_MGR] ✓ Synthesis complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return report, synthesis_data, synthesis_output
    
    # Keep all other existing methods unchanged
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
                    print(f"[RESEARCH_MGR] ⚠  Error loading {risk_type}: {e}")
        
        print(f"[RESEARCH_MGR] ✓ Loaded {loaded_count}/3 risk evaluations")
    
    def format_debate_for_analysis(self) -> str:
        """Format bull/bear debate for LLM analysis"""
        bull = self.research_inputs.get('bull_thesis', {})
        bear = self.research_inputs.get('bear_thesis', {})
        
        debate = f"""# Investment Debate for {self.ticker}

## BULL ANALYST POSITION

*Core Thesis:*
{bull.get('core_thesis', 'Not available')}

*Key Opportunities:*
"""
        # Add bull opportunities
        for category, opps in bull.get('opportunities', {}).items():
            if opps:
                debate += f"\n### {category.title()}:\n"
                for opp in opps[:3]:
                    debate += f"- {opp}\n"
        
        # Add bull catalysts
        debate += "\n*Upside Catalysts:*\n"
        for catalyst in bull.get('catalysts', [])[:5]:
            debate += f"- {catalyst.get('description', '')} ({catalyst.get('timeline', 'TBD')})\n"
        
        # Add bull R/R
        rr = bull.get('risk_reward', {})
        debate += f"\n*Risk/Reward:* {rr.get('upside_potential', 'N/A')} upside, {rr.get('downside_risk', 'N/A')} downside"
        debate += f" (Ratio: {rr.get('reward_risk_ratio', 0):.1f}:1)\n"
        
        # If debate history exists (from deep/research mode)
        if 'debate_history' in bull:
            debate += "\n*Debate Arguments:*\n"
            for entry in bull['debate_history']:
                debate += f"\nRound {entry['round']}: {entry['argument'][:400]}...\n"
        
        debate += "\n" + "="*70 + "\n\n"
        
        debate += "## BEAR ANALYST POSITION\n\n"
        
        debate += f"*Core Thesis:*\n{bear.get('core_thesis', 'Not available')}\n\n"
        
        # Add bear risks
        debate += "*Risk Factors:*\n"
        for category, risks in bear.get('risks', {}).items():
            if risks:
                debate += f"\n### {category.title()}:\n"
                for risk in risks[:3]:
                    debate += f"- {risk}\n"
        
        # Add bear triggers
        debate += "\n*Downside Triggers:*\n"
        for trigger in bear.get('downside_triggers', [])[:5]:
            debate += f"- {trigger.get('description', '')} ({trigger.get('timeline', 'TBD')})\n"
        
        # Add bear risk assessment
        ra = bear.get('risk_assessment', {})
        debate += f"\n*Risk Assessment:* {ra.get('downside_risk', 'N/A')} downside, {ra.get('limited_upside', 'N/A')} upside"
        debate += f" (Risk Level: {ra.get('risk_level', 'N/A')})\n"
        
        # If debate history exists
        if 'debate_history' in bear:
            debate += "\n*Debate Arguments:*\n"
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
        """Generate comprehensive synthesis using LLM"""
        if not self.client:
            print("[RESEARCH_MGR] ⚠  No API key - using fallback")
            return self._create_fallback_report(consensus, probabilities, conclusion)
        
        try:
            print(f"[RESEARCH_MGR] Generating synthesis with {self.model}...")
            
            context = f"""# Research Synthesis for {self.ticker}

{debate_formatted}

{'='*70}

## Preliminary Analysis

*Probability Assessment:*
{json.dumps(probabilities, indent=2)}

*Consensus Summary:*
{json.dumps(consensus, indent=2)}

*Initial Conclusion:*
{json.dumps(conclusion, indent=2)}

{'='*70}

As Research Manager, provide your definitive investment decision. Evaluate both sides critically and make a clear recommendation supported by the strongest evidence."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.6,
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
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Mode: Fallback (LLM unavailable)

## Probability Assessment
"""
        for scenario in probabilities['scenarios']:
            report += f"\n*{scenario['name']}:* {scenario['probability']}\n"
            report += f"  Outcome: {scenario['outcome']}\n"
            report += f"  {scenario['description']}\n"
        
        report += f"""
## Consensus Analysis

*Recommendations:*
"""
        for source, rec in consensus['recommendations'].items():
            report += f"  - {source}: {rec}\n"
        
        if consensus['key_agreements']:
            report += f"\n*Agreement:* {consensus['key_agreements'][0]}\n"
        
        if consensus['key_conflicts']:
            report += f"*Conflict:* {consensus['key_conflicts'][0]}\n"
        
        report += f"""
## Final Conclusion

*Recommendation:* {conclusion['recommendation']}
*Confidence:* {conclusion['confidence']}
*Position Size:* {conclusion['position_size_range']}
*Time Horizon:* {conclusion['time_horizon']}

*Rationale:* {conclusion['rationale']}

*Expected Value:* {conclusion['expected_value']:+.1f}%

RESEARCH CONCLUSION: {conclusion['recommendation']} - Confidence: {conclusion['confidence']}
"""
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Research Manager - Orchestrates debate and synthesizes decision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operation Modes:
  auto    - Auto-detect: debate if no bull/bear files exist, static if they do
  static  - Load pre-generated bull/bear files (original behavior)
  debate  - Orchestrate live debate between bull and bear

Examples:
  # Auto mode (default) - works with your current orchestrator!
  python research_manager.py AAPL
  
  # Force static files
  python research_manager.py AAPL --mode static
  
  # Force debate mode
  python research_manager.py AAPL --mode debate --rounds 3
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker")
    
    # Mode selection with static default for backward compatibility
    parser.add_argument("--mode", choices=['auto', 'static', 'debate'], default='static',
                       help="Operation mode: auto-detect, static files, or live debate (default: static)")
    
    # Debate-specific options
    parser.add_argument("--rounds", type=int, default=3,
                       help="Number of debate rounds (for debate mode, default: 3)")
    parser.add_argument("--debate-mode", choices=['shallow', 'deep', 'research'], default='deep',
                       help="Debate depth (for debate mode, default: deep)")
    parser.add_argument("--discussion-file", default="../../outputs/discussion_points.json",
                       help="Discussion points file (for debate mode)")
    
    # Static mode options
    parser.add_argument("--bull-file", default="../../outputs/bull_thesis.json",
                       help="Bull thesis JSON file (for static mode)")
    parser.add_argument("--bear-file", default="../../outputs/bear_thesis.json",
                       help="Bear thesis JSON file (for static mode)")
    
    # Common options
    parser.add_argument("--skip-evaluations", action="store_true",
                       help="Skip loading risk evaluations")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--save-synthesis", default="../../outputs/synthesis.json",
                       help="Save synthesis JSON (default: ../../outputs/synthesis.json)")
    
    args = parser.parse_args()
    
    try:
        manager = ResearchManager(ticker=args.ticker, api_key=args.api_key, model=args.model)
        
        # SMART AUTO-DETECTION LOGIC
        if args.mode == 'auto':
            # Detect if called by master orchestrator
            # Master orch always passes --bull-file and --bear-file
            called_by_orchestrator = ('--bull-file' in sys.argv and '--bear-file' in sys.argv)
            
            # Check if bull/bear thesis files exist
            bull_exists = os.path.exists(args.bull_file)
            bear_exists = os.path.exists(args.bear_file)
            discussion_exists = os.path.exists(args.discussion_file)
            
            if called_by_orchestrator and not (bull_exists and bear_exists):
                # Called by orchestrator but files don't exist yet
                # This means Phase 2 failed or we should run debate
                if discussion_exists:
                    print(f"[MAIN] AUTO: Orchestrator call without bull/bear files - running DEBATE mode")
                    args.mode = 'debate'
                else:
                    print(f"[MAIN] AUTO: Missing prerequisites - need Phase 1 first")
                    sys.exit(1)
            elif bull_exists and bear_exists:
                # Files exist - use static mode (normal orchestrator flow)
                print(f"[MAIN] AUTO: Found bull/bear files, using STATIC mode")
                args.mode = 'static'
            elif discussion_exists:
                # No bull/bear but have discussion - use debate mode
                print(f"[MAIN] AUTO: No bull/bear files but found discussion, using DEBATE mode")
                args.mode = 'debate'
            else:
                # Fallback to static, will error appropriately
                print(f"[MAIN] AUTO: No files found, defaulting to STATIC mode")
                args.mode = 'static'
        
        if args.mode == 'debate':
            # NEW: Debate mode
            print(f"[MAIN] Running in DEBATE mode")
            
            # Load discussion points
            if not os.path.exists(args.discussion_file):
                # Try to find it in standard location
                alt_discussion = "../../outputs/discussion_points.json"
                if os.path.exists(alt_discussion):
                    args.discussion_file = alt_discussion
                    print(f"[MAIN] Found discussion points at {alt_discussion}")
                else:
                    print(f"\n❌ Error: Discussion points file not found: {args.discussion_file}")
                    print("Run Phase 1 analysts first to generate discussion points!")
                    sys.exit(1)
            
            with open(args.discussion_file, 'r', encoding='utf-8') as f:
                discussion_points = json.load(f)
            
            # Orchestrate the debate
            debate_result = manager.orchestrate_debate(
                discussion_points,
                rounds=args.rounds,
                mode=args.debate_mode
            )
            
            print(f"[MAIN] Debate complete, synthesizing...")
            
        else:
            # ORIGINAL: Static mode (or enhanced static with debate)
            print(f"[MAIN] Running in STATIC mode")
            
            # Load research files
            manager.load_research_files(args.bull_file, args.bear_file)
            
            # Check we have minimum data
            if not manager.research_inputs['bull_thesis'] or not manager.research_inputs['bear_thesis']:
                print("\n❌ Error: Both bull and bear theses required")
                print("Run bull_researcher.py and bear_researcher.py first!")
                sys.exit(1)
            
            # NEW: Check if we can enhance with debate
            bull_rounds = manager.research_inputs['bull_thesis'].get('rounds', 1)
            bear_rounds = manager.research_inputs['bear_thesis'].get('rounds', 1)
            
            if max(bull_rounds, bear_rounds) > 1:
                print(f"[MAIN] Detected multi-round capable theses ({max(bull_rounds, bear_rounds)} rounds)")
                print(f"[MAIN] Enhancing with debate interactions...")
                manager.enhance_static_with_debate()
            else:
                print(f"[MAIN] Single-round theses, proceeding without debate")
        
        # Run synthesis (works for both modes)
        report, synthesis_data, synthesis_output = manager.synthesize(
            load_risk_evals=not args.skip_evaluations
        )
        
        print(report)
        
        # Save outputs
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to {args.output}")
        
        # Always save synthesis.json for downstream components
        with open(args.save_synthesis, 'w', encoding='utf-8') as f:
            json.dump(synthesis_output, f, indent=2)
        print(f"✓ Synthesis data saved to {args.save_synthesis}")
        
        # Optionally save bull/bear theses from debate
        if args.mode == 'debate':
            bull_path = args.bull_file or "../../outputs/bull_thesis.json"
            bear_path = args.bear_file or "../../outputs/bear_thesis.json"
            
            with open(bull_path, 'w', encoding='utf-8') as f:
                json.dump(manager.research_inputs['bull_thesis'], f, indent=2)
            print(f"✓ Bull thesis saved to {bull_path}")
            
            with open(bear_path, 'w', encoding='utf-8') as f:
                json.dump(manager.research_inputs['bear_thesis'], f, indent=2)
            print(f"✓ Bear thesis saved to {bear_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()