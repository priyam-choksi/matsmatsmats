"""
Master Trading System CLI - Interactive Control Center
Runs the complete multi-agent trading system with game theory integration

Usage: 
  python master_cli.py AAPL --mode quick
  python master_cli.py AAPL --mode full --research-depth deep
  python master_cli.py AAPL --mode game-theory
  python master_cli.py AAPL --interactive
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class MasterCLI:
    def __init__(self, ticker: str, portfolio_value: float = 100000):
        self.ticker = ticker.upper()
        self.portfolio_value = portfolio_value
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Track execution status
        self.completed_phases = []
        self.errors = []
        self.start_time = None
        
        # Define available components
        self.components = {
            'analysts': {
                'path': 'agents/orchestrators',
                'script': 'discussion_hub.py',
                'output': 'discussion_points.json',
                'description': 'Run 4 analysts (technical, fundamental, news, macro)'
            },
            'researchers': {
                'bull': {
                    'path': 'agents/researcher',
                    'script': 'bull_researcher.py',
                    'output': 'bull_thesis.json',
                    'description': 'Build bullish investment case'
                },
                'bear': {
                    'path': 'agents/researcher',
                    'script': 'bear_researcher.py',
                    'output': 'bear_thesis.json',
                    'description': 'Build bearish investment case'
                }
            },
            'research_manager': {
                'path': 'agents/managers',
                'script': 'research_manager.py',
                'output': 'research_synthesis.json',
                'description': 'Synthesize bull vs bear debate'
            },
            'risk_team': {
                'aggressive': {
                    'path': 'agents/risk_management',
                    'script': 'aggressive_debator.py',
                    'output': 'aggressive_eval.json'
                },
                'neutral': {
                    'path': 'agents/risk_management',
                    'script': 'neutral_debator.py',
                    'output': 'neutral_eval.json'
                },
                'conservative': {
                    'path': 'agents/risk_management',
                    'script': 'conservative_debator.py',
                    'output': 'conservative_eval.json'
                }
            },
            'risk_manager': {
                'path': 'agents/managers',
                'script': 'risk_manager.py',
                'output': 'risk_decision.json',
                'description': 'Make final trading decision'
            },
            'game_theory': {
                'path': 'agents/game_theory',
                'script': 'tournament_engine.py',
                'output': 'tournament_results.json',
                'description': 'Run game theory tournament'
            }
        }
    
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}\n")
    
    def print_phase(self, phase_name: str, status: str = "RUNNING"):
        """Print phase status"""
        icons = {
            'RUNNING': '⚙️',
            'SUCCESS': '✅',
            'ERROR': '❌',
            'SKIPPED': '⏭️'
        }
        icon = icons.get(status, '•')
        print(f"{icon} {phase_name:50} [{status}]")
    
    def run_command(self, cmd: List[str], cwd: Optional[str] = None, timeout: int = 120) -> tuple:
        """Execute command and return (success, output, error)"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                encoding='utf-8',
                errors='replace'
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    def run_analysts(self) -> bool:
        """Phase 1: Run discussion hub (4 analysts)"""
        self.print_phase("Phase 1: Analyst Team", "RUNNING")
        
        cmd = [
            "python", "discussion_hub.py", self.ticker,
            "--run-analysts",
            "--output", f"../../outputs/discussion_points.json",
            "--format", "json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, cwd="agents/orchestrators", timeout=180)
        
        if success:
            self.print_phase("Phase 1: Analyst Team", "SUCCESS")
            self.completed_phases.append('analysts')
            return True
        else:
            self.print_phase("Phase 1: Analyst Team", "ERROR")
            self.errors.append(f"Analysts failed: {stderr[:200]}")
            return False
    
    def run_researchers(self, mode: str = 'shallow', rounds: int = 1) -> bool:
        """Phase 2: Run bull and bear researchers"""
        self.print_phase(f"Phase 2: Researchers ({mode} mode)", "RUNNING")
        
        # Run bear researcher
        bear_cmd = [
            "python", "bear_researcher.py", self.ticker,
            "--mode", mode,
            "--rounds", str(rounds),
            "--save-data", "../../outputs/bear_thesis.json"
        ]
        
        success_bear, _, err_bear = self.run_command(bear_cmd, cwd="agents/researcher")
        
        # Run bull researcher
        bull_cmd = [
            "python", "bull_researcher.py", self.ticker,
            "--mode", mode,
            "--rounds", str(rounds),
            "--save-data", "../../outputs/bull_thesis.json"
        ]
        
        success_bull, _, err_bull = self.run_command(bull_cmd, cwd="agents/researcher")
        
        if success_bear and success_bull:
            self.print_phase(f"Phase 2: Researchers ({mode} mode)", "SUCCESS")
            self.completed_phases.append('researchers')
            return True
        else:
            self.print_phase(f"Phase 2: Researchers ({mode} mode)", "ERROR")
            if not success_bear:
                self.errors.append(f"Bear researcher failed: {err_bear[:100]}")
            if not success_bull:
                self.errors.append(f"Bull researcher failed: {err_bull[:100]}")
            return False
    
    def run_research_manager(self) -> bool:
        """Phase 3: Run research manager"""
        self.print_phase("Phase 3: Research Manager", "RUNNING")
        
        cmd = [
            "python", "research_manager.py", self.ticker,
            "--save-synthesis", "../../outputs/research_synthesis.json"
        ]
        
        success, _, stderr = self.run_command(cmd, cwd="agents/managers")
        
        if success:
            self.print_phase("Phase 3: Research Manager", "SUCCESS")
            self.completed_phases.append('research_manager')
            return True
        else:
            self.print_phase("Phase 3: Research Manager", "ERROR")
            self.errors.append(f"Research Manager failed: {stderr[:200]}")
            return False
    
    def run_risk_team(self) -> bool:
        """Phase 4: Run 3 risk debators"""
        self.print_phase("Phase 4: Risk Debators (3)", "RUNNING")
        
        risk_analysts = ['aggressive', 'neutral', 'conservative']
        success_count = 0
        
        for analyst in risk_analysts:
            cmd = [
                "python", f"{analyst}_debator.py", self.ticker,
                "--save-evaluation", f"../../outputs/{analyst}_eval.json"
            ]
            
            success, _, stderr = self.run_command(cmd, cwd="agents/risk_management")
            
            if success:
                success_count += 1
                print(f"  ✓ {analyst.capitalize()} complete")
            else:
                print(f"  ❌ {analyst.capitalize()} failed")
                self.errors.append(f"{analyst} failed: {stderr[:100]}")
        
        if success_count >= 2:  # At least 2 out of 3
            self.print_phase("Phase 4: Risk Debators (3)", "SUCCESS")
            self.completed_phases.append('risk_team')
            return True
        else:
            self.print_phase("Phase 4: Risk Debators (3)", "ERROR")
            return False
    
    def run_risk_manager(self) -> bool:
        """Phase 5: Run risk manager (final decision)"""
        self.print_phase("Phase 5: Risk Manager (Final Decision)", "RUNNING")
        
        cmd = [
            "python", "risk_manager.py", self.ticker,
            "--portfolio-value", str(self.portfolio_value),
            "--save-decision", "../../outputs/risk_decision.json",
            "--output", "../../outputs/final_decision.txt"
        ]
        
        success, stdout, stderr = self.run_command(cmd, cwd="agents/managers")
        
        if success:
            self.print_phase("Phase 5: Risk Manager (Final Decision)", "SUCCESS")
            self.completed_phases.append('risk_manager')
            
            # Try to extract verdict
            try:
                with open("outputs/risk_decision.json", 'r') as f:
                    decision = json.load(f)
                    verdict = decision.get('verdict', 'UNKNOWN')
                    position = decision.get('final_position_dollars', 0)
                    print(f"\n  📊 VERDICT: {verdict} - Position: ${position:,.0f}")
            except:
                pass
            
            return True
        else:
            self.print_phase("Phase 5: Risk Manager (Final Decision)", "ERROR")
            self.errors.append(f"Risk Manager failed: {stderr[:200]}")
            return False
    
    def run_game_theory(self) -> bool:
        """Phase 6: Run game theory tournament"""
        self.print_phase("Phase 6: Game Theory Tournament", "RUNNING")
        
        # Check if game theory orchestrator exists
        gt_script = Path("agents/orchestrators/game_theory_orchestrator.py")
        
        if not gt_script.exists():
            print("  ⚠️  Game theory orchestrator not found")
            print("  💡 Skipping game theory (optional component)")
            self.print_phase("Phase 6: Game Theory Tournament", "SKIPPED")
            return True
        
        cmd = [
            "python", "game_theory_orchestrator.py", self.ticker
        ]
        
        success, _, stderr = self.run_command(cmd, cwd="agents/orchestrators", timeout=60)
        
        if success:
            self.print_phase("Phase 6: Game Theory Tournament", "SUCCESS")
            self.completed_phases.append('game_theory')
            return True
        else:
            print(f"  ⚠️  Game theory failed: {stderr[:100]}")
            self.print_phase("Phase 6: Game Theory Tournament", "ERROR")
            return False
    
    def run_quick_mode(self) -> bool:
        """Quick Mode: Analysts → Researchers (shallow) → Research Manager → Risk Manager"""
        self.print_header(f"QUICK MODE: {self.ticker}")
        print("Fast analysis with shallow research depth (~2-3 minutes)\n")
        
        return (
            self.run_analysts() and
            self.run_researchers(mode='shallow') and
            self.run_research_manager() and
            self.run_risk_team() and
            self.run_risk_manager()
        )
    
    def run_full_mode(self, research_depth: str = 'deep', research_rounds: int = 3) -> bool:
        """Full Mode: Complete pipeline with deep research"""
        self.print_header(f"FULL MODE: {self.ticker}")
        print(f"Complete analysis with {research_depth} research ({research_rounds} rounds) (~5-10 minutes)\n")
        
        return (
            self.run_analysts() and
            self.run_researchers(mode=research_depth, rounds=research_rounds) and
            self.run_research_manager() and
            self.run_risk_team() and
            self.run_risk_manager() and
            self.run_game_theory()
        )
    
    def run_game_theory_only(self) -> bool:
        """Game Theory Only: Requires existing risk decision"""
        self.print_header(f"GAME THEORY MODE: {self.ticker}")
        print("Running tournament on existing decision\n")
        
        # Check if risk_decision.json exists
        if not (self.outputs_dir / "risk_decision.json").exists():
            print("❌ Error: risk_decision.json not found")
            print("💡 Run --mode full first to generate base decision")
            return False
        
        return self.run_game_theory()
    
    def run_custom_pipeline(self, phases: List[str], research_depth: str = 'shallow') -> bool:
        """Custom: Run specific phases only"""
        self.print_header(f"CUSTOM MODE: {self.ticker}")
        print(f"Running phases: {', '.join(phases)}\n")
        
        phase_map = {
            'analysts': self.run_analysts,
            'researchers': lambda: self.run_researchers(mode=research_depth),
            'research_manager': self.run_research_manager,
            'risk_team': self.run_risk_team,
            'risk_manager': self.run_risk_manager,
            'game_theory': self.run_game_theory
        }
        
        for phase in phases:
            if phase in phase_map:
                if not phase_map[phase]():
                    print(f"\n⚠️  Phase '{phase}' failed, continuing...")
            else:
                print(f"⚠️  Unknown phase: {phase}")
        
        return True
    
    def interactive_mode(self):
        """Interactive: User chooses what to run step-by-step"""
        self.print_header(f"INTERACTIVE MODE: {self.ticker}")
        
        print("""
What would you like to do?

1. Run Analysts (4 agents)
2. Run Researchers (Bull & Bear)
   - Shallow mode (fast)
   - Deep mode (debate)
   - Research mode (comprehensive)
3. Run Research Manager (synthesize)
4. Run Risk Team (3 debators)
5. Run Risk Manager (final decision)
6. Run Game Theory Tournament
7. Run Everything (Quick Mode)
8. Run Everything (Full Mode)
9. View Results
0. Exit

""")
        
        while True:
            try:
                choice = input(f"\nEnter choice [0-9]: ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye!")
                    break
                elif choice == '1':
                    self.run_analysts()
                elif choice == '2':
                    depth = input("  Research depth [shallow/deep/research]: ").strip() or 'shallow'
                    rounds = int(input("  Number of rounds [1-5]: ").strip() or '1')
                    self.run_researchers(mode=depth, rounds=rounds)
                elif choice == '3':
                    self.run_research_manager()
                elif choice == '4':
                    self.run_risk_team()
                elif choice == '5':
                    self.run_risk_manager()
                elif choice == '6':
                    self.run_game_theory()
                elif choice == '7':
                    self.run_quick_mode()
                    break
                elif choice == '8':
                    depth = input("  Research depth [shallow/deep/research]: ").strip() or 'deep'
                    rounds = int(input("  Number of rounds [1-5]: ").strip() or '3')
                    self.run_full_mode(research_depth=depth, research_rounds=rounds)
                    break
                elif choice == '9':
                    self.view_results()
                else:
                    print("Invalid choice. Try again.")
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def view_results(self):
        """Display summary of generated files"""
        print(f"\n{'='*70}")
        print("GENERATED OUTPUTS")
        print(f"{'='*70}\n")
        
        output_files = [
            ('discussion_points.json', 'Analyst consensus'),
            ('bull_thesis.json', 'Bull research'),
            ('bear_thesis.json', 'Bear research'),
            ('research_synthesis.json', 'Research synthesis'),
            ('aggressive_eval.json', 'Aggressive risk eval'),
            ('neutral_eval.json', 'Neutral risk eval'),
            ('conservative_eval.json', 'Conservative risk eval'),
            ('risk_decision.json', 'Final risk decision'),
            ('tournament_results.json', 'Game theory results')
        ]
        
        for filename, description in output_files:
            filepath = self.outputs_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"✅ {filename:30} ({size:,} bytes) - {description}")
                
                # Show key info
                if filename == 'risk_decision.json':
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        print(f"   → Verdict: {data.get('verdict', 'N/A')}")
                        print(f"   → Position: ${data.get('final_position_dollars', 0):,.0f}")
                    except:
                        pass
            else:
                print(f"⚪ {filename:30} - Not generated yet")
        
        print(f"\n{'='*70}\n")
    
    def generate_summary_report(self):
        """Generate summary of entire run"""
        elapsed = (datetime.now() - self.start_time).seconds if self.start_time else 0
        
        print(f"\n{'='*80}")
        print(f"EXECUTION SUMMARY: {self.ticker}")
        print(f"{'='*80}\n")
        
        print(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        print(f"Total Time: {elapsed}s ({elapsed/60:.1f} minutes)")
        print(f"\nCompleted Phases: {len(self.completed_phases)}")
        for phase in self.completed_phases:
            print(f"  ✅ {phase}")
        
        if self.errors:
            print(f"\nErrors Encountered: {len(self.errors)}")
            for error in self.errors[:3]:
                print(f"  ❌ {error}")
        
        # Try to show final decision
        try:
            decision_file = self.outputs_dir / "risk_decision.json"
            if decision_file.exists():
                with open(decision_file, 'r') as f:
                    decision = json.load(f)
                
                print(f"\n{'='*80}")
                print("FINAL DECISION")
                print(f"{'='*80}")
                print(f"Verdict: {decision.get('verdict', 'N/A')}")
                print(f"Position: ${decision.get('final_position_dollars', 0):,.0f} ({decision.get('final_position_pct', 0)*100:.1f}%)")
                print(f"Confidence: {decision.get('confidence', 'N/A')}")
                print(f"\nReasoning:")
                for reason in decision.get('reasoning', [])[:3]:
                    print(f"  • {reason}")
        except:
            pass
        
        print(f"\n{'='*80}")
        print(f"All outputs saved in: {self.outputs_dir.absolute()}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master Trading System CLI - Interactive Control Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXECUTION MODES:

1. QUICK MODE (Recommended for daily use)
   python master_cli.py AAPL --mode quick
   • Runs: Analysts → Researchers (shallow) → Risk Team → Final Decision
   • Time: ~2-3 minutes
   • Cost: ~10-15k tokens

2. FULL MODE (Comprehensive analysis)
   python master_cli.py AAPL --mode full --research-depth deep --research-rounds 3
   • Runs: Everything including multi-round debate
   • Time: ~5-10 minutes
   • Cost: ~30-50k tokens

3. GAME THEORY MODE (Requires existing decision)
   python master_cli.py AAPL --mode game-theory
   • Runs: Tournament on existing risk_decision.json
   • Time: ~30 seconds
   • Cost: ~5k tokens

4. CUSTOM MODE (Pick specific phases)
   python master_cli.py AAPL --phases analysts researchers risk_manager
   • Runs: Only specified components
   • Time: Varies
   • Cost: Varies

5. INTERACTIVE MODE (Step-by-step control)
   python master_cli.py AAPL --interactive
   • Choose: Each phase manually with options
   • Time: Your pace
   • Cost: What you choose

EXAMPLES:
  # Quick daily analysis
  python master_cli.py AAPL --mode quick
  
  # Deep research for important decision
  python master_cli.py AAPL --mode full --research-depth research --research-rounds 5
  
  # Just run analysts and researchers
  python master_cli.py AAPL --phases analysts researchers research_manager
  
  # Interactive control
  python master_cli.py AAPL --interactive
  
  # Just game theory on existing decision
  python master_cli.py AAPL --mode game-theory
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, MSFT)")
    
    # Mode selection
    parser.add_argument("--mode", choices=['quick', 'full', 'game-theory', 'custom'],
                       help="Execution mode")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive step-by-step mode")
    
    # Research depth options
    parser.add_argument("--research-depth", choices=['shallow', 'deep', 'research'],
                       default='shallow',
                       help="Research depth for bull/bear (default: shallow)")
    parser.add_argument("--research-rounds", type=int, default=3,
                       help="Debate rounds for deep/research mode (default: 3)")
    
    # Custom phase selection
    parser.add_argument("--phases", nargs="+",
                       choices=['analysts', 'researchers', 'research_manager', 'risk_team', 'risk_manager', 'game_theory'],
                       help="Specific phases to run (custom mode)")
    
    # Portfolio settings
    parser.add_argument("--portfolio-value", type=float, default=100000,
                       help="Portfolio value (default: $100,000)")
    
    # Utility
    parser.add_argument("--view-results", action="store_true",
                       help="View existing results without running")
    
    args = parser.parse_args()
    
    try:
        cli = MasterCLI(ticker=args.ticker, portfolio_value=args.portfolio_value)
        cli.start_time = datetime.now()
        
        # View results only
        if args.view_results:
            cli.view_results()
            sys.exit(0)
        
        # Interactive mode
        if args.interactive:
            cli.interactive_mode()
        
        # Quick mode
        elif args.mode == 'quick':
            cli.run_quick_mode()
        
        # Full mode
        elif args.mode == 'full':
            cli.run_full_mode(
                research_depth=args.research_depth,
                research_rounds=args.research_rounds
            )
        
        # Game theory only
        elif args.mode == 'game-theory':
            cli.run_game_theory_only()
        
        # Custom phases
        elif args.mode == 'custom' and args.phases:
            cli.run_custom_pipeline(args.phases, args.research_depth)
        
        else:
            print("❌ No mode specified. Use --mode or --interactive")
            print("💡 Try: python master_cli.py AAPL --interactive")
            sys.exit(1)
        
        # Generate summary
        cli.generate_summary_report()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()