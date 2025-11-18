"""
Fixed Master Orchestrator - Properly handles paths and encoding
Usage: python agents/orchestrators/master_orchestrator.py AAPL --run-all --research-mode deep
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # Set console code page to UTF-8
    os.system('chcp 65001 > nul')


class MasterOrchestrator:
    def __init__(self, ticker: str, portfolio_value: float = 100000, research_mode: str = 'shallow', research_rounds: int = 1):
        self.ticker = ticker.upper()
        self.portfolio_value = portfolio_value
        self.research_mode = research_mode
        
        # Map research mode to rounds if not specified
        if research_rounds == 1:
            if research_mode == 'deep':
                self.research_rounds = 3
            elif research_mode == 'research':
                self.research_rounds = 5
            else:
                self.research_rounds = 1
        else:
            self.research_rounds = research_rounds
        
        # Track execution
        self.execution_log = []
        self.phase_results = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
        
        # Setup paths
        self.setup_paths()
    
    def setup_paths(self):
        """Setup all paths properly"""
        # Find project root by looking for key directories
        current = Path.cwd()
        
        # Check if we're in orchestrators directory
        if current.name == 'orchestrators':
            self.project_root = current.parent.parent
        # Check if we're in agents directory
        elif current.name == 'agents':
            self.project_root = current.parent
        # Check if we have agents subdirectory (we're in project root)
        elif (current / 'agents').exists():
            self.project_root = current
        else:
            # Try to find project root by looking up
            temp = current
            while temp.parent != temp:
                if (temp / 'agents' / 'orchestrators').exists():
                    self.project_root = temp
                    break
                temp = temp.parent
            else:
                # Default to current directory
                self.project_root = current
        
        self.agents_root = self.project_root / "agents"
        self.outputs_path = self.project_root / "outputs"
        self.outputs_path.mkdir(exist_ok=True)
        
        # Define agent paths
        self.paths = {
            'orchestrators': self.agents_root / "orchestrators",
            'researcher': self.agents_root / "researcher",
            'managers': self.agents_root / "managers",
            'risk_management': self.agents_root / "risk_management"
        }
        
        print(f"[SETUP] Project root: {self.project_root}")
        print(f"[SETUP] Agents root: {self.agents_root}")
        print(f"[SETUP] Outputs: {self.outputs_path}")
        
        # Verify critical paths exist
        if not self.agents_root.exists():
            raise FileNotFoundError(f"Agents directory not found at {self.agents_root}")
        
        for name, path in self.paths.items():
            if not path.exists():
                print(f"[WARNING] {name} path not found: {path}")
    
    def log(self, message: str, phase: Optional[str] = None, status: str = 'INFO'):
        """Log execution progress"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'phase': phase,
            'message': message,
            'status': status
        }
        self.execution_log.append(log_entry)
        
        icons = {
            'ERROR': '[ERROR]',
            'SUCCESS': '[OK]',
            'RUNNING': '[RUN]',
            'INFO': '[INFO]'
        }
        icon = icons.get(status, '[*]')
        print(f"{icon} [{timestamp}] {message}")
    
    def run_command(self, cmd: List[str], cwd: Path, timeout: int = 120) -> tuple:
        """Execute command with proper encoding and Python interpreter"""
        try:
            # Use sys.executable instead of "python"
            if cmd[0] == "python":
                cmd[0] = sys.executable
            
            # Set environment for UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # Debug: Show actual command being run
            self.log(f"Running: {' '.join(cmd[:3])}...", status='INFO')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd),
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            # Show some output for debugging
            if result.stdout and len(result.stdout) > 0:
                preview = result.stdout[:200].replace('\n', ' ')
                self.log(f"Output preview: {preview}...", status='INFO')
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout}s", status='ERROR')
            return False, "", f"Timeout after {timeout}s"
        except FileNotFoundError as e:
            self.log(f"File not found: {e}", status='ERROR')
            return False, "", str(e)
        except Exception as e:
            self.log(f"Command error: {e}", status='ERROR')
            return False, "", str(e)
    
    def run_phase1_analysts(self) -> bool:
        """Phase 1: Run discussion hub"""
        self.log("Phase 1: Running Analysts (4)", "phase1", "RUNNING")
        
        script_path = self.paths['orchestrators'] / "discussion_hub.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "phase1", "ERROR")
            return False
        
        cmd = [
            sys.executable, 
            str(script_path),
            self.ticker,
            "--run-analysts",
            "--output", str(self.outputs_path / "discussion_points.json"),
            "--format", "json"
        ]
        
        success, stdout, stderr = self.run_command(
            cmd,
            cwd=self.paths['orchestrators'],
            timeout=180
        )
        
        if success:
            # Verify output file was created
            output_file = self.outputs_path / "discussion_points.json"
            if output_file.exists():
                self.log("Analysts complete - output file created", "phase1", "SUCCESS")
                self.phase_results['phase1'] = {'status': 'SUCCESS'}
                return True
            else:
                self.log("Analysts ran but no output file created", "phase1", "ERROR")
                self.phase_results['phase1'] = {'status': 'FAILED'}
                return False
        else:
            self.log(f"Analysts failed: {stderr[:200]}", "phase1", "ERROR")
            self.errors.append(f"Analysts: {stderr[:500]}")
            self.phase_results['phase1'] = {'status': 'FAILED'}
            return False
    
    def run_phase2_researchers(self) -> bool:
        """Phase 2: Run researchers"""
        self.log(f"Phase 2: Researchers ({self.research_mode}, {self.research_rounds} rounds)", "phase2", "RUNNING")
        
        # Check if discussion points exist
        discussion_file = self.outputs_path / "discussion_points.json"
        if not discussion_file.exists():
            self.log("Discussion points file not found", "phase2", "ERROR")
            return False
        
        # Bear researcher
        bear_script = self.paths['researcher'] / "bear_researcher.py"
        if not bear_script.exists():
            self.log(f"Bear script not found: {bear_script}", "phase2", "ERROR")
            return False
        
        bear_cmd = [
            sys.executable,
            str(bear_script),
            self.ticker,
            "--discussion-file", str(discussion_file),
            "--mode", self.research_mode,
            "--rounds", str(self.research_rounds),
            "--save-data", str(self.outputs_path / "bear_thesis.json")
        ]
        
        self.log("Running Bear Researcher...", "phase2", "INFO")
        success_bear, out_bear, err_bear = self.run_command(
            bear_cmd,
            cwd=self.paths['researcher'],
            timeout=300
        )
        
        # Bull researcher
        bull_script = self.paths['researcher'] / "bull_researcher.py"
        if not bull_script.exists():
            self.log(f"Bull script not found: {bull_script}", "phase2", "ERROR")
            return False
        
        bull_cmd = [
            sys.executable,
            str(bull_script),
            self.ticker,
            "--discussion-file", str(discussion_file),
            "--mode", self.research_mode,
            "--rounds", str(self.research_rounds),
            "--save-data", str(self.outputs_path / "bull_thesis.json")
        ]
        
        self.log("Running Bull Researcher...", "phase2", "INFO")
        success_bull, out_bull, err_bull = self.run_command(
            bull_cmd,
            cwd=self.paths['researcher'],
            timeout=300
        )
        
        # Check results
        bear_file = self.outputs_path / "bear_thesis.json"
        bull_file = self.outputs_path / "bull_thesis.json"
        
        if bear_file.exists() and bull_file.exists():
            self.log("Both researchers complete with output files", "phase2", "SUCCESS")
            self.phase_results['phase2'] = {'status': 'SUCCESS'}
            return True
        else:
            if not bear_file.exists():
                self.log("Bear thesis file not created", "phase2", "ERROR")
            if not bull_file.exists():
                self.log("Bull thesis file not created", "phase2", "ERROR")
            self.phase_results['phase2'] = {'status': 'PARTIAL'}
            return False
    
    def run_phase3_research_manager(self) -> bool:
        """Phase 3: Research Manager"""
        self.log("Phase 3: Research Manager", "phase3", "RUNNING")
        
        # Check prerequisites
        bear_file = self.outputs_path / "bear_thesis.json"
        bull_file = self.outputs_path / "bull_thesis.json"
        
        if not bear_file.exists() or not bull_file.exists():
            self.log("Missing bear or bull thesis files", "phase3", "ERROR")
            return False
        
        script_path = self.paths['managers'] / "research_manager.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "phase3", "ERROR")
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            self.ticker,
            "--bull-file", str(bull_file),
            "--bear-file", str(bear_file),
            "--save-synthesis", str(self.outputs_path / "research_synthesis.json")
        ]
        
        success, stdout, stderr = self.run_command(
            cmd,
            cwd=self.paths['managers'],
            timeout=120
        )
        
        output_file = self.outputs_path / "research_synthesis.json"
        if success and output_file.exists():
            self.log("Research Manager complete", "phase3", "SUCCESS")
            self.phase_results['phase3'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log(f"Research Manager failed", "phase3", "ERROR")
            if stderr:
                self.log(f"Error: {stderr[:200]}", "phase3", "ERROR")
            self.phase_results['phase3'] = {'status': 'FAILED'}
            return False
    
    def run_phase4_risk_team(self) -> bool:
        """Phase 4: Risk Team"""
        self.log("Phase 4: Risk Team (3 debators)", "phase4", "RUNNING")
        
        # Check prerequisites
        synthesis_file = self.outputs_path / "research_synthesis.json"
        if not synthesis_file.exists():
            self.log("Research synthesis file not found", "phase4", "ERROR")
            return False
        
        success_count = 0
        
        for analyst in ['aggressive', 'neutral', 'conservative']:
            script_path = self.paths['risk_management'] / f"{analyst}_debator.py"
            
            if not script_path.exists():
                self.log(f"Script not found: {script_path}", "phase4", "ERROR")
                continue
            
            cmd = [
                sys.executable,
                str(script_path),
                self.ticker,
                "--synthesis-file", str(synthesis_file),
                "--save-evaluation", str(self.outputs_path / f"{analyst}_eval.json")
            ]
            
            # Add optional files if they exist
            bear_file = self.outputs_path / "bear_thesis.json"
            bull_file = self.outputs_path / "bull_thesis.json"
            if bear_file.exists():
                cmd.extend(["--bear-file", str(bear_file)])
            if bull_file.exists():
                cmd.extend(["--bull-file", str(bull_file)])
            
            self.log(f"Running {analyst.capitalize()} debator...", "phase4", "INFO")
            success, stdout, stderr = self.run_command(
                cmd,
                cwd=self.paths['risk_management'],
                timeout=90
            )
            
            output_file = self.outputs_path / f"{analyst}_eval.json"
            if success and output_file.exists():
                success_count += 1
                self.log(f"  ✓ {analyst.capitalize()} complete", "phase4", "SUCCESS")
            else:
                self.log(f"  ✗ {analyst.capitalize()} failed", "phase4", "ERROR")
        
        if success_count >= 2:
            self.log(f"Risk team complete ({success_count}/3)", "phase4", "SUCCESS")
            self.phase_results['phase4'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Risk team failed (need at least 2/3)", "phase4", "ERROR")
            self.phase_results['phase4'] = {'status': 'FAILED'}
            return False
    
    def run_phase5_risk_manager(self) -> bool:
        """Phase 5: Risk Manager"""
        self.log("Phase 5: Risk Manager", "phase5", "RUNNING")
        
        synthesis_file = self.outputs_path / "research_synthesis.json"
        if not synthesis_file.exists():
            self.log("Research synthesis file not found", "phase5", "ERROR")
            return False
        
        script_path = self.paths['managers'] / "risk_manager.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "phase5", "ERROR")
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            self.ticker,
            "--synthesis-file", str(synthesis_file),
            "--portfolio-value", str(self.portfolio_value),
            "--save-decision", str(self.outputs_path / "risk_decision.json")
        ]
        
        success, stdout, stderr = self.run_command(
            cmd,
            cwd=self.paths['managers'],
            timeout=120
        )
        
        output_file = self.outputs_path / "risk_decision.json"
        if success and output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    decision = json.load(f)
                verdict = decision.get('verdict', '?')
                position = decision.get('final_position_dollars', 0)
                self.log(f"Final Decision: {verdict} (${position:,.0f})", "phase5", "SUCCESS")
            except:
                self.log("Risk Manager complete", "phase5", "SUCCESS")
            
            self.phase_results['phase5'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Risk Manager failed", "phase5", "ERROR")
            if stderr:
                self.log(f"Error: {stderr[:200]}", "phase5", "ERROR")
            self.phase_results['phase5'] = {'status': 'FAILED'}
            return False
    
    def run_phase6_game_theory(self) -> bool:
        """Phase 6: Game Theory Tournament"""
        self.log("Phase 6: Game Theory Tournament", "phase6", "RUNNING")
        
        script_path = self.paths['orchestrators'] / "game_theory_orchestrator.py"
        
        if not script_path.exists():
            self.log("Game theory script not available", "phase6", "INFO")
            self.phase_results['phase6'] = {'status': 'SKIPPED'}
            return True
        
        cmd = [
            sys.executable,
            str(script_path),
            self.ticker,
            "--outputs-dir", str(self.outputs_path)
        ]
        
        success, stdout, stderr = self.run_command(
            cmd,
            cwd=self.paths['orchestrators'],
            timeout=90
        )
        
        output_file = self.outputs_path / "game_theory_tournament.json"
        if success and output_file.exists():
            self.log("Game theory tournament complete", "phase6", "SUCCESS")
            self.phase_results['phase6'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Game theory failed", "phase6", "ERROR")
            if stderr:
                self.log(f"Error: {stderr[:200]}", "phase6", "ERROR")
            self.phase_results['phase6'] = {'status': 'FAILED'}
            return False
    
    def run_complete_workflow(self, include_game_theory: bool = True) -> Dict:
        """Run complete workflow"""
        self.start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print("MASTER ORCHESTRATOR - COMPLETE TRADING SYSTEM")
        print(f"{'='*80}")
        print(f"Ticker: {self.ticker}")
        print(f"Portfolio: ${self.portfolio_value:,.0f}")
        print(f"Research: {self.research_mode} ({self.research_rounds} rounds)")
        print(f"Python: {sys.executable}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Run phases sequentially
        phases = [
            (1, self.run_phase1_analysts, "Critical"),
            (2, self.run_phase2_researchers, "Critical"),
            (3, self.run_phase3_research_manager, "Critical"),
            (4, self.run_phase4_risk_team, "Important"),
            (5, self.run_phase5_risk_manager, "Critical"),
        ]
        
        if include_game_theory:
            phases.append((6, self.run_phase6_game_theory, "Optional"))
        
        for phase_num, phase_func, importance in phases:
            print(f"\n--- Phase {phase_num} ---")
            success = phase_func()
            
            # Stop on critical failures
            if not success and importance == "Critical":
                self.log(f"Stopping due to critical failure in Phase {phase_num}", status="ERROR")
                break
            
            # Add delay between phases
            time.sleep(1)
        
        self.end_time = datetime.now()
        
        # Print summary
        self.print_summary()
        self.save_logs()
        
        return self.phase_results
    
    def print_summary(self):
        """Print execution summary"""
        if not self.end_time:
            self.end_time = datetime.now()
            
        elapsed = (self.end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("EXECUTION SUMMARY")
        print(f"{'='*80}\n")
        print(f"Total Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        
        # Count successes
        success_count = sum(1 for r in self.phase_results.values() if r.get('status') == 'SUCCESS')
        total_count = len(self.phase_results)
        
        print(f"Phases Completed: {success_count}/{total_count}\n")
        
        print("Phase Results:")
        for phase, result in self.phase_results.items():
            status = result.get('status', 'UNKNOWN')
            icon = "✓" if status == 'SUCCESS' else "⚠" if status == 'PARTIAL' else "✗"
            print(f"  {icon} {phase}: {status}")
        
        # Show final decision if available
        try:
            decision_file = self.outputs_path / "risk_decision.json"
            if decision_file.exists():
                with open(decision_file, 'r', encoding='utf-8') as f:
                    decision = json.load(f)
                print(f"\nFINAL DECISION: {decision.get('verdict', 'N/A')}")
                print(f"Position Size: ${decision.get('final_position_dollars', 0):,.0f}")
        except:
            pass
        
        # Show errors if any
        if self.errors:
            print(f"\nErrors Encountered: {len(self.errors)}")
            for error in self.errors[:3]:  # Show first 3 errors
                print(f"  - {error[:100]}...")
        
        print(f"\n{'='*80}\n")
    
    def save_logs(self):
        """Save execution logs"""
        log_file = self.outputs_path / "execution_log.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'ticker': self.ticker,
                    'research_mode': self.research_mode,
                    'research_rounds': self.research_rounds,
                    'portfolio_value': self.portfolio_value,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': self.end_time.isoformat() if self.end_time else None,
                    'phases': self.phase_results,
                    'errors': self.errors,
                    'log': self.execution_log
                }, f, indent=2)
            print(f"Logs saved to: {log_file}")
        except Exception as e:
            print(f"Failed to save logs: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator - Complete Trading System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_orchestrator.py AAPL
  python master_orchestrator.py AAPL --run-all
  python master_orchestrator.py AAPL --run-all --research-mode deep
  python master_orchestrator.py AAPL --run-all --research-mode research --research-rounds 5
  
Research Modes:
  shallow  - Quick analysis (1 round, ~2 minutes)
  deep     - Detailed analysis (3 rounds, ~5 minutes)  
  research - Comprehensive analysis (5+ rounds, ~10 minutes)
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    parser.add_argument("--run-all", action="store_true", 
                       help="Include Phase 6 (Game Theory Tournament)")
    parser.add_argument("--research-mode", 
                       choices=['shallow', 'deep', 'research'], 
                       default='shallow',
                       help="Research depth (default: shallow)")
    parser.add_argument("--research-rounds", 
                       type=int, 
                       default=0,
                       help="Number of debate rounds (default: auto based on mode)")
    parser.add_argument("--portfolio-value", 
                       type=float, 
                       default=100000,
                       help="Portfolio value for position sizing (default: 100000)")
    
    args = parser.parse_args()
    
    # Auto-set rounds based on mode if not specified
    if args.research_rounds == 0:
        if args.research_mode == 'deep':
            args.research_rounds = 3
        elif args.research_mode == 'research':
            args.research_rounds = 5
        else:
            args.research_rounds = 1
    
    print(f"Starting orchestrator from: {os.getcwd()}")
    
    try:
        orchestrator = MasterOrchestrator(
            ticker=args.ticker,
            portfolio_value=args.portfolio_value,
            research_mode=args.research_mode,
            research_rounds=args.research_rounds
        )
        
        results = orchestrator.run_complete_workflow(
            include_game_theory=args.run_all
        )
        
        # Exit with appropriate code
        success_count = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        if success_count == len(results):
            sys.exit(0)  # All successful
        else:
            sys.exit(1)  # Some failures
        
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()