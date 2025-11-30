"""
Master Orchestrator - Complete Trading System Pipeline
Coordinates all phases: Analysts → Researchers → Debate → Risk Team → Decision

MODIFIED: Now supports historical backtesting via analysis_date parameter

Usage: 
  python master_orchestrator.py AAPL
  python master_orchestrator.py AAPL --research-mode deep
  python master_orchestrator.py AAPL --research-mode research --research-rounds 5
  python master_orchestrator.py AAPL --analysis-date 2024-06-15

Research Modes:
  shallow  - Quick analysis, no debate (~2 minutes)
  deep     - 3 debate rounds (~5 minutes)
  research - 5 debate rounds (~8 minutes)

Historical Backtesting:
  --analysis-date YYYY-MM-DD  - Analyze using data from specified date
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
    
# Find .env in project root (2 levels up from orchestration folder)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Goes up: orchestration → src → TradingAgent
env_path = project_root / '.env'

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
    os.system('chcp 65001 > nul')


class MasterOrchestrator:
    def __init__(self, ticker: str, portfolio_value: float = 100000, 
                 research_mode: str = 'shallow', research_rounds: int = 0,
                 analysis_date: Optional[str] = None):  # <-- NEW PARAMETER
        self.ticker = ticker.upper()
        self.portfolio_value = portfolio_value
        self.research_mode = research_mode
        self.analysis_date = analysis_date  # <-- NEW: Store analysis date
        self.market_context = None  # <-- NEW: Will hold full context if loaded
        
        # Map research mode to debate rounds if not explicitly specified
        if research_rounds == 0:
            if research_mode == 'deep':
                self.research_rounds = 3
            elif research_mode == 'research':
                self.research_rounds = 5
            else:  # shallow
                self.research_rounds = 0  # No debate for shallow mode
        else:
            self.research_rounds = research_rounds
        
        # Execution tracking
        self.execution_log = []
        self.phase_results = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
        
        # Setup paths
        self.setup_paths()
    
    def setup_paths(self):
        """Setup all paths properly"""
        current = Path.cwd()
        
        # Find project root
        if current.name == 'orchestrators':
            self.project_root = current.parent.parent
        elif current.name == 'agents':
            self.project_root = current.parent
        elif (current / 'agents').exists():
            self.project_root = current
        else:
            temp = current
            while temp.parent != temp:
                if (temp / 'agents' / 'orchestrators').exists():
                    self.project_root = temp
                    break
                temp = temp.parent
            else:
                self.project_root = current
        
        self.agents_root = self.project_root / "agents"
        self.outputs_path = self.project_root / "outputs"
        self.outputs_path.mkdir(exist_ok=True)
        
        # Agent paths
        self.paths = {
            'orchestrators': self.agents_root / "orchestrators",
            'researcher': self.agents_root / "researcher",
            'managers': self.agents_root / "managers",
            'risk_management': self.agents_root / "risk_management"
        }
        
        print(f"[SETUP] Project root: {self.project_root}")
        print(f"[SETUP] Outputs: {self.outputs_path}")
    
    # ============================================================
    # NEW METHOD: Load market context for historical backtesting
    # ============================================================
    def load_market_context(self):
        """
        Load market context if available (for batch runs).
        This is written by batch_collect_all.py before running the orchestrator.
        """
        context_file = self.outputs_path / "market_context.json"
        if context_file.exists():
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    context = json.load(f)
                
                # Extract analysis date from context (only if not already set via CLI)
                if not self.analysis_date:
                    self.analysis_date = context.get('analysis_date')
                self.market_context = context
                
                self.log(f"Loaded market context: {self.analysis_date}", "setup", "INFO")
                return context
            except Exception as e:
                self.log(f"Failed to load market context: {e}", "setup", "ERROR")
        return None
    
    def log(self, message: str, phase: Optional[str] = None, status: str = 'INFO'):
        """Log execution progress"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.execution_log.append({
            'timestamp': timestamp,
            'phase': phase,
            'message': message,
            'status': status
        })
        
        icons = {'ERROR': '[ERROR]', 'SUCCESS': '[OK]', 'RUNNING': '[RUN]', 'INFO': '[INFO]'}
        print(f"{icons.get(status, '[*]')} [{timestamp}] {message}")
    
    def run_command(self, cmd: List[str], cwd: Path, timeout: int = 120) -> tuple:
        """Execute command with proper encoding"""
        try:
            if cmd[0] == "python":
                cmd[0] = sys.executable
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
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
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    def run_phase1_analysts(self) -> bool:
        """Phase 1: Run analyst discussion hub"""
        self.log("Phase 1: Running Analysts (4 specialists)", "phase1", "RUNNING")
        
        # NEW: Log if using historical mode
        if self.analysis_date:
            self.log(f"Historical mode: {self.analysis_date}", "phase1", "INFO")
        
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
        
        # ============================================================
        # NEW: Pass analysis date to discussion hub
        # ============================================================
        if self.analysis_date:
            cmd.extend(["--analysis-date", self.analysis_date])
        
        success, stdout, stderr = self.run_command(cmd, self.paths['orchestrators'], timeout=180)
        
        output_file = self.outputs_path / "discussion_points.json"
        if success and output_file.exists():
            self.log("Analysts complete", "phase1", "SUCCESS")
            self.phase_results['phase1'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log(f"Analysts failed: {stderr[:200]}", "phase1", "ERROR")
            self.phase_results['phase1'] = {'status': 'FAILED'}
            return False
    
    def run_phase2_researchers(self) -> bool:
        """Phase 2: Run bull and bear researchers (single-pass analysis)"""
        self.log("Phase 2: Bull & Bear Researchers", "phase2", "RUNNING")
        
        discussion_file = self.outputs_path / "discussion_points.json"
        if not discussion_file.exists():
            self.log("Discussion points not found", "phase2", "ERROR")
            return False
        
        # Run Bull Researcher
        bull_script = self.paths['researcher'] / "bull_researcher.py"
        if bull_script.exists():
            self.log("Running Bull Researcher...", "phase2", "INFO")
            cmd = [
                sys.executable,
                str(bull_script),
                self.ticker,
                "--discussion-file", str(discussion_file),
                "--save-data", str(self.outputs_path / "bull_thesis.json")
            ]
            
            # NEW: Pass analysis date
            if self.analysis_date:
                cmd.extend(["--analysis-date", self.analysis_date])
            
            success, _, stderr = self.run_command(cmd, self.paths['researcher'], timeout=120)
            if not success:
                self.log(f"Bull researcher error: {stderr[:100]}", "phase2", "ERROR")
        else:
            self.log(f"Bull script not found: {bull_script}", "phase2", "ERROR")
        
        # Run Bear Researcher
        bear_script = self.paths['researcher'] / "bear_researcher.py"
        if bear_script.exists():
            self.log("Running Bear Researcher...", "phase2", "INFO")
            cmd = [
                sys.executable,
                str(bear_script),
                self.ticker,
                "--discussion-file", str(discussion_file),
                "--save-data", str(self.outputs_path / "bear_thesis.json")
            ]
            
            # NEW: Pass analysis date
            if self.analysis_date:
                cmd.extend(["--analysis-date", self.analysis_date])
            
            success, _, stderr = self.run_command(cmd, self.paths['researcher'], timeout=120)
            if not success:
                self.log(f"Bear researcher error: {stderr[:100]}", "phase2", "ERROR")
        else:
            self.log(f"Bear script not found: {bear_script}", "phase2", "ERROR")
        
        # Check results
        bull_file = self.outputs_path / "bull_thesis.json"
        bear_file = self.outputs_path / "bear_thesis.json"
        
        if bull_file.exists() and bear_file.exists():
            self.log("Both researchers complete", "phase2", "SUCCESS")
            self.phase_results['phase2'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("One or both researchers failed", "phase2", "ERROR")
            self.phase_results['phase2'] = {'status': 'PARTIAL' if bull_file.exists() or bear_file.exists() else 'FAILED'}
            return False
    
    def run_phase3_research_manager(self) -> bool:
        """Phase 3: Research Manager - Debate & Synthesis"""
        debate_desc = f"{self.research_rounds} debate rounds" if self.research_rounds > 0 else "direct synthesis"
        self.log(f"Phase 3: Research Manager ({debate_desc})", "phase3", "RUNNING")
        
        bull_file = self.outputs_path / "bull_thesis.json"
        bear_file = self.outputs_path / "bear_thesis.json"
        
        if not bull_file.exists() or not bear_file.exists():
            self.log("Missing bull or bear thesis files", "phase3", "ERROR")
            return False
        
        script_path = self.paths['managers'] / "research_manager.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "phase3", "ERROR")
            return False
        
        # Build command - pass debate-rounds to research manager
        cmd = [
            sys.executable,
            str(script_path),
            self.ticker,
            "--bull-file", str(bull_file),
            "--bear-file", str(bear_file),
            "--debate-rounds", str(self.research_rounds),  # KEY: Route rounds here
            "--save-synthesis", str(self.outputs_path / "research_synthesis.json")
        ]
        
        # NEW: Pass analysis date
        if self.analysis_date:
            cmd.extend(["--analysis-date", self.analysis_date])
        
        # Adjust timeout based on debate rounds
        timeout = 120 + (self.research_rounds * 45)  # ~45s per debate round
        
        success, stdout, stderr = self.run_command(cmd, self.paths['managers'], timeout=timeout)
        
        output_file = self.outputs_path / "research_synthesis.json"
        if success and output_file.exists():
            # Log debate info if available
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    synthesis = json.load(f)
                debate_conducted = synthesis.get('debate_conducted', False)
                if debate_conducted:
                    self.log(f"Debate complete ({self.research_rounds} rounds)", "phase3", "INFO")
            except:
                pass
            
            self.log("Research Manager complete", "phase3", "SUCCESS")
            self.phase_results['phase3'] = {'status': 'SUCCESS', 'debate_rounds': self.research_rounds}
            return True
        else:
            self.log(f"Research Manager failed: {stderr[:200]}", "phase3", "ERROR")
            self.phase_results['phase3'] = {'status': 'FAILED'}
            return False
    
    def run_phase4_risk_team(self) -> bool:
        """Phase 4: Risk Team (3 debators)"""
        self.log("Phase 4: Risk Team (3 evaluators)", "phase4", "RUNNING")
        
        synthesis_file = self.outputs_path / "research_synthesis.json"
        if not synthesis_file.exists():
            self.log("Research synthesis not found", "phase4", "ERROR")
            return False
        
        success_count = 0
        
        for analyst in ['aggressive', 'neutral', 'conservative']:
            script_path = self.paths['risk_management'] / f"{analyst}_debator.py"
            
            if not script_path.exists():
                continue
            
            cmd = [
                sys.executable,
                str(script_path),
                self.ticker,
                "--synthesis-file", str(synthesis_file),
                "--save-evaluation", str(self.outputs_path / f"{analyst}_eval.json")
            ]
            
            # Add optional thesis files
            bull_file = self.outputs_path / "bull_thesis.json"
            bear_file = self.outputs_path / "bear_thesis.json"
            if bull_file.exists():
                cmd.extend(["--bull-file", str(bull_file)])
            if bear_file.exists():
                cmd.extend(["--bear-file", str(bear_file)])
            
            # NEW: Pass analysis date
            if self.analysis_date:
                cmd.extend(["--analysis-date", self.analysis_date])
            
            self.log(f"Running {analyst.capitalize()} evaluator...", "phase4", "INFO")
            success, _, _ = self.run_command(cmd, self.paths['risk_management'], timeout=90)
            
            if success and (self.outputs_path / f"{analyst}_eval.json").exists():
                success_count += 1
        
        if success_count >= 2:
            self.log(f"Risk team complete ({success_count}/3)", "phase4", "SUCCESS")
            self.phase_results['phase4'] = {'status': 'SUCCESS', 'evaluators': success_count}
            return True
        else:
            self.log(f"Risk team failed ({success_count}/3)", "phase4", "ERROR")
            self.phase_results['phase4'] = {'status': 'FAILED'}
            return False
    
    def run_phase5_risk_manager(self) -> bool:
        """Phase 5: Risk Manager - Final Decision"""
        self.log("Phase 5: Risk Manager (Final Decision)", "phase5", "RUNNING")
        
        synthesis_file = self.outputs_path / "research_synthesis.json"
        if not synthesis_file.exists():
            self.log("Research synthesis not found", "phase5", "ERROR")
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
        
        # NEW: Pass analysis date
        if self.analysis_date:
            cmd.extend(["--analysis-date", self.analysis_date])
        
        success, stdout, stderr = self.run_command(cmd, self.paths['managers'], timeout=120)
        
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
            self.log(f"Risk Manager failed: {stderr[:200]}", "phase5", "ERROR")
            self.phase_results['phase5'] = {'status': 'FAILED'}
            return False
       
    def run_complete_workflow(self) -> Dict:
        """Run complete workflow (Phases 1–5 only)"""
        self.start_time = datetime.now()
        
        # ============================================================
        # NEW: Load market context first (for batch/historical runs)
        # ============================================================
        self.load_market_context()
        
        print(f"\n{'='*80}")
        print("MASTER ORCHESTRATOR - TRADING SYSTEM PIPELINE")
        print(f"{'='*80}")
        print(f"Ticker: {self.ticker}")
        print(f"Portfolio: ${self.portfolio_value:,.0f}")
        print(f"Research Mode: {self.research_mode}")
        print(f"Debate Rounds: {self.research_rounds if self.research_rounds > 0 else 'None (shallow)'}")
        
        # NEW: Show analysis date prominently
        if self.analysis_date:
            print(f"{'='*80}")
            print(f"*** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
            print(f"{'='*80}")
        
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Only phases 1–5
        phases = [
            (1, self.run_phase1_analysts, "Critical"),
            (2, self.run_phase2_researchers, "Critical"),
            (3, self.run_phase3_research_manager, "Critical"),
            (4, self.run_phase4_risk_team, "Important"),
            (5, self.run_phase5_risk_manager, "Critical"),
        ]
        
        for phase_num, phase_func, importance in phases:
            print(f"\n{'─'*40}")
            print(f"PHASE {phase_num}")
            print(f"{'─'*40}")
            
            success = phase_func()
            
            if not success and importance == "Critical":
                self.log(f"Stopping: Critical Phase {phase_num} failed", status="ERROR")
                break
            
            time.sleep(0.5)
        
        self.end_time = datetime.now()
        self.print_summary()
        self.save_logs()
        
        return self.phase_results

    
    def print_summary(self):
        """Print execution summary"""
        elapsed = (self.end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("EXECUTION SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Research Mode: {self.research_mode} ({self.research_rounds} debate rounds)")
        
        # NEW: Show analysis date in summary
        if self.analysis_date:
            print(f"Analysis Date: {self.analysis_date} (HISTORICAL)")
        
        success_count = sum(1 for r in self.phase_results.values() if r.get('status') == 'SUCCESS')
        print(f"Phases Completed: {success_count}/{len(self.phase_results)}\n")
        
        print("Phase Results:")
        for phase, result in self.phase_results.items():
            status = result.get('status', 'UNKNOWN')
            icon = "✓" if status == 'SUCCESS' else "⚠" if status in ['PARTIAL', 'SKIPPED'] else "✗"
            extra = f" ({result.get('debate_rounds', '')} rounds)" if result.get('debate_rounds') else ""
            print(f"  {icon} {phase}: {status}{extra}")
        
        # Show final decision
        try:
            decision_file = self.outputs_path / "risk_decision.json"
            if decision_file.exists():
                with open(decision_file, 'r', encoding='utf-8') as f:
                    decision = json.load(f)
                print(f"\n{'─'*40}")
                print(f"FINAL DECISION: {decision.get('verdict', 'N/A')}")
                print(f"Position Size: ${decision.get('final_position_dollars', 0):,.0f}")
                print(f"{'─'*40}")
        except:
            pass
        
        print(f"\n{'='*80}\n")
    
    def save_logs(self):
        """Save execution logs"""
        log_file = self.outputs_path / "execution_log.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'ticker': self.ticker,
                    'research_mode': self.research_mode,
                    'debate_rounds': self.research_rounds,
                    'portfolio_value': self.portfolio_value,
                    'analysis_date': self.analysis_date,  # NEW: Include in logs
                    'historical_mode': self.analysis_date is not None,  # NEW
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': self.end_time.isoformat() if self.end_time else None,
                    'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
                    'phases': self.phase_results,
                    'errors': self.errors,
                    'log': self.execution_log
                }, f, indent=2)
        except Exception as e:
            print(f"Failed to save logs: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator - Complete Trading System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_orchestrator.py AAPL
  python master_orchestrator.py AAPL --research-mode deep
  python master_orchestrator.py AAPL --research-mode research --research-rounds 5
  python master_orchestrator.py AAPL --analysis-date 2024-06-15

Research Modes:
  shallow  - Quick analysis, no debate (~2 minutes)
  deep     - 3 debate rounds (~5 minutes)
  research - 5 debate rounds (~8 minutes)

Historical Backtesting:
  --analysis-date YYYY-MM-DD  - Analyze using data from specified date
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--research-mode", 
                       choices=['shallow', 'deep', 'research'],
                       default='shallow',
                       help="Research depth (default: shallow)")
    parser.add_argument("--research-rounds",
                       type=int,
                       default=0,
                       help="Override debate rounds (0=auto based on mode)")
    parser.add_argument("--portfolio-value",
                       type=float,
                       default=100000,
                       help="Portfolio value (default: 100000)")
    
    # ============================================================
    # NEW: Add analysis-date argument for historical backtesting
    # ============================================================
    parser.add_argument("--analysis-date",
                       type=str,
                       default=None,
                       help="Historical analysis date (YYYY-MM-DD format)")
    
    args = parser.parse_args()
    
    print(f"Starting from: {os.getcwd()}")
    
    try:
        orchestrator = MasterOrchestrator(
            ticker=args.ticker,
            portfolio_value=args.portfolio_value,
            research_mode=args.research_mode,
            research_rounds=args.research_rounds,
            analysis_date=args.analysis_date  # NEW: Pass to orchestrator
        )
        
        results = orchestrator.run_complete_workflow()
        
        success_count = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        sys.exit(0 if success_count == len(results) else 1)
        
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()