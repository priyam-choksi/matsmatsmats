"""
Master Orchestrator - Complete Multi-Agent Trading System with Game Theory
Usage: python master_orchestrator.py AAPL --run-all --research-mode deep
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


class MasterOrchestrator:
    def __init__(self, ticker: str, portfolio_value: float = 100000, research_mode: str = 'shallow', research_rounds: int = 1):
        self.ticker = ticker.upper()
        self.portfolio_value = portfolio_value
        self.research_mode = research_mode
        self.research_rounds = research_rounds
        
        # Track execution
        self.execution_log = []
        self.phase_results = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
        
        # Setup paths - CRITICAL FIX
        self.setup_paths()
    
    def setup_paths(self):
        """Setup all paths relative to where script is run from"""
        # Current working directory
        self.cwd = Path.cwd()
        
        # Detect where we are
        if self.cwd.name == 'orchestrators':
            # Running from agents/orchestrators
            self.project_root = self.cwd.parent.parent
            self.agents_root = self.cwd.parent
        elif self.cwd.name == 'agents':
            # Running from agents/
            self.project_root = self.cwd.parent
            self.agents_root = self.cwd
        else:
            # Running from project root
            self.project_root = self.cwd
            self.agents_root = self.cwd / "agents"
        
        # Setup outputs
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
        print(f"[SETUP] Outputs: {self.outputs_path}")
    
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
        """Execute command with proper encoding"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd),  # Convert Path to string
                encoding='utf-8',
                errors='replace'
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            return False, "", str(e)
    
    def run_phase1_analysts(self) -> bool:
        """Phase 1: Run discussion hub"""
        self.log("Phase 1: Running Analysts (4)", "phase1", "RUNNING")
        
        cmd = [
            "python", "discussion_hub.py", self.ticker,
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
            self.log("Analysts complete", "phase1", "SUCCESS")
            self.phase_results['phase1'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log(f"Analysts failed: {stderr[:100]}", "phase1", "ERROR")
            self.errors.append(f"Analysts: {stderr[:200]}")
            self.phase_results['phase1'] = {'status': 'FAILED'}
            return False
    
    def run_phase2_researchers(self) -> bool:
        """Phase 2: Run researchers"""
        self.log(f"Phase 2: Researchers ({self.research_mode}, {self.research_rounds}r)", "phase2", "RUNNING")
        
        # Bear
        bear_cmd = [
            "python", "bear_researcher.py", self.ticker,
            "--discussion-file", str(self.outputs_path / "discussion_points.json"),
            "--mode", self.research_mode,
            "--rounds", str(self.research_rounds),
            "--save-data", str(self.outputs_path / "bear_thesis.json")
        ]
        
        success_bear, _, err_bear = self.run_command(
            bear_cmd,
            cwd=self.paths['researcher'],
            timeout=300
        )
        
        # Bull
        bull_cmd = [
            "python", "bull_researcher.py", self.ticker,
            "--discussion-file", str(self.outputs_path / "discussion_points.json"),
            "--mode", self.research_mode,
            "--rounds", str(self.research_rounds),
            "--save-data", str(self.outputs_path / "bull_thesis.json")
        ]
        
        success_bull, _, err_bull = self.run_command(
            bull_cmd,
            cwd=self.paths['researcher'],
            timeout=300
        )
        
        if success_bull and success_bear:
            self.log("Researchers complete", "phase2", "SUCCESS")
            self.phase_results['phase2'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Researchers incomplete", "phase2", "ERROR")
            self.phase_results['phase2'] = {'status': 'PARTIAL'}
            return False
    
    def run_phase3_research_manager(self) -> bool:
        """Phase 3: Research Manager"""
        self.log("Phase 3: Research Manager", "phase3", "RUNNING")
        
        cmd = [
            "python", "research_manager.py", self.ticker,
            "--bull-file", str(self.outputs_path / "bull_thesis.json"),
            "--bear-file", str(self.outputs_path / "bear_thesis.json"),
            "--save-synthesis", str(self.outputs_path / "research_synthesis.json")
        ]
        
        success, _, stderr = self.run_command(
            cmd,
            cwd=self.paths['managers'],
            timeout=120
        )
        
        if success:
            self.log("Research Manager complete", "phase3", "SUCCESS")
            self.phase_results['phase3'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log(f"Research Manager failed", "phase3", "ERROR")
            self.phase_results['phase3'] = {'status': 'FAILED'}
            return False
    
    def run_phase4_risk_team(self) -> bool:
        """Phase 4: Risk Team"""
        self.log("Phase 4: Risk Team (3)", "phase4", "RUNNING")
        
        success_count = 0
        
        for analyst in ['aggressive', 'neutral', 'conservative']:
            cmd = [
                "python", f"{analyst}_debator.py", self.ticker,
                "--synthesis-file", str(self.outputs_path / "research_synthesis.json"),
                "--bull-file", str(self.outputs_path / "bull_thesis.json"),
                "--bear-file", str(self.outputs_path / "bear_thesis.json"),
                "--save-evaluation", str(self.outputs_path / f"{analyst}_eval.json")
            ]
            
            success, _, _ = self.run_command(
                cmd,
                cwd=self.paths['risk_management'],
                timeout=90
            )
            
            if success:
                success_count += 1
                print(f"  [OK] {analyst.capitalize()}")
            else:
                print(f"  [X] {analyst.capitalize()}")
        
        if success_count >= 2:
            self.log(f"Risk team complete ({success_count}/3)", "phase4", "SUCCESS")
            self.phase_results['phase4'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Risk team failed", "phase4", "ERROR")
            self.phase_results['phase4'] = {'status': 'FAILED'}
            return False
    
    def run_phase5_risk_manager(self) -> bool:
        """Phase 5: Risk Manager"""
        self.log("Phase 5: Risk Manager", "phase5", "RUNNING")
        
        cmd = [
            "python", "risk_manager.py", self.ticker,
            "--synthesis-file", str(self.outputs_path / "research_synthesis.json"),
            "--portfolio-value", str(self.portfolio_value),
            "--save-decision", str(self.outputs_path / "risk_decision.json")
        ]
        
        success, _, stderr = self.run_command(
            cmd,
            cwd=self.paths['managers'],
            timeout=120
        )
        
        if success:
            try:
                with open(self.outputs_path / "risk_decision.json", 'r') as f:
                    decision = json.load(f)
                verdict = decision.get('verdict', '?')
                self.log(f"Final: {verdict}", "phase5", "SUCCESS")
            except:
                self.log("Risk Manager complete", "phase5", "SUCCESS")
            
            self.phase_results['phase5'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Risk Manager failed", "phase5", "ERROR")
            self.phase_results['phase5'] = {'status': 'FAILED'}
            return False
    
    def run_phase6_game_theory(self) -> bool:
        """Phase 6: Game Theory"""
        self.log("Phase 6: Game Theory", "phase6", "RUNNING")
        
        gt_script = self.paths['orchestrators'] / "game_theory_orchestrator.py"
        
        if not gt_script.exists():
            self.log("Game theory not available", "phase6", "INFO")
            return True
        
        cmd = [
            "python", "game_theory_orchestrator.py", self.ticker,
            "--outputs-dir", str(self.outputs_path)
        ]
        
        success, _, _ = self.run_command(
            cmd,
            cwd=self.paths['orchestrators'],
            timeout=90
        )
        
        if success:
            self.log("Game theory complete", "phase6", "SUCCESS")
            self.phase_results['phase6'] = {'status': 'SUCCESS'}
            return True
        else:
            self.log("Game theory failed", "phase6", "ERROR")
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
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Run phases
        self.run_phase1_analysts()
        self.run_phase2_researchers()
        self.run_phase3_research_manager()
        self.run_phase4_risk_team()
        self.run_phase5_risk_manager()
        
        if include_game_theory:
            self.run_phase6_game_theory()
        
        self.end_time = datetime.now()
        
        # Summary
        self.print_summary()
        self.save_logs()
        
        return self.phase_results
    
    def print_summary(self):
        """Print summary"""
        elapsed = (self.end_time - self.start_time).seconds
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        print(f"Time: {elapsed}s ({elapsed/60:.1f}min)")
        
        # Count successes
        success_count = sum(1 for r in self.phase_results.values() if r.get('status') == 'SUCCESS')
        total_count = len(self.phase_results)
        
        print(f"Phases: {success_count}/{total_count} successful\n")
        
        for phase, result in self.phase_results.items():
            status = result.get('status', 'UNKNOWN')
            icon = "[OK]" if status == 'SUCCESS' else "[!!]" if status == 'PARTIAL' else "[X]"
            print(f"  {icon} {phase}: {status}")
        
        # Final decision
        try:
            with open(self.outputs_path / "risk_decision.json") as f:
                dec = json.load(f)
            print(f"\n[FINAL] {dec.get('verdict')}: ${dec.get('final_position_dollars', 0):,.0f}")
        except:
            pass
        
        print(f"\n{'='*80}\n")
    
    def save_logs(self):
        """Save logs"""
        with open(self.outputs_path / "execution_log.json", 'w', encoding='utf-8') as f:
            json.dump({
                'ticker': self.ticker,
                'research_mode': self.research_mode,
                'phases': self.phase_results,
                'errors': self.errors
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator",
        epilog="""
Examples:
  python master_orchestrator.py AAPL
  python master_orchestrator.py AAPL --run-all
  python master_orchestrator.py AAPL --run-all --research-mode deep --research-rounds 3
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker")
    parser.add_argument("--run-all", action="store_true", help="Include game theory")
    parser.add_argument("--research-mode", choices=['shallow', 'deep', 'research'], default='shallow')
    parser.add_argument("--research-rounds", type=int, default=1)
    parser.add_argument("--portfolio-value", type=float, default=100000)
    
    args = parser.parse_args()
    
    try:
        orch = MasterOrchestrator(
            ticker=args.ticker,
            portfolio_value=args.portfolio_value,
            research_mode=args.research_mode,
            research_rounds=args.research_rounds
        )
        
        orch.run_complete_workflow(include_game_theory=args.run_all)
        
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()