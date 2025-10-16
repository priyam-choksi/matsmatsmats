"""
Master Orchestrator - Runs the complete multi-agent trading system workflow
Usage: python master_orchestrator.py AAPL --portfolio-value 100000 --run-all
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
from openai import OpenAI

class MasterOrchestrator:
    def __init__(self, ticker, portfolio_value=100000, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.portfolio_value = portfolio_value
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Track execution status
        self.execution_log = []
        self.phase_results = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
        
        # Define workflow phases
        self.workflow = {
            'phase1_analysts': {
                'name': 'Market Analysis',
                'agents': ['technical', 'news', 'fundamental', 'macro'],
                'orchestrator': 'discussion_hub',
                'path': 'orchestrators'
            },
            'phase2_researchers': {
                'name': 'Research Deep Dive',
                'agents': ['bull_researcher', 'bear_researcher'],
                'path': 'researcher'
            },
            'phase3_risk_team': {
                'name': 'Risk Evaluation',
                'agents': ['aggressive_debator', 'neutral_debator', 'conservative_debator'],
                'path': 'risk_management'
            },
            'phase4_managers': {
                'name': 'Management Decision',
                'agents': ['research_manager', 'risk_manager'],
                'path': 'managers'
            },
            'phase5_execution': {
                'name': 'Trade Execution',
                'agents': ['trader'],
                'path': 'execution'
            }
        }
        
        # Setup paths
        self.base_path = Path.cwd()
        self.setup_directories()
    
    def setup_directories(self):
        """Ensure outputs directory exists in project root"""
        # Go up two levels from orchestrators to get to project root
        # orchestrators -> agents -> project_root
        project_root = self.base_path.parent  # Go up from orchestrators to agents
        outputs_dir = project_root.parent / "outputs"  # Go up from agents to root, then outputs
        
        if not outputs_dir.exists():
            print(f"Creating outputs directory in project root...")
            outputs_dir.mkdir(exist_ok=True)
            
        # Store the outputs path for use in other methods
        self.outputs_path = outputs_dir
    
    def log(self, message, phase=None, status='INFO'):
        """Log execution progress"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'phase': phase,
            'message': message,
            'status': status
        }
        self.execution_log.append(log_entry)
        
        # Print with formatting
        if status == 'ERROR':
            print(f"❌ [{timestamp}] {message}")
        elif status == 'SUCCESS':
            print(f"✅ [{timestamp}] {message}")
        elif status == 'RUNNING':
            print(f"⚙️  [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")
    
    def run_command(self, cmd, timeout=60):
        """Execute a command and return result"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_path
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def run_phase1_analysts(self):
        """Run analyst agents and discussion hub"""
        self.log("Starting Phase 1: Market Analysis", "phase1", "RUNNING")
        
        # Use the outputs path from project root
        output_file = str(self.outputs_path / "discussion_points.json")
        
        # Run discussion hub which runs all analysts
        cmd = [
            "python", "discussion_hub.py", self.ticker,
            "--run-analysts",
            "--output", output_file,
            "--format", "json"
        ]
        
        success, stdout, stderr = self.run_command(cmd, timeout=120)
        
        if success:
            self.log("✓ Analysts complete: Technical, News, Fundamental, Macro", "phase1", "SUCCESS")
            self.log("✓ Discussion Hub synthesis complete", "phase1", "SUCCESS")
            self.phase_results['phase1'] = {'status': 'SUCCESS', 'output': 'outputs/discussion_points.json'}
        else:
            self.log(f"Analyst phase failed: {stderr}", "phase1", "ERROR")
            self.errors.append(f"Phase 1 failed: {stderr}")
            self.phase_results['phase1'] = {'status': 'FAILED'}
        
        return success
    
    def run_phase2_researchers(self):
        """Run bull and bear researchers"""
        self.log("Starting Phase 2: Research Deep Dive", "phase2", "RUNNING")
        
        # Use outputs path from project root
        discussion_file = str(self.outputs_path / "discussion_points.json")
        bull_data = str(self.outputs_path / "bull_thesis.json")
        bull_report = str(self.outputs_path / "bull_report.txt")
        bear_data = str(self.outputs_path / "bear_thesis.json")
        bear_report = str(self.outputs_path / "bear_report.txt")
        
        # Run Bull Researcher
        cmd = [
            "python", "../researcher/bull_researcher.py", self.ticker,
            "--discussion-file", discussion_file,
            "--save-data", bull_data,
            "--output", bull_report
        ]
        
        success_bull, _, stderr_bull = self.run_command(cmd)
        
        # Run Bear Researcher
        cmd = [
            "python", "../researcher/bear_researcher.py", self.ticker,
            "--discussion-file", discussion_file,
            "--save-data", bear_data,
            "--output", bear_report
        ]
        
        success_bear, _, stderr_bear = self.run_command(cmd)
        
        if success_bull and success_bear:
            self.log("✓ Bull thesis complete", "phase2", "SUCCESS")
            self.log("✓ Bear thesis complete", "phase2", "SUCCESS")
            self.phase_results['phase2'] = {'status': 'SUCCESS'}
        else:
            self.log("Research phase incomplete", "phase2", "ERROR")
            self.errors.append("Phase 2 had issues")
            self.phase_results['phase2'] = {'status': 'PARTIAL'}
        
        return success_bull and success_bear
    
    def run_phase3_risk_team(self):
        """Run risk management debators"""
        self.log("Starting Phase 3: Risk Evaluation", "phase3", "RUNNING")
        
        # Use outputs path from project root
        bull_file = str(self.outputs_path / "bull_thesis.json")
        bear_file = str(self.outputs_path / "bear_thesis.json")
        
        risk_profiles = ['aggressive', 'neutral', 'conservative']
        success_count = 0
        
        for profile in risk_profiles:
            eval_file = str(self.outputs_path / f"{profile}_eval.json")
            cmd = [
                "python", f"../risk_management/{profile}_debator.py", self.ticker,
                "--bull-file", bull_file,
                "--bear-file", bear_file,
                "--save-evaluation", eval_file
            ]
            
            success, _, _ = self.run_command(cmd)
            if success:
                self.log(f"✓ {profile.capitalize()} evaluation complete", "phase3", "SUCCESS")
                success_count += 1
            else:
                self.log(f"✗ {profile.capitalize()} evaluation failed", "phase3", "ERROR")
        
        self.phase_results['phase3'] = {
            'status': 'SUCCESS' if success_count == 3 else 'PARTIAL',
            'completed': success_count
        }
        
        return success_count >= 2  # Need at least 2 for consensus
    
    def run_phase4_managers(self):
        """Run research and risk managers"""
        self.log("Starting Phase 4: Management Decision", "phase4", "RUNNING")
        
        # Use outputs path from project root
        bull_file = str(self.outputs_path / "bull_thesis.json")
        bear_file = str(self.outputs_path / "bear_thesis.json")
        synthesis_file = str(self.outputs_path / "research_synthesis.json")
        research_report = str(self.outputs_path / "research_report.txt")
        decision_file = str(self.outputs_path / "risk_decision.json")
        risk_report = str(self.outputs_path / "risk_report.txt")
        
        # Run Research Manager
        cmd = [
            "python", "../managers/research_manager.py", self.ticker,
            "--bull-file", bull_file,
            "--bear-file", bear_file,
            "--save-synthesis", synthesis_file,
            "--output", research_report
        ]
        
        success_research, _, _ = self.run_command(cmd, timeout=90)
        
        if success_research:
            self.log("✓ Research synthesis complete", "phase4", "SUCCESS")
        
        # Run Risk Manager
        cmd = [
            "python", "../managers/risk_manager.py", self.ticker,
            "--synthesis-file", synthesis_file,
            "--portfolio-value", str(self.portfolio_value),
            "--save-decision", decision_file,
            "--output", risk_report
        ]
        
        success_risk, _, _ = self.run_command(cmd)
        
        if success_risk:
            self.log("✓ Risk decision complete", "phase4", "SUCCESS")
            self.phase_results['phase4'] = {'status': 'SUCCESS'}
        else:
            self.log("Management phase failed", "phase4", "ERROR")
            self.phase_results['phase4'] = {'status': 'FAILED'}
        
        return success_research and success_risk
    
    def run_phase5_execution(self):
        """Run trader for final execution"""
        self.log("Starting Phase 5: Trade Execution", "phase5", "RUNNING")
        
        # Use outputs path from project root
        decision_file = str(self.outputs_path / "risk_decision.json")
        order_file = str(self.outputs_path / "final_order.json")
        trade_report = str(self.outputs_path / "trade_order.txt")
        
        cmd = [
            "python", "../execution/trader.py", self.ticker,
            "--risk-decision", decision_file,
            "--save-order", order_file,
            "--output", trade_report
        ]
        
        success, stdout, _ = self.run_command(cmd)
        
        if success:
            # Check if trade was approved
            try:
                with open(order_file, 'r') as f:
                    order = json.load(f)
                    if order.get('order_status') == 'READY':
                        self.log("✓ TRADE APPROVED - Order ready for execution", "phase5", "SUCCESS")
                    elif order.get('order_status') == 'CANCELLED':
                        self.log("✓ TRADE REJECTED - No position taken", "phase5", "SUCCESS")
                    else:
                        self.log("✓ Trade evaluation complete", "phase5", "SUCCESS")
            except:
                self.log("✓ Execution phase complete", "phase5", "SUCCESS")
            
            self.phase_results['phase5'] = {'status': 'SUCCESS'}
        else:
            self.log("Execution phase failed", "phase5", "ERROR")
            self.phase_results['phase5'] = {'status': 'FAILED'}
        
        return success
    
    def generate_summary_report(self):
        """Generate comprehensive summary of entire workflow"""
        report = f"""
{'='*80}
                    MASTER ORCHESTRATOR - EXECUTION SUMMARY
{'='*80}

Ticker: {self.ticker}
Portfolio Value: ${self.portfolio_value:,.2f}
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runtime: {(self.end_time - self.start_time).seconds if self.end_time else 0} seconds

WORKFLOW STATUS:
----------------"""
        
        for phase_key, phase_info in self.workflow.items():
            if phase_key in self.phase_results:
                status = self.phase_results[phase_key].get('status', 'UNKNOWN')
                status_icon = "✅" if status == 'SUCCESS' else "⚠️" if status == 'PARTIAL' else "❌"
                report += f"\n{status_icon} {phase_info['name']:20} : {status}"
        
        # Try to load final decision
        decision_file = self.outputs_path / "risk_decision.json"
        try:
            with open(decision_file, 'r') as f:
                risk_decision = json.load(f)
                verdict = risk_decision.get('verdict', 'UNKNOWN')
                
                report += f"""

FINAL DECISION:
---------------
Risk Verdict: {verdict}"""
                
                if verdict in ['APPROVE', 'MODIFY']:
                    position = risk_decision.get('final_position', 0)
                    report += f"""
Position Size: ${position:,.2f} ({position/self.portfolio_value*100:.1f}% of portfolio)"""
        except:
            pass
        
        # Try to load final order
        order_file = self.outputs_path / "final_order.json"
        try:
            with open(order_file, 'r') as f:
                order = json.load(f)
                if order.get('order_status') == 'READY':
                    report += f"""

TRADE ORDER:
------------
Action: {order.get('action', 'N/A')}
Shares: {order.get('shares', 0):,}
Entry Price: ${order.get('entry_price', 0):.2f}
Total Value: ${order.get('total_value', 0):,.2f}
Stop Loss: ${order.get('stop_loss', {}).get('price', 0):.2f}"""
                elif order.get('order_status') == 'CANCELLED':
                    report += f"""

TRADE ORDER:
------------
Status: CANCELLED
Reason: {order.get('reason', 'Risk Management Rejection')}"""
        except:
            pass
        
        if self.errors:
            report += f"""

ERRORS ENCOUNTERED:
-------------------"""
            for error in self.errors[:5]:  # Show first 5 errors
                report += f"\n• {error}"
        
        report += f"""

OUTPUT FILES:
-------------
All results saved in: {self.outputs_path}
• discussion_points.json - Analyst consensus
• bull_thesis.json / bear_thesis.json - Research
• aggressive_eval.json / neutral_eval.json / conservative_eval.json - Risk evaluations
• research_synthesis.json - Research Manager synthesis
• risk_decision.json - Risk Manager decision
• final_order.json - Trade order details

{'='*80}
"""
        return report
    
    def run_workflow(self, phases_to_run=None, stop_on_error=False):
        """Execute the complete workflow"""
        self.start_time = datetime.now()
        
        print(f"""
{'='*80}
           MASTER ORCHESTRATOR - MULTI-AGENT TRADING SYSTEM
{'='*80}
Ticker: {self.ticker}
Portfolio: ${self.portfolio_value:,.2f}
Starting: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {self.outputs_path}
{'='*80}
        """)
        
        # Define phase execution order
        phase_functions = [
            ('phase1', self.run_phase1_analysts),
            ('phase2', self.run_phase2_researchers),
            ('phase3', self.run_phase3_risk_team),
            ('phase4', self.run_phase4_managers),
            ('phase5', self.run_phase5_execution)
        ]
        
        # Execute phases
        for phase_name, phase_func in phase_functions:
            if phases_to_run and phase_name not in phases_to_run:
                self.log(f"Skipping {phase_name} (not requested)", phase_name, "INFO")
                continue
            
            print(f"\n{'-'*60}")
            success = phase_func()
            
            if not success and stop_on_error:
                self.log("Workflow stopped due to error", phase_name, "ERROR")
                break
            
            # Small delay between phases
            time.sleep(1)
        
        self.end_time = datetime.now()
        
        # Generate and save summary
        summary = self.generate_summary_report()
        print(summary)
        
        # Save summary to file
        summary_file = self.outputs_path / "master_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Save execution log
        log_file = self.outputs_path / "execution_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                'ticker': self.ticker,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'phases': self.phase_results,
                'log': self.execution_log,
                'errors': self.errors
            }, f, indent=2)
        
        return self.phase_results
    
    def quick_analysis(self):
        """Run only analysis and research phases (no execution)"""
        self.log("Running Quick Analysis Mode (No Trading)", None, "INFO")
        return self.run_workflow(phases_to_run=['phase1', 'phase2', 'phase3'])
    
    def full_analysis(self):
        """Run all phases including execution"""
        self.log("Running Full Analysis with Trading Decision", None, "INFO")
        return self.run_workflow()

def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator - Complete Trading System")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--portfolio-value", type=float, default=100000,
                       help="Total portfolio value (default: $100,000)")
    parser.add_argument("--run-all", action="store_true",
                       help="Run complete workflow including trading")
    parser.add_argument("--quick", action="store_true",
                       help="Quick analysis only (no trading decision)")
    parser.add_argument("--phases", nargs="+",
                       choices=['phase1', 'phase2', 'phase3', 'phase4', 'phase5'],
                       help="Run specific phases only")
    parser.add_argument("--stop-on-error", action="store_true",
                       help="Stop workflow if any phase fails")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    
    args = parser.parse_args()
    
    # Initialize orchestrator (this will create outputs directory in project root)
    orchestrator = MasterOrchestrator(
        args.ticker,
        args.portfolio_value,
        args.api_key,
        args.model
    )
    
    # Run workflow based on arguments
    if args.run_all:
        results = orchestrator.full_analysis()
    elif args.quick:
        results = orchestrator.quick_analysis()
    elif args.phases:
        results = orchestrator.run_workflow(phases_to_run=args.phases)
    else:
        # Default: run analysis only
        print("\nNo mode specified. Use --run-all for complete workflow.")
        print("Running quick analysis mode (no trading)...")
        results = orchestrator.quick_analysis()
    
    # Check final status
    all_success = all(r.get('status') == 'SUCCESS' for r in results.values())
    if all_success:
        print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY")
    else:
        print("\n⚠️ WORKFLOW COMPLETED WITH ISSUES")
    
    print(f"\nResults saved in: {orchestrator.outputs_path}")
    print(f"View summary: {orchestrator.outputs_path}/master_summary.txt")
    print(f"View execution log: {orchestrator.outputs_path}/execution_log.json")

if __name__ == "__main__":
    main()