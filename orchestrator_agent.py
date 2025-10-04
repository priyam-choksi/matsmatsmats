"""
Orchestrator Agent - Combines outputs from other agents
Usage: python orchestrator_agent.py AAPL --agents technical news fundamental --output final_report.txt
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from openai import OpenAI

class OrchestratorAgent:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are the head trader synthesizing reports from multiple analysts.
Weigh their recommendations, resolve conflicts, and make the final trading decision.
Consider confidence levels and alignment. Provide clear rationale.
End with: FINAL DECISION: BUY/HOLD/SELL"""
    
    def run_agent(self, agent_script, ticker=None):
        """Run an individual agent script"""
        cmd = ["python", agent_script]
        if ticker:
            cmd.append(ticker)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout
        except subprocess.TimeoutExpired:
            return f"Agent {agent_script} timed out"
        except Exception as e:
            return f"Error running {agent_script}: {str(e)}"
    
    def extract_recommendation(self, report):
        """Extract recommendation from agent report"""
        if "RECOMMENDATION:" in report:
            lines = report.split('\n')
            for line in lines:
                if "RECOMMENDATION:" in line:
                    if "BUY" in line.upper():
                        return "BUY"
                    elif "SELL" in line.upper():
                        return "SELL"
        return "HOLD"
    
    def synthesize_with_llm(self, reports):
        """Use LLM to synthesize multiple reports"""
        if not self.client:
            # Simple voting mechanism
            votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
            for name, report in reports.items():
                rec = self.extract_recommendation(report)
                votes[rec] += 1
            
            decision = max(votes, key=votes.get)
            return f"""Synthesis of {len(reports)} analyst reports:

Votes: BUY={votes['BUY']}, HOLD={votes['HOLD']}, SELL={votes['SELL']}

FINAL DECISION: {decision}"""
        
        try:
            combined_reports = "\n\n".join([f"=== {name.upper()} ANALYST ===\n{report}" 
                                           for name, report in reports.items()])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Synthesize these reports for {self.ticker}:\n\n{combined_reports}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return "FINAL DECISION: HOLD (Error in synthesis)"
    
    def run(self, agents):
        """Execute orchestration"""
        reports = {}
        
        print(f"\nORCHESTRATOR: Analyzing {self.ticker}")
        print("=" * 50)
        
        # Map agent names to script files
        agent_scripts = {
            'technical': 'technical_agent.py',
            'news': 'news_agent.py',
            'fundamental': 'fundamental_agent.py',
            'macro': 'macro_agent.py'
        }
        
        # Run each requested agent
        for agent in agents:
            if agent in agent_scripts:
                print(f"\nRunning {agent} analyst...")
                script = agent_scripts[agent]
                
                if os.path.exists(script):
                    if agent == 'macro':
                        report = self.run_agent(script)
                    else:
                        report = self.run_agent(script, self.ticker)
                    reports[agent] = report
                    
                    rec = self.extract_recommendation(report)
                    print(f"  → {agent.capitalize()} recommendation: {rec}")
                else:
                    print(f"  → Script {script} not found")
        
        # Synthesize results
        print("\nSynthesizing recommendations...")
        synthesis = self.synthesize_with_llm(reports)
        
       # Create final report with full details
        final_report = f"""
        ================================================================================
                            ORCHESTRATED TRADING ANALYSIS                     
        ================================================================================

        Ticker: {self.ticker}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Agents Used: {', '.join(agents)}

        ================================================================================
                                QUICK SUMMARY
        ================================================================================
        """
        # Add quick summary of recommendations
        for agent, report in reports.items():
            rec = self.extract_recommendation(report)
            final_report += f"* {agent.capitalize():15} -> {rec}\n"

        # Add detailed reports from each agent
        final_report += f"""
        ================================================================================
                            DETAILED AGENT REPORTS
        ================================================================================
        """

        for agent, report in reports.items():
            final_report += f"""
        --------------------------------------------------------------------------------
        {agent.upper()} ANALYST REPORT                                    
        --------------------------------------------------------------------------------

        {report}

        """

        # Add final synthesis
        final_report += f"""
        ================================================================================
                            ORCHESTRATOR SYNTHESIS
        ================================================================================

        {synthesis}

        ================================================================================
        """

        return final_report

def main():
    parser = argparse.ArgumentParser(description="Orchestrator Agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--agents", nargs="+", 
                       choices=["technical", "news", "fundamental", "macro"],
                       default=["technical", "news", "fundamental"],
                       help="Which agents to run")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    
    args = parser.parse_args()
    
    orchestrator = OrchestratorAgent(args.ticker, args.api_key, args.model)
    result = orchestrator.run(args.agents)
    
    print(result)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:            
            f.write(result)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()