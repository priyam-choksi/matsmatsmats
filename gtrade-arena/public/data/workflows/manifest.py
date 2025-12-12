"""
Generate manifest.json and per-ticker list.json files for Pipeline History Page.

Run from: gtrade-arena/public/data/workflows/
Usage: python generate_manifest.py
"""

import os
import json
from pathlib import Path

def generate_manifest(workflows_dir: str = "."):
    """Scan workflow directories and generate manifest files."""
    
    workflows_path = Path(workflows_dir)
    all_workflows = []
    
    # Scan each ticker folder
    ticker_dirs = sorted([d for d in workflows_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"Found {len(ticker_dirs)} ticker directories")
    
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        ticker_workflows = []
        
        # Skip non-ticker folders
        if ticker in ['__pycache__', 'node_modules', '.git']:
            continue
            
        print(f"Scanning {ticker}...", end=" ")
        
        # Scan each date folder
        date_dirs = sorted([d for d in ticker_dir.iterdir() if d.is_dir()])
        
        for date_dir in date_dirs:
            folder_name = date_dir.name
            
            # Parse folder name: 2024-06-17_sample090_day268
            parts = folder_name.split('_')
            if len(parts) < 3:
                continue
            
            try:
                date = parts[0]
                sample = int(parts[1].replace('sample', '')) if 'sample' in parts[1] else 0
                day = int(parts[2].replace('day', '')) if 'day' in parts[2] else 0
            except (ValueError, IndexError):
                continue
            
            # Find portfolio subfolder(s)
            portfolio_dirs = [d for d in date_dir.iterdir() if d.is_dir() and d.name.startswith('portfolio_')]
            
            for portfolio_dir in portfolio_dirs:
                portfolio_folder = portfolio_dir.name
                
                # Read summary.json or risk_decision.json for verdict
                verdict = "HOLD"
                position = 0
                confidence = "MEDIUM"
                
                summary_file = portfolio_dir / "summary.json"
                risk_file = portfolio_dir / "risk_decision.json"
                
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            decision = summary.get('decision', {})
                            verdict = decision.get('verdict', 'HOLD')
                            position = decision.get('position', 0)
                            confidence = decision.get('confidence', 'MEDIUM')
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                elif risk_file.exists():
                    try:
                        with open(risk_file, 'r') as f:
                            risk = json.load(f)
                            verdict = risk.get('verdict', 'HOLD')
                            position = risk.get('final_position_dollars', 0)
                            confidence = risk.get('confidence', 'MEDIUM')
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                workflow = {
                    "ticker": ticker,
                    "date": date,
                    "folder": folder_name,
                    "portfolioFolder": portfolio_folder,
                    "sample": sample,
                    "day": day,
                    "verdict": verdict,
                    "position": position,
                    "confidence": confidence
                }
                
                ticker_workflows.append(workflow)
                all_workflows.append(workflow)
        
        # Save per-ticker list.json
        if ticker_workflows:
            list_file = ticker_dir / "list.json"
            with open(list_file, 'w') as f:
                json.dump(ticker_workflows, f, indent=2)
            print(f"{len(ticker_workflows)} workflows")
        else:
            print("0 workflows")
    
    # Save global manifest.json
    manifest_file = workflows_path / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(all_workflows, f, indent=2)
    
    print(f"\n✅ Generated manifest.json with {len(all_workflows)} total workflows")
    print(f"✅ Generated list.json for {len(ticker_dirs)} tickers")
    
    # Print summary by verdict
    verdict_counts = {}
    for w in all_workflows:
        v = w['verdict']
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    
    print("\nVerdict Distribution:")
    for v, count in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {count}")

if __name__ == "__main__":
    generate_manifest()