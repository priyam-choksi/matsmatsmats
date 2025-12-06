# fix_missing_summary.py
# Run from: agents/orchestrators/

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def fix_missing():
    project_root = Path.cwd().parent.parent
    outputs_path = project_root / "outputs"
    workflows_path = outputs_path / "workflows"
    game_theory_path = outputs_path / "game_theory"
    managers_path = project_root / "agents" / "managers"
    
    # DEBUG: Print paths
    print(f"Project root: {project_root}")
    print(f"Workflows path: {workflows_path}")
    print(f"Exists: {workflows_path.exists()}")
    
    if not workflows_path.exists():
        print("ERROR: workflows folder not found!")
        return
    
    # List what's in there
    tickers = [f.name for f in workflows_path.iterdir() if f.is_dir()]
    print(f"Found tickers: {tickers}")
    
    fixed = 0
    skipped = 0
    failed = 0
    needs_fix = 0
    
    for ticker_folder in workflows_path.iterdir():
        if not ticker_folder.is_dir():
            continue
        
        ticker = ticker_folder.name
            
        for date_folder in ticker_folder.iterdir():
            if not date_folder.is_dir():
                continue
                
            for portfolio_folder in date_folder.iterdir():
                if not portfolio_folder.is_dir():
                    continue
                
                synthesis_file = portfolio_folder / "research_synthesis.json"
                risk_file = portfolio_folder / "risk_decision.json"
                summary_file = portfolio_folder / "summary.json"
                
                # Skip if already complete
                if summary_file.exists() and risk_file.exists():
                    skipped += 1
                    continue
                
                # Need synthesis to proceed
                if not synthesis_file.exists():
                    continue
                
                needs_fix += 1
                print(f"Fixing: {ticker}/{date_folder.name}/{portfolio_folder.name}")
                
                # Parse folder name: 2024-10-17_sample001_day001
                try:
                    parts = date_folder.name.split('_sample')
                    date_str = parts[0]
                    rest = parts[1]
                    sample_str, day_str = rest.split('_day')
                    sample_num = int(sample_str)
                    actual_day = int(day_str)
                    portfolio_size = int(portfolio_folder.name.replace('portfolio_', ''))
                except Exception as e:
                    print(f"  ✗ Could not parse folder name: {e}")
                    failed += 1
                    continue
                
                # Step 1: Create risk_decision.json if missing
                if not risk_file.exists():
                    cmd = [
                        sys.executable,
                        str(managers_path / "risk_manager.py"),
                        ticker,
                        "--synthesis-file", str(synthesis_file),
                        "--portfolio-value", str(portfolio_size),
                        "--save-decision", str(risk_file)
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if not risk_file.exists():
                    print(f"  ✗ risk_decision.json failed")
                    failed += 1
                    continue
                
                print(f"  ✓ risk_decision.json")
                
                # Step 2: Get market_data from game_theory date_info.json
                date_info_file = game_theory_path / ticker / f"portfolio_{portfolio_size}" / f"sample_{sample_num}" / "date_info.json"
                
                market_data = None
                if date_info_file.exists():
                    try:
                        with open(date_info_file, 'r') as f:
                            date_info = json.load(f)
                            market_data = date_info.get('market_data', {})
                    except:
                        pass
                
                if not market_data:
                    market_data = {
                        "actual_day": actual_day,
                        "date": date_str,
                        "sample_num": sample_num
                    }
                
                # Step 3: Read decision from risk_decision.json
                try:
                    with open(risk_file, 'r') as f:
                        risk_data = json.load(f)
                    decision = {
                        "verdict": risk_data.get("verdict", "UNKNOWN"),
                        "position": risk_data.get("final_position_dollars", 0),
                        "confidence": risk_data.get("confidence", "LOW")
                    }
                except:
                    decision = {"verdict": "UNKNOWN", "position": 0, "confidence": "LOW"}
                
                # Step 4: Create summary.json
                summary = {
                    "ticker": ticker,
                    "date": date_str,
                    "sample_number": sample_num,
                    "actual_day_number": actual_day,
                    "portfolio_size": portfolio_size,
                    "market_data": market_data,
                    "timestamp": datetime.now().isoformat(),
                    "files_saved": len(list(portfolio_folder.glob("*.json"))),
                    "path": str(portfolio_folder.relative_to(outputs_path)),
                    "decision": decision
                }
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"  ✓ summary.json")
                fixed += 1
    
    print(f"\n{'='*50}")
    print(f"Needs fixing: {needs_fix}")
    print(f"Fixed: {fixed}")
    print(f"Skipped (already complete): {skipped}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    fix_missing()