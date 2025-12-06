# generate_manifest.py
import os, json
from pathlib import Path

workflows_path = Path("public/data/workflows")
manifest = []

print("Scanning workflows...")
for ticker_dir in workflows_path.iterdir():
    if not ticker_dir.is_dir():
        continue
    ticker = ticker_dir.name
    print(f"  {ticker}...", end=" ", flush=True)
    count = 0
    
    for sample_dir in ticker_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        
        portfolio_dir = sample_dir / "portfolio_100000"
        data_dir = portfolio_dir if portfolio_dir.is_dir() else sample_dir
        folder = f"{sample_dir.name}/portfolio_100000" if portfolio_dir.is_dir() else sample_dir.name
        
        summary_path = data_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    s = json.load(f)
                    manifest.append({
                        "ticker": ticker,
                        "folder": folder,
                        "date": s.get("date", ""),
                        "sample": s.get("sample_number", 0),
                        "day": s.get("actual_day_number", 0),
                        "verdict": s.get("decision", {}).get("verdict", "HOLD"),
                        "confidence": s.get("decision", {}).get("confidence", "MEDIUM"),
                        "position": s.get("decision", {}).get("position", 0)
                    })
                    count += 1
            except Exception as e:
                pass
    print(f"{count} samples")

# Sort by date descending
manifest.sort(key=lambda x: x["date"], reverse=True)

# Save manifest
output_path = Path("public/data/workflows/manifest.json")
with open(output_path, "w") as f:
    json.dump(manifest, f)

print(f"\nDone! Saved {len(manifest)} workflows to {output_path}")