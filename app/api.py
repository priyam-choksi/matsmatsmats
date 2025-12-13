"""FastAPI Worker API for trading proposals and webhooks."""

import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# ============================================================
# FIX PATH ISSUES - Ensure correct working directory
# ============================================================
# Get the TradingAgent root directory (parent of 'app' folder)
CURRENT_FILE = Path(__file__).resolve()
APP_FOLDER = CURRENT_FILE.parent  # app/
PROJECT_ROOT = APP_FOLDER.parent   # TradingAgent/

# Change to project root so all relative paths work
os.chdir(PROJECT_ROOT)

# Add to Python path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"[API INIT] Working directory: {os.getcwd()}")
print(f"[API INIT] Project root: {PROJECT_ROOT}")

# Now import after path is set
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.orchestrators.master_orchestrator import MasterOrchestrator

# Initialize FastAPI app
app = FastAPI(
    title="Trade Arena - AI Trading Agent API",
    description="11-Agent LLM Pipeline for Market Analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json_file(filepath: Path) -> Optional[Dict]:
    """Safely load a JSON file, return None if not found or invalid."""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
    return None


def collect_phase_outputs(outputs_path: Path) -> Dict[str, Any]:
    """Collect all output files from the pipeline run."""
    outputs = {}
    
    print(f"[API] Collecting outputs from: {outputs_path}")
    
    # Phase 1: Analysts - discussion_points.json
    discussion = load_json_file(outputs_path / "discussion_points.json")
    if discussion:
        outputs['phase1'] = {
            'discussion_points': discussion,
            'technical': discussion.get('technical') or discussion.get('Technical Analyst'),
            'news': discussion.get('news') or discussion.get('News Analyst'),
            'fundamental': discussion.get('fundamental') or discussion.get('Fundamental Analyst'),
            'macro': discussion.get('macro') or discussion.get('Macro Analyst'),
        }
    
    # Phase 2: Researchers
    bull_thesis = load_json_file(outputs_path / "bull_thesis.json")
    bear_thesis = load_json_file(outputs_path / "bear_thesis.json")
    if bull_thesis or bear_thesis:
        outputs['phase2'] = {
            'bull': bull_thesis,
            'bear': bear_thesis,
        }
    
    # Phase 3: Research Manager
    synthesis = load_json_file(outputs_path / "research_synthesis.json")
    if synthesis:
        outputs['phase3'] = {
            'synthesis': synthesis,
        }
    
    # Phase 4: Risk Team
    aggressive = load_json_file(outputs_path / "aggressive_eval.json")
    neutral = load_json_file(outputs_path / "neutral_eval.json")
    conservative = load_json_file(outputs_path / "conservative_eval.json")
    if aggressive or neutral or conservative:
        outputs['phase4'] = {
            'aggressive': aggressive,
            'neutral': neutral,
            'conservative': conservative,
        }
    
    # Phase 5: Risk Manager
    decision = load_json_file(outputs_path / "risk_decision.json")
    if decision:
        outputs['phase5'] = {
            'decision': decision,
        }
    
    print(f"[API] Collected {len(outputs)} phase outputs")
    return outputs


@app.post("/master_orch")
async def master_orchestrator_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol"),
    port_val: int = Query(100000, description="Portfolio value in dollars"),
    research_mode: str = Query("shallow", description="Research depth: shallow, deep, or research"),
    analysis_date: Optional[str] = Query(None, description="Historical analysis date (YYYY-MM-DD)")
) -> Dict[str, Any]:
    """
    Master orchestrator endpoint - runs complete trading system pipeline.
    """
    # Validate inputs
    if not ticker or len(ticker.strip()) == 0:
        raise HTTPException(status_code=400, detail="ticker is required")
    
    if port_val <= 0:
        raise HTTPException(status_code=400, detail="port_val must be greater than 0")
    
    if research_mode not in ['shallow', 'deep', 'research']:
        raise HTTPException(status_code=400, detail="research_mode must be shallow, deep, or research")
    
    ticker = ticker.upper().strip()
    
    print(f"\n{'='*60}")
    print(f"🚀 MASTER ORCHESTRATOR STARTED")
    print(f"{'='*60}")
    print(f"   Ticker: {ticker}")
    print(f"   Portfolio: ${port_val:,}")
    print(f"   Mode: {research_mode}")
    print(f"   Working Dir: {os.getcwd()}")
    if analysis_date:
        print(f"   Historical Date: {analysis_date}")
    print(f"{'='*60}\n")
    
    try:
        # Create orchestrator
        orchestrator = MasterOrchestrator(
            ticker=ticker,
            portfolio_value=port_val,
            research_mode=research_mode,
            analysis_date=analysis_date
        )
        
        # Track results for each phase
        phase_results = {}
        
        phases = [
            (1, "Analysts", orchestrator.run_phase1_analysts),
            (2, "Researchers", orchestrator.run_phase2_researchers),
            (3, "Research Manager", orchestrator.run_phase3_research_manager),
            (4, "Risk Team", orchestrator.run_phase4_risk_team),
            (5, "Risk Manager", orchestrator.run_phase5_risk_manager),
        ]
        
        for phase_num, phase_name, phase_func in phases:
            print(f"\n{'─'*50}")
            print(f"📍 Phase {phase_num}: {phase_name}")
            print(f"{'─'*50}")
            
            try:
                success = phase_func()
                
                if success:
                    phase_results[phase_num] = {
                        "status": "SUCCESS",
                        "name": phase_name,
                        "message": f"Phase {phase_num} completed successfully"
                    }
                    print(f"✅ Phase {phase_num} ({phase_name}) - SUCCESS")
                else:
                    phase_results[phase_num] = {
                        "status": "FAILED",
                        "name": phase_name,
                        "message": f"Phase {phase_num} returned failure"
                    }
                    print(f"❌ Phase {phase_num} ({phase_name}) - FAILED")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Phase {phase_num} ({phase_name}) failed"
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ Phase {phase_num} ({phase_name}) - ERROR:")
                traceback.print_exc()
                phase_results[phase_num] = {
                    "status": "ERROR",
                    "name": phase_name,
                    "message": str(e)
                }
                raise HTTPException(
                    status_code=500,
                    detail=f"Phase {phase_num} ({phase_name}) error: {str(e)}"
                )
        
        # Collect all output files
        phase_outputs = collect_phase_outputs(orchestrator.outputs_path)
        
        # Merge outputs into phase_results
        for phase_key, output_data in phase_outputs.items():
            phase_num = int(phase_key.replace('phase', ''))
            if phase_num in phase_results:
                phase_results[phase_num]['data'] = output_data
        
        # Get final decision
        final_decision = None
        decision_file = orchestrator.outputs_path / "risk_decision.json"
        if decision_file.exists():
            with open(decision_file, 'r', encoding='utf-8') as f:
                final_decision = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"🎯 PIPELINE COMPLETE")
        print(f"{'='*60}")
        if final_decision:
            print(f"   Verdict: {final_decision.get('verdict', 'N/A')}")
            print(f"   Confidence: {final_decision.get('confidence', 'N/A')}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "ticker": ticker,
            "portfolio_value": port_val,
            "research_mode": research_mode,
            "analysis_date": analysis_date,
            "message": f"Master orchestration completed for {ticker}",
            "phase_results": phase_results,
            "final_decision": final_decision,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ MASTER ORCHESTRATION ERROR:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Master orchestration error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Trade Arena API",
        "version": "1.0.0",
        "working_dir": os.getcwd(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Trade Arena API",
        "description": "11-Agent LLM Pipeline for Market Analysis",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)