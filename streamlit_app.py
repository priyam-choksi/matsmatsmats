import streamlit as st
import subprocess
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import sys
import os
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Fix for Windows encoding issues
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

# Import OpenAI for LLM explanations
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Game Theory Trading System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional trading theme with proper contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1976D2 0%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #388E3C;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1565C0 0%, #2E7D32 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .phase-card {
        background: #FAFAFA;
        border-left: 5px solid #E0E0E0;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s;
        color: #212121 !important;
    }
    
    .phase-card.running {
        border-left-color: #2196F3;
        background: linear-gradient(90deg, rgba(33, 150, 243, 0.15) 0%, #FAFAFA 100%);
        animation: pulse 2s infinite;
    }
    
    .phase-card.success {
        border-left-color: #4CAF50;
        background: linear-gradient(90deg, rgba(76, 175, 80, 0.15) 0%, #FAFAFA 100%);
    }
    
    .phase-card.error {
        border-left-color: #F44336;
        background: linear-gradient(90deg, rgba(244, 67, 54, 0.15) 0%, #FAFAFA 100%);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    .success-box {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #F57C00 0%, #FF9800 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white !important;
    }
    
    .log-container {
        background: #263238;
        color: #B2FF59 !important;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #37474F;
    }
    
    .llm-explanation {
        background: linear-gradient(135deg, #E3F2FD 0%, #E8F5E9 100%);
        border: 2px solid #1976D2;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #212121 !important;
        font-weight: 500;
    }
    
    /* Fix for Streamlit metric labels */
    [data-testid="metric-container"] {
        background-color: rgba(240, 240, 240, 0.9);
        border: 1px solid #E0E0E0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Status message styling */
    .status-running {
        background: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 0.75rem;
        border-radius: 4px;
        color: #0D47A1 !important;
    }
    
    .status-success {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 0.75rem;
        border-radius: 4px;
        color: #1B5E20 !important;
    }
    
    .status-error {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 0.75rem;
        border-radius: 4px;
        color: #B71C1C !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'phase_status' not in st.session_state:
    st.session_state.phase_status = {}
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'outputs' not in st.session_state:
    st.session_state.outputs = {}
if 'phase_outputs' not in st.session_state:
    st.session_state.phase_outputs = {}
if 'llm_explanations' not in st.session_state:
    st.session_state.llm_explanations = {}

# Phase definitions with correct paths
PHASES = {
    'phase1': {
        'name': 'Analyst Hub',
        'agents': ['Technical', 'Fundamental', 'News', 'Macro'],
        'description': 'Running 4 analyst agents',
        'icon': '📊',
        'timeout': 180,
        'output_files': ['discussion_points.json'],
        'script': 'agents/orchestrators/discussion_hub.py'
    },
    'phase2': {
        'name': 'Researchers',
        'agents': ['Bull Researcher', 'Bear Researcher'],
        'description': 'Bull vs Bear debate',
        'icon': '🔍',
        'timeout': 300,
        'output_files': ['bull_thesis.json', 'bear_thesis.json'],
        'scripts': {
            'bull': 'agents/researcher/bull_researcher.py',
            'bear': 'agents/researcher/bear_researcher.py'
        }
    },
    'phase3': {
        'name': 'Research Manager',
        'agents': ['Research Synthesizer'],
        'description': 'Synthesizing probabilities',
        'icon': '📈',
        'timeout': 120,
        'output_files': ['research_synthesis.json'],
        'script': 'agents/managers/research_manager.py'
    },
    'phase4': {
        'name': 'Risk Team',
        'agents': ['Aggressive', 'Neutral', 'Conservative'],
        'description': '3 risk perspectives',
        'icon': '⚖️',
        'timeout': 90,
        'output_files': ['aggressive_eval.json', 'neutral_eval.json', 'conservative_eval.json'],
        'scripts': {
            'aggressive': 'agents/risk_management/aggressive_debator.py',
            'neutral': 'agents/risk_management/neutral_debator.py',
            'conservative': 'agents/risk_management/conservative_debator.py'
        }
    },
    'phase5': {
        'name': 'Risk Manager',
        'agents': ['Final Decision'],
        'description': 'Risk-adjusted position',
        'icon': '🛡️',
        'timeout': 120,
        'output_files': ['risk_decision.json'],
        'script': 'agents/managers/risk_manager.py'
    },
    'phase6': {
        'name': 'Game Theory',
        'agents': ['5 Strategies'],
        'description': 'Tournament comparison',
        'icon': '🎮',
        'timeout': 90,
        'output_files': ['game_theory_tournament.json'],
        'script': 'agents/orchestrators/game_theory_orchestrator.py'
    }
}

class LLMExplainer:
    """Uses OpenAI to explain outputs in simple terms"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.model = "gpt-4o-mini"  # Use cheapest model
            except Exception as e:
                st.warning(f"Could not initialize OpenAI: {e}")
    
    def explain(self, data: Dict, context: str) -> str:
        """Generate simple explanation of complex data"""
        if not self.client:
            return None
            
        try:
            prompt = f"""You are explaining trading analysis results to someone who understands investing but needs clarity on the specific findings.

Context: {context}

Data to explain:
{json.dumps(data, indent=2)[:2000]}  # Limit data size

Provide a 2-3 sentence explanation that:
1. Highlights the key finding or decision
2. Explains WHY this conclusion was reached
3. Notes any important caveats or risks

Be concise and use simple language. Focus on actionable insights."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful trading analyst explaining complex results simply."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Could not generate explanation: {e}"

def get_project_root() -> Path:
    """Find the project root directory"""
    current = Path.cwd()
    
    # Check various possible locations
    if (current / 'agents').exists():
        return current
    elif (current.parent / 'agents').exists():
        return current.parent
    elif (current.parent.parent / 'agents').exists():
        return current.parent.parent
    else:
        # Try to find by looking for key files
        for parent in current.parents:
            if (parent / 'agents' / 'orchestrators').exists():
                return parent
    
    # Default to current directory
    return current

def safe_run_command(cmd, cwd=None, timeout=120):
    """Run command with proper encoding and error handling"""
    try:
        # Fix paths and use correct Python interpreter
        if cmd[0] in ['python', 'python3']:
            cmd[0] = sys.executable
        
        # Set UTF-8 environment
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        # Handle path conversions
        if cwd:
            cwd = Path(cwd)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except FileNotFoundError as e:
        return False, "", f"File not found: {e}"
    except Exception as e:
        return False, "", f"Error: {str(e)}"

def run_phase_command(phase_id, ticker, research_mode='shallow', portfolio_value=100000):
    """Run specific phase with correct paths and parameters"""
    project_root = get_project_root()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    phase_info = PHASES[phase_id]
    
    try:
        if phase_id == 'phase1':
            script_path = project_root / phase_info['script']
            cmd = [
                sys.executable,
                str(script_path),
                ticker,
                "--run-analysts",
                "--output", str(outputs_dir / "discussion_points.json"),
                "--format", "json"
            ]
            success, stdout, stderr = safe_run_command(cmd, cwd=script_path.parent, timeout=phase_info['timeout'])
            return success, {'stdout': stdout, 'stderr': stderr}
            
        elif phase_id == 'phase2':
            results = {}
            
            # Run bear researcher
            bear_script = project_root / phase_info['scripts']['bear']
            bear_cmd = [
                sys.executable,
                str(bear_script),
                ticker,
                "--discussion-file", str(outputs_dir / "discussion_points.json"),
                "--mode", research_mode,
                "--save-data", str(outputs_dir / "bear_thesis.json")
            ]
            bear_success, bear_out, bear_err = safe_run_command(bear_cmd, cwd=bear_script.parent, timeout=phase_info['timeout'])
            results['bear'] = {'success': bear_success, 'stdout': bear_out, 'stderr': bear_err}
            
            # Run bull researcher
            bull_script = project_root / phase_info['scripts']['bull']
            bull_cmd = [
                sys.executable,
                str(bull_script),
                ticker,
                "--discussion-file", str(outputs_dir / "discussion_points.json"),
                "--mode", research_mode,
                "--save-data", str(outputs_dir / "bull_thesis.json")
            ]
            bull_success, bull_out, bull_err = safe_run_command(bull_cmd, cwd=bull_script.parent, timeout=phase_info['timeout'])
            results['bull'] = {'success': bull_success, 'stdout': bull_out, 'stderr': bull_err}
            
            return bear_success and bull_success, results
            
        elif phase_id == 'phase3':
            script_path = project_root / phase_info['script']
            cmd = [
                sys.executable,
                str(script_path),
                ticker,
                "--bull-file", str(outputs_dir / "bull_thesis.json"),
                "--bear-file", str(outputs_dir / "bear_thesis.json"),
                "--save-synthesis", str(outputs_dir / "research_synthesis.json")
            ]
            success, stdout, stderr = safe_run_command(cmd, cwd=script_path.parent, timeout=phase_info['timeout'])
            return success, {'stdout': stdout, 'stderr': stderr}
            
        elif phase_id == 'phase4':
            results = {}
            for stance in ['aggressive', 'neutral', 'conservative']:
                script_path = project_root / phase_info['scripts'][stance]
                cmd = [
                    sys.executable,
                    str(script_path),
                    ticker,
                    "--synthesis-file", str(outputs_dir / "research_synthesis.json"),
                    "--save-evaluation", str(outputs_dir / f"{stance}_eval.json")
                ]
                success, stdout, stderr = safe_run_command(cmd, cwd=script_path.parent, timeout=phase_info['timeout'])
                results[stance] = {'success': success, 'stdout': stdout, 'stderr': stderr}
            
            success_count = sum(1 for r in results.values() if r['success'])
            return success_count >= 2, results
            
        elif phase_id == 'phase5':
            script_path = project_root / phase_info['script']
            cmd = [
                sys.executable,
                str(script_path),
                ticker,
                "--synthesis-file", str(outputs_dir / "research_synthesis.json"),
                "--portfolio-value", str(portfolio_value),
                "--save-decision", str(outputs_dir / "risk_decision.json")
            ]
            success, stdout, stderr = safe_run_command(cmd, cwd=script_path.parent, timeout=phase_info['timeout'])
            return success, {'stdout': stdout, 'stderr': stderr}
            
        elif phase_id == 'phase6':
            script_path = project_root / phase_info['script']
            cmd = [
                sys.executable,
                str(script_path),
                ticker,
                "--outputs-dir", str(outputs_dir)
            ]
            success, stdout, stderr = safe_run_command(cmd, cwd=script_path.parent, timeout=phase_info['timeout'])
            return success, {'stdout': stdout, 'stderr': stderr}
            
    except Exception as e:
        return False, {'error': str(e), 'traceback': traceback.format_exc()}

def load_output_file(filename):
    """Safely load a JSON output file"""
    try:
        project_root = get_project_root()
        filepath = project_root / "outputs" / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                if content.strip():
                    return json.loads(content)
        return None
    except json.JSONDecodeError as e:
        st.warning(f"JSON decode error in {filename}: {e}")
        return None
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        return None

def load_all_outputs():
    """Load all output files from outputs directory"""
    outputs = {}
    output_files = [
        'discussion_points.json',
        'bear_thesis.json', 
        'bull_thesis.json',
        'research_synthesis.json',
        'aggressive_eval.json',
        'neutral_eval.json',
        'conservative_eval.json',
        'risk_decision.json',
        'game_theory_tournament.json'
    ]
    
    for filename in output_files:
        key = filename.replace('.json', '')
        data = load_output_file(filename)
        if data:
            outputs[key] = data
            st.session_state.logs.append(f"✅ Loaded {filename}")
    
    return outputs

def create_phase_progress_visual():
    """Create visual progress indicator for all phases"""
    cols = st.columns(6)
    
    for i, (phase_id, phase) in enumerate(PHASES.items()):
        status = st.session_state.phase_status.get(phase_id, 'pending')
        
        with cols[i]:
            # Determine status styling
            if status == 'SUCCESS':
                bg_color = "#E8F5E9"
                border_color = "#4CAF50"
                text_color = "#1B5E20"
                icon_status = "✅"
            elif status == 'RUNNING':
                bg_color = "#E3F2FD"
                border_color = "#2196F3"
                text_color = "#0D47A1"
                icon_status = "⏳"
            elif status == 'ERROR':
                bg_color = "#FFEBEE"
                border_color = "#F44336"
                text_color = "#B71C1C"
                icon_status = "❌"
            else:
                bg_color = "#F5F5F5"
                border_color = "#BDBDBD"
                text_color = "#616161"
                icon_status = "⏸️"
            
            st.markdown(f"""
            <div style="
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 0.75rem;
                text-align: center;
                margin: 0.25rem;
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">{phase['icon']}</div>
                <div style="font-weight: 600; color: {text_color}; font-size: 0.9rem;">{phase['name']}</div>
                <div style="font-size: 1.5rem; margin-top: 0.25rem;">{icon_status}</div>
            </div>
            """, unsafe_allow_html=True)

def display_phase_output(phase_id, llm_explainer=None):
    """Display outputs for a specific phase - NO NESTED EXPANDERS"""
    phase = PHASES[phase_id]
    
    for output_file in phase['output_files']:
        key = output_file.replace('.json', '')
        if key in st.session_state.outputs:
            data = st.session_state.outputs[key]
            
            st.subheader(f"📄 {output_file}")
            
            # Display data visualization
            if key == 'discussion_points':
                display_discussion_points(data)
            elif key in ['bull_thesis', 'bear_thesis']:
                display_thesis(data, key)
            elif key == 'research_synthesis':
                display_synthesis(data)
            elif key == 'risk_decision':
                display_risk_decision(data)
            elif key == 'game_theory_tournament':
                display_game_theory(data)
            elif key in ['aggressive_eval', 'neutral_eval', 'conservative_eval']:
                display_risk_eval(data, key)
            
            # Add LLM explanation if available
            if llm_explainer and st.checkbox(f"🤖 Explain {key}", key=f"explain_{phase_id}_{key}"):
                explanation_key = f"{phase_id}_{key}"
                if explanation_key not in st.session_state.llm_explanations:
                    with st.spinner("Generating explanation..."):
                        explanation = llm_explainer.explain(data, f"Phase {phase_id} - {phase['name']}: {output_file}")
                        if explanation:
                            st.session_state.llm_explanations[explanation_key] = explanation
                
                if explanation_key in st.session_state.llm_explanations:
                    st.info(st.session_state.llm_explanations[explanation_key])

def map_confidence_to_score(conf_label: str) -> float:
    """Map string confidence labels to a 0-1 score for display."""
    if not conf_label:
        return 0.0
    label = str(conf_label).upper()
    mapping = {
        "VERY LOW": 0.2,
        "LOW": 0.3,
        "MEDIUM": 0.6,
        "HIGH": 0.8,
        "VERY HIGH": 0.9,
    }
    return mapping.get(label, 0.5)

def display_discussion_points(data):
    """Display formatted discussion points from analyst hub"""
    if not data:
        st.warning("No discussion data available")
        return
    
    summary = data.get('summary', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sentiment = summary.get('net_sentiment') or "N/A"
        st.metric("Net Sentiment", sentiment)
    with col2:
        bull = summary.get('bull_signal_count')
        st.metric("Bull Signals", bull if bull is not None else "N/A")
    with col3:
        bear = summary.get('bear_signal_count')
        st.metric("Bear Signals", bear if bear is not None else "N/A")
    with col4:
        if isinstance(bull, (int, float)) and isinstance(bear, (int, float)) and bear != 0:
            ratio = bull / bear
            st.metric("Bull/Bear Ratio", f"{ratio:.2f}")
        else:
            st.metric("Bull/Bear Ratio", "N/A")
    
    # Analyst recommendations
    recs = summary.get('recommendations', {}) or {}
    if recs:
        st.subheader("📊 Analyst Recommendations")
        rec_cols = st.columns(4)
        for i, (analyst, rec) in enumerate(recs.items()):
            with rec_cols[i % 4]:
                emoji = "🟢" if rec == "BUY" else "🔴" if rec == "SELL" else "🟡"
                st.write(f"{emoji} **{analyst.title()}**: {rec}")
    
    # Consensus points (if present in your discussion_points JSON)
    if 'consensus_points' in data and data['consensus_points']:
        st.subheader("🤝 Consensus Points")
        for point in data['consensus_points'][:3]:
            # description already has things like "Strong consensus: HOLD (4/4 analysts)"
            st.write(f"• {point.get('description', 'N/A')}")


def display_thesis(data, thesis_type):
    """Display bull or bear thesis with fields that actually exist in JSON."""
    if not data:
        st.warning(f"No {thesis_type} data available")
        return
    
    is_bull = "bull" in thesis_type
    title = "🐂 Bull Thesis" if is_bull else "🐻 Bear Thesis"
    color = "#4CAF50" if is_bull else "#F44336"
    
    st.markdown(f"<h3 style='color: {color};'>{title}</h3>", unsafe_allow_html=True)
    
    # High-level context from mode + analyst summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mode", data.get("mode", "N/A"))
    with col2:
        summary = data.get("discussion_summary", {}) or {}
        recs = summary.get("recommendations", {}) or {}
        if recs:
            stance_str = ", ".join(f"{k}: {v}" for k, v in recs.items())
        else:
            stance_str = "N/A"
        st.metric("Analyst Stance", stance_str)
    
    # Core thesis
    st.write("**Core Thesis:**")
    st.write(data.get("core_thesis", "N/A"))
    
    # Bull-specific details
    if is_bull:
        opportunities = data.get("opportunities") or {}
        if opportunities:
            st.write("**Bull Opportunities:**")
            for bucket, items in opportunities.items():
                st.markdown(f"**{bucket.replace('_', ' ').title()}**")
                for item in items[:5]:
                    st.write(f"• {item}")
        
        entry_strats = data.get("entry_strategies") or []
        if entry_strats:
            st.write("**Entry Strategies:**")
            for strat in entry_strats:
                st.write(f"- **{strat.get('strategy', 'N/A')}**: {strat.get('description', '')}")
    
    # Bear-specific details
    if not is_bull:
        risks = data.get("risks") or {}
        if risks:
            st.write("**Key Risks:**")
            for bucket, items in risks.items():
                st.markdown(f"**{bucket.replace('_', ' ').title()}**")
                for item in items[:5]:
                    st.write(f"• {item}")
        
        downside_triggers = data.get("downside_triggers") or []
        if downside_triggers:
            st.write("**Downside Triggers:**")
            for trig in downside_triggers:
                desc = trig.get("description", "N/A")
                impact = trig.get("impact")
                timeline = trig.get("timeline")
                prob = trig.get("probability")

                details = []
                if impact:
                    details.append(f"Impact: {impact}")
                if timeline:
                    details.append(f"Timeline: {timeline}")
                if prob:
                    details.append(f"Probability: {prob}")

                if details:
                    line = f"{desc} ({', '.join(details)})"
                else:
                    line = desc

                st.write(f"- {line}")


def display_synthesis(data):
    """Display research synthesis with probability gauges aligned to JSON schema."""
    if not data:
        st.warning("No synthesis data available")
        return
    
    probs = data.get("probabilities", {}) or {}
    bull_prob = float(probs.get("bull_case", 0.0))
    bear_prob = float(probs.get("bear_case", 0.0))
    
    conclusion = data.get("conclusion", {}) or {}
    conf_label = conclusion.get("confidence")
    conf_score = map_confidence_to_score(conf_label)
    
    # Create probability gauge chart
    fig = go.Figure()
    
    # Bull probability gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bull_prob,
        title={'text': "Bull Case"},
        domain={'x': [0, 0.45], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 33], 'color': "#FFEBEE"},
                {'range': [33, 66], 'color': "#FFF3E0"},
                {'range': [66, 100], 'color': "#E8F5E9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    # Bear probability gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bear_prob,
        title={'text': "Bear Case"},
        domain={'x': [0.55, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#F44336"},
            'steps': [
                {'range': [0, 33], 'color': "#E8F5E9"},
                {'range': [33, 66], 'color': "#FFF3E0"},
                {'range': [66, 100], 'color': "#FFEBEE"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence and recommendation
    col1, col2 = st.columns(2)
    with col1:
        if conf_label:
            st.metric("Confidence", f"{conf_score*100:.1f}% ({conf_label})")
        else:
            st.metric("Confidence", "N/A")
    with col2:
        st.metric("Recommendation", conclusion.get("recommendation", "N/A"))
    
    # Key insights from catalysts and risks
    catalysts = conclusion.get("key_catalysts") or []
    risks = conclusion.get("key_risks") or []
    if catalysts or risks:
        st.write("**Key Insights:**")
        for c in catalysts[:2]:
            st.write(f"• Catalyst: {c}")
        for r in risks[:2]:
            st.write(f"• Risk: {r}")


def display_risk_decision(data):
    """Display final risk decision with position sizing"""
    if not data:
        st.warning("No risk decision data available")
        return
    
    verdict = data.get('verdict', 'N/A')
    
    # Decision banner
    if verdict == "BUY":
        st.success(f"## 🟢 DECISION: {verdict}")
    elif verdict == "SELL":
        st.error(f"## 🔴 DECISION: {verdict}")
    else:
        st.info(f"## 🟡 DECISION: {verdict}")
    
    position_dollars = data.get("final_position_dollars", 0.0)
    position_pct = data.get("final_position_pct", None)
    portfolio_value = data.get("portfolio_value", 0.0)
    
    # Position metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Position Size ($)", f"${position_dollars:,.0f}")
    with col2:
        if isinstance(position_pct, (int, float)):
            st.metric("Position Size (%)", f"{position_pct*100:.2f}%")
        else:
            st.metric("Position Size (%)", "N/A")
    with col3:
        st.metric("Portfolio Value", f"${portfolio_value:,.0f}")
    with col4:
        st.metric("Confidence", data.get('confidence', 'N/A'))
    
    # Risk controls (stop loss, take profit, time limit)
    controls = data.get("risk_controls") or {}
    if controls:
        st.write("### Risk Controls")
        
        stop = controls.get("stop_loss", {}) or {}
        if stop:
            st.write(f"**Stop Loss:** {stop.get('percentage', 'N/A')}% ({stop.get('type', 'N/A')})")
        
        tps = controls.get("take_profit") or []
        if tps:
            st.write("**Take Profit Levels:**")
            for tp in tps:
                st.write(
                    f"- Level {tp.get('level')}: target {tp.get('target_pct')}%, "
                    f"exit {tp.get('exit_pct')}% ({tp.get('reasoning', '')})"
                )
        
        time_limit = controls.get("time_limit") or {}
        if time_limit:
            st.write("**Time Limit:**")
            st.write(f"- Max holding: {time_limit.get('max_holding_period', 'N/A')}")
            st.write(f"- Review frequency: {time_limit.get('review_frequency', 'N/A')}")
            if time_limit.get("mandatory_review"):
                st.write(f"- Mandatory review: {time_limit.get('mandatory_review')}")

def display_game_theory(data):
    """Display game theory tournament results"""
    if not data:
        st.warning("No game theory data available")
        return
    
    # Winner announcement
    winner = data.get('recommended_strategy', 'N/A')
    st.success(f"# 🏆 WINNER: {winner.upper()}")
    
    regime = data.get('regime', 'N/A')
    st.info(f"**Market Regime:** {regime}")
    
    # Strategy comparison
    if 'strategies' in data:
        strategies = data['strategies']
        
        # Create comparison dataframe
        strategy_data = []
        for name, strat in strategies.items():
            strategy_data.append({
                'Strategy': name.replace('_', ' ').title(),
                'Action': strat.get('action', 'N/A'),
                'Position Size': f"${strat.get('position_size', 0):,.0f}",
                'Confidence': f"{strat.get('confidence', 0)*100:.1f}%"
            })
        
        df = pd.DataFrame(strategy_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Position Sizes by Strategy', 'Confidence Levels'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        names = [s['Strategy'] for s in strategy_data]
        positions = [strategies[k].get('position_size', 0) for k in strategies.keys()]
        confidences = [strategies[k].get('confidence', 0)*100 for k in strategies.keys()]
        
        # Position sizes bar chart
        fig.add_trace(
            go.Bar(
                x=names,
                y=positions,
                marker_color=['#4CAF50' if n.lower() == winner.replace('_', ' ') else '#90CAF9' for n in names],
                text=[f"${p:,.0f}" for p in positions],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Confidence bar chart
        fig.add_trace(
            go.Bar(
                x=names,
                y=confidences,
                marker_color=['#FF9800' if n.lower() == winner.replace('_', ' ') else '#FFE082' for n in names],
                text=[f"{c:.1f}%" for c in confidences],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Position ($)", row=1, col=1)
        fig.update_yaxes(title_text="Confidence (%)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)

def display_risk_eval(data, eval_type):
    """Display risk evaluation from each stance aligned to JSON schema."""
    if not data:
        st.warning(f"No {eval_type} data available")
        return
    
    stance_key = eval_type.replace("_eval", "")
    stance_name = stance_key.title()
    stance_colors = {
        'aggressive': '#FF5252',
        'neutral': '#FFC107',
        'conservative': '#4CAF50'
    }
    color = stance_colors.get(stance_key, '#2196F3')
    
    st.markdown(f"<h4 style='color: {color};'>{stance_name} Risk Assessment</h4>", unsafe_allow_html=True)
    
    pos_size = data.get("position_size", None)
    pos_str = f"{pos_size*100:.1f}%" if isinstance(pos_size, (int, float)) else "N/A"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stance", data.get('stance', 'N/A'))
    with col2:
        st.metric("Position Size", pos_str)
    with col3:
        st.metric("Confidence", data.get('confidence', 'N/A'))
    
    reasoning = data.get("reasoning")
    if reasoning:
        st.write("**Reasoning:**")
        st.write(reasoning)
    
    trading_plan = data.get("trading_plan") or {}
    if trading_plan:
        st.write("**Trading Plan:**")
        st.write(f"- Entry: {trading_plan.get('entry_strategy', 'N/A')}")
        if trading_plan.get('position_building'):
            st.write(f"- Position building: {trading_plan.get('position_building')}")
        st.write(f"- Stop loss: {trading_plan.get('stop_loss', 'N/A')}")
        profit_targets = trading_plan.get('profit_targets') or []
        if profit_targets:
            st.write(f"- Profit targets: {', '.join(profit_targets)}")
        st.write(f"- Time frame: {trading_plan.get('time_frame', 'N/A')}")


def run_complete_pipeline(ticker, research_mode, portfolio_value):
    """Run the complete pipeline with proper error handling and logging"""
    st.session_state.running = True
    st.session_state.start_time = time.time()
    st.session_state.logs = []
    st.session_state.phase_status = {}
    st.session_state.outputs = {}
    
    # Initialize LLM explainer if API key available
    llm_explainer = LLMExplainer() if os.getenv("OPENAI_API_KEY") else None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()
    
    total_phases = len(PHASES)
    critical_failures = []
    
    for i, (phase_id, phase_info) in enumerate(PHASES.items()):
        # Update progress
        progress = (i / total_phases)
        progress_bar.progress(progress)
        
        # Update status
        status_text.info(f"🔄 Running {phase_info['name']}: {phase_info['description']}")
        
        # Update phase status
        st.session_state.phase_status[phase_id] = 'RUNNING'
        st.session_state.current_phase = phase_id
        
        # Log
        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Starting {phase_info['name']}"
        st.session_state.logs.append(log_msg)
        
        # Display recent logs
        with log_container.container():
            st.text_area("Execution Log", "\n".join(st.session_state.logs[-10:]), height=200, key=f"log_{i}")
        
        # Run the phase
        try:
            success, result = run_phase_command(phase_id, ticker, research_mode, portfolio_value)
            
            # Update status
            st.session_state.phase_status[phase_id] = 'SUCCESS' if success else 'ERROR'
            
            # Add result to logs
            status_emoji = "✅" if success else "❌"
            log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {status_emoji} {phase_info['name']} {'completed' if success else 'failed'}"
            st.session_state.logs.append(log_msg)
            
            # Store phase output
            st.session_state.phase_outputs[phase_id] = result
            
            # Load output files for this phase
            for output_file in phase_info['output_files']:
                key = output_file.replace('.json', '')
                data = load_output_file(output_file)
                if data:
                    st.session_state.outputs[key] = data
            
            # Check for critical failure
            if not success and phase_id in ['phase1', 'phase3', 'phase5']:
                critical_failures.append(phase_id)
                error_msg = f"Critical failure in {phase_info['name']}"
                status_text.error(f"❌ Pipeline stopped: {error_msg}")
                st.session_state.logs.append(f"[ERROR] {error_msg}")
                
                # Show error details
                if isinstance(result, dict):
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    if 'stderr' in result and result['stderr']:
                        with st.expander("Error Details"):
                            st.code(result['stderr'][:1000])
                break
                
        except Exception as e:
            st.session_state.phase_status[phase_id] = 'ERROR'
            error_msg = f"Exception in {phase_info['name']}: {str(e)}"
            st.session_state.logs.append(f"[ERROR] {error_msg}")
            st.error(error_msg)
            
            if phase_id in ['phase1', 'phase3', 'phase5']:
                critical_failures.append(phase_id)
                break
        
        # Small delay for visual feedback
        time.sleep(0.5)
    
    # Complete
    progress_bar.progress(1.0)
    elapsed = time.time() - st.session_state.start_time
    
    success_count = sum(1 for s in st.session_state.phase_status.values() if s == 'SUCCESS')
    
    if success_count == total_phases:
        status_text.success(f"✅ Pipeline completed successfully in {elapsed:.1f}s!")
        st.balloons()
    elif critical_failures:
        status_text.error(f"❌ Pipeline failed at critical phase(s): {', '.join(critical_failures)}")
    else:
        status_text.warning(f"⚠️ Pipeline completed with {total_phases - success_count} non-critical errors in {elapsed:.1f}s")
    
    st.session_state.running = False

# Main UI
st.markdown('<h1 class="main-header">Multi Agent Trading System : Axelrod Experiment</h1>', unsafe_allow_html=True)

# Initialize LLM explainer
llm_explainer = LLMExplainer() if os.getenv("OPENAI_API_KEY") else None

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker = st.text_input("Stock Ticker", value="AAPL", disabled=st.session_state.running).upper()
    
    research_mode = st.selectbox(
        "Research Mode",
        ['shallow', 'deep', 'research'],
        disabled=st.session_state.running,
        help="Shallow: 1 round | Deep: 3-5 rounds | Research: 5+ rounds"
    )
    
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000,
        disabled=st.session_state.running
    )
    
    st.divider()
    
    # AI Status
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        st.success("🤖 AI Explanations: Enabled (gpt-4o-mini)")
    elif not OPENAI_AVAILABLE:
        st.info("🤖 OpenAI library not installed")
        st.caption("Run: pip install openai")
    else:
        st.info("🤖 AI Explanations: API key not found")
    
    st.divider()
    
    # Action buttons
    if st.button("🚀 Run Analysis", disabled=st.session_state.running, use_container_width=True, type="primary"):
        run_complete_pipeline(ticker, research_mode, portfolio_value)
    
    if st.button("📥 Load Previous Results", use_container_width=True):
        st.session_state.outputs = load_all_outputs()
        if st.session_state.outputs:
            st.success(f"✅ Loaded {len(st.session_state.outputs)} output files")
            # Update phase status based on loaded files
            for phase_id, phase_info in PHASES.items():
                if all(f.replace('.json', '') in st.session_state.outputs for f in phase_info['output_files']):
                    st.session_state.phase_status[phase_id] = 'SUCCESS'
        else:
            st.warning("⚠️ No output files found in outputs directory")
    
    if st.button("🗑️ Clear All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Pipeline Status",
    "📈 Analysis Results", 
    "🎮 Game Theory",
    "📄 Executive Report",
    "🔍 Raw Data",
    "📋 Logs"
])

with tab1:
    st.header("📊 Pipeline Execution Status")
    
    # Visual progress
    create_phase_progress_visual()
    
    st.divider()
    
    # Phase details
    for phase_id, phase_info in PHASES.items():
        status = st.session_state.phase_status.get(phase_id, 'pending')
        
        # Create expander with status-based styling
        with st.expander(f"{phase_info['icon']} {phase_info['name']} - {status}", expanded=(status == 'RUNNING')):
            # Add status indicator
            if status == 'SUCCESS':
                st.success(f"✅ Phase completed successfully")
            elif status == 'RUNNING':
                st.info(f"⏳ Phase currently running...")
            elif status == 'ERROR':
                st.error(f"❌ Phase encountered an error")
            else:
                st.info(f"⏸️ Phase pending")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Description:** {phase_info['description']}")
                st.write(f"**Agents:** {', '.join(phase_info['agents'])}")
                st.write(f"**Timeout:** {phase_info['timeout']}s")
            
            with col2:
                if status == 'SUCCESS':
                    display_phase_output(phase_id, llm_explainer)
                elif status == 'ERROR' and phase_id in st.session_state.phase_outputs:
                    output = st.session_state.phase_outputs[phase_id]
                    if isinstance(output, dict):
                        if 'stderr' in output and output['stderr']:
                            st.code(output['stderr'][:500], language='text')

with tab2:
    st.header("📈 Detailed Analysis Results")
    
    if not st.session_state.outputs:
        st.info("No results available. Run an analysis or load previous results.")
    else:
        # Phase 1: Discussion Points
        if 'discussion_points' in st.session_state.outputs:
            st.subheader("📊 Phase 1: Analyst Consensus")
            display_discussion_points(st.session_state.outputs['discussion_points'])
            if llm_explainer and st.checkbox("🤖 Explain Analyst Consensus", key="explain_consensus"):
                with st.spinner("Generating explanation..."):
                    explanation = llm_explainer.explain(
                        st.session_state.outputs['discussion_points'],
                        "Analyst consensus from technical, fundamental, news, and macro analysis"
                    )
                    if explanation:
                        st.info(explanation)
            st.divider()
        
        # Phase 2: Bull & Bear Theses
        col1, col2 = st.columns(2)
        with col1:
            if 'bull_thesis' in st.session_state.outputs:
                display_thesis(st.session_state.outputs['bull_thesis'], 'bull_thesis')
        with col2:
            if 'bear_thesis' in st.session_state.outputs:
                display_thesis(st.session_state.outputs['bear_thesis'], 'bear_thesis')
        
        if 'bull_thesis' in st.session_state.outputs or 'bear_thesis' in st.session_state.outputs:
            st.divider()
        
        # Phase 3: Research Synthesis
        if 'research_synthesis' in st.session_state.outputs:
            st.subheader("📈 Phase 3: Research Synthesis")
            display_synthesis(st.session_state.outputs['research_synthesis'])
            st.divider()
        
        # Phase 4: Risk Evaluations
        risk_evals = ['aggressive_eval', 'neutral_eval', 'conservative_eval']
        if any(e in st.session_state.outputs for e in risk_evals):
            st.subheader("⚖️ Phase 4: Risk Team Evaluations")
            cols = st.columns(3)
            for i, eval_type in enumerate(risk_evals):
                if eval_type in st.session_state.outputs:
                    with cols[i]:
                        display_risk_eval(st.session_state.outputs[eval_type], eval_type)
            st.divider()
        
        # Phase 5: Final Decision
        if 'risk_decision' in st.session_state.outputs:
            st.subheader("🛡️ Phase 5: Risk Manager Decision")
            display_risk_decision(st.session_state.outputs['risk_decision'])

with tab3:
    st.header("🎮 Game Theory Tournament Results")
    
    if 'game_theory_tournament' not in st.session_state.outputs:
        st.info("No tournament data. Complete the full pipeline to see game theory results.")
    else:
        display_game_theory(st.session_state.outputs['game_theory_tournament'])
        
        if llm_explainer and st.checkbox("🤖 Explain Game Theory Winner", key="explain_game_theory"):
            with st.spinner("Generating explanation..."):
                explanation = llm_explainer.explain(
                    st.session_state.outputs['game_theory_tournament'],
                    "Game theory tournament comparing 5 trading strategies to find optimal approach"
                )
                if explanation:
                    st.info(explanation)

with tab4:
    st.header("📄 Executive Report")
    
    if not st.session_state.outputs:
        st.info("No data for report. Run an analysis first.")
    else:
        # Generate comprehensive report
        if 'risk_decision' in st.session_state.outputs:
            risk = st.session_state.outputs['risk_decision']
            synthesis = st.session_state.outputs.get('research_synthesis', {})
            game_theory = st.session_state.outputs.get('game_theory_tournament', {})
            
            synthesis_probs = {}
            synthesis_conclusion = {}
            synth_conf_label = None
            synth_conf_score = 0.0

            if synthesis:
                synthesis_probs = synthesis.get("probabilities", {}) or {}
                synthesis_conclusion = synthesis.get("conclusion", {}) or {}
                synth_conf_label = synthesis_conclusion.get("confidence")
                synth_conf_score = map_confidence_to_score(synth_conf_label)
            
                bull_prob = float(synthesis_probs.get("bull_case", 0.0))
                bear_prob = float(synthesis_probs.get("bear_case", 0.0))

            # Position size as fraction of portfolio
            pos_pct = float(risk.get("final_position_pct") or 0.0)
            
            
            # Executive Summary Box
            verdict = risk.get('verdict', 'N/A')
            if verdict == "BUY":
                st.success(f"# Investment Decision: {verdict}")
            elif verdict == "SELL":
                st.error(f"# Investment Decision: {verdict}")
            else:
                st.info(f"# Investment Decision: {verdict}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 💰 Position Details
                """)
                st.write(f"**Recommended Position:** ${risk.get('final_position_dollars', 0):,.0f}")
                st.write(f"**Portfolio Allocation:** {pos_pct*100:.2f}% of portfolio")
                
                if game_theory:
                    st.write(f"**Optimal Strategy:** {game_theory.get('recommended_strategy', 'N/A').replace('_', ' ').title()}")
                    st.write(f"**Market Regime:** {game_theory.get('regime', 'N/A')}")
            
            with col2:
                st.markdown("""
                ### 📊 Market Analysis
                """)
                bull_prob = synthesis_probs.get("bull_case", 0.0)
                bear_prob = synthesis_probs.get("bear_case", 0.0)
                st.write(f"**Bull Case Probability:** {bull_prob:.1f}%")
                st.write(f"**Bear Case Probability:** {bear_prob:.1f}%")
                if synth_conf_label:
                    st.write(f"**Analysis Confidence:** {synth_conf_score*100:.1f}% ({synth_conf_label})")
                else:
                    st.write("**Analysis Confidence:** N/A")
                st.write(f"**Recommendation:** {synthesis_conclusion.get('recommendation', 'N/A')}")

            
            # Risk Assessment
            st.markdown("### ⚠️ Risk Assessment")
            if 'risk_assessment' in risk:
                risk_data = risk['risk_assessment']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Downside Risk", risk_data.get('downside_risk', 'N/A'))
                with col2:
                    st.metric("Upside Potential", risk_data.get('upside_potential', 'N/A'))
                with col3:
                    st.metric("Risk/Reward", risk_data.get('risk_reward_ratio', 'N/A'))
            
            # Download report button
            report_text = f"""
GAME THEORY TRADING SYSTEM - EXECUTIVE REPORT
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {ticker}

INVESTMENT DECISION: {verdict}
============================
Position Size ($): ${risk.get('final_position_dollars', 0):,.0f}
Position Size (% of portfolio): {pos_pct*100:.2f}%

MARKET ANALYSIS
===============
Bull Probability: {synthesis_probs.get("bull_case", 0.0):.1f}%
Bear Probability: {synthesis_probs.get("bear_case", 0.0):.1f}%
Confidence: {synth_conf_score*100:.1f}% ({synth_conf_label or "N/A"})
Recommendation: {synthesis_conclusion.get("recommendation", "N/A")}

GAME THEORY RESULT
==================
Optimal Strategy: {game_theory.get('recommended_strategy', 'N/A') if game_theory else 'N/A'}
Market Regime: {game_theory.get('regime', 'N/A') if game_theory else 'N/A'}
"""
        
            st.download_button(
                label="📥 Download Full Report",
                data=report_text,
                file_name=f"{ticker}_trading_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

with tab5:
    st.header("🔍 Raw Output Data")
    
    if st.session_state.outputs:
        selected_output = st.selectbox(
            "Select Output File",
            list(st.session_state.outputs.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()}.json"
        )
        
        if selected_output:
            st.subheader(f"📁 {selected_output}.json")
            
            # Pretty print JSON
            json_str = json.dumps(st.session_state.outputs[selected_output], indent=2)
            st.code(json_str, language='json')
            
            # Download button
            st.download_button(
                label=f"📥 Download {selected_output}.json",
                data=json_str,
                file_name=f"{selected_output}.json",
                mime="application/json"
            )
    else:
        st.info("No output files available.")

with tab6:
    st.header("📋 Complete Execution Log")
    
    if st.session_state.logs:
        # Log statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            success_logs = sum(1 for log in st.session_state.logs if '✅' in log)
            st.metric("Successful Steps", success_logs)
        with col2:
            error_logs = sum(1 for log in st.session_state.logs if '❌' in log or 'ERROR' in log)
            st.metric("Errors", error_logs)
        with col3:
            st.metric("Total Log Entries", len(st.session_state.logs))
        
        # Full log display
        log_text = "\n".join(st.session_state.logs)
        st.text_area("Full Execution Log", log_text, height=400)
        
        # Download logs
        st.download_button(
            label="📥 Download Logs",
            data=log_text,
            file_name=f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.info("No logs available. Run an analysis to generate logs.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>Gtmats</strong><br>
    Group 08 DAMG 7374 Project<br>
    <small>Axelrod Fin Experiment</small>
</div>
""", unsafe_allow_html=True)