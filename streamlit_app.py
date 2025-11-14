# streamlit_app.py
"""
🎯 Game Theory Trading System - Advanced Dashboard
Real-time pipeline execution with live progress tracking
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import time
import threading
import queue

# Page config
st.set_page_config(
    page_title="Game Theory Trading System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-left: 5px solid #0d7377;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        color: white;
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left: 5px solid #c9184a;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        color: white;
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
    }
    
    .phase-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .phase-card.running {
        border-left-color: #ffc107;
        animation: pulse 2s infinite;
    }
    
    .phase-card.success {
        border-left-color: #28a745;
    }
    
    .phase-card.error {
        border-left-color: #dc3545;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .strategy-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .strategy-badge.winner {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = None
if 'phase_status' not in st.session_state:
    st.session_state.phase_status = {}
if 'log_output' not in st.session_state:
    st.session_state.log_output = []

# Helper functions
def load_json(filename):
    """Load JSON file from outputs"""
    try:
        filepath = Path("outputs") / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return None

def run_pipeline_with_progress(ticker, mode, portfolio):
    """Run pipeline and capture output in real-time"""
    cmd = [
        sys.executable,
        "agents/orchestrators/master_orchestrator.py",
        ticker,
        "--run-all",
        "--research-mode", mode,
        "--portfolio-value", str(portfolio)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    
    for line in process.stdout:
        st.session_state.log_output.append(line.strip())
        
        # Parse phase info
        if "Phase 1:" in line:
            st.session_state.current_phase = "Phase 1: Analysts"
        elif "Phase 2:" in line:
            st.session_state.current_phase = "Phase 2: Researchers"
        elif "Phase 3:" in line:
            st.session_state.current_phase = "Phase 3: Research Manager"
        elif "Phase 4:" in line:
            st.session_state.current_phase = "Phase 4: Risk Team"
        elif "Phase 5:" in line:
            st.session_state.current_phase = "Phase 5: Risk Manager"
        elif "Phase 6:" in line:
            st.session_state.current_phase = "Phase 6: Game Theory"
        
        # Track status
        if "[OK]" in line or "SUCCESS" in line:
            if st.session_state.current_phase:
                st.session_state.phase_status[st.session_state.current_phase] = "success"
        elif "[ERROR]" in line or "failed" in line:
            if st.session_state.current_phase:
                st.session_state.phase_status[st.session_state.current_phase] = "error"
    
    process.wait()
    return process.returncode == 0

def create_radar_chart(recommendations):
    """Create radar chart for analyst consensus"""
    categories = ['Technical', 'Fundamental', 'News', 'Macro']
    
    # Convert to scores: BUY=1, HOLD=0.5, SELL=0
    scores = []
    for cat in ['technical', 'fundamental', 'news', 'macro']:
        rec = recommendations.get(cat, 'HOLD')
        score = 1.0 if rec == 'BUY' else 0.0 if rec == 'SELL' else 0.5
        scores.append(score)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Analyst Sentiment',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=400,
        title="Analyst Sentiment Radar"
    )
    
    return fig

def create_strategy_sunburst(strategies):
    """Create sunburst chart for strategy breakdown"""
    data = {
        'labels': ['All Strategies'],
        'parents': [''],
        'values': [1]
    }
    
    for name, strat in strategies.items():
        data['labels'].append(name.upper())
        data['parents'].append('All Strategies')
        data['values'].append(abs(strat.get('position_size', 1000)))
    
    fig = go.Figure(go.Sunburst(
        labels=data['labels'],
        parents=data['parents'],
        values=data['values'],
        branchvalues="total",
        marker=dict(
            colors=px.colors.sequential.RdBu,
            line=dict(color='white', width=2)
        )
    ))
    
    fig.update_layout(height=500, title="Strategy Position Distribution")
    return fig

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "",
        ["🏠 Dashboard", "🚀 Live Analysis", "📊 Detailed Results", "🎭 Tournament", "📈 Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System stats
    st.markdown("### 📊 System Stats")
    
    risk = load_json("risk_decision.json")
    game = load_json("game_theory_tournament.json")
    
    if risk:
        st.success("✅ Latest Analysis Ready")
        verdict = risk.get('verdict', 'N/A')
        st.metric("Last Decision", verdict, delta=None)
    else:
        st.info("💤 No Recent Analysis")
    
    st.markdown("---")
    
    st.markdown("### 🧠 Components")
    components = {
        "Analysts": "4",
        "Researchers": "2", 
        "Risk Debators": "3",
        "Strategies": "5"
    }
    
    for comp, count in components.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(comp)
        with col2:
            st.markdown(f"**{count}**")

# Main Content
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🎯 Game Theory Trading System</div>', unsafe_allow_html=True)
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    risk_decision = load_json("risk_decision.json")
    game_theory = load_json("game_theory_tournament.json")
    discussion = load_json("discussion_points.json")
    
    with col1:
        if risk_decision:
            verdict = risk_decision.get('verdict', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='margin:0;'>Risk Decision</h4>
                <h2 style='margin:10px 0;'>{verdict}</h2>
                <p style='margin:0; opacity:0.9;'>✅ Available</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #868f96 0%, #596164 100%);">
                <h4 style='margin:0;'>Risk Decision</h4>
                <h2 style='margin:10px 0;'>N/A</h2>
                <p style='margin:0; opacity:0.9;'>❌ Not Run</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if game_theory:
            winner = game_theory.get('recommended_strategy', 'N/A').upper()
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='margin:0;'>Best Strategy</h4>
                <h2 style='margin:10px 0;'>{winner}</h2>
                <p style='margin:0; opacity:0.9;'>🏆 Winner</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #868f96 0%, #596164 100%);">
                <h4 style='margin:0;'>Best Strategy</h4>
                <h2 style='margin:10px 0;'>N/A</h2>
                <p style='margin:0; opacity:0.9;'>⏳ Pending</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if discussion:
            summary = discussion.get('summary', {})
            recs = summary.get('recommendations', {})
            buy_count = list(recs.values()).count('BUY')
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='margin:0;'>Analyst Consensus</h4>
                <h2 style='margin:10px 0;'>{buy_count}/4</h2>
                <p style='margin:0; opacity:0.9;'>🟢 Bullish Votes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #868f96 0%, #596164 100%);">
                <h4 style='margin:0;'>Analyst Consensus</h4>
                <h2 style='margin:10px 0;'>0/4</h2>
                <p style='margin:0; opacity:0.9;'>⏳ Pending</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if risk_decision:
            position = risk_decision.get('final_position_dollars', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='margin:0;'>Position Size</h4>
                <h2 style='margin:10px 0;'>${position/1000:.1f}K</h2>
                <p style='margin:0; opacity:0.9;'>💰 Allocated</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #868f96 0%, #596164 100%);">
                <h4 style='margin:0;'>Position Size</h4>
                <h2 style='margin:10px 0;'>$0</h2>
                <p style='margin:0; opacity:0.9;'>⏳ Pending</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🎯 System Overview")
        
        # Pipeline flowchart
        st.markdown("""
        ```
        📊 Phase 1: Analyst Team (4 agents)
              ↓
        🎓 Phase 2: Bull vs Bear Researchers  
              ↓
        🎯 Phase 3: Research Manager (Synthesis)
              ↓
        ⚖️  Phase 4: Risk Debators (3 perspectives)
              ↓
        🛡️  Phase 5: Risk Manager (Final Decision)
              ↓
        🎭 Phase 6: Game Theory Tournament (5 strategies)
        ```
        """)
        
        st.info("**Core Question:** Which trading strategy performs best under different market regimes?")
    
    with col2:
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("▶️ Run New Analysis", type="primary", use_container_width=True):
            st.session_state.page = "🚀 Live Analysis"
            st.rerun()
        
        if risk_decision:
            if st.button("📊 View Results", use_container_width=True):
                st.session_state.page = "📊 Detailed Results"
                st.rerun()
        
        if game_theory:
            if st.button("🎭 View Tournament", use_container_width=True):
                st.session_state.page = "🎭 Tournament"
                st.rerun()
    
    # Recent activity
    if risk_decision or game_theory:
        st.markdown("---")
        st.markdown("### 📈 Latest Analysis Summary")
        
        if risk_decision and discussion:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔍 Analysts")
                summary = discussion.get('summary', {})
                recs = summary.get('recommendations', {})
                for analyst, rec in recs.items():
                    emoji = "🟢" if rec == "BUY" else "🔴" if rec == "SELL" else "🟡"
                    st.markdown(f"{emoji} **{analyst.title()}**: {rec}")
            
            with col2:
                st.markdown("#### 🎯 Final Decision")
                verdict = risk_decision.get('verdict', 'N/A')
                position = risk_decision.get('final_position_dollars', 0)
                confidence = risk_decision.get('confidence', 'N/A')
                
                st.markdown(f"**Verdict:** {verdict}")
                st.markdown(f"**Position:** ${position:,.0f}")
                st.markdown(f"**Confidence:** {confidence}")
            
            with col3:
                st.markdown("#### 🏆 Best Strategy")
                if game_theory:
                    winner = game_theory.get('recommended_strategy', 'N/A')
                    regime = game_theory.get('regime', 'N/A')
                    
                    st.markdown(f"**Strategy:** {winner.upper()}")
                    st.markdown(f"**Regime:** {regime}")

elif page == "🚀 Live Analysis":
    st.markdown('<div class="main-header">🚀 Live Analysis Execution</div>', unsafe_allow_html=True)
    
    # Input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input(
            "📈 Stock Ticker",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, TSLA, NVDA",
            disabled=st.session_state.running
        ).upper()
    
    with col2:
        mode = st.selectbox(
            "🔬 Research Mode",
            ["shallow", "deep", "research"],
            disabled=st.session_state.running,
            help="shallow: 1 round (30s), deep: 3 rounds (2min), research: 5+ rounds (5min)"
        )
    
    with col3:
        portfolio = st.number_input(
            "💰 Portfolio ($)",
            value=100000,
            step=10000,
            disabled=st.session_state.running
        )
    
    # Run button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if not st.session_state.running:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                if ticker:
                    st.session_state.running = True
                    st.session_state.log_output = []
                    st.session_state.phase_status = {}
                    st.rerun()
        else:
            st.button("⏳ Running...", disabled=True, use_container_width=True)
    
    # Progress section
    if st.session_state.running:
        st.markdown("---")
        st.markdown("### 📊 Live Progress")
        
        # Phase progress cards
        phases = [
            ("Phase 1: Analysts", "🔍"),
            ("Phase 2: Researchers", "🎓"),
            ("Phase 3: Research Manager", "🎯"),
            ("Phase 4: Risk Team", "⚖️"),
            ("Phase 5: Risk Manager", "🛡️"),
            ("Phase 6: Game Theory", "🎭")
        ]
        
        for phase_name, emoji in phases:
            status = st.session_state.phase_status.get(phase_name, "pending")
            
            if status == "success":
                st.markdown(f"""
                <div class="phase-card success">
                    {emoji} <strong>{phase_name}</strong> ✅ <span style="color: #28a745;">Complete</span>
                </div>
                """, unsafe_allow_html=True)
            elif status == "error":
                st.markdown(f"""
                <div class="phase-card error">
                    {emoji} <strong>{phase_name}</strong> ❌ <span style="color: #dc3545;">Failed</span>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.current_phase == phase_name:
                st.markdown(f"""
                <div class="phase-card running">
                    {emoji} <strong>{phase_name}</strong> ⏳ <span style="color: #ffc107;">Running...</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="phase-card">
                    {emoji} <strong>{phase_name}</strong> ⏸️ <span style="color: #6c757d;">Pending</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Log output
        with st.expander("📋 Execution Log", expanded=True):
            log_container = st.container()
            with log_container:
                if st.session_state.log_output:
                    st.code("\n".join(st.session_state.log_output[-20:]), language="text")
        
        # Run the pipeline
        with st.spinner("🔄 Executing pipeline..."):
            success = run_pipeline_with_progress(ticker, mode, portfolio)
            
            st.session_state.running = False
            
            if success:
                st.success("✅ Analysis Complete!")
                st.balloons()
                time.sleep(2)
                st.session_state.page = "📊 Detailed Results"
                st.rerun()
            else:
                st.error("❌ Analysis failed. Check the log for details.")

elif page == "📊 Detailed Results":
    st.markdown('<div class="main-header">📊 Detailed Analysis Results</div>', unsafe_allow_html=True)
    
    discussion = load_json("discussion_points.json")
    bull_thesis = load_json("bull_thesis.json")
    bear_thesis = load_json("bear_thesis.json")
    synthesis = load_json("research_synthesis.json")
    risk_decision = load_json("risk_decision.json")
    
    if not discussion:
        st.warning("⚠️ No analysis data available. Please run an analysis first.")
        if st.button("🚀 Run Analysis"):
            st.session_state.page = "🚀 Live Analysis"
            st.rerun()
    else:
        # Tabs for organization
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysts", "🎓 Research", "⚖️ Risk", "📈 Charts"])
        
        with tab1:
            st.markdown("### 🔍 Analyst Team Recommendations")
            
            summary = discussion.get('summary', {})
            recs = summary.get('recommendations', {})
            
            # Analyst cards
            col1, col2 = st.columns(2)
            
            analysts = [
                ("technical", "📈 Technical Analysis", col1),
                ("fundamental", "💰 Fundamental Analysis", col2),
                ("news", "📰 News Sentiment", col1),
                ("macro", "🌍 Macro Economics", col2)
            ]
            
            for key, title, col in analysts:
                rec = recs.get(key, 'N/A')
                emoji = "🟢" if rec == "BUY" else "🔴" if rec == "SELL" else "🟡"
                
                with col:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <h4>{title}</h4>
                        <h2>{emoji} {rec}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Radar chart
            st.plotly_chart(create_radar_chart(recs), use_container_width=True)
        
        with tab2:
            st.markdown("### 🎓 Bull vs Bear Research")
            
            if bull_thesis and bear_thesis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🐂 Bull Case")
                    bull_prob = bull_thesis.get('probability_assessment', {}).get('bull_case_probability', 50)
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=bull_prob,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Bull Probability"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "green"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgray"},
                                {'range': [33, 66], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🐻 Bear Case")
                    bear_prob = bear_thesis.get('probability_assessment', {}).get('bear_case_probability', 50)
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=bear_prob,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Bear Probability"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "red"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgray"},
                                {'range': [33, 66], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "green", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### ⚖️ Risk Management Decision")
            
            if risk_decision:
                # Final decision hero
                verdict = risk_decision.get('verdict', 'N/A')
                position = risk_decision.get('final_position_dollars', 0)
                confidence = risk_decision.get('confidence', 'N/A')
                
                if verdict == "APPROVE":
                    st.markdown(f"""
                    <div class="success-box">
                        <h2>✅ {verdict}</h2>
                        <p style="font-size: 1.2rem; margin: 1rem 0;">
                            Position: <strong>${position:,.0f}</strong> | 
                            Confidence: <strong>{confidence}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif verdict == "MODIFY":
                    st.markdown(f"""
                    <div class="warning-box">
                        <h2>⚠️ {verdict}</h2>
                        <p style="font-size: 1.2rem; margin: 1rem 0;">
                            Position: <strong>${position:,.0f}</strong> | 
                            Confidence: <strong>{confidence}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"🛑 {verdict} - Position: ${position:,.0f}")
                
                # Risk debator perspectives
                st.markdown("#### 🗣️ Risk Debator Perspectives")
                
                col1, col2, col3 = st.columns(3)
                
                debators = [
                    ("aggressive_eval.json", "🔴 Aggressive", col1),
                    ("neutral_eval.json", "🟡 Neutral", col2),
                    ("conservative_eval.json", "🟢 Conservative", col3)
                ]
                
                for filename, title, col in debators:
                    eval_data = load_json(filename)
                    if eval_data:
                        with col:
                            stance = eval_data.get('stance', 'N/A')
                            position = eval_data.get('recommended_position_size', 0)
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px;">
                                <h5>{title}</h5>
                                <p><strong>Stance:</strong> {stance}</p>
                                <p><strong>Position:</strong> ${position:,.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 📈 Visual Analytics")
            
            # Create combined view
            if discussion and risk_decision:
                # Vote distribution
                summary = discussion.get('summary', {})
                recs = summary.get('recommendations', {})
                
                vote_data = pd.DataFrame({
                    'Analyst': list(recs.keys()),
                    'Recommendation': list(recs.values())
                })
                
                fig = px.histogram(
                    vote_data,
                    x='Recommendation',
                    color='Recommendation',
                    title="Analyst Vote Distribution",
                    color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "🎭 Tournament":
    st.markdown('<div class="main-header">🎭 Game Theory Tournament</div>', unsafe_allow_html=True)
    
    game_theory = load_json("game_theory_tournament.json")
    
    if not game_theory:
        st.warning("⚠️ No tournament data. Run the full pipeline first!")
        if st.button("🚀 Run Analysis"):
            st.session_state.page = "🚀 Live Analysis"
            st.rerun()
    else:
        winner = game_theory.get('recommended_strategy', 'N/A')
        regime = game_theory.get('regime', 'N/A')
        
        # Winner announcement
        st.markdown(f"""
        <div class="success-box">
            <h1 style="margin: 0;">🏆 Winner: {winner.upper()}</h1>
            <h3 style="margin-top: 1rem; opacity: 0.9;">Market Regime: {regime}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        strategies = game_theory.get('strategies', {})
        
        # Strategy badges
        st.markdown("### 🎯 All Strategies")
        badge_html = ""
        for name in strategies.keys():
            if name == winner:
                badge_html += f'<span class="strategy-badge winner">🏆 {name.upper()}</span>'
            else:
                badge_html += f'<span class="strategy-badge">{name.upper()}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Position size comparison
            data = []
            for name, strat in strategies.items():
                data.append({
                    'Strategy': name.upper(),
                    'Position': strat.get('position_size', 0),
                    'Action': strat.get('action', 'HOLD')
                })
            
            df = pd.DataFrame(data)
            
            fig = px.bar(
                df,
                x='Strategy',
                y='Position',
                color='Action',
                title="Position Size by Strategy",
                color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence comparison
            data = []
            for name, strat in strategies.items():
                data.append({
                    'Strategy': name.upper(),
                    'Confidence': strat.get('confidence', 0) * 100
                })
            
            df = pd.DataFrame(data)
            
            fig = px.bar(
                df,
                x='Strategy',
                y='Confidence',
                color='Confidence',
                title="Confidence Levels (%)",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sunburst chart
        st.plotly_chart(create_strategy_sunburst(strategies), use_container_width=True)
        
        # Detailed breakdown
        st.markdown("### 📋 Strategy Details")
        
        for name, strat in strategies.items():
            is_winner = (name == winner)
            
            with st.expander(f"{'🏆' if is_winner else '📊'} {name.upper()}", expanded=is_winner):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    action = strat.get('action', 'N/A')
                    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
                    st.metric("Action", f"{emoji} {action}")
                
                with col2:
                    st.metric("Position", f"${strat.get('position_size', 0):,.0f}")
                
                with col3:
                    st.metric("Confidence", f"{strat.get('confidence', 0)*100:.1f}%")
                
                with col4:
                    if is_winner:
                        st.metric("Status", "🏆 WINNER")
                
                st.markdown("**Reasoning:**")
                st.info(strat.get('reasoning', 'No reasoning provided'))

elif page == "📈 Insights":
    st.markdown('<div class="main-header">📈 System Insights</div>', unsafe_allow_html=True)
    
    st.info("🚧 Advanced analytics coming soon! This section will include:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Future Features:
        
        **Backtesting Results:**
        - Historical strategy performance
        - Win rate by regime
        - Risk-adjusted returns
        - Maximum drawdown analysis
        - Sharpe ratio comparison
        
        **Performance Tracking:**
        - Strategy evolution over time
        - Confidence calibration
        - Edge degradation analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Regime Analysis:
        
        **Market Regime Detection:**
        - Bull trend characteristics
        - Bear trend patterns
        - Sideways market behavior
        - High volatility periods
        
        **Strategy Selection:**
        - Best strategy per regime
        - Regime transition handling
        - Meta-learning insights
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 0.9rem;'>🎯 <strong>Game Theory Trading System</strong> | 
    Built with Streamlit & Plotly | © 2025</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
        Multi-Agent AI • Game Theory • Risk Management • Real-Time Analysis
    </p>
</div>
""", unsafe_allow_html=True)