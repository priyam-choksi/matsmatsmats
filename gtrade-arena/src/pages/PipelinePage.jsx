import React, { useState, useRef, useEffect } from 'react'
import { useTheme, usePipeline } from '../App'

// ========== DEMO DISPLAY OPTIONS ==========
const DEMO_OPTIONS = {
  BANNER: true,
  INLINE_SIDEBAR: false,
  HEADER_BADGE: false,
  FOOTER: true,
}

// ========== TICKERS (for dropdown) ==========
const TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'LLY']

// ========== PHASES (for UI display only) ==========
const PHASES = [
  { 
    num: 1, 
    name: 'Market Analysis', 
    icon: '📊',
    color: '#6366f1',
    agents: [
      { name: 'Technical Analyst', icon: '📈', desc: 'RSI, MACD, Bollinger Bands, support/resistance' },
      { name: 'News Analyst', icon: '📰', desc: 'Sentiment from Yahoo Finance, Finnhub, Reuters' },
      { name: 'Fundamental Analyst', icon: '📋', desc: 'P/E, growth rates, margins, valuation' },
      { name: 'Macro Analyst', icon: '🌍', desc: 'Fed policy, sector rotation, market regime' },
    ],
    duration: '~60s',
    output: 'market_context.json'
  },
  { 
    num: 2, 
    name: 'Bull/Bear Research', 
    icon: '⚔️',
    color: '#f59e0b',
    agents: [
      { name: 'Bull Researcher', icon: '🐂', desc: 'Builds bullish thesis with catalysts and upside targets' },
      { name: 'Bear Researcher', icon: '🐻', desc: 'Constructs bearish case with risks and downside scenarios' },
    ],
    duration: '~40s',
    output: 'research_theses.json'
  },
  { 
    num: 3, 
    name: 'Debate & Synthesis', 
    icon: '⚖️',
    color: '#10b981',
    agents: [
      { name: 'Research Manager', icon: '👨‍⚖️', desc: 'Moderates bull/bear debate, synthesizes findings' },
    ],
    duration: '~30s',
    output: 'synthesis.json'
  },
  { 
    num: 4, 
    name: 'Risk Evaluation', 
    icon: '🎯',
    color: '#ec4899',
    agents: [
      { name: 'Aggressive Evaluator', icon: '🔥', desc: 'Risk-seeking, higher positions, default: STRONG BUY' },
      { name: 'Neutral Evaluator', icon: '⚖️', desc: 'Balanced assessment, default: HOLD' },
      { name: 'Conservative Evaluator', icon: '🛡️', desc: 'Risk-averse, lower positions, default: AVOID' },
    ],
    duration: '~30s',
    output: 'evaluations.json'
  },
  { 
    num: 5, 
    name: 'Final Decision', 
    icon: '✅',
    color: '#8b5cf6',
    agents: [
      { name: 'Risk Manager', icon: '👔', desc: 'Makes final BUY/HOLD/REJECT with position sizing' },
    ],
    duration: '~10s',
    output: 'risk_decision.json'
  },
]

// ========== ICONS ==========
const Icons = {
  Play: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
  ),
  X: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18"/>
      <path d="m6 6 12 12"/>
    </svg>
  ),
  Check: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  ),
  Github: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
    </svg>
  ),
  Info: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 16v-4"/>
      <path d="M12 8h.01"/>
    </svg>
  ),
}

// ========== DEMO BANNER COMPONENT ==========
const DemoBanner = ({ isDark }) => {
  const [dismissed, setDismissed] = useState(false)
  
  if (dismissed) return null
  
  return (
    <div style={{
      background: isDark ? '#18181b' : '#f4f4f5',
      border: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}`,
      borderRadius: '10px',
      padding: '12px 16px',
      marginBottom: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      position: 'relative'
    }}>
      <div style={{
        width: '28px',
        height: '28px',
        borderRadius: '6px',
        background: isDark ? '#27272a' : '#e4e4e7',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        color: 'var(--text-muted)'
      }}>
        <Icons.Info size={14} />
      </div>
      
      <p style={{ 
        margin: 0, 
        fontSize: '12px', 
        color: 'var(--text-muted)',
        lineHeight: '1.4',
        flex: 1
      }}>
        <span style={{ fontWeight: '600', color: 'var(--text-secondary)' }}>Demo mode</span> — Interactive visualization of the 11-agent pipeline. Production system uses FastAPI with real-time LLM orchestration and live market data.
      </p>
      
      <button
        onClick={() => setDismissed(true)}
        style={{
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--text-muted)',
          padding: '4px',
          borderRadius: '4px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: 0.6
        }}
      >
        <Icons.X size={14} />
      </button>
    </div>
  )
}

// ========== SIDEBAR DEMO NOTE COMPONENT ==========
const SidebarDemoNote = ({ isDark }) => (
  <div style={{
    marginTop: '12px',
    padding: '10px 12px',
    background: isDark ? '#18181b' : '#fafafa',
    borderRadius: '8px',
    border: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}`,
  }}>
    <div style={{ 
      fontSize: '10px', 
      fontWeight: '600', 
      color: 'var(--text-muted)',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '4px'
    }}>
      Demo Mode
    </div>
    <p style={{ 
      margin: 0, 
      fontSize: '11px', 
      color: 'var(--text-muted)',
      lineHeight: '1.4'
    }}>
      Simulated pipeline execution. Production uses FastAPI + live LLM calls.
    </p>
  </div>
)

// ========== DEMO FOOTER COMPONENT ==========
const DemoFooter = ({ isDark }) => (
  <div style={{
    padding: '6px 20px',
    borderTop: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',
    background: 'var(--bg-secondary)',
    flexShrink: 0
  }}>
    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
      Interactive demo • Production system runs 11 LLM agents with live market data
    </span>
    <span style={{ color: 'var(--text-muted)', opacity: 0.4 }}>•</span>
    <a
      href="https://github.com/priyam-03/Capstone"
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        fontSize: '11px',
        color: 'var(--text-muted)',
        textDecoration: 'none',
      }}
    >
      <Icons.Github size={12} />
      View Source
    </a>
  </div>
)

// ========== AGENT CARD COMPONENT ==========
const AgentCard = ({ agent, status, output, isActive, phaseColor, isDark }) => {
  const [showOutput, setShowOutput] = useState(false)

  return (
    <div style={{
      padding: '14px',
      background: isActive 
        ? (isDark ? '#1a1a2e' : '#f0f0ff')
        : status === 'complete' 
          ? (isDark ? '#0f1f0f' : '#f0fdf4')
          : 'var(--bg-tertiary)',
      borderRadius: '10px',
      border: isActive 
        ? `2px solid ${phaseColor}` 
        : status === 'complete' 
          ? '1px solid #22c55e44' 
          : '1px solid var(--border-primary)',
      marginBottom: '10px',
      transition: 'all 0.3s'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '20px' }}>{agent.icon}</span>
          <div>
            <div style={{ fontWeight: '600', fontSize: '13px', color: 'var(--text-primary)' }}>
              {agent.name}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {agent.desc}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isActive && (
            <span style={{
              width: '14px',
              height: '14px',
              border: `2px solid ${phaseColor}`,
              borderTopColor: 'transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
          )}
          {status === 'complete' && (
            <>
              <span style={{ color: '#22c55e', display: 'flex' }}>
                <Icons.Check size={16} />
              </span>
              <button
                onClick={() => setShowOutput(!showOutput)}
                style={{
                  padding: '4px 10px',
                  background: isDark ? '#27272a' : '#e4e4e7',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  color: 'var(--text-secondary)'
                }}
              >
                {showOutput ? 'Hide' : 'View'}
              </button>
            </>
          )}
          {status === 'pending' && (
            <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Pending</span>
          )}
        </div>
      </div>
      
      {showOutput && output && (
        <div style={{
          marginTop: '12px',
          padding: '12px',
          background: isDark ? '#0a0a0a' : '#fafafa',
          borderRadius: '8px',
          fontSize: '11px',
          fontFamily: 'monospace'
        }}>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
            {JSON.stringify(output, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ========== RESULT CARD COMPONENT ==========
const ResultCard = ({ result, ticker, isDark }) => {
  if (!result) return null
  
  const colors = {
    BUY: { bg: isDark ? '#0f1f0f' : '#f0fdf4', border: '#22c55e', text: '#22c55e' },
    HOLD: { bg: isDark ? '#1f1a0f' : '#fffbeb', border: '#f59e0b', text: '#f59e0b' },
    REJECT: { bg: isDark ? '#1f0f0f' : '#fef2f2', border: '#ef4444', text: '#ef4444' },
  }
  const c = colors[result.verdict] || colors.HOLD

  return (
    <div style={{
      background: c.bg,
      border: `1px solid ${c.border}44`,
      borderRadius: '14px',
      padding: '20px',
      marginBottom: '16px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>
            {ticker} Analysis Complete
          </div>
          <div style={{ fontSize: '32px', fontWeight: '800', color: c.text }}>
            {result.verdict}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          {result.verdict !== 'REJECT' && (
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#22c55e' }}>
              {result.position_dollars}
            </div>
          )}
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {result.confidence} Confidence
          </div>
        </div>
      </div>
      
      <div style={{
        marginTop: '16px',
        padding: '14px',
        background: 'var(--bg-primary)',
        borderRadius: '10px'
      }}>
        <div style={{
          fontSize: '10px',
          fontWeight: '600',
          color: 'var(--text-muted)',
          marginBottom: '8px',
          textTransform: 'uppercase'
        }}>
          Reasoning
        </div>
        <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          {result.reasoning}
        </p>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '10px' }}>
        <div style={{ flex: 1, padding: '12px', background: 'var(--bg-primary)', borderRadius: '10px', textAlign: 'center' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Stop Loss</div>
          <div style={{ fontSize: '16px', fontWeight: '700', color: '#ef4444' }}>{result.stop_loss}</div>
        </div>
        <div style={{ flex: 1, padding: '12px', background: 'var(--bg-primary)', borderRadius: '10px', textAlign: 'center' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Target</div>
          <div style={{ fontSize: '16px', fontWeight: '700', color: '#22c55e' }}>{result.target}</div>
        </div>
      </div>
      
      {/* Evaluator Consensus */}
      {result.evaluator_consensus && (
        <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
          {Object.entries(result.evaluator_consensus).map(([type, stance]) => (
            <div key={type} style={{ 
              flex: 1, 
              padding: '10px', 
              background: 'var(--bg-primary)', 
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'capitalize' }}>
                {type}
              </div>
              <div style={{ 
                fontSize: '11px', 
                fontWeight: '700', 
                color: stance.includes('BUY') ? '#22c55e' : stance === 'HOLD' ? '#f59e0b' : '#ef4444'
              }}>
                {stance}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ========== MAIN PIPELINE PAGE COMPONENT ==========
export default function PipelinePage() {
  const { isDark } = useTheme()
  
  // Get all state and actions from PipelineContext (defined in App.jsx)
  const {
    ticker,
    setTicker,
    researchMode,
    setResearchMode,
    isRunning,
    currentPhase,
    currentAgent,
    progress,
    logs,
    agentStatuses,
    agentOutputs,
    result,
    selectedPhase,
    setSelectedPhase,
    runPipeline,
    resetPipeline,
  } = usePipeline()
  
  const logsEndRef = useRef(null)

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div style={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column', 
      background: 'var(--bg-secondary)' 
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* ==================== HEADER ==================== */}
      <header style={{
        background: 'var(--bg-primary)',
        borderBottom: '1px solid var(--border-primary)',
        padding: '12px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
            Analysis Pipeline
          </h1>
          <p style={{ margin: '2px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>
            5-Phase LLM Workflow • 11 Agents
          </p>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px' }}>
          {/* Header Badge (optional) */}
          {DEMO_OPTIONS.HEADER_BADGE && (
            <span style={{ 
              padding: '5px 10px', 
              background: isDark ? '#1e1b4b' : '#eef2ff', 
              borderRadius: '6px', 
              color: isDark ? '#a5b4fc' : '#4f46e5',
              fontSize: '11px',
              fontWeight: '500'
            }}>
              Demo Mode
            </span>
          )}
          
          <span style={{ 
            padding: '5px 10px', 
            background: 'var(--bg-tertiary)', 
            borderRadius: '6px', 
            color: 'var(--text-secondary)' 
          }}>
            {PHASES.reduce((sum, p) => sum + p.agents.length, 0)} Agents
          </span>
          
          {/* Processing indicator */}
          {isRunning && (
            <span style={{
              padding: '5px 10px',
              background: isDark ? '#1a1a2e' : '#f0f0ff',
              borderRadius: '6px',
              color: '#6366f1',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span style={{
                width: '8px',
                height: '8px',
                background: '#6366f1',
                borderRadius: '50%',
                animation: 'pulse 1s infinite'
              }} />
              Processing
            </span>
          )}
          
          {/* Complete indicator */}
          {!isRunning && progress === 100 && (
            <span style={{
              padding: '5px 10px',
              background: isDark ? '#0f1f0f' : '#f0fdf4',
              borderRadius: '6px',
              color: '#22c55e',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <Icons.Check size={12} />
              Complete
            </span>
          )}
        </div>
      </header>

      {/* ==================== MAIN LAYOUT ==================== */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        
        {/* ==================== LEFT SIDEBAR ==================== */}
        <aside style={{
          width: '300px',
          background: 'var(--bg-card)',
          borderRight: '1px solid var(--border-primary)',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* Configuration Section */}
          <div style={{ padding: '16px', borderBottom: '1px solid var(--border-primary)' }}>
            <div style={{ 
              fontSize: '11px', 
              fontWeight: '600', 
              color: 'var(--text-muted)', 
              marginBottom: '12px', 
              textTransform: 'uppercase' 
            }}>
              Configuration
            </div>
            
            {/* Ticker Select */}
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                Ticker
              </label>
              <select
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                disabled={isRunning}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  border: '1px solid var(--border-primary)',
                  borderRadius: '8px',
                  fontSize: '14px',
                  fontWeight: '600',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  cursor: isRunning ? 'not-allowed' : 'pointer'
                }}
              >
                {TICKERS.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>

            {/* Research Mode */}
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                Research Mode
              </label>
              <div style={{ display: 'flex', gap: '6px' }}>
                {[
                  { id: 'shallow', label: 'Fast', time: '~2 min' },
                  { id: 'deep', label: 'Deep', time: '~5 min' },
                  { id: 'research', label: 'Full', time: '~8 min' }
                ].map(mode => (
                  <button
                    key={mode.id}
                    onClick={() => setResearchMode(mode.id)}
                    disabled={isRunning}
                    style={{
                      flex: 1,
                      padding: '10px 8px',
                      border: researchMode === mode.id 
                        ? '2px solid var(--text-primary)' 
                        : '1px solid var(--border-primary)',
                      borderRadius: '8px',
                      background: researchMode === mode.id 
                        ? (isDark ? '#27272a' : '#f4f4f5') 
                        : 'var(--bg-secondary)',
                      cursor: isRunning ? 'not-allowed' : 'pointer',
                      fontSize: '11px'
                    }}
                  >
                    <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{mode.label}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>{mode.time}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Run / Cancel Button */}
            {!isRunning ? (
              <button
                onClick={runPipeline}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: 'var(--text-primary)',
                  color: 'var(--bg-primary)',
                  border: 'none',
                  borderRadius: '10px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                <Icons.Play size={14} /> 
                {progress === 100 ? 'Run Again' : 'Run Pipeline'}
              </button>
            ) : (
              <button
                onClick={resetPipeline}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: isDark ? '#2a1a1a' : '#fef2f2',
                  color: '#ef4444',
                  border: '1px solid #ef444444',
                  borderRadius: '10px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                <Icons.X size={14} /> Cancel
              </button>
            )}
            
            {/* Clear Results Button */}
            {!isRunning && progress === 100 && (
              <button
                onClick={resetPipeline}
                style={{
                  width: '100%',
                  marginTop: '8px',
                  padding: '10px',
                  background: 'transparent',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border-primary)',
                  borderRadius: '10px',
                  fontSize: '12px',
                  fontWeight: '500',
                  cursor: 'pointer',
                }}
              >
                Clear Results
              </button>
            )}
            
            {/* Sidebar Demo Note */}
            {DEMO_OPTIONS.INLINE_SIDEBAR && <SidebarDemoNote isDark={isDark} />}
          </div>

          {/* Progress Section */}
          {(isRunning || progress > 0) && (
            <div style={{ padding: '16px', borderBottom: '1px solid var(--border-primary)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
                <span style={{ fontWeight: '600', color: 'var(--text-primary)' }}>Progress</span>
                <span style={{ color: 'var(--text-muted)' }}>{Math.round(progress)}%</span>
              </div>
              <div style={{ 
                height: '6px', 
                background: 'var(--bg-tertiary)', 
                borderRadius: '3px', 
                overflow: 'hidden' 
              }}>
                <div style={{
                  height: '100%',
                  width: `${progress}%`,
                  background: progress === 100 
                    ? '#22c55e' 
                    : (PHASES[Math.max(0, currentPhase - 1)]?.color || 'var(--text-primary)'),
                  transition: 'width 0.3s',
                  borderRadius: '3px'
                }} />
              </div>
              {currentAgent && (
                <div style={{
                  marginTop: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '12px',
                  color: PHASES[Math.max(0, currentPhase - 1)]?.color || 'var(--text-primary)'
                }}>
                  <span style={{
                    width: '6px',
                    height: '6px',
                    background: 'currentColor',
                    borderRadius: '50%',
                    animation: 'pulse 1s infinite'
                  }} />
                  {currentAgent}
                </div>
              )}
            </div>
          )}

          {/* Phase Navigation */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }} className="hide-scrollbar">
            <div style={{ 
              fontSize: '11px', 
              fontWeight: '600', 
              color: 'var(--text-muted)', 
              marginBottom: '10px', 
              textTransform: 'uppercase' 
            }}>
              Phases
            </div>
            
            {PHASES.map((phase) => {
              const phaseComplete = currentPhase > phase.num || (!isRunning && progress === 100)
              const phaseActive = currentPhase === phase.num && isRunning
              
              return (
                <div
                  key={phase.num}
                  onClick={() => setSelectedPhase(phase.num)}
                  style={{
                    padding: '12px',
                    marginBottom: '8px',
                    borderRadius: '10px',
                    cursor: 'pointer',
                    border: selectedPhase === phase.num 
                      ? `2px solid ${phase.color}` 
                      : '1px solid var(--border-primary)',
                    background: phaseActive 
                      ? (isDark ? '#1a1a2e' : '#f5f5ff')
                      : phaseComplete 
                        ? (isDark ? '#0f1f0f' : '#f0fdf4')
                        : 'var(--bg-secondary)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '18px' }}>{phase.icon}</span>
                      <div>
                        <div style={{ fontWeight: '600', fontSize: '12px', color: 'var(--text-primary)' }}>
                          Phase {phase.num}: {phase.name}
                        </div>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                          {phase.agents.length} agents • {phase.duration}
                        </div>
                      </div>
                    </div>
                    
                    {phaseComplete && (
                      <span style={{ color: '#22c55e', display: 'flex' }}>
                        <Icons.Check size={14} />
                      </span>
                    )}
                    {phaseActive && (
                      <span style={{
                        width: '12px',
                        height: '12px',
                        border: `2px solid ${phase.color}`,
                        borderTopColor: 'transparent',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite'
                      }} />
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </aside>

        {/* ==================== MAIN CONTENT ==================== */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <main style={{ flex: 1, padding: '20px', overflowY: 'auto', minHeight: 0 }}>
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              
              {/* Demo Banner */}
              {DEMO_OPTIONS.BANNER && <DemoBanner isDark={isDark} />}
              
              {/* Result Card */}
              <ResultCard result={result} ticker={ticker} isDark={isDark} />

              {/* Phase Detail */}
              {selectedPhase && (
                <div style={{
                  background: 'var(--bg-card)',
                  borderRadius: '14px',
                  border: '1px solid var(--border-primary)',
                  padding: '20px',
                  marginBottom: '16px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                    <div style={{
                      width: '44px',
                      height: '44px',
                      borderRadius: '12px',
                      background: PHASES[selectedPhase - 1].color + '22',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '22px'
                    }}>
                      {PHASES[selectedPhase - 1].icon}
                    </div>
                    <div>
                      <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600', color: 'var(--text-primary)' }}>
                        Phase {selectedPhase}: {PHASES[selectedPhase - 1].name}
                      </h3>
                      <p style={{ margin: '2px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>
                        {PHASES[selectedPhase - 1].agents.length} agents • Output: {PHASES[selectedPhase - 1].output}
                      </p>
                    </div>
                  </div>
                  
                  {PHASES[selectedPhase - 1].agents.map(agent => (
                    <AgentCard
                      key={agent.name}
                      agent={agent}
                      status={agentStatuses[agent.name] || 'pending'}
                      output={agentOutputs[agent.name]}
                      isActive={currentAgent === agent.name}
                      phaseColor={PHASES[selectedPhase - 1].color}
                      isDark={isDark}
                    />
                  ))}
                </div>
              )}

              {/* Execution Log */}
              <div style={{
                background: isDark ? '#0a0a0a' : '#1e1e1e',
                borderRadius: '14px',
                padding: '16px',
                minHeight: '250px'
              }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center', 
                  marginBottom: '12px' 
                }}>
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#e5e7eb' }}>
                    Execution Log
                  </h3>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', color: '#6b7280' }}>{logs.length} entries</span>
                  </div>
                </div>
                
                <div 
                  style={{ 
                    maxHeight: '300px', 
                    overflowY: 'auto', 
                    fontFamily: 'monospace', 
                    fontSize: '12px' 
                  }} 
                  className="hide-scrollbar"
                >
                  {logs.length === 0 ? (
                    <div style={{ color: '#6b7280', padding: '20px', textAlign: 'center' }}>
                      Click "Run Pipeline" to start analysis...
                    </div>
                  ) : (
                    logs.map((log, i) => {
                      const colors = {
                        phase: '#6366f1',
                        success: '#22c55e',
                        result: '#f59e0b',
                        info: '#6b7280',
                        agent_start: '#9ca3af'
                      }
                      return (
                        <div
                          key={i}
                          style={{
                            padding: '4px 0',
                            color: colors[log.type] || '#6b7280',
                            fontWeight: log.type === 'phase' || log.type === 'result' ? '600' : '400',
                            borderLeft: log.type === 'phase' 
                              ? '3px solid #6366f1' 
                              : log.type === 'result' 
                                ? '3px solid #f59e0b' 
                                : 'none',
                            paddingLeft: log.type === 'phase' || log.type === 'result' ? '10px' : '0',
                            marginLeft: log.type === 'phase' || log.type === 'result' ? '0' : '13px'
                          }}
                        >
                          <span style={{ color: '#4b5563', marginRight: '10px' }}>[{log.time}]</span>
                          {log.message}
                        </div>
                      )
                    })
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
            </div>
          </main>
          
          {/* Demo Footer */}
          {DEMO_OPTIONS.FOOTER && <DemoFooter isDark={isDark} />}
        </div>
      </div>
    </div>
  )
}