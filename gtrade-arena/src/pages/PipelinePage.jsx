import React, { useState, useRef, useEffect } from 'react'
import { useTheme } from '../App'

// ========== CONSTANTS ==========
const TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'LLY']

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

// ========== GENERATE AGENT OUTPUTS ==========
const generateAgentOutput = (agentName, ticker) => {
  const outputs = {
    'Technical Analyst': {
      signal: ['BULLISH', 'BEARISH', 'NEUTRAL'][Math.floor(Math.random() * 3)],
      rsi: (30 + Math.random() * 40).toFixed(1),
      macd: Math.random() > 0.5 ? 'Bullish Crossover' : 'Bearish Divergence',
      trend: Math.random() > 0.4 ? 'Uptrend' : 'Consolidation',
      support: '$' + (100 + Math.random() * 50).toFixed(2),
      resistance: '$' + (150 + Math.random() * 50).toFixed(2),
    },
    'News Analyst': {
      sentiment: ['Positive', 'Negative', 'Mixed'][Math.floor(Math.random() * 3)],
      articles_analyzed: Math.floor(10 + Math.random() * 20),
      key_catalyst: `${ticker} earnings beat expectations`,
      sentiment_score: (Math.random() * 2 - 1).toFixed(2),
    },
    'Fundamental Analyst': {
      pe_ratio: (15 + Math.random() * 30).toFixed(1),
      revenue_growth: (5 + Math.random() * 25).toFixed(1) + '%',
      profit_margin: (10 + Math.random() * 20).toFixed(1) + '%',
      valuation: ['Undervalued', 'Fair Value', 'Overvalued'][Math.floor(Math.random() * 3)],
    },
    'Macro Analyst': {
      regime: ['Bull Market', 'Bear Market', 'Sideways'][Math.floor(Math.random() * 3)],
      sector_outlook: 'Favorable',
      fed_impact: 'Neutral',
      recommendation: ['Overweight', 'Neutral', 'Underweight'][Math.floor(Math.random() * 3)],
    },
    'Bull Researcher': {
      thesis: `Strong growth catalysts for ${ticker} including market expansion`,
      upside_target: '+' + (15 + Math.random() * 25).toFixed(0) + '%',
      catalysts: ['Earnings beat', 'New product launch', 'Market share gains'],
      confidence: ['HIGH', 'MEDIUM'][Math.floor(Math.random() * 2)],
    },
    'Bear Researcher': {
      thesis: `Key risks include valuation concerns and competitive pressures`,
      downside_risk: '-' + (10 + Math.random() * 20).toFixed(0) + '%',
      risks: ['Valuation stretched', 'Margin compression', 'Competition'],
      confidence: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)],
    },
    'Research Manager': {
      winner: Math.random() > 0.4 ? 'BULL' : 'BEAR',
      probability_bull: (40 + Math.random() * 30).toFixed(0) + '%',
      probability_bear: (20 + Math.random() * 25).toFixed(0) + '%',
      synthesis: `After debate, ${Math.random() > 0.4 ? 'bullish' : 'bearish'} thesis prevails`,
    },
    'Aggressive Evaluator': {
      stance: ['STRONG BUY', 'BUY', 'HOLD'][Math.floor(Math.random() * 3)],
      position_pct: (50 + Math.random() * 40).toFixed(0) + '%',
      reasoning: 'High conviction on growth catalysts',
    },
    'Neutral Evaluator': {
      stance: ['BUY', 'HOLD', 'AVOID'][Math.floor(Math.random() * 3)],
      position_pct: (25 + Math.random() * 30).toFixed(0) + '%',
      reasoning: 'Balanced view considering risks and rewards',
    },
    'Conservative Evaluator': {
      stance: ['HOLD', 'AVOID', 'STRONG AVOID'][Math.floor(Math.random() * 3)],
      position_pct: (5 + Math.random() * 20).toFixed(0) + '%',
      reasoning: 'Prefer capital preservation',
    },
    'Risk Manager': {
      verdict: ['BUY', 'HOLD', 'REJECT'][Math.floor(Math.random() * 3)],
      position_dollars: '$' + (Math.floor(Math.random() * 15000) + 5000).toLocaleString(),
      confidence: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)],
      stop_loss: '-8%',
      target: '+15%',
      reasoning: 'Final decision based on evaluator consensus and risk parameters',
    },
  }
  return outputs[agentName] || { status: 'Complete' }
}

// ========== ICONS ==========
const Icons = {
  Play: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
  ),
  X: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18"/><path d="m6 6 12 12"/>
    </svg>
  ),
  Check: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  ),
}

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
          <pre style={{ 
            margin: 0, 
            whiteSpace: 'pre-wrap', 
            color: 'var(--text-secondary)' 
          }}>
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
          <div style={{
            fontSize: '32px',
            fontWeight: '800',
            color: c.text,
          }}>
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
        <p style={{
          margin: 0,
          fontSize: '13px',
          color: 'var(--text-secondary)',
          lineHeight: '1.5'
        }}>
          {result.reasoning}
        </p>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '10px' }}>
        <div style={{ 
          flex: 1,
          padding: '12px', 
          background: 'var(--bg-primary)', 
          borderRadius: '10px', 
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Stop Loss</div>
          <div style={{ fontSize: '16px', fontWeight: '700', color: '#ef4444' }}>{result.stop_loss}</div>
        </div>
        <div style={{ 
          flex: 1,
          padding: '12px', 
          background: 'var(--bg-primary)', 
          borderRadius: '10px', 
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Target</div>
          <div style={{ fontSize: '16px', fontWeight: '700', color: '#22c55e' }}>{result.target}</div>
        </div>
      </div>
    </div>
  )
}

// ========== MAIN COMPONENT ==========
export default function PipelinePage() {
  const { isDark } = useTheme()
  const [ticker, setTicker] = useState('NVDA')
  const [researchMode, setResearchMode] = useState('deep')
  const [isRunning, setIsRunning] = useState(false)
  const [currentPhase, setCurrentPhase] = useState(0)
  const [currentAgent, setCurrentAgent] = useState('')
  const [progress, setProgress] = useState(0)
  const [logs, setLogs] = useState([])
  const [agentStatuses, setAgentStatuses] = useState({})
  const [agentOutputs, setAgentOutputs] = useState({})
  const [result, setResult] = useState(null)
  const [selectedPhase, setSelectedPhase] = useState(null)
  const logsEndRef = useRef(null)

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const addLog = (message, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs(prev => [...prev, { time, message, type }])
  }

  const runPipeline = async () => {
    setIsRunning(true)
    setLogs([])
    setResult(null)
    setAgentStatuses({})
    setAgentOutputs({})
    setProgress(0)
    setCurrentPhase(0)

    addLog(`Starting ${ticker} analysis pipeline`, 'info')
    addLog(`Mode: ${researchMode.toUpperCase()} | Model: GPT-4o-mini`, 'info')

    const totalAgents = PHASES.reduce((sum, p) => sum + p.agents.length, 0)
    let completedAgents = 0

    for (let i = 0; i < PHASES.length; i++) {
      const phase = PHASES[i]
      setCurrentPhase(i + 1)
      setSelectedPhase(i + 1)
      
      addLog(`Phase ${phase.num}: ${phase.name.toUpperCase()}`, 'phase')

      for (const agent of phase.agents) {
        setCurrentAgent(agent.name)
        setAgentStatuses(prev => ({ ...prev, [agent.name]: 'running' }))
        addLog(`  Running ${agent.name}...`, 'agent_start')

        const delay = researchMode === 'shallow' ? 400 : researchMode === 'deep' ? 700 : 1000
        await new Promise(r => setTimeout(r, delay + Math.random() * 300))

        const output = generateAgentOutput(agent.name, ticker)
        setAgentOutputs(prev => ({ ...prev, [agent.name]: output }))
        setAgentStatuses(prev => ({ ...prev, [agent.name]: 'complete' }))
        
        completedAgents++
        setProgress((completedAgents / totalAgents) * 100)

        let outputSummary = ''
        if (output.signal) outputSummary = `Signal: ${output.signal}`
        else if (output.sentiment) outputSummary = `Sentiment: ${output.sentiment}`
        else if (output.thesis) outputSummary = output.thesis.slice(0, 40) + '...'
        else if (output.stance) outputSummary = `${output.stance} @ ${output.position_pct}`
        else if (output.verdict) outputSummary = `${output.verdict} - ${output.position_dollars}`
        else if (output.winner) outputSummary = `Winner: ${output.winner}`

        addLog(`  ${agent.name} complete → ${outputSummary}`, 'success')
      }

      addLog(`  Output: ${phase.output}`, 'info')
    }

    const finalOutput = generateAgentOutput('Risk Manager', ticker)
    setResult(finalOutput)
    addLog(`FINAL: ${finalOutput.verdict} ${finalOutput.verdict !== 'REJECT' ? finalOutput.position_dollars : ''}`, 'result')
    
    setIsRunning(false)
    setCurrentPhase(0)
    setCurrentAgent('')
  }

  const resetPipeline = () => {
    setIsRunning(false)
    setLogs([])
    setResult(null)
    setAgentStatuses({})
    setAgentOutputs({})
    setProgress(0)
    setCurrentPhase(0)
    setSelectedPhase(null)
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)' }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* Header */}
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
          <span style={{ 
            padding: '5px 10px', 
            background: 'var(--bg-tertiary)', 
            borderRadius: '6px',
            color: 'var(--text-secondary)'
          }}>
            {PHASES.reduce((sum, p) => sum + p.agents.length, 0)} Agents
          </span>
          <span style={{ 
            padding: '5px 10px', 
            background: 'var(--bg-tertiary)', 
            borderRadius: '6px',
            color: 'var(--text-secondary)'
          }}>
            5 Phases
          </span>
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
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Panel - Config */}
        <aside style={{
          width: '300px',
          background: 'var(--bg-card)',
          borderRight: '1px solid var(--border-primary)',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* Configuration */}
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
            
            <div style={{ marginBottom: '14px' }}>
              <label style={{ 
                display: 'block', 
                fontSize: '11px', 
                color: 'var(--text-muted)', 
                marginBottom: '6px' 
              }}>
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

            <div style={{ marginBottom: '14px' }}>
              <label style={{ 
                display: 'block', 
                fontSize: '11px', 
                color: 'var(--text-muted)', 
                marginBottom: '6px' 
              }}>
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
                    <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>
                      {mode.label}
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
                      {mode.time}
                    </div>
                  </button>
                ))}
              </div>
            </div>

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
                <Icons.Play size={14} /> Run Pipeline
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
          </div>

          {/* Progress */}
          {(isRunning || progress > 0) && (
            <div style={{ padding: '16px', borderBottom: '1px solid var(--border-primary)' }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                fontSize: '12px', 
                marginBottom: '8px' 
              }}>
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
                  background: PHASES[Math.max(0, currentPhase - 1)]?.color || 'var(--text-primary)',
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
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between' 
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '18px' }}>{phase.icon}</span>
                      <div>
                        <div style={{ 
                          fontWeight: '600', 
                          fontSize: '12px',
                          color: 'var(--text-primary)'
                        }}>
                          Phase {phase.num}: {phase.name}
                        </div>
                        <div style={{ 
                          fontSize: '10px', 
                          color: 'var(--text-muted)' 
                        }}>
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

        {/* Main Content */}
        <main style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
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
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  marginBottom: '16px' 
                }}>
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
                    <h3 style={{ 
                      margin: 0, 
                      fontSize: '16px', 
                      fontWeight: '600',
                      color: 'var(--text-primary)'
                    }}>
                      Phase {selectedPhase}: {PHASES[selectedPhase - 1].name}
                    </h3>
                    <p style={{ 
                      margin: '2px 0 0', 
                      fontSize: '12px', 
                      color: 'var(--text-muted)' 
                    }}>
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
                <h3 style={{ 
                  margin: 0, 
                  fontSize: '14px', 
                  fontWeight: '600', 
                  color: '#e5e7eb' 
                }}>
                  Execution Log
                </h3>
                {logs.length > 0 && (
                  <button
                    onClick={() => setLogs([])}
                    style={{
                      padding: '4px 10px',
                      background: '#374151',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#9ca3af',
                      fontSize: '11px',
                      cursor: 'pointer'
                    }}
                  >
                    Clear
                  </button>
                )}
              </div>
              <div style={{ 
                maxHeight: '300px', 
                overflowY: 'auto', 
                fontFamily: 'monospace',
                fontSize: '12px'
              }} className="hide-scrollbar">
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
                        <span style={{ color: '#4b5563', marginRight: '10px' }}>
                          [{log.time}]
                        </span>
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
      </div>
    </div>
  )
}