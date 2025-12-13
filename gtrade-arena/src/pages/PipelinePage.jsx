import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useTheme } from '../App'

// ========== DISPLAY OPTIONS ==========
const DISPLAY_OPTIONS = {
  BANNER: false,
  FOOTER: true,
}

// ========== TICKERS (suggestions for dropdown) ==========
const TICKER_SUGGESTIONS = [
  'NVDA', 'AAPL', 'GOOG', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA', 'BRK.B', 
  'WMT', 'LLY', 'JPM', 'V', 'ORCL', 'XOM', 'JNJ', 'MA', 'NFLX', 'PLTR', 'ABBV', 
  'COST', 'BAC', 'AMD', 'HD', 'PG', 'CSCO', 'GE', 'KO', 'CVX', 'UNH', 'IBM', 
  'WFC', 'MS', 'CAT', 'MU', 'GS', 'AXP', 'MRK', 'CRM', 'APP', 'PM', 'RTX', 
  'TMUS', 'MCD', 'TMO', 'AMAT', 'ABT', 'LRCX', 'PEP'
]

// ========== PHASES ==========
const PHASES = [
  { 
    num: 1, 
    name: 'Market Analysis', 
    icon: '📊',
    color: '#6366f1',
    agents: [
      { name: 'Technical Analyst', icon: '📈', desc: 'RSI, MACD, Bollinger Bands, support/resistance', outputKey: 'technical' },
      { name: 'News Analyst', icon: '📰', desc: 'Sentiment from Yahoo Finance, Finnhub, Reuters', outputKey: 'news' },
      { name: 'Fundamental Analyst', icon: '📋', desc: 'P/E, growth rates, margins, valuation', outputKey: 'fundamental' },
      { name: 'Macro Analyst', icon: '🌍', desc: 'Fed policy, sector rotation, market regime', outputKey: 'macro' },
    ],
    duration: '~60s',
    output: 'discussion_points.json',
    outputKey: 'phase1'
  },
  { 
    num: 2, 
    name: 'Bull/Bear Research', 
    icon: '⚔️',
    color: '#f59e0b',
    agents: [
      { name: 'Bull Researcher', icon: '🐂', desc: 'Builds bullish thesis with catalysts and upside targets', outputKey: 'bull' },
      { name: 'Bear Researcher', icon: '🐻', desc: 'Constructs bearish case with risks and downside scenarios', outputKey: 'bear' },
    ],
    duration: '~40s',
    output: 'research_theses.json',
    outputKey: 'phase2'
  },
  { 
    num: 3, 
    name: 'Debate & Synthesis', 
    icon: '⚖️',
    color: '#10b981',
    agents: [
      { name: 'Research Manager', icon: '👨‍⚖️', desc: 'Moderates bull/bear debate, synthesizes findings', outputKey: 'synthesis' },
    ],
    duration: '~30s',
    output: 'research_synthesis.json',
    outputKey: 'phase3'
  },
  { 
    num: 4, 
    name: 'Risk Evaluation', 
    icon: '🎯',
    color: '#ec4899',
    agents: [
      { name: 'Aggressive Evaluator', icon: '🔥', desc: 'Risk-seeking, higher positions, default: STRONG BUY', outputKey: 'aggressive' },
      { name: 'Neutral Evaluator', icon: '⚖️', desc: 'Balanced assessment, default: HOLD', outputKey: 'neutral' },
      { name: 'Conservative Evaluator', icon: '🛡️', desc: 'Risk-averse, lower positions, default: AVOID', outputKey: 'conservative' },
    ],
    duration: '~30s',
    output: 'evaluations.json',
    outputKey: 'phase4'
  },
  { 
    num: 5, 
    name: 'Final Decision', 
    icon: '✅',
    color: '#8b5cf6',
    agents: [
      { name: 'Risk Manager', icon: '👔', desc: 'Makes final BUY/HOLD/REJECT with position sizing', outputKey: 'decision' },
    ],
    duration: '~10s',
    output: 'risk_decision.json',
    outputKey: 'phase5'
  },
]

const TOTAL_AGENTS = PHASES.reduce((sum, p) => sum + p.agents.length, 0)

// ========== LOCAL STORAGE ==========
const STORAGE_KEY = 'pipeline_state'
const saveState = (state) => {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...state, savedAt: Date.now() })) } catch (e) {}
}
const loadState = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      if (Date.now() - parsed.savedAt < 3600000 && !parsed.isRunning) return parsed
    }
  } catch (e) {}
  return null
}
const clearState = () => { try { localStorage.removeItem(STORAGE_KEY) } catch (e) {} }

// ========== ICONS ==========
const Icons = {
  Play: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>,
  X: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>,
  Check: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>,
  Github: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>,
  Calendar: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
}

// ========== TICKER COMBOBOX ==========
const TickerCombobox = ({ value, onChange, disabled, isDark }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [inputValue, setInputValue] = useState(value)
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)

  useEffect(() => { setInputValue(value) }, [value])

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) setIsOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const filtered = TICKER_SUGGESTIONS.filter(t => t.toLowerCase().includes(inputValue.toLowerCase()))

  const handleInputChange = (e) => {
    const val = e.target.value.toUpperCase()
    setInputValue(val)
    onChange(val)
    setIsOpen(true)
  }

  const handleSelect = (ticker) => {
    setInputValue(ticker)
    onChange(ticker)
    setIsOpen(false)
  }

  return (
    <div ref={dropdownRef} style={{ position: 'relative' }}>
      <input
        ref={inputRef}
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onFocus={() => setIsOpen(true)}
        disabled={disabled}
        placeholder="Enter ticker..."
        style={{
          width: '100%',
          padding: '10px 12px',
          border: '1px solid var(--border-primary)',
          borderRadius: '8px',
          fontSize: '14px',
          fontWeight: '600',
          background: 'var(--bg-secondary)',
          color: 'var(--text-primary)',
          cursor: disabled ? 'not-allowed' : 'text'
        }}
      />
      {isOpen && filtered.length > 0 && !disabled && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          maxHeight: '200px',
          overflowY: 'auto',
          background: isDark ? '#1f1f1f' : '#fff',
          border: '1px solid var(--border-primary)',
          borderRadius: '8px',
          marginTop: '4px',
          zIndex: 100,
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        }}>
          {filtered.slice(0, 10).map(ticker => (
            <div
              key={ticker}
              onClick={() => handleSelect(ticker)}
              style={{
                padding: '10px 12px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '500',
                color: 'var(--text-primary)',
                background: ticker === inputValue ? (isDark ? '#27272a' : '#f4f4f5') : 'transparent',
              }}
              onMouseEnter={(e) => e.target.style.background = isDark ? '#27272a' : '#f4f4f5'}
              onMouseLeave={(e) => e.target.style.background = ticker === inputValue ? (isDark ? '#27272a' : '#f4f4f5') : 'transparent'}
            >
              {ticker}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ========== AGENT CARD WITH OUTPUT ==========
const AgentCard = ({ agent, status, output, isActive, phaseColor, isDark }) => {
  const [showOutput, setShowOutput] = useState(false)
  
  return (
    <div style={{
      padding: '14px',
      background: isActive ? (isDark ? '#1a1a2e' : '#f0f0ff')
        : status === 'complete' ? (isDark ? '#0f1f0f' : '#f0fdf4')
        : 'var(--bg-tertiary)',
      borderRadius: '10px',
      border: isActive ? `2px solid ${phaseColor}` 
        : status === 'complete' ? '1px solid #22c55e44' 
        : '1px solid var(--border-primary)',
      marginBottom: '10px',
      transition: 'all 0.3s'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '20px' }}>{agent.icon}</span>
          <div>
            <div style={{ fontWeight: '600', fontSize: '13px', color: 'var(--text-primary)' }}>{agent.name}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{agent.desc}</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isActive && <span style={{ width: 14, height: 14, border: `2px solid ${phaseColor}`, borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />}
          {status === 'complete' && (
            <>
              <span style={{ color: '#22c55e' }}><Icons.Check size={16} /></span>
              {output && (
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
              )}
            </>
          )}
          {status === 'pending' && !isActive && <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Pending</span>}
        </div>
      </div>
      
      {/* OUTPUT DISPLAY */}
      {showOutput && output && (
        <div style={{
          marginTop: '12px',
          padding: '12px',
          background: isDark ? '#0a0a0a' : '#fafafa',
          borderRadius: '8px',
          fontSize: '11px',
          fontFamily: 'monospace',
          maxHeight: '300px',
          overflowY: 'auto'
        }}>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
            {typeof output === 'string' ? output : JSON.stringify(output, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ========== RESULT CARD ==========
const ResultCard = ({ result, ticker, isDark }) => {
  if (!result) return null
  const colors = {
    BUY: { bg: isDark ? '#0f1f0f' : '#f0fdf4', border: '#22c55e', text: '#22c55e' },
    HOLD: { bg: isDark ? '#1f1a0f' : '#fffbeb', border: '#f59e0b', text: '#f59e0b' },
    REJECT: { bg: isDark ? '#1f0f0f' : '#fef2f2', border: '#ef4444', text: '#ef4444' },
  }
  const c = colors[result.verdict] || colors.HOLD
  
  return (
    <div style={{ background: c.bg, border: `1px solid ${c.border}44`, borderRadius: '14px', padding: '20px', marginBottom: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>{ticker} Analysis Complete</div>
          <div style={{ fontSize: '32px', fontWeight: '800', color: c.text }}>{result.verdict}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          {result.verdict !== 'REJECT' && <div style={{ fontSize: '24px', fontWeight: '700', color: '#22c55e' }}>{result.position_dollars}</div>}
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{result.confidence} Confidence</div>
        </div>
      </div>
      <div style={{ marginTop: '16px', padding: '14px', background: 'var(--bg-primary)', borderRadius: '10px' }}>
        <div style={{ fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>Reasoning</div>
        <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{result.reasoning}</p>
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
      {result.evaluator_consensus && (
        <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
          {Object.entries(result.evaluator_consensus).map(([type, stance]) => (
            <div key={type} style={{ flex: 1, padding: '10px', background: 'var(--bg-primary)', borderRadius: '8px', textAlign: 'center' }}>
              <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'capitalize' }}>{type}</div>
              <div style={{ fontSize: '11px', fontWeight: '700', color: stance.includes('BUY') ? '#22c55e' : stance === 'HOLD' ? '#f59e0b' : '#ef4444' }}>{stance}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ========== FOOTER ==========
const PageFooter = ({ isDark }) => (
  <div style={{ padding: '6px 20px', borderTop: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', background: 'var(--bg-secondary)', flexShrink: 0 }}>
    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Trade Arena • 11 LLM agents with live market data</span>
    <span style={{ color: 'var(--text-muted)', opacity: 0.4 }}>•</span>
    <a href="https://github.com/priyam-03/Capstone" target="_blank" rel="noopener noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: 'var(--text-muted)', textDecoration: 'none' }}>
      <Icons.Github size={12} /> View Source
    </a>
  </div>
)

// ========== MAIN COMPONENT ==========
export default function PipelinePage() {
  const { isDark } = useTheme()
  
  // Config state
  const [ticker, setTicker] = useState('NVDA')
  const [researchMode, setResearchMode] = useState('shallow')
  const [portfolioValue, setPortfolioValue] = useState(100000)
  const [analysisDate, setAnalysisDate] = useState('')
  
  // Pipeline state
  const [isRunning, setIsRunning] = useState(false)
  const [currentPhase, setCurrentPhase] = useState(0)
  const [currentAgent, setCurrentAgent] = useState('')
  const [progress, setProgress] = useState(0)
  const [logs, setLogs] = useState([])
  const [agentStatuses, setAgentStatuses] = useState({})
  const [agentOutputs, setAgentOutputs] = useState({})
  const [result, setResult] = useState(null)
  const [selectedPhase, setSelectedPhase] = useState(1)
  const [completedAgents, setCompletedAgents] = useState(0)
  const [rawApiResponse, setRawApiResponse] = useState(null)
  
  const logsEndRef = useRef(null)
  const abortControllerRef = useRef(null)
  const progressIntervalRef = useRef(null)

  // Load persisted state
  useEffect(() => {
    const saved = loadState()
    if (saved) {
      setTicker(saved.ticker || 'NVDA')
      setResearchMode(saved.researchMode || 'shallow')
      setPortfolioValue(saved.portfolioValue || 100000)
      setProgress(saved.progress || 0)
      setLogs(saved.logs || [])
      setAgentStatuses(saved.agentStatuses || {})
      setAgentOutputs(saved.agentOutputs || {})
      setResult(saved.result || null)
      setCompletedAgents(saved.completedAgents || 0)
      setRawApiResponse(saved.rawApiResponse || null)
    }
  }, [])

  // Persist state
  useEffect(() => {
    if (!isRunning && (progress > 0 || result)) {
      saveState({ ticker, researchMode, portfolioValue, progress, logs, agentStatuses, agentOutputs, result, completedAgents, rawApiResponse, isRunning: false })
    }
  }, [ticker, researchMode, portfolioValue, progress, logs, agentStatuses, agentOutputs, result, completedAgents, rawApiResponse, isRunning])

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [logs])
  useEffect(() => () => { progressIntervalRef.current && clearInterval(progressIntervalRef.current); abortControllerRef.current?.abort() }, [])

  const addLog = useCallback((message, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs(prev => [...prev, { time, message, type }])
  }, [])

  const startProgressSimulation = useCallback(() => {
    let agentIndex = 0, phaseIndex = 0
    const allAgents = PHASES.flatMap((phase, pIdx) => phase.agents.map(agent => ({ ...agent, phaseNum: pIdx + 1 })))
    const interval = { shallow: 8000, deep: 20000, research: 35000 }[researchMode] || 8000
    
    if (allAgents.length > 0) {
      setCurrentPhase(1)
      setCurrentAgent(allAgents[0].name)
      setSelectedPhase(1)
      addLog(`▶ Phase 1: ${PHASES[0].name}`, 'phase')
      addLog(`  → ${allAgents[0].name} analyzing...`, 'agent_start')
    }
    
    progressIntervalRef.current = setInterval(() => {
      if (agentIndex < allAgents.length) {
        const agent = allAgents[agentIndex]
        setAgentStatuses(prev => ({ ...prev, [agent.name]: 'complete' }))
        setCompletedAgents(prev => { const n = prev + 1; setProgress((n / TOTAL_AGENTS) * 95); return n })
        addLog(`  ✓ ${agent.name} complete`, 'success')
        agentIndex++
        if (agentIndex < allAgents.length) {
          const next = allAgents[agentIndex]
          if (next.phaseNum !== agent.phaseNum) {
            phaseIndex++
            setCurrentPhase(next.phaseNum)
            setSelectedPhase(next.phaseNum)
            addLog(`▶ Phase ${next.phaseNum}: ${PHASES[phaseIndex].name}`, 'phase')
          }
          setCurrentAgent(next.name)
          addLog(`  → ${next.name} analyzing...`, 'agent_start')
        } else {
          setCurrentAgent('')
          addLog(`  ⏳ Finalizing...`, 'info')
        }
      }
    }, interval)
  }, [researchMode, addLog])

  // ========== EXTRACT OUTPUTS FROM API RESPONSE ==========
  const extractAgentOutputs = (data) => {
    const outputs = {}
    
    // Try to extract from phase_results if available
    if (data.phase_results) {
      // Phase 1 - Analysts (key can be 1 or "phase1")
      const p1 = data.phase_results[1]?.data || data.phase_results['phase1']?.data
      if (p1) {
        outputs['Technical Analyst'] = p1.technical || p1.discussion_points?.technical || p1
        outputs['News Analyst'] = p1.news || p1.discussion_points?.news || p1
        outputs['Fundamental Analyst'] = p1.fundamental || p1.discussion_points?.fundamental || p1
        outputs['Macro Analyst'] = p1.macro || p1.discussion_points?.macro || p1
        // If no individual outputs, use the whole discussion_points
        if (p1.discussion_points && !outputs['Technical Analyst']) {
          outputs['Technical Analyst'] = p1.discussion_points
          outputs['News Analyst'] = p1.discussion_points
          outputs['Fundamental Analyst'] = p1.discussion_points
          outputs['Macro Analyst'] = p1.discussion_points
        }
      }
      
      // Phase 2 - Researchers
      const p2 = data.phase_results[2]?.data || data.phase_results['phase2']?.data
      if (p2) {
        outputs['Bull Researcher'] = p2.bull || p2
        outputs['Bear Researcher'] = p2.bear || p2
      }
      
      // Phase 3 - Research Manager
      const p3 = data.phase_results[3]?.data || data.phase_results['phase3']?.data
      if (p3) {
        outputs['Research Manager'] = p3.synthesis || p3
      }
      
      // Phase 4 - Risk Team
      const p4 = data.phase_results[4]?.data || data.phase_results['phase4']?.data
      if (p4) {
        outputs['Aggressive Evaluator'] = p4.aggressive || p4
        outputs['Neutral Evaluator'] = p4.neutral || p4
        outputs['Conservative Evaluator'] = p4.conservative || p4
      }
      
      // Phase 5 - Risk Manager
      const p5 = data.phase_results[5]?.data || data.phase_results['phase5']?.data
      if (p5) {
        outputs['Risk Manager'] = p5.decision || p5
      }
    }
    
    // Also use final_decision for Risk Manager if available
    if (data.final_decision) {
      outputs['Risk Manager'] = data.final_decision
    }
    
    // Use risk_consensus for evaluators if available in final_decision
    if (data.final_decision?.risk_consensus?.stances) {
      const stances = data.final_decision.risk_consensus.stances
      const sizes = data.final_decision.risk_consensus.position_sizes || {}
      if (!outputs['Aggressive Evaluator']) {
        outputs['Aggressive Evaluator'] = { stance: stances.aggressive, position_size: sizes.aggressive }
      }
      if (!outputs['Neutral Evaluator']) {
        outputs['Neutral Evaluator'] = { stance: stances.neutral, position_size: sizes.neutral }
      }
      if (!outputs['Conservative Evaluator']) {
        outputs['Conservative Evaluator'] = { stance: stances.conservative, position_size: sizes.conservative }
      }
    }
    
    return outputs
  }

  const runPipeline = async () => {
    setIsRunning(true)
    setLogs([])
    setResult(null)
    setAgentStatuses({})
    setAgentOutputs({})
    setProgress(0)
    setCurrentPhase(0)
    setCurrentAgent('')
    setCompletedAgents(0)
    setRawApiResponse(null)
    clearState()

    addLog(`🚀 Starting ${ticker} analysis`, 'info')
    addLog(`Mode: ${researchMode.toUpperCase()} | Portfolio: $${portfolioValue.toLocaleString()}`, 'info')
    if (analysisDate) addLog(`📅 Historical Mode: ${analysisDate}`, 'info')

    abortControllerRef.current = new AbortController()
    startProgressSimulation()

    try {
      const params = new URLSearchParams({
        ticker,
        port_val: portfolioValue.toString(),
        research_mode: researchMode,
      })
      if (analysisDate) params.append('analysis_date', analysisDate)

      addLog(`📡 Calling API...`, 'info')
      
      const response = await fetch(`/api/master_orch?${params}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal
      })
      
      addLog(`📥 API responded with status: ${response.status}`, 'info')

      progressIntervalRef.current && clearInterval(progressIntervalRef.current)

      if (!response.ok) {
        const err = await response.json()
        addLog(`❌ API Error: ${err.detail || response.statusText}`, 'error')
        setIsRunning(false)
        return
      }

      const data = await response.json()
      setRawApiResponse(data) // Store raw response
      
      // DEBUG: Log the full API response to console
      console.log('=== FULL API RESPONSE ===')
      console.log(JSON.stringify(data, null, 2))
      console.log('=== PHASE RESULTS ===')
      console.log(data.phase_results)
      console.log('=== FINAL DECISION ===')
      console.log(data.final_decision)
      
      addLog(`✅ Pipeline complete`, 'success')

      // Mark all agents complete
      const statuses = {}
      PHASES.forEach(p => p.agents.forEach(a => { statuses[a.name] = 'complete' }))
      setAgentStatuses(statuses)

      // Extract outputs for each agent
      const outputs = extractAgentOutputs(data)
      setAgentOutputs(outputs)

      // Log phase results
      if (data.phase_results) {
        Object.entries(data.phase_results).forEach(([phase, r]) => {
          addLog(`Phase ${phase}: ${r.status}`, r.status === 'SUCCESS' ? 'success' : 'info')
        })
      }

      // Set final result
      if (data.final_decision) {
        const d = data.final_decision
        setResult({
          verdict: d.verdict,
          position_dollars: d.final_position_dollars ? `$${Math.round(d.final_position_dollars).toLocaleString()}` : '$0',
          confidence: d.confidence,
          stop_loss: d.risk_controls?.stop_loss?.percentage ? `-${d.risk_controls.stop_loss.percentage.toFixed(0)}%` : '-8%',
          target: d.risk_controls?.take_profit?.percentage ? `+${d.risk_controls.take_profit.percentage.toFixed(0)}%` : '+15%',
          reasoning: d.reasoning || 'Analysis complete',
          evaluator_consensus: d.evaluator_consensus || null
        })
        addLog(`━━━━━━━━━━━━━━━━━━━━━━━━`, 'result')
        addLog(`FINAL: ${d.verdict} | $${Math.round(d.final_position_dollars || 0).toLocaleString()}`, 'result')
        addLog(`Confidence: ${d.confidence}`, 'result')
        addLog(`━━━━━━━━━━━━━━━━━━━━━━━━`, 'result')
      }

      setProgress(100)
      setCompletedAgents(TOTAL_AGENTS)
      setCurrentPhase(6)
    } catch (error) {
      progressIntervalRef.current && clearInterval(progressIntervalRef.current)
      if (error.name === 'AbortError') {
        addLog(`⚠️ Cancelled`, 'info')
      } else {
        console.error('Pipeline error:', error)
        addLog(`❌ Error: ${error.message}`, 'error')
      }
    } finally {
      setIsRunning(false)
      setCurrentAgent('')
    }
  }

  const resetPipeline = () => {
    abortControllerRef.current?.abort()
    progressIntervalRef.current && clearInterval(progressIntervalRef.current)
    setIsRunning(false)
    setLogs([])
    setResult(null)
    setAgentStatuses({})
    setAgentOutputs({})
    setProgress(0)
    setCurrentPhase(0)
    setCurrentAgent('')
    setSelectedPhase(1)
    setCompletedAgents(0)
    setRawApiResponse(null)
    clearState()
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)' }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* HEADER */}
      <header style={{ background: 'var(--bg-primary)', borderBottom: '1px solid var(--border-primary)', padding: '12px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>Analysis Pipeline</h1>
          <p style={{ margin: '2px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>5-Phase LLM Workflow • 11 Agents</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px' }}>
          <span style={{ padding: '5px 10px', background: 'var(--bg-tertiary)', borderRadius: '6px', color: 'var(--text-secondary)' }}>{completedAgents}/{TOTAL_AGENTS} Agents</span>
          {isRunning && (
            <span style={{ padding: '5px 10px', background: isDark ? '#1a1a2e' : '#f0f0ff', borderRadius: '6px', color: '#6366f1', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: 8, height: 8, background: '#6366f1', borderRadius: '50%', animation: 'pulse 1s infinite' }} /> Processing
            </span>
          )}
          {!isRunning && progress === 100 && (
            <span style={{ padding: '5px 10px', background: isDark ? '#0f1f0f' : '#f0fdf4', borderRadius: '6px', color: '#22c55e', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Icons.Check size={12} /> Complete
            </span>
          )}
        </div>
      </header>

      {/* MAIN LAYOUT */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        
        {/* LEFT SIDEBAR */}
        <aside style={{ width: '300px', background: 'var(--bg-card)', borderRight: '1px solid var(--border-primary)', display: 'flex', flexDirection: 'column' }}>
          
          {/* Config */}
          <div style={{ padding: '16px', borderBottom: '1px solid var(--border-primary)' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase' }}>Configuration</div>
            
            {/* Ticker - Combobox */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>Ticker</label>
              <TickerCombobox value={ticker} onChange={setTicker} disabled={isRunning} isDark={isDark} />
            </div>

            {/* Portfolio Value */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>Portfolio Value ($)</label>
              <input
                type="number"
                value={portfolioValue}
                onChange={e => setPortfolioValue(Number(e.target.value) || 100000)}
                disabled={isRunning}
                style={{ width: '100%', padding: '10px 12px', border: '1px solid var(--border-primary)', borderRadius: '8px', fontSize: '14px', fontWeight: '600', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
              />
            </div>

            {/* Analysis Date (Historical) */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                <Icons.Calendar size={12} /> Analysis Date <span style={{ opacity: 0.6, fontSize: '10px' }}>(optional - for backtesting)</span>
              </label>
              <input
                type="date"
                value={analysisDate}
                onChange={e => setAnalysisDate(e.target.value)}
                disabled={isRunning}
                style={{ width: '100%', padding: '10px 12px', border: '1px solid var(--border-primary)', borderRadius: '8px', fontSize: '13px', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
              />
            </div>

            {/* Research Mode */}
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>Research Mode</label>
              <div style={{ display: 'flex', gap: '6px' }}>
                {[{ id: 'shallow', label: 'Fast', time: '~2 min', rounds: 0 }, { id: 'deep', label: 'Deep', time: '~5 min', rounds: 3 }, { id: 'research', label: 'Full', time: '~8 min', rounds: 5 }].map(mode => (
                  <button key={mode.id} onClick={() => setResearchMode(mode.id)} disabled={isRunning}
                    style={{ flex: 1, padding: '10px 8px', border: researchMode === mode.id ? '2px solid var(--text-primary)' : '1px solid var(--border-primary)', borderRadius: '8px', background: researchMode === mode.id ? (isDark ? '#27272a' : '#f4f4f5') : 'var(--bg-secondary)', cursor: isRunning ? 'not-allowed' : 'pointer', fontSize: '11px' }}>
                    <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{mode.label}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>{mode.time}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Run Button */}
            {!isRunning ? (
              <button onClick={runPipeline} style={{ width: '100%', padding: '12px', background: 'var(--text-primary)', color: 'var(--bg-primary)', border: 'none', borderRadius: '10px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                <Icons.Play size={14} /> {progress === 100 ? 'Run Again' : 'Run Pipeline'}
              </button>
            ) : (
              <button onClick={resetPipeline} style={{ width: '100%', padding: '12px', background: isDark ? '#2a1a1a' : '#fef2f2', color: '#ef4444', border: '1px solid #ef444444', borderRadius: '10px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                <Icons.X size={14} /> Cancel
              </button>
            )}

            {!isRunning && progress === 100 && (
              <button onClick={resetPipeline} style={{ width: '100%', marginTop: '8px', padding: '10px', background: 'transparent', color: 'var(--text-muted)', border: '1px solid var(--border-primary)', borderRadius: '10px', fontSize: '12px', cursor: 'pointer' }}>
                Clear Results
              </button>
            )}
          </div>

          {/* Progress */}
          {(isRunning || progress > 0) && (
            <div style={{ padding: '16px', borderBottom: '1px solid var(--border-primary)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '8px' }}>
                <span style={{ fontWeight: '600', color: 'var(--text-primary)' }}>Progress</span>
                <span style={{ color: 'var(--text-muted)' }}>{Math.round(progress)}%</span>
              </div>
              <div style={{ height: '6px', background: 'var(--bg-tertiary)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress}%`, background: progress === 100 ? '#22c55e' : (PHASES[Math.max(0, currentPhase - 1)]?.color || '#6366f1'), transition: 'width 0.5s', borderRadius: '3px' }} />
              </div>
              {currentAgent && (
                <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: PHASES[Math.max(0, currentPhase - 1)]?.color || '#6366f1' }}>
                  <span style={{ width: 6, height: 6, background: 'currentColor', borderRadius: '50%', animation: 'pulse 1s infinite' }} />
                  {currentAgent}
                </div>
              )}
            </div>
          )}

          {/* Phases */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }} className="hide-scrollbar">
            <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>Phases</div>
            {PHASES.map(phase => {
              const complete = phase.agents.every(a => agentStatuses[a.name] === 'complete')
              const active = currentPhase === phase.num && isRunning
              return (
                <div key={phase.num} onClick={() => setSelectedPhase(phase.num)}
                  style={{ padding: '12px', marginBottom: '8px', borderRadius: '10px', cursor: 'pointer', border: selectedPhase === phase.num ? `2px solid ${phase.color}` : '1px solid var(--border-primary)', background: active ? (isDark ? '#1a1a2e' : '#f5f5ff') : complete ? (isDark ? '#0f1f0f' : '#f0fdf4') : 'var(--bg-secondary)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '18px' }}>{phase.icon}</span>
                      <div>
                        <div style={{ fontWeight: '600', fontSize: '12px', color: 'var(--text-primary)' }}>Phase {phase.num}: {phase.name}</div>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{phase.agents.filter(a => agentStatuses[a.name] === 'complete').length}/{phase.agents.length} agents • {phase.duration}</div>
                      </div>
                    </div>
                    {complete && <span style={{ color: '#22c55e' }}><Icons.Check size={14} /></span>}
                    {active && !complete && <span style={{ width: 12, height: 12, border: `2px solid ${phase.color}`, borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />}
                  </div>
                </div>
              )
            })}
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <main style={{ flex: 1, padding: '20px', overflowY: 'auto' }} className="hide-scrollbar">
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              
              <ResultCard result={result} ticker={ticker} isDark={isDark} />

              {selectedPhase && (
                <div style={{ background: 'var(--bg-card)', borderRadius: '14px', border: '1px solid var(--border-primary)', padding: '20px', marginBottom: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                    <div style={{ width: 44, height: 44, borderRadius: '12px', background: PHASES[selectedPhase - 1].color + '22', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '22px' }}>
                      {PHASES[selectedPhase - 1].icon}
                    </div>
                    <div>
                      <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600', color: 'var(--text-primary)' }}>Phase {selectedPhase}: {PHASES[selectedPhase - 1].name}</h3>
                      <p style={{ margin: '2px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>{PHASES[selectedPhase - 1].agents.length} agents • Output: {PHASES[selectedPhase - 1].output}</p>
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
              <div style={{ background: isDark ? '#0a0a0a' : '#1e1e1e', borderRadius: '14px', padding: '16px', minHeight: '200px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#e5e7eb' }}>Execution Log</h3>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', color: '#6b7280' }}>{logs.length} entries</span>
                    {logs.length > 0 && !isRunning && (
                      <button onClick={() => setLogs([])} style={{ padding: '4px 10px', background: '#374151', border: 'none', borderRadius: '4px', color: '#9ca3af', fontSize: '11px', cursor: 'pointer' }}>Clear</button>
                    )}
                  </div>
                </div>
                <div style={{ maxHeight: '300px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '12px' }} className="hide-scrollbar">
                  {logs.length === 0 ? (
                    <div style={{ color: '#6b7280', padding: '20px', textAlign: 'center' }}>Click "Run Pipeline" to start...</div>
                  ) : logs.map((log, i) => {
                    const colors = { phase: '#6366f1', success: '#22c55e', result: '#f59e0b', error: '#ef4444', info: '#6b7280', agent_start: '#9ca3af' }
                    return (
                      <div key={i} style={{ padding: '4px 0', color: colors[log.type] || '#6b7280', fontWeight: ['phase', 'result', 'error'].includes(log.type) ? '600' : '400', borderLeft: log.type === 'phase' ? '3px solid #6366f1' : log.type === 'result' ? '3px solid #f59e0b' : log.type === 'error' ? '3px solid #ef4444' : 'none', paddingLeft: ['phase', 'result', 'error'].includes(log.type) ? '10px' : '0', marginLeft: ['phase', 'result', 'error'].includes(log.type) ? '0' : '13px' }}>
                        <span style={{ color: '#4b5563', marginRight: '10px' }}>[{log.time}]</span>{log.message}
                      </div>
                    )
                  })}
                  <div ref={logsEndRef} />
                </div>
              </div>
            </div>
          </main>
          
          {DISPLAY_OPTIONS.FOOTER && <PageFooter isDark={isDark} />}
        </div>
      </div>
    </div>
  )
}