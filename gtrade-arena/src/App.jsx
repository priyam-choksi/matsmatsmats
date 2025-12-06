import React, { useState, useEffect, createContext, useContext, useRef, useCallback } from 'react'
import { Routes, Route } from 'react-router-dom'

// Pages
import ArenaPage from './pages/ArenaPage'
import MarketPage from './pages/MarketPage'
import PipelinePage from './pages/PipelinePage'
import PipelineHistoryPage from './pages/PipelineHistoryPage'  // <-- ADD THIS
import AgentsPage from './pages/AgentsPage'
import DocsPage from './pages/DocsPage'

// Layout Components
import Nav from './components/layout/Nav'

// ========================================
// Theme Context
// ========================================

const ThemeContext = createContext()

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}

function ThemeProvider({ children }) {
  const getInitialTheme = () => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('gtrade-theme')
      if (stored) return stored
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark'
      }
    }
    return 'light'
  }

  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    localStorage.setItem('gtrade-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggleTheme, isDark: theme === 'dark' }}>
      {children}
    </ThemeContext.Provider>
  )
}

// ========================================
// Data Context
// ========================================

const DataContext = createContext()

export function useData() {
  const context = useContext(DataContext)
  if (!context) {
    throw new Error('useData must be used within DataProvider')
  }
  return context
}

function DataProvider({ children }) {
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [tournamentData, setTournamentData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const response = await fetch('/data/tournament/combined_summary.json')
        if (response.ok) {
          const data = await response.json()
          setTournamentData(data)
        }
      } catch (error) {
        console.error('Failed to load tournament data:', error)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  return (
    <DataContext.Provider value={{ selectedTicker, setSelectedTicker, tournamentData, setTournamentData, loading }}>
      {children}
    </DataContext.Provider>
  )
}

// ========================================
// Pipeline Context (persists across pages)
// ========================================

const PipelineContext = createContext()

export function usePipeline() {
  const context = useContext(PipelineContext)
  if (!context) {
    throw new Error('usePipeline must be used within PipelineProvider')
  }
  return context
}

// Pipeline phases config
const PHASES = [
  { num: 1, name: 'Market Analysis', icon: '📊', color: '#6366f1', agents: ['Technical Analyst', 'News Analyst', 'Fundamental Analyst', 'Macro Analyst'], duration: '~60s', output: 'market_context.json' },
  { num: 2, name: 'Bull/Bear Research', icon: '⚔️', color: '#f59e0b', agents: ['Bull Researcher', 'Bear Researcher'], duration: '~40s', output: 'research_theses.json' },
  { num: 3, name: 'Debate & Synthesis', icon: '⚖️', color: '#10b981', agents: ['Research Manager'], duration: '~30s', output: 'synthesis.json' },
  { num: 4, name: 'Risk Evaluation', icon: '🎯', color: '#ec4899', agents: ['Aggressive Evaluator', 'Neutral Evaluator', 'Conservative Evaluator'], duration: '~30s', output: 'evaluations.json' },
  { num: 5, name: 'Final Decision', icon: '✅', color: '#8b5cf6', agents: ['Risk Manager'], duration: '~10s', output: 'risk_decision.json' },
]

function PipelineProvider({ children }) {
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
  
  const pipelineRef = useRef(null)

  const addLog = useCallback((message, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs(prev => [...prev, { time, message, type }])
  }, [])

  const generateAgentOutput = useCallback((agentName, tickerSymbol) => {
    const outputs = {
      'Technical Analyst': { signal: ['BULLISH', 'BEARISH', 'NEUTRAL'][Math.floor(Math.random() * 3)], rsi: (30 + Math.random() * 40).toFixed(1), macd: Math.random() > 0.5 ? 'Bullish Crossover' : 'Bearish Divergence', trend: Math.random() > 0.4 ? 'Uptrend' : 'Consolidation', support: '$' + (100 + Math.random() * 50).toFixed(2), resistance: '$' + (150 + Math.random() * 50).toFixed(2) },
      'News Analyst': { sentiment: ['Positive', 'Negative', 'Mixed'][Math.floor(Math.random() * 3)], articles_analyzed: Math.floor(10 + Math.random() * 20), key_catalyst: `${tickerSymbol} earnings beat expectations`, sentiment_score: (Math.random() * 2 - 1).toFixed(2) },
      'Fundamental Analyst': { pe_ratio: (15 + Math.random() * 30).toFixed(1), revenue_growth: (5 + Math.random() * 25).toFixed(1) + '%', profit_margin: (10 + Math.random() * 20).toFixed(1) + '%', valuation: ['Undervalued', 'Fair Value', 'Overvalued'][Math.floor(Math.random() * 3)] },
      'Macro Analyst': { regime: ['Bull Market', 'Bear Market', 'Sideways'][Math.floor(Math.random() * 3)], sector_outlook: 'Favorable', fed_impact: 'Neutral', recommendation: ['Overweight', 'Neutral', 'Underweight'][Math.floor(Math.random() * 3)] },
      'Bull Researcher': { thesis: `Strong growth catalysts for ${tickerSymbol} including market expansion`, upside_target: '+' + (15 + Math.random() * 25).toFixed(0) + '%', catalysts: ['Earnings beat', 'New product launch', 'Market share gains'], confidence: ['HIGH', 'MEDIUM'][Math.floor(Math.random() * 2)] },
      'Bear Researcher': { thesis: `Key risks include valuation concerns and competitive pressures`, downside_risk: '-' + (10 + Math.random() * 20).toFixed(0) + '%', risks: ['Valuation stretched', 'Margin compression', 'Competition'], confidence: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)] },
      'Research Manager': { winner: Math.random() > 0.4 ? 'BULL' : 'BEAR', probability_bull: (40 + Math.random() * 30).toFixed(0) + '%', probability_bear: (20 + Math.random() * 25).toFixed(0) + '%', synthesis: `After debate, ${Math.random() > 0.4 ? 'bullish' : 'bearish'} thesis prevails` },
      'Aggressive Evaluator': { stance: ['STRONG BUY', 'BUY', 'HOLD'][Math.floor(Math.random() * 3)], position_pct: (50 + Math.random() * 40).toFixed(0) + '%', reasoning: 'High conviction on growth catalysts' },
      'Neutral Evaluator': { stance: ['BUY', 'HOLD', 'AVOID'][Math.floor(Math.random() * 3)], position_pct: (25 + Math.random() * 30).toFixed(0) + '%', reasoning: 'Balanced view considering risks and rewards' },
      'Conservative Evaluator': { stance: ['HOLD', 'AVOID', 'STRONG AVOID'][Math.floor(Math.random() * 3)], position_pct: (5 + Math.random() * 20).toFixed(0) + '%', reasoning: 'Prefer capital preservation' },
      'Risk Manager': { verdict: ['BUY', 'HOLD', 'REJECT'][Math.floor(Math.random() * 3)], position_dollars: '$' + (Math.floor(Math.random() * 15000) + 5000).toLocaleString(), confidence: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)], stop_loss: '-8%', target: '+15%', reasoning: 'Final decision based on evaluator consensus and risk parameters' },
    }
    return outputs[agentName] || { status: 'Complete' }
  }, [])

  const runPipeline = useCallback(async () => {
    if (isRunning) return
    
    setIsRunning(true)
    setLogs([])
    setResult(null)
    setAgentStatuses({})
    setAgentOutputs({})
    setProgress(0)
    setCurrentPhase(0)

    addLog(`Starting ${ticker} analysis pipeline`, 'info')
    addLog(`Mode: ${researchMode.toUpperCase()} | Model: GPT-4o-mini`, 'info')

    const allAgents = PHASES.flatMap(p => p.agents)
    const totalAgents = allAgents.length
    let completedAgents = 0

    pipelineRef.current = { cancelled: false }

    for (let i = 0; i < PHASES.length; i++) {
      if (pipelineRef.current?.cancelled) break
      
      const phase = PHASES[i]
      setCurrentPhase(i + 1)
      setSelectedPhase(i + 1)
      addLog(`Phase ${phase.num}: ${phase.name.toUpperCase()}`, 'phase')

      for (const agentName of phase.agents) {
        if (pipelineRef.current?.cancelled) break
        
        setCurrentAgent(agentName)
        setAgentStatuses(prev => ({ ...prev, [agentName]: 'running' }))
        addLog(`  Running ${agentName}...`, 'agent_start')

        const delay = researchMode === 'shallow' ? 400 : researchMode === 'deep' ? 700 : 1000
        await new Promise(r => setTimeout(r, delay + Math.random() * 300))

        if (pipelineRef.current?.cancelled) break

        const output = generateAgentOutput(agentName, ticker)
        setAgentOutputs(prev => ({ ...prev, [agentName]: output }))
        setAgentStatuses(prev => ({ ...prev, [agentName]: 'complete' }))
        
        completedAgents++
        setProgress((completedAgents / totalAgents) * 100)

        let outputSummary = ''
        if (output.signal) outputSummary = `Signal: ${output.signal}`
        else if (output.sentiment) outputSummary = `Sentiment: ${output.sentiment}`
        else if (output.thesis) outputSummary = output.thesis.slice(0, 40) + '...'
        else if (output.stance) outputSummary = `${output.stance} @ ${output.position_pct}`
        else if (output.verdict) outputSummary = `${output.verdict} - ${output.position_dollars}`
        else if (output.winner) outputSummary = `Winner: ${output.winner}`

        addLog(`  ${agentName} complete → ${outputSummary}`, 'success')
      }

      if (!pipelineRef.current?.cancelled) {
        addLog(`  Output: ${phase.output}`, 'info')
      }
    }

    if (!pipelineRef.current?.cancelled) {
      const finalOutput = generateAgentOutput('Risk Manager', ticker)
      setResult(finalOutput)
      addLog(`FINAL: ${finalOutput.verdict} ${finalOutput.verdict !== 'REJECT' ? finalOutput.position_dollars : ''}`, 'result')
    }
    
    setIsRunning(false)
    setCurrentPhase(0)
    setCurrentAgent('')
    pipelineRef.current = null
  }, [ticker, researchMode, isRunning, addLog, generateAgentOutput])

  const resetPipeline = useCallback(() => {
    if (pipelineRef.current) {
      pipelineRef.current.cancelled = true
    }
    setIsRunning(false)
    setLogs([])
    setResult(null)
    setAgentStatuses({})
    setAgentOutputs({})
    setProgress(0)
    setCurrentPhase(0)
    setCurrentAgent('')
    setSelectedPhase(null)
  }, [])

  return (
    <PipelineContext.Provider value={{
      ticker, setTicker,
      researchMode, setResearchMode,
      isRunning, currentPhase, currentAgent, progress, logs,
      agentStatuses, agentOutputs, result,
      selectedPhase, setSelectedPhase,
      runPipeline, resetPipeline, addLog,
      PHASES,
    }}>
      {children}
    </PipelineContext.Provider>
  )
}

// ========================================
// Main App Component
// ========================================

function AppContent() {
  return (
    <div 
      className="min-h-screen flex"
      style={{ 
        backgroundColor: 'var(--bg-secondary)',
        color: 'var(--text-primary)'
      }}
    >
      {/* Left Navigation */}
      <Nav />
      
      {/* Main Content Area */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<ArenaPage />} />
          <Route path="/market" element={<MarketPage />} />
          <Route path="/pipeline" element={<PipelinePage />} />
          <Route path="/history" element={<PipelineHistoryPage />} />  {/* <-- ADD THIS */}
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/docs" element={<DocsPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <DataProvider>
        <PipelineProvider>
          <AppContent />
        </PipelineProvider>
      </DataProvider>
    </ThemeProvider>
  )
}