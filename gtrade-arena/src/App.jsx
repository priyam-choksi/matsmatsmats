import React, { useState, useEffect, createContext, useContext, useRef, useCallback } from 'react'
import { Routes, Route } from 'react-router-dom'

// Pages
import ArenaPage from './pages/ArenaPage'
import MarketPage from './pages/MarketPage'
import PipelinePage from './pages/PipelinePage'
import LiveArenaPage from './pages/LiveArenaPage'
import PipelineHistoryPage from './pages/PipelineHistoryPage'
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

// Add this right after the TICKERS array in PipelinePage.jsx

const TICKER_DATA = {
  NVDA: { name: 'NVIDIA', sector: 'Semiconductors', price: 142.50, pe: 65.2, growth: 125.8, catalyst: 'AI chip demand surge', risk: 'Valuation concerns at 65x PE' },
  AAPL: { name: 'Apple', sector: 'Consumer Tech', price: 178.20, pe: 28.5, growth: 8.2, catalyst: 'iPhone 16 cycle + Vision Pro', risk: 'China market weakness' },
  MSFT: { name: 'Microsoft', sector: 'Enterprise Tech', price: 378.90, pe: 35.1, growth: 18.4, catalyst: 'Azure AI integration', risk: 'Enterprise spending slowdown' },
  GOOGL: { name: 'Alphabet', sector: 'Internet/AI', price: 141.80, pe: 24.2, growth: 15.1, catalyst: 'Gemini AI monetization', risk: 'Search market share erosion' },
  META: { name: 'Meta', sector: 'Social/AI', price: 505.75, pe: 27.8, growth: 22.3, catalyst: 'Reels monetization + AI ads', risk: 'Reality Labs losses' },
  AMZN: { name: 'Amazon', sector: 'E-Commerce/Cloud', price: 178.25, pe: 42.1, growth: 12.5, catalyst: 'AWS AI services growth', risk: 'Retail margin pressure' },
  TSLA: { name: 'Tesla', sector: 'EV/Energy', price: 248.50, pe: 78.3, growth: -8.2, catalyst: 'FSD licensing + Robotaxi', risk: 'EV price war, margin compression' },
  JPM: { name: 'JPMorgan', sector: 'Banking', price: 198.40, pe: 11.2, growth: 6.8, catalyst: 'Higher-for-longer rates', risk: 'Commercial real estate exposure' },
  V: { name: 'Visa', sector: 'Payments', price: 279.30, pe: 29.8, growth: 10.5, catalyst: 'Cross-border travel recovery', risk: 'Regulatory scrutiny on fees' },
  JNJ: { name: 'J&J', sector: 'Healthcare', price: 156.80, pe: 15.4, growth: 4.2, catalyst: 'Pharma pipeline', risk: 'Talc litigation overhang' },
  LLY: { name: 'Eli Lilly', sector: 'Pharma', price: 752.40, pe: 118.5, growth: 58.3, catalyst: 'GLP-1 obesity drugs', risk: 'Extreme valuation' },
}


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

  const generateAgentOutput = (agentName, ticker, mode = 'deep', prevOutputs = {}) => {
    const data = TICKER_DATA[ticker] || TICKER_DATA.NVDA
    const isDeep = mode === 'deep' || mode === 'research'
    const isFull = mode === 'research'
    const bullish = data.growth > 15
    const rsi = 30 + Math.random() * 40

    const outputs = {
      'Technical Analyst': {
        signal: rsi > 55 ? 'BULLISH' : rsi < 45 ? 'BEARISH' : 'NEUTRAL',
        rsi: rsi.toFixed(1),
        rsi_interpretation: rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral zone',
        macd: bullish ? 'Bullish Crossover' : 'Bearish Divergence',
        trend: data.growth > 10 ? 'Strong Uptrend' : data.growth > 0 ? 'Consolidation' : 'Downtrend',
        support: `$${(data.price * 0.92).toFixed(2)}`,
        resistance: `$${(data.price * 1.08).toFixed(2)}`,
        ...(isDeep && { volume: 'Above 20-day average', sma_50: `$${(data.price * 0.95).toFixed(2)}`, sma_200: `$${(data.price * 0.88).toFixed(2)}` }),
        ...(isFull && { fibonacci_618: `$${(data.price * 0.91).toFixed(2)}`, options_flow: 'Heavy call buying detected' })
      },

      'News Analyst': {
        sentiment: bullish ? 'Positive' : data.growth < 0 ? 'Negative' : 'Mixed',
        sentiment_score: (bullish ? 0.6 + Math.random() * 0.3 : -0.2 + Math.random() * 0.4).toFixed(2),
        articles_analyzed: isDeep ? 45 : 15,
        key_catalyst: data.catalyst,
        key_risk: data.risk,
        ...(isDeep && { sources: 'Reuters, Bloomberg, WSJ, CNBC', social_sentiment: bullish ? 'Positive' : 'Mixed' }),
        ...(isFull && { insider_activity: 'No significant recent trades', institutional_moves: 'Net accumulation last quarter' })
      },

      'Fundamental Analyst': {
        pe_ratio: data.pe.toFixed(1),
        pe_vs_sector: data.pe > 35 ? 'Premium' : data.pe < 20 ? 'Discount' : 'In-line',
        revenue_growth: `${data.growth.toFixed(1)}%`,
        valuation: data.pe > 50 ? 'Overvalued' : data.pe < 20 ? 'Undervalued' : 'Fair Value',
        ...(isDeep && { peg_ratio: (data.pe / Math.max(data.growth, 1)).toFixed(2), gross_margin: `${(45 + Math.random() * 15).toFixed(1)}%`, fcf_yield: `${(1 + Math.random() * 3).toFixed(1)}%` }),
        ...(isFull && { dcf_fair_value: `$${(data.price * (0.9 + Math.random() * 0.3)).toFixed(2)}`, moat: data.growth > 20 ? 'Strong' : 'Moderate' })
      },

      'Macro Analyst': {
        regime: data.growth > 10 ? 'Risk-On / Bull Market' : 'Defensive',
        sector_outlook: bullish ? 'Overweight' : 'Neutral',
        fed_impact: 'Higher for longer rates',
        recommendation: bullish ? 'Overweight' : data.growth < 0 ? 'Underweight' : 'Neutral',
        ...(isDeep && { gdp_outlook: 'Soft landing expected', inflation: 'Trending toward 2.5%' }),
        ...(isFull && { scenario_bull: 'AI productivity boom', scenario_bear: 'Recession + multiple compression' })
      },

      'Bull Researcher': {
        thesis: `${data.name} presents compelling upside driven by ${data.catalyst}. Growth of ${data.growth.toFixed(1)}% supports premium valuation.`,
        upside_target: `+${(15 + Math.random() * 20).toFixed(0)}%`,
        price_target: `$${(data.price * 1.2).toFixed(2)}`,
        catalysts: [data.catalyst, 'Earnings beat potential', 'Market share gains'],
        confidence: data.growth > 20 ? 'HIGH' : 'MEDIUM',
        ...(isDeep && { 
          counter_to_bears: `While bears cite ${data.risk.toLowerCase()}, we believe growth runway is underappreciated.`,
          competitive_edge: 'Market leadership and innovation pipeline'
        }),
        ...(isFull && { 
          historical_analog: 'Similar setup to AMZN 2015 - growth investment phase',
          institutional_support: 'Top funds increasing positions'
        })
      },

      'Bear Researcher': {
        thesis: `${data.name} faces headwinds from ${data.risk}. At ${data.pe.toFixed(1)}x PE, risk/reward skews negative.`,
        downside_target: `-${(12 + Math.random() * 15).toFixed(0)}%`,
        price_target: `$${(data.price * 0.82).toFixed(2)}`,
        risks: [data.risk, 'Valuation compression risk', 'Competitive pressure'],
        confidence: data.pe > 50 ? 'HIGH' : 'MEDIUM',
        ...(isDeep && { 
          counter_to_bulls: `Bulls cite ${data.catalyst.toLowerCase()}, but this is priced in at current multiples.`,
          structural_concerns: 'TAM may be overstated'
        }),
        ...(isFull && { 
          historical_warning: data.pe > 60 ? 'Echoes of dot-com valuations' : 'Cyclical risk elevated',
          short_interest: `${(3 + Math.random() * 8).toFixed(1)}% of float`
        })
      },

      'Research Manager': {
        winner: bullish && data.pe < 60 ? 'BULL' : 'BEAR',
        probability_bull: `${(bullish ? (55 + Math.random() * 20) : (30 + Math.random() * 15)).toFixed(0)}%`,
        probability_bear: `${(bullish ? (20 + Math.random() * 15) : (40 + Math.random() * 20)).toFixed(0)}%`,
        synthesis: bullish && data.pe < 60
          ? `Bull thesis prevails. ${data.catalyst} and ${data.growth.toFixed(1)}% growth outweigh valuation concerns.`
          : `Bear thesis prevails. ${data.risk} at ${data.pe.toFixed(1)}x PE creates unfavorable risk/reward.`,
        ...(isDeep && { 
          debate_rounds: isFull ? 3 : 2,
          key_contention: data.pe > 40 ? 'Valuation sustainability' : 'Growth durability',
          dissent: bullish ? 'Bears raise valid concerns on valuation' : 'Bulls note potential catalysts ahead'
        })
      },

      'Aggressive Evaluator': {
        stance: data.growth > 15 ? 'STRONG BUY' : data.growth > 5 ? 'BUY' : 'HOLD',
        position_pct: `${(45 + Math.random() * 25).toFixed(0)}%`,
        reasoning: `High conviction on ${data.catalyst}. Growth justifies aggressive positioning.`,
        ...(isDeep && { risk_tolerance: 'Willing to accept 25%+ drawdown for asymmetric upside' })
      },

      'Neutral Evaluator': {
        stance: data.growth > 10 ? 'BUY' : data.growth > 0 ? 'HOLD' : 'REDUCE',
        position_pct: `${(25 + Math.random() * 20).toFixed(0)}%`,
        reasoning: `Balanced view: ${data.catalyst} compelling but ${data.risk.toLowerCase()} warrants measured approach.`,
        ...(isDeep && { risk_tolerance: 'Targeting 12-15% annualized with controlled drawdowns' })
      },

      'Conservative Evaluator': {
        stance: data.pe > 45 ? 'AVOID' : data.growth < 5 ? 'AVOID' : 'HOLD',
        position_pct: `${(8 + Math.random() * 12).toFixed(0)}%`,
        reasoning: `Capital preservation focus. ${data.pe > 35 ? 'Valuation offers insufficient margin of safety.' : 'Prefer to wait for better entry.'}`,
        ...(isDeep && { risk_tolerance: 'Prioritizing downside protection' })
      },

      'Risk Manager': {
        verdict: data.growth > 15 && data.pe < 55 ? 'BUY' : data.pe > 70 || data.growth < 0 ? 'REJECT' : 'HOLD',
        position_dollars: data.growth > 15 && data.pe < 55 
          ? `$${(12000 + Math.floor(Math.random() * 6000)).toLocaleString()}`
          : data.pe > 70 || data.growth < 0 ? '$0' : `$${(5000 + Math.floor(Math.random() * 4000)).toLocaleString()}`,
        confidence: data.growth > 20 || data.pe > 70 ? 'HIGH' : 'MEDIUM',
        stop_loss: '-8%',
        target: '+15%',
        reasoning: data.growth > 15 && data.pe < 55
          ? `Evaluator consensus supports entry. ${data.catalyst} provides catalyst with ${data.growth.toFixed(1)}% growth.`
          : data.pe > 70 || data.growth < 0
            ? `Risk/reward unfavorable. ${data.risk} at ${data.pe.toFixed(1)}x PE. Preserving capital.`
            : `Mixed signals warrant caution. Small position to participate while limiting risk.`,
        ...(isDeep && { 
          evaluator_consensus: {
            aggressive: data.growth > 15 ? 'STRONG BUY' : 'BUY',
            neutral: data.growth > 10 ? 'BUY' : 'HOLD', 
            conservative: data.pe > 45 ? 'AVOID' : 'HOLD'
          }
        }),
        ...(isFull && {
          scenario_outcomes: {
            bull: `+${(18 + Math.random() * 12).toFixed(0)}%`,
            base: `+${(5 + Math.random() * 8).toFixed(0)}%`,
            bear: `-${(10 + Math.random() * 8).toFixed(0)}%`
          },
          next_review: 'Post earnings or on 10% price move'
        })
      },
    }

    return outputs[agentName] || { status: 'Complete' }
  }
  
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
        const output = generateAgentOutput(agentName, ticker, researchMode)
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
      // State
      ticker, setTicker,
      researchMode, setResearchMode,
      isRunning, setIsRunning,
      currentPhase, setCurrentPhase,
      currentAgent, setCurrentAgent,
      progress, setProgress,
      logs,
      agentStatuses, setAgentStatuses,
      agentOutputs, setAgentOutputs,
      result, setResult,
      selectedPhase, setSelectedPhase,
      // Actions
      addLog,
      runPipeline,
      resetPipeline,
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
          <Route path="/llm-arena" element={<LiveArenaPage />} />
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