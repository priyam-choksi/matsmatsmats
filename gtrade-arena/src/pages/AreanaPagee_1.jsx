import React, { useState, useRef, useEffect } from 'react'
import { useTheme } from '../App'
import { STRATEGY_CONFIG, TICKER_COLORS } from '../data'

// ============================================================
// CONFIGURATION - Update this to match your tournament folder
// ============================================================
const TOURNAMENT_FOLDER = 'gt_tournament_20251203_012947'
const BASE_PATH = '/data/game_theory_results'

// ============================================================
// DATA LOADING FUNCTIONS
// ============================================================

async function loadTickerSummary(ticker) {
  const url = `${BASE_PATH}/${TOURNAMENT_FOLDER}/by_ticker/${ticker}/${ticker}_summary.json`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load ${ticker} summary`)
  return res.json()
}

async function loadTickerDetailed(ticker) {
  const url = `${BASE_PATH}/${TOURNAMENT_FOLDER}/by_ticker/${ticker}/${ticker}_detailed.json`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load ${ticker} detailed`)
  return res.json()
}

async function getAvailableTickers() {
  try {
    const url = `${BASE_PATH}/${TOURNAMENT_FOLDER}/combined/all_tickers_summary.json`
    const res = await fetch(url)
    if (res.ok) {
      const data = await res.json()
      return data.tickers || Object.keys(data.by_ticker || {}) || []
    }
  } catch (e) {
    console.warn('Could not load ticker list')
  }
  return ['AAPL', 'NVDA', 'GOOGL', 'META', 'AMZN', 'MSFT', 'TSLA', 'JPM', 'V', 'JNJ', 'LLY']
}

// Transform summary data to match existing UI format
function transformStrategies(summary) {
  if (!summary?.strategies) return null
  
  const strategyMap = {
    'Signal Follower': { id: 'signal', icon: '🤖', color: '#18181b', personality: 'LLM-Powered', desc: 'Uses 11-agent pipeline' },
    'Tit-for-Tat': { id: 'tft', icon: '🔄', color: '#f59e0b', personality: 'Reactive', desc: 'Mirrors previous winner' },
    'Cooperator': { id: 'cooperator', icon: '🤝', color: '#3b82f6', personality: 'Trust Builder', desc: 'Follows market consensus' },
    'Defector': { id: 'defector', icon: '🎯', color: '#ef4444', personality: 'Contrarian', desc: 'Bets against consensus' },
    'Buy-and-Hold': { id: 'benchmark', icon: '📊', color: '#8b5cf6', personality: 'Passive', desc: 'Always 100% invested' }
  }

  return Object.entries(summary.strategies).map(([name, data]) => {
    const cfg = strategyMap[name] || { id: name.toLowerCase(), icon: '📈', color: '#666', personality: 'Unknown', desc: '' }
    const initialCap = data.initial_capital || 250000
    const finalCap = data.final_capital || initialCap
    return {
      id: cfg.id,
      name,
      icon: cfg.icon,
      color: cfg.color,
      personality: cfg.personality,
      desc: cfg.desc,
      returnPct: data.total_return_pct || ((finalCap - initialCap) / initialCap * 100),
      acctValue: finalCap,
      totalPnL: finalCap - initialCap,
      winRate: data.win_rate_pct || 0,
      wins: data.wins || 0,
      sharpe: data.sharpe_ratio || 0,
      maxDD: data.max_drawdown_pct || 0,
      avgPosition: data.avg_position_pct || (name === 'Buy-and-Hold' ? 100 : 50),
      cash: finalCap * 0.1 // Estimate
    }
  }).sort((a, b) => b.returnPct - a.returnPct)
}

// Transform detailed data to chart format
function transformChartData(detailed) {
  if (!detailed?.round_history) return null
  
  const strategyIdMap = {
    'Signal Follower': 'signal',
    'Tit-for-Tat': 'tft',
    'Cooperator': 'cooperator',
    'Defector': 'defector',
    'Buy-and-Hold': 'benchmark'
  }

  // Sample at intervals for chart (every ~10-15 rounds)
  const interval = Math.max(1, Math.floor(detailed.round_history.length / 7))
  const sampled = detailed.round_history.filter((_, i) => i % interval === 0 || i === detailed.round_history.length - 1)
  
  return sampled.map(round => {
    const point = { round: round.round }
    if (round.capital) {
      Object.entries(round.capital).forEach(([name, cap]) => {
        const id = strategyIdMap[name] || name.toLowerCase()
        // Convert capital to return % from initial (assuming 250K each)
        point[id] = ((cap - 250000) / 250000) * 100
      })
    }
    if (round.returns) {
      Object.entries(round.returns).forEach(([name, ret]) => {
        const id = strategyIdMap[name] || name.toLowerCase()
        point[id] = ret
      })
    }
    return point
  })
}

// Transform to holdings format
function transformHoldings(detailed, summary) {
  if (!detailed?.round_history || !summary?.strategies) return null

  const strategyIdMap = {
    'Signal Follower': 'signal',
    'Tit-for-Tat': 'tft',
    'Cooperator': 'cooperator',
    'Defector': 'defector',
    'Buy-and-Hold': 'benchmark'
  }

  const holdings = {}
  const lastRounds = detailed.round_history.slice(-3)

  Object.entries(summary.strategies).forEach(([name, data]) => {
    const id = strategyIdMap[name] || name.toLowerCase()
    
    // Get regime performance if available
    const regimePerf = data.regime_performance || { bull: 0, bear: 0, sideways: 0 }
    
    // Build recent rounds from actual data
    const recentRounds = lastRounds.map(r => ({
      round: r.round,
      regime: r.regime || 'sideways',
      position: r.positions?.[name] || data.avg_position_pct || 50,
      return: r.returns?.[name] || 0,
      won: r.winner === name
    }))

    holdings[id] = {
      positions: [], // We don't have individual stock positions from tournament data
      regimePerformance: {
        bull: regimePerf.bull || regimePerf.Bull || 0,
        bear: regimePerf.bear || regimePerf.Bear || 0,
        sideways: regimePerf.sideways || regimePerf.Sideways || 0
      },
      recentRounds
    }
  })

  return holdings
}

// ========== DEFAULT/FALLBACK DATA ==========
const defaultStrategies = [
  { id: 'signal', name: 'Signal Follower', icon: '🤖', color: '#18181b', returnPct: 13.66, acctValue: 227320, totalPnL: 27320, winRate: 31.1, wins: 28, sharpe: 1.47, maxDD: -8.2, cash: 22910, personality: 'LLM-Powered', desc: 'Uses 11-agent pipeline' },
  { id: 'tft', name: 'Tit-for-Tat', icon: '🔄', color: '#f59e0b', returnPct: 11.21, acctValue: 222420, totalPnL: 22420, winRate: 24.4, wins: 22, sharpe: 1.32, maxDD: -9.1, cash: 33370, personality: 'Reactive', desc: 'Mirrors previous winner' },
  { id: 'cooperator', name: 'Cooperator', icon: '🤝', color: '#3b82f6', returnPct: 5.45, acctValue: 210900, totalPnL: 10900, winRate: 14.4, wins: 13, sharpe: 0.89, maxDD: -7.8, cash: 28450, personality: 'Trust Builder', desc: 'Follows market consensus' },
  { id: 'defector', name: 'Defector', icon: '🎯', color: '#ef4444', returnPct: -4.24, acctValue: 191520, totalPnL: -8480, winRate: 7.8, wins: 7, sharpe: 0.31, maxDD: -18.6, cash: 41200, personality: 'Contrarian', desc: 'Bets against consensus' },
]

const defaultHoldings = {
  signal: {
    positions: [
      { symbol: 'NVDA', shares: 45, avgCost: 142.50, current: 185.30, pnl: 1926, pnlPct: 30.0 },
      { symbol: 'GOOGL', shares: 120, avgCost: 138.20, current: 161.88, pnl: 2841, pnlPct: 17.1 },
      { symbol: 'META', shares: 35, avgCost: 485.00, current: 512.40, pnl: 959, pnlPct: 5.6 },
    ],
    regimePerformance: { bull: 18.2, bear: -2.1, sideways: 4.8 },
    recentRounds: [
      { round: 88, regime: 'bull', position: 65, return: 2.1, won: true },
      { round: 89, regime: 'bull', position: 70, return: 1.8, won: true },
      { round: 90, regime: 'sideways', position: 45, return: 0.4, won: false },
    ]
  },
  tft: {
    positions: [
      { symbol: 'MRVL', shares: 280, avgCost: 68.40, current: 81.00, pnl: 3528, pnlPct: 18.4 },
      { symbol: 'AMZN', shares: 95, avgCost: 178.50, current: 197.89, pnl: 1842, pnlPct: 10.9 },
    ],
    regimePerformance: { bull: 14.5, bear: -3.2, sideways: 5.1 },
    recentRounds: [
      { round: 88, regime: 'bull', position: 60, return: 1.9, won: false },
      { round: 89, regime: 'bull', position: 65, return: 1.6, won: false },
      { round: 90, regime: 'sideways', position: 50, return: 0.5, won: true },
    ]
  },
  cooperator: {
    positions: [
      { symbol: 'AAPL', shares: 180, avgCost: 188.50, current: 200.98, pnl: 2245, pnlPct: 6.6 },
      { symbol: 'JNJ', shares: 145, avgCost: 152.30, current: 158.90, pnl: 957, pnlPct: 4.3 },
    ],
    regimePerformance: { bull: 8.2, bear: -1.8, sideways: 3.2 },
    recentRounds: [
      { round: 88, regime: 'bull', position: 55, return: 1.1, won: false },
      { round: 89, regime: 'bull', position: 58, return: 0.9, won: false },
      { round: 90, regime: 'sideways', position: 52, return: 0.2, won: false },
    ]
  },
  defector: {
    positions: [
      { symbol: 'TSLA', shares: 125, avgCost: 285.60, current: 238.88, pnl: -5840, pnlPct: -16.4 },
    ],
    regimePerformance: { bull: -12.4, bear: 6.2, sideways: -2.1 },
    recentRounds: [
      { round: 88, regime: 'bull', position: 20, return: -0.8, won: false },
      { round: 89, regime: 'bull', position: 15, return: -0.6, won: false },
      { round: 90, regime: 'sideways', position: 30, return: -0.2, won: false },
    ]
  },
}

const defaultFeedData = [
  { id: 1, ticker: 'NVDA', timestamp: '2:32 PM', verdict: 'BUY', positionPct: 65, confidence: 'HIGH', reasoning: 'Strong technical breakout above $180 with volume confirmation. Bull thesis supported by AI demand.', keyFactors: ['Technical breakout', 'AI catalyst', 'Bullish consensus'] },
  { id: 2, ticker: 'GOOGL', timestamp: '11:15 AM', verdict: 'BUY', positionPct: 42, confidence: 'MEDIUM', reasoning: 'Fundamental value play with reasonable P/E. Cloud growth accelerating.', keyFactors: ['Cloud growth', 'Reasonable valuation'] },
  { id: 3, ticker: 'TSLA', timestamp: '9:45 AM', verdict: 'REJECT', positionPct: 0, confidence: 'HIGH', reasoning: 'High valuation risk with slowing delivery growth. Conservative evaluator raised red flags.', keyFactors: ['Valuation risk', 'Delivery slowdown'] },
  { id: 4, ticker: 'AAPL', timestamp: 'Yesterday', verdict: 'HOLD', positionPct: 28, confidence: 'MEDIUM', reasoning: 'Reduced position from initial recommendation. iPhone cycle concerns balanced by services growth.', keyFactors: ['Services growth', 'iPhone uncertainty'] },
]

const defaultChartData = [
  { round: 1, signal: 0.8, tft: 0.5, cooperator: 0.4, defector: 1.8 },
  { round: 15, signal: 4.1, tft: 3.5, cooperator: 2.2, defector: -0.5 },
  { round: 30, signal: 6.8, tft: 6.2, cooperator: 3.5, defector: -2.8 },
  { round: 45, signal: 8.5, tft: 7.8, cooperator: 4.0, defector: -3.5 },
  { round: 60, signal: 10.2, tft: 9.4, cooperator: 4.6, defector: -3.9 },
  { round: 75, signal: 12.1, tft: 10.5, cooperator: 5.1, defector: -4.0 },
  { round: 90, signal: 13.66, tft: 11.21, cooperator: 5.45, defector: -4.24 },
]

// ========== PERFORMANCE CHART ==========
const PerformanceChart = ({ isDark, chartData, strategies }) => {
  const data = chartData || defaultChartData
  const strats = strategies || defaultStrategies
  
  const width = 560
  const height = 220
  const padding = { top: 15, right: 80, bottom: 30, left: 45 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom
  
  // Calculate min/max from data
  let allVals = []
  data.forEach(d => {
    strats.forEach(s => {
      if (d[s.id] !== undefined) allVals.push(d[s.id])
    })
  })
  const minY = Math.min(-6, Math.floor(Math.min(...allVals) - 2))
  const maxY = Math.max(16, Math.ceil(Math.max(...allVals) + 2))

  const yScale = (val) => padding.top + chartHeight - ((val - minY) / (maxY - minY)) * chartHeight
  const xScale = (idx) => padding.left + (idx / (data.length - 1)) * chartWidth

  const createPath = (key) => {
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yScale(d[key] || 0)}`).join(' ')
  }

  const gridColor = isDark ? '#27272a' : '#e4e4e7'
  const textColor = isDark ? '#a1a1aa' : '#71717a'

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '100%' }}>
      {/* Grid */}
      {[minY, 0, Math.floor(maxY/2), maxY].map((v, i) => (
        <g key={i}>
          <line x1={padding.left} y1={yScale(v)} x2={width - padding.right} y2={yScale(v)} stroke={gridColor} strokeDasharray={v === 0 ? "0" : "4,4"} strokeWidth={v === 0 ? 1.5 : 1} />
          <text x={padding.left - 8} y={yScale(v) + 4} textAnchor="end" fontSize="10" fill={textColor}>{v}%</text>
        </g>
      ))}
      
      {/* X axis labels */}
      {data.map((d, i) => (
        <text key={i} x={xScale(i)} y={height - 10} textAnchor="middle" fontSize="10" fill={textColor}>R{d.round}</text>
      ))}

      {/* Lines */}
      {strats.map(s => (
        <path key={s.id} d={createPath(s.id)} fill="none" stroke={s.color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      ))}

      {/* Legend */}
      {strats.map((s, i) => (
        <g key={s.id} transform={`translate(${width - padding.right + 10}, ${padding.top + i * 18})`}>
          <line x1="0" y1="6" x2="16" y2="6" stroke={s.color} strokeWidth="2.5" />
          <text x="22" y="10" fontSize="10" fill={textColor}>{s.name.split(' ')[0]}</text>
        </g>
      ))}

      {/* End dots */}
      {strats.map(s => {
        const lastVal = data[data.length - 1][s.id] || 0
        return <circle key={s.id} cx={xScale(data.length - 1)} cy={yScale(lastVal)} r="4" fill={s.color} />
      })}
    </svg>
  )
}

// ========== MAIN COMPONENT ==========
export default function ArenaPage() {
  const { isDark } = useTheme()
  
  // Data state
  const [strategies, setStrategies] = useState(defaultStrategies)
  const [holdings, setHoldings] = useState(defaultHoldings)
  const [chartData, setChartData] = useState(defaultChartData)
  const [feedData, setFeedData] = useState(defaultFeedData)
  const [tickers, setTickers] = useState(['AAPL'])
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [loading, setLoading] = useState(true)
  const [dataLoaded, setDataLoaded] = useState(false)
  const [tournamentInfo, setTournamentInfo] = useState({ rounds: 90, winner: null })
  
  // UI state (keep your original)
  const [expandedStrategy, setExpandedStrategy] = useState('signal')
  const [expandedFeed, setExpandedFeed] = useState(null)
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', content: "👋 Hi! Ask me about strategies, game theory, or who's winning!" }
  ])
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)

  // Load available tickers on mount
  useEffect(() => {
    getAvailableTickers().then(t => {
      if (t.length > 0) {
        setTickers(t)
        setSelectedTicker(t[0])
      }
    })
  }, [])

  // Load data when ticker changes
  useEffect(() => {
    if (!selectedTicker) return
    
    setLoading(true)
    Promise.all([
      loadTickerSummary(selectedTicker),
      loadTickerDetailed(selectedTicker)
    ])
      .then(([summary, detailed]) => {
        // Transform and set data
        const newStrategies = transformStrategies(summary)
        const newChartData = transformChartData(detailed)
        const newHoldings = transformHoldings(detailed, summary)
        
        if (newStrategies) setStrategies(newStrategies)
        if (newChartData) setChartData(newChartData)
        if (newHoldings) setHoldings({ ...defaultHoldings, ...newHoldings })
        
        setTournamentInfo({
          rounds: summary.total_rounds || 90,
          winner: summary.winner || null
        })
        setDataLoaded(true)
      })
      .catch(err => {
        console.warn('Using default data:', err.message)
        setDataLoaded(false)
      })
      .finally(() => setLoading(false))
  }, [selectedTicker])

  // Chat scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Chat handler (keep your original logic)
  const handleSend = () => {
    if (!input.trim()) return
    const q = input.toLowerCase()
    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: input }])
    setInput('')

    let response = "Try asking about specific strategies, game theory mechanics, or who's winning!"
    
    if (q.includes('signal')) {
      const s = strategies.find(x => x.id === 'signal')
      response = `🤖 **Signal Follower** is at ${s?.returnPct?.toFixed(2) || 13.66}% using the 11-agent LLM pipeline.\n\n• Won ${s?.wins || 28}/${tournamentInfo.rounds} rounds\n• Sharpe ratio: ${s?.sharpe?.toFixed(2) || 1.47}\n• Best in bull markets`
    } else if (q.includes('defector')) {
      const s = strategies.find(x => x.id === 'defector')
      response = `🎯 **Defector** is at ${s?.returnPct?.toFixed(2) || -4.24}%.\n\n• Won ${s?.wins || 7}/${tournamentInfo.rounds} rounds\n• Sharpe ratio: ${s?.sharpe?.toFixed(2) || 0.31}\n• Contrarian approach\n• Best in bear markets`
    } else if (q.includes('cooperator')) {
      const s = strategies.find(x => x.id === 'cooperator')
      response = `🤝 **Cooperator** is at ${s?.returnPct?.toFixed(2) || 5.45}% by following consensus.\n\n• Won ${s?.wins || 13}/${tournamentInfo.rounds} rounds\n• Sharpe ratio: ${s?.sharpe?.toFixed(2) || 0.89}\n• Works well in sideways markets`
    } else if (q.includes('tit') || q.includes('tat')) {
      const s = strategies.find(x => x.id === 'tft')
      response = `🔄 **Tit-for-Tat** is at ${s?.returnPct?.toFixed(2) || 11.21}% by mirroring winners.\n\n• Won ${s?.wins || 22}/${tournamentInfo.rounds} rounds\n• Sharpe ratio: ${s?.sharpe?.toFixed(2) || 1.32}\n• Classic game theory strategy`
    } else if (q.includes('win') || q.includes('lead') || q.includes('best')) {
      const sorted = [...strategies].sort((a, b) => b.returnPct - a.returnPct)
      response = `🏆 **Current Standings (${selectedTicker}):**\n\n${sorted.map((s, i) => `${i + 1}. ${s.icon} ${s.name}: ${s.returnPct >= 0 ? '+' : ''}${s.returnPct.toFixed(2)}%`).join('\n')}`
    } else if (q.includes('game theory') || (q.includes('how') && q.includes('work'))) {
      response = `🎮 **Game Theory Tournament:**\n\n• ${strategies.length} strategies share $1M capital\n• ${tournamentInfo.rounds} rounds of competition\n• After each round, capital reallocates\n• Winners take from losers via z-score\n• Creates true strategic interdependence`
    }

    setTimeout(() => {
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: response }])
    }, 300)
  }

  const bgTertiary = isDark ? '#27272a' : '#f4f4f5'
  const borderColor = isDark ? '#3f3f46' : '#e4e4e7'
  const h = holdings[expandedStrategy] || holdings.signal || defaultHoldings.signal
  const selectedStrat = strategies.find(s => s.id === expandedStrategy) || strategies[0]

  return (
    <div style={{ minHeight: '100vh', background: isDark ? '#09090b' : '#fafafa' }}>
      {/* Header */}
      <header style={{ background: isDark ? '#18181b' : 'white', borderBottom: `1px solid ${borderColor}`, padding: '12px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '36px', height: '36px', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: '700', fontSize: '14px' }}>GT</div>
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: isDark ? '#fafafa' : '#18181b' }}>Game Theory Arena</h1>
            <p style={{ margin: 0, fontSize: '11px', color: isDark ? '#a1a1aa' : '#71717a' }}>Capital Allocation Tournament</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Ticker Selector */}
          <select
            value={selectedTicker}
            onChange={e => setSelectedTicker(e.target.value)}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: `1px solid ${borderColor}`,
              background: bgTertiary,
              color: isDark ? '#fafafa' : '#18181b',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            {tickers.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <span style={{ padding: '5px 10px', background: bgTertiary, borderRadius: '6px', fontSize: '11px', color: isDark ? '#a1a1aa' : '#52525b' }}>{tournamentInfo.rounds} Rounds</span>
          <span style={{ padding: '5px 10px', background: bgTertiary, borderRadius: '6px', fontSize: '11px', color: isDark ? '#a1a1aa' : '#52525b' }}>{tickers.length} Tickers</span>
          <span style={{ padding: '5px 10px', background: bgTertiary, borderRadius: '6px', fontSize: '11px', color: isDark ? '#a1a1aa' : '#52525b' }}>$1M Pool</span>
          <span style={{ 
            padding: '5px 10px', 
            background: dataLoaded ? '#dcfce7' : '#fef3c7', 
            borderRadius: '6px', 
            fontSize: '10px', 
            fontWeight: '600',
            color: dataLoaded ? '#166534' : '#92400e' 
          }}>
            {loading ? '⏳ Loading...' : dataLoaded ? '✓ Live Data' : '⚠ Demo Data'}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <div style={{ display: 'flex', height: 'calc(100vh - 60px)' }}>
        {/* Left: Strategies */}
        <div style={{ width: '320px', borderRight: `1px solid ${borderColor}`, background: isDark ? '#18181b' : 'white', overflowY: 'auto', padding: '12px' }}>
          <div style={{ fontSize: '11px', fontWeight: '600', color: isDark ? '#71717a' : '#a1a1aa', marginBottom: '10px', textTransform: 'uppercase' }}>Strategies</div>
          {strategies.map((s, idx) => {
            const isExpanded = expandedStrategy === s.id
            const isUp = s.returnPct >= 0
            return (
              <div key={s.id} onClick={() => setExpandedStrategy(s.id)} style={{
                background: isExpanded ? (isDark ? '#27272a' : '#f4f4f5') : 'transparent',
                borderRadius: '10px',
                padding: '12px',
                marginBottom: '6px',
                cursor: 'pointer',
                border: isExpanded ? `1px solid ${s.color}40` : '1px solid transparent'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '22px' }}>{s.icon}</span>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontWeight: '600', fontSize: '13px', color: isDark ? '#fafafa' : '#18181b' }}>{s.name}</span>
                        {idx === 0 && <span style={{ fontSize: '12px' }}>👑</span>}
                      </div>
                      <div style={{ fontSize: '10px', color: isDark ? '#71717a' : '#a1a1aa' }}>{s.personality}</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '15px', fontWeight: '700', color: isUp ? '#10b981' : '#ef4444' }}>
                      {isUp ? '+' : ''}{s.returnPct.toFixed(2)}%
                    </div>
                    <div style={{ fontSize: '10px', color: isDark ? '#71717a' : '#a1a1aa' }}>{s.wins} wins</div>
                  </div>
                </div>

                {isExpanded && (
                  <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: `1px solid ${borderColor}` }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '12px' }}>
                      {[
                        { label: 'Account', value: `$${(s.acctValue/1000).toFixed(0)}K` },
                        { label: 'Win Rate', value: `${s.winRate.toFixed(1)}%` },
                        { label: 'Sharpe', value: s.sharpe.toFixed(2) },
                        { label: 'Max DD', value: `${s.maxDD.toFixed(1)}%`, isNeg: true },
                        { label: 'Avg Pos', value: `${s.avgPosition?.toFixed(0) || 50}%` },
                        { label: 'Cash', value: `$${(s.cash/1000).toFixed(0)}K` },
                      ].map((m, i) => (
                        <div key={i} style={{ textAlign: 'center', padding: '6px', background: isDark ? '#1a1a1a' : '#fafafa', borderRadius: '6px' }}>
                          <div style={{ fontSize: '9px', color: isDark ? '#71717a' : '#a1a1aa' }}>{m.label}</div>
                          <div style={{ fontSize: '11px', fontWeight: '600', color: m.isNeg ? '#ef4444' : (isDark ? '#fafafa' : '#18181b') }}>{m.value}</div>
                        </div>
                      ))}
                    </div>

                    {/* Regime Performance */}
                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '10px', color: isDark ? '#71717a' : '#a1a1aa', marginBottom: '6px' }}>Regime Performance</div>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        {Object.entries(h?.regimePerformance || {}).map(([regime, ret]) => (
                          <div key={regime} style={{ flex: 1, textAlign: 'center', padding: '6px', background: isDark ? '#1a1a1a' : '#fafafa', borderRadius: '6px' }}>
                            <div style={{ fontSize: '9px', color: regime === 'bull' ? '#10b981' : regime === 'bear' ? '#ef4444' : '#f59e0b' }}>
                              {regime.charAt(0).toUpperCase() + regime.slice(1)}
                            </div>
                            <div style={{ fontSize: '11px', fontWeight: '600', color: ret >= 0 ? '#10b981' : '#ef4444' }}>
                              {ret >= 0 ? '+' : ''}{ret}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Recent Rounds */}
                    <div>
                      <div style={{ fontSize: '10px', color: isDark ? '#71717a' : '#a1a1aa', marginBottom: '6px' }}>Recent Rounds</div>
                      {(h?.recentRounds || []).map(r => (
                        <div key={r.round} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 8px', background: isDark ? '#1a1a1a' : '#fafafa', borderRadius: '6px', marginBottom: '4px', fontSize: '11px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontWeight: '600', color: isDark ? '#fafafa' : '#18181b' }}>R{r.round}</span>
                            <span style={{ padding: '2px 6px', borderRadius: '4px', fontSize: '9px', background: r.regime === 'bull' ? '#dcfce7' : r.regime === 'bear' ? '#fee2e2' : '#fef3c7', color: r.regime === 'bull' ? '#166534' : r.regime === 'bear' ? '#991b1b' : '#92400e' }}>
                              {r.regime?.toUpperCase()}
                            </span>
                            <span style={{ color: isDark ? '#71717a' : '#a1a1aa' }}>{r.position}% pos</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ fontWeight: '500', color: r.return >= 0 ? '#10b981' : '#ef4444' }}>{r.return >= 0 ? '+' : ''}{r.return}%</span>
                            {r.won && <span style={{ fontSize: '10px' }}>🏆</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Center: Chart + Feed */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Chart */}
          <div style={{ padding: '16px 20px', borderBottom: `1px solid ${borderColor}`, background: isDark ? '#18181b' : 'white' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <h2 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: isDark ? '#fafafa' : '#18181b' }}>Tournament Performance</h2>
                <p style={{ margin: 0, fontSize: '11px', color: isDark ? '#71717a' : '#a1a1aa' }}>{selectedTicker} • Return % over {tournamentInfo.rounds} rounds</p>
              </div>
              {tournamentInfo.winner && (
                <div style={{ padding: '6px 12px', background: '#dcfce7', borderRadius: '8px', fontSize: '12px', fontWeight: '600', color: '#166534' }}>
                  🏆 Winner: {tournamentInfo.winner}
                </div>
              )}
            </div>
            <div style={{ height: '220px' }}>
              <PerformanceChart isDark={isDark} chartData={chartData} strategies={strategies} />
            </div>
          </div>

          {/* Feed */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', background: isDark ? '#09090b' : '#fafafa' }}>
            <h3 style={{ margin: '0 0 12px', fontSize: '13px', fontWeight: '600', color: isDark ? '#fafafa' : '#18181b' }}>Recent Pipeline Decisions</h3>
            {feedData.map(item => {
              const isExpanded = expandedFeed === item.id
              const verdictColors = { BUY: '#10b981', HOLD: '#f59e0b', REJECT: '#ef4444', SELL: '#ef4444' }
              return (
                <div key={item.id} onClick={() => setExpandedFeed(isExpanded ? null : item.id)} style={{
                  background: isDark ? '#18181b' : 'white',
                  borderRadius: '10px',
                  padding: '12px',
                  marginBottom: '8px',
                  cursor: 'pointer',
                  border: `1px solid ${borderColor}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '16px', fontWeight: '700', color: TICKER_COLORS[item.ticker] || (isDark ? '#fafafa' : '#18181b') }}>{item.ticker}</span>
                      <span style={{ padding: '3px 8px', borderRadius: '6px', fontSize: '10px', fontWeight: '600', background: `${verdictColors[item.verdict]}20`, color: verdictColors[item.verdict] }}>{item.verdict}</span>
                      <span style={{ fontSize: '11px', color: isDark ? '#71717a' : '#a1a1aa' }}>{item.timestamp}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '12px', fontWeight: '600', color: isDark ? '#fafafa' : '#18181b' }}>{item.positionPct}%</span>
                      <span style={{ padding: '2px 6px', borderRadius: '4px', fontSize: '9px', background: item.confidence === 'HIGH' ? '#dcfce7' : '#fef3c7', color: item.confidence === 'HIGH' ? '#166534' : '#92400e' }}>{item.confidence}</span>
                    </div>
                  </div>
                  {isExpanded && (
                    <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: `1px solid ${borderColor}` }}>
                      <p style={{ margin: '0 0 8px', fontSize: '12px', color: isDark ? '#a1a1aa' : '#52525b', lineHeight: 1.5 }}>{item.reasoning}</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {item.keyFactors.map((f, i) => (
                          <span key={i} style={{ padding: '3px 8px', background: bgTertiary, borderRadius: '4px', fontSize: '10px', color: isDark ? '#a1a1aa' : '#52525b' }}>{f}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Right: Chat */}
        <div style={{ width: '300px', borderLeft: `1px solid ${borderColor}`, background: isDark ? '#18181b' : 'white', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '12px', borderBottom: `1px solid ${borderColor}` }}>
            <div style={{ fontSize: '13px', fontWeight: '600', color: isDark ? '#fafafa' : '#18181b' }}>💬 Research Assistant</div>
            <div style={{ fontSize: '10px', color: isDark ? '#71717a' : '#a1a1aa' }}>Ask about strategies & results</div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {messages.map(msg => (
              <div key={msg.id} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <div style={{
                  maxWidth: '85%',
                  padding: '10px 12px',
                  borderRadius: msg.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                  fontSize: '12px',
                  lineHeight: 1.5,
                  whiteSpace: 'pre-wrap',
                  background: msg.role === 'user' ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' : bgTertiary,
                  color: msg.role === 'user' ? 'white' : (isDark ? '#fafafa' : '#18181b')
                }}>
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div style={{ padding: '12px', borderTop: `1px solid ${borderColor}` }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                placeholder="Ask about strategies..."
                style={{
                  flex: 1,
                  padding: '10px 12px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  fontSize: '12px',
                  background: bgTertiary,
                  color: isDark ? '#fafafa' : '#18181b'
                }}
              />
              <button onClick={handleSend} style={{
                padding: '10px 16px',
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer'
              }}>Send</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}