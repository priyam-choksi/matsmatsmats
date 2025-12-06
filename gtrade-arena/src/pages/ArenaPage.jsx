import React, { useState, useRef, useEffect, useMemo } from 'react'
import { useTheme } from '../App'

// ========== CONFIGURATION ==========
const TOURNAMENT_FOLDER = 'gt_tournament_20251203_012947'
const BASE_PATH = '/data/game_theory_results'
const AVAILABLE_TICKERS = ['AAPL', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'JPM', 'MSFT']

const STRATEGY_CONFIG = {
  'Signal Follower': { id: 'signal', color: '#8b5cf6', icon: '🤖', short: 'Signal' },
  'Cooperator': { id: 'cooperator', color: '#22c55e', icon: '🤝', short: 'Coop' },
  'Defector': { id: 'defector', color: '#ef4444', icon: '🎯', short: 'Defect' },
  'Tit-for-Tat': { id: 'tft', color: '#3b82f6', icon: '🔄', short: 'TFT' },
  'Buy-and-Hold': { id: 'benchmark', color: '#64748b', icon: '📊', short: 'B&H' }
}

const STRATEGY_ORDER = ['Signal Follower', 'Cooperator', 'Defector', 'Tit-for-Tat']

// ========== DATA LOADING ==========
async function loadTickerData(ticker) {
  try {
    const [summaryRes, detailedRes] = await Promise.all([
      fetch(`${BASE_PATH}/${TOURNAMENT_FOLDER}/by_ticker/${ticker}/${ticker}_summary.json`),
      fetch(`${BASE_PATH}/${TOURNAMENT_FOLDER}/by_ticker/${ticker}/${ticker}_detailed.json`)
    ])
    if (!summaryRes.ok || !detailedRes.ok) throw new Error('Failed to load')
    const summary = await summaryRes.json()
    const detailed = await detailedRes.json()
    if (detailed.rounds) detailed.rounds.sort((a, b) => a.round_num - b.round_num)
    return { summary, detailed }
  } catch (err) {
    console.error('Load error:', err)
    return null
  }
}

// ========== STRATEGY CARDS ==========
const StrategyCards = ({ summary, isDark }) => {
  const returns = summary?.total_returns_pct || {}
  const allocs = summary?.final_allocations || {}
  const wins = summary?.wins_per_strategy || {}
  
  const sorted = STRATEGY_ORDER.map(name => ({
    name, ...STRATEGY_CONFIG[name],
    return: returns[name] || 0,
    capital: allocs[name] || 250000,
    wins: wins[name] || 0
  })).sort((a, b) => b.return - a.return)

  const border = isDark ? '#27272a' : '#e4e4e7'

  return (
    <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', overflowX: 'auto', paddingBottom: '4px' }}>
      {sorted.map((s, i) => (
        <div key={s.name} style={{
          display: 'flex', alignItems: 'center', gap: '12px',
          padding: '14px 18px', background: isDark ? '#18181b' : '#fff',
          borderRadius: '12px', border: `1px solid ${border}`,
          minWidth: '200px', flex: '1'
        }}>
          <span style={{ fontSize: '28px' }}>{s.icon}</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '2px' }}>
              {s.name}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              ${(s.capital / 1000).toFixed(0)}K • {s.wins} wins
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              fontSize: '18px', fontWeight: '700',
              color: s.return >= 0 ? '#22c55e' : '#ef4444'
            }}>
              {s.return >= 0 ? '+' : ''}{s.return.toFixed(1)}%
            </div>
            {i === 0 && (
              <span style={{
                fontSize: '9px', padding: '2px 6px', borderRadius: '4px',
                background: '#fef3c7', color: '#b45309', fontWeight: '600'
              }}>LEADER</span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

// ========== CAPITAL CHART ==========
const CapitalChart = ({ rounds, isDark, summary }) => {
  const width = 1000, height = 400
  const padding = { top: 50, right: 160, bottom: 60, left: 80 }
  const chartW = width - padding.left - padding.right
  const chartH = height - padding.top - padding.bottom

  const capitalHistory = useMemo(() => {
    if (!rounds.length) return []
    const history = [{
      round: 0, regime: null,
      'Signal Follower': 250000, 'Cooperator': 250000, 'Defector': 250000, 'Tit-for-Tat': 250000, 'Buy-and-Hold': 1000000
    }]
    let benchmarkCapital = 1000000
    rounds.forEach(r => {
      const allocs = r.round_results?.allocations_after || {}
      benchmarkCapital *= (1 + (r.market?.daily_return || 0))
      history.push({
        round: r.round_num, regime: r.regime,
        'Signal Follower': allocs['Signal Follower'] || 250000,
        'Cooperator': allocs['Cooperator'] || 250000,
        'Defector': allocs['Defector'] || 250000,
        'Tit-for-Tat': allocs['Tit-for-Tat'] || 250000,
        'Buy-and-Hold': benchmarkCapital
      })
    })
    return history
  }, [rounds])

  if (!capitalHistory.length) return null

  const maxRound = rounds.length
  const allVals = capitalHistory.flatMap(d => Object.entries(d).filter(([k]) => !['round', 'regime'].includes(k)).map(([, v]) => v))
  const maxY = Math.ceil(Math.max(...allVals) / 200000) * 200000
  const minY = 0

  const xScale = (round) => padding.left + (round / maxRound) * chartW
  const yScale = (val) => padding.top + chartH - ((val - minY) / (maxY - minY)) * chartH

  const createPath = (key) => capitalHistory.map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.round)},${yScale(d[key])}`).join(' ')

  // Regime bands
  const regimeBands = useMemo(() => {
    const bands = []
    let curr = null, start = 0
    capitalHistory.forEach(d => {
      if (d.regime && d.regime !== curr) {
        if (curr) bands.push({ regime: curr, start, end: d.round })
        curr = d.regime; start = d.round
      }
    })
    if (curr) bands.push({ regime: curr, start, end: maxRound })
    return bands
  }, [capitalHistory, maxRound])

  const regimeColors = { bull: 'rgba(34,197,94,0.08)', bear: 'rgba(239,68,68,0.08)', sideways: 'rgba(250,204,21,0.06)' }
  const gridColor = isDark ? '#27272a' : '#e4e4e7'
  const textColor = isDark ? '#71717a' : '#a1a1aa'
  const finalPoint = capitalHistory[capitalHistory.length - 1]

  // Calculate returns for legend
  const getReturn = (name) => {
    const initial = name === 'Buy-and-Hold' ? 1000000 : 250000
    return ((finalPoint[name] / initial - 1) * 100)
  }

  return (
    <div style={{ background: isDark ? '#09090b' : '#fff', borderRadius: '16px', padding: '20px', border: `1px solid ${isDark ? '#27272a' : '#e4e4e7'}` }}>
      <h3 style={{ margin: '0 0 16px', fontSize: '16px', fontWeight: '600', color: 'var(--text-primary)' }}>
        Capital Progression Through Tournament
      </h3>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%' }}>
        {/* Regime bands */}
        {regimeBands.map((b, i) => (
          <rect key={i} x={xScale(b.start)} y={padding.top} width={xScale(b.end) - xScale(b.start)} height={chartH} fill={regimeColors[b.regime]} />
        ))}

        {/* Grid */}
        {[0, 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000].filter(v => v <= maxY).map(v => (
          <g key={v}>
            <line x1={padding.left} y1={yScale(v)} x2={padding.left + chartW} y2={yScale(v)} stroke={gridColor} strokeDasharray="4,4" />
            <text x={padding.left - 12} y={yScale(v)} fill={textColor} fontSize="11" textAnchor="end" dominantBaseline="middle">
              ${(v / 1000).toFixed(0)}K
            </text>
          </g>
        ))}

        {/* X axis */}
        {[0, 20, 40, 60, 80, maxRound].map(v => (
          <text key={v} x={xScale(v)} y={padding.top + chartH + 25} fill={textColor} fontSize="11" textAnchor="middle">{v}</text>
        ))}
        <text x={padding.left + chartW / 2} y={height - 15} fill={textColor} fontSize="12" textAnchor="middle">Round</text>

        {/* Strategy lines */}
        {[...STRATEGY_ORDER, 'Buy-and-Hold'].map(name => {
          const cfg = STRATEGY_CONFIG[name]
          return (
            <path key={name} d={createPath(name)} fill="none" stroke={cfg.color} strokeWidth={2.5}
              strokeDasharray={name === 'Buy-and-Hold' ? '8,4' : 'none'} strokeLinecap="round" strokeLinejoin="round" />
          )
        })}

        {/* End dots */}
        {[...STRATEGY_ORDER, 'Buy-and-Hold'].map(name => {
          const cfg = STRATEGY_CONFIG[name]
          const yPos = yScale(finalPoint[name])
          return <circle key={`dot-${name}`} cx={xScale(maxRound)} cy={yPos} r="5" fill={cfg.color} />
        })}

        {/* Legend with values */}
        {[...STRATEGY_ORDER, 'Buy-and-Hold'].map((name, i) => {
          const cfg = STRATEGY_CONFIG[name]
          const ret = getReturn(name)
          const yPos = padding.top + 20 + i * 28
          return (
            <g key={`legend-${name}`} transform={`translate(${padding.left + chartW + 20}, ${yPos})`}>
              <line x1="0" y1="0" x2="24" y2="0" stroke={cfg.color} strokeWidth="3" strokeDasharray={name === 'Buy-and-Hold' ? '6,3' : 'none'} />
              <circle cx="12" cy="0" r="4" fill={cfg.color} />
              <text x="32" y="4" fill={isDark ? '#e4e4e7' : '#18181b'} fontSize="12" fontWeight="500">{cfg.short}</text>
              <text x="130" y="4" fill={ret >= 0 ? '#22c55e' : '#ef4444'} fontSize="12" fontWeight="600" textAnchor="end">
                {ret >= 0 ? '+' : ''}{ret.toFixed(1)}%
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ========== STATS ROW ==========
const StatsRow = ({ summary, rounds, isDark }) => {
  const returns = summary?.total_returns_pct || {}
  const wins = summary?.wins_per_strategy || {}
  const benchReturn = summary?.benchmark?.total_return_pct || 0

  // Regime counts
  const regimeCounts = useMemo(() => {
    const c = { bull: 0, bear: 0, sideways: 0 }
    rounds.forEach(r => { if (r.regime) c[r.regime]++ })
    return c
  }, [rounds])

  const total = rounds.length
  const winner = Object.entries(returns).sort((a, b) => b[1] - a[1])[0]

  const border = isDark ? '#27272a' : '#e4e4e7'
  const cardBg = isDark ? '#18181b' : '#fff'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
      {/* Wins Distribution */}
      <div style={{ background: cardBg, border: `1px solid ${border}`, borderRadius: '12px', padding: '16px' }}>
        <div style={{ fontSize: '13px', fontWeight: '600', marginBottom: '14px', color: 'var(--text-primary)' }}>Wins Distribution</div>
        {STRATEGY_ORDER.map(name => {
          const w = wins[name] || 0
          const cfg = STRATEGY_CONFIG[name]
          return (
            <div key={name} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px', gap: '10px' }}>
              <div style={{ width: '50px', fontSize: '11px', color: 'var(--text-muted)' }}>{cfg.short}</div>
              <div style={{ flex: 1, height: '8px', background: isDark ? '#27272a' : '#f4f4f5', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${(w / Math.max(...Object.values(wins))) * 100}%`, height: '100%', background: cfg.color, borderRadius: '4px' }} />
              </div>
              <div style={{ width: '28px', fontSize: '12px', fontWeight: '600', color: 'var(--text-primary)', textAlign: 'right' }}>{w}</div>
            </div>
          )
        })}
      </div>

      {/* vs Benchmark */}
      <div style={{ background: cardBg, border: `1px solid ${border}`, borderRadius: '12px', padding: '16px' }}>
        <div style={{ fontSize: '13px', fontWeight: '600', marginBottom: '14px', color: 'var(--text-primary)' }}>vs Benchmark</div>
        {STRATEGY_ORDER.map(name => {
          const excess = (returns[name] || 0) - benchReturn
          const cfg = STRATEGY_CONFIG[name]
          const maxExcess = Math.max(...STRATEGY_ORDER.map(n => Math.abs((returns[n] || 0) - benchReturn)))
          const barW = (Math.abs(excess) / maxExcess) * 40
          return (
            <div key={name} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px', gap: '6px' }}>
              <div style={{ width: '50px', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'right' }}>{cfg.short}</div>
              <div style={{ width: '80px', height: '8px', position: 'relative' }}>
                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '1px', background: isDark ? '#3f3f46' : '#d4d4d8' }} />
                <div style={{
                  position: 'absolute', top: 0, height: '100%',
                  left: excess < 0 ? `${50 - barW}%` : '50%',
                  width: `${barW}%`, background: cfg.color, borderRadius: '2px'
                }} />
              </div>
              <div style={{ width: '55px', fontSize: '11px', fontWeight: '600', color: excess >= 0 ? '#22c55e' : '#ef4444' }}>
                {excess >= 0 ? '+' : ''}{excess.toFixed(1)}%
              </div>
            </div>
          )
        })}
      </div>

      {/* Tournament Metrics */}
      <div style={{ background: cardBg, border: `1px solid ${border}`, borderRadius: '12px', padding: '16px', fontFamily: 'ui-monospace, monospace', fontSize: '11px' }}>
        <div style={{ fontSize: '13px', fontWeight: '600', marginBottom: '12px', color: 'var(--text-primary)', fontFamily: 'inherit' }}>Tournament Metrics</div>
        <div style={{ color: 'var(--text-secondary)', lineHeight: '1.8' }}>
          <div>Total Rounds: <strong>{total}</strong></div>
          <div>Cooperation: <strong>{((summary?.cooperation_rate || 0) * 100).toFixed(1)}%</strong></div>
          <div>Gini Coeff: <strong>{(summary?.allocation_gini || 0).toFixed(3)}</strong></div>
          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: `1px dashed ${border}` }}>
            <span style={{ color: '#22c55e' }}>Bull: {regimeCounts.bull} ({((regimeCounts.bull/total)*100).toFixed(0)}%)</span>
          </div>
          <div><span style={{ color: '#ef4444' }}>Bear: {regimeCounts.bear} ({((regimeCounts.bear/total)*100).toFixed(0)}%)</span></div>
          <div><span style={{ color: '#eab308' }}>Sideways: {regimeCounts.sideways} ({((regimeCounts.sideways/total)*100).toFixed(0)}%)</span></div>
          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: `1px dashed ${border}` }}>
            Winner: <strong style={{ color: STRATEGY_CONFIG[winner?.[0]]?.color }}>{winner?.[0]}</strong>
          </div>
        </div>
      </div>

      {/* Final Standings */}
      <div style={{ background: cardBg, border: `1px solid ${border}`, borderRadius: '12px', overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)', borderBottom: `1px solid ${border}` }}>Final Standings</div>
        <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: isDark ? '#27272a' : '#f4f4f5' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: '500', color: 'var(--text-muted)' }}>Strategy</th>
              <th style={{ padding: '8px 6px', textAlign: 'right', fontWeight: '500', color: 'var(--text-muted)' }}>Return</th>
              <th style={{ padding: '8px 6px', textAlign: 'right', fontWeight: '500', color: 'var(--text-muted)' }}>Wins</th>
              <th style={{ padding: '8px 6px', textAlign: 'right', fontWeight: '500', color: 'var(--text-muted)' }}>Alloc</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: '500', color: 'var(--text-muted)' }}>Beat</th>
            </tr>
          </thead>
          <tbody>
            {STRATEGY_ORDER.map(name => {
              const ret = returns[name] || 0
              const cfg = STRATEGY_CONFIG[name]
              const beat = ret > benchReturn
              return (
                <tr key={name} style={{ borderBottom: `1px solid ${border}` }}>
                  <td style={{ padding: '8px 10px' }}>
                    <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: cfg.color, marginRight: '6px' }} />
                    <span style={{ color: 'var(--text-primary)' }}>{cfg.short}</span>
                  </td>
                  <td style={{ padding: '8px 6px', textAlign: 'right', fontWeight: '600', color: ret >= 0 ? '#22c55e' : '#ef4444' }}>
                    {ret >= 0 ? '+' : ''}{ret.toFixed(1)}%
                  </td>
                  <td style={{ padding: '8px 6px', textAlign: 'right', color: 'var(--text-secondary)' }}>{summary?.win_rates?.[name]?.toFixed(0)}%</td>
                  <td style={{ padding: '8px 6px', textAlign: 'right', color: 'var(--text-secondary)' }}>{summary?.allocation_pcts?.[name]?.toFixed(1)}%</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', fontWeight: '600', color: beat ? '#22c55e' : '#ef4444' }}>{beat ? 'Y' : 'N'}</td>
                </tr>
              )
            })}
            <tr style={{ background: isDark ? '#27272a' : '#f4f4f5' }}>
              <td style={{ padding: '8px 10px', color: 'var(--text-muted)' }}>📊 B&H</td>
              <td style={{ padding: '8px 6px', textAlign: 'right', fontWeight: '600', color: '#22c55e' }}>+{benchReturn.toFixed(1)}%</td>
              <td style={{ padding: '8px 6px', textAlign: 'right', color: 'var(--text-muted)' }}>—</td>
              <td style={{ padding: '8px 6px', textAlign: 'right', color: 'var(--text-secondary)' }}>100%</td>
              <td style={{ padding: '8px 10px', textAlign: 'center', color: 'var(--text-muted)' }}>REF</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ========== ROUND FEED ==========
const RoundFeed = ({ rounds, isDark }) => {
  const [expanded, setExpanded] = useState(null)
  const sorted = useMemo(() => [...rounds].reverse(), [rounds])
  const border = isDark ? '#27272a' : '#e4e4e7'

  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      {sorted.map(r => {
        const isExp = expanded === r.round_num
        const market = r.market?.daily_return_pct || 0
        const isUp = market >= 0
        const winner = r.round_results?.winner
        const winnerCfg = STRATEGY_CONFIG[winner]

        return (
          <div key={r.round_num} style={{ borderBottom: `1px solid ${border}` }}>
            <div onClick={() => setExpanded(isExp ? null : r.round_num)}
              style={{ padding: '12px 16px', cursor: 'pointer', background: isExp ? (isDark ? '#18181b' : '#fafafa') : 'transparent' }}>
              {/* Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontWeight: '700', fontSize: '13px', color: 'var(--text-primary)' }}>Round {r.round_num}</span>
                  <span style={{
                    fontSize: '10px', padding: '2px 8px', borderRadius: '4px', fontWeight: '500',
                    background: r.regime === 'bull' ? '#dcfce7' : r.regime === 'bear' ? '#fee2e2' : '#fef9c3',
                    color: r.regime === 'bull' ? '#166534' : r.regime === 'bear' ? '#991b1b' : '#854d0e'
                  }}>{r.regime}</span>
                </div>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{r.date}</span>
              </div>
              {/* Market & Winner */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '14px' }}>{isUp ? '📈' : '📉'}</span>
                  <span style={{ fontWeight: '600', fontSize: '13px', color: isUp ? '#22c55e' : '#ef4444' }}>
                    {isUp ? '+' : ''}{market.toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Winner:</span>
                  <span style={{ fontWeight: '600', color: winnerCfg?.color }}>
                    {winnerCfg?.icon} {winnerCfg?.short}
                  </span>
                </div>
              </div>
            </div>

            {/* Expanded details */}
            {isExp && (
              <div style={{ padding: '0 16px 16px', background: isDark ? '#18181b' : '#fafafa' }}>
                <div style={{ fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Strategy Positions & Returns
                </div>
                {STRATEGY_ORDER.map(name => {
                  const dec = r.strategy_decisions?.[name] || {}
                  const ret = r.round_results?.returns_pct?.[name] || 0
                  const dollarRet = r.round_results?.dollar_returns?.[name] || 0
                  const isWinner = winner === name
                  const cfg = STRATEGY_CONFIG[name]

                  return (
                    <div key={name} style={{
                      padding: '12px', marginBottom: '8px',
                      background: isDark ? '#09090b' : '#fff',
                      borderRadius: '8px',
                      border: isWinner ? `2px solid ${cfg.color}` : `1px solid ${border}`
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '16px' }}>{cfg.icon}</span>
                          <span style={{ fontWeight: '600', color: cfg.color }}>{name}</span>
                          {isWinner && <span style={{ fontSize: '14px' }}>🏆</span>}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            Position: <strong style={{ color: 'var(--text-primary)' }}>{dec.position_pct?.toFixed(1)}%</strong>
                          </span>
                          <span style={{ fontSize: '13px', fontWeight: '700', color: ret >= 0 ? '#22c55e' : '#ef4444' }}>
                            {ret >= 0 ? '+' : ''}{ret.toFixed(2)}% <span style={{ fontSize: '11px', fontWeight: '500' }}>
                              (${dollarRet >= 0 ? '+' : ''}{dollarRet.toFixed(0)})
                            </span>
                          </span>
                        </div>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                        {dec.reasoning}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ========== CHAT ==========
const Chat = ({ isDark, summary }) => {
  const [msgs, setMsgs] = useState([{ id: 1, role: 'assistant', content: "👋 Ask about strategies or tournament results!" }])
  const [input, setInput] = useState('')
  const ref = useRef(null)
  useEffect(() => { ref.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const send = () => {
    if (!input.trim()) return
    const q = input.toLowerCase()
    setMsgs(p => [...p, { id: Date.now(), role: 'user', content: input }])
    setInput('')

    const returns = summary?.total_returns_pct || {}
    const wins = summary?.wins_per_strategy || {}
    let res = "Try asking about strategies or who's winning!"

    if (q.includes('winner') || q.includes('best')) {
      const sorted = Object.entries(returns).sort((a, b) => b[1] - a[1])
      res = `🏆 **Rankings:**\n${sorted.map(([n, r], i) => `${i + 1}. ${STRATEGY_CONFIG[n]?.icon} ${n}: ${r >= 0 ? '+' : ''}${r.toFixed(1)}%`).join('\n')}`
    } else if (q.includes('defector')) {
      res = `🎯 **Defector**: +${returns['Defector']?.toFixed(1)}%\nWins: ${wins['Defector']} | Strategy: Contrarian bets`
    } else if (q.includes('signal')) {
      res = `🤖 **Signal Follower**: +${returns['Signal Follower']?.toFixed(1)}%\nWins: ${wins['Signal Follower']} | Strategy: LLM signals`
    }

    setTimeout(() => setMsgs(p => [...p, { id: Date.now() + 1, role: 'assistant', content: res }]), 200)
  }

  const border = isDark ? '#27272a' : '#e4e4e7'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {msgs.map(m => (
          <div key={m.id} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '85%', padding: '10px 14px',
              borderRadius: m.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
              fontSize: '13px', lineHeight: '1.5', whiteSpace: 'pre-wrap',
              background: m.role === 'user' ? '#18181b' : (isDark ? '#27272a' : '#f4f4f5'),
              color: m.role === 'user' ? '#fff' : 'var(--text-primary)'
            }}>{m.content}</div>
          </div>
        ))}
        <div ref={ref} />
      </div>
      <div style={{ padding: '12px', borderTop: `1px solid ${border}` }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="Ask about the tournament..." style={{
              flex: 1, padding: '12px 14px', borderRadius: '10px', border: `1px solid ${border}`,
              background: isDark ? '#18181b' : '#fff', color: 'var(--text-primary)', fontSize: '13px'
            }} />
          <button onClick={send} style={{
            padding: '12px 18px', borderRadius: '10px', border: 'none',
            background: '#18181b', color: '#fff', fontWeight: '600', cursor: 'pointer'
          }}>→</button>
        </div>
      </div>
    </div>
  )
}

// ========== MAIN ==========
export default function ArenaPage() {
  const { isDark } = useTheme()
  const [ticker, setTicker] = useState('AAPL')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('feed')

  useEffect(() => {
    setLoading(true)
    loadTickerData(ticker).then(d => { setData(d); setLoading(false) })
  }, [ticker])

  const summary = data?.summary
  const rounds = data?.detailed?.rounds || []
  const isLive = !!data

  const bg = isDark ? '#09090b' : '#f4f4f5'
  const cardBg = isDark ? '#18181b' : '#fff'
  const border = isDark ? '#27272a' : '#e4e4e7'

  return (
    <div style={{ display: 'flex', height: '100%', background: bg }}>
      {/* Main */}
      <div style={{ flex: 1, padding: '24px', overflowY: 'auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Game Theory Tournament: {ticker}
            </h1>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
              Capital: $1,000,000 | Realloc: 10% | Benchmark: Buy-and-Hold
            </p>
          </div>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <span style={{
              padding: '6px 12px', borderRadius: '8px', fontSize: '12px', fontWeight: '600',
              background: loading ? '#fef9c3' : isLive ? '#dcfce7' : '#fee2e2',
              color: loading ? '#854d0e' : isLive ? '#166534' : '#991b1b'
            }}>
              {loading ? '⏳ Loading' : isLive ? '✓ Live Data' : '✗ Error'}
            </span>
            <select value={ticker} onChange={e => setTicker(e.target.value)} style={{
              padding: '10px 14px', borderRadius: '10px', border: `1px solid ${border}`,
              background: cardBg, color: 'var(--text-primary)', fontWeight: '600', fontSize: '14px', cursor: 'pointer'
            }}>
              {AVAILABLE_TICKERS.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '80px', color: 'var(--text-muted)' }}>Loading tournament data...</div>
        ) : (
          <>
            <StrategyCards summary={summary} isDark={isDark} />
            <CapitalChart rounds={rounds} isDark={isDark} summary={summary} />
            <div style={{ height: '20px' }} />
            <StatsRow summary={summary} rounds={rounds} isDark={isDark} />
          </>
        )}
      </div>

      {/* Sidebar */}
      <div style={{ width: '400px', borderLeft: `1px solid ${border}`, background: cardBg, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', borderBottom: `1px solid ${border}` }}>
          {[{ id: 'feed', label: 'Round History', icon: '📋' }, { id: 'chat', label: 'Ask AI', icon: '💬' }].map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex: 1, padding: '16px', border: 'none', cursor: 'pointer', fontSize: '13px', fontWeight: '600',
              background: tab === t.id ? cardBg : (isDark ? '#09090b' : '#f4f4f5'),
              color: tab === t.id ? 'var(--text-primary)' : 'var(--text-muted)',
              borderBottom: tab === t.id ? '2px solid var(--text-primary)' : '2px solid transparent'
            }}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {tab === 'feed' && <RoundFeed rounds={rounds} isDark={isDark} />}
          {tab === 'chat' && <Chat isDark={isDark} summary={summary} />}
        </div>
      </div>
    </div>
  )
}