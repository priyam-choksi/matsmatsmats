import React, { useState, useRef, useEffect, useMemo } from 'react'
import { useTheme } from '../App'

const FOLDER = 'gt_tournament_20251206_141939'
const PATH = '/data/game_theory_results'
const TICKERS = ['NVDA','AAPL','MSFT','GOOGL','META','JPM','GS','V','LLY','JNJ','UNH','AMZN','TSLA','WMT','XOM','CVX','SPY','QQQ','PG','KO']

const S = {
  'Signal Follower': { c: '#a855f7', s: 'Signal', d: 'LLM-powered AI signals' },
  'Cooperator': { c: '#22c55e', s: 'Coop', d: 'Follows group consensus' },
  'Defector': { c: '#f97316', s: 'Defect', d: 'Bets against the crowd' },
  'Tit-for-Tat': { c: '#3b82f6', s: 'TFT', d: 'Copies previous winner' }
}
const O = Object.keys(S)

async function load(t) {
  try {
    const [sum, det] = await Promise.all([
      fetch(`${PATH}/${FOLDER}/by_ticker/${t}/${t}_summary.json`).then(r => r.json()),
      fetch(`${PATH}/${FOLDER}/by_ticker/${t}/${t}_detailed.json`).then(r => r.json())
    ])
    det.rounds?.sort((a, b) => a.round_num - b.round_num)
    return { sum, det, t }
  } catch { return null }
}

async function loadAll() {
  return (await Promise.all(TICKERS.map(load))).filter(Boolean)
}

const fmtPct = n => `${n >= 0 ? '+' : ''}${(n || 0).toFixed(2)}%`
const fmtK = n => n >= 1e6 ? `$${(n/1e6).toFixed(2)}M` : `$${(n/1e3).toFixed(0)}K`

export default function ArenaPage() {
  const { isDark } = useTheme()
  const [tab, setTab] = useState('overview')
  const [all, setAll] = useState([])
  const [loading, setLoading] = useState(true)
  const [ticker, setTicker] = useState('AAPL')
  const [sidebarTab, setSidebarTab] = useState('feed')

  useEffect(() => { loadAll().then(d => { setAll(d); setLoading(false) }) }, [])

  const bdr = isDark ? '#27272a' : '#e4e4e7'
  const bg2 = isDark ? '#18181b' : '#f4f4f5'
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'race', label: 'Race' },
    { id: 'regimes', label: 'Regimes' },
    { id: 'deepdive', label: 'Deep Dive' },
    { id: 'dna', label: 'Strategy DNA' },
    { id: 'whatif', label: 'What-If' },
    { id: 'h2h', label: 'Head to Head' },
  ]

  return (
    <div style={{ height: '100%', display: 'flex', background: isDark ? '#09090b' : '#fafafa' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '16px 20px', borderBottom: `1px solid ${bdr}`, background: isDark ? '#0a0a0a' : '#fff', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div>
              <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: 'var(--text-primary)', letterSpacing: '-0.3px' }}>
                Strategy Arena
              </h1>
              <p style={{ margin: '2px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>
                Game theory tournament · {all.length} tickers · {(all.length * 90).toLocaleString()} rounds
              </p>
            </div>
            <div style={{ display: 'flex', gap: '12px' }}>
              {O.map(s => (
                <div key={s} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: S[s].c }} />
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{S[s].s}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '2px' }}>
            {tabs.map(t => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                padding: '8px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer',
                fontSize: '12px', fontWeight: '500',
                background: tab === t.id ? (isDark ? '#27272a' : '#e4e4e7') : 'transparent',
                color: tab === t.id ? 'var(--text-primary)' : 'var(--text-muted)'
              }}>{t.label}</button>
            ))}
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)', fontSize: '13px' }}>
              Loading tournament data...
            </div>
          ) : (
            <>
              {tab === 'overview' && <Overview all={all} dark={isDark} />}
              {tab === 'race' && <Race all={all} dark={isDark} ticker={ticker} setTicker={setTicker} />}
              {tab === 'regimes' && <Regimes all={all} dark={isDark} />}
              {tab === 'deepdive' && <DeepDive all={all} dark={isDark} ticker={ticker} setTicker={setTicker} />}
              {tab === 'dna' && <StrategyDNA all={all} dark={isDark} />}
              {tab === 'whatif' && <WhatIf all={all} dark={isDark} />}
              {tab === 'h2h' && <HeadToHead all={all} dark={isDark} />}
            </>
          )}
        </div>
      </div>
      <RightSidebar all={all} dark={isDark} sidebarTab={sidebarTab} setSidebarTab={setSidebarTab} />
    </div>
  )
}

function RightSidebar({ all, dark, sidebarTab, setSidebarTab }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const [expandedFeed, setExpandedFeed] = useState(null)
  const [expandedPortfolio, setExpandedPortfolio] = useState(null)

  const feedData = useMemo(() => {
    const items = []
    all.slice(0, 5).forEach(({ det, t }) => {
      (det?.rounds?.slice(-3) || []).forEach(r => {
        const winner = r.round_results?.winner
        if (winner) items.push({
          id: `${t}-${r.round_num}`, ticker: t, round: r.round_num, date: r.date,
          regime: r.regime, winner, market: r.market?.daily_return_pct || 0,
          decisions: r.strategy_decisions, returns: r.round_results?.returns_pct
        })
      })
    })
    return items.sort((a, b) => b.round - a.round).slice(0, 10)
  }, [all])

  const portfolioData = useMemo(() => {
    if (!all.length) return []
    const final = {}
    O.forEach(s => { final[s] = { name: s, totalValue: 0, totalReturn: 0, wins: 0, holdings: [] } })
    all.forEach(({ sum, det, t }) => {
      const lastRound = det?.rounds?.[det.rounds.length - 1]
      O.forEach(s => {
        const alloc = lastRound?.round_results?.allocations_after?.[s] || 250000
        final[s].totalValue += alloc
        final[s].totalReturn += sum?.total_returns_pct?.[s] || 0
        final[s].wins += sum?.wins_per_strategy?.[s] || 0
        final[s].holdings.push({ ticker: t, value: alloc, return: sum?.total_returns_pct?.[s] || 0 })
      })
    })
    O.forEach(s => { final[s].avgReturn = final[s].totalReturn / all.length; final[s].holdings.sort((a, b) => b.value - a.value) })
    return Object.values(final).sort((a, b) => b.totalValue - a.totalValue)
  }, [all])

  return (
    <aside style={{ width: '320px', background: dark ? '#0a0a0a' : '#fff', borderLeft: `1px solid ${bdr}`, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ display: 'flex', borderBottom: `1px solid ${bdr}` }}>
        {['feed', 'portfolios', 'chat'].map(tab => (
          <button key={tab} onClick={() => setSidebarTab(tab)} style={{
            flex: 1, padding: '12px', fontSize: '11px', fontWeight: '500', background: 'none', border: 'none', cursor: 'pointer',
            borderBottom: sidebarTab === tab ? '2px solid var(--text-primary)' : '2px solid transparent',
            color: sidebarTab === tab ? 'var(--text-primary)' : 'var(--text-muted)'
          }}>{tab === 'feed' ? 'Feed' : tab === 'portfolios' ? 'Portfolios' : 'Ask AI'}</button>
        ))}
      </div>
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {sidebarTab === 'feed' && <FeedTab feedData={feedData} expandedFeed={expandedFeed} setExpandedFeed={setExpandedFeed} dark={dark} />}
        {sidebarTab === 'portfolios' && <PortfoliosTab portfolioData={portfolioData} expandedPortfolio={expandedPortfolio} setExpandedPortfolio={setExpandedPortfolio} dark={dark} />}
        {sidebarTab === 'chat' && <ChatTab dark={dark} all={all} />}
      </div>
    </aside>
  )
}

function FeedTab({ feedData, expandedFeed, setExpandedFeed, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Recent Rounds</div>
      {feedData.map(item => {
        const isExp = expandedFeed === item.id
        const regimeColors = { bull: '#22c55e', bear: '#ef4444', sideways: '#eab308' }
        return (
          <div key={item.id} onClick={() => setExpandedFeed(isExp ? null : item.id)} style={{
            padding: '10px 12px', borderRadius: '8px', background: dark ? '#18181b' : '#f4f4f5', border: `1px solid ${bdr}`, cursor: 'pointer'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontWeight: '600', fontSize: '13px', color: 'var(--text-primary)' }}>{item.ticker}</span>
                <span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', background: `${regimeColors[item.regime]}15`, color: regimeColors[item.regime], fontWeight: '500', textTransform: 'uppercase' }}>{item.regime}</span>
              </div>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>R{item.round}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[item.winner].c }} />
                <span style={{ fontSize: '11px', fontWeight: '500', color: S[item.winner].c }}>{S[item.winner].s}</span>
              </div>
              <span style={{ fontSize: '11px', fontWeight: '600', color: item.market >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(item.market)}</span>
            </div>
            {isExp && item.decisions && (
              <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: `1px dashed ${bdr}` }}>
                {O.map(s => {
                  const dec = item.decisions[s], ret = item.returns?.[s] || 0
                  return (
                    <div key={s} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', fontSize: '10px' }}>
                      <div style={{ width: '4px', height: '4px', borderRadius: '1px', background: S[s].c }} />
                      <span style={{ color: 'var(--text-muted)', flex: 1 }}>{dec?.position_pct?.toFixed(0)}%</span>
                      <span style={{ fontWeight: '600', color: ret >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(ret)}</span>
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

function PortfoliosTab({ portfolioData, expandedPortfolio, setExpandedPortfolio, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Strategy Portfolios</div>
      {portfolioData.map(p => {
        const isExp = expandedPortfolio === p.name
        const pnl = p.totalValue - (250000 * (p.holdings?.length || 1))
        const pnlPct = (pnl / (250000 * (p.holdings?.length || 1))) * 100
        return (
          <div key={p.name} onClick={() => setExpandedPortfolio(isExp ? null : p.name)} style={{ borderRadius: '8px', border: `1px solid ${S[p.name].c}30`, overflow: 'hidden', cursor: 'pointer' }}>
            <div style={{ padding: '12px', background: `${S[p.name].c}08` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: S[p.name].c }} />
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: S[p.name].c }}>{S[p.name].s}</div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{p.wins} wins</div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>{fmtK(p.totalValue)}</div>
                  <div style={{ fontSize: '10px', color: pnlPct >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(pnlPct)}</div>
                </div>
              </div>
            </div>
            {isExp && (
              <div style={{ background: dark ? '#18181b' : '#fff', maxHeight: '160px', overflowY: 'auto' }}>
                <table style={{ width: '100%', fontSize: '10px', borderCollapse: 'collapse' }}>
                  <thead><tr style={{ background: dark ? '#27272a' : '#f4f4f5' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: '500' }}>Ticker</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--text-muted)', fontWeight: '500' }}>Value</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--text-muted)', fontWeight: '500' }}>Return</th>
                  </tr></thead>
                  <tbody>{p.holdings.slice(0, 8).map((h, i) => (
                    <tr key={i} style={{ borderTop: `1px solid ${bdr}` }}>
                      <td style={{ padding: '5px 10px', fontWeight: '500', color: 'var(--text-primary)' }}>{h.ticker}</td>
                      <td style={{ padding: '5px 10px', textAlign: 'right', color: 'var(--text-secondary)' }}>{fmtK(h.value)}</td>
                      <td style={{ padding: '5px 10px', textAlign: 'right', fontWeight: '500', color: h.return >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(h.return)}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function ChatTab({ dark, all }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const [messages, setMessages] = useState([{ id: 1, role: 'assistant', content: 'Ask me about the tournament results, strategies, or specific tickers.' }])
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const handleSend = () => {
    if (!input.trim()) return
    const q = input.toLowerCase()
    setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: input }])
    setInput('')
    let response = "I can help you analyze the tournament data. Try asking about winners, strategies, or specific tickers."
    if (q.includes('winner') || q.includes('best')) {
      const returns = {}
      O.forEach(s => { returns[s] = 0 })
      all.forEach(({ sum }) => { O.forEach(s => { returns[s] += sum?.total_returns_pct?.[s] || 0 }) })
      const sorted = Object.entries(returns).sort((a, b) => b[1] - a[1])
      response = `Overall Rankings:\n${sorted.map(([s, r], i) => `${i + 1}. ${s}: ${fmtPct(r / all.length)} avg`).join('\n')}`
    }
    setTimeout(() => { setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: response }]) }, 300)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {messages.map(m => (
          <div key={m.id} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '85%', padding: '10px 12px', borderRadius: m.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
              fontSize: '12px', lineHeight: '1.5', whiteSpace: 'pre-wrap',
              background: m.role === 'user' ? '#a855f7' : (dark ? '#27272a' : '#f4f4f5'),
              color: m.role === 'user' ? '#fff' : 'var(--text-primary)'
            }}>{m.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div style={{ padding: '12px', borderTop: `1px solid ${bdr}` }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Ask about the tournament..." style={{
              flex: 1, padding: '10px 12px', borderRadius: '8px', border: `1px solid ${bdr}`,
              background: dark ? '#18181b' : '#fff', color: 'var(--text-primary)', fontSize: '12px'
            }} />
          <button onClick={handleSend} style={{ padding: '10px 16px', borderRadius: '8px', border: 'none', background: '#a855f7', color: '#fff', fontWeight: '500', cursor: 'pointer', fontSize: '12px' }}>Send</button>
        </div>
      </div>
    </div>
  )
}

function Overview({ all, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const bg2 = dark ? '#18181b' : '#f4f4f5'
  const ref = useRef()
  const [hovRound, setHovRound] = useState(null)

  const { hist, stats } = useMemo(() => {
    if (!all.length) return { hist: [], stats: null }
    const n = all.length, maxR = Math.max(...all.map(d => d.det?.rounds?.length || 0)), init = 250000 * n
    const h = [{ r: 0, holdings: {} }]
    O.forEach(s => { h[0][s] = init; h[0].holdings[s] = all.map(({ t }) => ({ ticker: t, value: 250000, position: 0 })) })
    h[0].bh = init
    for (let r = 1; r <= maxR; r++) {
      const pt = { r, holdings: {} }
      O.forEach(s => { pt[s] = 0; pt.holdings[s] = [] })
      pt.bh = 0
      all.forEach(({ det, t }) => {
        const rd = det?.rounds?.[r - 1]
        if (rd) {
          const a = rd.round_results?.allocations_after || {}
          O.forEach(s => { const val = a[s] || 0; pt[s] += val; pt.holdings[s].push({ ticker: t, value: val, position: rd.strategy_decisions?.[s]?.position_pct || 0 }) })
          let bh = 250000
          for (let i = 0; i < r; i++) bh *= (1 + (det.rounds[i]?.market?.daily_return || 0))
          pt.bh += bh
        }
      })
      h.push(pt)
    }
    const final = h[h.length - 1]
    const strats = O.map(s => ({ name: s, final: final[s], ret: ((final[s] / init) - 1) * 100, wins: all.reduce((a, { sum }) => a + (sum?.wins_per_strategy?.[s] || 0), 0) })).sort((a, b) => b.ret - a.ret)
    return { hist: h, stats: { strats, init, final, benchRet: ((final.bh / init) - 1) * 100, n, maxR } }
  }, [all])

  if (!stats) return null

  const w = 700, ht = 260, p = { t: 25, r: 70, b: 35, l: 60 }
  const cw = w - p.l - p.r, ch = ht - p.t - p.b
  const allV = hist.flatMap(d => [...O.map(s => d[s]), d.bh])
  const maxY = Math.ceil(Math.max(...allV) / 1e6) * 1e6, minY = Math.floor(Math.min(...allV) * 0.95 / 1e6) * 1e6
  const x = r => p.l + (r / stats.maxR) * cw, y = v => p.t + ch - ((v - minY) / (maxY - minY)) * ch
  const path = k => hist.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(d.r)},${y(d[k])}`).join(' ')
  const winner = stats.strats[0]
  const currentData = hovRound !== null ? hist[hovRound] : hist[hist.length - 1]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {stats.strats.map((s, i) => (
          <div key={s.name} style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '16px', border: `1px solid ${bdr}`, borderLeft: `3px solid ${S[s.name].c}` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: S[s.name].c }} />
              <span style={{ fontSize: '12px', fontWeight: '500', color: 'var(--text-secondary)' }}>{S[s.name].s}</span>
              {i === 0 && <span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '3px', background: '#22c55e15', color: '#22c55e', fontWeight: '500', marginLeft: 'auto' }}>LEADER</span>}
            </div>
            <div style={{ fontSize: '22px', fontWeight: '600', color: s.ret >= 0 ? '#22c55e' : '#ef4444', marginBottom: '4px' }}>{fmtPct(s.ret)}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{fmtK(s.final)} · {s.wins} wins</div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Portfolio Growth</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{stats.n} tickers × $250K initial</div>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>B&H: <span style={{ color: '#71717a', fontWeight: '500' }}>{fmtPct(stats.benchRet)}</span></div>
        </div>
        <svg ref={ref} viewBox={`0 0 ${w} ${ht}`} style={{ width: '100%', cursor: 'crosshair' }}
          onMouseMove={e => { const rect = ref.current?.getBoundingClientRect(); if (!rect) return; const mx = (e.clientX - rect.left) * (w / rect.width); const idx = Math.round((mx - p.l) / cw * stats.maxR); if (idx >= 0 && idx < hist.length) setHovRound(idx) }}
          onMouseLeave={() => setHovRound(null)}>
          {[minY, minY + (maxY - minY) * 0.5, maxY].map(v => (<g key={v}><line x1={p.l} y1={y(v)} x2={p.l + cw} y2={y(v)} stroke={bdr} /><text x={p.l - 8} y={y(v)} fill={dark ? '#525252' : '#a1a1aa'} fontSize="9" textAnchor="end" dominantBaseline="middle">{fmtK(v)}</text></g>))}
          {[0, 30, 60, 90].filter(v => v <= stats.maxR).map(v => (<text key={v} x={x(v)} y={ht - 10} fill={dark ? '#525252' : '#a1a1aa'} fontSize="9" textAnchor="middle">R{v}</text>))}
          <path d={path('bh')} fill="none" stroke="#71717a" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.5" />
          {O.map(s => (<path key={s} d={path(s)} fill="none" stroke={S[s].c} strokeWidth="2" strokeLinecap="round" />))}
          {O.map(s => (<g key={`e-${s}`}><circle cx={x(stats.maxR)} cy={y(stats.final[s])} r="4" fill={S[s].c} /><text x={x(stats.maxR) + 8} y={y(stats.final[s]) + 1} fill={S[s].c} fontSize="10" fontWeight="600" dominantBaseline="middle">{fmtK(stats.final[s])}</text></g>))}
          {hovRound !== null && (<g><line x1={x(hovRound)} y1={p.t} x2={x(hovRound)} y2={p.t + ch} stroke={dark ? '#fff' : '#000'} strokeWidth="1" strokeDasharray="2,2" opacity="0.3" />{O.map(s => (<circle key={`h-${s}`} cx={x(hovRound)} cy={y(currentData[s])} r="5" fill={S[s].c} stroke={dark ? '#18181b' : '#fff'} strokeWidth="2" />))}</g>)}
        </svg>
        {hovRound !== null && (
          <div style={{ marginTop: '12px', padding: '10px 16px', borderRadius: '8px', background: bg2, display: 'flex', gap: '16px', alignItems: 'center', fontSize: '12px' }}>
            <span style={{ fontWeight: '500', color: 'var(--text-primary)' }}>R{hovRound}</span>
            {O.map(s => (<span key={s} style={{ color: S[s].c, fontWeight: '500' }}>{S[s].s}: {fmtK(currentData[s])}</span>))}
          </div>
        )}
      </div>

      {/* Ticker Table */}
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Performance by Ticker</div>
        </div>
        <div style={{ overflowX: 'auto', maxHeight: '280px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
            <thead style={{ position: 'sticky', top: 0, background: bg2 }}>
              <tr>
                <th style={{ padding: '10px 16px', textAlign: 'left', fontWeight: '500', color: 'var(--text-muted)', fontSize: '10px', textTransform: 'uppercase' }}>Ticker</th>
                {O.map(s => (<th key={s} style={{ padding: '10px 12px', textAlign: 'center', fontWeight: '500', color: S[s].c, fontSize: '10px' }}>{S[s].s}</th>))}
                <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: '500', color: '#71717a', fontSize: '10px' }}>B&H</th>
              </tr>
            </thead>
            <tbody>
              {all.map(({ sum, t }) => {
                const rets = sum?.total_returns_pct || {}, bench = sum?.benchmark?.total_return_pct || 0
                const best = O.reduce((a, b) => (rets[b] || 0) > (rets[a] || 0) ? b : a)
                return (
                  <tr key={t} style={{ borderBottom: `1px solid ${bdr}` }}>
                    <td style={{ padding: '10px 16px', fontWeight: '500', color: 'var(--text-primary)' }}>{t}</td>
                    {O.map(s => { const r = rets[s] || 0; return (
                      <td key={s} style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <span style={{ padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: s === best ? '600' : '400', background: s === best ? (r >= 0 ? '#22c55e15' : '#ef444415') : 'transparent', color: r >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(r)}</span>
                      </td>
                    )})}
                    <td style={{ padding: '8px 12px', textAlign: 'center', color: bench >= 0 ? '#22c55e' : '#ef4444', fontWeight: '400', fontSize: '11px' }}>{fmtPct(bench)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function Race({ all, dark, ticker, setTicker }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const bg2 = dark ? '#18181b' : '#f4f4f5'
  const data = all.find(d => d.t === ticker)
  const rounds = data?.det?.rounds || []
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const ref = useRef()
  const maxF = rounds.length

  const hist = useMemo(() => {
    const h = [{ r: 0 }]; O.forEach(s => h[0][s] = 250000); h[0].bh = 250000; let bh = 250000
    rounds.forEach(r => {
      const a = r.round_results?.allocations_after || {}
      bh *= (1 + (r.market?.daily_return || 0))
      const pt = { r: r.round_num, date: r.date, regime: r.regime, mkt: r.market?.daily_return_pct, winner: r.round_results?.winner, bh }
      O.forEach(s => pt[s] = a[s] || 250000)
      h.push(pt)
    })
    return h
  }, [rounds])

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => { setFrame(f => { if (f >= maxF) { setPlaying(false); return maxF }; return f + 1 }) }, 50)
    return () => clearInterval(id)
  }, [playing, maxF])

  const slice = hist.slice(0, frame + 1), cur = slice[slice.length - 1]
  const w = 650, ht = 280, p = { t: 25, r: 20, b: 35, l: 60 }
  const cw = w - p.l - p.r, ch = ht - p.t - p.b
  const allV = hist.flatMap(d => [...O.map(s => d[s]), d.bh])
  const maxY = Math.ceil(Math.max(...allV) / 100000) * 100000
  const x = r => p.l + (r / maxF) * cw, y = v => p.t + ch - (v / maxY) * ch
  const path = k => slice.map(d => `${d.r === 0 ? 'M' : 'L'}${x(d.r)},${y(d[k])}`).join(' ')
  const sorted = O.map(s => ({ s, v: cur[s] })).sort((a, b) => b.v - a.v)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        {all.map(({ t }) => (
          <button key={t} onClick={() => { setTicker(t); setFrame(0); setPlaying(false) }} style={{
            padding: '6px 12px', borderRadius: '6px', border: `1px solid ${ticker === t ? '#a855f7' : bdr}`,
            background: ticker === t ? '#a855f715' : 'transparent', color: ticker === t ? '#a855f7' : 'var(--text-muted)',
            fontWeight: '500', fontSize: '11px', cursor: 'pointer'
          }}>{t}</button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '16px' }}>
        <div style={{ flex: 1, background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <div style={{ fontSize: '16px', fontWeight: '500', color: 'var(--text-primary)' }}>{ticker}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                Round {frame}/{maxF}
                {cur?.regime && <span style={{ padding: '2px 6px', borderRadius: '3px', fontSize: '9px', fontWeight: '500', textTransform: 'uppercase', background: cur.regime === 'bull' ? '#22c55e15' : cur.regime === 'bear' ? '#ef444415' : '#eab30815', color: cur.regime === 'bull' ? '#22c55e' : cur.regime === 'bear' ? '#ef4444' : '#eab308' }}>{cur.regime}</span>}
                {cur?.mkt !== undefined && <span style={{ color: cur.mkt >= 0 ? '#22c55e' : '#ef4444', fontWeight: '500' }}>{fmtPct(cur.mkt)}</span>}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button onClick={() => { setFrame(0); setPlaying(false) }} style={{ padding: '8px 12px', borderRadius: '6px', border: `1px solid ${bdr}`, background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '12px' }}>Reset</button>
              <button onClick={() => setPlaying(!playing)} style={{ padding: '8px 16px', borderRadius: '6px', border: 'none', background: playing ? '#ef4444' : '#22c55e', color: '#fff', cursor: 'pointer', fontSize: '12px', fontWeight: '500' }}>{playing ? 'Pause' : 'Play'}</button>
              <button onClick={() => setFrame(maxF)} style={{ padding: '8px 12px', borderRadius: '6px', border: `1px solid ${bdr}`, background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '12px' }}>End</button>
            </div>
          </div>
          <svg ref={ref} viewBox={`0 0 ${w} ${ht}`} style={{ width: '100%' }}>
            {[0, 250000, 500000, 750000, 1000000].filter(v => v <= maxY).map(v => (<g key={v}><line x1={p.l} y1={y(v)} x2={p.l + cw} y2={y(v)} stroke={bdr} /><text x={p.l - 8} y={y(v)} fill={dark ? '#525252' : '#a1a1aa'} fontSize="9" textAnchor="end" dominantBaseline="middle">{fmtK(v)}</text></g>))}
            {[0, 30, 60, 90].filter(v => v <= maxF).map(v => (<text key={v} x={x(v)} y={ht - 8} fill={dark ? '#525252' : '#a1a1aa'} fontSize="9" textAnchor="middle">R{v}</text>))}
            <path d={path('bh')} fill="none" stroke="#71717a" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.5" />
            {O.map(s => (<path key={s} d={path(s)} fill="none" stroke={S[s].c} strokeWidth="2" strokeLinecap="round" />))}
            {O.map(s => (<circle key={`d-${s}`} cx={x(frame)} cy={y(cur[s])} r="5" fill={S[s].c} stroke={dark ? '#18181b' : '#fff'} strokeWidth="2" />))}
          </svg>
          <input type="range" min="0" max={maxF} value={frame} onChange={e => { setFrame(+e.target.value); setPlaying(false) }} style={{ width: '100%', marginTop: '12px', accentColor: '#a855f7' }} />
        </div>

        <div style={{ width: '200px', background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '16px', border: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '10px', fontWeight: '500', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '12px' }}>Standings</div>
          {sorted.map(({ s, v }, i) => {
            const ret = ((v / 250000) - 1) * 100
            return (
              <div key={s} style={{ marginBottom: '10px', padding: '10px', borderRadius: '8px', background: i === 0 ? `${S[s].c}10` : 'transparent', border: i === 0 ? `1px solid ${S[s].c}30` : 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[s].c }} />
                  <span style={{ fontSize: '11px', fontWeight: '500', color: S[s].c }}>{S[s].s}</span>
                  {i === 0 && <span style={{ fontSize: '9px', color: '#22c55e', marginLeft: 'auto' }}>1st</span>}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                  <span style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)' }}>{fmtK(v)}</span>
                  <span style={{ fontSize: '11px', fontWeight: '500', color: ret >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(ret)}</span>
                </div>
              </div>
            )
          })}
          <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: `1px dashed ${bdr}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>B&H</span>
            <span style={{ fontSize: '12px', fontWeight: '500', color: '#71717a' }}>{fmtK(cur.bh)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function Regimes({ all, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const regimeStats = useMemo(() => {
    const data = { bull: {}, bear: {}, sideways: {} }
    Object.keys(data).forEach(r => { O.forEach(s => data[r][s] = { returns: [], wins: 0, rounds: 0 }) })
    all.forEach(({ det }) => {
      det?.rounds?.forEach(r => {
        const regime = r.regime; if (!regime || !data[regime]) return
        const rets = r.round_results?.returns_pct || {}, winner = r.round_results?.winner
        O.forEach(s => { if (rets[s] !== undefined) { data[regime][s].returns.push(rets[s]); data[regime][s].rounds++; if (winner === s) data[regime][s].wins++ } })
      })
    })
    return Object.entries(data).map(([regime, strats]) => ({
      regime, strategies: O.map(s => ({ name: s, avgRet: strats[s].returns.length ? strats[s].returns.reduce((a, b) => a + b, 0) / strats[s].returns.length : 0, wins: strats[s].wins, rounds: strats[s].rounds })).sort((a, b) => b.avgRet - a.avgRet),
      totalRounds: O.reduce((a, s) => a + strats[s].rounds, 0) / O.length
    }))
  }, [all])
  const cfg = { bull: { label: 'Bull', color: '#22c55e', bg: '#22c55e10' }, bear: { label: 'Bear', color: '#ef4444', bg: '#ef444410' }, sideways: { label: 'Sideways', color: '#eab308', bg: '#eab30810' } }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div><div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Regime Analysis</div><div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Strategy performance across market conditions</div></div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
        {regimeStats.map(({ regime, strategies, totalRounds }) => {
          const c = cfg[regime], winner = strategies[0]
          return (
            <div key={regime} style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
              <div style={{ padding: '16px', background: c.bg, borderBottom: `1px solid ${bdr}` }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: c.color }}>{c.label} Market</div>
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{Math.round(totalRounds)} rounds</span>
                </div>
                <div style={{ padding: '12px', borderRadius: '8px', background: dark ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.7)', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: S[winner.name].c }} />
                  <div style={{ flex: 1 }}><div style={{ fontSize: '12px', fontWeight: '500', color: S[winner.name].c }}>{winner.name}</div><div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>dominates</div></div>
                  <div style={{ fontSize: '16px', fontWeight: '600', color: winner.avgRet >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(winner.avgRet)}/rd</div>
                </div>
              </div>
              <div style={{ padding: '12px 16px' }}>
                {strategies.map((s, i) => (
                  <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px', padding: '8px', borderRadius: '6px', background: i === 0 ? `${S[s.name].c}08` : 'transparent' }}>
                    <span style={{ width: '20px', fontSize: '11px', color: 'var(--text-muted)', fontWeight: '500' }}>#{i + 1}</span>
                    <div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[s.name].c }} />
                    <span style={{ flex: 1, fontSize: '11px', color: 'var(--text-secondary)' }}>{S[s.name].s}</span>
                    <div style={{ textAlign: 'right' }}><div style={{ fontSize: '11px', fontWeight: '500', color: s.avgRet >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(s.avgRet)}</div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{s.wins}W</div></div>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function DeepDive({ all, dark, ticker, setTicker }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const bg2 = dark ? '#18181b' : '#f4f4f5'
  const data = all.find(d => d.t === ticker), sum = data?.sum, rounds = data?.det?.rounds || []
  const [expanded, setExpanded] = useState(null)
  const rets = sum?.total_returns_pct || {}, wins = sum?.wins_per_strategy || {}, allocs = sum?.final_allocations || {}, bench = sum?.benchmark?.total_return_pct || 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        {all.map(({ t }) => (
          <button key={t} onClick={() => { setTicker(t); setExpanded(null) }} style={{
            padding: '6px 12px', borderRadius: '6px', border: `1px solid ${ticker === t ? '#a855f7' : bdr}`,
            background: ticker === t ? '#a855f715' : 'transparent', color: ticker === t ? '#a855f7' : 'var(--text-muted)',
            fontWeight: '500', fontSize: '11px', cursor: 'pointer'
          }}>{t}</button>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {O.map(s => {
          const r = rets[s] || 0, w = wins[s] || 0, a = allocs[s] || 250000
          return (
            <div key={s} style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '16px', border: `1px solid ${bdr}`, borderTop: `3px solid ${S[s].c}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: S[s].c }} />
                <span style={{ fontSize: '12px', fontWeight: '500', color: S[s].c }}>{S[s].s}</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: '600', color: r >= 0 ? '#22c55e' : '#ef4444', marginBottom: '10px' }}>{fmtPct(r)}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.7 }}>
                <div>Final: <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>{fmtK(a)}</span></div>
                <div>Wins: <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>{w} ({((w / rounds.length) * 100).toFixed(0)}%)</span></div>
                <div style={{ color: r > bench ? '#22c55e' : '#ef4444' }}>{r > bench ? '↑' : '↓'} vs B&H ({fmtPct(bench)})</div>
              </div>
            </div>
          )
        })}
      </div>
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}><div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Round History</div><div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{rounds.length} rounds</div></div>
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {[...rounds].reverse().map(r => {
            const isExp = expanded === r.round_num, w = r.round_results?.winner, m = r.market?.daily_return_pct || 0
            return (
              <div key={r.round_num} style={{ borderBottom: `1px solid ${bdr}` }}>
                <div onClick={() => setExpanded(isExp ? null : r.round_num)} style={{ padding: '12px 16px', cursor: 'pointer', background: isExp ? bg2 : 'transparent' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontWeight: '500', color: 'var(--text-primary)', fontSize: '12px' }}>R{r.round_num}</span>
                      <span style={{ padding: '2px 6px', borderRadius: '3px', fontSize: '9px', fontWeight: '500', textTransform: 'uppercase', background: r.regime === 'bull' ? '#22c55e15' : r.regime === 'bear' ? '#ef444415' : '#eab30815', color: r.regime === 'bull' ? '#22c55e' : r.regime === 'bear' ? '#ef4444' : '#eab308' }}>{r.regime}</span>
                      <span style={{ fontSize: '11px', color: m >= 0 ? '#22c55e' : '#ef4444', fontWeight: '500' }}>{fmtPct(m)}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[w]?.c }} />
                      <span style={{ fontSize: '11px', fontWeight: '500', color: S[w]?.c }}>{S[w]?.s}</span>
                    </div>
                  </div>
                </div>
                {isExp && (
                  <div style={{ padding: '12px 16px', background: bg2 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
                      {O.map(s => {
                        const dec = r.strategy_decisions?.[s] || {}, ret = r.round_results?.returns_pct?.[s] || 0, isWin = w === s
                        return (
                          <div key={s} style={{ padding: '12px', borderRadius: '8px', background: dark ? '#0a0a0a' : '#fff', border: isWin ? `2px solid ${S[s].c}` : `1px solid ${bdr}` }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                              <div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[s].c }} />
                              <span style={{ fontSize: '11px', fontWeight: '500', color: S[s].c }}>{S[s].s}</span>
                              {isWin && <span style={{ fontSize: '9px', color: '#22c55e', marginLeft: 'auto' }}>WIN</span>}
                            </div>
                            <div style={{ display: 'flex', gap: '12px', marginBottom: '8px', fontSize: '11px' }}>
                              <div><div style={{ color: 'var(--text-muted)', marginBottom: '2px' }}>Pos</div><div style={{ fontWeight: '500', color: 'var(--text-primary)' }}>{dec.position_pct?.toFixed(0)}%</div></div>
                              <div><div style={{ color: 'var(--text-muted)', marginBottom: '2px' }}>Ret</div><div style={{ fontWeight: '500', color: ret >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(ret)}</div></div>
                            </div>
                            {dec.reasoning && <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.4, padding: '8px', borderRadius: '4px', background: dark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)', maxHeight: '50px', overflow: 'hidden' }}>{dec.reasoning.slice(0, 120)}...</div>}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function StrategyDNA({ all, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const dna = useMemo(() => {
    return O.map(s => {
      const positions = [], winsAgg = { c: 0, t: 0 }, winsCon = { c: 0, t: 0 }
      const winsByR = { bull: 0, bear: 0, sideways: 0 }, roundsByR = { bull: 0, bear: 0, sideways: 0 }
      let totalWins = 0, totalRounds = 0
      all.forEach(({ det }) => {
        det?.rounds?.forEach(r => {
          const pos = r.strategy_decisions?.[s]?.position_pct || 0
          positions.push(pos)
          const won = r.round_results?.winner === s, regime = r.regime || 'sideways'
          totalRounds++; roundsByR[regime]++
          if (won) { totalWins++; winsByR[regime]++ }
          if (pos > 50) { winsAgg.t++; if (won) winsAgg.c++ } else { winsCon.t++; if (won) winsCon.c++ }
        })
      })
      const avgPos = positions.reduce((a, b) => a + b, 0) / positions.length
      return {
        name: s, avgPos, maxPos: Math.max(...positions), minPos: Math.min(...positions), posRange: Math.max(...positions) - Math.min(...positions),
        aggWin: winsAgg.t > 0 ? (winsAgg.c / winsAgg.t) * 100 : 0, conWin: winsCon.t > 0 ? (winsCon.c / winsCon.t) * 100 : 0,
        bullWin: roundsByR.bull > 0 ? (winsByR.bull / roundsByR.bull) * 100 : 0, bearWin: roundsByR.bear > 0 ? (winsByR.bear / roundsByR.bear) * 100 : 0, sideWin: roundsByR.sideways > 0 ? (winsByR.sideways / roundsByR.sideways) * 100 : 0
      }
    })
  }, [all])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div><div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Strategy DNA</div><div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Behavioral fingerprints from {(all.length * 90).toLocaleString()} decisions</div></div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
        {dna.map(d => (
          <div key={d.name} style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: S[d.name].c }} />
              <div><div style={{ fontSize: '14px', fontWeight: '600', color: S[d.name].c }}>{d.name}</div><div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{S[d.name].d}</div></div>
            </div>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '10px', fontWeight: '500', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Position Behavior</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)', width: '32px' }}>{d.minPos.toFixed(0)}%</span>
                <div style={{ flex: 1, height: '8px', background: bdr, borderRadius: '4px', position: 'relative' }}>
                  <div style={{ position: 'absolute', left: `${d.minPos}%`, width: `${d.posRange}%`, height: '100%', background: `linear-gradient(90deg, ${S[d.name].c}50, ${S[d.name].c})`, borderRadius: '4px' }} />
                  <div style={{ position: 'absolute', left: `${d.avgPos}%`, top: '-4px', width: '3px', height: '16px', background: S[d.name].c, borderRadius: '2px', transform: 'translateX(-50%)' }} />
                </div>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)', width: '32px', textAlign: 'right' }}>{d.maxPos.toFixed(0)}%</span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textAlign: 'center' }}>Avg: <span style={{ fontWeight: '500', color: S[d.name].c }}>{d.avgPos.toFixed(1)}%</span></div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '16px' }}>
              <div style={{ padding: '10px', borderRadius: '8px', background: '#22c55e08', textAlign: 'center' }}><div style={{ fontSize: '16px', fontWeight: '600', color: '#22c55e' }}>{d.bullWin.toFixed(0)}%</div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Bull</div></div>
              <div style={{ padding: '10px', borderRadius: '8px', background: '#ef444408', textAlign: 'center' }}><div style={{ fontSize: '16px', fontWeight: '600', color: '#ef4444' }}>{d.bearWin.toFixed(0)}%</div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Bear</div></div>
              <div style={{ padding: '10px', borderRadius: '8px', background: '#eab30808', textAlign: 'center' }}><div style={{ fontSize: '16px', fontWeight: '600', color: '#eab308' }}>{d.sideWin.toFixed(0)}%</div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Side</div></div>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <div style={{ flex: 1, padding: '12px', borderRadius: '8px', background: '#f9731608', border: '1px solid #f9731620' }}><div style={{ fontSize: '14px', fontWeight: '600', color: '#f97316' }}>{d.aggWin.toFixed(1)}%</div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Win when aggressive</div></div>
              <div style={{ flex: 1, padding: '12px', borderRadius: '8px', background: '#3b82f608', border: '1px solid #3b82f620' }}><div style={{ fontSize: '14px', fontWeight: '600', color: '#3b82f6' }}>{d.conWin.toFixed(1)}%</div><div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Win when conservative</div></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function WhatIf({ all, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const [strategy, setStrategy] = useState('Signal Follower')
  const [startRound, setStartRound] = useState(30)

  const sim = useMemo(() => {
    const maxR = 90, init = 250000 * all.length
    let simCap = init, bhCap = init
    const hist = []
    for (let r = startRound; r <= maxR; r++) {
      let rRet = 0, bhRet = 0, cnt = 0
      all.forEach(({ det }) => { const rd = det?.rounds?.[r - 1]; if (rd) { rRet += rd.round_results?.returns_pct?.[strategy] || 0; bhRet += (rd.market?.daily_return || 0) * 100; cnt++ } })
      if (cnt > 0) { rRet /= cnt; bhRet /= cnt }
      simCap *= (1 + rRet / 100); bhCap *= (1 + bhRet / 100)
      hist.push({ r, sim: simCap, bh: bhCap })
    }
    return { hist, ret: ((simCap / init) - 1) * 100, bhRet: ((bhCap / init) - 1) * 100, alpha: ((simCap / init) - 1) * 100 - ((bhCap / init) - 1) * 100, simCap, bhCap, init }
  }, [all, strategy, startRound])

  const w = 450, ht = 160, p = { t: 20, r: 20, b: 25, l: 55 }
  const cw = w - p.l - p.r, ch = ht - p.t - p.b
  const allV = sim.hist.flatMap(d => [d.sim, d.bh])
  const maxY = Math.max(...allV) * 1.05, minY = Math.min(...allV) * 0.95
  const x = r => p.l + ((r - startRound) / (90 - startRound)) * cw, y = v => p.t + ch - ((v - minY) / (maxY - minY)) * ch

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div><div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>What-If Simulator</div><div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>What if you started following <span style={{ color: S[strategy].c, fontWeight: '500' }}>{strategy}</span> at round {startRound}?</div></div>
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}` }}>
        <div style={{ display: 'flex', gap: '24px' }}>
          <div style={{ width: '180px' }}>
            <div style={{ fontSize: '10px', fontWeight: '500', color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Strategy</div>
            {O.map(s => (
              <button key={s} onClick={() => setStrategy(s)} style={{ width: '100%', padding: '10px 12px', marginBottom: '6px', borderRadius: '8px', border: strategy === s ? `2px solid ${S[s].c}40` : `1px solid ${bdr}`, background: strategy === s ? `${S[s].c}15` : 'transparent', color: strategy === s ? S[s].c : 'var(--text-secondary)', fontWeight: strategy === s ? '500' : '400', fontSize: '12px', cursor: 'pointer', textAlign: 'left', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[s].c }} />{S[s].s}
              </button>
            ))}
            <div style={{ fontSize: '10px', fontWeight: '500', color: 'var(--text-muted)', marginTop: '14px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Start: R{startRound}</div>
            <input type="range" min="1" max="60" value={startRound} onChange={e => setStartRound(+e.target.value)} style={{ width: '100%', accentColor: S[strategy].c }} />
          </div>
          <div style={{ flex: 1 }}>
            <svg viewBox={`0 0 ${w} ${ht}`} style={{ width: '100%' }}>
              {[minY, (minY + maxY) / 2, maxY].map(v => (<g key={v}><line x1={p.l} y1={y(v)} x2={w - p.r} y2={y(v)} stroke={bdr} strokeDasharray="3,3" /><text x={p.l - 6} y={y(v)} fill={dark ? '#525252' : '#a1a1aa'} fontSize="8" textAnchor="end" dominantBaseline="middle">{fmtK(v)}</text></g>))}
              <path d={sim.hist.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(d.r)},${y(d.bh)}`).join(' ')} fill="none" stroke="#71717a" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
              <path d={sim.hist.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(d.r)},${y(d.sim)}`).join(' ')} fill="none" stroke={S[strategy].c} strokeWidth="2" />
              <circle cx={x(startRound)} cy={y(sim.hist[0]?.sim || 0)} r="4" fill={S[strategy].c} />
              <circle cx={x(90)} cy={y(sim.simCap)} r="4" fill={S[strategy].c} />
            </svg>
            <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
              <div style={{ flex: 1, padding: '14px', borderRadius: '10px', background: sim.ret >= 0 ? '#22c55e08' : '#ef444408', border: `1px solid ${sim.ret >= 0 ? '#22c55e20' : '#ef444420'}` }}>
                <div style={{ fontSize: '22px', fontWeight: '600', color: sim.ret >= 0 ? '#22c55e' : '#ef4444' }}>{fmtPct(sim.ret)}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Simulated return</div>
              </div>
              <div style={{ flex: 1, padding: '14px', borderRadius: '10px', background: sim.alpha >= 0 ? '#22c55e08' : '#ef444408', border: `1px solid ${sim.alpha >= 0 ? '#22c55e20' : '#ef444420'}` }}>
                <div style={{ fontSize: '22px', fontWeight: '600', color: sim.alpha >= 0 ? '#22c55e' : '#ef4444' }}>{sim.alpha >= 0 ? '+' : ''}{sim.alpha.toFixed(2)}%</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Alpha vs B&H</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function HeadToHead({ all, dark }) {
  const bdr = dark ? '#27272a' : '#e4e4e7'
  const bg2 = dark ? '#18181b' : '#f4f4f5'
  const [s1, setS1] = useState('Signal Follower')
  const [s2, setS2] = useState('Defector')

  const matrix = useMemo(() => {
    const m = {}
    O.forEach(strat1 => { m[strat1] = {}; O.forEach(strat2 => {
      if (strat1 === strat2) { m[strat1][strat2] = null; return }
      let wins = 0, total = 0
      all.forEach(({ det }) => { det?.rounds?.forEach(r => { const r1 = r.round_results?.returns_pct?.[strat1] || 0, r2 = r.round_results?.returns_pct?.[strat2] || 0; total++; if (r1 > r2) wins++ }) })
      m[strat1][strat2] = total > 0 ? (wins / total * 100) : 50
    })})
    return m
  }, [all])

  const comparison = useMemo(() => {
    let w1 = 0, w2 = 0, ties = 0
    const tickerResults = []
    all.forEach(({ det, sum, t }) => {
      let s1WinsT = 0, s2WinsT = 0
      det?.rounds?.forEach(r => { const r1 = r.round_results?.returns_pct?.[s1] || 0, r2 = r.round_results?.returns_pct?.[s2] || 0; if (r1 > r2) { w1++; s1WinsT++ } else if (r2 > r1) { w2++; s2WinsT++ } else ties++ })
      const ret1 = sum?.total_returns_pct?.[s1] || 0, ret2 = sum?.total_returns_pct?.[s2] || 0
      tickerResults.push({ ticker: t, ret1, ret2, winner: ret1 > ret2 ? s1 : ret2 > ret1 ? s2 : 'tie', s1WinsT, s2WinsT })
    })
    return { w1, w2, ties, total: w1 + w2 + ties, tickerResults }
  }, [all, s1, s2])

  const s1Pct = (comparison.w1 / comparison.total) * 100, s2Pct = (comparison.w2 / comparison.total) * 100

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div><div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Head-to-Head Analysis</div><div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Compare strategies across {(all.length * 90).toLocaleString()} rounds</div></div>
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}`, marginBottom: '8px' }}>
        <div style={{ fontSize: '12px', fontWeight: '500', color: 'var(--text-primary)', marginBottom: '12px' }}>Conflict Matrix</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'auto repeat(4, 1fr)', gap: '4px' }}>
          <div />
          {O.map(s => (<div key={s} style={{ textAlign: 'center', padding: '6px' }}><div style={{ width: '8px', height: '8px', borderRadius: '2px', background: S[s].c, margin: '0 auto' }} /></div>))}
          {O.map(strat1 => (
            <React.Fragment key={strat1}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 10px', fontSize: '11px', fontWeight: '500', color: S[strat1].c }}><div style={{ width: '6px', height: '6px', borderRadius: '1px', background: S[strat1].c }} />{S[strat1].s}</div>
              {O.map(strat2 => {
                const val = matrix[strat1][strat2]
                if (val === null) return <div key={strat2} style={{ background: bg2, borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '10px', padding: '10px' }}>—</div>
                const win = val > 50
                return (
                  <div key={strat2} onClick={() => { setS1(strat1); setS2(strat2) }} style={{ padding: '10px 6px', borderRadius: '6px', textAlign: 'center', background: win ? `rgba(34,197,94,${(val - 50) / 50 * 0.2})` : `rgba(239,68,68,${(50 - val) / 50 * 0.2})`, cursor: 'pointer', border: (strat1 === s1 && strat2 === s2) ? '2px solid var(--text-primary)' : '2px solid transparent' }}>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: win ? '#22c55e' : '#ef4444' }}>{val.toFixed(0)}%</div>
                  </div>
                )
              })}
            </React.Fragment>
          ))}
        </div>
        <div style={{ marginTop: '10px', fontSize: '10px', color: 'var(--text-muted)' }}>Click a cell to compare</div>
      </div>
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}` }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ width: '14px', height: '14px', borderRadius: '3px', background: S[s1].c, margin: '0 auto 8px' }} />
            <div style={{ fontSize: '13px', fontWeight: '600', color: S[s1].c }}>{s1}</div>
            <div style={{ fontSize: '28px', fontWeight: '600', color: 'var(--text-primary)', marginTop: '8px' }}>{comparison.w1}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>wins ({s1Pct.toFixed(1)}%)</div>
          </div>
          <div style={{ padding: '16px', textAlign: 'center' }}><div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--text-muted)' }}>VS</div><div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>{comparison.total} rounds</div></div>
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ width: '14px', height: '14px', borderRadius: '3px', background: S[s2].c, margin: '0 auto 8px' }} />
            <div style={{ fontSize: '13px', fontWeight: '600', color: S[s2].c }}>{s2}</div>
            <div style={{ fontSize: '28px', fontWeight: '600', color: 'var(--text-primary)', marginTop: '8px' }}>{comparison.w2}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>wins ({s2Pct.toFixed(1)}%)</div>
          </div>
        </div>
        <div style={{ marginTop: '16px', height: '8px', borderRadius: '4px', overflow: 'hidden', display: 'flex', background: bdr }}>
          <div style={{ width: `${s1Pct}%`, background: S[s1].c, transition: 'width 0.3s' }} />
          <div style={{ width: `${s2Pct}%`, background: S[s2].c, transition: 'width 0.3s' }} />
        </div>
      </div>
      <div style={{ background: dark ? '#0a0a0a' : '#fff', borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', borderBottom: `1px solid ${bdr}`, fontSize: '12px', fontWeight: '500', color: 'var(--text-primary)' }}>Per-Ticker Results</div>
        <div style={{ maxHeight: '240px', overflowY: 'auto' }}>
          {comparison.tickerResults.map(r => (
            <div key={r.ticker} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 16px', borderBottom: `1px solid ${bdr}` }}>
              <span style={{ fontWeight: '500', color: 'var(--text-primary)', width: '45px', fontSize: '12px' }}>{r.ticker}</span>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '11px', fontWeight: '500', color: S[s1].c, width: '55px' }}>{fmtPct(r.ret1)}</span>
                <div style={{ flex: 1, height: '5px', background: bdr, borderRadius: '2px', overflow: 'hidden', display: 'flex' }}>
                  <div style={{ width: `${(r.s1WinsT / (r.s1WinsT + r.s2WinsT || 1)) * 100}%`, background: S[s1].c }} />
                  <div style={{ width: `${(r.s2WinsT / (r.s1WinsT + r.s2WinsT || 1)) * 100}%`, background: S[s2].c }} />
                </div>
                <span style={{ fontSize: '11px', fontWeight: '500', color: S[s2].c, width: '55px', textAlign: 'right' }}>{fmtPct(r.ret2)}</span>
              </div>
              <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: r.winner === s1 ? S[s1].c : r.winner === s2 ? S[s2].c : '#71717a' }} />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}