/**
 * PipelineHistoryPage.jsx - Fixed layout + proper dark/light theming
 */

import React, { useState, useEffect, useMemo } from 'react'
import { useTheme } from '../App'
import { loadWorkflowOutput, ACTION_CONFIG, TICKER_COLORS } from '../data'

// ============================================================
// CONSTANTS
// ============================================================

const PHASES = [
  { num: 1, name: 'Market Analysis', icon: '📊' },
  { num: 2, name: 'Bull/Bear Research', icon: '⚔️' },
  { num: 3, name: 'Synthesis', icon: '⚖️' },
  { num: 4, name: 'Risk Evaluation', icon: '🎯' },
  { num: 5, name: 'Final Decision', icon: '✅' }
]

const VERDICT_CONFIG = {
  'BUY': { emoji: '📈', color: '#10b981', label: 'BUY' },
  'STRONG BUY': { emoji: '🚀', color: '#10b981', label: 'STRONG BUY' },
  'HOLD': { emoji: '⏸️', color: '#f59e0b', label: 'HOLD' },
  'SELL': { emoji: '📉', color: '#ef4444', label: 'SELL' },
  'REJECT': { emoji: '🚫', color: '#ef4444', label: 'REJECT' },
  'AVOID': { emoji: '⛔', color: '#dc2626', label: 'AVOID' },
  'APPROVE': { emoji: '✅', color: '#10b981', label: 'APPROVE' },
  'MODIFY': { emoji: '✏️', color: '#3b82f6', label: 'MODIFY' },
}

const getVerdictConfig = (verdict) => {
  return VERDICT_CONFIG[verdict] || { emoji: '❓', color: '#6b7280', label: verdict || 'N/A' }
}

// ============================================================
// ICONS (SVG)
// ============================================================

const Icons = {
  X: () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18M6 6l12 12" /></svg>,
  Search: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>,
  Loader: () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: 'spin 1s linear infinite' }}><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>,
  Clock: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>,
  FileText: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>,
}

// ============================================================
// PHASE COMPONENTS
// ============================================================

function MarketContextPhase({ data }) {
  const ctx = data.market_context || {}
  const price = ctx.price_data || {}
  const isUp = (price.daily_return || 0) >= 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {[['Open', price.open], ['Close', price.close], ['High', price.high], ['Low', price.low]].map(([label, val], i) => (
          <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: '8px', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>{label}</div>
            <div style={{ fontSize: '16px', fontWeight: '600', color: 'var(--text-primary)' }}>${(val || 0).toFixed(2)}</div>
          </div>
        ))}
      </div>
      <div style={{ background: 'var(--bg-tertiary)', borderRadius: '8px', padding: '16px', display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ color: 'var(--text-secondary)' }}>Daily Return</span>
        <span style={{ fontSize: '20px', fontWeight: '700', color: isUp ? 'var(--success)' : 'var(--danger)' }}>
          {isUp ? '+' : ''}{((price.daily_return || 0) * 100).toFixed(2)}%
        </span>
      </div>
      <div style={{ background: 'var(--bg-tertiary)', borderRadius: '8px', padding: '16px' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>VOLUME</div>
        <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--text-primary)' }}>
          {price.volume ? (price.volume / 1e6).toFixed(1) + 'M' : 'N/A'}
        </div>
      </div>
    </div>
  )
}

function ThesisCard({ title, icon, thesis, isBull }) {
  const [open, setOpen] = useState(false)
  const color = isBull ? 'var(--success)' : 'var(--danger)'
  const bg = isBull ? 'var(--success-bg)' : 'var(--danger-bg)'
  const border = isBull ? 'var(--success-border)' : 'var(--danger-border)'
  const signals = isBull ? thesis?.key_bullish_signals : thesis?.key_risk_signals
  const pct = isBull ? thesis?.risk_reward?.upside_pct : thesis?.risk_assessment?.downside_pct

  return (
    <div style={{ background: bg, borderRadius: '12px', border: `1px solid ${border}`, overflow: 'hidden' }}>
      <div onClick={() => setOpen(!open)} style={{ padding: '16px', cursor: 'pointer' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '24px' }}>{icon}</span>
            <div>
              <div style={{ fontWeight: '700', color }}>{title}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{thesis?.conviction?.level || 'N/A'} Conviction</div>
            </div>
          </div>
          {pct != null && <span style={{ padding: '4px 12px', background: `${color}20`, color, borderRadius: '6px', fontWeight: '700', fontSize: '13px' }}>{isBull ? '+' : '-'}{Math.abs(pct).toFixed(1)}%</span>}
        </div>
        <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          {thesis?.core_thesis}
        </p>
      </div>
      {open && signals?.length > 0 && (
        <div style={{ padding: '16px', borderTop: `1px solid ${border}` }}>
          {signals.slice(0, 4).map((s, i) => (
            <div key={i} style={{ background: 'var(--bg-primary)', borderRadius: '6px', padding: '10px', marginBottom: '8px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-primary)' }}>{s.signal}</div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>{s.source} • {isBull ? s.strength : s.severity}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function BullBearPhase({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
      <ThesisCard title="Bull Thesis" icon="🐂" thesis={data.bull_thesis} isBull={true} />
      <ThesisCard title="Bear Thesis" icon="🐻" thesis={data.bear_thesis} isBull={false} />
    </div>
  )
}

function SynthesisPhase({ data }) {
  const syn = data.research_synthesis || {}
  const winner = syn.debate_winner?.winner || 'unknown'
  const isBull = winner === 'bull'
  const color = isBull ? 'var(--success)' : 'var(--danger)'
  const bg = isBull ? 'var(--success-bg)' : 'var(--danger-bg)'
  const probs = syn.probabilities || {}

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ background: bg, borderRadius: '16px', padding: '24px', textAlign: 'center', border: `2px solid ${color}40` }}>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>DEBATE WINNER</div>
        <div style={{ fontSize: '32px', fontWeight: '800', color }}>{isBull ? '🐂 BULL' : '🐻 BEAR'}</div>
        <p style={{ margin: '12px 0 0', fontSize: '13px', color: 'var(--text-secondary)' }}>{syn.debate_winner?.reasoning}</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
        {[['Bull', probs.bull_case, 'var(--success)'], ['Base', probs.base_case, 'var(--text-muted)'], ['Bear', probs.bear_case, 'var(--danger)']].map(([l, v, c], i) => (
          <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
            <div style={{ fontSize: '11px', color: c }}>{l} Case</div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: c }}>{(v || 0).toFixed(0)}%</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function EvaluatorsPhase({ data }) {
  const evals = [['Aggressive', '🔥', data.aggressive_eval], ['Neutral', '⚖️', data.neutral_eval], ['Conservative', '🛡️', data.conservative_eval]]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
      {evals.map(([name, icon, ev], i) => {
        const stance = ev?.recommendation || ev?.stance || 'N/A'
        const cfg = getVerdictConfig(stance)
        return (
          <div key={i} style={{ background: 'var(--bg-tertiary)', borderRadius: '12px', padding: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
              <span style={{ fontSize: '24px' }}>{icon}</span>
              <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{name}</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span style={{ padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '600', background: `${cfg.color}20`, color: cfg.color }}>{cfg.emoji} {stance}</span>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{(ev?.position_pct || 0).toFixed(0)}%</span>
            </div>
            {ev?.reasoning && <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{ev.reasoning}</p>}
          </div>
        )
      })}
    </div>
  )
}

function FinalDecisionPhase({ data }) {
  const dec = data.risk_decision || {}
  const verdict = dec.verdict || 'HOLD'
  const cfg = getVerdictConfig(verdict)

  return (
    <div style={{ maxWidth: '500px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ background: `${cfg.color}15`, borderRadius: '16px', border: `2px solid ${cfg.color}40`, padding: '32px', textAlign: 'center' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>FINAL VERDICT</div>
        <div style={{ fontSize: '42px', fontWeight: '800', color: cfg.color }}>{cfg.emoji} {verdict}</div>
        <div style={{ color: 'var(--text-secondary)', marginTop: '8px', fontSize: '14px' }}>{dec.confidence} Confidence</div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ background: 'var(--bg-tertiary)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Position</div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: 'var(--text-primary)' }}>${(dec.final_position_dollars || 0).toLocaleString()}</div>
        </div>
        <div style={{ background: 'var(--bg-tertiary)', borderRadius: '12px', padding: '16px', textAlign: 'center' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Stop Loss</div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: 'var(--danger)' }}>-{dec.stop_loss_pct || 15}%</div>
        </div>
      </div>
      {dec.reasoning && (
        <div style={{ background: 'var(--bg-tertiary)', borderRadius: '12px', padding: '16px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>REASONING</div>
          <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{dec.reasoning}</p>
        </div>
      )}
    </div>
  )
}

// ============================================================
// MODAL
// ============================================================

function RunDetailModal({ run, onClose, isDark }) {
  const [phase, setPhase] = useState(5)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    loadWorkflowOutput(run.ticker, run.folder)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [run])

  const verdict = data?.risk_decision?.verdict || run.verdict || 'HOLD'
  const cfg = getVerdictConfig(verdict)

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
      <div style={{ background: 'var(--bg-card)', borderRadius: '16px', border: '1px solid var(--border-primary)', width: '100%', maxWidth: '900px', maxHeight: '90vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-primary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '28px', fontWeight: '800', color: TICKER_COLORS[run.ticker] || 'var(--text-primary)' }}>{run.ticker}</span>
              <span style={{ padding: '6px 14px', borderRadius: '8px', fontSize: '13px', fontWeight: '700', background: `${cfg.color}20`, color: cfg.color }}>{cfg.emoji} {verdict}</span>
            </div>
            <div style={{ marginTop: '6px', fontSize: '13px', color: 'var(--text-muted)' }}>
              {run.date} • Sample #{run.sample} • Day {run.day}
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'var(--bg-tertiary)', border: 'none', borderRadius: '10px', padding: '10px', cursor: 'pointer', color: 'var(--text-secondary)' }}><Icons.X /></button>
        </div>

        <div style={{ display: 'flex', gap: '8px', padding: '14px 24px', borderBottom: '1px solid var(--border-primary)', background: 'var(--bg-secondary)', overflowX: 'auto' }}>
          {PHASES.map(p => (
            <button key={p.num} onClick={() => setPhase(p.num)} style={{
              padding: '10px 18px', borderRadius: '10px', border: 'none', cursor: 'pointer', fontSize: '13px', fontWeight: '600', whiteSpace: 'nowrap', transition: 'all 0.2s',
              background: phase === p.num ? (isDark ? '#fafafa' : '#18181b') : 'var(--bg-tertiary)',
              color: phase === p.num ? (isDark ? '#18181b' : '#fafafa') : 'var(--text-secondary)'
            }}>
              {p.icon} {p.name}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '24px' }}>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', gap: '12px', color: 'var(--text-muted)' }}>
              <Icons.Loader /> Loading...
            </div>
          ) : !data ? (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>❌</div>
              <div>Failed to load workflow data</div>
            </div>
          ) : (
            <>
              {phase === 1 && <MarketContextPhase data={data} />}
              {phase === 2 && <BullBearPhase data={data} />}
              {phase === 3 && <SynthesisPhase data={data} />}
              {phase === 4 && <EvaluatorsPhase data={data} />}
              {phase === 5 && <FinalDecisionPhase data={data} />}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ============================================================
// WORKFLOW CARD
// ============================================================

function WorkflowCard({ workflow, onClick }) {
  const cfg = getVerdictConfig(workflow.verdict)
  const hasPosition = workflow.position && workflow.position > 0

  return (
    <div
      onClick={onClick}
      style={{
        background: 'var(--bg-card)',
        borderRadius: '12px',
        border: '1px solid var(--border-primary)',
        padding: '16px',
        cursor: 'pointer',
        transition: 'all 0.2s',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = cfg.color
        e.currentTarget.style.transform = 'translateY(-2px)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border-primary)'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '18px', fontWeight: '700', color: TICKER_COLORS[workflow.ticker] || 'var(--text-primary)' }}>
          {workflow.ticker}
        </span>
        <span style={{ padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '600', background: `${cfg.color}15`, color: cfg.color, display: 'flex', alignItems: 'center', gap: '4px' }}>
          {cfg.emoji} {workflow.verdict}
        </span>
      </div>
      <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{workflow.date}</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px', borderTop: '1px solid var(--border-primary)' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Sample #{workflow.sample}</span>
        {hasPosition ? (
          <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--success)' }}>${workflow.position.toLocaleString()}</span>
        ) : (
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>No position</span>
        )}
      </div>
    </div>
  )
}

// ============================================================
// MAIN PAGE
// ============================================================

export default function PipelineHistoryPage() {
  const { isDark } = useTheme()
  const [manifest, setManifest] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [ticker, setTicker] = useState('All')
  const [verdictFilter, setVerdictFilter] = useState('All')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState(null)
  const [sort, setSort] = useState('desc')
  const [page, setPage] = useState(1)
  const pageSize = 30

  useEffect(() => {
    fetch('/data/workflows/manifest.json')
      .then(r => {
        if (!r.ok) throw new Error('Manifest not found')
        return r.json()
      })
      .then(setManifest)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tickers = useMemo(() => [...new Set(manifest.map(w => w.ticker))].sort(), [manifest])
  const verdicts = useMemo(() => [...new Set(manifest.map(w => w.verdict))].sort(), [manifest])

  const filtered = useMemo(() => {
    let list = manifest
    if (ticker !== 'All') list = list.filter(w => w.ticker === ticker)
    if (verdictFilter !== 'All') list = list.filter(w => w.verdict === verdictFilter)
    if (search) {
      const q = search.toUpperCase()
      list = list.filter(w => w.ticker.includes(q) || w.date.includes(search))
    }
    return list.sort((a, b) => sort === 'desc' ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date))
  }, [manifest, ticker, verdictFilter, search, sort])

  const paginated = filtered.slice(0, page * pageSize)
  const hasMore = paginated.length < filtered.length

  const stats = useMemo(() => {
    const verdictCounts = {}
    filtered.forEach(w => {
      verdictCounts[w.verdict] = (verdictCounts[w.verdict] || 0) + 1
    })
    return verdictCounts
  }, [filtered])

  // Common select style (matches PipelinePage)
  const selectStyle = {
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid var(--border-primary)',
    background: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    fontSize: '12px',
    cursor: 'pointer',
    outline: 'none'
  }

  // Common button style
  const buttonStyle = {
    padding: '8px 12px',
    borderRadius: '8px',
    border: '1px solid var(--border-primary)',
    background: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    fontSize: '12px',
    cursor: 'pointer'
  }

  if (loading) {
    return (
      <div style={{ height: '100vh', background: 'var(--bg-secondary)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', color: 'var(--text-muted)' }}>
        <Icons.Loader /> Loading workflows...
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ height: '100vh', background: 'var(--bg-secondary)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px', padding: '40px' }}>
        <div style={{ fontSize: '48px' }}>📂</div>
        <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--text-primary)' }}>Manifest not found</div>
        <div style={{ color: 'var(--text-muted)', textAlign: 'center', maxWidth: '400px' }}>
          Run <code style={{ background: 'var(--bg-tertiary)', padding: '2px 8px', borderRadius: '4px' }}>python generate_manifest.py</code> first.
        </div>
      </div>
    )
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)' }}>
      {/* Header */}
      <header style={{
        background: 'var(--bg-primary)',
        borderBottom: '1px solid var(--border-primary)',
        padding: '12px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
            Pipeline History
          </h1>
          <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
            {filtered.length.toLocaleString()} workflows • {tickers.length} tickers
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Search */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'var(--bg-primary)',
            borderRadius: '8px',
            padding: '8px 12px',
            border: '1px solid var(--border-primary)'
          }}>
            <Icons.Search />
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(1) }}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-primary)',
                fontSize: '12px',
                width: '120px',
                outline: 'none'
              }}
            />
          </div>

          {/* Stats badge */}
          <span style={{
            padding: '5px 10px',
            background: 'var(--bg-tertiary)',
            borderRadius: '6px',
            color: 'var(--text-secondary)',
            fontSize: '12px'
          }}>
            {manifest.length} Total
          </span>
        </div>
      </header>

      {/* Filters Bar */}
      <div style={{
        padding: '12px 20px',
        borderBottom: '1px solid var(--border-primary)',
        display: 'flex',
        gap: '10px',
        flexWrap: 'wrap',
        alignItems: 'center',
        background: 'var(--bg-primary)',
        flexShrink: 0
      }}>
        <select
          value={ticker}
          onChange={e => { setTicker(e.target.value); setPage(1) }}
          style={selectStyle}
        >
          <option value="All">All Tickers</option>
          {tickers.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <select
          value={verdictFilter}
          onChange={e => { setVerdictFilter(e.target.value); setPage(1) }}
          style={selectStyle}
        >
          <option value="All">All Decisions</option>
          {verdicts.map(v => {
            const cfg = getVerdictConfig(v)
            return <option key={v} value={v}>{cfg.emoji} {v}</option>
          })}
        </select>

        <button
          onClick={() => setSort(s => s === 'desc' ? 'asc' : 'desc')}
          style={buttonStyle}
        >
          {sort === 'desc' ? '↓ Newest' : '↑ Oldest'}
        </button>

        {(ticker !== 'All' || verdictFilter !== 'All' || search) && (
          <button
            onClick={() => { setTicker('All'); setVerdictFilter('All'); setSearch(''); setPage(1) }}
            style={{ ...buttonStyle, background: 'transparent', color: 'var(--text-muted)' }}
          >
            ✕ Clear
          </button>
        )}

        {/* Verdict Stats */}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
          {Object.entries(stats).slice(0, 4).map(([verdict, count]) => {
            const cfg = getVerdictConfig(verdict)
            return (
              <span key={verdict} style={{
                padding: '4px 8px',
                background: `${cfg.color}15`,
                borderRadius: '6px',
                fontSize: '11px',
                color: cfg.color,
                fontWeight: '600'
              }}>
                {cfg.emoji} {count}
              </span>
            )
          })}
        </div>
      </div>

      {/* Scrollable Content */}
      <main style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '16px'
          }}>
            {paginated.map((w, i) => (
              <WorkflowCard
                key={`${w.ticker}-${w.sample}-${i}`}
                workflow={w}
                onClick={() => setSelected(w)}
              />
            ))}
          </div>

          {hasMore && (
            <div style={{ textAlign: 'center', marginTop: '32px' }}>
              <button
                onClick={() => setPage(p => p + 1)}
                style={{
                  padding: '12px 32px',
                  borderRadius: '10px',
                  border: '1px solid var(--border-primary)',
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}
              >
                Load More ({filtered.length - paginated.length} remaining)
              </button>
            </div>
          )}

          {filtered.length === 0 && (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔍</div>
              <div style={{ fontSize: '16px', fontWeight: '500' }}>No workflows found</div>
              <div style={{ fontSize: '13px', marginTop: '8px' }}>Try adjusting your filters</div>
            </div>
          )}
        </div>
      </main>

      {selected && <RunDetailModal run={selected} onClose={() => setSelected(null)} isDark={isDark} />}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}