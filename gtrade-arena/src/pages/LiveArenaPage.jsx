import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useTheme } from '../App';
import TournamentChat from '../components/chat/llm_arena_rag';

// Model configuration
const MODELS = {
  'GPT-4o-mini': { color: '#10b981', short: 'GPT-4o', desc: 'OpenAI flagship mini', icon: '🤖' },
  'Llama-3.3-70B': { color: '#6366f1', short: 'Llama-70B', desc: 'Meta large model', icon: '🦙' },
  'Llama-4-Maverick': { color: '#f59e0b', short: 'Maverick', desc: 'Llama 4 MoE', icon: '🎯' },
  'Kimi-K2': { color: '#06b6d4', short: 'Kimi', desc: 'Moonshot reasoning', icon: '🌙' },
  'Qwen3-32B': { color: '#ec4899', short: 'Qwen', desc: 'Alibaba model', icon: '🐼' },
  'GPT-OSS-120B': { color: '#8b5cf6', short: 'GPT-OSS', desc: 'Open source GPT', icon: '⚡' },
  'Allam-2-7B': { color: '#f97316', short: 'Allam', desc: 'Multilingual model', icon: '🌍' },
};

const MODEL_FILES = {
  'GPT-4o-mini': 'GPT_4o_mini.json',
  'Llama-3.3-70B': 'Llama_3_3_70B.json',
  'Llama-4-Maverick': 'Llama_4_Maverick.json',
  'Kimi-K2': 'Kimi_K2.json',
  'Qwen3-32B': 'Qwen3_32B.json',
  'GPT-OSS-120B': 'GPT_OSS_120B.json',
  'Allam-2-7B': 'Allam_2_7B.json',
};

// Helpers
const fmtPct = n => `${n >= 0 ? '+' : ''}${(n || 0).toFixed(2)}%`;
const fmtK = n => n >= 1e6 ? `$${(n / 1e6).toFixed(2)}M` : n >= 1000 ? `$${(n / 1e3).toFixed(1)}K` : `$${(n || 0).toFixed(0)}`;
const fmtNum = n => (n || 0).toLocaleString();

export default function LLMArenaPage() {
  const { isDark } = useTheme();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mainTab, setMainTab] = useState('overview');

  const [raceFrame, setRaceFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const raceInterval = useRef(null);

  const [modelA, setModelA] = useState('Llama-3.3-70B');
  const [modelB, setModelB] = useState('Qwen3-32B');

  const [highlightedModels, setHighlightedModels] = useState(new Set(Object.keys(MODELS)));
  const [hoveredModel, setHoveredModel] = useState(null);

  const [selectedModel, setSelectedModel] = useState(null);
  const [modelTrades, setModelTrades] = useState({});
  const [loadingTrades, setLoadingTrades] = useState(false);
  const [expandedTrade, setExpandedTrade] = useState(null);

  // Use CSS variables from App theme (same as Market page)
  const borderColor = 'var(--border-primary)';
  const mainBg = 'var(--bg-secondary)';
  const cardBg = 'var(--bg-card)';
  const secondaryBg = 'var(--bg-tertiary)';
  const textPrimary = 'var(--text-primary)';
  const textMuted = 'var(--text-muted)';

  useEffect(() => {
    fetch('/data/llm_arena/llm_arena_summary.json')
      .then(r => { if (!r.ok) throw new Error('Failed to load'); return r.json(); })
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  useEffect(() => {
    if (isPlaying && data) {
      raceInterval.current = setInterval(() => {
        setRaceFrame(f => {
          if (f >= (data.equity_curves?.[Object.keys(data.equity_curves)[0]]?.length || 385) - 1) {
            setIsPlaying(false);
            return f;
          }
          return f + 1;
        });
      }, 50);
    }
    return () => clearInterval(raceInterval.current);
  }, [isPlaying, data]);

  const loadModelTrades = useCallback(async (model) => {
    if (!model || modelTrades[model]) return;
    setLoadingTrades(true);
    try {
      const file = MODEL_FILES[model];
      const r = await fetch(`/data/llm_arena/model_trades/${file}`);
      if (!r.ok) throw new Error('Failed');
      const d = await r.json();
      setModelTrades(prev => ({ ...prev, [model]: d }));
    } catch (e) {
      console.error(e);
    }
    setLoadingTrades(false);
  }, [modelTrades]);

  useEffect(() => {
    if (selectedModel) loadModelTrades(selectedModel);
  }, [selectedModel, loadModelTrades]);

  const toggleModel = (model) => {
    setHighlightedModels(prev => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  };

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: mainBg, color: textPrimary }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>🏟️</div>
        <div style={{ fontSize: 18 }}>Loading Arena Data...</div>
      </div>
    </div>
  );

  if (error) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: mainBg, color: textPrimary }}>
      <div style={{ textAlign: 'center', padding: 32, background: cardBg, borderRadius: 12, border: `1px solid ${borderColor}` }}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>⚠️</div>
        <div style={{ fontSize: 18, marginBottom: 8 }}>Failed to Load Data</div>
        <div style={{ color: textMuted }}>{error}</div>
      </div>
    </div>
  );

  const { leaderboard, equity_curves, model_stats, ticker_stats, recent_trades } = data;
  const top3 = leaderboard?.slice(0, 3) || [];
  const medals = ['🥇', '🥈', '🥉'];
  const maxFrames = equity_curves?.[Object.keys(equity_curves)[0]]?.length || 385;

  const mainTabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'race', label: 'Race' },
    { id: 'dna', label: 'Model DNA' },
    { id: 'tickers', label: 'Tickers' },
    { id: 'h2h', label: 'Head to Head' },
  ];

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: mainBg, color: textPrimary, overflow: 'hidden' }}>
      <style>{`.hide-scroll::-webkit-scrollbar{display:none}.hide-scroll{-ms-overflow-style:none;scrollbar-width:none}`}</style>

      {/* Compact Header */}
      <div style={{ padding: '12px 20px', borderBottom: `1px solid ${borderColor}`, background: cardBg, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>LLM Trading Arena</h1>
            <span style={{ fontSize: 12, color: textMuted }}>7 models · {data.rounds} rounds · $250K</span>
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            {top3.map((m, i) => (
              <div key={m.model} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 10px', background: secondaryBg, borderRadius: 6, fontSize: 12 }}>
                <span>{medals[i]}</span>
                <span style={{ fontWeight: 600 }}>{MODELS[m.model]?.short}</span>
                <span style={{ color: m.return_pct >= 0 ? '#10b981' : '#ef4444', fontWeight: 500 }}>{fmtPct(m.return_pct)}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 4, marginTop: 12 }}>
          {mainTabs.map(t => (
            <button key={t.id} onClick={() => setMainTab(t.id)} style={{
              padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
              background: mainTab === t.id ? (isDark ? '#27272a' : '#e4e4e7') : 'transparent',
              color: mainTab === t.id ? textPrimary : textMuted,
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      {/* Body - flex row, no scroll on parent */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Main Content - scrollable */}
        <main className="hide-scroll" style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
          <div style={{ maxWidth: 1000, margin: '0 auto' }}>
            {mainTab === 'overview' && (
              <OverviewTab
                data={data}
                isDark={isDark}
                colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }}
                highlightedModels={highlightedModels}
                toggleModel={toggleModel}
                hoveredModel={hoveredModel}
                setHoveredModel={setHoveredModel}
                onModelSelect={setSelectedModel}
              />
            )}
            {mainTab === 'race' && <RaceTab data={data} frame={raceFrame} setFrame={setRaceFrame} isPlaying={isPlaying} setIsPlaying={setIsPlaying} maxFrames={maxFrames} isDark={isDark} colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }} />}
            {mainTab === 'dna' && <DNATab data={data} isDark={isDark} colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }} onModelSelect={setSelectedModel} />}
            {mainTab === 'tickers' && <TickersTab data={data} isDark={isDark} colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }} />}
            {mainTab === 'h2h' && <H2HTab data={data} modelA={modelA} modelB={modelB} setModelA={setModelA} setModelB={setModelB} isDark={isDark} colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }} />}
          </div>
        </main>

        {/* Right Sidebar - fixed, internal scroll only */}
        <aside style={{ width: 340, borderLeft: `1px solid ${borderColor}`, background: cardBg, display: 'flex', flexDirection: 'column', flexShrink: 0, overflow: 'hidden' }}>
          {selectedModel ? (
            <ModelPortfolio
              model={selectedModel}
              data={data}
              modelTrades={modelTrades}
              loadingTrades={loadingTrades}
              onClose={() => setSelectedModel(null)}
              isDark={isDark}
              colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }}
            />
          ) : (
            <FeedSidebar
              trades={recent_trades}
              isDark={isDark}
              colors={{ borderColor, cardBg, secondaryBg, textPrimary, textMuted }}
              expandedTrade={expandedTrade}
              setExpandedTrade={setExpandedTrade}
              onModelClick={setSelectedModel}
              data={data}
            />
          )}
        </aside>
      </div>
    </div>
  );
}

// ============ OVERVIEW TAB ============
function OverviewTab({ data, isDark, colors, highlightedModels, toggleModel, hoveredModel, setHoveredModel, onModelSelect }) {
  const { borderColor, cardBg, secondaryBg, textPrimary, textMuted } = colors;
  const { leaderboard, equity_curves, model_stats } = data;
  const winner = leaderboard?.[0];
  const top4 = leaderboard?.slice(0, 4) || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Winner Banner */}
      {winner && (
        <div style={{
          background: `linear-gradient(135deg, ${MODELS[winner.model]?.color}22 0%, ${cardBg} 100%)`,
          border: `1px solid ${MODELS[winner.model]?.color}44`,
          borderRadius: 12, padding: 20, display: 'flex', alignItems: 'center', gap: 20
        }}>
          <div style={{ fontSize: 48 }}>🏆</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: textMuted, marginBottom: 4 }}>Tournament Champion</div>
            <div style={{ fontSize: 22, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>{MODELS[winner.model]?.icon}</span>
              <span>{winner.model}</span>
            </div>
            <div style={{ color: textMuted, fontSize: 13, marginTop: 4 }}>{MODELS[winner.model]?.desc}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{fmtK(winner.value)}</div>
            <div style={{ fontSize: 16, color: '#10b981', fontWeight: 600 }}>{fmtPct(winner.return_pct)}</div>
            <div style={{ fontSize: 12, color: textMuted }}>{winner.trades} trades</div>
          </div>
        </div>
      )}

      {/* Top 4 Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        {top4.map((m, i) => (
          <div
            key={m.model}
            onClick={() => onModelSelect(m.model)}
            style={{
              background: cardBg,
              border: `1px solid ${borderColor}`,
              borderRadius: 10,
              padding: 14,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = MODELS[m.model]?.color}
            onMouseLeave={e => e.currentTarget.style.borderColor = borderColor}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <span style={{ fontSize: 20 }}>{['🥇', '🥈', '🥉', '4️⃣'][i]}</span>
              <span style={{ fontSize: 18 }}>{MODELS[m.model]?.icon}</span>
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{MODELS[m.model]?.short}</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtK(m.value)}</div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 12, color: textMuted }}>
              <span style={{ color: m.return_pct >= 0 ? '#10b981' : '#ef4444' }}>{fmtPct(m.return_pct)}</span>
              <span>{m.trades} trades</span>
            </div>
          </div>
        ))}
      </div>

      {/* Equity Chart */}
      <div style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, padding: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Portfolio Growth</h3>
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={() => { Object.keys(MODELS).forEach(m => { if (!highlightedModels.has(m)) toggleModel(m); }); }}
              style={{ padding: '4px 10px', fontSize: 11, borderRadius: 4, border: `1px solid ${borderColor}`, background: 'transparent', color: textMuted, cursor: 'pointer' }}
            >All</button>
            <button
              onClick={() => { highlightedModels.forEach(m => toggleModel(m)); }}
              style={{ padding: '4px 10px', fontSize: 11, borderRadius: 4, border: `1px solid ${borderColor}`, background: 'transparent', color: textMuted, cursor: 'pointer' }}
            >None</button>
          </div>
        </div>
        <EquityChart
          curves={equity_curves}
          isDark={isDark}
          highlightedModels={highlightedModels}
          hoveredModel={hoveredModel}
          setHoveredModel={setHoveredModel}
        />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 14 }}>
          {Object.entries(MODELS).map(([name, cfg]) => {
            const isActive = highlightedModels.has(name);
            const isHovered = hoveredModel === name;
            return (
              <button
                key={name}
                onClick={() => toggleModel(name)}
                onMouseEnter={() => setHoveredModel(name)}
                onMouseLeave={() => setHoveredModel(null)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', fontSize: 11, borderRadius: 6,
                  border: `1px solid ${isActive ? cfg.color : borderColor}`,
                  background: isActive ? `${cfg.color}15` : 'transparent',
                  color: isActive ? (isHovered ? '#fff' : cfg.color) : textMuted,
                  opacity: isActive ? 1 : 0.5, cursor: 'pointer', transition: 'all 0.2s',
                }}
              >
                <div style={{ width: 12, height: 3, background: cfg.color, borderRadius: 2 }} />
                <span>{cfg.short}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Leaderboard Table */}
      <div style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, overflow: 'hidden' }}>
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${borderColor}` }}>
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Final Standings</h3>
          <p style={{ margin: '4px 0 0', fontSize: 11, color: textMuted }}>Click any row to view model portfolio</p>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: secondaryBg }}>
              <th style={{ padding: '10px 14px', textAlign: 'left', fontWeight: 600 }}>Rank</th>
              <th style={{ padding: '10px 14px', textAlign: 'left', fontWeight: 600 }}>Model</th>
              <th style={{ padding: '10px 14px', textAlign: 'right', fontWeight: 600 }}>Final Value</th>
              <th style={{ padding: '10px 14px', textAlign: 'right', fontWeight: 600 }}>Return</th>
              <th style={{ padding: '10px 14px', textAlign: 'right', fontWeight: 600 }}>Trades</th>
              <th style={{ padding: '10px 14px', textAlign: 'right', fontWeight: 600 }}>$/Trade</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard?.map((m, i) => {
              const stats = model_stats?.[m.model];
              const perTrade = stats?.total_trades > 0 ? (m.value - 250000) / stats.total_trades : 0;
              return (
                <tr
                  key={m.model}
                  onClick={() => onModelSelect(m.model)}
                  style={{ borderBottom: `1px solid ${borderColor}`, cursor: 'pointer', transition: 'background 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = `${MODELS[m.model]?.color}10`}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '11px 14px' }}>{['🥇', '🥈', '🥉'][i] || `#${i + 1}`}</td>
                  <td style={{ padding: '11px 14px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span>{MODELS[m.model]?.icon}</span>
                      <span style={{ fontWeight: 500 }}>{m.model}</span>
                    </div>
                  </td>
                  <td style={{ padding: '11px 14px', textAlign: 'right', fontWeight: 600 }}>{fmtK(m.value)}</td>
                  <td style={{ padding: '11px 14px', textAlign: 'right', color: m.return_pct >= 0 ? '#10b981' : '#ef4444', fontWeight: 500 }}>{fmtPct(m.return_pct)}</td>
                  <td style={{ padding: '11px 14px', textAlign: 'right' }}>{m.trades}</td>
                  <td style={{ padding: '11px 14px', textAlign: 'right', color: perTrade >= 0 ? '#10b981' : '#ef4444' }}>{fmtK(perTrade)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============ EQUITY CHART ============
function EquityChart({ curves, isDark, height = 260, highlightedModels, hoveredModel, setHoveredModel }) {
  const [tooltip, setTooltip] = useState(null);
  if (!curves) return null;

  const models = Object.keys(curves);
  const dataLen = curves[models[0]]?.length || 0;
  const visibleModels = models.filter(m => highlightedModels?.has(m));
  const allValues = visibleModels.length > 0 ? visibleModels.flatMap(m => curves[m] || []) : models.flatMap(m => curves[m] || []);

  const minVal = Math.min(...allValues) * 0.98;
  const maxVal = Math.max(...allValues) * 1.02;
  const width = 900;
  const padding = { top: 20, right: 20, bottom: 30, left: 70 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const xScale = i => padding.left + (i / (dataLen - 1)) * chartW;
  const yScale = v => padding.top + chartH - ((v - minVal) / (maxVal - minVal)) * chartH;
  const yTicks = Array.from({ length: 5 }, (_, i) => minVal + (i / 4) * (maxVal - minVal));

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}
      onMouseMove={e => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * width;
        const idx = Math.round(((x - padding.left) / chartW) * (dataLen - 1));
        if (idx >= 0 && idx < dataLen) {
          setTooltip({ x, idx, values: models.reduce((a, m) => ({ ...a, [m]: curves[m]?.[idx] }), {}) });
        }
      }}
      onMouseLeave={() => { setTooltip(null); setHoveredModel && setHoveredModel(null); }}
    >
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={padding.left} x2={width - padding.right} y1={yScale(v)} y2={yScale(v)} stroke={isDark ? '#27272a' : '#e4e4e7'} strokeDasharray="2,4" />
          <text x={padding.left - 8} y={yScale(v)} textAnchor="end" alignmentBaseline="middle" fill={isDark ? '#52525b' : '#a1a1aa'} fontSize={11}>{fmtK(v)}</text>
        </g>
      ))}
      <line x1={padding.left} x2={width - padding.right} y1={yScale(250000)} y2={yScale(250000)} stroke={isDark ? '#3f3f46' : '#d4d4d8'} strokeDasharray="4,4" />
      <text x={width - padding.right + 4} y={yScale(250000)} alignmentBaseline="middle" fill={isDark ? '#52525b' : '#a1a1aa'} fontSize={10}>Start</text>

      {models.map(m => {
        const isHighlighted = highlightedModels?.has(m) ?? true;
        const isHovered = hoveredModel === m;
        const shouldDim = hoveredModel && !isHovered;
        if (!isHighlighted) return null;
        const path = curves[m]?.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yScale(v)}`).join(' ');
        return (
          <path
            key={m} d={path} fill="none" stroke={MODELS[m]?.color || '#888'}
            strokeWidth={isHovered ? 3 : 2} opacity={shouldDim ? 0.15 : 0.85}
            style={{ transition: 'opacity 0.2s, stroke-width 0.2s' }}
            onMouseEnter={() => setHoveredModel && setHoveredModel(m)}
          />
        );
      })}

      {tooltip && (
        <g>
          <line x1={tooltip.x} x2={tooltip.x} y1={padding.top} y2={height - padding.bottom} stroke={isDark ? '#52525b' : '#a1a1aa'} strokeDasharray="4,4" />
          <rect x={Math.min(tooltip.x + 10, width - 150)} y={padding.top} width={140} height={visibleModels.length * 18 + 24} fill={isDark ? '#18181b' : '#fff'} stroke={isDark ? '#27272a' : '#e4e4e7'} rx={6} />
          <text x={Math.min(tooltip.x + 18, width - 142)} y={padding.top + 16} fill={isDark ? '#a1a1aa' : '#71717a'} fontSize={11}>Round {tooltip.idx}</text>
          {visibleModels.sort((a, b) => (tooltip.values[b] || 0) - (tooltip.values[a] || 0)).map((m, i) => (
            <text key={m} x={Math.min(tooltip.x + 18, width - 142)} y={padding.top + 34 + i * 18} fill={MODELS[m]?.color} fontSize={11} fontWeight={hoveredModel === m ? 700 : 500}>{MODELS[m]?.short}: {fmtK(tooltip.values[m])}</text>
          ))}
        </g>
      )}
    </svg>
  );
}

// ============ RACE TAB ============
function RaceTab({ data, frame, setFrame, isPlaying, setIsPlaying, maxFrames, isDark, colors }) {
  const { borderColor, cardBg, secondaryBg, textPrimary, textMuted } = colors;
  const { equity_curves } = data;
  if (!equity_curves) return null;

  const models = Object.keys(equity_curves);
  const currentValues = models.map(m => ({ model: m, value: equity_curves[m]?.[frame] || 250000, profit: (equity_curves[m]?.[frame] || 250000) - 250000 })).sort((a, b) => b.profit - a.profit);
  const maxProfit = Math.max(...currentValues.map(c => Math.abs(c.profit)), 1000); // Min 1000 to avoid division issues

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button onClick={() => { setFrame(0); setIsPlaying(false); }} style={{ padding: '8px 16px', borderRadius: 6, border: `1px solid ${borderColor}`, background: cardBg, color: textPrimary, cursor: 'pointer', fontSize: 13 }}>Reset</button>
        <button onClick={() => setIsPlaying(!isPlaying)} style={{ padding: '8px 20px', borderRadius: 6, border: 'none', background: isPlaying ? '#ef4444' : '#10b981', color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 500 }}>{isPlaying ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setFrame(maxFrames - 1); setIsPlaying(false); }} style={{ padding: '8px 16px', borderRadius: 6, border: `1px solid ${borderColor}`, background: cardBg, color: textPrimary, cursor: 'pointer', fontSize: 13 }}>End</button>
        <div style={{ flex: 1, marginLeft: 16 }}><input type="range" min={0} max={maxFrames - 1} value={frame} onChange={e => setFrame(+e.target.value)} style={{ width: '100%' }} /></div>
        <div style={{ fontSize: 13, color: textMuted, minWidth: 100 }}>Round {frame} / {maxFrames - 1}</div>
      </div>

      <div style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Profit Race</h3>
          <span style={{ fontSize: 12, color: textMuted }}>Starting: $250K each</span>
        </div>
        {currentValues.map((m, i) => {
          const cfg = MODELS[m.model];
          const isPositive = m.profit >= 0;
          const barWidth = (Math.abs(m.profit) / maxProfit) * 100;
          return (
            <div key={m.model} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: i < currentValues.length - 1 ? 14 : 0 }}>
              <div style={{ width: 28, fontSize: 14, color: textMuted, textAlign: 'center', fontWeight: 600 }}>{i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`}</div>
              <span style={{ fontSize: 22, width: 32 }}>{cfg?.icon}</span>
              <div style={{ width: 80, fontSize: 13, fontWeight: 500 }}>{cfg?.short}</div>
              <div style={{ flex: 1, height: 28, background: secondaryBg, borderRadius: 6, overflow: 'hidden', position: 'relative' }}>
                <div style={{
                  height: '100%',
                  width: `${barWidth}%`,
                  background: isPositive
                    ? `linear-gradient(90deg, ${cfg?.color}66, ${cfg?.color})`
                    : `linear-gradient(90deg, #ef444466, #ef4444)`,
                  borderRadius: 6,
                  transition: 'width 0.1s ease-out'
                }} />
              </div>
              <div style={{ width: 90, textAlign: 'right', fontSize: 14, fontWeight: 700, fontFamily: 'monospace', color: isPositive ? '#10b981' : '#ef4444' }}>
                {isPositive ? '+' : ''}{m.profit >= 1000 || m.profit <= -1000 ? `${(m.profit / 1000).toFixed(1)}K` : `${m.profit.toFixed(0)}`}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, padding: 20 }}>
        <h3 style={{ margin: '0 0 16px', fontSize: 16, fontWeight: 600 }}>Progress to Round {frame}</h3>
        <RaceLineChart curves={equity_curves} frame={frame} isDark={isDark} />
      </div>
    </div>
  );
}

function RaceLineChart({ curves, frame, isDark, height = 200 }) {
  if (!curves || frame === 0) return null;
  const models = Object.keys(curves);
  const visibleData = models.reduce((a, m) => ({ ...a, [m]: curves[m]?.slice(0, frame + 1) || [] }), {});
  const allVisible = Object.values(visibleData).flat();
  const minVal = allVisible.length ? Math.min(...allVisible) * 0.98 : 240000;
  const maxVal = allVisible.length ? Math.max(...allVisible) * 1.02 : 260000;

  const width = 900, padding = { top: 10, right: 20, bottom: 20, left: 60 };
  const chartW = width - padding.left - padding.right, chartH = height - padding.top - padding.bottom;
  const xScale = i => padding.left + (i / Math.max(frame, 1)) * chartW;
  const yScale = v => padding.top + chartH - ((v - minVal) / (maxVal - minVal || 1)) * chartH;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      {[0, 0.5, 1].map((p, i) => {
        const v = minVal + p * (maxVal - minVal);
        return (
          <g key={i}>
            <line x1={padding.left} x2={width - padding.right} y1={yScale(v)} y2={yScale(v)} stroke={isDark ? '#27272a' : '#e4e4e7'} />
            <text x={padding.left - 8} y={yScale(v)} textAnchor="end" alignmentBaseline="middle" fill={isDark ? '#52525b' : '#a1a1aa'} fontSize={10}>{fmtK(v)}</text>
          </g>
        );
      })}
      {models.map(m => {
        const d = visibleData[m];
        if (!d.length) return null;
        const path = d.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yScale(v)}`).join(' ');
        return <path key={m} d={path} fill="none" stroke={MODELS[m]?.color || '#888'} strokeWidth={2} strokeLinecap="round" />;
      })}
      {models.map(m => {
        const d = visibleData[m];
        if (!d.length) return null;
        return <circle key={m} cx={xScale(d.length - 1)} cy={yScale(d[d.length - 1])} r={4} fill={MODELS[m]?.color || '#888'} />;
      })}
    </svg>
  );
}

// ============ DNA TAB ============
function DNATab({ data, isDark, colors, onModelSelect }) {
  const { borderColor, cardBg, textPrimary, textMuted } = colors;
  const { leaderboard, model_stats } = data;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {leaderboard?.map((m, i) => {
        const stats = model_stats?.[m.model] || {};
        const cfg = MODELS[m.model] || {};
        const profitPerTrade = stats.total_trades > 0 ? (m.value - 250000) / stats.total_trades : 0;
        return (
          <div
            key={m.model}
            onClick={() => onModelSelect && onModelSelect(m.model)}
            style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, padding: 20, borderLeft: `4px solid ${cfg.color}`, cursor: 'pointer', transition: 'transform 0.15s, box-shadow 0.15s' }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = `0 4px 12px ${cfg.color}22`; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 16 }}>
              <span style={{ fontSize: 32 }}>{cfg.icon}</span>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 16, fontWeight: 600 }}>{m.model}</span>
                  <span style={{ fontSize: 12, padding: '2px 8px', background: `${cfg.color}22`, color: cfg.color, borderRadius: 4 }}>#{i + 1}</span>
                </div>
                <div style={{ fontSize: 13, color: textMuted }}>{cfg.desc}</div>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              <div><div style={{ fontSize: 11, color: textMuted }}>Final Value</div><div style={{ fontSize: 16, fontWeight: 600 }}>{fmtK(m.value)}</div></div>
              <div><div style={{ fontSize: 11, color: textMuted }}>Return</div><div style={{ fontSize: 16, fontWeight: 600, color: m.return_pct >= 0 ? '#10b981' : '#ef4444' }}>{fmtPct(m.return_pct)}</div></div>
              <div><div style={{ fontSize: 11, color: textMuted }}>Total Trades</div><div style={{ fontSize: 14, fontWeight: 500 }}>{stats.total_trades || 0}</div></div>
              <div><div style={{ fontSize: 11, color: textMuted }}>Profit/Trade</div><div style={{ fontSize: 14, fontWeight: 500, color: profitPerTrade >= 0 ? '#10b981' : '#ef4444' }}>{fmtK(profitPerTrade)}</div></div>
              <div><div style={{ fontSize: 11, color: textMuted }}>Trades/Day</div><div style={{ fontSize: 14, fontWeight: 500 }}>{(stats.trades_per_day || 0).toFixed(2)}</div></div>
              <div><div style={{ fontSize: 11, color: textMuted }}>Favorite Ticker</div><div style={{ fontSize: 14, fontWeight: 500 }}>{stats.favorite_ticker || 'N/A'} ({stats.favorite_ticker_count || 0})</div></div>
              <div style={{ gridColumn: 'span 2' }}>
                <div style={{ fontSize: 11, color: textMuted, marginBottom: 4 }}>Buy/Sell Ratio</div>
                <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', background: isDark ? '#27272a' : '#e4e4e7' }}>
                  <div style={{ width: `${stats.buy_ratio || 50}%`, background: '#10b981' }} />
                  <div style={{ width: `${100 - (stats.buy_ratio || 50)}%`, background: '#ef4444' }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: textMuted, marginTop: 4 }}>
                  <span>Buy {(stats.buy_ratio || 0).toFixed(1)}%</span>
                  <span>Sell {(100 - (stats.buy_ratio || 0)).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ============ TICKERS TAB ============
function TickersTab({ data, isDark, colors }) {
  const { borderColor, cardBg, secondaryBg, textMuted } = colors;
  const { ticker_stats } = data;
  const top10 = ticker_stats?.slice(0, 10) || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
        {top10.map((t, i) => (
          <div key={t.ticker} style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 10, padding: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{ fontSize: 16, fontWeight: 700 }}>{t.ticker}</span>
              <span style={{ fontSize: 11, color: textMuted }}>#{i + 1}</span>
            </div>
            <div style={{ fontSize: 13, color: textMuted }}>{t.trades} trades</div>
            <div style={{ fontSize: 12, color: textMuted }}>{fmtK(t.volume)} vol</div>
            <div style={{ display: 'flex', gap: 8, marginTop: 8, fontSize: 11 }}>
              <span style={{ color: '#10b981' }}>{t.buys} buys</span>
              <span style={{ color: '#ef4444' }}>{t.sells} sells</span>
            </div>
          </div>
        ))}
      </div>
      <div style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: secondaryBg }}>
              <th style={{ padding: '10px 16px', textAlign: 'left', fontWeight: 600 }}>Ticker</th>
              <th style={{ padding: '10px 16px', textAlign: 'right', fontWeight: 600 }}>Trades</th>
              <th style={{ padding: '10px 16px', textAlign: 'right', fontWeight: 600 }}>Volume</th>
              <th style={{ padding: '10px 16px', textAlign: 'right', fontWeight: 600 }}>Buys</th>
              <th style={{ padding: '10px 16px', textAlign: 'right', fontWeight: 600 }}>Sells</th>
              <th style={{ padding: '10px 16px', textAlign: 'right', fontWeight: 600 }}>Buy %</th>
            </tr>
          </thead>
          <tbody>
            {ticker_stats?.map(t => (
              <tr key={t.ticker} style={{ borderBottom: `1px solid ${borderColor}` }}>
                <td style={{ padding: '10px 16px', fontWeight: 600 }}>{t.ticker}</td>
                <td style={{ padding: '10px 16px', textAlign: 'right' }}>{t.trades}</td>
                <td style={{ padding: '10px 16px', textAlign: 'right' }}>{fmtK(t.volume)}</td>
                <td style={{ padding: '10px 16px', textAlign: 'right', color: '#10b981' }}>{t.buys}</td>
                <td style={{ padding: '10px 16px', textAlign: 'right', color: '#ef4444' }}>{t.sells}</td>
                <td style={{ padding: '10px 16px', textAlign: 'right' }}>{(t.buy_ratio || 0).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============ H2H TAB ============
function H2HTab({ data, modelA, modelB, setModelA, setModelB, isDark, colors }) {
  const { borderColor, cardBg, textPrimary, textMuted } = colors;
  const { leaderboard, model_stats } = data;
  const modelNames = Object.keys(MODELS);
  const statsA = leaderboard?.find(m => m.model === modelA);
  const statsB = leaderboard?.find(m => m.model === modelB);
  const detailA = model_stats?.[modelA] || {};
  const detailB = model_stats?.[modelB] || {};
  const selectStyle = { padding: '8px 12px', borderRadius: 6, border: `1px solid ${borderColor}`, background: cardBg, color: textPrimary, fontSize: 14, cursor: 'pointer' };

  const CompareRow = ({ label, valA, valB, format = v => v, better = 'high' }) => {
    const numA = typeof valA === 'number' ? valA : 0, numB = typeof valB === 'number' ? valB : 0;
    const winner = better === 'high' ? (numA > numB ? 'A' : numA < numB ? 'B' : null) : (numA < numB ? 'A' : numA > numB ? 'B' : null);
    return (
      <div style={{ display: 'flex', alignItems: 'center', padding: '12px 0', borderBottom: `1px solid ${borderColor}` }}>
        <div style={{ flex: 1, textAlign: 'right', fontWeight: winner === 'A' ? 600 : 400, color: winner === 'A' ? '#10b981' : textPrimary }}>{format(valA)}</div>
        <div style={{ width: 120, textAlign: 'center', fontSize: 13, color: textMuted }}>{label}</div>
        <div style={{ flex: 1, textAlign: 'left', fontWeight: winner === 'B' ? 600 : 400, color: winner === 'B' ? '#10b981' : textPrimary }}>{format(valB)}</div>
      </div>
    );
  };

  const totalReturn = (statsA?.value || 250000) + (statsB?.value || 250000) - 500000;
  const pctA = totalReturn > 0 ? ((statsA?.value || 250000) - 250000) / totalReturn * 100 : 50;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24 }}>
        <select value={modelA} onChange={e => setModelA(e.target.value)} style={selectStyle}>{modelNames.map(m => <option key={m} value={m}>{MODELS[m]?.icon} {m}</option>)}</select>
        <span style={{ fontSize: 20, fontWeight: 700, color: textMuted }}>VS</span>
        <select value={modelB} onChange={e => setModelB(e.target.value)} style={selectStyle}>{modelNames.map(m => <option key={m} value={m}>{MODELS[m]?.icon} {m}</option>)}</select>
      </div>
      <div style={{ background: cardBg, border: `1px solid ${borderColor}`, borderRadius: 12, padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 20 }}>
          <div style={{ flex: 1, textAlign: 'center' }}><div style={{ fontSize: 40, marginBottom: 8 }}>{MODELS[modelA]?.icon}</div><div style={{ fontSize: 18, fontWeight: 600 }}>{MODELS[modelA]?.short}</div><div style={{ fontSize: 13, color: textMuted }}>{MODELS[modelA]?.desc}</div></div>
          <div style={{ fontSize: 24, fontWeight: 700, color: textMuted, padding: '0 20px' }}>⚔️</div>
          <div style={{ flex: 1, textAlign: 'center' }}><div style={{ fontSize: 40, marginBottom: 8 }}>{MODELS[modelB]?.icon}</div><div style={{ fontSize: 18, fontWeight: 600 }}>{MODELS[modelB]?.short}</div><div style={{ fontSize: 13, color: textMuted }}>{MODELS[modelB]?.desc}</div></div>
        </div>
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden' }}>
            <div style={{ width: `${pctA}%`, background: MODELS[modelA]?.color, transition: 'width 0.3s' }} />
            <div style={{ width: `${100 - pctA}%`, background: MODELS[modelB]?.color, transition: 'width 0.3s' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: textMuted, marginTop: 4 }}><span>{pctA.toFixed(1)}% of gains</span><span>{(100 - pctA).toFixed(1)}% of gains</span></div>
        </div>
        <CompareRow label="Rank" valA={statsA?.rank} valB={statsB?.rank} format={v => `#${v}`} better="low" />
        <CompareRow label="Final Value" valA={statsA?.value} valB={statsB?.value} format={fmtK} />
        <CompareRow label="Return %" valA={statsA?.return_pct} valB={statsB?.return_pct} format={fmtPct} />
        <CompareRow label="Total Trades" valA={detailA.total_trades} valB={detailB.total_trades} format={v => v} />
        <CompareRow label="Volume" valA={detailA.total_volume} valB={detailB.total_volume} format={fmtK} />
        <CompareRow label="Trades/Day" valA={detailA.trades_per_day} valB={detailB.trades_per_day} format={v => (v || 0).toFixed(2)} />
        <CompareRow label="Favorite Ticker" valA={detailA.favorite_ticker} valB={detailB.favorite_ticker} format={v => v || 'N/A'} better={null} />
      </div>
    </div>
  );
}

// ============ MODEL PORTFOLIO SIDEBAR ============
function ModelPortfolio({ model, data, modelTrades, loadingTrades, onClose, isDark, colors }) {
  const { borderColor, cardBg, secondaryBg, textPrimary, textMuted } = colors;
  const cfg = MODELS[model];
  const stats = data.model_stats?.[model] || {};
  const lb = data.leaderboard?.find(m => m.model === model) || {};
  const perTrade = stats.total_trades > 0 ? (lb.value - 250000) / stats.total_trades : 0;
  const trades = modelTrades[model]?.trades || data.recent_trades?.filter(t => t.model === model) || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div style={{ padding: 20, borderBottom: `1px solid ${borderColor}`, background: `linear-gradient(135deg, ${cfg?.color}18 0%, ${cardBg} 100%)`, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 36 }}>{cfg?.icon}</span>
            <div><div style={{ fontSize: 18, fontWeight: 600 }}>{cfg?.short}</div><div style={{ fontSize: 12, color: textMuted }}>{cfg?.desc}</div></div>
          </div>
          <button onClick={onClose} style={{ width: 32, height: 32, borderRadius: 8, border: `1px solid ${borderColor}`, background: 'transparent', color: textMuted, cursor: 'pointer', fontSize: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>✕</button>
        </div>
        <div style={{ display: 'flex', gap: 20 }}>
          <div><div style={{ fontSize: 11, color: textMuted, marginBottom: 2 }}>Rank</div><div style={{ fontSize: 22, fontWeight: 700 }}>#{lb.rank}</div></div>
          <div><div style={{ fontSize: 11, color: textMuted, marginBottom: 2 }}>Final Value</div><div style={{ fontSize: 22, fontWeight: 700 }}>{fmtK(lb.value)}</div></div>
          <div><div style={{ fontSize: 11, color: textMuted, marginBottom: 2 }}>Return</div><div style={{ fontSize: 22, fontWeight: 700, color: lb.return_pct >= 0 ? '#10b981' : '#ef4444' }}>{fmtPct(lb.return_pct)}</div></div>
        </div>
      </div>
      <div style={{ padding: 16, borderBottom: `1px solid ${borderColor}`, flexShrink: 0 }}>
        <h4 style={{ margin: '0 0 12px', fontSize: 13, fontWeight: 600, color: textMuted }}>STATISTICS</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
          <div style={{ background: secondaryBg, borderRadius: 8, padding: 12 }}><div style={{ fontSize: 11, color: textMuted }}>Total Trades</div><div style={{ fontSize: 18, fontWeight: 600 }}>{stats.total_trades || lb.trades || 0}</div></div>
          <div style={{ background: secondaryBg, borderRadius: 8, padding: 12 }}><div style={{ fontSize: 11, color: textMuted }}>Profit/Trade</div><div style={{ fontSize: 18, fontWeight: 600, color: perTrade >= 0 ? '#10b981' : '#ef4444' }}>{fmtK(perTrade)}</div></div>
          <div style={{ background: secondaryBg, borderRadius: 8, padding: 12 }}><div style={{ fontSize: 11, color: textMuted }}>Total Volume</div><div style={{ fontSize: 18, fontWeight: 600 }}>{fmtK(stats.total_volume || 0)}</div></div>
          <div style={{ background: secondaryBg, borderRadius: 8, padding: 12 }}><div style={{ fontSize: 11, color: textMuted }}>Favorite Ticker</div><div style={{ fontSize: 18, fontWeight: 600 }}>{stats.favorite_ticker || 'N/A'}</div></div>
        </div>
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: textMuted, marginBottom: 4 }}><span>Buy {(stats.buy_ratio || 50).toFixed(1)}%</span><span>Sell {(100 - (stats.buy_ratio || 50)).toFixed(1)}%</span></div>
          <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden' }}><div style={{ width: `${stats.buy_ratio || 50}%`, background: '#10b981' }} /><div style={{ width: `${100 - (stats.buy_ratio || 50)}%`, background: '#ef4444' }} /></div>
        </div>
      </div>
      <div className="hide-scroll" style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        <h4 style={{ margin: '0 0 12px', fontSize: 13, fontWeight: 600, color: textMuted }}>RECENT TRADES</h4>
        {loadingTrades && <div style={{ textAlign: 'center', color: textMuted, padding: 20 }}>Loading trades...</div>}
        {!loadingTrades && trades.length === 0 && <div style={{ textAlign: 'center', color: textMuted, padding: 20 }}>No trades available</div>}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {trades.slice(0, 30).map((t, i) => (
            <div key={i} style={{ background: secondaryBg, borderRadius: 8, padding: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 4, background: t.action === 'BUY' ? '#10b98120' : '#ef444420', color: t.action === 'BUY' ? '#10b981' : '#ef4444', fontWeight: 600 }}>{t.action}</span>
                <span style={{ fontSize: 14, fontWeight: 600 }}>{t.ticker}</span>
                <span style={{ fontSize: 11, color: textMuted, marginLeft: 'auto' }}>{t.timestamp?.split('T')[0]}</span>
              </div>
              <div style={{ fontSize: 12, color: textMuted, marginTop: 6 }}>{fmtK(t.amount_usd)} · {t.shares?.toFixed(2)} shares @ ${t.price?.toFixed(2)}</div>
              {(t.reasoning || t.model_reasoning) && (
                <div style={{ fontSize: 11, color: textMuted, marginTop: 8, padding: 10, background: cardBg, borderRadius: 6, borderLeft: `3px solid ${cfg?.color}`, lineHeight: 1.5, maxHeight: '200px', overflowY: 'auto' }}>{t.reasoning || t.model_reasoning}</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============ FEED SIDEBAR (3-TAB VERSION) ============
function FeedSidebar({ trades, isDark, colors, expandedTrade, setExpandedTrade, onModelClick, data }) {
  const { borderColor, cardBg, secondaryBg, textPrimary, textMuted } = colors;
  const [sidebarTab, setSidebarTab] = useState('feed');
  const [expandedPortfolio, setExpandedPortfolio] = useState(null);

  const grouped = (trades || []).reduce((acc, t, idx) => {
    const date = t.timestamp?.split('T')[0] || 'Unknown';
    if (!acc[date]) acc[date] = [];
    acc[date].push({ ...t, globalIdx: idx });
    return acc;
  }, {});

  const leaderboard = data?.leaderboard || [];
  const modelStats = data?.model_stats || {};

  const portfolios = leaderboard.map((m, idx) => {
    const stats = modelStats[m.model] || {};
    const modelTrades = (trades || []).filter(t => t.model === m.model);
    const tickerMap = {};
    modelTrades.forEach(t => {
      if (!tickerMap[t.ticker]) tickerMap[t.ticker] = { ticker: t.ticker, volume: 0, trades: 0, buys: 0, sells: 0 };
      tickerMap[t.ticker].volume += t.amount_usd || 0;
      tickerMap[t.ticker].trades++;
      if (t.action === 'BUY') tickerMap[t.ticker].buys++; else tickerMap[t.ticker].sells++;
    });
    const holdings = Object.values(tickerMap).sort((a, b) => b.volume - a.volume);
    return { model: m.model, rank: idx + 1, value: m.value || 250000, returnPct: m.return_pct || 0, trades: m.trades || stats.total_trades || 0, holdings, bestTicker: holdings[0], worstTicker: holdings[holdings.length - 1] };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div style={{ display: 'flex', borderBottom: `1px solid ${borderColor}`, flexShrink: 0 }}>
        {['feed', 'holdings', 'stats', 'chat'].map(tab => (
          <button key={tab} onClick={() => setSidebarTab(tab)} style={{
            flex: 1, padding: '11px 8px', fontSize: '10px', fontWeight: '600',
            background: 'none', border: 'none', cursor: 'pointer',
            textTransform: 'uppercase', letterSpacing: '0.5px',
            borderBottom: sidebarTab === tab ? `2px solid ${textPrimary}` : '2px solid transparent',
            color: sidebarTab === tab ? textPrimary : textMuted
          }}>
            {tab === 'feed' ? 'Feed' : tab === 'holdings' ? 'Holdings' : tab === 'stats' ? 'Stats' : 'Ask AI'}
          </button>
        ))}
      </div>
      <div className="hide-scroll" style={{ flex: 1, overflowY: 'auto' }}>
        {sidebarTab === 'feed' && (
          <div style={{ padding: '12px' }}>
            <div style={{ fontSize: '10px', color: textMuted, marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px', display: 'flex', justifyContent: 'space-between' }}>
              <span>Recent Trades</span>
              <span>{trades?.length || 0}</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {(trades || []).slice(0, 25).map((t, idx) => {
                const isExp = expandedTrade === idx;
                const cfg = MODELS[t.model];
                return (
                  <div key={idx} onClick={() => setExpandedTrade(isExp ? null : idx)} style={{
                    padding: '10px 12px',
                    borderRadius: '6px',
                    background: secondaryBg,
                    border: `1px solid ${isExp ? (cfg?.color || borderColor) + '40' : borderColor}`,
                    cursor: 'pointer'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: '600', fontSize: '12px', color: textPrimary, fontFamily: 'monospace' }}>{t.ticker}</span>
                        <span style={{
                          fontSize: '9px', padding: '2px 5px', borderRadius: '3px',
                          background: t.action === 'BUY' ? '#10b98115' : '#ef444415',
                          color: t.action === 'BUY' ? '#10b981' : '#ef4444',
                          fontWeight: '500', textTransform: 'uppercase'
                        }}>{t.action}</span>
                      </div>
                      <span style={{ fontSize: '10px', color: textMuted, fontFamily: 'monospace' }}>{t.timestamp?.split('T')[0]}</span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontSize: '14px', cursor: 'pointer' }} onClick={e => { e.stopPropagation(); onModelClick(t.model); }}>{cfg?.icon}</span>
                        <span style={{ fontSize: '11px', fontWeight: '500', color: cfg?.color }}>{cfg?.short}</span>
                      </div>
                      <span style={{ fontSize: '11px', fontWeight: '500', color: textPrimary, fontFamily: 'monospace' }}>
                        {fmtK(t.amount_usd)}
                      </span>
                    </div>

                    {isExp && (
                      <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: `1px solid ${borderColor}` }}>
                        <div style={{ fontSize: '10px', color: textMuted, marginBottom: '6px' }}>
                          {t.shares?.toFixed(2)} shares @ ${t.price?.toFixed(2)}
                        </div>
                        {t.reasoning && (
                          <div style={{
                            fontSize: '10px', color: textMuted, lineHeight: 1.5,
                            padding: '8px', borderRadius: '4px',
                            background: cardBg,
                            borderLeft: `2px solid ${cfg?.color}`,
                            maxHeight: '300px',
                            overflowY: 'auto',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word'
                          }}>
                            {t.reasoning}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {sidebarTab === 'holdings' && (
          <div style={{ padding: '12px' }}>
            <div style={{ fontSize: '10px', color: textMuted, marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Model Portfolios
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {portfolios.map((p) => {
                const cfg = MODELS[p.model];
                const isExp = expandedPortfolio === p.model;
                const profit = p.value - 250000;
                const profitPct = (profit / 250000) * 100;

                return (
                  <div key={p.model} onClick={() => setExpandedPortfolio(isExp ? null : p.model)} style={{
                    borderRadius: '6px',
                    border: `1px solid ${cfg?.color}25`,
                    overflow: 'hidden',
                    cursor: 'pointer',
                    background: cardBg
                  }}>
                    {/* Header */}
                    <div style={{ padding: '12px', background: `${cfg?.color}08`, borderLeft: `3px solid ${cfg?.color}` }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '10px', color: textMuted, fontFamily: 'monospace' }}>#{p.rank}</span>
                          <span style={{ fontSize: '16px' }}>{cfg?.icon}</span>
                          <span style={{ fontSize: '12px', fontWeight: '600', color: cfg?.color }}>{cfg?.short}</span>
                        </div>
                        <span style={{ fontSize: '14px', fontWeight: '700', color: textPrimary, fontFamily: 'monospace' }}>{fmtK(p.value)}</span>
                      </div>

                      <div style={{ display: 'flex', gap: '16px', fontSize: '10px' }}>
                        <span style={{ color: textMuted }}>{p.trades} trades</span>
                        <span style={{ color: profitPct >= 0 ? '#10b981' : '#ef4444', fontWeight: '500', fontFamily: 'monospace' }}>
                          {profitPct >= 0 ? '+' : ''}{profitPct.toFixed(2)}%
                        </span>
                      </div>
                    </div>

                    {/* Expanded */}
                    {isExp && (
                      <div style={{ background: secondaryBg }}>
                        {p.bestTicker && p.worstTicker && (
                          <div style={{ display: 'flex', borderBottom: `1px solid ${borderColor}` }}>
                            <div style={{ flex: 1, padding: '8px 12px', borderRight: `1px solid ${borderColor}` }}>
                              <div style={{ fontSize: '9px', color: textMuted, marginBottom: '2px' }}>BEST</div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <span style={{ fontWeight: '600', color: textPrimary, fontSize: '11px' }}>{p.bestTicker.ticker}</span>
                                <span style={{ fontSize: '10px', color: '#10b981', fontFamily: 'monospace' }}>+{((p.bestTicker.volume / p.value) * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                            <div style={{ flex: 1, padding: '8px 12px' }}>
                              <div style={{ fontSize: '9px', color: textMuted, marginBottom: '2px' }}>WORST</div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <span style={{ fontWeight: '600', color: textPrimary, fontSize: '11px' }}>{p.worstTicker.ticker}</span>
                                <span style={{ fontSize: '10px', color: '#ef4444', fontFamily: 'monospace' }}>{((p.worstTicker.volume / p.value) * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          </div>
                        )}

                        <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
                          {p.holdings.map((h) => (
                            <div key={h.ticker} style={{ display: 'flex', alignItems: 'center', padding: '6px 12px', borderBottom: `1px solid ${borderColor}`, fontSize: '10px' }}>
                              <span style={{ fontWeight: '500', color: textPrimary, width: '40px' }}>{h.ticker}</span>
                              <span style={{ color: textMuted, flex: 1, fontFamily: 'monospace' }}>{fmtK(h.volume)}</span>
                              <span style={{ fontWeight: '500', color: h.buys > h.sells ? '#10b981' : '#ef4444', fontFamily: 'monospace' }}>
                                {h.buys > h.sells ? '+' : ''}{((h.volume / p.value) * 100).toFixed(2)}%
                              </span>
                            </div>
                          ))}
                          {p.holdings.length === 0 && <div style={{ padding: '16px', textAlign: 'center', color: textMuted, fontSize: '10px' }}>No holdings</div>}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {sidebarTab === 'stats' && (
          <div style={{ padding: '12px' }}>
            {/* Overview */}
            <div style={{ marginBottom: '14px' }}>
              <div style={{ fontSize: '10px', color: textMuted, marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Overview
              </div>
              <div style={{ background: secondaryBg, borderRadius: '6px', padding: '12px', border: `1px solid ${borderColor}` }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '12px' }}>
                  <div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: textPrimary, fontFamily: 'monospace' }}>7</div>
                    <div style={{ fontSize: '10px', color: textMuted }}>Models</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: textPrimary, fontFamily: 'monospace' }}>{trades?.length || 0}</div>
                    <div style={{ fontSize: '10px', color: textMuted }}>Trades</div>
                  </div>
                </div>
                {leaderboard[0] && (
                  <div style={{ padding: '10px', background: `${MODELS[leaderboard[0].model]?.color}10`, borderRadius: '4px', borderLeft: `3px solid ${MODELS[leaderboard[0].model]?.color}` }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div>
                        <div style={{ fontSize: '11px', fontWeight: '600', color: MODELS[leaderboard[0].model]?.color }}>{MODELS[leaderboard[0].model]?.short}</div>
                        <div style={{ fontSize: '9px', color: textMuted }}>Leader</div>
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: MODELS[leaderboard[0].model]?.color, fontFamily: 'monospace' }}>{fmtK(leaderboard[0].value)}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Performance */}
            <div style={{ marginBottom: '14px' }}>
              <div style={{ fontSize: '10px', color: textMuted, marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Returns
              </div>
              <div style={{ background: secondaryBg, borderRadius: '6px', padding: '12px', border: `1px solid ${borderColor}` }}>
                {leaderboard.map((m) => {
                  const cfg = MODELS[m.model];
                  const pct = Math.min(100, Math.max(0, ((m.return_pct || 0) + 10) * 2));
                  return (
                    <div key={m.model} style={{ marginBottom: '8px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <div style={{ width: '6px', height: '6px', borderRadius: '2px', background: cfg?.color }} />
                          <span style={{ fontSize: '10px', color: textMuted }}>{cfg?.short}</span>
                        </div>
                        <span style={{ fontSize: '10px', fontWeight: '500', color: textPrimary, fontFamily: 'monospace' }}>{fmtPct(m.return_pct)}</span>
                      </div>
                      <div style={{ height: '4px', background: borderColor, borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${pct}%`, background: cfg?.color, borderRadius: '2px' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Action Distribution */}
            <div style={{ marginBottom: '14px' }}>
              <div style={{ fontSize: '10px', color: textMuted, marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Buy/Sell Ratio
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
                {(() => {
                  const buys = (trades || []).filter(t => t.action === 'BUY').length;
                  const sells = (trades || []).filter(t => t.action === 'SELL').length;
                  return [
                    { key: 'buy', label: 'Buy', color: '#10b981', count: buys },
                    { key: 'sell', label: 'Sell', color: '#ef4444', count: sells }
                  ].map(r => (
                    <div key={r.key} style={{
                      background: secondaryBg,
                      borderRadius: '6px',
                      padding: '10px',
                      textAlign: 'center',
                      border: `1px solid ${borderColor}`
                    }}>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: r.color, fontFamily: 'monospace' }}>{r.count}</div>
                      <div style={{ fontSize: '9px', color: textMuted }}>{r.label}</div>
                    </div>
                  ));
                })()}
              </div>
            </div>

            {/* Top Tickers */}
            <div>
              <div style={{ fontSize: '10px', color: textMuted, marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Top Tickers
              </div>
              <div style={{ background: secondaryBg, borderRadius: '6px', padding: '10px', border: `1px solid ${borderColor}` }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
                  {(() => {
                    const tickerCounts = {};
                    (trades || []).forEach(t => { tickerCounts[t.ticker] = (tickerCounts[t.ticker] || 0) + 1; });
                    return Object.entries(tickerCounts).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([ticker, count]) => (
                      <div key={ticker} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 8px', borderRadius: '4px', background: cardBg }}>
                        <span style={{ fontSize: '10px', color: textMuted, flex: 1 }}>{ticker}</span>
                        <span style={{ fontSize: '12px', fontWeight: '600', color: textPrimary, fontFamily: 'monospace' }}>{count}</span>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            </div>
          </div>
        )}

        {sidebarTab === 'chat' && (
          <TournamentChat data={data} isDark={isDark} />
        )}
      </div>
    </div>
  );
}