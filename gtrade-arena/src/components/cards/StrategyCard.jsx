import React, { useState } from 'react'
import { useTheme } from '../../App'

// Icons
const Icons = {
  TrendingUp: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" /><polyline points="16 7 22 7 22 13" />
    </svg>
  ),
  TrendingDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 17 13.5 8.5 8.5 13.5 2 7" /><polyline points="16 17 22 17 22 11" />
    </svg>
  ),
  Trophy: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" /><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" /><path d="M4 22h16" /><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" /><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" /><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
    </svg>
  ),
  Target: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" />
    </svg>
  ),
  BarChart: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" x2="12" y1="20" y2="10" /><line x1="18" x2="18" y1="20" y2="4" /><line x1="6" x2="6" y1="20" y2="14" />
    </svg>
  ),
  ChevronDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  )
}

// Strategy configurations
const strategyConfig = {
  'Signal Follower': {
    icon: '🤖',
    color: '#9b59b6',
    bgLight: 'rgba(155, 89, 182, 0.08)',
    bgDark: 'rgba(155, 89, 182, 0.15)',
    description: 'LLM-powered signals',
    personality: 'Analytical'
  },
  'Cooperator': {
    icon: '🤝',
    color: '#27ae60',
    bgLight: 'rgba(39, 174, 96, 0.08)',
    bgDark: 'rgba(39, 174, 96, 0.15)',
    description: 'Follows consensus',
    personality: 'Trust Builder'
  },
  'Defector': {
    icon: '🎯',
    color: '#e74c3c',
    bgLight: 'rgba(231, 76, 60, 0.08)',
    bgDark: 'rgba(231, 76, 60, 0.15)',
    description: 'Contrarian positions',
    personality: 'Opportunist'
  },
  'Tit-for-Tat': {
    icon: '🔄',
    color: '#3498db',
    bgLight: 'rgba(52, 152, 219, 0.08)',
    bgDark: 'rgba(52, 152, 219, 0.15)',
    description: 'Mimics last winner',
    personality: 'Adaptive'
  },
  'Buy-and-Hold': {
    icon: '📊',
    color: '#64748b',
    bgLight: 'rgba(100, 116, 139, 0.08)',
    bgDark: 'rgba(100, 116, 139, 0.15)',
    description: 'Benchmark strategy',
    personality: 'Passive'
  }
}

export default function StrategyCard({ 
  strategy,
  allocation = 250000,
  returnPct = 0,
  wins = 0,
  totalRounds = 90,
  sharpe = 0,
  maxDrawdown = 0,
  vsBenchmark = 0,
  regimePerformance = null,
  isLeader = false,
  rank = null,
  compact = false
}) {
  const { isDark } = useTheme()
  const [isExpanded, setIsExpanded] = useState(false)
  
  const config = strategyConfig[strategy] || strategyConfig['Buy-and-Hold']
  const winRate = totalRounds > 0 ? ((wins / totalRounds) * 100).toFixed(1) : 0
  const isPositive = returnPct >= 0

  // Compact version for grid displays
  if (compact) {
    return (
      <div
        className="p-4 rounded-xl border transition-all duration-200 cursor-pointer"
        style={{
          backgroundColor: isDark ? config.bgDark : config.bgLight,
          borderColor: isLeader ? config.color : 'var(--border-primary)'
        }}
        onClick={() => setIsExpanded(!isExpanded)}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = config.color
          e.currentTarget.style.transform = 'translateY(-2px)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = isLeader ? config.color : 'var(--border-primary)'
          e.currentTarget.style.transform = 'translateY(0)'
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xl">{config.icon}</span>
            <div>
              <div 
                className="text-sm font-semibold"
                style={{ color: config.color }}
              >
                {strategy}
              </div>
              <div 
                className="text-[10px]"
                style={{ color: 'var(--text-muted)' }}
              >
                {config.personality}
              </div>
            </div>
          </div>
          
          {isLeader && (
            <span 
              className="text-[9px] font-bold px-1.5 py-0.5 rounded"
              style={{ 
                backgroundColor: 'rgba(245, 158, 11, 0.15)',
                color: '#f59e0b'
              }}
            >
              👑 LEADER
            </span>
          )}
          
          {rank && !isLeader && (
            <span 
              className="text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center"
              style={{ 
                backgroundColor: 'var(--bg-hover)',
                color: 'var(--text-tertiary)'
              }}
            >
              {rank}
            </span>
          )}
        </div>

        {/* Return */}
        <div className="flex items-center gap-2 mb-3">
          {isPositive ? (
            <Icons.TrendingUp size={18} style={{ color: 'var(--success)' }} />
          ) : (
            <Icons.TrendingDown size={18} style={{ color: 'var(--danger)' }} />
          )}
          <span 
            className="text-2xl font-bold"
            style={{ color: isPositive ? 'var(--success)' : 'var(--danger)' }}
          >
            {isPositive ? '+' : ''}{returnPct.toFixed(2)}%
          </span>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 gap-2">
          <div 
            className="p-2 rounded-lg"
            style={{ backgroundColor: 'var(--bg-card)' }}
          >
            <div 
              className="text-[9px] uppercase tracking-wider mb-0.5"
              style={{ color: 'var(--text-muted)' }}
            >
              Allocation
            </div>
            <div 
              className="text-sm font-semibold"
              style={{ color: 'var(--text-primary)' }}
            >
              ${(allocation / 1000).toFixed(0)}K
            </div>
          </div>
          
          <div 
            className="p-2 rounded-lg"
            style={{ backgroundColor: 'var(--bg-card)' }}
          >
            <div 
              className="text-[9px] uppercase tracking-wider mb-0.5"
              style={{ color: 'var(--text-muted)' }}
            >
              Win Rate
            </div>
            <div 
              className="text-sm font-semibold"
              style={{ color: 'var(--text-primary)' }}
            >
              {winRate}%
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Full version with expandable details
  return (
    <div
      className="rounded-xl border overflow-hidden transition-all duration-200"
      style={{
        backgroundColor: 'var(--bg-card)',
        borderColor: isLeader ? config.color : 'var(--border-primary)'
      }}
    >
      {/* Main Content */}
      <div 
        className="p-4 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div 
              className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl"
              style={{ backgroundColor: isDark ? config.bgDark : config.bgLight }}
            >
              {config.icon}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span 
                  className="text-base font-bold"
                  style={{ color: config.color }}
                >
                  {strategy}
                </span>
                {isLeader && (
                  <span 
                    className="text-[9px] font-bold px-1.5 py-0.5 rounded"
                    style={{ 
                      backgroundColor: 'rgba(245, 158, 11, 0.15)',
                      color: '#f59e0b'
                    }}
                  >
                    👑 #1
                  </span>
                )}
              </div>
              <div 
                className="text-xs"
                style={{ color: 'var(--text-tertiary)' }}
              >
                {config.description}
              </div>
            </div>
          </div>

          {/* Return Badge */}
          <div className="text-right">
            <div className="flex items-center gap-1.5 justify-end">
              {isPositive ? (
                <Icons.TrendingUp size={20} style={{ color: 'var(--success)' }} />
              ) : (
                <Icons.TrendingDown size={20} style={{ color: 'var(--danger)' }} />
              )}
              <span 
                className="text-2xl font-bold"
                style={{ color: isPositive ? 'var(--success)' : 'var(--danger)' }}
              >
                {isPositive ? '+' : ''}{returnPct.toFixed(2)}%
              </span>
            </div>
            <div 
              className="text-xs mt-0.5"
              style={{ color: 'var(--text-muted)' }}
            >
              ${allocation.toLocaleString()}
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: 'Wins', value: `${wins}/${totalRounds}`, icon: Icons.Trophy },
            { label: 'Win Rate', value: `${winRate}%`, icon: Icons.Target },
            { label: 'Sharpe', value: sharpe.toFixed(2), icon: Icons.BarChart },
            { 
              label: 'vs Bench', 
              value: `${vsBenchmark >= 0 ? '+' : ''}${vsBenchmark.toFixed(1)}%`,
              isColored: true,
              isPositive: vsBenchmark >= 0
            }
          ].map((stat, i) => (
            <div 
              key={i}
              className="p-2 rounded-lg text-center"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <div 
                className="text-[9px] uppercase tracking-wider mb-1"
                style={{ color: 'var(--text-muted)' }}
              >
                {stat.label}
              </div>
              <div 
                className="text-sm font-semibold"
                style={{ 
                  color: stat.isColored 
                    ? (stat.isPositive ? 'var(--success)' : 'var(--danger)')
                    : 'var(--text-primary)'
                }}
              >
                {stat.value}
              </div>
            </div>
          ))}
        </div>

        {/* Expand Button */}
        <div 
          className="flex items-center justify-center mt-3 pt-3 border-t"
          style={{ borderColor: 'var(--border-primary)' }}
        >
          <div 
            className="flex items-center gap-1 text-[10px] font-medium"
            style={{ color: 'var(--text-muted)' }}
          >
            <span>{isExpanded ? 'Less' : 'More'} details</span>
            <Icons.ChevronDown 
              size={14} 
              style={{ 
                transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s'
              }} 
            />
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div 
          className="px-4 pb-4 pt-0 border-t"
          style={{ borderColor: 'var(--border-primary)' }}
        >
          {/* Regime Performance */}
          {regimePerformance && (
            <div className="mt-3">
              <div 
                className="text-[10px] font-semibold uppercase tracking-wider mb-2"
                style={{ color: 'var(--text-muted)' }}
              >
                Performance by Regime
              </div>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { regime: 'Bull', value: regimePerformance.bull, color: 'var(--success)' },
                  { regime: 'Bear', value: regimePerformance.bear, color: 'var(--danger)' },
                  { regime: 'Sideways', value: regimePerformance.sideways, color: 'var(--warning)' }
                ].map((r, i) => (
                  <div 
                    key={i}
                    className="p-2 rounded-lg text-center"
                    style={{ backgroundColor: 'var(--bg-tertiary)' }}
                  >
                    <div 
                      className="text-[9px] font-medium mb-1"
                      style={{ color: r.color }}
                    >
                      {r.regime}
                    </div>
                    <div 
                      className="text-sm font-bold"
                      style={{ color: r.value >= 0 ? 'var(--success)' : 'var(--danger)' }}
                    >
                      {r.value >= 0 ? '+' : ''}{r.value.toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk Metrics */}
          <div className="mt-3">
            <div 
              className="text-[10px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Risk Metrics
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div 
                className="p-2 rounded-lg"
                style={{ backgroundColor: 'var(--bg-tertiary)' }}
              >
                <div 
                  className="text-[9px] uppercase tracking-wider mb-0.5"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Max Drawdown
                </div>
                <div 
                  className="text-sm font-semibold"
                  style={{ color: 'var(--danger)' }}
                >
                  {maxDrawdown.toFixed(1)}%
                </div>
              </div>
              <div 
                className="p-2 rounded-lg"
                style={{ backgroundColor: 'var(--bg-tertiary)' }}
              >
                <div 
                  className="text-[9px] uppercase tracking-wider mb-0.5"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Sharpe Ratio
                </div>
                <div 
                  className="text-sm font-semibold"
                  style={{ color: sharpe >= 1 ? 'var(--success)' : 'var(--text-primary)' }}
                >
                  {sharpe.toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          {/* Strategy Description */}
          <div 
            className="mt-3 p-3 rounded-lg"
            style={{ 
              backgroundColor: isDark ? config.bgDark : config.bgLight,
              borderLeft: `3px solid ${config.color}`
            }}
          >
            <div 
              className="text-[10px] font-semibold mb-1"
              style={{ color: config.color }}
            >
              Strategy Logic
            </div>
            <p 
              className="text-xs leading-relaxed"
              style={{ color: 'var(--text-secondary)' }}
            >
              {strategy === 'Signal Follower' && 'Uses the 11-agent LLM pipeline to generate trading signals. Follows aggressive/neutral/conservative evaluator consensus.'}
              {strategy === 'Cooperator' && 'Blends LLM signals with group consensus. Weights decisions toward the average of all strategies for stable, cooperative behavior.'}
              {strategy === 'Defector' && 'Takes contrarian positions against group consensus. Bets opposite to what most strategies are doing. High risk, potential high reward.'}
              {strategy === 'Tit-for-Tat' && 'Classic game theory strategy. Copies the position of whichever strategy won the previous round. Adapts to changing market conditions.'}
              {strategy === 'Buy-and-Hold' && 'Passive benchmark strategy. Always maintains 100% market exposure regardless of signals or conditions.'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}