import React, { useState } from 'react'
import { useTheme } from '../../App'

// Icons
const Icons = {
  TrendingUp: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" /><polyline points="16 7 22 7 22 13" />
    </svg>
  ),
  TrendingDown: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 17 13.5 8.5 8.5 13.5 2 7" /><polyline points="16 17 22 17 22 11" />
    </svg>
  ),
  Trophy: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" /><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" /><path d="M4 22h16" /><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" /><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" /><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
    </svg>
  ),
  ChevronDown: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
  Clock: ({ size = 10 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
    </svg>
  )
}

// Strategy colors
const strategyColors = {
  'Signal Follower': { color: '#9b59b6', bg: 'rgba(155, 89, 182, 0.1)' },
  'Cooperator': { color: '#27ae60', bg: 'rgba(39, 174, 96, 0.1)' },
  'Defector': { color: '#e74c3c', bg: 'rgba(231, 76, 60, 0.1)' },
  'Tit-for-Tat': { color: '#3498db', bg: 'rgba(52, 152, 219, 0.1)' }
}

// Action badge colors
const actionColors = {
  'BUY': { color: 'var(--success)', bg: 'var(--success-bg)' },
  'SELL': { color: 'var(--danger)', bg: 'var(--danger-bg)' },
  'HOLD': { color: 'var(--warning)', bg: 'var(--warning-bg)' },
  'REDUCE': { color: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
  'INCREASE': { color: '#22c55e', bg: 'rgba(34, 197, 94, 0.1)' }
}

// Format timestamp to relative time
const formatTimeAgo = (timestamp) => {
  if (!timestamp) return ''
  
  const now = new Date()
  const date = new Date(timestamp)
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export default function FeedItem({ item }) {
  const { isDark } = useTheme()
  const [isExpanded, setIsExpanded] = useState(false)
  
  const strategy = strategyColors[item.strategy] || { color: '#6b7280', bg: 'rgba(107, 114, 128, 0.1)' }
  const action = actionColors[item.action] || actionColors['HOLD']
  const hasResult = item.result && typeof item.result.return_pct === 'number'
  const isPositive = hasResult && item.result.return_pct >= 0
  const wonRound = item.result?.won_round

  return (
    <div
      className="rounded-lg overflow-hidden transition-all duration-200"
      style={{ 
        backgroundColor: 'var(--bg-tertiary)',
        border: wonRound ? `1px solid ${strategy.color}40` : '1px solid transparent'
      }}
    >
      {/* Main Content - Always Visible */}
      <div 
        className="p-3 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {/* Header Row */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {/* Strategy Badge */}
            <span
              className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
              style={{ 
                color: strategy.color,
                backgroundColor: strategy.bg
              }}
            >
              {item.strategy}
            </span>
            
            {/* Win indicator */}
            {wonRound && (
              <span 
                className="flex items-center gap-0.5 text-[9px] font-semibold px-1 py-0.5 rounded"
                style={{ 
                  color: '#f59e0b',
                  backgroundColor: 'rgba(245, 158, 11, 0.1)'
                }}
              >
                <Icons.Trophy size={9} />
                WON
              </span>
            )}
          </div>
          
          {/* Timestamp */}
          <div 
            className="flex items-center gap-1 text-[9px]"
            style={{ color: 'var(--text-muted)' }}
          >
            <Icons.Clock size={9} />
            {formatTimeAgo(item.timestamp)}
          </div>
        </div>

        {/* Action Row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Action Badge */}
            <span
              className="text-[10px] font-bold px-1.5 py-0.5 rounded"
              style={{ 
                color: action.color,
                backgroundColor: action.bg
              }}
            >
              {item.action}
            </span>
            
            {/* Ticker */}
            <span 
              className="text-sm font-bold"
              style={{ color: 'var(--text-primary)' }}
            >
              {item.ticker}
            </span>
            
            {/* Position % */}
            <span 
              className="text-xs"
              style={{ color: 'var(--text-tertiary)' }}
            >
              @ {item.position_pct}%
            </span>
          </div>

          {/* Result */}
          {hasResult && (
            <div className="flex items-center gap-1">
              {isPositive ? (
                <Icons.TrendingUp size={12} style={{ color: 'var(--success)' }} />
              ) : (
                <Icons.TrendingDown size={12} style={{ color: 'var(--danger)' }} />
              )}
              <span 
                className="text-xs font-semibold"
                style={{ color: isPositive ? 'var(--success)' : 'var(--danger)' }}
              >
                {isPositive ? '+' : ''}{item.result.return_pct.toFixed(2)}%
              </span>
            </div>
          )}
        </div>

        {/* Expand indicator */}
        {item.reasoning && (
          <div 
            className="flex items-center justify-center mt-2 pt-2 border-t"
            style={{ borderColor: 'var(--border-primary)' }}
          >
            <div 
              className="flex items-center gap-1 text-[9px] font-medium transition-transform"
              style={{ 
                color: 'var(--text-muted)',
                transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)'
              }}
            >
              <Icons.ChevronDown size={12} />
            </div>
          </div>
        )}
      </div>

      {/* Expanded Content */}
      {isExpanded && item.reasoning && (
        <div 
          className="px-3 pb-3 pt-0 border-t"
          style={{ borderColor: 'var(--border-primary)' }}
        >
          {/* Reasoning */}
          <div className="mt-2">
            <span 
              className="text-[9px] font-semibold uppercase tracking-wider"
              style={{ color: 'var(--text-muted)' }}
            >
              Reasoning
            </span>
            <p 
              className="text-[11px] mt-1 leading-relaxed"
              style={{ color: 'var(--text-secondary)' }}
            >
              {item.reasoning}
            </p>
          </div>

          {/* Additional Details */}
          {item.round_num && (
            <div 
              className="flex items-center gap-4 mt-3 pt-2 border-t"
              style={{ borderColor: 'var(--border-primary)' }}
            >
              <div className="flex items-center gap-1">
                <span 
                  className="text-[9px]"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Round
                </span>
                <span 
                  className="text-[10px] font-semibold"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  #{item.round_num}
                </span>
              </div>
              
              {item.regime && (
                <div className="flex items-center gap-1">
                  <span 
                    className="text-[9px]"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Regime
                  </span>
                  <span 
                    className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${
                      item.regime === 'bull' ? 'regime-bull' :
                      item.regime === 'bear' ? 'regime-bear' : 'regime-sideways'
                    }`}
                  >
                    {item.regime?.toUpperCase()}
                  </span>
                </div>
              )}

              {item.confidence && (
                <div className="flex items-center gap-1">
                  <span 
                    className="text-[9px]"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Confidence
                  </span>
                  <span 
                    className="text-[10px] font-semibold"
                    style={{ 
                      color: item.confidence === 'HIGH' ? 'var(--success)' :
                             item.confidence === 'LOW' ? 'var(--danger)' : 'var(--warning)'
                    }}
                  >
                    {item.confidence}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Key Factors (if available) */}
          {item.keyFactors && item.keyFactors.length > 0 && (
            <div className="mt-3">
              <span 
                className="text-[9px] font-semibold uppercase tracking-wider"
                style={{ color: 'var(--text-muted)' }}
              >
                Key Factors
              </span>
              <div className="flex flex-wrap gap-1 mt-1">
                {item.keyFactors.map((factor, i) => (
                  <span
                    key={i}
                    className="text-[9px] px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor: 'var(--bg-hover)',
                      color: 'var(--text-secondary)'
                    }}
                  >
                    ✓ {factor}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}