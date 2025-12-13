import React, { useState, useEffect } from 'react'
import { useTheme } from '../../App'
import Chat from '../ui/Chat'
import FeedItem from '../cards/FeedItem'

// Icons
const Icons = {
  Feed: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 11a9 9 0 0 1 9 9" /><path d="M4 4a16 16 0 0 1 16 16" /><circle cx="5" cy="19" r="1" />
    </svg>
  ),
  Portfolio: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="20" height="14" x="2" y="7" rx="2" ry="2" /><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  ),
  Chat: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  ),
  ChevronLeft: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m15 18-6-6 6-6" />
    </svg>
  ),
  ChevronRight: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m9 18 6-6-6-6" />
    </svg>
  ),
  TrendingUp: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" /><polyline points="16 7 22 7 22 13" />
    </svg>
  ),
  TrendingDown: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 17 13.5 8.5 8.5 13.5 2 7" /><polyline points="16 17 22 17 22 11" />
    </svg>
  )
}

// Sample feed data (in production, load from JSON)
const sampleFeed = [
  {
    id: 1,
    timestamp: '2025-12-03T14:32:00',
    strategy: 'Signal Follower',
    ticker: 'NVDA',
    action: 'BUY',
    position_pct: 65,
    reasoning: 'Strong technical breakout above $180 resistance with volume confirmation.',
    result: { return_pct: 2.14, won_round: true }
  },
  {
    id: 2,
    timestamp: '2025-12-03T11:15:00',
    strategy: 'Cooperator',
    ticker: 'AAPL',
    action: 'HOLD',
    position_pct: 42,
    reasoning: 'Following group consensus on mixed signals.',
    result: { return_pct: 0.38, won_round: false }
  },
  {
    id: 3,
    timestamp: '2025-12-03T09:45:00',
    strategy: 'Defector',
    ticker: 'TSLA',
    action: 'REDUCE',
    position_pct: 15,
    reasoning: 'Contrarian position against bullish consensus.',
    result: { return_pct: -1.24, won_round: false }
  },
  {
    id: 4,
    timestamp: '2025-12-02T16:20:00',
    strategy: 'Tit-for-Tat',
    ticker: 'GOOGL',
    action: 'BUY',
    position_pct: 55,
    reasoning: 'Mimicking Signal Follower from previous winning round.',
    result: { return_pct: 1.67, won_round: true }
  }
]

// Sample portfolio data
const samplePortfolios = [
  { strategy: 'Signal Follower', allocation: 287320, return_pct: 14.93, position_pct: 65, ticker: 'NVDA' },
  { strategy: 'Cooperator', allocation: 275840, return_pct: 10.34, position_pct: 42, ticker: 'AAPL' },
  { strategy: 'Tit-for-Tat', allocation: 218418, return_pct: -12.64, position_pct: 55, ticker: 'GOOGL' },
  { strategy: 'Defector', allocation: 218420, return_pct: -12.63, position_pct: 15, ticker: 'TSLA' }
]

const tabs = [
  { id: 'feed', label: 'Feed', icon: 'Feed' },
  { id: 'portfolios', label: 'Portfolios', icon: 'Portfolio' },
  { id: 'chat', label: 'Ask AI', icon: 'Chat' }
]

// Strategy color mapping
const strategyColors = {
  'Signal Follower': '#9b59b6',
  'Cooperator': '#27ae60',
  'Defector': '#e74c3c',
  'Tit-for-Tat': '#3498db'
}

export default function RightSidebar({ isOpen = true, onToggle }) {
  const { isDark } = useTheme()
  const [activeTab, setActiveTab] = useState('feed')
  const [feed, setFeed] = useState(sampleFeed)
  const [portfolios, setPortfolios] = useState(samplePortfolios)

  // Load feed data from JSON (if available)
  useEffect(() => {
    const loadFeed = async () => {
      try {
        const response = await fetch('/data/feed/recent_decisions.json')
        if (response.ok) {
          const data = await response.json()
          if (data.activities && data.activities.length > 0) {
            setFeed(data.activities)
          }
        }
      } catch (error) {
        console.log('Using sample feed data')
      }
    }
    loadFeed()
  }, [])

  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        className="fixed right-0 top-1/2 -translate-y-1/2 w-6 h-16 rounded-l-lg flex items-center justify-center z-50 transition-colors"
        style={{
          backgroundColor: 'var(--bg-card)',
          borderColor: 'var(--border-primary)',
          borderWidth: '1px 0 1px 1px',
          color: 'var(--text-tertiary)'
        }}
      >
        <Icons.ChevronLeft size={14} />
      </button>
    )
  }

  return (
    <aside
      className="w-80 flex flex-col border-l"
      style={{
        backgroundColor: 'var(--bg-card)',
        borderColor: 'var(--border-primary)'
      }}
    >
      {/* Header with collapse button */}
      <div 
        className="h-12 px-4 flex items-center justify-between border-b"
        style={{ borderColor: 'var(--border-primary)' }}
      >
        <span 
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: 'var(--text-tertiary)' }}
        >
          Activity
        </span>
        <button
          onClick={onToggle}
          className="w-6 h-6 rounded flex items-center justify-center transition-colors"
          style={{ color: 'var(--text-tertiary)' }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-hover)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          <Icons.ChevronRight size={14} />
        </button>
      </div>

      {/* Tabs */}
      <div 
        className="flex border-b"
        style={{ borderColor: 'var(--border-primary)' }}
      >
        {tabs.map(tab => {
          const Icon = Icons[tab.icon]
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="flex-1 py-3 flex items-center justify-center gap-1.5 text-xs font-medium transition-all relative"
              style={{
                color: isActive ? 'var(--accent-primary)' : 'var(--text-tertiary)',
                backgroundColor: isActive ? (isDark ? 'rgba(99, 102, 241, 0.05)' : 'rgba(99, 102, 241, 0.05)') : 'transparent'
              }}
            >
              <Icon size={14} />
              <span>{tab.label}</span>
              {isActive && (
                <div 
                  className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{ backgroundColor: 'var(--accent-primary)' }}
                />
              )}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Feed Tab */}
        {activeTab === 'feed' && (
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            <p 
              className="text-[10px] px-1 mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Recent strategy decisions and outcomes
            </p>
            {feed.map((item, i) => (
              <FeedItem key={item.id || i} item={item} />
            ))}
          </div>
        )}

        {/* Portfolios Tab */}
        {activeTab === 'portfolios' && (
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            <p 
              className="text-[10px] px-1 mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Current strategy allocations
            </p>
            {portfolios.map((p, i) => (
              <div
                key={i}
                className="p-3 rounded-lg transition-colors"
                style={{ backgroundColor: 'var(--bg-tertiary)' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: strategyColors[p.strategy] }}
                    />
                    <span 
                      className="text-sm font-semibold"
                      style={{ color: strategyColors[p.strategy] }}
                    >
                      {p.strategy}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {p.return_pct >= 0 ? (
                      <Icons.TrendingUp size={12} style={{ color: 'var(--success)' }} />
                    ) : (
                      <Icons.TrendingDown size={12} style={{ color: 'var(--danger)' }} />
                    )}
                    <span 
                      className="text-xs font-semibold"
                      style={{ color: p.return_pct >= 0 ? 'var(--success)' : 'var(--danger)' }}
                    >
                      {p.return_pct >= 0 ? '+' : ''}{p.return_pct}%
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs mb-2">
                  <span style={{ color: 'var(--text-tertiary)' }}>Allocation</span>
                  <span style={{ color: 'var(--text-primary)' }} className="font-medium">
                    ${p.allocation.toLocaleString()}
                  </span>
                </div>
                
                <div className="flex items-center justify-between text-xs">
                  <span style={{ color: 'var(--text-tertiary)' }}>Position</span>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {p.position_pct}% in {p.ticker}
                  </span>
                </div>

                {/* Mini progress bar */}
                <div 
                  className="h-1 rounded-full mt-2 overflow-hidden"
                  style={{ backgroundColor: 'var(--bg-hover)' }}
                >
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ 
                      width: `${p.position_pct}%`,
                      backgroundColor: strategyColors[p.strategy],
                      opacity: 0.7
                    }}
                  />
                </div>
              </div>
            ))}

            {/* Total Capital */}
            <div 
              className="mt-3 p-3 rounded-lg border"
              style={{ 
                backgroundColor: isDark ? 'rgba(99, 102, 241, 0.05)' : 'rgba(99, 102, 241, 0.05)',
                borderColor: 'var(--accent-primary)'
              }}
            >
              <div className="flex items-center justify-between">
                <span 
                  className="text-xs font-medium"
                  style={{ color: 'var(--text-tertiary)' }}
                >
                  Total Capital
                </span>
                <span 
                  className="text-sm font-bold"
                  style={{ color: 'var(--accent-primary)' }}
                >
                  $1,000,000
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <Chat />
        )}
      </div>

      {/* Footer */}
      <div 
        className="px-3 py-2 border-t text-center"
        style={{ borderColor: 'var(--border-primary)' }}
      >
        <span 
          className="text-[9px]"
          style={{ color: 'var(--text-muted)' }}
        >
          Data refreshes every 30s • {new Date().toLocaleTimeString()}
        </span>
      </div>
    </aside>
  )
}