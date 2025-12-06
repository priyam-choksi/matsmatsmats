import React, { useState } from 'react'
import { useTheme } from '../../App'

// Strategy configurations
const STRATEGY_CONFIG = {
  'Signal Follower': { 
    color: '#9b59b6', 
    icon: '🤖',
    gradient: 'linear-gradient(90deg, #9b59b6, #8e44ad)'
  },
  'Cooperator': { 
    color: '#27ae60', 
    icon: '🤝',
    gradient: 'linear-gradient(90deg, #27ae60, #1e8449)'
  },
  'Defector': { 
    color: '#e74c3c', 
    icon: '🎯',
    gradient: 'linear-gradient(90deg, #e74c3c, #c0392b)'
  },
  'Tit-for-Tat': { 
    color: '#3498db', 
    icon: '🔄',
    gradient: 'linear-gradient(90deg, #3498db, #2980b9)'
  },
  'Buy-and-Hold': { 
    color: '#64748b', 
    icon: '📊',
    gradient: 'linear-gradient(90deg, #64748b, #475569)'
  }
}

// Format currency
const formatCurrency = (value) => {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`
  if (value >= 1000) return `$${(value / 1000).toFixed(0)}K`
  return `$${value.toFixed(0)}`
}

export default function AllocationBar({ 
  allocations = {},
  totalCapital = 1000000,
  showLabels = true,
  showLegend = true,
  showPercentages = true,
  height = 'normal', // 'slim', 'normal', 'tall'
  animated = true
}) {
  const { isDark } = useTheme()
  const [hoveredStrategy, setHoveredStrategy] = useState(null)

  // Calculate total and percentages
  const total = Object.values(allocations).reduce((sum, val) => sum + (val || 0), 0) || totalCapital
  
  // Sort strategies by allocation (highest first)
  const sortedStrategies = Object.entries(allocations)
    .sort(([, a], [, b]) => b - a)
    .map(([strategy, value]) => ({
      strategy,
      value: value || 0,
      percentage: ((value || 0) / total) * 100,
      config: STRATEGY_CONFIG[strategy] || { color: '#64748b', icon: '📊', gradient: '#64748b' }
    }))

  // Height mapping
  const barHeight = height === 'slim' ? 'h-2' : height === 'tall' ? 'h-6' : 'h-4'
  const barHeightPx = height === 'slim' ? 8 : height === 'tall' ? 24 : 16

  // If no data, show placeholder
  if (sortedStrategies.length === 0) {
    return (
      <div 
        className="w-full rounded-full overflow-hidden"
        style={{ 
          height: barHeightPx,
          backgroundColor: 'var(--bg-tertiary)'
        }}
      >
        <div 
          className="h-full flex items-center justify-center text-[10px]"
          style={{ color: 'var(--text-muted)' }}
        >
          No allocation data
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      {/* Main Bar */}
      <div 
        className={`w-full ${barHeight} rounded-full overflow-hidden flex`}
        style={{ backgroundColor: 'var(--bg-tertiary)' }}
      >
        {sortedStrategies.map(({ strategy, percentage, config }, index) => (
          <div
            key={strategy}
            className="h-full relative transition-all duration-300 cursor-pointer"
            style={{ 
              width: `${percentage}%`,
              background: config.gradient,
              opacity: hoveredStrategy && hoveredStrategy !== strategy ? 0.5 : 1,
              transform: hoveredStrategy === strategy ? 'scaleY(1.2)' : 'scaleY(1)',
              zIndex: hoveredStrategy === strategy ? 10 : 1
            }}
            onMouseEnter={() => setHoveredStrategy(strategy)}
            onMouseLeave={() => setHoveredStrategy(null)}
          >
            {/* Percentage label on bar (only for tall bars) */}
            {height === 'tall' && percentage > 10 && (
              <span 
                className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white"
                style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}
              >
                {percentage.toFixed(0)}%
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Hover Tooltip */}
      {hoveredStrategy && (
        <div 
          className="mt-2 p-2 rounded-lg inline-flex items-center gap-2 animate-fadeIn"
          style={{ backgroundColor: 'var(--bg-tertiary)' }}
        >
          <span className="text-base">
            {STRATEGY_CONFIG[hoveredStrategy]?.icon}
          </span>
          <span 
            className="text-xs font-semibold"
            style={{ color: STRATEGY_CONFIG[hoveredStrategy]?.color }}
          >
            {hoveredStrategy}
          </span>
          <span 
            className="text-xs"
            style={{ color: 'var(--text-secondary)' }}
          >
            {formatCurrency(allocations[hoveredStrategy])}
          </span>
          <span 
            className="text-xs font-medium px-1.5 py-0.5 rounded"
            style={{ 
              backgroundColor: `${STRATEGY_CONFIG[hoveredStrategy]?.color}20`,
              color: STRATEGY_CONFIG[hoveredStrategy]?.color
            }}
          >
            {((allocations[hoveredStrategy] / total) * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Legend */}
      {showLegend && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
          {sortedStrategies.map(({ strategy, value, percentage, config }) => (
            <div
              key={strategy}
              className="flex items-center gap-2 cursor-pointer transition-opacity"
              style={{ 
                opacity: hoveredStrategy && hoveredStrategy !== strategy ? 0.4 : 1 
              }}
              onMouseEnter={() => setHoveredStrategy(strategy)}
              onMouseLeave={() => setHoveredStrategy(null)}
            >
              {/* Color dot */}
              <div 
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: config.color }}
              />
              
              {/* Strategy name */}
              <span 
                className="text-xs font-medium"
                style={{ color: 'var(--text-secondary)' }}
              >
                {strategy}
              </span>

              {/* Value */}
              {showLabels && (
                <span 
                  className="text-xs font-semibold"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {formatCurrency(value)}
                </span>
              )}

              {/* Percentage */}
              {showPercentages && (
                <span 
                  className="text-[10px] font-medium px-1.5 py-0.5 rounded"
                  style={{ 
                    backgroundColor: `${config.color}15`,
                    color: config.color
                  }}
                >
                  {percentage.toFixed(1)}%
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Total */}
      {showLabels && (
        <div 
          className="mt-3 pt-3 border-t flex items-center justify-between"
          style={{ borderColor: 'var(--border-primary)' }}
        >
          <span 
            className="text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            Total Capital
          </span>
          <span 
            className="text-sm font-bold"
            style={{ color: 'var(--text-primary)' }}
          >
            {formatCurrency(total)}
          </span>
        </div>
      )}
    </div>
  )
}

// Vertical Allocation Bars (alternative view)
export function AllocationBars({ 
  allocations = {},
  showChange = true,
  previousAllocations = null
}) {
  const { isDark } = useTheme()
  
  const total = Object.values(allocations).reduce((sum, val) => sum + (val || 0), 0) || 1000000
  
  const sortedStrategies = Object.entries(allocations)
    .sort(([, a], [, b]) => b - a)
    .map(([strategy, value]) => {
      const percentage = ((value || 0) / total) * 100
      const prevValue = previousAllocations?.[strategy] || value
      const change = prevValue ? ((value - prevValue) / prevValue) * 100 : 0
      
      return {
        strategy,
        value: value || 0,
        percentage,
        change,
        config: STRATEGY_CONFIG[strategy] || { color: '#64748b', icon: '📊' }
      }
    })

  return (
    <div className="space-y-3">
      {sortedStrategies.map(({ strategy, value, percentage, change, config }) => (
        <div key={strategy}>
          {/* Label row */}
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-2">
              <span className="text-base">{config.icon}</span>
              <span 
                className="text-sm font-semibold"
                style={{ color: config.color }}
              >
                {strategy}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span 
                className="text-sm font-semibold"
                style={{ color: 'var(--text-primary)' }}
              >
                {formatCurrency(value)}
              </span>
              {showChange && change !== 0 && (
                <span 
                  className="text-[10px] font-medium px-1.5 py-0.5 rounded"
                  style={{ 
                    backgroundColor: change >= 0 ? 'var(--success-bg)' : 'var(--danger-bg)',
                    color: change >= 0 ? 'var(--success)' : 'var(--danger)'
                  }}
                >
                  {change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
                </span>
              )}
            </div>
          </div>
          
          {/* Bar */}
          <div 
            className="h-2 rounded-full overflow-hidden"
            style={{ backgroundColor: 'var(--bg-tertiary)' }}
          >
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ 
                width: `${percentage}%`,
                backgroundColor: config.color
              }}
            />
          </div>
          
          {/* Percentage label */}
          <div className="flex justify-end mt-0.5">
            <span 
              className="text-[10px]"
              style={{ color: 'var(--text-muted)' }}
            >
              {percentage.toFixed(1)}% of total
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

// Mini allocation indicator (for compact displays)
export function AllocationMini({ allocations = {}, size = 100 }) {
  const total = Object.values(allocations).reduce((sum, val) => sum + (val || 0), 0) || 1
  
  const segments = Object.entries(allocations)
    .sort(([, a], [, b]) => b - a)
    .map(([strategy, value]) => ({
      strategy,
      percentage: ((value || 0) / total) * 100,
      color: STRATEGY_CONFIG[strategy]?.color || '#64748b'
    }))

  // Calculate stroke-dasharray for donut chart
  let cumulative = 0
  const circumference = 2 * Math.PI * 35 // radius = 35

  return (
    <svg width={size} height={size} viewBox="0 0 100 100">
      {/* Background circle */}
      <circle
        cx="50"
        cy="50"
        r="35"
        fill="none"
        stroke="var(--bg-tertiary)"
        strokeWidth="12"
      />
      
      {/* Segments */}
      {segments.map(({ strategy, percentage, color }) => {
        const dashLength = (percentage / 100) * circumference
        const dashOffset = -((cumulative / 100) * circumference) + (circumference / 4)
        cumulative += percentage
        
        return (
          <circle
            key={strategy}
            cx="50"
            cy="50"
            r="35"
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeDasharray={`${dashLength} ${circumference - dashLength}`}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dasharray 0.5s ease' }}
          />
        )
      })}
      
      {/* Center text */}
      <text
        x="50"
        y="47"
        textAnchor="middle"
        fontSize="12"
        fontWeight="600"
        fill="var(--text-primary)"
      >
        $1M
      </text>
      <text
        x="50"
        y="60"
        textAnchor="middle"
        fontSize="8"
        fill="var(--text-muted)"
      >
        Total
      </text>
    </svg>
  )
}