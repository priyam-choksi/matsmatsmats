import React, { useState, useMemo } from 'react'
import { useTheme } from '../../App'

// Regime configurations
const REGIME_CONFIG = {
  bull: {
    color: '#10b981',
    bgLight: 'rgba(16, 185, 129, 0.15)',
    bgDark: 'rgba(16, 185, 129, 0.25)',
    icon: '🟢',
    label: 'Bull',
    description: 'Upward trending market'
  },
  bear: {
    color: '#ef4444',
    bgLight: 'rgba(239, 68, 68, 0.15)',
    bgDark: 'rgba(239, 68, 68, 0.25)',
    icon: '🔴',
    label: 'Bear',
    description: 'Downward trending market'
  },
  sideways: {
    color: '#f59e0b',
    bgLight: 'rgba(245, 158, 11, 0.15)',
    bgDark: 'rgba(245, 158, 11, 0.25)',
    icon: '🟡',
    label: 'Sideways',
    description: 'Consolidating market'
  }
}

// Generate sample data if none provided
const generateSampleData = (rounds = 90) => {
  const data = []
  const regimes = ['bull', 'bear', 'sideways']
  let currentRegime = regimes[Math.floor(Math.random() * 3)]
  
  for (let i = 1; i <= rounds; i++) {
    // Occasionally change regime
    if (Math.random() < 0.08) {
      currentRegime = regimes[Math.floor(Math.random() * 3)]
    }
    
    data.push({
      round: i,
      date: `2025-${String(Math.floor(i / 30) + 7).padStart(2, '0')}-${String((i % 28) + 1).padStart(2, '0')}`,
      regime: currentRegime,
      return: (Math.random() - 0.48) * 4
    })
  }
  
  return data
}

export default function RegimeTimeline({
  data = null,
  height = 'normal', // 'slim', 'normal', 'tall'
  showLegend = true,
  showStats = true,
  showTooltip = true,
  onRoundClick = null
}) {
  const { isDark } = useTheme()
  const [hoveredRound, setHoveredRound] = useState(null)
  
  // Use provided data or generate sample
  const timelineData = useMemo(() => data || generateSampleData(), [data])
  
  // Calculate regime statistics
  const stats = useMemo(() => {
    const counts = { bull: 0, bear: 0, sideways: 0 }
    const returns = { bull: [], bear: [], sideways: [] }
    
    timelineData.forEach(d => {
      if (d.regime && counts[d.regime] !== undefined) {
        counts[d.regime]++
        if (d.return != null) {
          returns[d.regime].push(d.return)
        }
      }
    })
    
    const total = timelineData.length
    
    return Object.entries(counts).map(([regime, count]) => {
      const avgReturn = returns[regime].length > 0 
        ? returns[regime].reduce((a, b) => a + b, 0) / returns[regime].length 
        : 0
      
      return {
        regime,
        count,
        percentage: (count / total) * 100,
        avgReturn,
        config: REGIME_CONFIG[regime]
      }
    })
  }, [timelineData])

  // Group consecutive regimes for better visualization
  const groupedRegimes = useMemo(() => {
    const groups = []
    let currentGroup = null
    
    timelineData.forEach((d, i) => {
      if (!currentGroup || currentGroup.regime !== d.regime) {
        if (currentGroup) {
          currentGroup.endIndex = i - 1
          groups.push(currentGroup)
        }
        currentGroup = {
          regime: d.regime,
          startIndex: i,
          startRound: d.round,
          config: REGIME_CONFIG[d.regime] || REGIME_CONFIG.sideways
        }
      }
    })
    
    if (currentGroup) {
      currentGroup.endIndex = timelineData.length - 1
      groups.push(currentGroup)
    }
    
    return groups
  }, [timelineData])

  // Height mapping
  const barHeight = height === 'slim' ? 8 : height === 'tall' ? 24 : 16

  return (
    <div className="w-full">
      {/* Timeline Bar */}
      <div 
        className="w-full rounded-lg overflow-hidden flex"
        style={{ 
          height: barHeight,
          backgroundColor: 'var(--bg-tertiary)'
        }}
      >
        {timelineData.map((d, i) => {
          const config = REGIME_CONFIG[d.regime] || REGIME_CONFIG.sideways
          const isHovered = hoveredRound === i
          
          return (
            <div
              key={i}
              className="h-full transition-all duration-150 cursor-pointer"
              style={{
                flex: 1,
                backgroundColor: isDark ? config.bgDark : config.bgLight,
                borderLeft: i > 0 && timelineData[i - 1]?.regime !== d.regime 
                  ? `1px solid ${config.color}` 
                  : 'none',
                opacity: hoveredRound !== null && !isHovered ? 0.5 : 1,
                transform: isHovered ? 'scaleY(1.3)' : 'scaleY(1)',
                zIndex: isHovered ? 10 : 1
              }}
              onMouseEnter={() => setHoveredRound(i)}
              onMouseLeave={() => setHoveredRound(null)}
              onClick={() => onRoundClick?.(d, i)}
            />
          )
        })}
      </div>

      {/* Round labels */}
      <div className="flex justify-between mt-1.5 px-0.5">
        <span 
          className="text-[9px]"
          style={{ color: 'var(--text-muted)' }}
        >
          R1
        </span>
        <span 
          className="text-[9px]"
          style={{ color: 'var(--text-muted)' }}
        >
          R{Math.floor(timelineData.length / 2)}
        </span>
        <span 
          className="text-[9px]"
          style={{ color: 'var(--text-muted)' }}
        >
          R{timelineData.length}
        </span>
      </div>

      {/* Tooltip */}
      {showTooltip && hoveredRound !== null && timelineData[hoveredRound] && (
        <div 
          className="mt-2 p-2.5 rounded-lg inline-flex items-center gap-3 animate-fadeIn"
          style={{ backgroundColor: 'var(--bg-tertiary)' }}
        >
          <div className="flex items-center gap-1.5">
            <span className="text-sm">
              {REGIME_CONFIG[timelineData[hoveredRound].regime]?.icon}
            </span>
            <span 
              className="text-xs font-semibold"
              style={{ color: REGIME_CONFIG[timelineData[hoveredRound].regime]?.color }}
            >
              {REGIME_CONFIG[timelineData[hoveredRound].regime]?.label}
            </span>
          </div>
          
          <div 
            className="text-[10px]"
            style={{ color: 'var(--text-secondary)' }}
          >
            Round {timelineData[hoveredRound].round}
          </div>
          
          {timelineData[hoveredRound].date && (
            <div 
              className="text-[10px]"
              style={{ color: 'var(--text-muted)' }}
            >
              {timelineData[hoveredRound].date}
            </div>
          )}
          
          {timelineData[hoveredRound].return != null && (
            <div 
              className="text-[10px] font-semibold"
              style={{ 
                color: timelineData[hoveredRound].return >= 0 
                  ? 'var(--success)' 
                  : 'var(--danger)' 
              }}
            >
              {timelineData[hoveredRound].return >= 0 ? '+' : ''}
              {timelineData[hoveredRound].return.toFixed(2)}%
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      {showLegend && (
        <div className="flex flex-wrap gap-4 mt-3">
          {Object.entries(REGIME_CONFIG).map(([regime, config]) => (
            <div 
              key={regime}
              className="flex items-center gap-1.5"
            >
              <div 
                className="w-3 h-3 rounded"
                style={{ backgroundColor: isDark ? config.bgDark : config.bgLight, border: `1px solid ${config.color}` }}
              />
              <span 
                className="text-xs"
                style={{ color: 'var(--text-secondary)' }}
              >
                {config.label}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Statistics */}
      {showStats && (
        <div className="grid grid-cols-3 gap-3 mt-4">
          {stats.map(({ regime, count, percentage, avgReturn, config }) => (
            <div
              key={regime}
              className="p-3 rounded-lg"
              style={{ 
                backgroundColor: isDark ? config.bgDark : config.bgLight,
                border: `1px solid ${config.color}30`
              }}
            >
              <div className="flex items-center gap-1.5 mb-2">
                <span className="text-sm">{config.icon}</span>
                <span 
                  className="text-xs font-semibold"
                  style={{ color: config.color }}
                >
                  {config.label}
                </span>
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span 
                    className="text-[10px]"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Rounds
                  </span>
                  <span 
                    className="text-xs font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {count} ({percentage.toFixed(0)}%)
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span 
                    className="text-[10px]"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Avg Return
                  </span>
                  <span 
                    className="text-xs font-semibold"
                    style={{ color: avgReturn >= 0 ? 'var(--success)' : 'var(--danger)' }}
                  >
                    {avgReturn >= 0 ? '+' : ''}{avgReturn.toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Compact regime badge (for use in other components)
export function RegimeBadge({ regime, size = 'normal' }) {
  const config = REGIME_CONFIG[regime] || REGIME_CONFIG.sideways
  const { isDark } = useTheme()
  
  const sizeClasses = {
    small: 'text-[9px] px-1.5 py-0.5',
    normal: 'text-[10px] px-2 py-1',
    large: 'text-xs px-2.5 py-1.5'
  }

  return (
    <span
      className={`inline-flex items-center gap-1 rounded font-semibold ${sizeClasses[size]}`}
      style={{ 
        backgroundColor: isDark ? config.bgDark : config.bgLight,
        color: config.color
      }}
    >
      <span>{config.icon}</span>
      <span>{config.label}</span>
    </span>
  )
}

// Regime distribution pie/donut
export function RegimeDistribution({ data = null, size = 120 }) {
  const { isDark } = useTheme()
  const timelineData = useMemo(() => data || generateSampleData(), [data])
  
  // Calculate percentages
  const counts = { bull: 0, bear: 0, sideways: 0 }
  timelineData.forEach(d => {
    if (d.regime && counts[d.regime] !== undefined) {
      counts[d.regime]++
    }
  })
  
  const total = timelineData.length
  const segments = Object.entries(counts)
    .map(([regime, count]) => ({
      regime,
      percentage: (count / total) * 100,
      config: REGIME_CONFIG[regime]
    }))
    .sort((a, b) => b.percentage - a.percentage)

  // SVG donut chart
  const radius = 40
  const circumference = 2 * Math.PI * radius
  let cumulative = 0

  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} viewBox="0 0 100 100">
        {/* Background */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke="var(--bg-tertiary)"
          strokeWidth="14"
        />
        
        {/* Segments */}
        {segments.map(({ regime, percentage, config }) => {
          const dashLength = (percentage / 100) * circumference
          const dashOffset = -((cumulative / 100) * circumference) + (circumference / 4)
          cumulative += percentage
          
          return (
            <circle
              key={regime}
              cx="50"
              cy="50"
              r={radius}
              fill="none"
              stroke={config.color}
              strokeWidth="14"
              strokeDasharray={`${dashLength} ${circumference - dashLength}`}
              strokeDashoffset={dashOffset}
              strokeLinecap="round"
              style={{ transition: 'stroke-dasharray 0.5s ease' }}
            />
          )
        })}
        
        {/* Center text */}
        <text
          x="50"
          y="46"
          textAnchor="middle"
          fontSize="14"
          fontWeight="700"
          fill="var(--text-primary)"
        >
          {total}
        </text>
        <text
          x="50"
          y="60"
          textAnchor="middle"
          fontSize="8"
          fill="var(--text-muted)"
        >
          rounds
        </text>
      </svg>
      
      {/* Legend */}
      <div className="space-y-2">
        {segments.map(({ regime, percentage, config }) => (
          <div key={regime} className="flex items-center gap-2">
            <div 
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: config.color }}
            />
            <span 
              className="text-xs"
              style={{ color: 'var(--text-secondary)' }}
            >
              {config.label}
            </span>
            <span 
              className="text-xs font-semibold"
              style={{ color: 'var(--text-primary)' }}
            >
              {percentage.toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}