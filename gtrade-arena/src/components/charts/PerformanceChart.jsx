import React, { useState, useMemo } from 'react'
import { useTheme } from '../../App'

// Strategy colors
const STRATEGY_COLORS = {
  'Signal Follower': '#9b59b6',
  'Cooperator': '#27ae60',
  'Defector': '#e74c3c',
  'Tit-for-Tat': '#3498db',
  'Buy-and-Hold': '#64748b'
}

// Sample data generator (when no data provided)
const generateSampleData = () => {
  const data = []
  const strategies = ['Signal Follower', 'Cooperator', 'Defector', 'Tit-for-Tat']
  
  // Starting values
  const values = {
    'Signal Follower': 250000,
    'Cooperator': 250000,
    'Defector': 250000,
    'Tit-for-Tat': 250000
  }
  
  for (let round = 0; round <= 90; round += 5) {
    const point = { round }
    
    strategies.forEach(strategy => {
      // Simulate different growth patterns
      const volatility = strategy === 'Defector' ? 0.03 : 0.015
      const trend = strategy === 'Signal Follower' ? 0.002 : 
                    strategy === 'Cooperator' ? 0.0015 :
                    strategy === 'Defector' ? -0.001 : 0.0005
      
      const change = (Math.random() - 0.45) * volatility + trend
      values[strategy] *= (1 + change)
      point[strategy] = Math.round(values[strategy])
    })
    
    data.push(point)
  }
  
  return data
}

export default function PerformanceChart({ 
  data = null, 
  strategies = ['Signal Follower', 'Cooperator', 'Defector', 'Tit-for-Tat'],
  height = 300,
  showLegend = true,
  showGrid = true,
  showTooltip = true,
  animate = true
}) {
  const { isDark } = useTheme()
  const [hoveredPoint, setHoveredPoint] = useState(null)
  const [hoveredStrategy, setHoveredStrategy] = useState(null)
  
  // Use provided data or generate sample
  const chartData = useMemo(() => data || generateSampleData(), [data])
  
  // Chart dimensions
  const width = 700
  const padding = { top: 20, right: 80, bottom: 40, left: 60 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  // Calculate min/max values
  const allValues = chartData.flatMap(d => strategies.map(s => d[s]).filter(v => v != null))
  const minValue = Math.min(...allValues) * 0.98
  const maxValue = Math.max(...allValues) * 1.02
  
  // Scale functions
  const xScale = (index) => padding.left + (index / (chartData.length - 1)) * chartWidth
  const yScale = (value) => padding.top + chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight

  // Generate path for each strategy
  const generatePath = (strategy) => {
    const points = chartData
      .map((d, i) => d[strategy] != null ? `${xScale(i)},${yScale(d[strategy])}` : null)
      .filter(p => p != null)
    
    return `M${points.join(' L')}`
  }

  // Generate area path
  const generateAreaPath = (strategy) => {
    const points = chartData
      .map((d, i) => d[strategy] != null ? { x: xScale(i), y: yScale(d[strategy]) } : null)
      .filter(p => p != null)
    
    if (points.length === 0) return ''
    
    const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ')
    return `${linePath} L${points[points.length - 1].x},${padding.top + chartHeight} L${points[0].x},${padding.top + chartHeight} Z`
  }

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const ticks = []
    const range = maxValue - minValue
    const step = range / 4
    for (let i = 0; i <= 4; i++) {
      ticks.push(minValue + step * i)
    }
    return ticks
  }, [minValue, maxValue])

  // X-axis ticks (show every nth round)
  const xTicks = useMemo(() => {
    const step = Math.ceil(chartData.length / 6)
    return chartData.filter((_, i) => i % step === 0 || i === chartData.length - 1)
  }, [chartData])

  // Format currency
  const formatValue = (v) => {
    if (v >= 1000000) return `$${(v / 1000000).toFixed(2)}M`
    if (v >= 1000) return `$${(v / 1000).toFixed(0)}K`
    return `$${v.toFixed(0)}`
  }

  // Handle mouse move
  const handleMouseMove = (e) => {
    if (!showTooltip) return
    
    const svg = e.currentTarget
    const rect = svg.getBoundingClientRect()
    const x = e.clientX - rect.left
    
    // Find closest data point
    const index = Math.round(((x - padding.left) / chartWidth) * (chartData.length - 1))
    const clampedIndex = Math.max(0, Math.min(chartData.length - 1, index))
    
    setHoveredPoint({ index: clampedIndex, data: chartData[clampedIndex] })
  }

  return (
    <div className="w-full">
      {/* Legend */}
      {showLegend && (
        <div className="flex flex-wrap gap-4 mb-4 px-2">
          {strategies.map(strategy => {
            const lastValue = chartData[chartData.length - 1]?.[strategy]
            const firstValue = chartData[0]?.[strategy]
            const change = firstValue ? ((lastValue - firstValue) / firstValue * 100) : 0
            
            return (
              <div 
                key={strategy}
                className="flex items-center gap-2 cursor-pointer transition-opacity"
                style={{ opacity: hoveredStrategy && hoveredStrategy !== strategy ? 0.3 : 1 }}
                onMouseEnter={() => setHoveredStrategy(strategy)}
                onMouseLeave={() => setHoveredStrategy(null)}
              >
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: STRATEGY_COLORS[strategy] }}
                />
                <span 
                  className="text-xs font-medium"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {strategy}
                </span>
                <span 
                  className="text-xs font-semibold"
                  style={{ color: change >= 0 ? 'var(--success)' : 'var(--danger)' }}
                >
                  {change >= 0 ? '+' : ''}{change.toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>
      )}

      {/* Chart */}
      <svg 
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ maxHeight: height }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredPoint(null)}
      >
        {/* Background */}
        <rect 
          x={padding.left} 
          y={padding.top} 
          width={chartWidth} 
          height={chartHeight}
          fill={isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)'}
          rx="4"
        />

        {/* Grid lines */}
        {showGrid && yTicks.map((tick, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={yScale(tick)}
              x2={padding.left + chartWidth}
              y2={yScale(tick)}
              stroke={isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'}
              strokeDasharray="4,4"
            />
            <text
              x={padding.left - 8}
              y={yScale(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize="10"
              fill="var(--text-muted)"
            >
              {formatValue(tick)}
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {xTicks.map((d, i) => {
          const index = chartData.indexOf(d)
          return (
            <text
              key={i}
              x={xScale(index)}
              y={height - 10}
              textAnchor="middle"
              fontSize="10"
              fill="var(--text-muted)"
            >
              R{d.round}
            </text>
          )
        })}

        {/* Area fills (subtle) */}
        {strategies.map(strategy => (
          <path
            key={`area-${strategy}`}
            d={generateAreaPath(strategy)}
            fill={STRATEGY_COLORS[strategy]}
            opacity={hoveredStrategy === strategy ? 0.15 : 0.05}
            style={{ transition: 'opacity 0.2s' }}
          />
        ))}

        {/* Lines */}
        {strategies.map(strategy => (
          <path
            key={`line-${strategy}`}
            d={generatePath(strategy)}
            fill="none"
            stroke={STRATEGY_COLORS[strategy]}
            strokeWidth={hoveredStrategy === strategy ? 3 : 2}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={hoveredStrategy && hoveredStrategy !== strategy ? 0.3 : 1}
            style={{ transition: 'opacity 0.2s, stroke-width 0.2s' }}
          />
        ))}

        {/* End points with values */}
        {strategies.map(strategy => {
          const lastIndex = chartData.length - 1
          const lastValue = chartData[lastIndex]?.[strategy]
          if (lastValue == null) return null
          
          return (
            <g key={`end-${strategy}`}>
              {/* Dot */}
              <circle
                cx={xScale(lastIndex)}
                cy={yScale(lastValue)}
                r={hoveredStrategy === strategy ? 5 : 4}
                fill={STRATEGY_COLORS[strategy]}
                stroke="var(--bg-card)"
                strokeWidth="2"
                style={{ transition: 'r 0.2s' }}
              />
              
              {/* Value label */}
              <g transform={`translate(${xScale(lastIndex) + 8}, ${yScale(lastValue)})`}>
                <rect
                  x="0"
                  y="-10"
                  width="55"
                  height="20"
                  rx="4"
                  fill={STRATEGY_COLORS[strategy]}
                  opacity="0.9"
                />
                <text
                  x="27"
                  y="2"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="10"
                  fontWeight="600"
                  fill="white"
                >
                  {formatValue(lastValue)}
                </text>
              </g>
            </g>
          )
        })}

        {/* Hover line and tooltip */}
        {hoveredPoint && showTooltip && (
          <g>
            {/* Vertical line */}
            <line
              x1={xScale(hoveredPoint.index)}
              y1={padding.top}
              x2={xScale(hoveredPoint.index)}
              y2={padding.top + chartHeight}
              stroke="var(--text-muted)"
              strokeWidth="1"
              strokeDasharray="4,4"
              opacity="0.5"
            />
            
            {/* Points on line */}
            {strategies.map(strategy => {
              const value = hoveredPoint.data?.[strategy]
              if (value == null) return null
              
              return (
                <circle
                  key={`hover-${strategy}`}
                  cx={xScale(hoveredPoint.index)}
                  cy={yScale(value)}
                  r="4"
                  fill={STRATEGY_COLORS[strategy]}
                  stroke="var(--bg-card)"
                  strokeWidth="2"
                />
              )
            })}
          </g>
        )}
      </svg>

      {/* Tooltip */}
      {hoveredPoint && showTooltip && (
        <div 
          className="mt-2 p-3 rounded-lg"
          style={{ backgroundColor: 'var(--bg-tertiary)' }}
        >
          <div 
            className="text-xs font-semibold mb-2"
            style={{ color: 'var(--text-secondary)' }}
          >
            Round {hoveredPoint.data?.round}
          </div>
          <div className="grid grid-cols-2 gap-2">
            {strategies.map(strategy => {
              const value = hoveredPoint.data?.[strategy]
              const firstValue = chartData[0]?.[strategy]
              const change = firstValue ? ((value - firstValue) / firstValue * 100) : 0
              
              return (
                <div key={strategy} className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <div 
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: STRATEGY_COLORS[strategy] }}
                    />
                    <span 
                      className="text-[10px]"
                      style={{ color: 'var(--text-tertiary)' }}
                    >
                      {strategy.split(' ')[0]}
                    </span>
                  </div>
                  <div className="text-right">
                    <span 
                      className="text-[10px] font-semibold"
                      style={{ color: 'var(--text-primary)' }}
                    >
                      {formatValue(value)}
                    </span>
                    <span 
                      className="text-[9px] ml-1"
                      style={{ color: change >= 0 ? 'var(--success)' : 'var(--danger)' }}
                    >
                      {change >= 0 ? '+' : ''}{change.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}