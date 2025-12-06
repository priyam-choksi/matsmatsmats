import React, { useState } from 'react'
import { useTheme } from '../../App'

// Icons
const Icons = {
  ChevronDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
  ArrowRight: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" /><path d="m12 5 7 7-7 7" />
    </svg>
  ),
  Check: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  Zap: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  )
}

// Phase colors
const phaseColors = {
  1: { color: '#6366f1', bg: 'rgba(99, 102, 241, 0.1)', label: 'Analysis' },
  2: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', label: 'Research' },
  3: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', label: 'Synthesis' },
  4: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', label: 'Evaluation' },
  5: { color: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.1)', label: 'Decision' }
}

// Agent configurations with detailed info
const agentDetails = {
  'Technical Analyst': {
    icon: '📊',
    responsibilities: [
      'Calculate RSI, MACD, Bollinger Bands',
      'Identify support/resistance levels',
      'Detect chart patterns',
      'Analyze volume trends'
    ],
    inputs: ['Historical price data', 'Volume data', 'Market context'],
    outputs: ['Technical signal', 'Key levels', 'Pattern detection'],
    sampleOutput: { signal: 'BULLISH', rsi: 58.4, trend: 'Uptrend', support: '$142.50' }
  },
  'Fundamental Analyst': {
    icon: '📋',
    responsibilities: [
      'Analyze P/E, P/B, P/S ratios',
      'Evaluate revenue and earnings growth',
      'Assess profit margins and ROE',
      'Compare to sector peers'
    ],
    inputs: ['Financial statements', 'Earnings reports', 'Sector data'],
    outputs: ['Valuation assessment', 'Growth metrics', 'Health score'],
    sampleOutput: { pe_ratio: 28.5, revenue_growth: '22.4%', valuation: 'Fair Value' }
  },
  'News Analyst': {
    icon: '📰',
    responsibilities: [
      'Aggregate news from multiple sources',
      'Perform sentiment analysis',
      'Identify key catalysts',
      'Track analyst ratings'
    ],
    inputs: ['News feeds', 'Press releases', 'Analyst reports'],
    outputs: ['Sentiment score', 'Key headlines', 'Catalyst timeline'],
    sampleOutput: { sentiment: 'Positive', score: 0.72, articles: 24 }
  },
  'Macro Analyst': {
    icon: '🌍',
    responsibilities: [
      'Monitor Fed policy and rates',
      'Track sector rotation patterns',
      'Detect market regime',
      'Analyze macro correlations'
    ],
    inputs: ['Economic indicators', 'Fed statements', 'Sector flows'],
    outputs: ['Market regime', 'Sector outlook', 'Risk factors'],
    sampleOutput: { regime: 'Bull Market', sector: 'Tech Overweight', fed: 'Neutral' }
  },
  'Bull Researcher': {
    icon: '🐂',
    responsibilities: [
      'Build comprehensive bull case',
      'Identify growth catalysts',
      'Set upside price targets',
      'Counter bear arguments'
    ],
    inputs: ['Phase 1 outputs', 'Company guidance', 'Industry trends'],
    outputs: ['Bull thesis', 'Upside target', 'Confidence level'],
    sampleOutput: { thesis: 'Strong AI tailwinds', upside: '+28%', confidence: 'HIGH' }
  },
  'Bear Researcher': {
    icon: '🐻',
    responsibilities: [
      'Build comprehensive bear case',
      'Identify key risks',
      'Set downside targets',
      'Counter bull arguments'
    ],
    inputs: ['Phase 1 outputs', 'Risk factors', 'Competitive analysis'],
    outputs: ['Bear thesis', 'Downside risk', 'Confidence level'],
    sampleOutput: { thesis: 'Valuation stretched', downside: '-18%', confidence: 'MEDIUM' }
  },
  'Research Manager': {
    icon: '⚖️',
    responsibilities: [
      'Facilitate structured debate',
      'Evaluate argument strength',
      'Assign probability weights',
      'Synthesize recommendation'
    ],
    inputs: ['Bull thesis', 'Bear thesis', 'Discussion points'],
    outputs: ['Debate winner', 'Probability distribution', 'Synthesis'],
    sampleOutput: { winner: 'BULL', bull_prob: '62%', bear_prob: '28%' }
  },
  'Aggressive Evaluator': {
    icon: '🔥',
    responsibilities: [
      'Evaluate from risk-seeking perspective',
      'Recommend higher positions',
      'Focus on upside potential',
      'Default: STRONG BUY'
    ],
    inputs: ['Research synthesis', 'Risk parameters'],
    outputs: ['Stance', 'Position %', 'Reasoning'],
    personality: 'Risk-Seeking',
    sampleOutput: { stance: 'STRONG BUY', position: '75%' }
  },
  'Neutral Evaluator': {
    icon: '⚖️',
    responsibilities: [
      'Provide balanced assessment',
      'Recommend moderate positions',
      'Weigh upside and downside',
      'Default: HOLD'
    ],
    inputs: ['Research synthesis', 'Risk parameters'],
    outputs: ['Stance', 'Position %', 'Reasoning'],
    personality: 'Balanced',
    sampleOutput: { stance: 'BUY', position: '40%' }
  },
  'Conservative Evaluator': {
    icon: '🛡️',
    responsibilities: [
      'Evaluate from risk-averse perspective',
      'Recommend smaller positions',
      'Focus on downside protection',
      'Default: AVOID'
    ],
    inputs: ['Research synthesis', 'Risk parameters'],
    outputs: ['Stance', 'Position %', 'Reasoning'],
    personality: 'Risk-Averse',
    sampleOutput: { stance: 'HOLD', position: '15%' }
  },
  'Risk Manager': {
    icon: '👨‍💼',
    responsibilities: [
      'Aggregate evaluator recommendations',
      'Make final BUY/HOLD/REJECT',
      'Determine position size',
      'Set stop-loss and targets'
    ],
    inputs: ['All evaluator outputs', 'Portfolio constraints'],
    outputs: ['Final verdict', 'Position $', 'Risk controls'],
    sampleOutput: { verdict: 'BUY', position: '$12,450', confidence: 'MEDIUM' }
  }
}

export default function AgentCard({ agent, compact = false }) {
  const { isDark } = useTheme()
  const [isExpanded, setIsExpanded] = useState(false)
  
  // Get agent details - support both object format and string name
  const agentName = typeof agent === 'string' ? agent : (agent?.name || 'Unknown Agent')
  const agentPhase = typeof agent === 'object' ? agent?.phase : 1
  const agentDesc = typeof agent === 'object' ? agent?.description : ''
  
  const details = agentDetails[agentName] || {
    icon: '🤖',
    responsibilities: ['Process data', 'Generate output'],
    inputs: ['Input data'],
    outputs: ['Output data'],
    sampleOutput: {}
  }
  
  const phase = phaseColors[agentPhase] || phaseColors[1]

  // Compact version
  if (compact) {
    return (
      <div
        className="p-3 rounded-lg border transition-all duration-200 cursor-pointer"
        style={{
          backgroundColor: 'var(--bg-card)',
          borderColor: 'var(--border-primary)'
        }}
        onClick={() => setIsExpanded(!isExpanded)}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = phase.color
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = 'var(--border-primary)'
        }}
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">{details.icon}</span>
          <div className="flex-1 min-w-0">
            <div 
              className="text-sm font-semibold truncate"
              style={{ color: 'var(--text-primary)' }}
            >
              {agentName}
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              <span 
                className="text-[9px] font-semibold px-1.5 py-0.5 rounded"
                style={{ backgroundColor: phase.bg, color: phase.color }}
              >
                Phase {agentPhase}
              </span>
              {details.personality && (
                <span 
                  className="text-[9px]"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {details.personality}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Full version
  return (
    <div
      className="rounded-xl border overflow-hidden transition-all duration-200"
      style={{
        backgroundColor: 'var(--bg-card)',
        borderColor: isExpanded ? phase.color : 'var(--border-primary)'
      }}
    >
      {/* Header - Always visible */}
      <div 
        className="p-4 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            {/* Icon */}
            <div 
              className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl flex-shrink-0"
              style={{ backgroundColor: phase.bg }}
            >
              {details.icon}
            </div>
            
            {/* Info */}
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h3 
                  className="text-base font-bold"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {agentName}
                </h3>
                <span 
                  className="text-[10px] font-semibold px-2 py-0.5 rounded"
                  style={{ backgroundColor: phase.bg, color: phase.color }}
                >
                  Phase {agentPhase} • {phase.label}
                </span>
              </div>
              
              {/* Personality badge */}
              {details.personality && (
                <span 
                  className="inline-block text-[10px] font-medium px-2 py-0.5 rounded mt-1.5"
                  style={{ 
                    backgroundColor: 'var(--warning-bg)', 
                    color: 'var(--warning)' 
                  }}
                >
                  {details.personality}
                </span>
              )}
              
              {/* Description */}
              <p 
                className="text-xs mt-2 leading-relaxed"
                style={{ color: 'var(--text-secondary)' }}
              >
                {agentDesc || `Specialized agent for ${phase.label.toLowerCase()} tasks in the trading pipeline.`}
              </p>
            </div>
          </div>

          {/* Expand icon */}
          <div 
            className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0 transition-transform"
            style={{ 
              backgroundColor: 'var(--bg-tertiary)',
              color: 'var(--text-muted)',
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)'
            }}
          >
            <Icons.ChevronDown size={14} />
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div 
          className="px-4 pb-4 border-t"
          style={{ borderColor: 'var(--border-primary)' }}
        >
          {/* Responsibilities */}
          <div className="mt-4">
            <div 
              className="text-[10px] font-semibold uppercase tracking-wider mb-2"
              style={{ color: 'var(--text-muted)' }}
            >
              Responsibilities
            </div>
            <ul className="space-y-1.5">
              {details.responsibilities.map((resp, i) => (
                <li 
                  key={i}
                  className="flex items-start gap-2 text-xs"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  <Icons.Check 
                    size={12} 
                    className="mt-0.5 flex-shrink-0" 
                    style={{ color: phase.color }} 
                  />
                  <span>{resp}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Inputs & Outputs */}
          <div className="grid grid-cols-2 gap-3 mt-4">
            {/* Inputs */}
            <div>
              <div 
                className="text-[10px] font-semibold uppercase tracking-wider mb-2"
                style={{ color: 'var(--text-muted)' }}
              >
                Inputs
              </div>
              <div className="flex flex-wrap gap-1">
                {details.inputs.map((input, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-2 py-1 rounded"
                    style={{ 
                      backgroundColor: 'var(--info-bg)', 
                      color: 'var(--info)' 
                    }}
                  >
                    {input}
                  </span>
                ))}
              </div>
            </div>

            {/* Outputs */}
            <div>
              <div 
                className="text-[10px] font-semibold uppercase tracking-wider mb-2"
                style={{ color: 'var(--text-muted)' }}
              >
                Outputs
              </div>
              <div className="flex flex-wrap gap-1">
                {details.outputs.map((output, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-2 py-1 rounded"
                    style={{ 
                      backgroundColor: 'var(--success-bg)', 
                      color: 'var(--success)' 
                    }}
                  >
                    {output}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Sample Output */}
          <div className="mt-4">
            <div 
              className="text-[10px] font-semibold uppercase tracking-wider mb-2 flex items-center gap-1"
              style={{ color: 'var(--text-muted)' }}
            >
              <Icons.Zap size={10} />
              Sample Output
            </div>
            <pre
              className="text-[10px] p-3 rounded-lg overflow-x-auto"
              style={{ 
                backgroundColor: isDark ? '#1a1a1a' : '#f8fafc',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border-primary)'
              }}
            >
              {JSON.stringify(details.sampleOutput, null, 2)}
            </pre>
          </div>

          {/* Flow indicator */}
          {agentPhase < 5 && (
            <div 
              className="mt-4 pt-3 border-t flex items-center justify-center gap-2"
              style={{ borderColor: 'var(--border-primary)' }}
            >
              <span 
                className="text-[10px]"
                style={{ color: 'var(--text-muted)' }}
              >
                Output feeds into
              </span>
              <span 
                className="text-[10px] font-semibold flex items-center gap-1"
                style={{ color: phaseColors[agentPhase + 1]?.color }}
              >
                Phase {agentPhase + 1}
                <Icons.ArrowRight size={10} />
              </span>
            </div>
          )}

          {/* Final decision indicator */}
          {agentPhase === 5 && (
            <div 
              className="mt-4 pt-3 border-t flex items-center justify-center"
              style={{ borderColor: 'var(--border-primary)' }}
            >
              <span 
                className="text-[10px] font-semibold px-3 py-1 rounded-full"
                style={{ 
                  backgroundColor: phase.bg, 
                  color: phase.color 
                }}
              >
                ✓ Final Output → Trading Decision
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}