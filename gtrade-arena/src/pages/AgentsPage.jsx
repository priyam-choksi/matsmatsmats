import React, { useState } from 'react'
import { useTheme } from '../App'

// ========== AGENTS DATA ==========
const agents = [
  // Phase 1 - Market Analysis
  { 
    name: 'Technical Analyst', 
    icon: '📈', 
    phase: 1,
    phaseName: 'Market Analysis',
    category: 'Analyst',
    color: '#6366f1',
    description: 'Analyzes price action, chart patterns, and technical indicators to identify trading signals.',
    responsibilities: [
      'Calculate RSI, MACD, Bollinger Bands',
      'Identify support and resistance levels',
      'Detect chart patterns (flags, triangles, H&S)',
      'Analyze volume trends and momentum'
    ],
    inputs: ['Historical price data', 'Volume data', 'Market context'],
    outputs: ['Technical signal (BULLISH/BEARISH/NEUTRAL)', 'Key levels', 'Pattern detection'],
    sampleOutput: {
      signal: 'BULLISH',
      rsi: 58.4,
      macd: 'Bullish Crossover',
      trend: 'Uptrend',
      support: '$142.50',
      resistance: '$165.00'
    }
  },
  { 
    name: 'News Analyst', 
    icon: '📰', 
    phase: 1,
    phaseName: 'Market Analysis',
    category: 'Analyst',
    color: '#6366f1',
    description: 'Processes news from multiple sources to gauge market sentiment and identify catalysts.',
    responsibilities: [
      'Aggregate news from Yahoo, Finnhub, Reuters',
      'Perform sentiment analysis on headlines',
      'Identify key catalysts and events',
      'Track analyst ratings and price targets'
    ],
    inputs: ['News feeds', 'Press releases', 'Analyst reports'],
    outputs: ['Sentiment score', 'Key headlines', 'Catalyst timeline'],
    sampleOutput: {
      sentiment: 'Positive',
      score: 0.72,
      articles_analyzed: 24,
      key_catalyst: 'Earnings beat expectations by 15%'
    }
  },
  { 
    name: 'Fundamental Analyst', 
    icon: '📊', 
    phase: 1,
    phaseName: 'Market Analysis',
    category: 'Analyst',
    color: '#6366f1',
    description: 'Evaluates financial statements, valuation metrics, and company fundamentals.',
    responsibilities: [
      'Analyze P/E, P/B, P/S ratios',
      'Evaluate revenue and earnings growth',
      'Assess profit margins and ROE',
      'Compare to sector peers'
    ],
    inputs: ['Financial statements', 'Earnings reports', 'Sector data'],
    outputs: ['Valuation assessment', 'Growth metrics', 'Financial health score'],
    sampleOutput: {
      pe_ratio: 28.5,
      revenue_growth: '22.4%',
      profit_margin: '18.7%',
      valuation: 'Fair Value',
      vs_sector: 'Premium justified by growth'
    }
  },
  { 
    name: 'Macro Analyst', 
    icon: '🌍', 
    phase: 1,
    phaseName: 'Market Analysis',
    category: 'Analyst',
    color: '#6366f1',
    description: 'Assesses broader market conditions, Fed policy, and sector rotation trends.',
    responsibilities: [
      'Monitor Fed policy and interest rates',
      'Track sector rotation patterns',
      'Detect market regime (bull/bear/sideways)',
      'Analyze correlation with macro factors'
    ],
    inputs: ['Economic indicators', 'Fed statements', 'Sector ETF flows'],
    outputs: ['Market regime', 'Sector outlook', 'Macro risk factors'],
    sampleOutput: {
      regime: 'Bull Market',
      fed_stance: 'Neutral - Holding rates',
      sector_outlook: 'Technology Overweight',
      risk_factors: ['Inflation data', 'Earnings season']
    }
  },
  // Phase 2 - Research
  { 
    name: 'Bull Researcher', 
    icon: '🐂', 
    phase: 2,
    phaseName: 'Bull/Bear Research',
    category: 'Researcher',
    color: '#10b981',
    description: 'Constructs the bullish investment thesis with catalysts, upside targets, and supporting evidence.',
    responsibilities: [
      'Build comprehensive bull case',
      'Identify growth catalysts',
      'Set upside price targets',
      'Counter bear arguments'
    ],
    inputs: ['Phase 1 analyst outputs', 'Company guidance', 'Industry trends'],
    outputs: ['Bull thesis document', 'Upside target', 'Catalyst timeline', 'Confidence level'],
    sampleOutput: {
      thesis: 'Strong AI tailwinds driving accelerated growth',
      upside_target: '+28%',
      catalysts: ['AI product launch Q1', 'Data center expansion', 'Market share gains'],
      confidence: 'HIGH'
    }
  },
  { 
    name: 'Bear Researcher', 
    icon: '🐻', 
    phase: 2,
    phaseName: 'Bull/Bear Research',
    category: 'Researcher',
    color: '#ef4444',
    description: 'Constructs the bearish case with risks, headwinds, and downside scenarios.',
    responsibilities: [
      'Build comprehensive bear case',
      'Identify key risks and headwinds',
      'Set downside price targets',
      'Counter bull arguments'
    ],
    inputs: ['Phase 1 analyst outputs', 'Risk factors', 'Competitive analysis'],
    outputs: ['Bear thesis document', 'Downside target', 'Risk factors', 'Confidence level'],
    sampleOutput: {
      thesis: 'Valuation stretched, competition intensifying',
      downside_risk: '-18%',
      risks: ['Multiple compression', 'Margin pressure', 'Regulatory risk'],
      confidence: 'MEDIUM'
    }
  },
  // Phase 3 - Synthesis
  { 
    name: 'Research Manager', 
    icon: '⚖️', 
    phase: 3,
    phaseName: 'Debate & Synthesis',
    category: 'Manager',
    color: '#f59e0b',
    description: 'Moderates the bull/bear debate, weighs arguments, and synthesizes into actionable conclusions.',
    responsibilities: [
      'Facilitate structured debate',
      'Evaluate argument strength',
      'Assign probability weights',
      'Synthesize final recommendation'
    ],
    inputs: ['Bull thesis', 'Bear thesis', 'Discussion points'],
    outputs: ['Debate winner', 'Probability distribution', 'Synthesized recommendation'],
    sampleOutput: {
      debate_winner: 'BULL',
      bull_probability: '62%',
      bear_probability: '28%',
      base_probability: '10%',
      synthesis: 'Bull case prevails on strong growth catalysts'
    }
  },
  // Phase 4 - Risk Evaluation
  { 
    name: 'Aggressive Evaluator', 
    icon: '🔥', 
    phase: 4,
    phaseName: 'Risk Evaluation',
    category: 'Evaluator',
    color: '#f97316',
    description: 'Risk-seeking personality that favors larger positions and bullish interpretations.',
    responsibilities: [
      'Evaluate from risk-seeking perspective',
      'Recommend higher position sizes',
      'Focus on upside potential',
      'Default stance: STRONG BUY'
    ],
    inputs: ['Research synthesis', 'Risk parameters'],
    outputs: ['Stance', 'Position size %', 'Reasoning'],
    personality: 'Risk-Seeking',
    defaultStance: 'STRONG BUY',
    sampleOutput: {
      stance: 'STRONG BUY',
      position_pct: '75%',
      reasoning: 'High conviction on growth catalysts, willing to take concentrated position'
    }
  },
  { 
    name: 'Neutral Evaluator', 
    icon: '⚖️', 
    phase: 4,
    phaseName: 'Risk Evaluation',
    category: 'Evaluator',
    color: '#6b7280',
    description: 'Balanced personality that weighs bull and bear cases equally.',
    responsibilities: [
      'Provide balanced assessment',
      'Recommend moderate positions',
      'Weigh both upside and downside',
      'Default stance: HOLD'
    ],
    inputs: ['Research synthesis', 'Risk parameters'],
    outputs: ['Stance', 'Position size %', 'Reasoning'],
    personality: 'Balanced',
    defaultStance: 'HOLD',
    sampleOutput: {
      stance: 'BUY',
      position_pct: '40%',
      reasoning: 'Balanced view - upside potential outweighs risks but position sized for volatility'
    }
  },
  { 
    name: 'Conservative Evaluator', 
    icon: '🛡️', 
    phase: 4,
    phaseName: 'Risk Evaluation',
    category: 'Evaluator',
    color: '#3b82f6',
    description: 'Risk-averse personality that prioritizes capital preservation.',
    responsibilities: [
      'Evaluate from risk-averse perspective',
      'Recommend smaller positions',
      'Focus on downside protection',
      'Default stance: AVOID'
    ],
    inputs: ['Research synthesis', 'Risk parameters'],
    outputs: ['Stance', 'Position size %', 'Reasoning'],
    personality: 'Risk-Averse',
    defaultStance: 'AVOID',
    sampleOutput: {
      stance: 'HOLD',
      position_pct: '15%',
      reasoning: 'Prefer to wait for pullback, current valuation leaves little margin of safety'
    }
  },
  // Phase 5 - Decision
  { 
    name: 'Risk Manager', 
    icon: '👔', 
    phase: 5,
    phaseName: 'Final Decision',
    category: 'Decision Maker',
    color: '#8b5cf6',
    description: 'Makes the final trading decision with position sizing based on evaluator consensus.',
    responsibilities: [
      'Aggregate evaluator recommendations',
      'Make final BUY/HOLD/REJECT decision',
      'Determine exact position size',
      'Set stop-loss and profit targets'
    ],
    inputs: ['All evaluator outputs', 'Portfolio constraints', 'Risk limits'],
    outputs: ['Final verdict', 'Position $ amount', 'Risk controls'],
    sampleOutput: {
      verdict: 'BUY',
      position_dollars: '$12,450',
      confidence: 'MEDIUM',
      stop_loss: '-8%',
      profit_targets: ['+12%', '+18%', '+25%']
    }
  },
]

// Phase definitions for navigation
const phases = [
  { num: 1, name: 'Market Analysis', icon: '📊', agents: 4 },
  { num: 2, name: 'Bull/Bear Research', icon: '⚔️', agents: 2 },
  { num: 3, name: 'Debate & Synthesis', icon: '⚖️', agents: 1 },
  { num: 4, name: 'Risk Evaluation', icon: '🎯', agents: 3 },
  { num: 5, name: 'Final Decision', icon: '✅', agents: 1 },
]

// Icons
const Icons = {
  ChevronDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6"/>
    </svg>
  ),
  Brain: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/>
    </svg>
  ),
  ArrowRight: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
    </svg>
  )
}

// Agent Card Component
const AgentCard = ({ agent, isExpanded, onToggle }) => {
  return (
    <div style={{
      background: 'var(--bg-primary)',
      borderRadius: '14px',
      border: '1px solid var(--border-primary)',
      overflow: 'hidden',
      transition: 'all 0.2s'
    }}>
      {/* Header - Always Visible */}
      <div 
        onClick={onToggle}
        style={{ 
          padding: '16px', 
          cursor: 'pointer',
          borderBottom: isExpanded ? '1px solid var(--border-primary)' : 'none'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', gap: '14px' }}>
            <div style={{
              width: '50px',
              height: '50px',
              background: `${agent.color}15`,
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '26px'
            }}>
              {agent.icon}
            </div>
            <div>
              <h3 style={{ 
                margin: 0, 
                fontSize: '15px', 
                fontWeight: '700',
                color: 'var(--text-primary)'
              }}>
                {agent.name}
              </h3>
              <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
                <span style={{
                  padding: '3px 8px',
                  background: `${agent.color}20`,
                  color: agent.color,
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontWeight: '600'
                }}>
                  Phase {agent.phase}
                </span>
                <span style={{
                  padding: '3px 8px',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-muted)',
                  borderRadius: '6px',
                  fontSize: '10px'
                }}>
                  {agent.category}
                </span>
                {agent.personality && (
                  <span style={{
                    padding: '3px 8px',
                    background: 'var(--warning-bg)',
                    color: 'var(--warning)',
                    borderRadius: '6px',
                    fontSize: '10px'
                  }}>
                    {agent.personality}
                  </span>
                )}
              </div>
              <p style={{
                margin: '8px 0 0',
                fontSize: '12px',
                color: 'var(--text-secondary)',
                lineHeight: '1.4'
              }}>
                {agent.description}
              </p>
            </div>
          </div>
          <span style={{
            fontSize: '18px',
            color: 'var(--text-muted)',
            transform: isExpanded ? 'rotate(180deg)' : 'none',
            transition: 'transform 0.2s'
          }}>
            <Icons.ChevronDown size={18} />
          </span>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div style={{ 
          padding: '16px', 
          background: 'var(--bg-secondary)'
        }}>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(2, 1fr)', 
            gap: '16px', 
            marginBottom: '16px' 
          }}>
            {/* Responsibilities */}
            <div>
              <div style={{
                fontSize: '10px',
                fontWeight: '700',
                color: 'var(--text-muted)',
                marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                Responsibilities
              </div>
              <ul style={{
                margin: 0,
                paddingLeft: '16px',
                fontSize: '12px',
                color: 'var(--text-secondary)'
              }}>
                {agent.responsibilities.map((r, i) => (
                  <li key={i} style={{ marginBottom: '4px' }}>{r}</li>
                ))}
              </ul>
            </div>
            
            {/* Inputs & Outputs */}
            <div>
              <div style={{
                fontSize: '10px',
                fontWeight: '700',
                color: 'var(--text-muted)',
                marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                Inputs
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '12px' }}>
                {agent.inputs.map((inp, i) => (
                  <span key={i} style={{
                    padding: '4px 8px',
                    background: 'var(--info-bg)',
                    color: 'var(--info)',
                    borderRadius: '6px',
                    fontSize: '10px'
                  }}>
                    {inp}
                  </span>
                ))}
              </div>
              
              <div style={{
                fontSize: '10px',
                fontWeight: '700',
                color: 'var(--text-muted)',
                marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                Outputs
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {agent.outputs.map((out, i) => (
                  <span key={i} style={{
                    padding: '4px 8px',
                    background: 'var(--success-bg)',
                    color: 'var(--success)',
                    borderRadius: '6px',
                    fontSize: '10px'
                  }}>
                    {out}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Sample Output */}
          <div>
            <div style={{
              fontSize: '10px',
              fontWeight: '700',
              color: 'var(--text-muted)',
              marginBottom: '8px',
              textTransform: 'uppercase'
            }}>
              Sample Output
            </div>
            <pre style={{
              margin: 0,
              padding: '12px',
              background: 'var(--bg-primary)',
              borderRadius: '8px',
              fontSize: '11px',
              color: 'var(--text-secondary)',
              overflow: 'auto',
              border: '1px solid var(--border-primary)'
            }}>
              {JSON.stringify(agent.sampleOutput, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

// Main Component
export default function AgentsPage() {
  const { isDark } = useTheme()
  const [expandedAgent, setExpandedAgent] = useState(null)
  const [selectedPhase, setSelectedPhase] = useState(null)

  const filteredAgents = selectedPhase 
    ? agents.filter(a => a.phase === selectedPhase) 
    : agents

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'var(--bg-secondary)',
      paddingTop: '56px'
    }}>
      {/* Header */}
      <header style={{
        background: 'var(--bg-primary)',
        borderBottom: '1px solid var(--border-primary)',
        padding: '12px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white'
          }}>
            <Icons.Brain size={18} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              LLM Agents
            </h1>
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
              11 Specialized Agents • 5 Phases
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
          <span style={{ 
            padding: '5px 10px', 
            background: 'var(--bg-tertiary)', 
            borderRadius: '6px',
            color: 'var(--text-secondary)'
          }}>
            GPT-4o-mini
          </span>
          <span style={{ 
            padding: '5px 10px', 
            background: 'var(--info-bg)', 
            borderRadius: '6px',
            color: 'var(--info)'
          }}>
            LangGraph
          </span>
        </div>
      </header>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
        {/* Phase Filter */}
        <div style={{ 
          display: 'flex', 
          gap: '8px', 
          marginBottom: '20px', 
          flexWrap: 'wrap' 
        }}>
          <button
            onClick={() => setSelectedPhase(null)}
            style={{
              padding: '8px 14px',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: '600',
              background: selectedPhase === null 
                ? 'var(--text-primary)' 
                : 'var(--bg-tertiary)',
              color: selectedPhase === null 
                ? 'var(--bg-primary)' 
                : 'var(--text-secondary)'
            }}
          >
            All Agents (11)
          </button>
          {phases.map(phase => (
            <button
              key={phase.num}
              onClick={() => setSelectedPhase(phase.num)}
              style={{
                padding: '8px 14px',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '12px',
                background: selectedPhase === phase.num 
                  ? 'var(--accent-primary)' 
                  : 'var(--bg-tertiary)',
                color: selectedPhase === phase.num 
                  ? 'white' 
                  : 'var(--text-secondary)'
              }}
            >
              {phase.icon} Phase {phase.num} ({phase.agents})
            </button>
          ))}
        </div>

        {/* Pipeline Flow Visualization */}
        <div style={{
          background: 'var(--bg-primary)',
          borderRadius: '14px',
          border: '1px solid var(--border-primary)',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <h3 style={{ 
            margin: '0 0 16px', 
            fontSize: '14px', 
            fontWeight: '600',
            color: 'var(--text-primary)'
          }}>
            📊 Pipeline Flow
          </h3>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between', 
            gap: '8px' 
          }}>
            {phases.map((p, i) => (
              <React.Fragment key={p.num}>
                <div
                  onClick={() => setSelectedPhase(p.num)}
                  style={{
                    flex: 1,
                    padding: '14px',
                    borderRadius: '12px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    background: selectedPhase === p.num 
                      ? 'var(--accent-primary)' 
                      : 'var(--bg-tertiary)',
                    color: selectedPhase === p.num 
                      ? 'white' 
                      : 'var(--text-secondary)',
                    border: selectedPhase === p.num 
                      ? '2px solid var(--accent-primary)' 
                      : '1px solid var(--border-primary)',
                    transition: 'all 0.2s'
                  }}
                >
                  <div style={{ fontSize: '24px', marginBottom: '6px' }}>{p.icon}</div>
                  <div style={{ fontWeight: '700', fontSize: '12px' }}>Phase {p.num}</div>
                  <div style={{ fontSize: '11px', opacity: 0.8 }}>{p.name}</div>
                  <div style={{ fontSize: '10px', opacity: 0.7, marginTop: '4px' }}>
                    {p.agents} agent{p.agents > 1 ? 's' : ''}
                  </div>
                </div>
                {i < 4 && (
                  <div style={{ color: 'var(--text-muted)', fontSize: '20px' }}>
                    <Icons.ArrowRight size={20} />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Agent Cards Grid */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(2, 1fr)', 
          gap: '16px' 
        }}>
          {filteredAgents.map(agent => (
            <AgentCard
              key={agent.name}
              agent={agent}
              isExpanded={expandedAgent === agent.name}
              onToggle={() => setExpandedAgent(
                expandedAgent === agent.name ? null : agent.name
              )}
            />
          ))}
        </div>

        {/* Research Framework Card */}
        <div style={{
          background: 'var(--bg-primary)',
          borderRadius: '14px',
          border: '1px solid var(--border-primary)',
          padding: '20px',
          marginTop: '20px'
        }}>
          <h3 style={{ 
            margin: '0 0 16px', 
            fontSize: '14px', 
            fontWeight: '600',
            color: 'var(--text-primary)'
          }}>
            🔬 Research Framework
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            <div style={{
              padding: '16px',
              background: 'var(--bg-tertiary)',
              borderRadius: '10px'
            }}>
              <div style={{ 
                fontSize: '11px', 
                fontWeight: '600', 
                color: 'var(--text-muted)',
                marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                Research Question
              </div>
              <p style={{ 
                margin: 0, 
                fontSize: '13px', 
                color: 'var(--text-secondary)',
                lineHeight: '1.5'
              }}>
                Do different trading strategies perform optimally under different market regimes?
              </p>
            </div>
            <div style={{
              padding: '16px',
              background: 'var(--bg-tertiary)',
              borderRadius: '10px'
            }}>
              <div style={{ 
                fontSize: '11px', 
                fontWeight: '600', 
                color: 'var(--text-muted)',
                marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                Hypothesis
              </div>
              <p style={{ 
                margin: 0, 
                fontSize: '13px', 
                color: 'var(--text-secondary)',
                lineHeight: '1.5'
              }}>
                LLM-powered signal generation combined with game theory capital allocation can outperform traditional strategies.
              </p>
            </div>
            <div style={{
              padding: '16px',
              background: 'var(--bg-tertiary)',
              borderRadius: '10px'
            }}>
              <div style={{ 
                fontSize: '11px', 
                fontWeight: '600', 
                color: 'var(--text-muted)',
                marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                Validation
              </div>
              <p style={{ 
                margin: 0, 
                fontSize: '13px', 
                color: 'var(--text-secondary)',
                lineHeight: '1.5'
              }}>
                1000 Monte Carlo simulations across 11 tickers with 990 samples. 73.2% signal accuracy achieved.
              </p>
            </div>
          </div>
        </div>

        {/* Data Collection Stats */}
        <div style={{
          background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
          borderRadius: '14px',
          padding: '20px',
          marginTop: '20px',
          color: 'white'
        }}>
          <h3 style={{ margin: '0 0 16px', fontSize: '14px', fontWeight: '600' }}>
            📦 Data Collection
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            {[
              { label: 'Total Samples', value: '990' },
              { label: 'Tickers', value: '11' },
              { label: 'Sample Interval', value: '3 days' },
              { label: 'Time Period', value: '~9 months' }
            ].map((stat, i) => (
              <div key={i} style={{
                padding: '12px',
                background: 'rgba(255,255,255,0.15)',
                borderRadius: '8px',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '10px', opacity: 0.8, marginBottom: '4px' }}>
                  {stat.label}
                </div>
                <div style={{ fontSize: '18px', fontWeight: '700' }}>
                  {stat.value}
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '16px' }}>
            <div style={{ fontSize: '11px', opacity: 0.8, marginBottom: '8px' }}>
              TICKERS ANALYZED
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ', 'LLY'].map((t, i) => (
                <span key={i} style={{
                  padding: '6px 12px',
                  background: 'rgba(255,255,255,0.2)',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: '600'
                }}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}