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

// ========== GAME THEORY STRATEGIES DATA ==========
const gameTheoryStrategies = [
  {
    name: 'Signal Follower',
    icon: '📡',
    color: '#6366f1',
    shortName: 'Signal',
    description: 'Uses LLM-generated trading signals to make decisions. Represents AI-powered analysis.',
    behavior: 'Follows the pipeline\'s BUY/HOLD/SELL recommendations with position sizing based on confidence levels. Adapts position size (0-100%) based on signal strength.',
    mechanism: [
      'Receives signal from 11-agent pipeline',
      'Maps BUY/HOLD/SELL to position allocation',
      'Scales position by confidence score',
      'Adjusts for market regime context'
    ],
    strengths: [
      'Adapts to market conditions dynamically',
      'Uses comprehensive multi-source analysis',
      'Data-driven with explainable reasoning',
      'Can capture complex market patterns'
    ],
    weaknesses: [
      'Dependent on LLM signal quality',
      'May lag in fast-moving markets',
      'Complex reasoning can miss simple patterns',
      'Higher computational cost'
    ],
    gameTheoryRole: 'The "informed player" - has access to sophisticated analysis that other strategies lack.',
    performanceNote: 'Achieved 73.2% signal accuracy with +14.93% returns in tournament.'
  },
  {
    name: 'Cooperator',
    icon: '🤝',
    color: '#10b981',
    shortName: 'Coop',
    description: 'Always cooperates with the group consensus. Follows the majority strategy allocation.',
    behavior: 'Mirrors the average position of all other strategies. Builds trust through consistent, predictable behavior. Never defects or takes contrarian positions.',
    mechanism: [
      'Calculates average position of other strategies',
      'Matches that position exactly',
      'Maintains consistent cooperative behavior',
      'Never deviates regardless of market conditions'
    ],
    strengths: [
      'Low conflict with other strategies',
      'Benefits from collective wisdom',
      'Stable, predictable returns',
      'Reduces overall portfolio volatility'
    ],
    weaknesses: [
      'Vulnerable to coordinated defection',
      'No alpha generation potential',
      'Follows rather than leads',
      'Cannot capitalize on unique insights'
    ],
    gameTheoryRole: 'The "nice player" - from Axelrod\'s tournaments, cooperation can be evolutionarily stable.',
    performanceNote: 'Provides baseline for measuring value of active strategies.'
  },
  {
    name: 'Defector',
    icon: '🎯',
    color: '#f59e0b',
    shortName: 'Defect',
    description: 'Contrarian strategy that bets against the crowd. Seeks alpha through disagreement.',
    behavior: 'Takes opposite positions to the consensus. When others are bullish, it\'s bearish and vice versa. Profits when the crowd is wrong.',
    mechanism: [
      'Monitors consensus position direction',
      'Calculates inverse allocation',
      'Takes contrarian position consistently',
      'Maximizes when crowd is wrong'
    ],
    strengths: [
      'High alpha potential in reversals',
      'Profits from crowd mistakes',
      'Independent thinking approach',
      'Can capture mean reversion'
    ],
    weaknesses: [
      'High risk in trending markets',
      'Often wrong during momentum',
      'Misses extended trends entirely',
      'Can suffer large drawdowns'
    ],
    gameTheoryRole: 'The "defector" - exploits cooperative strategies but suffers when facing other defectors.',
    performanceNote: 'Won most individual rounds but had inconsistent overall returns.'
  },
  {
    name: 'Tit-for-Tat',
    icon: '🔄',
    color: '#06b6d4',
    shortName: 'TFT',
    description: 'Classic game theory strategy. Cooperates first, then mirrors the previous round\'s winning strategy.',
    behavior: 'Starts with a cooperative position. After each round, copies whatever strategy won the previous round. Adapts based on what\'s currently working.',
    mechanism: [
      'Round 1: Takes cooperative position',
      'Observes which strategy won previous round',
      'Copies winning strategy\'s approach',
      'Resets after extended losing streaks'
    ],
    strengths: [
      'Highly adaptive to conditions',
      'Punishes consistent defection',
      'Rewards cooperative behavior',
      'Simple yet effective logic'
    ],
    weaknesses: [
      'One round lag in adaptation',
      'Can get stuck in echo cycles',
      'Reactive rather than proactive',
      'Whipsaws in volatile markets'
    ],
    gameTheoryRole: 'The "reciprocator" - Axelrod showed TFT wins iterated prisoner\'s dilemma tournaments.',
    performanceNote: 'Most consistent performer across different market regimes.'
  }
]

// ========== LLM MODELS DATA ==========
const llmModels = [
  {
    name: 'GPT-4o',
    icon: '🧠',
    color: '#10b981',
    provider: 'OpenAI',
    tier: 'Premium',
    description: 'OpenAI\'s flagship multimodal model with superior reasoning capabilities.',
    tradingStyle: 'Balanced and thorough analysis with strong risk assessment.',
    strengths: [
      'Superior reasoning and nuance',
      'Excellent at weighing multiple factors',
      'Strong risk/reward analysis',
      'Best at complex market narratives'
    ],
    weaknesses: [
      'Higher latency',
      'Most expensive per call',
      'Can overthink simple setups'
    ],
    sampleReasoning: 'Given the mixed signals - bullish technicals but elevated valuation - I recommend a moderate 40% position with tight stops.',
    stats: { trades: 280, return: '+32.63%', perTrade: '$291' }
  },
  {
    name: 'Llama-3.3-70B',
    icon: '🦙',
    color: '#8b5cf6',
    provider: 'Meta',
    tier: 'Large',
    description: 'Meta\'s open-source large model with excellent performance on financial tasks.',
    tradingStyle: 'Aggressive positioning with high conviction calls.',
    strengths: [
      'Fast inference speed',
      'Strong quantitative reasoning',
      'Good pattern recognition',
      'Open source - auditable'
    ],
    weaknesses: [
      'Can be overconfident',
      'Less nuanced than GPT-4o',
      'Occasional hallucinations on dates'
    ],
    sampleReasoning: 'Clear bullish breakout pattern confirmed by volume. Taking 70% position targeting previous highs.',
    stats: { trades: 280, return: '+32.63%', perTrade: '$291' }
  },
  {
    name: 'Qwen3-32B',
    icon: '🌐',
    color: '#06b6d4',
    provider: 'Alibaba',
    tier: 'Large',
    description: 'Strong multilingual model with excellent quantitative and mathematical reasoning.',
    tradingStyle: 'Data-driven with emphasis on numerical analysis.',
    strengths: [
      'Excellent math/quant skills',
      'Strong on financial metrics',
      'Efficient token usage',
      'Good multilingual market coverage'
    ],
    weaknesses: [
      'Less creative in thesis building',
      'Can miss qualitative factors',
      'Smaller context window'
    ],
    sampleReasoning: 'P/E of 18.5 vs sector 24.2 indicates undervaluation. RSI at 42 suggests room to run. 55% allocation.',
    stats: { trades: 365, return: '+32.19%', perTrade: '$221' }
  },
  {
    name: 'Kimi',
    icon: '🌙',
    color: '#f59e0b',
    provider: 'Moonshot AI',
    tier: 'Specialized',
    description: 'Specialized model with exceptional long-context capabilities for document analysis.',
    tradingStyle: 'Research-heavy with deep fundamental analysis.',
    strengths: [
      'Excellent long-context handling',
      'Strong document analysis',
      'Good at synthesizing multiple sources',
      'Thorough due diligence'
    ],
    weaknesses: [
      'Slower on quick decisions',
      'Can over-analyze',
      'Less reactive to technicals'
    ],
    sampleReasoning: 'After reviewing the 10-K, earnings transcript, and analyst reports, the growth trajectory supports a 45% position.',
    stats: { trades: 1068, return: '+30.29%', perTrade: '$71' }
  },
  {
    name: 'GPT-o3-mini',
    icon: '⚡',
    color: '#ef4444',
    provider: 'OpenAI',
    tier: 'Efficient',
    description: 'Smaller, faster model optimized for speed and cost efficiency.',
    tradingStyle: 'Quick decisions based on key signals.',
    strengths: [
      'Very fast inference',
      'Low cost per call',
      'Good for high-frequency signals',
      'Efficient pattern matching'
    ],
    weaknesses: [
      'Less nuanced analysis',
      'Can miss complex setups',
      'Shorter reasoning chains'
    ],
    sampleReasoning: 'MACD bullish crossover + above 50 SMA = BUY signal. 50% position.',
    stats: { trades: 1408, return: '+29.88%', perTrade: '$53' }
  },
  {
    name: 'Maverick',
    icon: '🦅',
    color: '#ec4899',
    provider: 'Meta',
    tier: 'Experimental',
    description: 'Experimental model pushing boundaries on creative and unconventional reasoning.',
    tradingStyle: 'Contrarian with unconventional thesis generation.',
    strengths: [
      'Creative thesis building',
      'Finds non-obvious plays',
      'Good at sentiment extremes',
      'Novel pattern recognition'
    ],
    weaknesses: [
      'Higher variance in outputs',
      'Can be too contrarian',
      'Inconsistent confidence calibration'
    ],
    sampleReasoning: 'Everyone is bearish but insider buying is up 300%. Classic capitulation setup. Going 65% long.',
    stats: { trades: 892, return: '+27.45%', perTrade: '$77' }
  },
  {
    name: 'Allam',
    icon: '🏛️',
    color: '#6b7280',
    provider: 'IBM/SDAIA',
    tier: 'Specialized',
    description: 'Arabic-focused model with strong regional market knowledge and conservative approach.',
    tradingStyle: 'Conservative with emphasis on capital preservation.',
    strengths: [
      'Strong risk management',
      'Conservative position sizing',
      'Good at avoiding losses',
      'Regional market expertise'
    ],
    weaknesses: [
      'Misses aggressive opportunities',
      'Lower overall returns',
      'Can be too cautious'
    ],
    sampleReasoning: 'Fundamentals are solid but macro uncertainty warrants caution. Taking 25% position with wide stops.',
    stats: { trades: 756, return: '+18.92%', perTrade: '$63' }
  }
]



// Icons
const Icons = {
  ChevronDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
  Brain: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" /><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" /><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" /><path d="M17.599 6.5a3 3 0 0 0 .399-1.375" /><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" /><path d="M3.477 10.896a4 4 0 0 1 .585-.396" /><path d="M19.938 10.5a4 4 0 0 1 .585.396" /><path d="M6 18a4 4 0 0 1-1.967-.516" /><path d="M19.967 17.484A4 4 0 0 1 18 18" />
    </svg>
  ),
  ArrowRight: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" /><path d="m12 5 7 7-7 7" />
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

// ========== GAME THEORY TAB COMPONENT ==========
function GameTheoryTab({ isDark }) {
  const bdr = isDark ? '#27272a' : '#e4e4e7'
  const bg = isDark ? '#0a0a0a' : '#fff'
  const bg2 = isDark ? '#18181b' : '#f4f4f5'
  const [expandedStrategy, setExpandedStrategy] = useState(null)

  return (
    <>
      {/* Tournament Design Overview */}
      <div style={{ background: bg, borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}`, marginBottom: '16px' }}>
        <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)', marginBottom: '6px' }}>Tournament Design</div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '16px' }}>
          Game-theoretic validation of LLM trading signals through strategy competition
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          <div style={{ padding: '14px', background: bg2, borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#6366f1', marginBottom: '6px', textTransform: 'uppercase' }}>Research Question</div>
            <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Can AI-generated signals outperform traditional game-theoretic strategies across different market regimes?
            </p>
          </div>
          <div style={{ padding: '14px', background: bg2, borderRadius: '8px', borderLeft: '3px solid #10b981' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#10b981', marginBottom: '6px', textTransform: 'uppercase' }}>Hypothesis</div>
            <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              No single strategy dominates all conditions. Optimal approach varies by market regime (bull/bear/sideways).
            </p>
          </div>
          <div style={{ padding: '14px', background: bg2, borderRadius: '8px', borderLeft: '3px solid #f59e0b' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#f59e0b', marginBottom: '6px', textTransform: 'uppercase' }}>Validation Method</div>
            <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              1,800 rounds across 20 tickers with shared capital pool. Strategies compete for allocation each round.
            </p>
          </div>
        </div>
      </div>

      {/* Strategy Comparison Grid */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, marginBottom: '16px', overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Competing Strategies</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>4 strategies based on classic game theory principles</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)' }}>
          {gameTheoryStrategies.map((strategy, i) => {
            const isExp = expandedStrategy === strategy.name
            const isLeftCol = i % 2 === 0
            const isLastRow = i >= gameTheoryStrategies.length - 2

            return (
              <div
                key={strategy.name}
                style={{
                  borderRight: isLeftCol ? `1px solid ${bdr}` : 'none',
                  borderBottom: isLastRow ? 'none' : `1px solid ${bdr}`
                }}
              >
                <div
                  onClick={() => setExpandedStrategy(isExp ? null : strategy.name)}
                  style={{
                    padding: '14px 16px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderBottom: isExp ? `1px solid ${bdr}` : 'none'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                      width: '36px',
                      height: '36px',
                      background: bg2,
                      borderRadius: '8px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px'
                    }}>
                      {strategy.icon}
                    </div>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>{strategy.name}</span>
                        <span style={{
                          padding: '2px 6px',
                          background: `${strategy.color}15`,
                          color: strategy.color,
                          borderRadius: '4px',
                          fontSize: '9px',
                          fontWeight: '600'
                        }}>{strategy.shortName}</span>
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{strategy.description}</div>
                    </div>
                  </div>
                  <span style={{
                    color: 'var(--text-muted)',
                    transform: isExp ? 'rotate(180deg)' : 'none',
                    transition: 'transform 0.2s',
                    fontSize: '12px'
                  }}>▼</span>
                </div>

                {isExp && (
                  <div style={{ padding: '14px 16px', background: bg2 }}>
                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '10px', fontWeight: '600', color: strategy.color, marginBottom: '4px', textTransform: 'uppercase' }}>Behavior</div>
                      <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{strategy.behavior}</p>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>Mechanism</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
                        {strategy.mechanism.map((step, j) => (
                          <div key={j} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{
                              width: '14px', height: '14px', background: `${strategy.color}20`, color: strategy.color,
                              borderRadius: '3px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                              fontSize: '8px', fontWeight: '600', flexShrink: 0
                            }}>{j + 1}</span>
                            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div>
                        <div style={{ fontSize: '9px', fontWeight: '600', color: '#10b981', marginBottom: '4px', textTransform: 'uppercase' }}>Strengths</div>
                        {strategy.strengths.slice(0, 3).map((s, j) => (
                          <div key={j} style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>+ {s}</div>
                        ))}
                      </div>
                      <div>
                        <div style={{ fontSize: '9px', fontWeight: '600', color: '#ef4444', marginBottom: '4px', textTransform: 'uppercase' }}>Weaknesses</div>
                        {strategy.weaknesses.slice(0, 3).map((w, j) => (
                          <div key={j} style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>− {w}</div>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginTop: '12px', padding: '10px', background: bg, borderRadius: '6px', border: `1px solid ${bdr}` }}>
                      <div style={{ fontSize: '9px', fontWeight: '600', color: strategy.color, marginBottom: '4px', textTransform: 'uppercase' }}>Game Theory Role</div>
                      <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>{strategy.gameTheoryRole}</p>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Game Theory Foundation */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden', marginBottom: '16px' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Theoretical Foundation</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Classic game theory concepts applied to market competition</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)' }}>
          {[
            {
              title: 'Prisoner\'s Dilemma',
              desc: 'Each round, strategies choose positions without knowing others\' choices. Cooperation (following signals) vs defection (contrarian bets) creates classic dilemma dynamics.',
              color: '#6366f1'
            },
            {
              title: 'Iterated Games',
              desc: 'Unlike one-shot games, repeated play allows for reputation, retaliation, and learning. Tit-for-Tat famously wins in iterated tournaments.',
              color: '#10b981'
            },
            {
              title: 'Nash Equilibrium',
              desc: 'No stable dominant strategy exists across all market conditions. This validates our hypothesis about regime-dependent performance.',
              color: '#f59e0b'
            }
          ].map((concept, i) => (
            <div key={i} style={{ padding: '14px 16px', borderRight: i < 2 ? `1px solid ${bdr}` : 'none' }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: concept.color, marginBottom: '6px' }}>{concept.title}</div>
              <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.5' }}>{concept.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Regime Analysis Framework */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Market Regime Framework</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>How strategies perform varies by market conditions</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)' }}>
          {[
            {
              regime: 'Bull Market',
              icon: '📈',
              color: '#10b981',
              characteristics: ['Sustained uptrend', 'High momentum', 'Positive sentiment'],
              expected: 'Signal Follower and Cooperator expected to lead'
            },
            {
              regime: 'Bear Market',
              icon: '📉',
              color: '#ef4444',
              characteristics: ['Sustained downtrend', 'Fear dominant', 'Risk-off flows'],
              expected: 'Defector may capitalize on crowd mistakes'
            },
            {
              regime: 'Sideways',
              icon: '↔️',
              color: '#f59e0b',
              characteristics: ['Range-bound', 'Low conviction', 'Mixed signals'],
              expected: 'Tit-for-Tat adapts best to choppy conditions'
            }
          ].map((r, i) => (
            <div key={i} style={{ padding: '14px 16px', borderRight: i < 2 ? `1px solid ${bdr}` : 'none' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                <span style={{ fontSize: '18px' }}>{r.icon}</span>
                <span style={{ fontSize: '12px', fontWeight: '600', color: r.color }}>{r.regime}</span>
              </div>
              <div style={{ marginBottom: '10px' }}>
                {r.characteristics.map((c, j) => (
                  <div key={j} style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>• {c}</div>
                ))}
              </div>
              <div style={{ padding: '8px', background: bg2, borderRadius: '4px' }}>
                <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginBottom: '2px', textTransform: 'uppercase' }}>Hypothesis</div>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{r.expected}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Research Insights */}
      <div style={{ marginTop: '16px', padding: '16px', background: bg2, borderRadius: '10px', border: `1px solid ${bdr}` }}>
        <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '10px' }}>Key Research Insights</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
          {[
            { insight: 'No universal winner', detail: 'Different strategies excel in different regimes, validating our core hypothesis.' },
            { insight: 'Cooperation has value', detail: 'Following LLM signals (cooperation) provides consistent, if not spectacular, returns.' },
            { insight: 'Defection is high-variance', detail: 'Contrarian bets win big or lose big - high individual round wins but inconsistent overall.' },
            { insight: 'Adaptation beats rigidity', detail: 'Tit-for-Tat\'s ability to adapt makes it the most consistent across all regimes.' }
          ].map((item, i) => (
            <div key={i} style={{ padding: '10px 12px', background: bg, borderRadius: '6px', border: `1px solid ${bdr}` }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>{item.insight}</div>
              <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.4' }}>{item.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

// ========== LLM ARENA TAB COMPONENT ==========
function LLMArenaTab({ isDark }) {
  const bdr = isDark ? '#27272a' : '#e4e4e7'
  const bg = isDark ? '#0a0a0a' : '#fff'
  const bg2 = isDark ? '#18181b' : '#f4f4f5'
  const [expandedModel, setExpandedModel] = useState(null)

  return (
    <>
      {/* Arena Design Overview */}
      <div style={{ background: bg, borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}`, marginBottom: '16px' }}>
        <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)', marginBottom: '6px' }}>Arena Design</div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '16px' }}>
          Head-to-head comparison of LLM reasoning capabilities on identical market scenarios
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          <div style={{ padding: '14px', background: bg2, borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#6366f1', marginBottom: '6px', textTransform: 'uppercase' }}>Controlled Variable</div>
            <div style={{ fontSize: '12px', color: 'var(--text-primary)', fontWeight: '500', marginBottom: '4px' }}>Identical Context</div>
            <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Every model receives the exact same market data, news, and technical indicators for each round.
            </p>
          </div>
          <div style={{ padding: '14px', background: bg2, borderRadius: '8px', borderLeft: '3px solid #10b981' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#10b981', marginBottom: '6px', textTransform: 'uppercase' }}>Independent Variable</div>
            <div style={{ fontSize: '12px', color: 'var(--text-primary)', fontWeight: '500', marginBottom: '4px' }}>Model Architecture</div>
            <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Different LLMs with varying sizes, training data, and reasoning approaches compete.
            </p>
          </div>
          <div style={{ padding: '14px', background: bg2, borderRadius: '8px', borderLeft: '3px solid #f59e0b' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#f59e0b', marginBottom: '6px', textTransform: 'uppercase' }}>Dependent Variable</div>
            <div style={{ fontSize: '12px', color: 'var(--text-primary)', fontWeight: '500', marginBottom: '4px' }}>Trading Performance</div>
            <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Returns are calculated based on actual market movements following each decision.
            </p>
          </div>
        </div>
      </div>

      {/* Model Comparison Grid */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, marginBottom: '16px', overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Competing Models</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>7 models across different providers and architectures</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)' }}>
          {llmModels.map((model, i) => {
            const isExp = expandedModel === model.name
            const isLastRow = i >= llmModels.length - (llmModels.length % 2 === 0 ? 2 : 1)
            const isLeftCol = i % 2 === 0

            return (
              <div
                key={model.name}
                style={{
                  borderRight: isLeftCol ? `1px solid ${bdr}` : 'none',
                  borderBottom: isLastRow ? 'none' : `1px solid ${bdr}`
                }}
              >
                <div
                  onClick={() => setExpandedModel(isExp ? null : model.name)}
                  style={{
                    padding: '14px 16px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderBottom: isExp ? `1px solid ${bdr}` : 'none'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                      width: '36px',
                      height: '36px',
                      background: bg2,
                      borderRadius: '8px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px'
                    }}>
                      {model.icon}
                    </div>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>{model.name}</span>
                        <span style={{
                          padding: '2px 6px',
                          background: `${model.color}15`,
                          color: model.color,
                          borderRadius: '4px',
                          fontSize: '9px',
                          fontWeight: '600'
                        }}>{model.tier}</span>
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{model.provider}</div>
                    </div>
                  </div>
                  <span style={{
                    color: 'var(--text-muted)',
                    transform: isExp ? 'rotate(180deg)' : 'none',
                    transition: 'transform 0.2s',
                    fontSize: '12px'
                  }}>▼</span>
                </div>

                {isExp && (
                  <div style={{ padding: '14px 16px', background: bg2 }}>
                    <p style={{ margin: '0 0 12px', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                      {model.description}
                    </p>

                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '10px', fontWeight: '600', color: model.color, marginBottom: '4px', textTransform: 'uppercase' }}>Trading Style</div>
                      <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)' }}>{model.tradingStyle}</p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div>
                        <div style={{ fontSize: '9px', fontWeight: '600', color: '#10b981', marginBottom: '4px', textTransform: 'uppercase' }}>Strengths</div>
                        {model.strengths.slice(0, 3).map((s, j) => (
                          <div key={j} style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>+ {s}</div>
                        ))}
                      </div>
                      <div>
                        <div style={{ fontSize: '9px', fontWeight: '600', color: '#ef4444', marginBottom: '4px', textTransform: 'uppercase' }}>Weaknesses</div>
                        {model.weaknesses.slice(0, 3).map((w, j) => (
                          <div key={j} style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>− {w}</div>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginTop: '12px', padding: '10px', background: bg, borderRadius: '6px', border: `1px solid ${bdr}` }}>
                      <div style={{ fontSize: '9px', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>Sample Reasoning</div>
                      <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-secondary)', fontStyle: 'italic', lineHeight: '1.4' }}>
                        "{model.sampleReasoning}"
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Evaluation Criteria */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Evaluation Metrics</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>How models are scored and ranked</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {[
            { metric: 'Total Return', desc: 'Cumulative P&L from all trades', weight: '40%', color: '#10b981' },
            { metric: 'Win Rate', desc: 'Percentage of profitable trades', weight: '25%', color: '#6366f1' },
            { metric: 'Risk-Adjusted', desc: 'Sharpe ratio of returns', weight: '20%', color: '#f59e0b' },
            { metric: 'Consistency', desc: 'Standard deviation of returns', weight: '15%', color: '#8b5cf6' }
          ].map((m, i) => (
            <div key={i} style={{ padding: '14px 16px', borderRight: i < 3 ? `1px solid ${bdr}` : 'none' }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: m.color, marginBottom: '4px' }}>{m.metric}</div>
              <p style={{ margin: '0 0 8px', fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.4' }}>{m.desc}</p>
              <div style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-primary)', fontFamily: 'monospace' }}>{m.weight}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Insights */}
      <div style={{ marginTop: '16px', padding: '16px', background: bg2, borderRadius: '10px', border: `1px solid ${bdr}` }}>
        <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '10px' }}>Key Research Insights</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
          {[
            { insight: 'Larger models don\'t always win', detail: 'Efficiency matters - GPT-o3-mini often matches larger models on simple setups.' },
            { insight: 'Trading style affects regime performance', detail: 'Aggressive models excel in trends, conservative ones in choppy markets.' },
            { insight: 'Reasoning quality > speed', detail: 'Models that explain their logic tend to have better risk management.' },
            { insight: 'Provider diversity reduces bias', detail: 'Cross-provider comparison reveals different training biases.' }
          ].map((item, i) => (
            <div key={i} style={{ padding: '10px 12px', background: bg, borderRadius: '6px', border: `1px solid ${bdr}` }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>{item.insight}</div>
              <p style={{ margin: 0, fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.4' }}>{item.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

// ========== ARCHITECTURE TAB COMPONENT ==========
function ArchitectureTab({ isDark }) {
  const bdr = isDark ? '#27272a' : '#e4e4e7'
  const bg = isDark ? '#0a0a0a' : '#fff'
  const bg2 = isDark ? '#18181b' : '#f4f4f5'

  return (
    <>
      {/* System Overview */}
      <div style={{ background: bg, borderRadius: '10px', padding: '20px', border: `1px solid ${bdr}`, marginBottom: '16px' }}>
        <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)', marginBottom: '6px' }}>Trade Arena System</div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '20px' }}>
          Multi-agent LLM market analysis with game-theoretic validation
        </div>

        {/* Data Flow */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

          {/* Row 1: Data Sources */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '100px', fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Data Layer</div>
            <div style={{ flex: 1, display: 'flex', gap: '8px' }}>
              {[
                { name: 'Yahoo Finance', desc: 'Price & volume' },
                { name: 'Finnhub', desc: 'News & sentiment' },
                { name: 'NewsAPI', desc: 'Headlines' },
                { name: 'SEC Edgar', desc: 'Filings' }
              ].map((src, i) => (
                <div key={i} style={{
                  flex: 1,
                  padding: '10px 12px',
                  background: bg2,
                  borderRadius: '6px',
                  border: `1px solid ${bdr}`,
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-primary)' }}>{src.name}</div>
                  <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>{src.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Arrow Down */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '100px' }} />
            <div style={{ flex: 1, textAlign: 'center', color: 'var(--text-muted)', fontSize: '16px' }}>↓</div>
          </div>

          {/* Row 2: Analysis Pipeline */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '100px', fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Analysis</div>
            <div style={{
              flex: 1,
              padding: '14px 16px',
              background: '#6366f115',
              borderRadius: '8px',
              border: '1px solid #6366f130',
              borderLeft: '3px solid #6366f1'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <div style={{ fontSize: '12px', fontWeight: '600', color: '#6366f1' }}>11-Agent Analysis Pipeline</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>5 phases → BUY/HOLD/SELL signal</div>
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                {['Technical', 'News', 'Fundamental', 'Macro', 'Bull/Bear', 'Debate', 'Risk Eval', 'Decision'].map((phase, i) => (
                  <div key={i} style={{
                    padding: '4px 8px',
                    background: bg,
                    borderRadius: '4px',
                    fontSize: '9px',
                    color: 'var(--text-secondary)',
                    border: `1px solid ${bdr}`
                  }}>{phase}</div>
                ))}
              </div>
            </div>
          </div>

          {/* Arrow Down Split */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '100px' }} />
            <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '40px' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: '16px' }}>↓</div>
                <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Signal</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: '16px' }}>↓</div>
                <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>Context</div>
              </div>
            </div>
          </div>

          {/* Row 3: Validation Systems */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
            <div style={{ width: '100px', fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', paddingTop: '14px' }}>Validation</div>
            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{
                padding: '14px 16px',
                background: '#f59e0b15',
                borderRadius: '8px',
                border: '1px solid #f59e0b30',
                borderLeft: '3px solid #f59e0b'
              }}>
                <div style={{ fontSize: '12px', fontWeight: '600', color: '#f59e0b', marginBottom: '6px' }}>Strategy Arena</div>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  4 game-theoretic strategies compete using LLM signals. Tests if AI beats traditional approaches.
                </div>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {['Signal', 'Coop', 'Defect', 'TFT'].map((s, i) => (
                    <span key={i} style={{ padding: '3px 6px', background: bg, borderRadius: '3px', fontSize: '9px', color: 'var(--text-muted)', border: `1px solid ${bdr}` }}>{s}</span>
                  ))}
                </div>
              </div>
              <div style={{
                padding: '14px 16px',
                background: '#8b5cf615',
                borderRadius: '8px',
                border: '1px solid #8b5cf630',
                borderLeft: '3px solid #8b5cf6'
              }}>
                <div style={{ fontSize: '12px', fontWeight: '600', color: '#8b5cf6', marginBottom: '6px' }}>LLM Arena</div>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  7 models analyze identical context. Compares reasoning capabilities across providers.
                </div>
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {['GPT-4o', 'Llama', 'Qwen', 'Kimi', '+3'].map((m, i) => (
                    <span key={i} style={{ padding: '3px 6px', background: bg, borderRadius: '3px', fontSize: '9px', color: 'var(--text-muted)', border: `1px solid ${bdr}` }}>{m}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Arrow Down */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '100px' }} />
            <div style={{ flex: 1, textAlign: 'center', color: 'var(--text-muted)', fontSize: '16px' }}>↓</div>
          </div>

          {/* Row 4: Outputs */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '100px', fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Output</div>
            <div style={{ flex: 1, display: 'flex', gap: '8px' }}>
              {[
                { name: '55K+ Decisions', color: '#10b981' },
                { name: 'Regime Analysis', color: '#6366f1' },
                { name: 'Model Rankings', color: '#8b5cf6' },
                { name: 'Strategy Insights', color: '#f59e0b' }
              ].map((out, i) => (
                <div key={i} style={{
                  flex: 1,
                  padding: '10px 12px',
                  background: `${out.color}10`,
                  borderRadius: '6px',
                  border: `1px solid ${out.color}25`,
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '11px', fontWeight: '600', color: out.color }}>{out.name}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Technical Stack */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden', marginBottom: '16px' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Technical Stack</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {[
            { category: 'Frontend', items: ['React', 'Vite', 'Next.js'], color: '#06b6d4' },
            { category: 'Backend', items: ['Python', 'FastAPI', 'Pydantic'], color: '#10b981' },
            { category: 'LLM', items: ['OpenAI API', 'Groq', 'OpenRouter'], color: '#8b5cf6' },
            {
              category: 'Data',
              color: '#f59e0b',
              groups: [
                { label: 'Market', items: ['Yahoo Finance', 'Finnhub', 'Alpaca'] },
                { label: 'News & Social', items: ['NewsAPI', 'PRAW'] },
                { label: 'Historical Records', items: [] }
              ]
            }
          ].map((stack, i) => (
            <div key={i} style={{ padding: '14px 16px', borderRight: i < 3 ? `1px solid ${bdr}` : 'none' }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: stack.color, marginBottom: '8px', textTransform: 'uppercase' }}>{stack.category}</div>
              {stack.items ? (
                stack.items.map((item, j) => (
                  <div key={j} style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px' }}>• {item}</div>
                ))
              ) : (
                stack.groups.map((group, j) => (
                  <div key={j} style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    • <span style={{ color: stack.color }}>{group.label}</span>
                    {group.items.length > 0 && (
                      <span style={{ color: 'var(--text-tertiary)', fontSize: '10px' }}> ({group.items.join(', ')})</span>
                    )}
                  </div>
                ))
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', marginBottom: '16px' }}>
        {[
          { label: 'Pipeline Agents', value: '11', color: '#6366f1' },
          { label: 'Tickers Analyzed', value: '20', color: '#10b981' },
          { label: 'Tournament Rounds', value: '1,800', color: '#f59e0b' },
          { label: 'LLM Arena Rounds', value: '383', color: '#8b5cf6' },
          { label: 'Total Decisions', value: '55K+', color: '#06b6d4' }
        ].map((stat, i) => (
          <div key={i} style={{ background: bg, borderRadius: '8px', padding: '14px', border: `1px solid ${bdr}`, textAlign: 'center' }}>
            <div style={{ fontSize: '22px', fontWeight: '700', color: stat.color, fontFamily: 'monospace' }}>{stat.value}</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Research Findings */}
      <div style={{ background: bg, borderRadius: '10px', border: `1px solid ${bdr}`, overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px', borderBottom: `1px solid ${bdr}` }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--text-primary)' }}>Research Findings</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Key conclusions from the Trade Arena experiment</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)' }}>
          {[
            {
              title: 'Core Hypothesis Validated',
              finding: 'No single strategy wins universally. Performance is regime-dependent, confirming our research question.',
              color: '#10b981'
            },
            {
              title: 'LLM Signals Have Value',
              finding: 'Signal Follower achieved 73.2% accuracy. AI-generated signals provide actionable alpha when properly validated.',
              color: '#6366f1'
            },
            {
              title: 'Model Differences Matter',
              finding: 'Different LLMs show distinct trading personalities. Architecture and training affect financial reasoning.',
              color: '#8b5cf6'
            }
          ].map((finding, i) => (
            <div key={i} style={{ padding: '16px', borderRight: i < 2 ? `1px solid ${bdr}` : 'none' }}>
              <div style={{ fontSize: '11px', fontWeight: '600', color: finding.color, marginBottom: '8px' }}>{finding.title}</div>
              <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{finding.finding}</p>
            </div>
          ))}
        </div>
      </div>


    </>
  )
}

// Main Component
export default function AgentsPage() {
  const { isDark } = useTheme()
  const [activeTab, setActiveTab] = useState('pipeline')
  const [expandedAgent, setExpandedAgent] = useState(null)
  const [selectedPhase, setSelectedPhase] = useState(null)

  const filteredAgents = selectedPhase
    ? agents.filter(a => a.phase === selectedPhase)
    : agents

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-secondary)',
      overflow: 'hidden'
    }}>
      <style>{`.hide-scroll::-webkit-scrollbar{display:none}.hide-scroll{-ms-overflow-style:none;scrollbar-width:none}`}</style>
      {/* Header */}
      <header style={{
        background: 'var(--bg-primary)',
        borderBottom: '1px solid var(--border-primary)',
        padding: '12px 20px',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              System Architecture
            </h1>
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
              Multi-agent LLM trading system with game-theoretic validation
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
            <span style={{
              padding: '5px 10px',
              background: 'var(--bg-tertiary)',
              borderRadius: '6px',
              color: 'var(--text-secondary)'
            }}>
              11 Agents
            </span>
            <span style={{
              padding: '5px 10px',
              background: 'var(--info-bg)',
              borderRadius: '6px',
              color: 'var(--info)'
            }}>
              4 Strategies
            </span>
            <span style={{
              padding: '5px 10px',
              background: 'var(--success-bg)',
              borderRadius: '6px',
              color: 'var(--success)'
            }}>
              7 Models
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '4px' }}>
          {[
            { id: 'pipeline', label: 'Analysis Pipeline' },
            { id: 'gametheory', label: 'Strategy Arena' },
            { id: 'llmarena', label: 'LLM Arena' },
            { id: 'architecture', label: 'Architecture' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setSelectedPhase(null); setExpandedAgent(null) }}
              style={{
                padding: '8px 14px',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: '500',
                background: activeTab === tab.id ? (isDark ? '#27272a' : '#e4e4e7') : 'transparent',
                color: activeTab === tab.id ? 'var(--text-primary)' : 'var(--text-muted)'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>
      <main className="hide-scroll" style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>

          {/* PIPELINE TAB */}
          {activeTab === 'pipeline' && (
            <>
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
                        ? (isDark ? '#27272a' : '#e4e4e7')
                        : 'var(--bg-tertiary)',
                      color: selectedPhase === phase.num
                        ? 'var(--text-primary)'
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
                            ? (isDark ? '#27272a' : '#e4e4e7')
                            : 'var(--bg-tertiary)',
                          color: selectedPhase === p.num
                            ? 'var(--text-primary)'
                            : 'var(--text-secondary)',
                          border: selectedPhase === p.num
                            ? '2px solid var(--border-secondary)'
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
                background: isDark ? 'var(--bg-tertiary)' : 'var(--bg-tertiary)',
                borderRadius: '14px',
                padding: '20px',
                marginTop: '20px',
                border: '1px solid var(--border-primary)'
              }}>
                <h3 style={{ margin: '0 0 16px', fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)' }}>
                  📦 Data Collection
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                  {[
                    { label: 'Total Samples', value: '1,800' },
                    { label: 'Tickers', value: '20' },
                    { label: 'Sample Interval', value: '3 days' },
                    { label: 'Time Period', value: '~12 months' }
                  ].map((stat, i) => (
                    <div key={i} style={{
                      padding: '12px',
                      background: 'var(--bg-card)',
                      borderRadius: '8px',
                      textAlign: 'center',
                      border: '1px solid var(--border-primary)'
                    }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                        {stat.label}
                      </div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
                        {stat.value}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: '16px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Tickers Analyzed
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {[
                      { sector: 'Tech', tickers: ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA'] },
                      { sector: 'Finance', tickers: ['GS', 'JPM', 'V'] },
                      { sector: 'Healthcare', tickers: ['JNJ', 'LLY', 'UNH'] },
                      { sector: 'Consumer', tickers: ['KO', 'PG', 'WMT'] },
                      { sector: 'Energy', tickers: ['CVX', 'XOM'] },
                      { sector: 'ETFs', tickers: ['QQQ', 'SPY'] },
                    ].map((group, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)', width: '70px', flexShrink: 0 }}>{group.sector}</span>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {group.tickers.map((t, j) => (
                            <span key={j} style={{
                              padding: '4px 10px',
                              background: 'var(--bg-card)',
                              border: '1px solid var(--border-primary)',
                              borderRadius: '4px',
                              fontSize: '11px',
                              fontWeight: '600',
                              color: 'var(--text-primary)',
                              fontFamily: 'monospace'
                            }}>
                              {t}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                  <p style={{ margin: '14px 0 0', fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                    Selected for sector diversity, high liquidity, and representation across market caps to ensure robust strategy evaluation across different market conditions.
                  </p>
                </div>
              </div>
            </>
          )}

          {/* GAME THEORY TAB */}
          {activeTab === 'gametheory' && (
            <GameTheoryTab isDark={isDark} />
          )}

          {/* LLM ARENA TAB */}
          {activeTab === 'llmarena' && (
            <LLMArenaTab isDark={isDark} />
          )}

          {/* ARCHITECTURE TAB */}
          {activeTab === 'architecture' && (
            <ArchitectureTab isDark={isDark} />
          )}

        </div>
      </main>
    </div>
  )
}