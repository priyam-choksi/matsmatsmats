import React, { useState, useRef, useEffect } from 'react'
import { useTheme } from '../../App'

// Icons
const Icons = {
  Send: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" />
    </svg>
  ),
  Bot: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 8V4H8" /><rect width="16" height="12" x="4" y="8" rx="2" /><path d="M2 14h2" /><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" />
    </svg>
  ),
  User: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
    </svg>
  ),
  Sparkles: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    </svg>
  )
}

// Knowledge base for simple Q&A
const knowledgeBase = {
  // Strategy info
  'signal follower': `🤖 **Signal Follower** is the LLM-powered strategy that uses our 11-agent pipeline.

**How it works:**
• Follows risk evaluator signals directly
• Uses aggressive/neutral/conservative consensus
• Currently leading with +14.93% returns

**Performance by regime:**
• Bull markets: +18.2%
• Bear markets: -2.1%
• Sideways: +4.8%

It won 32 out of 90 rounds and has the highest Sharpe ratio (1.47).`,

  'cooperator': `🤝 **Cooperator** blends LLM signals with group consensus.

**How it works:**
• Weights its decisions toward group average
• Builds trust through consistent behavior
• Performs best in stable markets

**Performance:**
• Overall return: +10.34%
• Wins: 28 out of 90 rounds
• Best in sideways markets (+5.2%)

Conservative approach that avoids extreme positions.`,

  'defector': `🎯 **Defector** takes contrarian positions against consensus.

**How it works:**
• Bets against what others are doing
• High risk, high potential reward
• Struggles in trending markets

**Performance:**
• Overall return: -12.63%
• Wins: 12 out of 90 rounds
• Actually profitable in bear markets (+6.2%)

Worst performer in bull markets but provides hedge value.`,

  'tit-for-tat': `🔄 **Tit-for-Tat** mimics the previous round's winner.

**How it works:**
• Classic game theory strategy
• Copies whatever worked last round
• Adapts to changing conditions

**Performance:**
• Overall return: -12.64%
• Wins: 18 out of 90 rounds
• Decent in sideways markets (+5.1%)

Reactive strategy that can capture momentum shifts.`,

  // General topics
  'game theory': `🎮 **Game Theory Tournament**

Our tournament creates TRUE strategic interdependence:

**Setup:**
• $1M shared capital pool
• 4 strategies compete
• 90 rounds per ticker

**Mechanics:**
• After each round, capital reallocates
• Winners take from losers via z-score
• 10% reallocation rate per round

**Why it matters:**
Unlike parallel backtesting, payoffs here depend on what OTHER strategies do. This creates genuine game-theoretic dynamics.`,

  'pipeline': `🔬 **Analysis Pipeline**

5-phase LLM workflow with 11 specialized agents:

**Phase 1 - Analysis (4 agents):**
Technical, Fundamental, News, Macro

**Phase 2 - Research (2 agents):**
Bull Researcher vs Bear Researcher

**Phase 3 - Synthesis (1 agent):**
Research Manager weighs evidence

**Phase 4 - Risk Evaluation (3 agents):**
Aggressive, Neutral, Conservative

**Phase 5 - Decision (1 agent):**
Risk Manager makes final call

Each phase outputs structured JSON for the next.`,

  'regime': `📊 **Market Regimes**

We classify markets into three regimes:

**🟢 Bull Market:**
• Signal Follower dominates (+18.2%)
• Defector struggles (-12.4%)
• Best time to follow LLM signals

**🔴 Bear Market:**
• Defector actually wins (+6.2%)
• Signal Follower loses (-6.2%)
• Contrarian approach works

**🟡 Sideways:**
• Cooperator edges ahead (+5.2%)
• Tit-for-Tat adapts well (+5.1%)
• Consensus following is safe

The tournament tests how strategies perform across ALL regimes.`,

  'winner': `🏆 **Tournament Results**

**Current Standings:**
1. 🥇 Signal Follower: +14.93% (32 wins)
2. 🥈 Cooperator: +10.34% (28 wins)
3. 🥉 Tit-for-Tat: -12.64% (18 wins)
4. Defector: -12.63% (12 wins)

**Key Insights:**
• LLM signals outperform in trending markets
• Contrarian fails in sustained bull runs
• Cooperation beats defection long-term

Signal Follower beats Buy-and-Hold benchmark by +2.48%!`,

  'agents': `🤖 **The 11 Agents**

**Analysts (Phase 1):**
📊 Technical - RSI, MACD, patterns
📋 Fundamental - P/E, growth, margins
📰 News - Sentiment analysis
🌍 Macro - Fed policy, sector rotation

**Researchers (Phase 2):**
🐂 Bull - Builds bullish case
🐻 Bear - Builds bearish case

**Synthesizer (Phase 3):**
🔬 Research Manager - Weighs evidence

**Evaluators (Phase 4):**
🔥 Aggressive - Risk-seeking
⚖️ Neutral - Balanced
🛡️ Conservative - Risk-averse

**Decision Maker (Phase 5):**
👨‍💼 Risk Manager - Final call`
}

// Find best matching response
const findResponse = (query) => {
  const q = query.toLowerCase()
  
  // Check for direct matches
  for (const [key, value] of Object.entries(knowledgeBase)) {
    if (q.includes(key)) {
      return value
    }
  }
  
  // Keyword matching
  if (q.includes('win') || q.includes('best') || q.includes('leader') || q.includes('standing')) {
    return knowledgeBase['winner']
  }
  if (q.includes('how') && q.includes('work')) {
    return knowledgeBase['pipeline']
  }
  if (q.includes('bull') || q.includes('bear') || q.includes('sideways') || q.includes('market')) {
    return knowledgeBase['regime']
  }
  if (q.includes('agent') || q.includes('llm') || q.includes('ai')) {
    return knowledgeBase['agents']
  }
  
  // Default response
  return `I can help you understand the GTrade tournament! Try asking about:

• **Strategies** - "Tell me about Signal Follower"
• **Game Theory** - "How does the tournament work?"
• **Pipeline** - "Explain the analysis pipeline"
• **Regimes** - "How do strategies perform in bull markets?"
• **Results** - "Who is winning?"
• **Agents** - "What are the 11 agents?"

What would you like to know?`
}

// Suggested questions
const suggestions = [
  "Who is winning the tournament?",
  "How does Signal Follower work?",
  "Explain the game theory",
  "What are market regimes?"
]

export default function Chat() {
  const { isDark } = useTheme()
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: `👋 Hi! I'm your GTrade research assistant.

I can answer questions about:
• Tournament strategies & performance
• The 11-agent analysis pipeline
• Game theory mechanics
• Market regime analysis

What would you like to know?`
    }
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async (text = input) => {
    if (!text.trim()) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: text.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsTyping(true)

    // Simulate AI thinking delay
    await new Promise(r => setTimeout(r, 500 + Math.random() * 500))

    const response = findResponse(text)
    
    setIsTyping(false)
    setMessages(prev => [...prev, {
      id: Date.now() + 1,
      role: 'assistant',
      content: response
    }])
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[90%] rounded-xl px-3 py-2 ${
                msg.role === 'user' ? 'rounded-br-sm' : 'rounded-bl-sm'
              }`}
              style={{
                backgroundColor: msg.role === 'user' 
                  ? 'var(--accent-primary)' 
                  : 'var(--bg-tertiary)',
                color: msg.role === 'user' 
                  ? 'white' 
                  : 'var(--text-primary)'
              }}
            >
              {/* Avatar for assistant */}
              {msg.role === 'assistant' && (
                <div className="flex items-center gap-1.5 mb-1.5">
                  <div 
                    className="w-4 h-4 rounded flex items-center justify-center"
                    style={{ backgroundColor: 'var(--accent-primary)' }}
                  >
                    <Icons.Bot size={10} className="text-white" />
                  </div>
                  <span 
                    className="text-[10px] font-semibold"
                    style={{ color: 'var(--accent-primary)' }}
                  >
                    GTrade AI
                  </span>
                </div>
              )}
              
              {/* Message content with markdown-like formatting */}
              <div 
                className="text-xs leading-relaxed whitespace-pre-wrap"
                dangerouslySetInnerHTML={{
                  __html: msg.content
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br/>')
                }}
              />
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div 
              className="px-3 py-2 rounded-xl rounded-bl-sm"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <div className="flex items-center gap-1">
                <div 
                  className="w-1.5 h-1.5 rounded-full animate-bounce"
                  style={{ backgroundColor: 'var(--text-muted)', animationDelay: '0ms' }}
                />
                <div 
                  className="w-1.5 h-1.5 rounded-full animate-bounce"
                  style={{ backgroundColor: 'var(--text-muted)', animationDelay: '150ms' }}
                />
                <div 
                  className="w-1.5 h-1.5 rounded-full animate-bounce"
                  style={{ backgroundColor: 'var(--text-muted)', animationDelay: '300ms' }}
                />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions (show only at start) */}
      {messages.length <= 2 && (
        <div className="px-3 pb-2">
          <div className="flex flex-wrap gap-1.5">
            {suggestions.map((s, i) => (
              <button
                key={i}
                onClick={() => handleSend(s)}
                className="px-2 py-1 rounded-md text-[10px] font-medium transition-colors"
                style={{
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-primary)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--accent-primary)'
                  e.currentTarget.style.color = 'var(--accent-primary)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-primary)'
                  e.currentTarget.style.color = 'var(--text-secondary)'
                }}
              >
                <Icons.Sparkles size={10} className="inline mr-1" />
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div 
        className="p-3 border-t"
        style={{ borderColor: 'var(--border-primary)' }}
      >
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about strategies..."
            className="flex-1 px-3 py-2 rounded-lg text-xs outline-none transition-colors"
            style={{
              backgroundColor: 'var(--bg-tertiary)',
              border: '1px solid var(--border-primary)',
              color: 'var(--text-primary)'
            }}
            onFocus={(e) => e.currentTarget.style.borderColor = 'var(--accent-primary)'}
            onBlur={(e) => e.currentTarget.style.borderColor = 'var(--border-primary)'}
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || isTyping}
            className="w-9 h-9 rounded-lg flex items-center justify-center transition-all"
            style={{
              background: input.trim() && !isTyping
                ? 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))'
                : 'var(--bg-tertiary)',
              color: input.trim() && !isTyping ? 'white' : 'var(--text-muted)',
              cursor: input.trim() && !isTyping ? 'pointer' : 'not-allowed'
            }}
          >
            <Icons.Send size={14} />
          </button>
        </div>
      </div>
    </div>
  )
}