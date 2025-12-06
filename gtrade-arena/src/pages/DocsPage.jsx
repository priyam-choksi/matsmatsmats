import React, { useState } from 'react'
import { useTheme } from '../App'

// ========== FAQ DATA ==========
const faqCategories = [
  {
    id: 'overview',
    name: 'Overview',
    icon: '🏠',
    faqs: [
      {
        q: 'What is GTrade?',
        a: `GTrade is a research platform studying multi-agent market analysis with game-theoretic validation. It was built for DAMG 7374 (Intelligent Analytics) at Northeastern University.

The platform combines:
• **11 LLM-powered agents** for market analysis
• **5-phase analysis pipeline** from data collection to final decision
• **Game theory tournament** where strategies compete for capital
• **Real-time market data** visualization

This is an academic research project exploring whether AI agents can make better trading decisions than traditional strategies.`
      },
      {
        q: 'What is the research question?',
        a: `**Primary Research Question:**
"Do different trading strategies perform optimally under different market regimes?"

**Hypothesis:**
LLM-powered signal generation combined with game theory capital allocation can outperform traditional strategies by adapting to market conditions.

**Key Findings:**
• Signal Follower (LLM-based) achieved +14.93% returns
• Outperformed Buy-and-Hold benchmark by +2.48%
• Different strategies excel in different market regimes
• 73.2% signal accuracy validated through Monte Carlo simulations`
      },
      {
        q: 'Is this financial advice?',
        a: `**No. This is purely academic research.**

GTrade is a research project for educational purposes only. It should never be used for actual trading decisions. The simulated returns and performance metrics are based on historical backtesting and do not guarantee future results.

**Disclaimers:**
• Past performance does not indicate future results
• Simulated trading does not reflect real market conditions
• No real money is involved in this research
• This is not investment advice

Always consult a qualified financial advisor before making investment decisions.`
      }
    ]
  },
  {
    id: 'pipeline',
    name: 'Analysis Pipeline',
    icon: '🔄',
    faqs: [
      {
        q: 'How does the analysis pipeline work?',
        a: `The pipeline consists of **5 phases** with **11 specialized agents**:

**Phase 1: Market Analysis** (~60s)
4 analysts gather data:
• Technical Analyst - RSI, MACD, support/resistance
• News Analyst - Sentiment from news sources
• Fundamental Analyst - P/E, growth, margins
• Macro Analyst - Fed policy, market regime

**Phase 2: Bull/Bear Research** (~40s)
2 researchers build opposing cases:
• Bull Researcher - Upside thesis with catalysts
• Bear Researcher - Downside risks and headwinds

**Phase 3: Debate & Synthesis** (~30s)
1 manager moderates:
• Research Manager - Weighs arguments, assigns probabilities

**Phase 4: Risk Evaluation** (~30s)
3 evaluators with different personalities:
• Aggressive - Risk-seeking, higher positions
• Neutral - Balanced view
• Conservative - Risk-averse, capital preservation

**Phase 5: Final Decision** (~10s)
1 decision maker:
• Risk Manager - Final BUY/HOLD/REJECT with position sizing`
      },
      {
        q: 'What LLM model is used?',
        a: `The pipeline uses **GPT-4o-mini** through OpenAI's API.

**Why GPT-4o-mini?**
• Cost-effective for high-volume analysis
• Fast inference time (~2-3s per agent)
• Good balance of capability and speed
• Supports structured JSON outputs

**Technical Stack:**
• LangGraph for agent orchestration
• LangChain for LLM interactions
• Pydantic for output validation
• Async processing for parallel execution

Each agent has a specialized system prompt that defines its personality, responsibilities, and output format.`
      },
      {
        q: 'How are position sizes determined?',
        a: `Position sizing follows a **consensus-based approach**:

**Step 1: Evaluator Recommendations**
Each of the 3 risk evaluators recommends a position size (0-100%):
• Aggressive: Typically 50-90%
• Neutral: Typically 25-50%
• Conservative: Typically 5-25%

**Step 2: Weighted Average**
The Risk Manager calculates a weighted average based on:
• Evaluator confidence levels
• Market regime (bull/bear/sideways)
• Volatility conditions

**Step 3: Portfolio Constraints**
Final position is adjusted for:
• Maximum position size limits
• Available capital
• Risk budget allocation

**Example:**
If Aggressive says 80%, Neutral says 40%, Conservative says 15%
→ Weighted average might be ~45%
→ Applied to $100K portfolio = $45,000 position`
      }
    ]
  },
  {
    id: 'tournament',
    name: 'Game Theory Tournament',
    icon: '🏆',
    faqs: [
      {
        q: 'What is the tournament?',
        a: `The tournament is a **game-theoretic competition** where 4 trading strategies compete for shared capital over 90 rounds.

**Key Concept: Strategic Interdependence**
Unlike traditional backtesting where strategies run independently, our tournament creates genuine competition:
• All strategies share a **$1M capital pool**
• After each round, capital **reallocates** based on performance
• Winners take capital from losers
• Your payoff depends on others' actions

**This creates true game theory dynamics:**
• Strategies must consider what others will do
• Cooperation vs defection tradeoffs
• Regime-dependent optimal strategies`
      },
      {
        q: 'What are the 4 strategies?',
        a: `**1. Signal Follower 🤖**
• Uses the 11-agent LLM pipeline
• Follows risk evaluator recommendations
• Adapts position size to confidence
• Best in: Bull markets (+18.2%)

**2. Cooperator 🤝**
• Blends LLM signals with group consensus
• Weights decisions toward group average
• Builds trust through consistency
• Best in: Sideways markets (+5.2%)

**3. Defector 🎯**
• Takes contrarian positions
• Bets against consensus
• High risk, high potential reward
• Best in: Bear markets (+6.2%)

**4. Tit-for-Tat 🔄**
• Classic game theory strategy
• Mirrors previous round winner
• Adaptive and reactive
• Performs well across regimes`
      },
      {
        q: 'How does capital reallocation work?',
        a: `Capital reallocates using **z-score based transfers**:

**Step 1: Calculate Returns**
Each strategy's return for the round is calculated based on:
• Position size
• Market movement
• Win/loss outcome

**Step 2: Compute Z-Scores**
Returns are standardized to z-scores:
\`z = (return - mean) / std_dev\`

**Step 3: Transfer Capital**
Capital flows from underperformers to outperformers:
• Positive z-score → Gain capital
• Negative z-score → Lose capital
• Transfer amount = 10% of pool × z-score

**Example:**
If Signal Follower has z = +1.5 and Defector has z = -1.2:
• Signal Follower gains: $1M × 10% × 1.5 = $150K
• Defector loses: $1M × 10% × 1.2 = $120K

This creates pressure on underperformers and rewards consistency.`
      },
      {
        q: 'How is accuracy measured?',
        a: `Accuracy is validated through **Monte Carlo simulations**:

**Methodology:**
• 1000 simulation runs
• 11 tickers analyzed
• 90 samples per ticker
• Total: 990 data points per simulation

**Metrics Calculated:**
• **Signal Accuracy**: 73.2%
  - Percentage of correct BUY/HOLD/REJECT calls
• **Sharpe Ratio**: Risk-adjusted returns
• **Win Rate**: Percentage of profitable trades
• **Max Drawdown**: Largest peak-to-trough decline

**Confidence Intervals:**
Results include 95% confidence intervals to quantify uncertainty.

**Benchmark Comparison:**
All strategies compared against Buy-and-Hold (passive investing) to measure alpha generation.`
      }
    ]
  },
  {
    id: 'data',
    name: 'Data & Sources',
    icon: '📊',
    faqs: [
      {
        q: 'What tickers are analyzed?',
        a: `The platform analyzes **11 tickers** across 4 sectors:

**Technology (6):**
• NVDA - NVIDIA Corp (Semiconductors)
• AAPL - Apple Inc (Consumer Tech)
• MSFT - Microsoft Corp (Enterprise Tech)
• GOOGL - Alphabet Inc (Internet)
• META - Meta Platforms (Social Media)
• AMZN - Amazon.com (E-Commerce)
• TSLA - Tesla Inc (EV/Auto)

**Financials (2):**
• JPM - JPMorgan Chase (Banking)
• V - Visa Inc (Payments)

**Healthcare (2):**
• JNJ - Johnson & Johnson (Pharma)
• LLY - Eli Lilly (Biotech)

These were selected for:
• High liquidity and trading volume
• Diverse sector representation
• Quality data availability
• Research relevance`
      },
      {
        q: 'What data sources are used?',
        a: `**Price Data:**
• Yahoo Finance API - OHLCV data
• Real-time quotes and historical prices

**News & Sentiment:**
• Finnhub API - News aggregation
• Yahoo Finance - Headlines and articles
• Reuters - Market news

**Fundamental Data:**
• Finnhub API - Financial metrics
• Company filings and earnings reports

**Macro Data:**
• Economic indicators
• Fed policy statements
• Sector ETF flows

**Data Freshness:**
• Prices: Real-time (2-second refresh)
• News: Updated every 15 minutes
• Fundamentals: Updated quarterly`
      },
      {
        q: 'What is the data collection period?',
        a: `**Collection Parameters:**
• **Samples per ticker**: 90
• **Sample interval**: 3 days
• **Total period**: ~9 months
• **Total data points**: 990 (11 × 90)

**Why 3-day intervals?**
• Reduces noise from daily fluctuations
• Captures meaningful price movements
• Balances frequency with signal quality
• Aligns with typical holding periods

**Historical Range:**
Data collected from various market conditions:
• Bull market periods
• Bear market periods
• Sideways/consolidation periods

This ensures strategies are tested across different market regimes.`
      }
    ]
  },
  {
    id: 'technical',
    name: 'Technical Details',
    icon: '⚙️',
    faqs: [
      {
        q: 'What is the tech stack?',
        a: `**Backend:**
• Python 3.11+
• FastAPI - REST API framework
• LangGraph - Agent orchestration
• LangChain - LLM interactions
• Pydantic - Data validation

**AI/ML:**
• OpenAI GPT-4o-mini - LLM model
• Structured outputs with JSON mode
• Async parallel processing

**Frontend:**
• React 18 with Vite
• CSS Variables for theming
• Responsive design
• Real-time data updates

**Data:**
• Yahoo Finance API
• Finnhub API
• JSON file storage for results

**Deployment:**
• Docker containers
• Environment-based configuration`
      },
      {
        q: 'How are agents orchestrated?',
        a: `Agents are orchestrated using **LangGraph**:

**Workflow Definition:**
\`\`\`
Phase 1 (Parallel):
  Technical → 
  News      → Merge → Phase 2
  Fundamental →
  Macro     →

Phase 2 (Parallel):
  Bull Researcher →
                   → Phase 3
  Bear Researcher →

Phase 3:
  Research Manager → Phase 4

Phase 4 (Parallel):
  Aggressive →
  Neutral    → Merge → Phase 5
  Conservative →

Phase 5:
  Risk Manager → Final Output
\`\`\`

**Key Features:**
• Parallel execution where possible
• State management between phases
• Error handling and retries
• Structured output validation`
      },
      {
        q: 'How can I run this locally?',
        a: `**Prerequisites:**
• Node.js 18+
• Python 3.11+
• OpenAI API key

**Frontend Setup:**
\`\`\`bash
cd gtrade-arena
npm install
npm run dev
\`\`\`

**Backend Setup:**
\`\`\`bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python main.py
\`\`\`

**Environment Variables:**
• OPENAI_API_KEY - Required for LLM
• FINNHUB_API_KEY - For news data
• DEBUG - Enable debug logging

**Development Mode:**
The frontend runs on http://localhost:5173
The backend runs on http://localhost:8000

See the GitHub repository for full documentation.`
      }
    ]
  }
]

// Icons
const Icons = {
  ChevronDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6"/>
    </svg>
  ),
  Book: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
    </svg>
  ),
  Search: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>
    </svg>
  ),
  ExternalLink: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" x2="21" y1="14" y2="3"/>
    </svg>
  )
}

// FAQ Item Component
const FAQItem = ({ faq, isOpen, onToggle }) => {
  // Simple markdown-like formatting
  const formatText = (text) => {
    return text
      .split('\n')
      .map((line, i) => {
        // Bold text
        line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Code blocks
        line = line.replace(/`([^`]+)`/g, '<code style="background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px; font-size: 11px;">$1</code>')
        // Bullet points
        if (line.trim().startsWith('•')) {
          return `<div style="display: flex; gap: 8px; margin: 4px 0;"><span>•</span><span>${line.slice(1).trim()}</span></div>`
        }
        return line
      })
      .join('<br/>')
  }

  return (
    <div style={{
      background: 'var(--bg-primary)',
      borderRadius: '12px',
      border: '1px solid var(--border-primary)',
      overflow: 'hidden',
      marginBottom: '8px'
    }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%',
          padding: '16px',
          textAlign: 'left',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          gap: '12px'
        }}
      >
        <span style={{
          fontSize: '14px',
          fontWeight: '600',
          color: 'var(--text-primary)'
        }}>
          {faq.q}
        </span>
        <span style={{
          color: 'var(--text-muted)',
          transform: isOpen ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
          flexShrink: 0
        }}>
          <Icons.ChevronDown size={18} />
        </span>
      </button>
      {isOpen && (
        <div style={{
          padding: '0 16px 16px',
          fontSize: '13px',
          color: 'var(--text-secondary)',
          lineHeight: '1.6',
          borderTop: '1px solid var(--border-primary)',
          paddingTop: '16px'
        }}
          dangerouslySetInnerHTML={{ __html: formatText(faq.a) }}
        />
      )}
    </div>
  )
}

// Main Component
export default function DocsPage() {
  const { isDark } = useTheme()
  const [selectedCategory, setSelectedCategory] = useState('overview')
  const [openFAQ, setOpenFAQ] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')

  const currentCategory = faqCategories.find(c => c.id === selectedCategory)

  // Filter FAQs based on search
  const filteredFAQs = searchQuery
    ? faqCategories.flatMap(cat => 
        cat.faqs.filter(faq => 
          faq.q.toLowerCase().includes(searchQuery.toLowerCase()) ||
          faq.a.toLowerCase().includes(searchQuery.toLowerCase())
        ).map(faq => ({ ...faq, category: cat.name }))
      )
    : currentCategory?.faqs || []

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
            <Icons.Book size={18} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Documentation
            </h1>
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
              FAQ & Technical Reference
            </p>
          </div>
        </div>

        {/* Search */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'var(--bg-tertiary)',
          borderRadius: '10px',
          padding: '0 12px',
          width: '300px'
        }}>
          <Icons.Search size={16} style={{ color: 'var(--text-muted)' }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documentation..."
            style={{
              flex: 1,
              padding: '9px 0',
              border: 'none',
              background: 'transparent',
              fontSize: '13px',
              outline: 'none',
              color: 'var(--text-primary)'
            }}
          />
          {searchQuery && (
            <button 
              onClick={() => setSearchQuery('')}
              style={{ 
                background: 'none', 
                border: 'none', 
                cursor: 'pointer', 
                color: 'var(--text-muted)',
                padding: '4px'
              }}
            >
              ✕
            </button>
          )}
        </div>
      </header>

      <div style={{ display: 'flex', maxWidth: '1200px', margin: '0 auto' }}>
        {/* Sidebar */}
        <aside style={{
          width: '240px',
          padding: '20px',
          borderRight: '1px solid var(--border-primary)',
          background: 'var(--bg-primary)',
          minHeight: 'calc(100vh - 120px)'
        }}>
          <div style={{
            fontSize: '11px',
            fontWeight: '600',
            color: 'var(--text-muted)',
            marginBottom: '12px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            Categories
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {faqCategories.map(cat => (
              <button
                key={cat.id}
                onClick={() => {
                  setSelectedCategory(cat.id)
                  setSearchQuery('')
                  setOpenFAQ(null)
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: selectedCategory === cat.id && !searchQuery
                    ? 'var(--accent-primary)'
                    : 'transparent',
                  color: selectedCategory === cat.id && !searchQuery
                    ? 'white'
                    : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: '500',
                  textAlign: 'left',
                  transition: 'all 0.15s'
                }}
              >
                <span style={{ fontSize: '16px' }}>{cat.icon}</span>
                {cat.name}
              </button>
            ))}
          </div>

          {/* Quick Links */}
          <div style={{ marginTop: '24px' }}>
            <div style={{
              fontSize: '11px',
              fontWeight: '600',
              color: 'var(--text-muted)',
              marginBottom: '12px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Quick Links
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { label: 'GitHub Repository', url: '#' },
                { label: 'API Documentation', url: '#' },
                { label: 'Research Paper', url: '#' }
              ].map((link, i) => (
                <a
                  key={i}
                  href={link.url}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '12px',
                    color: 'var(--accent-primary)',
                    textDecoration: 'none'
                  }}
                >
                  {link.label}
                  <Icons.ExternalLink size={12} />
                </a>
              ))}
            </div>
          </div>

          {/* Project Info */}
          <div style={{
            marginTop: '24px',
            padding: '14px',
            background: 'var(--bg-tertiary)',
            borderRadius: '10px'
          }}>
            <div style={{
              fontSize: '11px',
              fontWeight: '600',
              color: 'var(--text-muted)',
              marginBottom: '8px'
            }}>
              PROJECT INFO
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              <div><strong>Course:</strong> DAMG 7374</div>
              <div><strong>University:</strong> Northeastern</div>
              <div><strong>Version:</strong> 2.0</div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main style={{ flex: 1, padding: '20px', maxWidth: '800px' }}>
          {/* Search Results Header */}
          {searchQuery ? (
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{
                margin: 0,
                fontSize: '16px',
                fontWeight: '600',
                color: 'var(--text-primary)'
              }}>
                Search Results for "{searchQuery}"
              </h2>
              <p style={{
                margin: '4px 0 0',
                fontSize: '13px',
                color: 'var(--text-muted)'
              }}>
                {filteredFAQs.length} result{filteredFAQs.length !== 1 ? 's' : ''} found
              </p>
            </div>
          ) : (
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{
                margin: 0,
                fontSize: '20px',
                fontWeight: '700',
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <span style={{ fontSize: '24px' }}>{currentCategory?.icon}</span>
                {currentCategory?.name}
              </h2>
              <p style={{
                margin: '4px 0 0',
                fontSize: '13px',
                color: 'var(--text-muted)'
              }}>
                {currentCategory?.faqs.length} questions in this category
              </p>
            </div>
          )}

          {/* FAQ List */}
          <div>
            {filteredFAQs.length === 0 ? (
              <div style={{
                textAlign: 'center',
                padding: '40px',
                color: 'var(--text-muted)'
              }}>
                <p style={{ fontSize: '14px' }}>No results found for "{searchQuery}"</p>
                <p style={{ fontSize: '12px' }}>Try different keywords or browse categories</p>
              </div>
            ) : (
              filteredFAQs.map((faq, i) => (
                <div key={i}>
                  {searchQuery && faq.category && i === 0 || 
                   (searchQuery && filteredFAQs[i-1]?.category !== faq.category) ? (
                    <div style={{
                      fontSize: '11px',
                      fontWeight: '600',
                      color: 'var(--text-muted)',
                      marginTop: i > 0 ? '20px' : '0',
                      marginBottom: '8px',
                      textTransform: 'uppercase'
                    }}>
                      {faq.category}
                    </div>
                  ) : null}
                  <FAQItem
                    faq={faq}
                    isOpen={openFAQ === `${selectedCategory}-${i}` || openFAQ === `search-${i}`}
                    onToggle={() => setOpenFAQ(
                      openFAQ === `${selectedCategory}-${i}` || openFAQ === `search-${i}`
                        ? null 
                        : searchQuery ? `search-${i}` : `${selectedCategory}-${i}`
                    )}
                  />
                </div>
              ))
            )}
          </div>

          {/* Help Card */}
          {!searchQuery && (
            <div style={{
              marginTop: '32px',
              padding: '20px',
              background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
              borderRadius: '14px',
              color: 'white'
            }}>
              <h3 style={{ margin: '0 0 8px', fontSize: '16px', fontWeight: '600' }}>
                Still have questions?
              </h3>
              <p style={{ margin: '0 0 16px', fontSize: '13px', opacity: 0.9 }}>
                Check out the AI Assistant in the sidebar or explore the GitHub repository for more details.
              </p>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button style={{
                  padding: '10px 16px',
                  background: 'white',
                  color: 'var(--accent-primary)',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}>
                  💬 Ask AI Assistant
                </button>
                <button style={{
                  padding: '10px 16px',
                  background: 'rgba(255,255,255,0.2)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  View on GitHub <Icons.ExternalLink size={12} />
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}