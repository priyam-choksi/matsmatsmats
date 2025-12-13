import React, { useState, useEffect, useRef } from 'react';

// Strategies in the game theory tournament
const STRATEGIES = {
  'Signal Follower': { short: 'Signal', color: '#10b981', desc: 'Follows LLM market signals with conviction weighting' },
  'Cooperator': { short: 'Coop', color: '#6366f1', desc: 'Momentum-based, follows trends' },
  'Defector': { short: 'Defect', color: '#ef4444', desc: 'Mean reversion, bets against trends' },
  'Tit-for-Tat': { short: 'TFT', color: '#f59e0b', desc: 'Copies the previous round winner' },
};

// Tickers available in the tournament
const TICKERS = ['AAPL', 'AMZN', 'CVX', 'GOOGL', 'GS', 'JNJ', 'JPM', 'KO', 'LLY', 'META', 'MSFT', 'NVDA', 'PG', 'QQQ', 'SPY', 'TSLA', 'UNH', 'V', 'WMT', 'XOM'];

export default function GameTheoryChat({ isDark, tournamentPath = '/data/game_theory_results/gt_tournament_20251206_141939' }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi! Ask me about the Game Theory Trading Tournament - strategy performance, ticker analysis, cooperation vs defection patterns, or how strategies competed for capital allocation." }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [allData, setAllData] = useState({});
  const [ready, setReady] = useState(false);
  const messagesEndRef = useRef(null);

  const borderColor = isDark ? '#27272a' : '#e4e4e7';
  const secondaryBg = isDark ? '#18181b' : '#f4f4f5';
  const textPrimary = isDark ? '#fafafa' : '#18181b';
  const textMuted = isDark ? '#a1a1aa' : '#71717a';

  // Load data files
  useEffect(() => {
    const loadData = async () => {
      const loaded = {};
      
      // Load combined summary (has all tickers data)
      try {
        const res = await fetch(`${tournamentPath}/combined/all_tickers_summary.json`);
        if (res.ok) {
          loaded.summary = await res.json();
          console.log('✓ Loaded all_tickers_summary');
        }
      } catch (e) { console.warn('Failed to load summary'); }

      // Load per-ticker detailed data for key tickers
      const topTickers = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'SPY'];
      for (const ticker of topTickers) {
        // Load detailed.json (has round-by-round data with reasoning)
        try {
          const res = await fetch(`${tournamentPath}/by_ticker/${ticker}/${ticker}_detailed.json`);
          if (res.ok) {
            loaded[`detailed_${ticker}`] = await res.json();
            console.log(`✓ Loaded ${ticker} detailed`);
          }
        } catch (e) { }
      }

      setAllData(loaded);
      setReady(true);
      console.log('Game Theory data loaded:', Object.keys(loaded));
    };
    loadData();
  }, [tournamentPath]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check if detailed response needed
  const wantsDetail = (q) => {
    return ['detail', 'explain', 'why', 'reason', 'how', 'elaborate', 'analysis', 'deep', 'round'].some(w => q.includes(w));
  };

  // Build context based on question
  const buildContext = (question) => {
    const q = question.toLowerCase();
    const needsDetail = wantsDetail(q);
    const summary = allData.summary;
    
    let context = `GAME THEORY TRADING TOURNAMENT
- 4 strategies compete for $1M capital allocation across 90 rounds per ticker
- 20 tickers analyzed: ${TICKERS.join(', ')}
- Benchmark: Buy-and-Hold (100% invested)
- Reallocation rate: 10% shift per round based on performance\n\n`;

    // Add strategy descriptions
    context += `STRATEGIES:
- Signal Follower: Follows LLM market signals (aggressive/neutral/conservative) with conviction weighting
- Cooperator: Momentum-based strategy, follows market trends
- Defector: Mean reversion strategy, bets against recent trends
- Tit-for-Tat: Copies the previous round's winning strategy\n\n`;

    // Detect mentioned tickers
    const mentionedTickers = [];
    TICKERS.forEach(ticker => {
      if (q.includes(ticker.toLowerCase())) mentionedTickers.push(ticker);
    });
    
    // Check aliases
    const tickerAliases = { 'nvidia': 'NVDA', 'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT', 'amazon': 'AMZN', 'google': 'GOOGL' };
    Object.entries(tickerAliases).forEach(([alias, ticker]) => {
      if (q.includes(alias) && !mentionedTickers.includes(ticker)) mentionedTickers.push(ticker);
    });

    // Detect mentioned strategies
    const mentionedStrategies = [];
    Object.keys(STRATEGIES).forEach(strat => {
      const stratLower = strat.toLowerCase();
      if (q.includes(stratLower) || 
          (strat === 'Tit-for-Tat' && (q.includes('tit') || q.includes('tft') || q.includes('t4t'))) ||
          (strat === 'Signal Follower' && q.includes('signal')) ||
          (strat === 'Cooperator' && (q.includes('coop') || q.includes('momentum'))) ||
          (strat === 'Defector' && (q.includes('defect') || q.includes('mean reversion')))) {
        mentionedStrategies.push(strat);
      }
    });

    // Add overall performance summary
    if (summary?.by_ticker && (q.includes('overall') || q.includes('best') || q.includes('winner') || q.includes('perform') || mentionedTickers.length === 0)) {
      // Aggregate across all tickers
      const totals = { 'Signal Follower': 0, 'Cooperator': 0, 'Defector': 0, 'Tit-for-Tat': 0 };
      const wins = { 'Signal Follower': 0, 'Cooperator': 0, 'Defector': 0, 'Tit-for-Tat': 0 };
      
      Object.values(summary.by_ticker).forEach(ticker => {
        Object.entries(ticker.final_allocations || {}).forEach(([strat, val]) => {
          totals[strat] = (totals[strat] || 0) + val;
        });
        Object.entries(ticker.wins_per_strategy || {}).forEach(([strat, val]) => {
          wins[strat] = (wins[strat] || 0) + val;
        });
      });

      context += `OVERALL PERFORMANCE (across all 20 tickers):
Final Capital Allocation:
${Object.entries(totals).sort((a,b) => b[1] - a[1]).map(([s, v]) => `- ${s}: ${(v/1e6).toFixed(2)}M`).join('\n')}

Total Wins:
${Object.entries(wins).sort((a,b) => b[1] - a[1]).map(([s, w]) => `- ${s}: ${w} wins`).join('\n')}\n\n`;
    }

    // Add ticker-specific data
    if (mentionedTickers.length > 0 && summary?.by_ticker) {
      mentionedTickers.forEach(ticker => {
        const tickerData = summary.by_ticker[ticker];
        if (tickerData) {
          context += `\n${ticker} RESULTS (${tickerData.total_rounds} rounds):
Final Allocations: ${Object.entries(tickerData.final_allocations || {}).map(([s, v]) => `${s}: ${(v/1000).toFixed(0)}K (${tickerData.allocation_pcts?.[s]}%)`).join(', ')}
Win Rates: ${Object.entries(tickerData.win_rates || {}).map(([s, r]) => `${s}: ${r}%`).join(', ')}
Returns: ${Object.entries(tickerData.total_returns_pct || {}).map(([s, r]) => `${s}: ${r.toFixed(1)}%`).join(', ')}
Cooperation Rate: ${(tickerData.cooperation_rate * 100).toFixed(1)}%
Benchmark (Buy-Hold): ${tickerData.benchmark_return_pct?.toFixed(1)}%\n`;
        }

        // Add round-by-round reasoning if asking for details
        if (needsDetail && allData[`detailed_${ticker}`]?.rounds) {
          const rounds = allData[`detailed_${ticker}`].rounds.slice(0, 5);
          context += `\nSAMPLE ROUNDS FOR ${ticker}:\n`;
          rounds.forEach(r => {
            context += `Round ${r.round_num} (${r.date}): Market ${r.market?.daily_return_pct >= 0 ? '+' : ''}${r.market?.daily_return_pct}%, Winner: ${r.round_results?.winner}\n`;
            if (r.strategy_decisions) {
              Object.entries(r.strategy_decisions).forEach(([strat, dec]) => {
                if (mentionedStrategies.length === 0 || mentionedStrategies.includes(strat)) {
                  context += `  ${strat}: ${dec.position_pct}% position - "${dec.reasoning?.slice(0, 100)}..."\n`;
                }
              });
            }
          });
        }
      });
    }

    // Add strategy comparison if asking about specific strategies
    if (mentionedStrategies.length > 0 && summary?.by_ticker) {
      context += `\n${mentionedStrategies.join(' vs ')} COMPARISON:\n`;
      
      // Get top 5 tickers by total volume
      const tickerResults = Object.entries(summary.by_ticker).slice(0, 5);
      tickerResults.forEach(([ticker, data]) => {
        const stratResults = mentionedStrategies.map(s => 
          `${s}: ${data.allocation_pcts?.[s]}% (${(data.final_allocations?.[s]/1000).toFixed(0)}K)`
        ).join(' vs ');
        context += `${ticker}: ${stratResults}\n`;
      });
    }

    // Add cooperation/defection analysis
    if (q.includes('cooperat') || q.includes('defect') || q.includes('game theory')) {
      context += `\nCOOPERATION VS DEFECTION ANALYSIS:\n`;
      if (summary?.by_ticker) {
        const coopRates = Object.entries(summary.by_ticker)
          .map(([t, d]) => ({ ticker: t, rate: d.cooperation_rate, defectorWins: d.wins_per_strategy?.['Defector'] || 0 }))
          .sort((a, b) => b.rate - a.rate);
        
        context += `Highest cooperation: ${coopRates.slice(0, 3).map(c => `${c.ticker} (${(c.rate*100).toFixed(0)}%)`).join(', ')}\n`;
        context += `Lowest cooperation: ${coopRates.slice(-3).map(c => `${c.ticker} (${(c.rate*100).toFixed(0)}%)`).join(', ')}\n`;
        context += `Defector dominated: ${coopRates.filter(c => c.defectorWins > 40).map(c => c.ticker).join(', ') || 'None'}\n`;
      }
    }

    console.log('Context length:', context.length);
    return context;
  };

  const handleSend = async () => {
    if (!input.trim() || loading || !ready) return;

    const question = input.trim();
    const qLower = question.toLowerCase();
    const needsDetail = wantsDetail(qLower);
    
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setLoading(true);

    try {
      const context = buildContext(question);
      
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          max_tokens: needsDetail ? 350 : 150,
          temperature: 0.4,
          messages: [
            {
              role: 'system',
              content: `You are an AI assistant that ONLY answers questions about the Game Theory Trading Tournament.

TOURNAMENT OVERVIEW:
- 4 trading strategies compete for a $1M capital pool across 90 rounds per ticker
- Each round, strategies decide position sizes (0-100% invested)
- Winner of each round gets capital reallocated from losers (10% reallocation rate)
- Benchmark: Buy-and-Hold (always 100% invested)

STRATEGIES:
- Signal Follower: Uses LLM-generated market signals (aggressive/neutral/conservative) to size positions
- Cooperator: Momentum strategy - follows market trends, goes long in uptrends
- Defector: Mean reversion strategy - bets against recent trends, contrarian
- Tit-for-Tat: Adaptive - copies whatever the previous round's winner did

KEY METRICS:
- Final Allocation: How much capital each strategy ended with (started at $250K each)
- Win Rate: % of rounds where strategy had the best return
- Cooperation Rate: How often strategies aligned with market direction
- Allocation Gini: Inequality measure of capital distribution

ALLOWED TOPICS:
- Strategy performance, rankings, win rates, capital allocation
- Ticker-specific analysis (AAPL, NVDA, TSLA, etc.)
- Cooperation vs defection patterns, game theory dynamics
- Strategy reasoning and decision-making
- Comparisons between strategies
- Insights and learnings from the tournament

OFF-TOPIC (redirect):
Weather, personal questions, pokemon, coding, recipes, general knowledge.

REDIRECT RESPONSE:
"I only answer questions about the Game Theory Trading Tournament. Try asking about strategy performance, ticker results, or cooperation patterns."

GREETING RESPONSE:
"Hi! I can help you explore the Game Theory Tournament. Ask about strategy rankings, how Defector vs Cooperator performed, or specific ticker results!"

RESPONSE STYLE:
- Write naturally in paragraphs, NOT bullet points
- ${needsDetail ? 'Give ~200 words with specific reasoning from the data.' : 'Keep it to ~80-100 words.'}
- Use specific numbers: "Defector won 44 rounds on AAPL with 33.7% final allocation"
- Explain game theory dynamics naturally

TOURNAMENT DATA:
${context}`
            },
            { role: 'user', content: question }
          ]
        })
      });

      const result = await res.json();
      
      if (result.error) {
        setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${result.error.message}` }]);
      } else {
        const answer = result.choices?.[0]?.message?.content || "Couldn't process that.";
        setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    }

    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{ padding: '10px 12px', borderBottom: `1px solid ${borderColor}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '11px', fontWeight: '600', color: textPrimary }}>🎮 Ask AI</span>
          <span style={{ fontSize: '9px', color: ready ? '#10b981' : '#f59e0b' }}>
            {ready ? `● ${Object.keys(allData).length} sources` : '○ Loading...'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="hide-scroll" style={{ flex: 1, overflowY: 'auto', padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <div style={{
              maxWidth: '88%',
              padding: '8px 10px',
              borderRadius: m.role === 'user' ? '10px 10px 4px 10px' : '10px 10px 10px 4px',
              fontSize: '11px',
              lineHeight: '1.5',
              whiteSpace: 'pre-wrap',
              background: m.role === 'user' ? '#6366f1' : secondaryBg,
              color: m.role === 'user' ? '#fff' : textPrimary,
              border: m.role === 'user' ? 'none' : `1px solid ${borderColor}`
            }}>
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ fontSize: '10px', color: textMuted, padding: '6px' }}>Analyzing...</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Questions */}
      {messages.length < 3 && (
        <div style={{ padding: '6px 10px', display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {['Best strategy?', 'NVDA results', 'TitForTat vs Defector'].map(q => (
            <button key={q} onClick={() => setInput(q)} style={{
              padding: '3px 7px', fontSize: '9px', borderRadius: '4px',
              border: `1px solid ${borderColor}`, background: 'transparent',
              color: textMuted, cursor: 'pointer'
            }}>{q}</button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ padding: '10px', borderTop: `1px solid ${borderColor}` }}>
        <div style={{ display: 'flex', gap: '6px' }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Ask about strategies..."
            disabled={loading || !ready}
            style={{
              flex: 1, padding: '8px 10px', borderRadius: '6px',
              border: `1px solid ${borderColor}`, background: secondaryBg,
              color: textPrimary, fontSize: '11px', outline: 'none'
            }}
          />
          <button onClick={handleSend} disabled={loading || !ready} style={{
            padding: '8px 12px', borderRadius: '6px', border: 'none',
            background: '#6366f1', color: '#fff', fontSize: '11px',
            fontWeight: '600', cursor: 'pointer', opacity: (loading || !ready) ? 0.5 : 1
          }}>
            Go
          </button>
        </div>
      </div>
    </div>
  );
}