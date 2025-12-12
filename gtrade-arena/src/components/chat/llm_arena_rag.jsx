import React, { useState, useEffect, useRef } from 'react';

const MODELS = {
  'GPT-4o-mini': { short: 'GPT-4o', color: '#10b981' },
  'Llama-3.3-70B': { short: 'Llama-70B', color: '#6366f1' },
  'Llama-4-Maverick': { short: 'Maverick', color: '#f59e0b' },
  'Kimi-K2': { short: 'Kimi', color: '#06b6d4' },
  'Qwen3-32B': { short: 'Qwen', color: '#ec4899' },
  'GPT-OSS-120B': { short: 'GPT-OSS', color: '#8b5cf6' },
  'Allam-2-7B': { short: 'Allam', color: '#f97316' },
};

const MODEL_FILES = {
  'GPT-4o-mini': 'GPT_4o_mini.json',
  'Llama-3.3-70B': 'Llama_3_3_70B.json',
  'Llama-4-Maverick': 'Llama_4_Maverick.json',
  'Kimi-K2': 'Kimi_K2.json',
  'Qwen3-32B': 'Qwen3_32B.json',
  'GPT-OSS-120B': 'GPT_OSS_120B.json',
  'Allam-2-7B': 'Allam_2_7B.json',
};

export default function TournamentChat({ data, isDark }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi! Ask me about the LLM Trading Arena - rankings, strategies, or why models made specific trades." }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelData, setModelData] = useState({});
  const [tradeHistory, setTradeHistory] = useState([]);
  const [ready, setReady] = useState(false);
  const messagesEndRef = useRef(null);

  const borderColor = isDark ? '#27272a' : '#e4e4e7';
  const secondaryBg = isDark ? '#18181b' : '#f4f4f5';
  const textPrimary = isDark ? '#fafafa' : '#18181b';
  const textMuted = isDark ? '#a1a1aa' : '#71717a';

  // Load all model files + trade history
  useEffect(() => {
    const loadData = async () => {
      const loaded = {};
      
      // Load individual model files (has detailed reasoning)
      for (const [model, file] of Object.entries(MODEL_FILES)) {
        try {
          const res = await fetch(`/data/llm_arena/model_trades/${file}`);
          if (res.ok) {
            const data = await res.json();
            loaded[model] = data;
          }
        } catch (e) {
          console.warn(`Failed to load ${model}`);
        }
      }
      
      // Load trade history (has all trades with model field)
      try {
        const res = await fetch('/data/llm_arena/trade_history.json');
        if (res.ok) {
          const history = await res.json();
          setTradeHistory(Array.isArray(history) ? history : history.trades || []);
        }
      } catch (e) {
        console.warn('Failed to load trade history');
      }

      setModelData(loaded);
      setReady(true);
      console.log('Loaded models:', Object.keys(loaded));
    };
    loadData();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check if detailed response needed
  const wantsDetail = (q) => {
    return ['detail', 'explain', 'why', 'reason', 'how', 'elaborate', 'analysis', 'think'].some(w => q.includes(w));
  };

  // Build context based on question
  const buildContext = (question) => {
    const q = question.toLowerCase();
    const needsDetail = wantsDetail(q);
    
    // Compact leaderboard
    let context = `LEADERBOARD:\n${data?.leaderboard?.map(m => 
      `${m.rank}. ${m.model}: $${(m.value/1000).toFixed(0)}K (${m.return_pct?.toFixed(1)}%) - ${m.trades} trades`
    ).join('\n')}`;

    // Detect ticker mentions
    const tickerAliases = {
      'nvidia': 'NVDA', 'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
      'amazon': 'AMZN', 'google': 'GOOGL', 'meta': 'META', 'amd': 'AMD'
    };
    let mentionedTicker = null;
    
    // Check direct ticker symbols
    ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'JPM', 'UNH', 'CVX', 'XOM'].forEach(t => {
      if (q.includes(t.toLowerCase())) mentionedTicker = t;
    });
    // Check aliases
    Object.entries(tickerAliases).forEach(([alias, ticker]) => {
      if (q.includes(alias)) mentionedTicker = ticker;
    });

    // Detect mentioned models
    const mentionedModels = [];
    Object.keys(MODELS).forEach(model => {
      const short = MODELS[model].short.toLowerCase();
      const nameLower = model.toLowerCase();
      if (q.includes(nameLower) || q.includes(short) || 
          (model.includes('Llama') && q.includes('llama')) ||
          (model.includes('GPT') && q.includes('gpt'))) {
        mentionedModels.push(model);
      }
    });

    // Check if asking about "all" models
    const askingAll = q.includes('all') || q.includes('every') || q.includes('each') || 
                      q.includes('they') || q.includes('models') || mentionedModels.length === 0;

    // Get relevant trades with reasoning
    const getTradesWithReasoning = (modelName, ticker = null) => {
      const mData = modelData[modelName];
      if (!mData?.trades) return [];
      
      let trades = mData.trades.filter(t => t.model_reasoning && t.model_reasoning.length > 50);
      
      if (ticker) {
        trades = trades.filter(t => t.ticker === ticker);
      }
      
      return trades.slice(0, 3); // Top 3 trades with reasoning
    };

    // Add reasoning based on question type
    if (q.includes('why') || q.includes('reason') || needsDetail) {
      const modelsToQuery = askingAll ? Object.keys(MODELS) : mentionedModels;
      
      context += `\n\n=== TRADE REASONING ===`;
      
      modelsToQuery.forEach(model => {
        const trades = getTradesWithReasoning(model, mentionedTicker);
        
        if (trades.length > 0) {
          context += `\n\n**${model}:**`;
          trades.forEach(t => {
            const reasoning = t.model_reasoning || t.reasoning || '';
            // Truncate but keep meaningful content
            const truncated = reasoning.length > 400 ? reasoning.slice(0, 400) + '...' : reasoning;
            context += `\n[${t.action} ${t.ticker} $${(t.amount_usd/1000).toFixed(0)}K on ${t.timestamp}]`;
            context += `\nReasoning: "${truncated}"`;
          });
        }
      });
    }

    // Add ticker-specific info
    if (mentionedTicker) {
      const tickerStats = data?.ticker_stats?.find(t => t.ticker === mentionedTicker);
      if (tickerStats) {
        context += `\n\n${mentionedTicker} STATS: ${tickerStats.trades} total trades, ${tickerStats.buys} buys, ${tickerStats.sells} sells`;
      }

      // Get trades for this ticker from trade history
      if (!q.includes('why') && !needsDetail) {
        context += `\n\n${mentionedTicker} TRADES BY MODEL:`;
        Object.keys(MODELS).forEach(model => {
          const mData = modelData[model];
          if (!mData?.trades) return;
          
          const tickerTrades = mData.trades.filter(t => t.ticker === mentionedTicker);
          const buys = tickerTrades.filter(t => t.action === 'BUY').length;
          const sells = tickerTrades.filter(t => t.action === 'SELL').length;
          const volume = tickerTrades.reduce((sum, t) => sum + (t.amount_usd || 0), 0);
          
          if (tickerTrades.length > 0) {
            context += `\n${MODELS[model].short}: ${buys} buys, ${sells} sells, $${(volume/1000).toFixed(0)}K volume`;
          }
        });
      }
    }

    // Add model stats if specific model mentioned
    if (mentionedModels.length > 0 && !q.includes('why')) {
      mentionedModels.forEach(model => {
        const mData = modelData[model];
        if (mData?.stats) {
          context += `\n\n${model} STATS:`;
          context += `\n- Total trades: ${mData.stats.total_trades}`;
          context += `\n- Buy/Sell: ${mData.stats.buy_count}/${mData.stats.sell_count} (${mData.stats.buy_ratio?.toFixed(1)}% buys)`;
          context += `\n- Favorite ticker: ${mData.stats.favorite_ticker} (${mData.stats.favorite_ticker_count} trades)`;
          context += `\n- Avg trade size: $${(mData.stats.avg_trade_size/1000).toFixed(1)}K`;
        }
      });
    }

    console.log('Context length:', context.length);
    console.log('Context preview:', context.slice(0, 800));
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
              content: `You are an AI assistant that ONLY answers questions about the LLM Trading Arena tournament.

ALLOWED TOPICS (answer these):
- Model rankings, performance, returns (Llama, GPT, Qwen, Kimi, Allam, Maverick)
- Trading decisions: why models bought/sold stocks
- Stock analysis: NVDA, AAPL, TSLA, MSFT, AMZN, etc.
- Strategies, comparisons, market conditions during the tournament
- Insights, learnings, takeaways, conclusions from the tournament
- Questions about "this", "the tournament", "the arena", "the data", "the results"
- General greetings like "hi" or "hello" (respond briefly and offer to help with tournament questions)

OFF-TOPIC (redirect these):
Weather, personal questions, pokemon, coding help, recipes, general knowledge, news, math problems, writing tasks unrelated to the tournament.

HOW TO DECIDE:
- If the question mentions tournament, models, trading, stocks, or asks about insights/learnings → ANSWER IT
- If the question could reasonably relate to analyzing this trading competition → ANSWER IT
- If the question is clearly about something else entirely → REDIRECT

REDIRECT RESPONSE (use only for clearly off-topic):
"I only answer questions about the LLM Trading Arena tournament. Try asking about model rankings, trading strategies, or why models bought specific stocks like NVDA or AAPL."

GREETING RESPONSE (for hi/hello):
"Hi! I can help you explore the LLM Trading Arena results. Ask me about model rankings, why they bought certain stocks, or trading strategies!"

RESPONSE STYLE (for relevant questions):
- Write naturally in flowing paragraphs, NOT bullet points or numbered lists
- ${needsDetail ? 'Give ~200 words with specific quotes from reasoning.' : 'Keep it to ~80-100 words max.'}
- Summarize patterns across models, don't list each separately
- Use phrases like "Most models...", "The key insight is...", "The data shows..."
- Sound like a human analyst giving a quick briefing

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
          <span style={{ fontSize: '11px', fontWeight: '600', color: textPrimary }}>🤖 Ask AI</span>
          <span style={{ fontSize: '9px', color: ready ? '#10b981' : '#f59e0b' }}>
            {ready ? `● ${Object.keys(modelData).length} models loaded` : '○ Loading...'}
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
              background: m.role === 'user' ? '#10b981' : secondaryBg,
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
          {['Who won?', 'Why buy NVDA?', 'Compare Llama vs Qwen'].map(q => (
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
            placeholder="Ask anything..."
            disabled={loading || !ready}
            style={{
              flex: 1, padding: '8px 10px', borderRadius: '6px',
              border: `1px solid ${borderColor}`, background: secondaryBg,
              color: textPrimary, fontSize: '11px', outline: 'none'
            }}
          />
          <button onClick={handleSend} disabled={loading || !ready} style={{
            padding: '8px 12px', borderRadius: '6px', border: 'none',
            background: '#10b981', color: '#fff', fontSize: '11px',
            fontWeight: '600', cursor: 'pointer', opacity: (loading || !ready) ? 0.5 : 1
          }}>
            Go
          </button>
        </div>
      </div>
    </div>
  );
}