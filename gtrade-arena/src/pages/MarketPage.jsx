import React, { useState, useEffect, useCallback } from 'react'
import { useTheme } from '../App'

// ========== CONFIG ==========
const FINNHUB_KEY = import.meta.env.VITE_FINNHUB_KEY || import.meta.env.VITE_FINNHUB_API_KEY || ''

// CORS Proxies (try multiple in case one fails)
const CORS_PROXIES = [
  'https://corsproxy.io/?',
  'https://api.allorigins.win/raw?url=',
  'https://cors-anywhere.herokuapp.com/',
]

let currentProxyIndex = 0

async function fetchWithProxy(url) {
  // Try each proxy until one works
  for (let i = 0; i < CORS_PROXIES.length; i++) {
    const proxyIndex = (currentProxyIndex + i) % CORS_PROXIES.length
    const proxy = CORS_PROXIES[proxyIndex]
    try {
      const res = await fetch(proxy + encodeURIComponent(url))
      if (res.ok) {
        currentProxyIndex = proxyIndex // Remember working proxy
        return res
      }
    } catch (e) {
      console.warn(`Proxy ${proxyIndex} failed, trying next...`)
    }
  }
  throw new Error('All proxies failed')
}

const DEFAULT_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ']

const COMPANY_INFO = {
  NVDA: { name: 'NVIDIA Corp', sector: 'Semiconductors' },
  AAPL: { name: 'Apple Inc', sector: 'Consumer Tech' },
  MSFT: { name: 'Microsoft Corp', sector: 'Enterprise Tech' },
  GOOGL: { name: 'Alphabet Inc', sector: 'Internet' },
  META: { name: 'Meta Platforms', sector: 'Social Media' },
  AMZN: { name: 'Amazon.com', sector: 'E-Commerce' },
  TSLA: { name: 'Tesla Inc', sector: 'EV/Auto' },
  JPM: { name: 'JPMorgan Chase', sector: 'Banking' },
  V: { name: 'Visa Inc', sector: 'Payments' },
  JNJ: { name: 'Johnson & Johnson', sector: 'Pharma' },
  LLY: { name: 'Eli Lilly', sector: 'Biotech' },
  WMT: { name: 'Walmart Inc', sector: 'Retail' },
  PG: { name: 'Procter & Gamble', sector: 'Consumer Goods' },
  UNH: { name: 'UnitedHealth Group', sector: 'Healthcare' },
  HD: { name: 'Home Depot', sector: 'Retail' },
  MA: { name: 'Mastercard Inc', sector: 'Payments' },
  DIS: { name: 'Walt Disney Co', sector: 'Entertainment' },
  NFLX: { name: 'Netflix Inc', sector: 'Streaming' },
  AMD: { name: 'AMD Inc', sector: 'Semiconductors' },
  INTC: { name: 'Intel Corp', sector: 'Semiconductors' },
}

// ========== YAHOO FINANCE API (Primary - No API Key Needed) ==========

// Get quote + chart data in one call
async function yahooQuoteAndChart(symbol, range = '1mo', interval = '1d') {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${interval}&range=${range}`
    const res = await fetchWithProxy(url)
    const data = await res.json()
    const result = data.chart?.result?.[0]
    
    if (!result) return { quote: null, chart: [] }
    
    const meta = result.meta
    const quotes = result.indicators?.quote?.[0] || {}
    const timestamps = result.timestamp || []
    
    // Build chart from timestamps + quotes
    const chart = timestamps.map((t, i) => ({
      date: new Date(t * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      open: quotes.open?.[i],
      high: quotes.high?.[i],
      low: quotes.low?.[i],
      close: quotes.close?.[i],
      volume: quotes.volume?.[i],
    })).filter(d => d.close != null)
    
    // Build quote from meta + today's candle data
    const prevClose = meta.chartPreviousClose || meta.previousClose || meta.regularMarketPreviousClose
    const price = meta.regularMarketPrice
    const todayCandle = chart.length > 0 ? chart[chart.length - 1] : {}
    
    const quote = {
      price,
      change: price - prevClose,
      changePercent: ((price - prevClose) / prevClose) * 100,
      high: meta.regularMarketDayHigh || todayCandle.high,
      low: meta.regularMarketDayLow || todayCandle.low,
      open: meta.regularMarketOpen || todayCandle.open || quotes.open?.[0],
      prevClose,
      volume: meta.regularMarketVolume || todayCandle.volume,
      marketCap: meta.marketCap,
      fiftyTwoWeekHigh: meta.fiftyTwoWeekHigh,
      fiftyTwoWeekLow: meta.fiftyTwoWeekLow,
    }
    
    return { quote, chart }
  } catch (e) {
    console.warn(`Yahoo error for ${symbol}:`, e.message)
    return { quote: null, chart: [] }
  }
}

// Fetch additional quote data (for market cap, etc)
async function yahooQuoteSummary(symbol) {
  try {
    const url = `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${symbol}?modules=price,summaryDetail`
    const res = await fetchWithProxy(url)
    const data = await res.json()
    const price = data.quoteSummary?.result?.[0]?.price
    const summary = data.quoteSummary?.result?.[0]?.summaryDetail
    
    if (price) {
      return {
        marketCap: price.marketCap?.raw,
        open: price.regularMarketOpen?.raw,
        volume: price.regularMarketVolume?.raw,
        avgVolume: price.averageDailyVolume3Month?.raw,
        fiftyTwoWeekHigh: summary?.fiftyTwoWeekHigh?.raw,
        fiftyTwoWeekLow: summary?.fiftyTwoWeekLow?.raw,
        beta: summary?.beta?.raw,
        peRatio: summary?.trailingPE?.raw,
        dividendYield: summary?.dividendYield?.raw,
      }
    }
  } catch (e) {
    console.warn(`Yahoo summary error for ${symbol}:`, e.message)
  }
  return null
}

// Just get quote (faster for bulk)
async function yahooQuote(symbol) {
  const { quote } = await yahooQuoteAndChart(symbol, '1d', '1d')
  return quote
}

// ========== FINNHUB API (Backup + Fundamentals) ==========

async function finnhubMetrics(symbol) {
  if (!FINNHUB_KEY) return null
  try {
    const res = await fetch(`https://finnhub.io/api/v1/stock/metric?symbol=${symbol}&metric=all&token=${FINNHUB_KEY}`)
    const d = await res.json()
    return d?.metric || null
  } catch (e) { return null }
}

async function finnhubNews(symbol) {
  if (!FINNHUB_KEY) return []
  try {
    const to = new Date().toISOString().split('T')[0]
    const from = new Date(Date.now() - 7 * 86400000).toISOString().split('T')[0]
    const res = await fetch(`https://finnhub.io/api/v1/company-news?symbol=${symbol}&from=${from}&to=${to}&token=${FINNHUB_KEY}`)
    const d = await res.json()
    return (d || []).slice(0, 15).map((n, i) => ({
      id: i,
      headline: n.headline,
      summary: n.summary,
      source: n.source,
      url: n.url,
      datetime: n.datetime,
      sentiment: /surge|jump|beat|soar|rally|gain|upgrade/i.test(n.headline || '') ? 'bullish' :
                 /drop|fall|miss|sink|plunge|crash|downgrade/i.test(n.headline || '') ? 'bearish' : 'neutral'
    }))
  } catch (e) { return [] }
}

// ========== GOOGLE NEWS (Free Fallback) ==========

async function googleNews(symbol) {
  try {
    const company = COMPANY_INFO[symbol]?.name || symbol
    const res = await fetch(`https://news.google.com/rss/search?q=${encodeURIComponent(symbol + ' ' + company + ' stock')}&hl=en-US&gl=US&ceid=US:en`)
    const text = await res.text()
    const parser = new DOMParser()
    const xml = parser.parseFromString(text, 'text/xml')
    return Array.from(xml.querySelectorAll('item')).slice(0, 12).map((item, i) => {
      const title = item.querySelector('title')?.textContent || ''
      return {
        id: `g-${i}`,
        headline: title,
        source: item.querySelector('source')?.textContent || 'News',
        url: item.querySelector('link')?.textContent || '',
        datetime: Math.floor(new Date(item.querySelector('pubDate')?.textContent || '').getTime() / 1000),
        sentiment: /surge|jump|beat|soar|rally|gain|upgrade/i.test(title) ? 'bullish' :
                   /drop|fall|miss|sink|plunge|crash|downgrade/i.test(title) ? 'bearish' : 'neutral'
      }
    })
  } catch (e) { return [] }
}

// ========== SEC FILINGS (via proxy) ==========

async function secFilings(symbol) {
  try {
    const url = `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${symbol}&type=&dateb=&owner=include&count=10&output=atom`
    const res = await fetchWithProxy(url)
    const text = await res.text()
    const parser = new DOMParser()
    const xml = parser.parseFromString(text, 'text/xml')
    return Array.from(xml.querySelectorAll('entry')).slice(0, 10).map((e, i) => ({
      id: i,
      title: e.querySelector('title')?.textContent || '',
      link: e.querySelector('link')?.getAttribute('href') || '',
      updated: e.querySelector('updated')?.textContent || '',
      type: e.querySelector('category')?.getAttribute('term') || 'Filing'
    }))
  } catch (e) { 
    console.warn('SEC error:', e.message)
    return [] 
  }
}

// ========== CHART COMPONENT ==========
function PriceChart({ data, price, isDark }) {
  const [hover, setHover] = React.useState(null) // { x, y, index, price, date }
  const svgRef = React.useRef(null)
  
  if (!data || data.length < 2) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>Loading chart...</div>
  }

  const closes = data.map(d => d.close)
  const currentPrice = price || closes[closes.length - 1]
  const allPrices = [...closes, currentPrice]
  const min = Math.min(...allPrices) * 0.995
  const max = Math.max(...allPrices) * 1.005
  const range = max - min || 1
  const isUp = currentPrice >= closes[0]
  const color = isUp ? '#22c55e' : '#ef4444'

  const w = 760, h = 200
  const pad = { t: 20, r: 60, b: 25, l: 50 }
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b

  const x = (i) => pad.l + (i / (closes.length - 1)) * cw
  const y = (v) => pad.t + ch - ((v - min) / range) * ch

  const line = closes.map((c, i) => `${i === 0 ? 'M' : 'L'}${x(i)},${y(c)}`).join(' ')
  const area = `${line} L${x(closes.length - 1)},${pad.t + ch} L${pad.l},${pad.t + ch} Z`

  // Handle mouse move
  const handleMouseMove = (e) => {
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const mouseX = ((e.clientX - rect.left) / rect.width) * w
    const mouseY = ((e.clientY - rect.top) / rect.height) * h
    
    // Find closest data point
    const chartX = mouseX - pad.l
    if (chartX < 0 || chartX > cw) {
      setHover(null)
      return
    }
    
    const index = Math.round((chartX / cw) * (closes.length - 1))
    const clampedIndex = Math.max(0, Math.min(closes.length - 1, index))
    const dataPoint = data[clampedIndex]
    
    if (dataPoint) {
      setHover({
        x: x(clampedIndex),
        y: y(dataPoint.close),
        index: clampedIndex,
        price: dataPoint.close,
        open: dataPoint.open,
        high: dataPoint.high,
        low: dataPoint.low,
        volume: dataPoint.volume,
        date: dataPoint.date,
        mouseY
      })
    }
  }

  const handleMouseLeave = () => setHover(null)

  return (
    <svg 
      ref={svgRef}
      viewBox={`0 0 ${w} ${h}`} 
      style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <defs>
        <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      
      {/* Grid */}
      {[0, 0.5, 1].map((p, i) => (
        <g key={i}>
          <line x1={pad.l} y1={pad.t + ch * (1-p)} x2={w - pad.r} y2={pad.t + ch * (1-p)} stroke={isDark ? '#333' : '#e5e5e5'} strokeDasharray="4,4" />
          <text x={pad.l - 8} y={pad.t + ch * (1-p) + 4} fill={isDark ? '#666' : '#999'} fontSize="10" textAnchor="end">${(min + range * p).toFixed(0)}</text>
        </g>
      ))}
      
      {/* Area and Line */}
      <path d={area} fill="url(#chartGrad)" />
      <path d={line} fill="none" stroke={color} strokeWidth="2" />
      
      {/* Hover crosshair and tooltip */}
      {hover && (
        <g>
          {/* Vertical line */}
          <line x1={hover.x} y1={pad.t} x2={hover.x} y2={pad.t + ch} stroke={isDark ? '#555' : '#ccc'} strokeWidth="1" strokeDasharray="4,4" />
          {/* Horizontal line */}
          <line x1={pad.l} y1={hover.y} x2={w - pad.r} y2={hover.y} stroke={isDark ? '#555' : '#ccc'} strokeWidth="1" strokeDasharray="4,4" />
          {/* Data point circle */}
          <circle cx={hover.x} cy={hover.y} r="5" fill={color} stroke={isDark ? '#18181b' : '#fff'} strokeWidth="2" />
          
          {/* Price label on right */}
          <rect x={w - pad.r + 5} y={hover.y - 10} width="50" height="20" rx="4" fill={isDark ? '#333' : '#1a1a1a'} />
          <text x={w - pad.r + 30} y={hover.y + 4} fill="#fff" fontSize="10" fontWeight="600" textAnchor="middle">${hover.price.toFixed(2)}</text>
          
          {/* Date label on bottom */}
          <rect x={hover.x - 30} y={pad.t + ch + 5} width="60" height="16" rx="3" fill={isDark ? '#333' : '#1a1a1a'} />
          <text x={hover.x} y={pad.t + ch + 16} fill="#fff" fontSize="9" textAnchor="middle">{hover.date}</text>
          
          {/* Tooltip box */}
          <g transform={`translate(${hover.x < w/2 ? hover.x + 15 : hover.x - 125}, ${Math.max(pad.t, Math.min(hover.y - 40, pad.t + ch - 80))})`}>
            <rect x="0" y="0" width="110" height="72" rx="6" fill={isDark ? '#27272a' : '#1a1a1a'} opacity="0.95" />
            <text x="8" y="16" fill="#fff" fontSize="11" fontWeight="600">{hover.date}</text>
            <text x="8" y="32" fill="#9ca3af" fontSize="9">O: <tspan fill="#fff">${hover.open?.toFixed(2) || '—'}</tspan></text>
            <text x="60" y="32" fill="#9ca3af" fontSize="9">H: <tspan fill="#22c55e">${hover.high?.toFixed(2) || '—'}</tspan></text>
            <text x="8" y="46" fill="#9ca3af" fontSize="9">L: <tspan fill="#ef4444">${hover.low?.toFixed(2) || '—'}</tspan></text>
            <text x="60" y="46" fill="#9ca3af" fontSize="9">C: <tspan fill="#fff">${hover.price?.toFixed(2) || '—'}</tspan></text>
            <text x="8" y="62" fill="#9ca3af" fontSize="9">Vol: <tspan fill="#fff">{hover.volume ? (hover.volume >= 1e6 ? (hover.volume/1e6).toFixed(1) + 'M' : hover.volume.toLocaleString()) : '—'}</tspan></text>
          </g>
        </g>
      )}
      
      {/* Current price marker (only show when not hovering) */}
      {!hover && (
        <g>
          <circle cx={x(closes.length - 1)} cy={y(currentPrice)} r="4" fill={isDark ? '#18181b' : '#fff'} stroke={color} strokeWidth="2" />
          <rect x={w - pad.r + 5} y={y(currentPrice) - 10} width="50" height="20" rx="4" fill={color} />
          <text x={w - pad.r + 30} y={y(currentPrice) + 4} fill="#fff" fontSize="10" fontWeight="600" textAnchor="middle">${currentPrice.toFixed(2)}</text>
        </g>
      )}
    </svg>
  )
}

// ========== NEWS ITEM ==========
function NewsItem({ item, expanded, onToggle }) {
  const colors = { bullish: { bg: '#dcfce7', c: '#166534' }, bearish: { bg: '#fee2e2', c: '#991b1b' }, neutral: { bg: '#f4f4f5', c: '#666' } }
  const s = colors[item.sentiment] || colors.neutral
  
  const ago = item.datetime ? (() => {
    const mins = Math.floor((Date.now() / 1000 - item.datetime) / 60)
    if (mins < 60) return `${mins}m`
    if (mins < 1440) return `${Math.floor(mins / 60)}h`
    return `${Math.floor(mins / 1440)}d`
  })() : ''

  return (
    <div onClick={onToggle} style={{ padding: '12px 0', borderBottom: '1px solid var(--border-primary)', cursor: 'pointer' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ padding: '2px 8px', borderRadius: '4px', fontSize: '9px', fontWeight: '700', background: s.bg, color: s.c, textTransform: 'uppercase' }}>{item.sentiment}</span>
        <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{ago}</span>
      </div>
      <div style={{ fontSize: '12px', fontWeight: '500', color: 'var(--text-primary)', lineHeight: 1.4 }}>{item.headline}</div>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>{item.source}</div>
      {expanded && (
        <div style={{ marginTop: '10px' }}>
          {item.summary && <p style={{ margin: '0 0 8px', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{item.summary.slice(0, 200)}...</p>}
          {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()} style={{ fontSize: '11px', color: 'var(--accent-primary)' }}>Read article →</a>}
        </div>
      )}
    </div>
  )
}

// ========== FILING ITEM ==========
function FilingItem({ item }) {
  const typeColors = { '10-K': '#8b5cf6', '10-Q': '#3b82f6', '8-K': '#f59e0b', '4': '#22c55e' }
  const color = Object.entries(typeColors).find(([k]) => item.type?.includes(k))?.[1] || '#888'
  
  return (
    <a href={item.link} target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 0', borderBottom: '1px solid var(--border-primary)', textDecoration: 'none' }}>
      <span style={{ padding: '3px 8px', borderRadius: '4px', fontSize: '9px', fontWeight: '700', background: `${color}22`, color }}>{item.type || 'SEC'}</span>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.title}</div>
        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{new Date(item.updated).toLocaleDateString()}</div>
      </div>
    </a>
  )
}

// ========== MAIN COMPONENT ==========
export default function MarketPage() {
  const { isDark } = useTheme()
  
  const [tickers, setTickers] = useState(DEFAULT_TICKERS)
  const [selected, setSelected] = useState('NVDA')
  const [quotes, setQuotes] = useState({})
  const [chart, setChart] = useState([])
  const [news, setNews] = useState([])
  const [filings, setFilings] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [tab, setTab] = useState('News')
  const [period, setPeriod] = useState('1M')
  const [expandedNews, setExpandedNews] = useState(null)
  const [searchInput, setSearchInput] = useState('')
  const [loadingQuotes, setLoadingQuotes] = useState(true)

  // Fetch all quotes using Yahoo
  const fetchAllQuotes = useCallback(async () => {
    setLoadingQuotes(true)
    const results = await Promise.all(tickers.map(t => yahooQuote(t)))
    const newQuotes = {}
    tickers.forEach((t, i) => { if (results[i]) newQuotes[t] = results[i] })
    setQuotes(newQuotes)
    setLastUpdate(new Date())
    setLoadingQuotes(false)
  }, [tickers])

  // Fetch selected ticker data
  const fetchTickerData = useCallback(async (ticker, timeRange) => {
    // Get chart with appropriate range
    const periodMap = {
      '1D': { range: '1d', interval: '5m' },
      '1W': { range: '5d', interval: '15m' },
      '1M': { range: '1mo', interval: '1d' },
      '3M': { range: '3mo', interval: '1d' }
    }
    const { range, interval } = periodMap[timeRange] || periodMap['1M']
    
    // Fetch chart + quote together
    const [{ quote, chart: chartData }, summary] = await Promise.all([
      yahooQuoteAndChart(ticker, range, interval),
      yahooQuoteSummary(ticker)
    ])
    
    // Merge quote with summary data
    if (quote) {
      const mergedQuote = {
        ...quote,
        marketCap: quote.marketCap || summary?.marketCap,
        open: quote.open || summary?.open,
        beta: summary?.beta,
        peRatio: summary?.peRatio,
        dividendYield: summary?.dividendYield,
      }
      setQuotes(prev => ({ ...prev, [ticker]: mergedQuote }))
    }
    setChart(chartData)
    
    // Fetch news - try Finnhub first, fallback to Google
    let newsData = await finnhubNews(ticker)
    if (newsData.length === 0) newsData = await googleNews(ticker)
    setNews(newsData)
    
    // Fetch filings
    const filingsData = await secFilings(ticker)
    setFilings(filingsData)
    
    // Fetch metrics from Finnhub (if available)
    const metricsData = await finnhubMetrics(ticker)
    // Merge with Yahoo summary data
    const mergedMetrics = {
      ...metricsData,
      beta: metricsData?.beta || summary?.beta,
      '52WeekHigh': metricsData?.['52WeekHigh'] || summary?.fiftyTwoWeekHigh,
      '52WeekLow': metricsData?.['52WeekLow'] || summary?.fiftyTwoWeekLow,
    }
    setMetrics(mergedMetrics)
  }, [])

  // Initial load
  useEffect(() => {
    fetchAllQuotes()
    fetchTickerData(selected, period)
    const interval = setInterval(fetchAllQuotes, 30000)
    return () => clearInterval(interval)
  }, [])

  // When ticker or period changes
  useEffect(() => {
    fetchTickerData(selected, period)
  }, [selected, period])

  // Add ticker
  const addTicker = (ticker) => {
    const t = (ticker || searchInput).toUpperCase().trim()
    if (t && t.length <= 5 && !tickers.includes(t)) {
      setTickers(prev => [...prev, t])
      setSelected(t)
      setSearchInput('')
      yahooQuote(t).then(q => q && setQuotes(prev => ({ ...prev, [t]: q })))
    }
  }

  // Remove ticker
  const removeTicker = (ticker, e) => {
    e.stopPropagation()
    if (tickers.length <= 1) return // Keep at least 1
    setTickers(prev => prev.filter(t => t !== ticker))
    setQuotes(prev => { const n = { ...prev }; delete n[ticker]; return n })
    if (selected === ticker) setSelected(tickers.find(t => t !== ticker) || tickers[0])
  }

  const quote = quotes[selected] || {}
  const company = COMPANY_INFO[selected] || { name: selected, sector: 'Equity' }
  const isUp = (quote.change || 0) >= 0
  const upCount = Object.values(quotes).filter(q => (q.change || 0) > 0).length
  const totalLoaded = Object.keys(quotes).length

  const fmt = {
    price: v => v != null ? `$${v.toFixed(2)}` : '—',
    pct: v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—',
    vol: v => !v ? '—' : v >= 1e9 ? `${(v/1e9).toFixed(1)}B` : v >= 1e6 ? `${(v/1e6).toFixed(1)}M` : v.toLocaleString(),
    cap: v => !v ? '—' : v >= 1e12 ? `$${(v/1e12).toFixed(2)}T` : v >= 1e9 ? `$${(v/1e9).toFixed(0)}B` : `$${(v/1e6).toFixed(0)}M`,
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)' }}>
      <style>{`.hide-scroll::-webkit-scrollbar{display:none}.hide-scroll{-ms-overflow-style:none;scrollbar-width:none}`}</style>

      {/* Header */}
      <header style={{ background: 'var(--bg-primary)', borderBottom: '1px solid var(--border-primary)', padding: '12px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>Market Terminal</h1>
          <p style={{ margin: '2px 0 0', fontSize: '11px', color: 'var(--text-muted)' }}>{tickers.length} tickers • Yahoo Finance + Finnhub + SEC</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <input value={searchInput} onChange={e => setSearchInput(e.target.value.toUpperCase())} onKeyDown={e => e.key === 'Enter' && addTicker()} placeholder="Add ticker..." style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-primary)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '12px', width: '100px' }} />
          <button onClick={() => addTicker()} style={{ padding: '8px 14px', borderRadius: '6px', border: 'none', background: 'var(--accent-primary)', color: 'white', fontSize: '12px', cursor: 'pointer' }}>Add</button>
          <button onClick={fetchAllQuotes} disabled={loadingQuotes} style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-primary)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '11px', cursor: 'pointer', opacity: loadingQuotes ? 0.6 : 1 }}>↻ {loadingQuotes ? 'Loading...' : 'Refresh'}</button>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{lastUpdate?.toLocaleTimeString() || '—'}</span>
          <span style={{ padding: '5px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '600', background: upCount > totalLoaded / 2 ? '#dcfce7' : '#fee2e2', color: upCount > totalLoaded / 2 ? '#166534' : '#991b1b' }}>
            {upCount > totalLoaded / 2 ? 'Bull' : 'Bear'} ({upCount}/{totalLoaded})
          </span>
        </div>
      </header>

      {/* Ticker Strip */}
      <div className="hide-scroll" style={{ display: 'flex', gap: '8px', padding: '10px 20px', background: 'var(--bg-primary)', borderBottom: '1px solid var(--border-primary)', overflowX: 'auto' }}>
        {tickers.map(t => {
          const q = quotes[t] || {}
          const up = (q.change || 0) >= 0
          const sel = t === selected
          const hasData = q.price > 0
          return (
            <button key={t} onClick={() => setSelected(t)} style={{
              display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 14px', borderRadius: '8px', cursor: 'pointer', whiteSpace: 'nowrap', position: 'relative',
              border: sel ? `2px solid ${up ? '#22c55e' : '#ef4444'}` : '1px solid var(--border-primary)',
              background: sel ? (up ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)') : 'var(--bg-card)',
            }}>
              <span style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-primary)' }}>{t}</span>
              {hasData ? (
                <span style={{ fontSize: '12px', fontWeight: '600', color: up ? '#22c55e' : '#ef4444' }}>{up ? '▲' : '▼'}{Math.abs(q.changePercent || 0).toFixed(2)}%</span>
              ) : <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>...</span>}
              {tickers.length > 1 && (
                <span onClick={(e) => removeTicker(t, e)} style={{ marginLeft: '4px', width: '16px', height: '16px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', color: 'var(--text-muted)', cursor: 'pointer', background: 'var(--bg-tertiary)' }} title="Remove">×</span>
              )}
            </button>
          )
        })}
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <main className="hide-scroll" style={{ flex: 1, padding: '16px 20px', overflowY: 'auto' }}>
          
          {/* Stock Header */}
          <div style={{ background: 'var(--bg-card)', borderRadius: '12px', padding: '20px', marginBottom: '16px', border: '1px solid var(--border-primary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
                  <span style={{ fontSize: '28px', fontWeight: '700', color: 'var(--text-primary)' }}>{selected}</span>
                  <span style={{ padding: '4px 10px', background: 'var(--bg-tertiary)', borderRadius: '6px', fontSize: '11px', color: 'var(--text-muted)' }}>{company.sector}</span>
                </div>
                <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{company.name}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '32px', fontWeight: '700', color: 'var(--text-primary)' }}>{fmt.price(quote.price)}</div>
                <div style={{ fontSize: '16px', fontWeight: '600', color: isUp ? '#22c55e' : '#ef4444' }}>
                  {quote.change != null ? `${isUp ? '+' : ''}${quote.change.toFixed(2)} (${fmt.pct(quote.changePercent)})` : '—'}
                </div>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '10px' }}>
              {[
                { l: 'Open', v: fmt.price(quote.open) },
                { l: 'High', v: fmt.price(quote.high), c: '#22c55e' },
                { l: 'Low', v: fmt.price(quote.low), c: '#ef4444' },
                { l: 'Volume', v: fmt.vol(quote.volume) },
                { l: 'Mkt Cap', v: fmt.cap(quote.marketCap) },
                { l: 'Prev Close', v: fmt.price(quote.prevClose) },
              ].map((s, i) => (
                <div key={i} style={{ padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '10px', textAlign: 'center' }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>{s.l}</div>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: s.c || 'var(--text-primary)' }}>{s.v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Chart */}
          <div style={{ background: 'var(--bg-card)', borderRadius: '12px', padding: '20px', marginBottom: '16px', border: '1px solid var(--border-primary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)' }}>Price Chart</h3>
              <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-tertiary)', borderRadius: '8px', padding: '3px' }}>
                {['1D', '1W', '1M', '3M'].map(p => (
                  <button key={p} onClick={() => setPeriod(p)} style={{
                    padding: '6px 12px', fontSize: '11px', fontWeight: '500', borderRadius: '6px', border: 'none', cursor: 'pointer',
                    background: period === p ? 'var(--bg-primary)' : 'transparent',
                    color: period === p ? 'var(--text-primary)' : 'var(--text-muted)',
                  }}>{p}</button>
                ))}
              </div>
            </div>
            <div style={{ height: '200px' }}>
              <PriceChart data={chart} price={quote.price} isDark={isDark} />
            </div>
          </div>

          {/* Watchlist */}
          <div style={{ background: 'var(--bg-card)', borderRadius: '12px', border: '1px solid var(--border-primary)', overflow: 'hidden' }}>
            <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border-primary)', display: 'flex', justifyContent: 'space-between' }}>
              <h3 style={{ margin: 0, fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)' }}>Watchlist</h3>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{totalLoaded} loaded</span>
            </div>
            <div className="hide-scroll" style={{ maxHeight: '300px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-tertiary)' }}>
                    {['Symbol', 'Company', 'Price', 'Change', '%', ''].map((h, i) => (
                      <th key={i} style={{ padding: '10px 14px', textAlign: i < 2 ? 'left' : 'right', fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', width: i === 5 ? '40px' : 'auto' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tickers.map(t => {
                    const q = quotes[t] || {}
                    const c = COMPANY_INFO[t] || { name: t }
                    const up = (q.change || 0) >= 0
                    const hasData = q.price > 0
                    return (
                      <tr key={t} onClick={() => setSelected(t)} style={{ cursor: 'pointer', borderBottom: '1px solid var(--border-primary)', background: t === selected ? (isDark ? '#1a1a1a' : '#f9f9f9') : 'transparent' }}>
                        <td style={{ padding: '12px 14px', fontWeight: '700', color: 'var(--text-primary)' }}>{t}</td>
                        <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>{c.name}</td>
                        <td style={{ padding: '12px 14px', textAlign: 'right', fontFamily: 'monospace', fontWeight: '600', color: 'var(--text-primary)' }}>{hasData ? fmt.price(q.price) : '—'}</td>
                        <td style={{ padding: '12px 14px', textAlign: 'right', fontFamily: 'monospace', color: up ? '#22c55e' : '#ef4444' }}>{hasData ? `${up ? '+' : ''}${q.change.toFixed(2)}` : '—'}</td>
                        <td style={{ padding: '12px 14px', textAlign: 'right', fontWeight: '600', color: up ? '#22c55e' : '#ef4444' }}>{hasData ? `${up ? '▲' : '▼'}${Math.abs(q.changePercent).toFixed(2)}%` : '—'}</td>
                        <td style={{ padding: '12px 14px', textAlign: 'center' }}>
                          {tickers.length > 1 && (
                            <span onClick={(e) => removeTicker(t, e)} style={{ width: '20px', height: '20px', borderRadius: '4px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px', color: 'var(--text-muted)', cursor: 'pointer', background: 'var(--bg-tertiary)' }} title="Remove">×</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </main>

        {/* Sidebar */}
        <aside style={{ width: '340px', background: 'var(--bg-card)', borderLeft: '1px solid var(--border-primary)', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border-primary)' }}>
            {['News', 'Filings', 'Fundamentals', 'Technical'].map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                flex: 1, padding: '12px 4px', fontSize: '10px', fontWeight: '500', background: 'none', border: 'none', cursor: 'pointer',
                borderBottom: tab === t ? '2px solid var(--text-primary)' : '2px solid transparent',
                color: tab === t ? 'var(--text-primary)' : 'var(--text-muted)',
              }}>{t}</button>
            ))}
          </div>
          <div style={{ padding: '8px 14px', borderBottom: '1px solid var(--border-primary)', fontSize: '11px', color: 'var(--text-muted)' }}>{selected} • {tab}</div>
          
          <div className="hide-scroll" style={{ flex: 1, overflowY: 'auto', padding: '0 14px' }}>
            {tab === 'News' && (
              news.length > 0 ? news.map(n => <NewsItem key={n.id} item={n} expanded={expandedNews === n.id} onToggle={() => setExpandedNews(expandedNews === n.id ? null : n.id)} />)
              : <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading news...</div>
            )}
            
            {tab === 'Filings' && (
              filings.length > 0 ? filings.map(f => <FilingItem key={f.id} item={f} />)
              : <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading filings...</div>
            )}
            
            {tab === 'Fundamentals' && (
              <div style={{ padding: '12px 0' }}>
                {metrics ? [
                  { l: 'P/E Ratio', v: metrics.peBasicExclExtraTTM?.toFixed(2) },
                  { l: 'P/S Ratio', v: metrics.psTTM?.toFixed(2) },
                  { l: 'P/B Ratio', v: metrics.pbQuarterly?.toFixed(2) },
                  { l: 'EPS (TTM)', v: metrics.epsBasicExclExtraItemsTTM ? `$${metrics.epsBasicExclExtraItemsTTM.toFixed(2)}` : null },
                  { l: 'ROE', v: metrics.roeTTM ? `${metrics.roeTTM.toFixed(1)}%` : null },
                  { l: 'ROA', v: metrics.roaTTM ? `${metrics.roaTTM.toFixed(1)}%` : null },
                  { l: 'Gross Margin', v: metrics.grossMarginTTM ? `${metrics.grossMarginTTM.toFixed(1)}%` : null },
                  { l: 'Net Margin', v: metrics.netProfitMarginTTM ? `${metrics.netProfitMarginTTM.toFixed(1)}%` : null },
                  { l: 'Revenue Growth', v: metrics.revenueGrowthTTMYoy ? `${metrics.revenueGrowthTTMYoy.toFixed(1)}%` : null },
                  { l: 'Debt/Equity', v: metrics.totalDebtToEquityQuarterly?.toFixed(2) },
                  { l: 'Current Ratio', v: metrics.currentRatioQuarterly?.toFixed(2) },
                  { l: '52W High', v: metrics['52WeekHigh'] ? `$${metrics['52WeekHigh'].toFixed(2)}` : null },
                  { l: '52W Low', v: metrics['52WeekLow'] ? `$${metrics['52WeekLow'].toFixed(2)}` : null },
                  { l: 'Beta', v: metrics.beta?.toFixed(2) },
                  { l: 'Dividend Yield', v: metrics.dividendYieldIndicatedAnnual ? `${metrics.dividendYieldIndicatedAnnual.toFixed(2)}%` : null },
                ].filter(x => x.v).map((x, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-primary)' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{x.l}</span>
                    <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-primary)' }}>{x.v}</span>
                  </div>
                )) : <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>{FINNHUB_KEY ? 'Loading...' : 'Add VITE_FINNHUB_KEY for fundamentals'}</div>}
              </div>
            )}
            
            {tab === 'Technical' && (
              <div style={{ padding: '12px 0' }}>
                {[
                  { l: 'Current Price', v: fmt.price(quote.price) },
                  { l: 'Day Range', v: quote.low && quote.high ? `${fmt.price(quote.low)} - ${fmt.price(quote.high)}` : '—' },
                  { l: 'Previous Close', v: fmt.price(quote.prevClose) },
                  { l: 'Open', v: fmt.price(quote.open) },
                  { l: 'Volume', v: fmt.vol(quote.volume) },
                  { l: 'Market Cap', v: fmt.cap(quote.marketCap) },
                  { l: 'Day Change', v: quote.change != null ? `${isUp ? '+' : ''}${quote.change.toFixed(2)}` : '—', c: isUp ? '#22c55e' : '#ef4444' },
                  { l: 'Day Change %', v: fmt.pct(quote.changePercent), c: isUp ? '#22c55e' : '#ef4444' },
                  { l: '52W High', v: quote.fiftyTwoWeekHigh ? fmt.price(quote.fiftyTwoWeekHigh) : (metrics?.['52WeekHigh'] ? `${metrics['52WeekHigh'].toFixed(2)}` : '—') },
                  { l: '52W Low', v: quote.fiftyTwoWeekLow ? fmt.price(quote.fiftyTwoWeekLow) : (metrics?.['52WeekLow'] ? `${metrics['52WeekLow'].toFixed(2)}` : '—') },
                  { l: 'Beta', v: quote.beta?.toFixed(2) || metrics?.beta?.toFixed(2) || '—' },
                  { l: 'P/E Ratio', v: quote.peRatio?.toFixed(2) || metrics?.peBasicExclExtraTTM?.toFixed(2) || '—' },
                  { l: 'Trend', v: isUp ? 'Uptrend' : 'Downtrend', c: isUp ? '#22c55e' : '#ef4444' },
                  { l: 'Signal', v: isUp ? 'BULLISH' : 'BEARISH', c: isUp ? '#22c55e' : '#ef4444' },
                ].map((x, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-primary)' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{x.l}</span>
                    <span style={{ fontSize: '12px', fontWeight: '600', color: x.c || 'var(--text-primary)' }}>{x.v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}