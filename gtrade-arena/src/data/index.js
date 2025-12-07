/**
 * GTrade Data Layer
 * Central data management for the application
 */

// ============================================================
// CONFIGURATION
// ============================================================

// Tournament folder name - UPDATE THIS when you run a new tournament
export const TOURNAMENT_FOLDER = 'gt_tournament_20251206_141939'

export const STRATEGY_CONFIG = {
  'Signal Follower': {
    id: 'signal',
    color: '#a855f7',
    bgLight: '#faf5ff',
    bgDark: '#2e1065',
    icon: '✦',
    emoji: '🤖',
    personality: 'LLM-Powered',
    description: 'Uses 11-agent LLM pipeline for decisions'
  },
  'Cooperator': {
    id: 'cooperator',
    color: '#22c55e',
    bgLight: '#f0fdf4',
    bgDark: '#14532d',
    icon: '◆',
    emoji: '🤝',
    personality: 'Consensus',
    description: 'Follows market consensus'
  },
  'Defector': {
    id: 'defector',
    color: '#ef4444',
    bgLight: '#fef2f2',
    bgDark: '#450a0a',
    icon: '✕',
    emoji: '🎯',
    personality: 'Contrarian',
    description: 'Bets against consensus'
  },
  'Tit-for-Tat': {
    id: 'tft',
    color: '#3b82f6',
    bgLight: '#eff6ff',
    bgDark: '#1e3a8a',
    icon: '↻',
    emoji: '🔄',
    personality: 'Reactive',
    description: 'Mirrors previous round winner'
  },
  'Buy-and-Hold': {
    id: 'benchmark',
    color: '#6b7280',
    bgLight: '#f9fafb',
    bgDark: '#1f2937',
    icon: '◎',
    emoji: '📊',
    personality: 'Passive',
    description: 'Always 100% invested (benchmark)'
  }
}

export const TICKER_COLORS = {
  'AAPL': '#555555',
  'NVDA': '#76b900',
  'GOOGL': '#ea4335',
  'META': '#0866ff',
  'AMZN': '#ff9900',
  'MSFT': '#00a4ef',
  'TSLA': '#ef4444',
  'AMD': '#ed1c24',
  'MU': '#22c55e',
  'MRVL': '#8b5cf6',
  'SPY': '#0066cc',
  'IWM': '#10b981',
  'QQQ': '#6366f1',
  'CVX': '#0056a4',
  'GS': '#6fa8dc',
  'JNJ': '#d5232a',
  'JPM': '#005eb8',
  'KO': '#ed1c16',
  'LLY': '#c41230',
  'PG': '#003da5',
  'UNH': '#002677',
  'V': '#1a1f71',
  'WMT': '#0071ce',
  'XOM': '#ed1b2d'
}

export const ACTION_CONFIG = {
  'BUY': { emoji: '📈', color: '#22c55e', label: 'Bought' },
  'SELL': { emoji: '📉', color: '#ef4444', label: 'Sold' },
  'HOLD': { emoji: '⏸️', color: '#f59e0b', label: 'Hold' },
  'REJECT': { emoji: '🚫', color: '#6b7280', label: 'Rejected' },
  'STRONG BUY': { emoji: '🚀', color: '#10b981', label: 'Strong Buy' },
  'AVOID': { emoji: '⛔', color: '#dc2626', label: 'Avoid' }
}

// ============================================================
// BASE PATHS
// ============================================================

const BASE_PATH = '/data'
const GT_PATH = `${BASE_PATH}/game_theory_results/${TOURNAMENT_FOLDER}`
const WORKFLOWS_PATH = `${BASE_PATH}/workflows`

// ============================================================
// GAME THEORY DATA LOADERS
// ============================================================

/**
 * Load tournament summary for a specific ticker
 */
export async function loadTickerSummary(ticker) {
  const url = `${GT_PATH}/by_ticker/${ticker}/${ticker}_summary.json`
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to load ${ticker} summary from ${url}`)
  return response.json()
}

/**
 * Load detailed round-by-round tournament data for a ticker
 */
export async function loadTickerDetailed(ticker) {
  const url = `${GT_PATH}/by_ticker/${ticker}/${ticker}_detailed.json`
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to load ${ticker} detailed data from ${url}`)
  return response.json()
}

/**
 * Load combined summary across all tickers
 */
export async function loadCombinedSummary() {
  const url = `${GT_PATH}/combined/all_tickers_summary.json`
  const response = await fetch(url)
  if (!response.ok) throw new Error('Failed to load combined summary')
  return response.json()
}

/**
 * Get list of available tickers from the tournament
 */
export async function getAvailableTickers() {
  try {
    const combined = await loadCombinedSummary()
    // Try different possible structures
    if (combined.tickers && Array.isArray(combined.tickers)) {
      return combined.tickers
    }
    if (combined.by_ticker) {
      return Object.keys(combined.by_ticker)
    }
    // Fallback: try to list directories (won't work in browser, but good for reference)
    return ['AAPL', 'NVDA', 'GOOGL', 'META', 'AMZN', 'MSFT', 'TSLA', 'JPM', 'V', 'JNJ', 'LLY',
            'CVX', 'GS', 'KO', 'PG', 'QQQ', 'SPY', 'UNH', 'WMT', 'XOM']
  } catch (e) {
    console.warn('Could not load tickers from combined summary:', e)
    return ['AAPL', 'NVDA', 'GOOGL', 'META', 'AMZN', 'MSFT', 'TSLA', 'JPM', 'V', 'JNJ', 'LLY']
  }
}

// ============================================================
// WORKFLOW DATA LOADERS
// ============================================================

/**
 * Load the workflow manifest (list of all pipeline runs)
 */
export async function loadWorkflowManifest() {
  const response = await fetch(`${WORKFLOWS_PATH}/manifest.json`)
  if (!response.ok) throw new Error('Failed to load workflow manifest')
  return response.json()
}

/**
 * Load all data for Pipeline History page
 */
export async function loadPipelineHistoryData() {
  const manifest = await loadWorkflowManifest()
  return {
    workflows: manifest,
    totalCount: manifest.length,
    tickers: [...new Set(manifest.map(w => w.ticker))].sort()
  }
}

/**
 * Load workflow output for a specific ticker/sample
 */
export async function loadWorkflowOutput(ticker, sampleFolder) {
  const basePath = `${WORKFLOWS_PATH}/${ticker}/${sampleFolder}`
  
  const files = [
    'summary.json',
    'risk_decision.json',
    'aggressive_eval.json',
    'neutral_eval.json', 
    'conservative_eval.json',
    'research_synthesis.json',
    'bull_thesis.json',
    'bear_thesis.json',
    'discussion_points.json',
    'market_context.json',
    'execution_log.json'
  ]
  
  const results = {}
  
  for (const file of files) {
    try {
      const response = await fetch(`${basePath}/${file}`)
      if (response.ok) {
        const key = file.replace('.json', '')
        results[key] = await response.json()
      }
    } catch (e) {
      console.warn(`Could not load ${file}`)
    }
  }
  
  return results
}

/**
 * List available workflow samples for a ticker
 */
export async function listWorkflowSamples(ticker) {
  // Returns known sample folders - expand based on your data
  return [
    { folder: '2024-06-17_sample090_day268/portfolio_100000', date: '2024-06-17', sample: 90 },
    { folder: '2024-06-21_sample089_day265/portfolio_100000', date: '2024-06-21', sample: 89 },
    { folder: '2024-06-26_sample088_day262/portfolio_100000', date: '2024-06-26', sample: 88 },
  ]
}

// ============================================================
// DATA TRANSFORMERS
// ============================================================

/**
 * Transform tournament summary into strategy cards format
 */
export function transformToStrategyCards(summaryData) {
  if (!summaryData?.strategies) return []
  
  return Object.entries(summaryData.strategies).map(([name, data]) => {
    const config = STRATEGY_CONFIG[name] || {}
    return {
      id: config.id || name.toLowerCase().replace(/\s+/g, '_'),
      name,
      color: config.color || '#6b7280',
      icon: config.icon || '•',
      emoji: config.emoji || '📊',
      personality: config.personality || 'Unknown',
      description: config.description || '',
      returnPct: data.total_return_pct || 0,
      acctValue: data.final_capital || 0,
      totalPnL: (data.final_capital || 0) - (data.initial_capital || 250000),
      winRate: data.win_rate_pct || 0,
      wins: data.wins || 0,
      sharpe: data.sharpe_ratio || 0,
      maxDD: data.max_drawdown_pct || 0,
      avgPosition: data.avg_position_pct || 0
    }
  })
}

/**
 * Transform round history into chart data format
 */
export function transformToChartData(detailedData) {
  if (!detailedData?.round_history) return []
  
  return detailedData.round_history.map((round, idx) => ({
    round: round.round || idx + 1,
    date: round.date || '',
    regime: round.regime || 'sideways',
    ...Object.fromEntries(
      Object.entries(round.capital || {}).map(([strategy, capital]) => [
        STRATEGY_CONFIG[strategy]?.id || strategy.toLowerCase(),
        capital
      ])
    )
  }))
}

/**
 * Transform workflow risk_decision into feed item format
 */
export function transformToFeedItem(riskDecision, workflowSummary, ticker) {
  if (!riskDecision) return null
  
  const config = ACTION_CONFIG[riskDecision.verdict] || ACTION_CONFIG['HOLD']
  
  return {
    id: `${ticker}_${workflowSummary?.sample_number || Date.now()}`,
    ticker,
    timestamp: workflowSummary?.timestamp || new Date().toISOString(),
    analysisDate: workflowSummary?.analysis_date || '',
    action: config.label,
    actionEmoji: config.emoji,
    verdict: riskDecision.verdict,
    positionPct: riskDecision.position_sizing?.final_pct || 0,
    positionDollars: riskDecision.position_sizing?.final_dollars || 0,
    confidence: riskDecision.confidence || 'MEDIUM',
    reasoning: riskDecision.reasoning || '',
    keyFactors: riskDecision.key_factors || [],
    riskConsensus: {
      aggressive: {
        stance: riskDecision.risk_consensus?.stances?.aggressive || 'HOLD',
        position: riskDecision.risk_consensus?.position_sizes?.aggressive || 0
      },
      neutral: {
        stance: riskDecision.risk_consensus?.stances?.neutral || 'HOLD',
        position: riskDecision.risk_consensus?.position_sizes?.neutral || 0
      },
      conservative: {
        stance: riskDecision.risk_consensus?.stances?.conservative || 'HOLD',
        position: riskDecision.risk_consensus?.position_sizes?.conservative || 0
      }
    },
    redFlags: riskDecision.risk_consensus?.red_flags || []
  }
}

/**
 * Transform evaluator outputs into summary format
 */
export function transformEvaluatorSummary(aggressive, neutral, conservative) {
  return {
    aggressive: {
      stance: aggressive?.stance || 'N/A',
      position: aggressive?.position_pct || 0,
      confidence: aggressive?.confidence || 'N/A',
      reasoning: aggressive?.reasoning || ''
    },
    neutral: {
      stance: neutral?.stance || 'N/A',
      position: neutral?.position_pct || 0,
      confidence: neutral?.confidence || 'N/A',
      reasoning: neutral?.reasoning || ''
    },
    conservative: {
      stance: conservative?.stance || 'N/A',
      position: conservative?.position_pct || 0,
      confidence: conservative?.confidence || 'N/A',
      reasoning: conservative?.reasoning || ''
    },
    consensus: calculateConsensus(aggressive, neutral, conservative)
  }
}

function calculateConsensus(agg, neu, cons) {
  const stances = [agg?.stance, neu?.stance, cons?.stance].filter(Boolean)
  const buyStances = ['BUY', 'STRONG BUY']
  const sellStances = ['SELL', 'AVOID', 'REJECT']
  
  const buyCount = stances.filter(s => buyStances.includes(s)).length
  const sellCount = stances.filter(s => sellStances.includes(s)).length
  
  if (buyCount >= 2) return 'BULLISH'
  if (sellCount >= 2) return 'BEARISH'
  return 'MIXED'
}

// ============================================================
// COMPOSITE LOADERS
// ============================================================

/**
 * Load all data needed for Arena page
 */
export async function loadArenaPageData(ticker = 'AAPL') {
  try {
    const [summary, detailed] = await Promise.all([
      loadTickerSummary(ticker),
      loadTickerDetailed(ticker)
    ])
    
    return {
      strategies: transformToStrategyCards(summary),
      chartData: transformToChartData(detailed),
      tournament: {
        ticker: summary.ticker,
        totalRounds: summary.total_rounds,
        totalCapital: summary.total_capital,
        winner: summary.winner,
        regimes: summary.regime_distribution,
        cooperationRate: summary.cooperation_rate
      },
      raw: { summary, detailed }
    }
  } catch (error) {
    console.error('Failed to load arena data:', error)
    return null
  }
}

/**
 * Load all data needed for Pipeline/Market page
 */
export async function loadPipelinePageData(ticker = 'AAPL') {
  try {
    const samples = await listWorkflowSamples(ticker)
    if (samples.length === 0) return null
    
    const latestSample = samples[0]
    const workflow = await loadWorkflowOutput(ticker, latestSample.folder)
    
    return {
      ticker,
      sample: latestSample,
      decision: workflow.risk_decision,
      evaluators: transformEvaluatorSummary(
        workflow.aggressive_eval,
        workflow.neutral_eval,
        workflow.conservative_eval
      ),
      synthesis: workflow.research_synthesis,
      bullThesis: workflow.bull_thesis,
      bearThesis: workflow.bear_thesis,
      discussionPoints: workflow.discussion_points,
      marketContext: workflow.market_context,
      executionLog: workflow.execution_log,
      summary: workflow.summary,
      feedItem: transformToFeedItem(workflow.risk_decision, workflow.summary, ticker)
    }
  } catch (error) {
    console.error('Failed to load pipeline data:', error)
    return null
  }
}

/**
 * Load recent activity feed from workflow outputs
 */
export async function loadActivityFeed(tickers = ['AAPL', 'NVDA', 'GOOGL']) {
  const feedItems = []
  
  for (const ticker of tickers) {
    try {
      const data = await loadPipelinePageData(ticker)
      if (data?.feedItem) {
        feedItems.push(data.feedItem)
      }
    } catch (e) {
      console.warn(`Could not load feed for ${ticker}`)
    }
  }
  
  return feedItems.sort((a, b) => 
    new Date(b.timestamp) - new Date(a.timestamp)
  )
}

// ============================================================
// EXPORT DEFAULTS
// ============================================================

export default {
  TOURNAMENT_FOLDER,
  STRATEGY_CONFIG,
  TICKER_COLORS,
  ACTION_CONFIG,
  loadTickerSummary,
  loadTickerDetailed,
  loadCombinedSummary,
  getAvailableTickers,
  loadWorkflowManifest,
  loadWorkflowOutput,
  loadArenaPageData,
  loadPipelinePageData,
  loadActivityFeed,
  transformToStrategyCards,
  transformToChartData,
  transformToFeedItem,
  transformEvaluatorSummary
}