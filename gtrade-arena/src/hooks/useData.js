/**
 * React hooks for data loading
 * Provides easy access to data layer with loading/error states
 */

import { useState, useEffect, useCallback } from 'react'
import {
  loadArenaPageData,
  loadPipelinePageData,
  loadActivityFeed,
  loadTickerSummary,
  loadTickerDetailed,
  loadCombinedSummary,
  STRATEGY_CONFIG,
  TICKER_COLORS,
  ACTION_CONFIG
} from '../data'

// ============================================================
// GENERIC DATA HOOK
// ============================================================

/**
 * Generic hook for loading any async data
 */
export function useAsyncData(loadFn, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const reload = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await loadFn()
      setData(result)
    } catch (err) {
      setError(err.message || 'Failed to load data')
      console.error('Data loading error:', err)
    } finally {
      setLoading(false)
    }
  }, [loadFn])

  useEffect(() => {
    reload()
  }, deps)

  return { data, loading, error, reload }
}

// Add to your existing useData.js

/**
 * Hook for Pipeline History page
 */
export function usePipelineHistory() {
  const [data, setData] = useState({ workflows: [], tickers: [], totalCount: 0 })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const result = await loadPipelineHistoryData()
        setData(result)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return { ...data, loading, error }
}


// ============================================================
// PAGE-SPECIFIC HOOKS
// ============================================================

/**
 * Hook for Arena page data
 */
export function useArenaData(ticker = 'AAPL') {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [currentTicker, setCurrentTicker] = useState(ticker)

  const loadData = useCallback(async (t) => {
    setLoading(true)
    setError(null)
    try {
      const result = await loadArenaPageData(t)
      if (result) {
        setData(result)
        setCurrentTicker(t)
      } else {
        throw new Error('No data returned')
      }
    } catch (err) {
      setError(err.message)
      console.error('Arena data error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData(ticker)
  }, [ticker, loadData])

  const changeTicker = useCallback((newTicker) => {
    loadData(newTicker)
  }, [loadData])

  return {
    data,
    loading,
    error,
    currentTicker,
    changeTicker,
    reload: () => loadData(currentTicker),
    // Convenience accessors
    strategies: data?.strategies || [],
    chartData: data?.chartData || [],
    tournament: data?.tournament || null
  }
}

/**
 * Hook for Pipeline/Market page data
 */
export function usePipelineData(ticker = 'AAPL') {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadData = useCallback(async (t) => {
    setLoading(true)
    setError(null)
    try {
      const result = await loadPipelinePageData(t)
      setData(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData(ticker)
  }, [ticker, loadData])

  return {
    data,
    loading,
    error,
    reload: () => loadData(ticker),
    // Convenience accessors
    decision: data?.decision || null,
    evaluators: data?.evaluators || null,
    synthesis: data?.synthesis || null,
    feedItem: data?.feedItem || null
  }
}

/**
 * Hook for activity feed data
 */
export function useActivityFeed(tickers = ['AAPL', 'NVDA', 'GOOGL']) {
  const [feed, setFeed] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await loadActivityFeed(tickers)
      setFeed(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [tickers.join(',')])

  useEffect(() => {
    loadData()
  }, [loadData])

  return { feed, loading, error, reload: loadData }
}

/**
 * Hook for tournament summary across all tickers
 */
export function useCombinedSummary() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const result = await loadCombinedSummary()
        setData(result)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  return { data, loading, error }
}

// ============================================================
// CONFIG HOOKS
// ============================================================

/**
 * Hook for strategy configuration
 */
export function useStrategyConfig(strategyName) {
  return STRATEGY_CONFIG[strategyName] || null
}

/**
 * Hook for all strategy configs
 */
export function useAllStrategyConfigs() {
  return STRATEGY_CONFIG
}

/**
 * Hook for ticker color
 */
export function useTickerColor(ticker) {
  return TICKER_COLORS[ticker] || '#6b7280'
}

/**
 * Hook for action config
 */
export function useActionConfig(action) {
  return ACTION_CONFIG[action] || ACTION_CONFIG['HOLD']
}

// ============================================================
// UTILITY HOOKS
// ============================================================

/**
 * Hook for data with auto-refresh
 */
export function useAutoRefresh(loadFn, intervalMs = 30000, deps = []) {
  const { data, loading, error, reload } = useAsyncData(loadFn, deps)

  useEffect(() => {
    const interval = setInterval(reload, intervalMs)
    return () => clearInterval(interval)
  }, [reload, intervalMs])

  return { data, loading, error, reload }
}

/**
 * Hook for comparing multiple tickers
 */
export function useTickerComparison(tickers = ['AAPL', 'NVDA']) {
  const [data, setData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function loadAll() {
      setLoading(true)
      const results = {}
      
      for (const ticker of tickers) {
        try {
          results[ticker] = await loadTickerSummary(ticker)
        } catch (err) {
          console.warn(`Failed to load ${ticker}`)
        }
      }
      
      setData(results)
      setLoading(false)
    }
    loadAll()
  }, [tickers.join(',')])

  return { data, loading, error }
}

// ============================================================
// EXPORTS
// ============================================================

export default {
  useAsyncData,
  useArenaData,
  usePipelineData,
  useActivityFeed,
  useCombinedSummary,
  useStrategyConfig,
  useAllStrategyConfigs,
  useTickerColor,
  useActionConfig,
  useAutoRefresh,
  useTickerComparison
}