/**
 * Formatting utilities for numbers, dates, and display values
 */

// Format number as currency
export function formatCurrency(value, decimals = 0) {
  if (value === null || value === undefined) return '-'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value)
}

// Format number as percentage
export function formatPercent(value, decimals = 2) {
  if (value === null || value === undefined) return '-'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(decimals)}%`
}

// Format large numbers with K/M/B suffixes
export function formatCompact(value) {
  if (value === null || value === undefined) return '-'
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return value.toFixed(0)
}

// Format date string
export function formatDate(dateStr) {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}

// Format timestamp
export function formatTimestamp(timestamp) {
  if (!timestamp) return '-'
  const date = new Date(timestamp)
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Get color class for strategy
export function getStrategyColor(strategy) {
  const colors = {
    'Signal Follower': 'text-purple-400',
    'Cooperator': 'text-green-400',
    'Defector': 'text-red-400',
    'Tit-for-Tat': 'text-blue-400',
    'Buy-and-Hold': 'text-gray-400'
  }
  return colors[strategy] || 'text-gray-400'
}

// Get background color for regime
export function getRegimeColor(regime) {
  const colors = {
    'bull': 'bg-green-500/20 text-green-400',
    'bear': 'bg-red-500/20 text-red-400',
    'sideways': 'bg-yellow-500/20 text-yellow-400'
  }
  return colors[regime] || 'bg-gray-500/20 text-gray-400'
}

// Get color for positive/negative values
export function getValueColor(value) {
  if (value > 0) return 'text-green-400'
  if (value < 0) return 'text-red-400'
  return 'text-gray-400'
}
