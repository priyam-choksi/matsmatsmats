/**
 * Data loader utilities for fetching JSON from public/data folder
 */

const BASE_PATH = '/data'

export async function loadTournamentData(ticker) {
  const response = await fetch(`${BASE_PATH}/tournament/${ticker}_detailed.json`)
  if (!response.ok) throw new Error(`Failed to load ${ticker} data`)
  return response.json()
}

export async function loadCombinedSummary() {
  const response = await fetch(`${BASE_PATH}/tournament/combined_summary.json`)
  if (!response.ok) throw new Error('Failed to load combined summary')
  return response.json()
}

export async function loadRecentDecisions() {
  const response = await fetch(`${BASE_PATH}/feed/recent_decisions.json`)
  if (!response.ok) throw new Error('Failed to load feed')
  return response.json()
}

export async function loadAgentsConfig() {
  const response = await fetch(`${BASE_PATH}/agents/agents_config.json`)
  if (!response.ok) throw new Error('Failed to load agents config')
  return response.json()
}

export async function loadStrategiesConfig() {
  const response = await fetch(`${BASE_PATH}/strategies/strategies_config.json`)
  if (!response.ok) throw new Error('Failed to load strategies config')
  return response.json()
}

export async function loadHoldings() {
  const response = await fetch(`${BASE_PATH}/strategies/holdings.json`)
  if (!response.ok) throw new Error('Failed to load holdings')
  return response.json()
}

// Load all data needed for Arena page
export async function loadArenaData() {
  const [summary, strategies, holdings, feed] = await Promise.all([
    loadCombinedSummary(),
    loadStrategiesConfig(),
    loadHoldings(),
    loadRecentDecisions()
  ])
  return { summary, strategies, holdings, feed }
}
