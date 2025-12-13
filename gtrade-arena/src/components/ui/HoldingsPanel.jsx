import React from 'react'
import { useDataLoader } from '../../hooks/useDataLoader'
import FeedItem from '../cards/FeedItem'

export default function HoldingsPanel() {
  const { data, loading, error } = useDataLoader('feed/recent_decisions.json')

  if (loading) return <div className="p-4 text-gray-500">Loading...</div>
  if (error) return <div className="p-4 text-red-400">Error loading feed</div>

  return (
    <div className="p-4 space-y-3 overflow-y-auto">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Recent Decisions</h3>
      {data?.activities?.map((item, i) => (
        <FeedItem key={i} item={item} />
      ))}
    </div>
  )
}
