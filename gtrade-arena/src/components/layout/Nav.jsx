import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { useTheme } from '../../App'

// Icons as simple SVG components
const Icons = {
  Arena: ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20V10" /><path d="M18 20V4" /><path d="M6 20v-4" />
    </svg>
  ),
  Market: ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" /><path d="m19 9-5 5-4-4-3 3" />
    </svg>
  ),
  Pipeline: ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="18" cy="18" r="3" /><circle cx="6" cy="6" r="3" /><path d="M6 21V9a9 9 0 0 0 9 9" />
    </svg>
  ),
  // ADD THIS: History icon
  History: ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M12 7v5l4 2" />
    </svg>
  ),
  Agents: ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M22 21v-2a4 4 0 0 0-3-3.87" /><path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  ),
  Docs: ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /><path d="M16 13H8" /><path d="M16 17H8" /><path d="M10 9H8" />
    </svg>
  ),
  Sun: ({ size = 18 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4" /><path d="M12 2v2" /><path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" /><path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" />
    </svg>
  ),
  Moon: ({ size = 18 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  ),
  Settings: ({ size = 18 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" /><circle cx="12" cy="12" r="3" />
    </svg>
  )
}

const navItems = [
  { path: '/', icon: 'Arena', label: 'Arena' },
  { path: '/market', icon: 'Market', label: 'Market' },
  { path: '/pipeline', icon: 'Pipeline', label: 'Pipeline' },
  { path: '/history', icon: 'History', label: 'History' },  // <-- ADD THIS
  { path: '/agents', icon: 'Agents', label: 'Agents' },
  { path: '/docs', icon: 'Docs', label: 'Docs' },
]

export default function Nav() {
  const { theme, toggleTheme, isDark } = useTheme()
  const location = useLocation()

  return (
    <nav 
      className="w-20 flex flex-col items-center py-5 border-r"
      style={{ 
        backgroundColor: 'var(--bg-card)',
        borderColor: 'var(--border-primary)'
      }}
    >
      {/* Logo */}
      <div 
        className="w-11 h-11 rounded-xl flex items-center justify-center mb-8 cursor-pointer transition-transform hover:scale-105"
        style={{ 
          background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))'
        }}
      >
        <span className="text-white font-bold text-base">GT</span>
      </div>
      
      {/* Navigation Items */}
      <div className="flex flex-col gap-1 flex-1">
        {navItems.map(({ path, icon, label }) => {
          const Icon = Icons[icon]
          const isActive = location.pathname === path
          
          return (
            <NavLink
              key={path}
              to={path}
              className="w-14 h-14 rounded-xl flex flex-col items-center justify-center gap-1 transition-all duration-200 group"
              style={{
                backgroundColor: isActive ? (isDark ? 'rgba(99, 102, 241, 0.15)' : 'rgba(99, 102, 241, 0.1)') : 'transparent',
                color: isActive ? 'var(--accent-primary)' : 'var(--text-tertiary)'
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.backgroundColor = 'var(--bg-hover)'
                  e.currentTarget.style.color = 'var(--text-secondary)'
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.backgroundColor = 'transparent'
                  e.currentTarget.style.color = 'var(--text-tertiary)'
                }
              }}
            >
              <Icon size={20} />
              <span className="text-[10px] font-medium">{label}</span>
            </NavLink>
          )
        })}
      </div>

      {/* Bottom Section - Theme Toggle & Settings */}
      <div className="flex flex-col gap-2 mt-auto">
        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200"
          style={{
            backgroundColor: 'var(--bg-tertiary)',
            color: 'var(--text-secondary)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-hover)'
            e.currentTarget.style.color = 'var(--accent-primary)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'
            e.currentTarget.style.color = 'var(--text-secondary)'
          }}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? <Icons.Sun size={18} /> : <Icons.Moon size={18} />}
        </button>

        {/* Settings */}
        <button
          className="w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200"
          style={{
            backgroundColor: 'transparent',
            color: 'var(--text-tertiary)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-hover)'
            e.currentTarget.style.color = 'var(--text-secondary)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent'
            e.currentTarget.style.color = 'var(--text-tertiary)'
          }}
          title="Settings"
        >
          <Icons.Settings size={18} />
        </button>
      </div>

      {/* Status Indicator */}
      <div 
        className="mt-4 flex items-center gap-1.5 px-2 py-1 rounded-md"
        style={{ backgroundColor: 'var(--success-bg)' }}
      >
        <span 
          className="w-1.5 h-1.5 rounded-full animate-pulse"
          style={{ backgroundColor: 'var(--success)' }}
        />
        <span 
          className="text-[9px] font-semibold"
          style={{ color: 'var(--success)' }}
        >
          LIVE
        </span>
      </div>
    </nav>
  )
}