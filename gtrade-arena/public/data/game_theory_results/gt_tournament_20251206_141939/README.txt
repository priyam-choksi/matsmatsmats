
GAME THEORY TOURNAMENT RESULTS
==============================
Generated: 2025-12-06 14:19:39

OUTPUT STRUCTURE:
-----------------
by_ticker/          - Individual ticker results
  AAPL/
    AAPL_dashboard.png    - Main visual summary
    AAPL_readable.txt     - Human-readable round-by-round log
    AAPL_detailed.json    - Complete data for analysis
    AAPL_summary.json     - Summary metrics
    AAPL_trades.csv       - Trade data for import
    AAPL_decisions.png    - Position vs return analysis
    AAPL_race.gif         - Animated capital race

combined/           - Cross-ticker analysis
  results_table.txt       - Easy-to-read summary table
  results_table.csv       - For spreadsheet import
  all_tickers_summary.json - Complete combined data
  all_trades.csv          - All trades across tickers
  cross_ticker_dashboard.png - Visual comparison
  strategy_rankings.png   - Who won where

STRATEGIES:
-----------
  - Signal Follower: Follows LLM signals as portfolio manager - scaled positions
  - Cooperator: MOMENTUM: Rides the trend, cooperates with market direction
  - Defector: MEAN REVERSION: Fades extremes, defects from momentum
  - Tit-for-Tat: Smart adaptive: learns from winner's approach, not just position

BENCHMARK: Buy-and-Hold

CONFIGURATION:
--------------
  Total Capital: $1,000,000
  Reallocation Rate: 10%
  Initial Per Strategy: $250,000
