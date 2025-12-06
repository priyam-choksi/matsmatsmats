
GAME THEORY TOURNAMENT RESULTS
==============================
Generated: 2025-12-03 01:29:47

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
  - Signal Follower: Follows LLM agent signals - tests if AI analysis adds value
  - Cooperator: Follows the pack - takes similar positions to others
  - Defector: Goes against the pack - contrarian positions
  - Tit-for-Tat: Copies what worked last round - adaptive learner

BENCHMARK: Buy-and-Hold

CONFIGURATION:
--------------
  Total Capital: $1,000,000
  Reallocation Rate: 10%
  Initial Per Strategy: $250,000
