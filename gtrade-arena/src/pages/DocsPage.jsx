import React, { useState } from 'react'
import { useTheme } from '../App'
import ReactMarkdown from 'react-markdown'

// ========== FAQ DATA ==========
const faqCategories = [
  // OVERVIEW CATEGORY - Paste this into the faqCategories array

  {
    id: 'overview',
    name: 'Overview',
    icon: '🏠',
    faqs: [
      {
        q: 'What is Trade Arena?',
        a: `Trade Arena is a capstone project for DAMG 7374 at Northeastern University, built by Priyam Choksi and Vishodhan Krishnan during Fall 2025.

The project answers one question: do different trading strategies perform optimally under different market regimes? We test this by building three interconnected systems:

### 1. Analysis Pipeline (11 LLM agents, 5 phases)

The pipeline processes market data through specialized agents:

- **Phase 1** - Runs 4 analysts in parallel: Technical (RSI, MACD, Bollinger Bands, support/resistance), News (Yahoo Finance, Finnhub, NewsAPI sentiment), Fundamental (P/E, margins, peer comparison), and Macro (Fed policy, VIX, sector rotation)
- **Phase 2** - Bull and Bear researchers build opposing theses from Phase 1 data
- **Phase 3** - Research Manager facilitates debate rounds and synthesizes into probability-weighted recommendations
- **Phase 4** - Runs 3 risk evaluators with different personalities (Aggressive defaults to STRONG BUY, Neutral to HOLD, Conservative to AVOID)
- **Phase 5** - Risk Manager makes the final BUY/HOLD/REJECT decision with exact position sizing

### 2. Strategy Arena (Game Theory Tournament)

Four strategies compete for a shared $1M capital pool over 90 rounds per ticker:

- **Signal Follower** - Uses the 11-agent pipeline output
- **Cooperator** - Momentum-based, blends 70% group average + 30% LLM signal
- **Defector** - Mean reversion, bets against recent trends
- **Tit-for-Tat** - Copies whatever strategy won last round

After each round, capital reallocates via z-score: winners take from losers. This creates genuine strategic interdependence - your returns depend on what others do.

### 3. LLM Arena (Model Comparison)

7 language models compete on identical trading tasks: GPT-4o-mini, Llama-3.3-70B, Llama-4-Maverick, Kimi-K2, Qwen3-32B, GPT-OSS-120B, Allam-2-7B

Each starts with $250K, receives same data/prompts. Tests which model architectures generate better signals.

**Scale:** 20 tickers, 1,800 tournament rounds, 2,695 LLM arena rounds, 35,000+ workflow JSON outputs.`
      },
      {
        q: 'What were the actual results?',
        a: `## Tournament Results (across 1,800 rounds, 20 tickers)

### Signal Follower (LLM-based)

- Total Return: +14.93%
- Signal Accuracy: 73.2% (directional calls correct vs 3-day forward returns)
- Best in: Bull markets where systematic analysis captures momentum
- Average Position: ~15% (conservative due to evaluator consensus)

### Cooperator (Momentum)

- Total Return: +10.34%
- Win Rate: 28/90 average
- Best in: Trending markets where following the crowd works
- Behavior: Warmup period (3 rounds) then 70% group + 30% LLM blend

### Defector (Mean Reversion)

- Total Return: -4.24% overall, but +6.2% in bear markets
- Win Rate: 7/90 average (wins few rounds but big when it wins)
- Best in: Reversals and corrections
- Behavior: Monitors rolling 3-day returns, bets contrarian when |cumulative| > 1.5%

### Tit-for-Tat (Adaptive)

- Total Return: -12.64%
- Win Rate: 18/90 average
- Best in: Regime transitions (quickly adapts to new winner)
- Behavior: 50-90% copy weight based on winner's streak length

---

**Benchmark:** Buy-and-Hold tracked separately with +12.45% average, meaning Signal Follower generated +2.48% alpha.

**Cooperation Rate:** 78% of rounds saw strategies taking similar positions. The 22% "defection" rounds often determined who gained/lost significant capital.

**Key Finding:** No single strategy dominated all conditions. Signal Follower excelled in 44% of rounds (bull markets), Defector in 2% (bear markets), Cooperator and TFT split the rest. Market regime detection became the most valuable signal.`
      },
      {
        q: 'What are the 20 tickers analyzed?',
        a: `The tournament covers 20 tickers across 6 sectors:

### Technology (7 tickers)

- **AAPL** - Apple Inc (largest by market cap, consumer tech)
- **AMZN** - Amazon.com (e-commerce + AWS cloud)
- **GOOGL** - Alphabet Inc (advertising, search, cloud)
- **META** - Meta Platforms (social media, VR/metaverse)
- **MSFT** - Microsoft Corp (enterprise software, cloud)
- **NVDA** - NVIDIA Corp (AI chips, data center GPUs)
- **TSLA** - Tesla Inc (EVs, energy, high volatility)

### Financials (3 tickers)

- **GS** - Goldman Sachs (investment banking, trading)
- **JPM** - JPMorgan Chase (commercial banking, largest US bank)
- **V** - Visa Inc (payments processing, high margins)

### Healthcare (3 tickers)

- **JNJ** - Johnson & Johnson (diversified pharma, consumer health)
- **LLY** - Eli Lilly (pharma, GLP-1 weight loss drugs driving growth)
- **UNH** - UnitedHealth Group (health insurance, Optum services)

### Consumer (3 tickers)

- **KO** - Coca-Cola (beverages, dividend aristocrat)
- **PG** - Procter & Gamble (consumer staples, defensive)
- **WMT** - Walmart Inc (retail, e-commerce growth)

### Energy (2 tickers)

- **CVX** - Chevron Corp (integrated oil, dividend focus)
- **XOM** - ExxonMobil (integrated oil, largest by revenue)

### ETFs (2 tickers)

- **SPY** - S&P 500 ETF (broad market benchmark)
- **QQQ** - Nasdaq 100 ETF (tech-heavy benchmark)

---

**Selection criteria:** High liquidity (options market depth), sector representation, analyst coverage, meaningful price history, and volatility range (TSLA high, KO low).`
      },
      {
        q: 'What data sources power the system?',
        a: `### Market Data

- **Yahoo Finance** (\`yfinance\` library) - Primary source for OHLCV price data. Downloads 90 days history per ticker. Provides \`ticker.info\` for fundamentals, \`ticker.news\` for headlines.
- **Alpaca API** - Supplementary historical data, used for validation.

### News & Sentiment

- **Finnhub API** (\`/company-news\` endpoint) - Financial news with pre-labeled sentiment. Requires \`FINNHUB_KEY\`. Best for earnings, M&A, analyst actions.
- **NewsAPI** (\`/everything\` endpoint) - Broader headline coverage. Searches ticker + company name. Requires \`NEWSAPI_KEY\`.
- **Reddit/PRAW** - Social sentiment from r/stocks, r/investing, r/wallstreetbets. Captures retail investor sentiment. Requires \`REDDIT_CLIENT_ID\`, \`REDDIT_CLIENT_SECRET\`.

### Fundamental Data

- **Finnhub API** (\`/stock/metric\`) - EV/EBITDA, PEG ratio, additional metrics.
- **yfinance** \`ticker.financials\` - Income statements, balance sheets, cash flow.
- **SEC EDGAR** - 10-K, 10-Q filings for deep fundamental analysis.

### Macro Data (all via yfinance)

- **Indices:** \`^GSPC\` (S&P 500), \`^DJI\` (Dow), \`^IXIC\` (Nasdaq), \`^RUT\` (Russell 2000)
- **Volatility:** \`^VIX\` (fear gauge)
- **Rates:** \`^TNX\` (10-year Treasury yield)
- **Sectors:** XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC (11 sector ETFs)
- **Economic proxies:** DX-Y.NYB (Dollar), GC=F (Gold), CL=F (Crude), BTC-USD

### Data Freshness

- **Historical mode** - Queries end on \`analysis_date\`, simulating what you'd know then
- **Live mode** - Real-time prices every 2 seconds, news every 15 minutes, fundamentals quarterly`
      },
      {
        q: 'Is this financial advice?',
        a: `**No.** This is purely academic research for educational purposes.

Trade Arena should never be used for actual trading decisions. The simulated returns are based on historical backtesting with significant limitations:

### What's NOT modeled

- Transaction costs (commissions, spreads)
- Slippage (price impact of trades)
- Market impact (moving the market with large orders)
- Liquidity constraints (can't always trade at quoted prices)
- Short-selling costs and restrictions
- Margin requirements
- Tax implications

### Backtest limitations

- Lookahead bias risk (we tried to avoid, but historical data is imperfect)
- Survivorship bias (we only analyze tickers that exist today)
- Regime-specific results (2024-2025 was mostly bullish)
- LLM behavior may change with model updates

### What this IS

- Academic exploration of multi-agent LLM systems
- Game theory validation of strategy interdependence
- Comparison of LLM architectures on structured tasks
- Portfolio construction research

---

The +14.93% Signal Follower return and 73.2% accuracy are research metrics, not investment promises. Past performance does not indicate future results. Always consult a qualified financial advisor.`
      }
    ]
  }

  ,

  // ============================================================
  // PIPELINE CATEGORY - Paste into faqCategories array
  // ============================================================

  {
    id: 'pipeline',
    name: 'Analysis Pipeline',
    icon: '🔄',
    faqs: [
      {
        q: 'How does Phase 1 (Market Analysis) work?',
        a: `Phase 1 runs 4 analyst agents in parallel via subprocess from master_orchestrator.py, taking roughly 60 seconds total. Each agent fetches its own data and produces structured JSON output.

---

### Technical Analyst (technical_agent.py)

Fetches 90 days of OHLCV from yfinance ending on the analysis_date. The calculate_indicators() method computes RSI(14), MACD (12/26/9 periods), Bollinger Bands (20-period SMA +/- 2 std), moving averages (SMA_5, EMA_10, SMA_20, SMA_50), ATR(14), and Stochastic(14,3). Support/resistance uses identify_support_resistance() with a 20-day lookback, finding rolling highs/lows.

**Output format includes:**
- "RSI: 72.3 - OVERBOUGHT"
- "MACD: 2.34 | Signal: 1.89 | Histogram: +0.45 - Bullish momentum"
- Specific support ($140.00) and resistance ($155.00) levels with entry/stop/target price targets

---

### News Analyst (news_agent.py)

Aggregates from ticker.news (Yahoo Finance), Finnhub's /company-news endpoint, and optionally Reddit via PRAW. Articles are filtered to the --days window (default 7). The full article list is sent to the LLM for sentiment scoring (-1 to +1), with output including key headlines, impact assessments, and a catalyst timeline.

---

### Fundamental Analyst (fundamental_agent.py)

Pulls from yfinance ticker.info: P/E, P/B, P/S ratios, market cap, revenue, earnings, and margins. Calculates peer comparison against sector averages.

**Output:** Valuation assessment (Undervalued/Fair/Overvalued), growth trajectory, financial health score.

---

### Macro Analyst (macro_agent.py)

Fetches market indices (SPY, DIA, QQQ), VIX, 11 sector ETFs, Dollar index, Gold, Oil, and Treasury yields. Determines regime as:

- **RISK-ON** - VIX < 18, tech leading
- **RISK-OFF** - VIX > 25, defensives leading
- **NEUTRAL** - Mixed signals

**Output:** "Market regime: RISK-ON (VIX at 14.2, tech leading +2.3% WoW)"

---

All 4 outputs merge into discussion_points.json via discussion_hub.py, which runs an LLM synthesis pass to extract key signals, identify conflicts, and set research priorities.`
      },
      {
        q: 'How does Phase 2 (Bull/Bear Research) work?',
        a: `Phase 2 runs 2 researcher agents in parallel (~40 seconds total), each receiving the same discussion_points.json from Phase 1.

---

### Bull Researcher (bull_researcher.py)

Builds the comprehensive bullish investment thesis. It reads the Phase 1 synthesis, extracts all positive signals (technical breakouts, positive sentiment, strong fundamentals, favorable macro), and constructs arguments around growth catalysts, upside price targets, and competitive advantages.

**Output includes:**
- Thesis summary
- Upside target percentage
- Catalyst timeline (what events could drive the stock higher)
- Confidence level (HIGH/MEDIUM/LOW)
- Explicit counter-arguments to anticipated bear points

**Example output:** "Strong AI tailwinds driving accelerated growth. Upside target: +28%. Catalysts: AI product launch Q1, data center expansion. Confidence: HIGH."

---

### Bear Researcher (bear_researcher.py)

Constructs the bearish case with the same rigor. It extracts all risk signals (overbought technicals, negative sentiment, valuation concerns, macro headwinds), and builds arguments around downside scenarios, key risks, and competitive threats.

**Output includes:**
- Bear thesis
- Downside risk percentage
- Risk factors list
- Confidence level
- Counter-arguments to bull points

**Example:** "Valuation stretched at 35x forward earnings, competition intensifying from peers. Downside risk: -18%. Key risks: Multiple compression, margin pressure, regulatory overhang. Confidence: MEDIUM."

---

Both researchers produce validated JSON with guardrails checking for logical consistency (e.g., bullish thesis can't have negative target). Validation scores (0-100) track data quality and argument coherence.`
      },
      {
        q: 'How does Phase 3 (Debate & Synthesis) work?',
        a: `Phase 3 runs the Research Manager (research_manager.py) which orchestrates a structured debate between bull and bear theses, then synthesizes the final recommendation (~30 seconds per debate round).

---

### Debate Format

The debate follows a strict format defined in research_manager.py. Each side gets 300-500 words per round to:

- Directly address the opponent's specific points
- Use data/evidence from their own thesis only (no fabricating new data)
- Be persuasive but factual
- End with their strongest argument (bull) or key risk concern (bear)

---

### Evaluation Rules

The manager evaluates argument strength using these consistency rules:
- strong_buy requires bull majority
- strong_sell requires bear majority
- High confidence requires good data quality

After debate (configurable 1-3 rounds via --debate-rounds flag), the manager assigns probability weights:

- Bull case probability (e.g., 62%)
- Bear case probability (e.g., 28%)
- Base case probability (e.g., 10%)

---

### Output

Output includes:
- debate_winner (BULL or BEAR)
- The probability distribution
- A synthesized recommendation with rationale

The synthesis explicitly states which arguments were most convincing and which evidence was strongest. This becomes the foundation for Phase 4 risk evaluation.

The validation_rules dict enforces: position bounds (0-20% max), valid recommendations (STRONG BUY through STRONG SELL), valid confidence levels, and required fields in the conclusion.`
      },
      {
        q: 'How does Phase 4 (Risk Evaluation) work?',
        a: `Phase 4 runs 3 evaluator agents in parallel (~30 seconds total), each with a distinct risk personality that colors their interpretation of the Phase 3 synthesis.

---

### Aggressive Evaluator (aggressive_evaluator.py)

Evaluates from a risk-seeking perspective. Default stance: STRONG BUY. It focuses on upside potential, recommends larger position sizes (50-80% range), and is willing to take concentrated positions on high-conviction ideas.

**Example output:** "STRONG BUY @ 75%. High conviction on growth catalysts, willing to take concentrated position."

---

### Neutral Evaluator (neutral_evaluator.py)

Provides balanced assessment. Default stance: HOLD. It weighs both upside and downside equally, recommends moderate positions (25-45% range), and adjusts based on evidence strength.

**Example:** "BUY @ 40%. Balanced view - upside potential outweighs risks but position sized for volatility."

---

### Conservative Evaluator (conservative_evaluator.py)

Prioritizes capital preservation. Default stance: AVOID. It focuses on downside protection, recommends smaller positions (5-20% range), and flags red_flags list of concerns.

**Example:** "HOLD @ 15%. Prefer to wait for pullback, current valuation leaves little margin of safety."

---

### Output Format

Each evaluator outputs:
- stance (STRONG BUY/BUY/HOLD/AVOID)
- position_pct (integer 0-100)
- reasoning (paragraph explaining logic)
- confidence

**Guardrails enforce consistency:**
- AVOID stance must have 0% position
- BUY stance must have minimum 5%
- Risk/reward ratio below threshold downgrades STRONG BUY to BUY

The three evaluations provide the spectrum of perspectives for the final Risk Manager decision.`
      },
      {
        q: 'How does Phase 5 (Final Decision) work?',
        a: `Phase 5 runs the Risk Manager (risk_manager.py) which makes the final BUY/HOLD/REJECT decision with exact dollar position sizing (~10 seconds).

---

### Consensus Analysis

The Risk Manager first runs analyze_evaluator_consensus() which:

- Tallies stances across all 3 evaluators (how many BUY vs AVOID)
- Calculates average recommended position size
- Identifies agreements ("All evaluators favor buying") and disagreements ("Mixed views: 2 buy, 1 avoid")
- Extracts red_flags from the conservative evaluator

---

### 2-LLM Decision Process

**LLM #1 (Position Calibrator)** takes the base range from evaluator consensus and adjusts based on conviction alignment. Inputs: Research recommendation, evaluator positions, agent votes (how many of the 11 agents were bullish/bearish/neutral). It can widen or narrow the position range.

**LLM #2 (Final Decision)** receives the calibrated range plus all context and outputs:

- verdict: BUY, HOLD, or REJECT
- position_dollars: Exact amount to invest (e.g., "$12,450" from a $100K portfolio)
- confidence: HIGH/MEDIUM/LOW
- reasoning: Paragraph explaining the decision

---

### Risk Controls

Risk controls are set via set_risk_controls():
- stop_loss percentage (typically -8%)
- profit_targets array (e.g., [+12%, +18%, +25%])

---

The final output JSON includes all fields plus metadata: ticker, analysis_date, total pipeline duration, and references to all intermediate outputs. This becomes one row in the 35,000+ workflow outputs generated across the full tournament.`
      },
      {
        q: 'How do I run the pipeline?',
        a: `The pipeline is triggered through master_orchestrator.py with various flags:

---

### Basic Run (live mode, today's date)

python master_orchestrator.py AAPL

---

### Historical Mode (specific date)

python master_orchestrator.py AAPL --analysis-date 2024-06-17

This is critical for the tournament - each of the 90 rounds uses a different historical date to test strategy performance across real market conditions.

---

### Control Debate Depth

- --research-mode shallow (fast, 1 debate round)
- --research-mode deep (thorough, 3 debate rounds)
- --debate-rounds 2 (explicit override)

---

### Output Control

- --output-dir ./my_outputs (custom output location)
- --format json (default, structured output)

---

### Output Files

The orchestrator creates timestamped directories under outputs/ containing:

- **discussion_points.json** - Phase 1
- **bull_thesis.json, bear_thesis.json** - Phase 2
- **research_synthesis.json, debate_transcript.txt** - Phase 3
- **aggressive_eval.json, neutral_eval.json, conservative_eval.json** - Phase 4
- **final_decision.json** - Phase 5
- **execution_log.json** - Timing and status for each phase

---

### Execution Times

- **Shallow mode:** ~2.5 minutes
- **Deep mode:** ~4-5 minutes per ticker

The tournament runs batch_collect_all.py which loops through all 20 tickers x 90 rounds = 1,800 pipeline executions.`
      }
    ]
  },

  // ============================================================
  // STRATEGY ARENA CATEGORY
  // ============================================================

  {
    id: 'strategy-arena',
    name: 'Strategy Arena',
    icon: '⚔️',
    faqs: [
      {
        q: 'What is the Strategy Arena?',
        a: `The Strategy Arena is a game-theoretic tournament where 4 trading strategies compete for a shared $1M capital pool across 90 rounds per ticker. This isn't parallel backtesting—it's genuine strategic interdependence where each strategy's capital depends on how it performs relative to the others.

The core innovation: capital flows from losers to winners via z-score reallocation each round. If Signal Follower outperforms and Defector underperforms, capital transfers from Defector to Signal Follower. Over 90 rounds, the best-performing strategies accumulate capital while poor performers get squeezed.

The 4 competing strategies are:
- Signal Follower: Pure LLM signal execution (trusts pipeline output directly)
- Cooperator: Momentum strategy (rides trends, follows recent price direction)
- Defector: Mean reversion strategy (fades extremes, contrarian positioning)
- Tit-for-Tat: Adaptive strategy (copies approach of last round's winner)

Buy-and-Hold serves as external benchmark at 100% position always—it doesn't compete for capital but provides the baseline: can active strategies beat passive?

The research question: "Do different strategies perform optimally under different market regimes?" After 1,800 tournament rounds (20 tickers × 90 rounds), we have the data to answer this definitively.`
      },
      {
        q: 'How does Signal Follower work?',
        a: `Signal Follower (signal_follower.py) is the pure LLM strategy—it directly executes whatever the 11-agent pipeline recommends, ignoring what other strategies are doing.

Position logic from the code:
1. Read aggressive_position, neutral_position, conservative_position from MarketContext (these are the Phase 4 evaluator outputs, scaled to 0.0-1.0)
2. If any signal > BULLISH_THRESHOLD (0.12): follow the strongest signal
3. Otherwise: follow the average of all three
4. Apply consensus multiplier (0.7-1.3x) based on evaluator agreement
5. Scale up by SCALE_FACTOR (3.5x) to convert small percentages into meaningful positions
6. Clamp to MIN_POSITION (15%) and MAX_POSITION (90%)

Example calculation: If aggressive=0.15, neutral=0.10, conservative=0.05:
- Max signal (0.15) exceeds threshold → use max
- Base position = 0.15 × 100 × 3.5 = 52.5%
- Consensus (medium agreement) → mult = 0.85
- Final = 52.5% × 0.85 = 44.6%

This strategy wins when: LLM signals are correct. It struggles when: LLM is wrong, or when market doesn't move as expected.

The strategy achieved 73.2% signal accuracy and +14.93% return across all tournaments, but showed high variance—it's right often but can have large drawdowns when wrong.`
      },
      {
        q: 'How does Cooperator (Momentum) work?',
        a: `Cooperator (cooperator.py) implements momentum trading logic with a game theory wrapper. The name "Cooperator" comes from game theory (cooperates with market direction), but the actual implementation is pure momentum strategy.

Position logic from decide_position():
1. Look back LOOKBACK (3) rounds of market returns from game.rounds
2. Calculate average return over that window
3. If avg_return > STRONG_UP (1.5%): Very bullish position (75-90%)
4. If avg_return > MILD_UP (0.5%): Moderately bullish (55-70%)
5. If avg_return > MILD_DOWN (-0.5%): Neutral (40-55%)
6. If avg_return > STRONG_DOWN (-1.5%): Moderately bearish (25-40%)
7. Else: Very bearish (15-25%)
8. Clamp between MIN_POSITION (15%) and MAX_POSITION (90%)

First few rounds (before enough history): Falls back to LLM signals scaled by SCALE_FACTOR (3.5x).

Momentum logic: "What's been going up keeps going up." This wins in trending markets where price moves continue. It loses at reversals (slow to turn) and in choppy mean-reverting markets.

Example: If last 3 rounds had returns of +1.2%, +0.8%, +0.5%, average = +0.83%. This exceeds MILD_UP threshold → position around 60% (bullish).

From results: Cooperator returned +10.9% but won only 3% of rounds—it makes steady gains but rarely beats the others decisively.`
      },
      {
        q: 'How does Defector (Mean Reversion) work?',
        a: `Defector (defector.py) implements mean reversion trading with a contrarian approach. The "Defector" name means it defects from the crowd when markets are extreme.

Position logic from decide_position():
1. Get last round's market return from game.rounds[-1].market_return
2. Also check 2-day cumulative return (LOOKBACK=2) for stronger signals
3. If return > BIG_UP (2%): Market overbought → reduce position to 25-35%
4. If return > MODERATE_UP (1%): Mildly overbought → position 35-45%
5. If return > MODERATE_DOWN (-1%): Neutral range → follow LLM signals
6. If return > BIG_DOWN (-2%): Oversold → expect bounce → position 60-70%
7. Else (very big drop): Strong bounce expected → position 70-85%
8. Clamp between MIN_POSITION (15%) and MAX_POSITION (90%)

Cumulative extreme check: If 2-day cumulative move exceeds CUMULATIVE_EXTREME (3%), take stronger contrarian position.

Mean reversion logic: "What goes up must come down. Extremes revert to mean." This wins at turning points and in range-bound markets. It loses in trending markets (fighting the trend).

Example: If yesterday's return was -2.5% (big down), Defector goes bullish (70%+), expecting a bounce. If the bounce happens, it profits. If the drop continues, it loses.

From results: Defector had highest win rate (60%) and +17.9% return, but this includes AAPL's strong period where contrarian dip-buying worked well. It accumulated capital because it won the most individual rounds.`
      },
      {
        q: 'How does Tit-for-Tat (Adaptive) work?',
        a: `Tit-for-Tat (tit_for_tat.py) is the classic game theory strategy adapted for trading—it learns from the last winner and copies their approach (not just their position).

Position logic from decide_position():

Round 1 (start "nice"): Follow LLM signals, position around 50-60%

Later rounds: Analyze last winner and WHY they won:
- If Cooperator (momentum) won: Market was trending → copy bullish/bearish momentum approach
  - If last_return > 0: Be bullish (70-80%)
  - If last_return < 0: Be bearish (30-40%)

- If Defector (mean reversion) won: Market reversed → copy contrarian approach
  - Position for expected normalization (60-70%)
  - If market down, expect bounce

- If Signal Follower won: LLM signals were correct → trust pipeline
  - Get LLM position and boost confidence (+10% if bullish)

Winner streak bonus: BASE_COPY_WEIGHT (70%) + STREAK_BONUS (5%) per consecutive win. If Cooperator won 3 rounds in a row, copy weight = 70% + 15% = 85%.

Logic: "Do what worked. But understand WHY it worked." This wins when market regime is consistent (winners repeat). It loses when winners alternate randomly (always one step behind).

From results: Tit-for-Tat achieved +13.0% return and 7% win rate—solidly middle-of-pack. It never dominates but rarely catastrophically fails.`
      },
      {
        q: 'How does z-score reallocation work?',
        a: `Z-score reallocation (in gt_engine.py) is how capital flows between strategies after each round. It's what makes this a game theory tournament rather than parallel backtesting.

After each round:
1. Calculate each strategy's return based on: position_size × market_move
   - If Signal Follower had 60% position and market moved +1.5%, return = 0.6 × 1.5% = +0.9%

2. Convert returns to z-scores (standardized scores):
   - z = (return - mean_return) / std_dev(returns)
   - This normalizes across different market conditions

3. Calculate capital transfers:
   - transfer = REALLOCATION_RATE (10%) × pool × z-score
   - Positive z-score → receive capital
   - Negative z-score → lose capital

Example round:
- Signal Follower: +0.9% return, z = +1.2 → gains $120K
- Cooperator: +0.6% return, z = +0.3 → gains $30K
- Defector: -0.3% return, z = -0.8 → loses $80K
- Tit-for-Tat: -0.2% return, z = -0.7 → loses $70K
(Transfers sum to $0—zero-sum game)

Why z-scores instead of raw returns? A +2% return in a +1% market (outperformance) is different from +2% in a +3% market (underperformance). Z-scores capture relative performance.

Reallocation rate (10%) is tunable: higher = faster concentration toward winners, lower = more stable allocations. 10% creates meaningful differentiation without making early rounds deterministic.

Over 90 rounds: Winners compound gains while losers get squeezed. Starting capital ($250K each) diverges dramatically—the AAPL tournament ended with Defector at ~$800K and Signal Follower near $0.`
      },
      {
        q: 'What metrics are tracked?',
        a: `The tournament tracks comprehensive metrics at multiple levels—per round, per ticker, and aggregate.

Primary performance metrics:
- Total Return: Cumulative profit/loss as percentage of starting capital
- Win Rate: Percentage of rounds won by each strategy (who had highest return that round)
- Signal Accuracy: For Signal Follower—were directional calls correct against 3-day forward returns? (73.2% achieved)
- Capital Progression: How strategy's capital changed over 90 rounds

Risk metrics:
- Sharpe Ratio: Return per unit of volatility (higher = better risk-adjusted returns)
- Max Drawdown: Largest peak-to-trough decline during tournament
- Volatility: Standard deviation of round-by-round returns

Game theory metrics:
- Cooperation Rate: How often strategies took similar positions (~78% observed)
- Gini Coefficient: Capital inequality (0 = equal, 1 = one winner takes all)
- Winner Concentration: Did one strategy dominate or was it competitive?

Regime-specific analysis breaks performance by detected market regime:
- Bull (>5% over period): Signal Follower and Cooperator excel
- Bear (<-5% over period): Defector's mean reversion captures bounces
- Sideways (-5% to +5%): Tit-for-Tat's adaptiveness helps

Output volume: 35,000+ workflow JSON files across all tournaments, 1,800 tournament rounds (20 tickers × 90 rounds), full trade-level data exportable to CSV for external analysis. Each JSON includes complete agent reasoning, timestamps, confidence scores, and intermediate calculations.`
      }
    ]
  },

  // ============================================================
  // LLM ARENA CATEGORY - Paste into faqCategories array
  // ============================================================

  {
    id: 'llm-arena',
    name: 'LLM Arena',
    icon: '🏟️',
    faqs: [
      {
        q: 'What is the LLM Arena?',
        a: `The LLM Arena (llm_arena.py) is a controlled experiment comparing 7 different language models on identical trading tasks. Each model starts with $250,000 (STARTING_CAPITAL constant). All models receive the same market data, news, and trading prompts. The only variable is the model itself.

---

### The 7 Competing Models

- **GPT-4o-mini (OpenAI)** - provider "openai", model "gpt-4o-mini"
- **Llama-3.3-70B (Meta)** - provider "groq", model "llama-3.3-70b-versatile"
- **Llama-4-Maverick (Meta)** - provider "groq", model "meta-llama/llama-4-maverick-17b-128e-instruct"
- **Kimi-K2 (Moonshot AI)** - provider "groq", model "moonshotai/kimi-k2-instruct-0905"
- **Qwen3-32B (Alibaba)** - provider "groq", model "qwen/qwen3-32b"
- **GPT-OSS-120B** - provider "groq", model "openai/gpt-oss-120b"
- **Allam-2-7B** - provider "groq", model "allam-2-7b" (Arabic-focused but multilingual)

---

### How It Works

Over 2,695 rounds, models receive market context and must decide: BUY, SELL, or HOLD for each of the 20 tickers. Each round they submit trades with dollar amounts and reasoning. Portfolio values track cumulative performance.

**Research question:** "Given identical inputs and pipeline structure, which model architectures generate better trading signals?"`
      },
      {
        q: 'How does the Arena execution work?',
        a: `The Arena runs via the TradingArena class in llm_arena.py with sophisticated checkpoint/resume functionality for crash recovery.

---

### Data Sources

Data comes in two types (determined by get_all_rounds()):

- **Rounds 1-90:** "Rich" workflow data from outputs/workflows/ - includes full 6-phase pipeline outputs with debates
- **Rounds 91-473:** "Lite" market data from outputs/market_data/ - technicals, fundamentals, news only

---

### Each Round Executes

1. Load market data for that date (prices dict with all 20 tickers)
2. Build prompt with portfolio state, market context, and available cash
3. Call each model sequentially with PROVIDER_DELAYS (OpenAI: 0.5s, Groq: 2s between requests)
4. Parse JSON response for trades array with ticker, action, amount_usd, reasoning
5. Execute trades via portfolio.execute_buy() or portfolio.execute_sell()
6. Record equity via portfolio.record_equity(date)
7. Save checkpoint every 5 rounds (TIMING["checkpoint_every"])

---

### API Configuration

- request_timeout: 90 seconds
- max_retries: 3 per failed call
- temperature: 0.7
- max_tokens: 2500

If any model fails after retries, the Arena raises ModelFailedError and stops cleanly - we need ALL models to participate for fair comparison. No partial rounds are saved.`
      },
      {
        q: 'What makes models perform differently?',
        a: `Even with identical prompts and inputs, models exhibit distinct "trading personalities" that emerge from their architecture and training.

---

### Reasoning Style

Some models (Llama-3.3-70B, Kimi-K2) provide thorough multi-factor analysis weighing technical, fundamental, and sentiment signals. Others (GPT-4o-mini) make quicker decisions focusing on key signals. The model_reasoning field in each Trade captures this verbatim.

---

### Risk Calibration

- **Llama models** tend toward aggressive positioning - larger position sizes on conviction
- **Allam-2-7B** leans conservative, rarely taking positions above 20% of portfolio
- **GPT-4o-mini** shows balanced risk-taking

This shows up in average position sizes and max drawdowns.

---

### Regime Adaptation

Reveals model robustness:

- **Aggressive models (Llama-4-Maverick)** excel during trending markets with sustained moves
- **Conservative models (Allam)** preserve capital better during volatility spikes
- Some models naturally adapt style based on detected market regime (risk-on vs risk-off signals), others maintain consistent behavior

---

### Edge Case Handling

How does the model handle:

- Conflicting signals (technicals bullish, fundamentals bearish)?
- High volatility periods?
- Prices near key support/resistance?
- Earnings announcements?

The self.reasoning dict stores per-round reasoning for post-hoc analysis of these scenarios.

---

### Portfolio Tracking

The portfolio dataclass tracks: cash, positions dict (ticker to Position with shares, avg_cost, current_price), trades list (full history), and equity_curve for charting.`
      },
      {
        q: 'How is data structured for the frontend?',
        a: `The preprocess_llm_arena.py script transforms raw arena output into optimized files for React frontend loading.

---

### Input Files

- **arena_results.json** - Final portfolio states, equity curves, round-by-round results
- **trade_history.json** - All trades with timestamps, amounts, reasoning

---

### Output Structure

**llm_arena_summary.json (~150KB)** - Main file loaded on page mount:

- Leaderboard with all 7 models (sorted by portfolio value)
- Equity curves (one point per date for each model)
- Top 20 tickers by activity
- Daily activity summary
- Recent 100 trades (truncated reasoning to 150 chars)

**model_trades/ directory** - One file per model for lazy loading:

- GPT_4o_mini.json
- Llama_3_3_70B.json
- Llama_4_Maverick.json
- Kimi_K2.json
- Qwen3_32B.json
- GPT_OSS_120B.json
- Allam_2_7B.json

Each model file contains: full trade history with complete reasoning, position history, daily equity values, performance metrics (return %, win rate, avg trade size).

---

### React Loading Pattern

Summary loads fast on initial page mount. Individual model trade files load lazily on-demand when user selects a specific model to view details.`
      },
      {
        q: 'What metrics compare model performance?',
        a: `The Arena tracks comprehensive metrics for each model, computed from Portfolio class properties.

---

### Portfolio Metrics

- **total_value** - cash + positions_value (sum of position.market_value)
- **return_pct** - ((total_value - STARTING_CAPITAL) / STARTING_CAPITAL) x 100
- **positions_value** - Sum of shares x current_price across all positions
- **unrealized_pnl_pct per position** - ((market_value - cost) / cost) x 100

---

### Trading Activity Metrics

- Total trades executed
- Buy count vs Sell count
- Average trade size (amount_usd)
- Trade frequency (trades per round)
- Ticker concentration (how spread across 20 tickers)

---

### Risk Metrics

- **Max drawdown** - Largest peak-to-trough decline in equity_curve
- **Volatility** - Standard deviation of daily returns
- **Sharpe ratio** - (avg_return - risk_free) / volatility
- **Win rate** - Percentage of profitable trades

---

### Behavioral Metrics (from reasoning analysis)

- Average reasoning length (verbosity)
- Signal emphasis (which factors mentioned most: technical, fundamental, sentiment)
- Conviction language (strong/weak modifiers)
- Contrarian vs consensus positioning

---

### Leaderboard Display

The leaderboard in LiveArenaPage.jsx shows: rank (with medal emojis), model name, portfolio value formatted via fmtK(), return percentage via fmtPct(), and color-coded performance (green positive, red negative).`
      },
      {
        q: 'How do I run the LLM Arena?',
        a: `The Arena supports multiple run modes via command-line arguments:

---

### Run Commands

**Full run (all available rounds):**

python llm_arena.py --run

**Limited run (specific number of rounds):**

python llm_arena.py --run --rounds 50

**Resume from checkpoint (crash recovery):**

python llm_arena.py --resume

**Check status (see progress without running):**

python llm_arena.py --status

**Reset checkpoint (start fresh):**

python llm_arena.py --reset

**Wait mode (auto-retry on API failures):**

python llm_arena.py --run --wait

---

### Timing Configuration

The TIMING dict controls execution:

- inter_model_delay: 3 seconds between each model's API call
- inter_round_delay: 3 seconds after completing a round
- checkpoint_every: 5 rounds between saves
- failure_cooldown: 300 seconds (5 min) wait on failure in --wait mode
- max_failure_retries: 3 attempts per failed round

---

### Required API Keys (from .env)

- **OPENAI_API_KEY** - For GPT-4o-mini
- **GROQ_API_KEY** - For all other models (Groq hosts Llama, Qwen, Kimi, Allam)

---

### Estimated Runtime

(7 models x 3s delay) + 3s round delay = ~24s per round. For 473 rounds: ~3.2 hours. The print_progress() method shows a progress bar with ETA.

---

### Output Files

Saved to outputs/llm_arena/:

- **arena_results.json** - Main results
- **trade_history.json** - All trades
- **checkpoint.json** - Resume state
- **arena_log.txt** - Execution log`
      }
    ]
  },

  {
    id: 'data',
    name: 'Data & Sources',
    icon: '📊',
    faqs: [
      {
        q: 'What tickers are analyzed?',
        a: `The platform analyzes 20 tickers across 6 sectors, defined in the TICKERS array in data_collector.py and llm_arena.py:

---

### Technology (7)

- **AAPL (Apple)** - Consumer tech, $2.8T market cap
- **AMZN (Amazon)** - E-commerce/cloud, $1.8T market cap
- **GOOGL (Alphabet)** - Internet/advertising, $2.1T market cap
- **META (Meta)** - Social media, $1.3T market cap
- **MSFT (Microsoft)** - Enterprise software/cloud, $3.1T market cap
- **NVDA (NVIDIA)** - Semiconductors/AI chips, $3.5T market cap
- **TSLA (Tesla)** - EV/automotive, $780B market cap

### Financials (3)

- **GS (Goldman Sachs)** - Investment banking, $168B market cap
- **JPM (JPMorgan Chase)** - Commercial banking, $570B market cap
- **V (Visa)** - Payment networks, $580B market cap

### Healthcare (3)

- **JNJ (Johnson & Johnson)** - Pharma/consumer health, $375B market cap
- **LLY (Eli Lilly)** - Biotech/pharmaceuticals, $740B market cap
- **UNH (UnitedHealth)** - Health insurance, $485B market cap

### Consumer Staples (3)

- **KO (Coca-Cola)** - Beverages
- **PG (Procter & Gamble)** - Consumer products
- **WMT (Walmart)** - Retail

### Energy (2)

- **CVX (Chevron)** - Integrated oil & gas
- **XOM (Exxon Mobil)** - Integrated oil & gas

### ETFs (2)

- **SPY (S&P 500 ETF)** - Broad market benchmark
- **QQQ (Nasdaq 100 ETF)** - Tech-heavy benchmark

---

**Selection criteria:** High liquidity, sector diversity, quality data availability, and research relevance. Heavy tech exposure tests performance during both AI rallies and corrections.`
      },
      {
        q: 'What data sources are used?',
        a: `The platform uses multiple data sources with fallbacks, configured in data_collector.py and individual agent files:

---

### Price Data (yfinance - primary)

- OHLCV data: Open, High, Low, Close, Volume
- 90 days of historical data per analysis
- Real-time quotes for live monitoring
- Called via yf.download() with progress=False

### Technical Indicators (calculated in technical_agent.py)

- RSI, MACD, Bollinger Bands, ATR, Stochastics
- Moving averages (SMA_5, EMA_10, SMA_20, SMA_50)
- Support/resistance levels
- Volume analysis (volume_ratio vs 20-day average)

### News & Sentiment (multiple sources with fallbacks)

- **Finnhub API (FINNHUB_API_KEY)** - Primary news source, /company-news endpoint, excellent historical support
- **NewsAPI (NEWSAPI_KEY)** - Fallback, limited to ~30 days historical
- **Yahoo Finance** - ticker.news attribute, limited historical
- **Alpha Vantage (ALPHAVANTAGE_KEY)** - Alternative news source
- **Reddit (REDDIT_CLIENT_ID/SECRET)** - Optional social sentiment via PRAW

### Fundamental Data

- **yfinance ticker.info** - P/E, P/B, P/S, market cap, revenue, margins
- **Finnhub /stock/metric** - Additional metrics
- **SEC EDGAR** - Quarterly filings (10-Q, 10-K, 8-K) via TICKER_TO_CIK mapping

### Macro Data (get_macro_data() in data_collector.py)

- SPY, DIA, QQQ for market indices
- VIX for volatility
- Sector ETFs (XLK, XLF, XLV, etc.)
- Treasury yields, Dollar index, Gold, Oil`
      },
      {
        q: 'What is the data collection period?',
        a: `Data collection spans multiple time windows depending on the analysis type:

---

### Strategy Arena (tournament)

- Samples per ticker: 90 rounds
- Sample interval: 3 days (INTERVAL_DAYS constant)
- Total period: ~9 months of historical data
- Total data points: 1,800 (20 tickers x 90 rounds)

### LLM Arena

- Rich data (Rounds 1-90): Full workflow outputs from outputs/workflows/
- Lite data (Rounds 91-473): Market data only from outputs/market_data/
- Total rounds: Up to 473 depending on available data
- Each round corresponds to a specific historical date

### Per-analysis data windows

- Price history: 90 days lookback from analysis_date
- News: 7 days lookback (configurable via --days flag)
- SEC filings: 90 days lookback
- Macro indicators: Point-in-time snapshot

---

### Why 3-day intervals for tournaments?

- Reduces noise from daily fluctuations
- Captures meaningful price movements (typical swing = 2-5 days)
- Balances signal frequency with quality
- Aligns with typical holding periods
- Generates 90 rounds which is statistically significant

### Historical range coverage ensures testing across:

- Bull market periods (tech rally 2023-2024)
- Bear market periods (2022 correction)
- Sideways/consolidation periods
- High volatility events (earnings, Fed announcements)
- Different market regimes (risk-on vs risk-off)`
      },
      {
        q: 'How is data collected and stored?',
        a: `Data collection is orchestrated by data_collector.py with output to outputs/market_data/{date}/ directories.

---

### Collection Process (run_collection() function)

1. Generate date range: Every INTERVAL_DAYS (3) from start to end date
2. For each date and ticker:
   - get_price_data(): 90 days OHLCV from yfinance + calculate technicals
   - get_news(): Finnhub first, NewsAPI fallback, up to 5 articles
   - get_sec_filings(): Recent 10-Q, 10-K, 8-K from EDGAR
3. get_macro_data(): Market-wide context (indices, VIX, yields)
4. Save as JSON to outputs/market_data/{YYYY-MM-DD}/{ticker}.json

---

### File Structure Per Date

outputs/market_data/
- 2024-06-17/
  - AAPL.json
  - NVDA.json
  - ... (20 ticker files)
  - macro.json
- 2024-06-20/
  - ...

---

### Each Ticker JSON Contains

- ticker: "AAPL"
- date: "2024-06-17"
- price: {close: 178.32, open: 177.50, high: 179.10, low: 176.80, volume: 45000000}
- technicals: {rsi: 58.4, macd: 2.34, macd_signal: 1.89, ...}
- news: [{headline: "...", source: "finnhub", sentiment: 0.5}, ...]
- sec_filings: [{type: "10-Q", date: "2024-05-15", ...}]

---

### Rate Limiting

- **Finnhub:** 60 calls/minute, handled with time.sleep(1) on 429 errors
- **SEC EDGAR:** Requires User-Agent header (SEC_USER_AGENT env var)
- **NewsAPI:** Free tier ~100 calls/day, 30-day historical limit`
      },
      {
        q: 'How do I collect new data?',
        a: `Data collection can be run manually or scheduled, using scripts in the agents/ directory.

---

### Collect all market data for a date range

python data_collector.py --start 2024-01-01 --end 2024-06-30

This populates outputs/market_data/ with one directory per date (every 3 days).

---

### Run the full analysis pipeline for one ticker

python master_orchestrator.py AAPL --analysis-date 2024-06-17

This generates outputs/workflows/{timestamp}/ with all phase outputs.

---

### Batch collection for tournaments

python batch_collect_all.py --tickers all --rounds 90

This runs the full pipeline for all 20 tickers x 90 rounds.

---

### Required Environment Variables (.env file in project root)

- OPENAI_API_KEY=sk-... (Required for LLM agents)
- FINNHUB_API_KEY=... (News data)
- NEWSAPI_KEY=... (Fallback news)
- GROQ_API_KEY=... (For LLM Arena models)
- SEC_USER_AGENT="YourApp your@email.com" (Required for EDGAR)

---

### Verify data availability

python llm_arena.py --status

This shows: workflow dates found, market-only dates, total rounds available, API key status.

Data quality checks are built into collection - missing data returns None and is logged but doesn't stop the process. The Arena's get_all_rounds() function automatically determines which rounds have sufficient data.`
      }
    ]
  },

  // ============================================================
  // TECHNICAL DETAILS CATEGORY
  // ============================================================

  {
    id: 'technical',
    name: 'Technical Details',
    icon: '⚙️',
    faqs: [
      {
        q: 'What is the tech stack?',
        a: `### Backend

- **Python 3.11+** - Core language for all backend processing
- **FastAPI** - REST API framework for serving data and running pipelines
- **Custom MasterOrchestrator** - Agent coordination class (no LangGraph/LangChain)
- **Pydantic** - Data validation and structured output parsing from LLMs

### AI/ML

- **OpenAI API** - GPT-4o-mini powers the Strategy Arena pipeline
- **Groq** - Fast inference for some LLM Arena models (Llama)
- **OpenRouter** - Access to additional models (Qwen, Kimi, Allam)
- **Structured JSON mode** - Ensures reliable, parseable agent outputs
- **Async parallel processing** - Runs independent agents concurrently

### Frontend

- **React 18 with Vite** - Fast development with hot module replacement
- **CSS Variables** - Consistent theming with full light/dark mode support
- **Responsive design** - Works across desktop and tablet viewports

### Data Sources

- **Yahoo Finance API** - Primary source for OHLCV price data
- **Finnhub API** - News aggregation and fundamental metrics
- **NewsAPI** - Broader market headlines and sentiment
- **Alpaca API** - Supplementary historical data
- **PRAW (Reddit API)** - Social sentiment from investing subreddits

### Storage & Output

- JSON file storage for all workflow outputs
- 35,000+ structured output files with complete agent reasoning
- CSV exports available for tournament results and trade history
- Per-ticker detailed analysis with visualizations`
      },
      {
        q: 'How are agents orchestrated?',
        a: `Agents are orchestrated using a **custom MasterOrchestrator** class built with direct OpenAI API calls. This was a deliberate design choice over frameworks like LangGraph or LangChain to maintain full control over execution flow, state management, and error handling.

---

### Execution Flow

**Phase 1 (Parallel - ~60 seconds):**

Technical Analyst, News Analyst, Fundamental Analyst, Macro Analyst all run in parallel, then merge outputs for Phase 2.

**Phase 2 (Parallel - ~40 seconds):**

Bull Researcher and Bear Researcher both receive Phase 1 context and run in parallel, then pass to Phase 3.

**Phase 3 (Sequential - ~30 seconds):**

Research Manager receives both theses, runs debate synthesis, then passes to Phase 4.

**Phase 4 (Parallel - ~30 seconds):**

Aggressive Evaluator, Neutral Evaluator, Conservative Evaluator all run in parallel, then merge outputs for Phase 5.

**Phase 5 (Sequential - ~10 seconds):**

Risk Manager makes final BUY/HOLD/REJECT decision with position sizing.

---

### Key Implementation Details

- Parallel execution where dependencies allow (Phases 1, 2, 4)
- Sequential execution where outputs feed into next agent (Phases 3, 5)
- State management carries context between phases
- Error handling includes retries with exponential backoff
- Structured output validation via Pydantic models
- Each agent has specialized system prompt defining personality and output format
- Total pipeline runtime: ~3 minutes per ticker`
      },
      {
        q: 'How can I run this locally?',
        a: `### Prerequisites

- Node.js 18+
- Python 3.11+
- OpenAI API key

---

### Frontend Setup

cd gtrade-arena
npm install
npm run dev

---

### Backend Setup

cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python main.py

---

### Environment Variables

- OPENAI_API_KEY - Required for LLM agents
- FINNHUB_API_KEY - For news data
- DEBUG=true - Enable verbose logging

---

### Development Mode

- Frontend: http://localhost:5173
- Backend: http://localhost:8000

---

### GitHub Repository

https://github.com/priyam-choksi/matsmatsmats

Full documentation, sample data, and instructions for running complete tournament simulations available in the repo.`
      }
    ]
  },

  // CODEBASE CATEGORY - Paste this into the faqCategories array

  {
    id: 'codebase',
    name: 'Codebase',
    icon: '📁',
    faqs: [
      {
        q: 'Project folder structure',
        a: `The project has two main parts: Python backend under /agents and React frontend under /gtrade-arena.

---

### Backend Structure (/agents)

**Orchestrators (/agents/orchestrators/):**

- **master_orchestrator.py** - Main pipeline coordinator, runs all 5 phases
- **discussion_hub.py** - Phase 1 aggregator, synthesizes 4 analyst outputs
- **orchestrator_agent.py** - Legacy single-ticker orchestration
- **batch_collect_all.py** - Parallel data collection across all 20 tickers

**Analysts (/agents/analyst/):**

- **technical_agent.py** - RSI, MACD, Bollinger, support/resistance
- **news_agent.py** - Yahoo Finance, Finnhub, NewsAPI, Reddit sentiment
- **fundamental_agent.py** - P/E, P/B, margins, peer comparison
- **macro_agent.py** - Fed policy, VIX, sector rotation, regime detection

**Researchers (/agents/researcher/):**

- **bull_researcher.py** - Constructs bullish thesis with catalysts
- **bear_researcher.py** - Constructs bearish thesis with risks

**Managers (/agents/managers/):**

- **research_manager.py** - Facilitates debate, synthesizes into recommendation

**Risk Management (/agents/risk_management/):**

- **aggressive_debator.py** - Risk-seeking evaluator (default: STRONG BUY)
- **neutral_debator.py** - Balanced evaluator (default: HOLD)
- **conservative_debator.py** - Risk-averse evaluator (default: AVOID)
- **risk_manager.py** - Final decision maker, aggregates votes

**Game Theory (/agents/game_theory/):**

- **gt_engine.py** - Main tournament engine
- **game_state.py** - Tracks allocations, positions, round history
- **data_loader.py** - Loads workflow JSONs for tournament
- **market_context.py** - Market data structure for strategies
- **metrics_calculator.py** - Sharpe, drawdown, win rate calculations
- **run_analysis.py** - CLI entry point for tournaments

**Strategies (/agents/game_theory/strategies/):**

- **base.py** - Strategy abstract base class
- **signal_follower.py** - LLM signal following
- **cooperator.py** - Momentum/group following
- **defector.py** - Mean reversion/contrarian
- **tit_for_tat.py** - Adaptive copying
- **buy_hold.py** - Passive benchmark

**LLM Arena (/agents/llm_arena/):**

- **llm_arena.py** - Multi-model competition runner
- **preprocess_llm_arena.py** - Converts results to frontend JSON

**Outputs (/outputs/):**

- **workflows/{ticker}/{date}/** - Individual analysis outputs
- **game_theory_results/** - Tournament results by ticker
- **llm_arena/** - LLM competition results

---

### Frontend Structure (/gtrade-arena)

**Source (/src/):**

- **App.jsx** - Main app, theme provider, routing

**Pages (/src/pages/):**

- **MarketPage.jsx** - Real-time market terminal
- **PipelinePage.jsx** - Pipeline runner UI
- **TournamentPage.jsx** - Game theory results viewer
- **LiveArenaPage.jsx** - LLM Arena dashboard
- **AgentsPage.jsx** - 11 agents documentation
- **DocsPage.jsx** - This documentation page

**Components (/src/components/):**

- **Sidebar.jsx** - Navigation sidebar
- **chat/** - RAG chat components

**Data (/src/data/):**

- **tournament/** - Pre-processed tournament JSONs
- **llm_arena/** - Pre-processed LLM arena JSONs`
      },
      {
        q: 'technical_agent.py - What it does',
        a: `**Location:** /agents/analyst/technical_agent.py

This agent calculates technical indicators and generates trading signals. It's the most computationally intensive analyst because it processes raw price data.

---

### Class: TechnicalAgent

**Key methods:**

**get_price_data(days)** - Fetches OHLCV from yfinance. If analysis_date is set (historical mode), it fetches data ending on that date. Otherwise uses current data. Downloads 90 days to ensure enough data for longer-period indicators.

**calculate_indicators(df)** - Computes all technical indicators on the DataFrame:

- RSI (14-period): delta.where(delta > 0, 0).rolling(14).mean() / abs delta for gains/losses
- MACD: EMA(12) - EMA(26), Signal = EMA(9) of MACD, Histogram = MACD - Signal
- Bollinger Bands: 20-period SMA +/- 2 standard deviations
- Moving Averages: SMA_5, EMA_10, SMA_20, SMA_50
- ATR: max(High-Low, |High-PrevClose|, |Low-PrevClose|) rolling 14
- Stochastic: (Close - Low14) / (High14 - Low14) * 100

**identify_support_resistance(df)** - Finds key price levels using rolling min/max over 20 periods. Support = recent lows that held, Resistance = recent highs that rejected.

**format_technical_summary(df, levels, targets, days)** - Builds the text summary sent to the LLM. Includes current price, period change, each indicator value with interpretation (e.g., "RSI: 72.3 - OVERBOUGHT").

**analyze_with_llm(technical_summary)** - Sends the formatted summary to GPT-4o-mini with a detailed system prompt defining the analysis framework. Returns BUY/HOLD/SELL with confidence.

**_get_llm_decision(analysis)** - Fallback extraction if the main LLM response doesn't have clear recommendation. Uses a second LLM call with strict output format.

---

### Usage

python technical_agent.py AAPL --days 7 --analysis-date 2024-06-15

---

### Output

Markdown report with trend assessment, indicator values, support/resistance levels, and RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low`
      },
      {
        q: 'news_agent.py - What it does',
        a: `**Location:** /agents/analyst/news_agent.py

This agent aggregates news from multiple sources and performs sentiment analysis.

---

### Class: NewsAgent

**Data sources (configurable via --sources flag):**

**yahoo** - Uses yfinance ticker.news to get recent headlines. Extracts title, publisher, link, publish time.

**finnhub** - Calls Finnhub API /company-news endpoint. Requires FINNHUB_KEY env var. Returns headline, summary, source, datetime.

**newsapi** - Calls NewsAPI /everything endpoint searching for ticker + company name. Requires NEWSAPI_KEY. Broader coverage but less financial-specific.

**reddit** - Uses PRAW to search r/stocks, r/investing, r/wallstreetbets. Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET. Captures social sentiment.

---

### Key Methods

**fetch_yahoo_news(days)** - Gets news from yfinance, filters to last N days, formats as list of dicts.

**fetch_finnhub_news(days)** - API call with date range, returns articles with sentiment pre-labeled by Finnhub.

**fetch_reddit_sentiment(days)** - Searches subreddits for ticker mentions, extracts post titles and scores.

**aggregate_news(sources, days)** - Combines all sources, deduplicates by title similarity, sorts by date.

**analyze_sentiment_with_llm(articles)** - Sends article list to GPT-4o-mini. System prompt instructs it to:

- Identify overall sentiment (Bullish/Bearish/Neutral)
- Extract key catalysts and their impact
- Note sentiment shifts vs. historical
- Flag time-sensitive news vs. already-priced-in

---

### Output

The agent outputs a structured report with:

- Overall Sentiment Score (-1 to +1)
- Key Headlines (top 5 with impact assessment)
- Catalyst Timeline (upcoming events)
- RECOMMENDATION based on news environment

**Historical mode:** When analysis_date is set, only news published before that date is included.`
      },
      {
        q: 'fundamental_agent.py - What it does',
        a: `**Location:** /agents/analyst/fundamental_agent.py

This agent evaluates company financials and valuation metrics.

---

### Class: FundamentalAgent

**Data sources:**

- **yfinance ticker.info** - Gets P/E, P/B, P/S, market cap, revenue, earnings, margins, etc.
- **yfinance ticker.financials** - Quarterly/annual income statements, balance sheets.
- **Finnhub /stock/metric** - Additional metrics like EV/EBITDA, PEG ratio.
- **SEC EDGAR (optional)** - For 10-K/10-Q filings text extraction.

---

### Key Metrics Calculated

**Valuation:**

- P/E Ratio (trailing and forward)
- P/B Ratio (Price to Book)
- P/S Ratio (Price to Sales)
- EV/EBITDA
- PEG Ratio (P/E / Growth Rate)

**Growth:**

- Revenue Growth (YoY)
- Earnings Growth (YoY)
- Quarterly growth trends

**Profitability:**

- Gross Margin
- Operating Margin
- Net Profit Margin
- ROE (Return on Equity)
- ROA (Return on Assets)

**Financial Health:**

- Debt/Equity Ratio
- Current Ratio
- Quick Ratio
- Free Cash Flow

**Peer Comparison:** Compares metrics to sector averages from yfinance.

---

### LLM Analysis

The LLM receives all metrics and determines:

- Valuation: Undervalued / Fair Value / Overvalued
- Growth Quality: Accelerating / Stable / Decelerating
- Financial Health: Strong / Adequate / Weak
- RECOMMENDATION with reasoning

**Output includes specific numbers:** "P/E of 28.5 vs sector average of 22.1 suggests 29% premium, but justified by 35% earnings growth vs sector 12%."`
      },
      {
        q: 'macro_agent.py - What it does',
        a: `**Location:** /agents/analyst/macro_agent.py

This agent assesses broader market conditions that affect individual stocks.

---

### Class: MacroAgent

**Key methods (each is a "tool" the agent can call):**

**get_market_indicators(days)** - Fetches major indices via yfinance:

- S&P 500 (^GSPC) - Broad market trend
- Dow Jones (^DJI) - Blue chip sentiment
- NASDAQ (^IXIC) - Tech sector
- VIX (^VIX) - Volatility/fear gauge
- 10-Year Treasury (^TNX) - Rate environment
- Russell 2000 (^RUT) - Small cap risk appetite

For each: calculates % change over period, annualized volatility, trend direction.

**get_sector_performance(days)** - Fetches 11 sector ETFs:

XLK (Tech), XLF (Financials), XLE (Energy), XLV (Healthcare), XLY (Consumer Disc), XLP (Consumer Staples), XLI (Industrials), XLB (Materials), XLU (Utilities), XLRE (Real Estate), XLC (Communications)

Identifies: Leading sectors, lagging sectors, rotation patterns.

**get_economic_indicators(days)** - Fetches macro proxies:

- DX-Y.NYB (Dollar Index) - USD strength
- GC=F (Gold) - Safe haven demand
- CL=F (Crude Oil) - Economic activity proxy
- BTC-USD (Bitcoin) - Risk appetite
- HYG (High Yield Bonds) - Credit conditions
- TLT (20Y Treasury) - Flight to safety

**detect_regime()** - Determines market regime based on:

- VIX level (<15 = complacent, 15-25 = normal, >25 = fearful, >35 = crisis)
- S&P 500 trend (above/below 50-day and 200-day MA)
- Yield curve (normal/flat/inverted)
- Sector rotation (cyclical vs defensive leadership)

---

### Output

RISK-ON / RISK-OFF / NEUTRAL with confidence level

The final output answers: "Is this a good environment for this stock's sector?" Not just "is the market up/down."`
      },
      {
        q: 'discussion_hub.py - What it does',
        a: `**Location:** /agents/orchestrators/discussion_hub.py

This is the Phase 1 aggregator that synthesizes all 4 analyst outputs into structured discussion points for Phase 2 researchers.

---

### Class: DiscussionHub

**Workflow:**

1. Spawns each analyst agent as subprocess (parallel execution)
2. Captures stdout from each agent
3. Parses recommendations and confidence from each report
4. Identifies agreements and conflicts between analysts
5. Generates synthesis for downstream agents

---

### Key Methods

**run_analyst(agent_name, agent_script)** - Executes single analyst via subprocess.run(). Passes ticker and --analysis-date if in historical mode. Timeout: 90 seconds.

**extract_recommendation(report)** - Regex patterns to find "RECOMMENDATION: BUY/HOLD/SELL" and "Confidence: High/Medium/Low" from report text.

**create_analyst_summary(report, analyst_type)** - Creates one-liner summary for each analyst. Example: "TECHNICAL: Uptrend confirmed, RSI 58 - BUY (High)"

**extract_key_points(report, analyst_type)** - Scans for bullish/bearish signals. Uses keyword matching: ["uptrend", "oversold", "support level", "bullish crossover"] for bullish, ["downtrend", "overbought", "resistance", "bearish"] for bearish.

**identify_conflicts()** - Compares recommendations across analysts. If Technical says BUY and Fundamental says SELL, flags as CRITICAL conflict. If 2 vs 2 split, flags as MODERATE.

**find_consensus()** - What do multiple analysts agree on? "3/4 analysts bullish on near-term momentum"

**create_llm_synthesis(discussion_data)** - Sends all 4 full reports to GPT-4o-mini with system prompt:

- Identify 2-3 most important factors
- Build Bull Case (3-5 points with source)
- Build Bear Case (3-5 points with source)
- Flag key conflicts to resolve
- Provide SYNTHESIS DIRECTION: BULLISH/BEARISH/NEUTRAL

---

### Output File: discussion_points.json

Contains:

- Full analyst reports (preserved for downstream)
- Analyst summaries
- Bull/bear evidence extracted
- Conflicts and consensus
- LLM synthesis
- synthesis_direction and synthesis_confidence

This becomes the input for both bull_researcher.py and bear_researcher.py.`
      },
      {
        q: 'research_manager.py - What it does',
        a: `**Location:** /agents/managers/research_manager.py

This is the Phase 3 synthesis agent that facilitates bull/bear debate and produces the final research recommendation.

---

### Class: ResearchManager

**Inputs:**

- bull_thesis.json (from Phase 2)
- bear_thesis.json (from Phase 2)
- Optional: risk evaluations if running post-Phase 4

---

### Key Methods

**load_research_files(bull_file, bear_file)** - Loads both thesis JSONs, extracts validation scores, core thesis, risk/reward metrics.

**run_debate(rounds)** - If debate_rounds > 0, facilitates structured debate:

**Round structure:**

1. Bull presents argument (300-500 words)
2. Bear responds, countering Bull's points
3. Bull rebuts Bear's response
4. Repeat for N rounds

Each argument is generated by LLM with prompt: "You are Bull Analyst in Round {N}. Your thesis: {summary}. Previous argument from opponent: {text}. DEBATE RULES: Address opponent's points directly, use evidence from your thesis, don't fabricate new data."

**_build_thesis_summary(thesis, side)** - Formats thesis for debate context:

- Core thesis statement
- Upside/downside percentages
- Risk/reward ratio
- Top 3 signals
- Key catalysts/triggers

**_build_synthesis_prompt()** - Creates final prompt for LLM synthesis: "You are Research Manager. Synthesize all research and make FINAL decision. Consider: Which thesis has stronger evidence? Who won debate? What do risk evaluators say?"

**_validate_synthesis(synthesis)** - Guardrails on LLM output:

- Probabilities must sum to ~100
- Recommendation must match probability (can't say BUY if bull_prob < 40)
- Position size must be within bounds (0-20%)
- Confidence must align with data quality

---

### Output: research_synthesis.json

Contains:

- probabilities: {bull_case: 62, bear_case: 28, base_case: 10}
- debate_winner: "bull" with reasoning
- thesis_quality: which thesis was better supported
- conclusion: {recommendation: "BUY", confidence: "MEDIUM", position_size: "12%"}
- key_factors: [{factor: "AI growth catalysts", impact: "bullish", weight: "high"}]
- full_synthesis: 2-3 paragraph written summary

The debate rounds (0, 3, or 5 depending on research_mode) are the key differentiator between shallow/deep/research modes.`
      },
      {
        q: 'risk_manager.py - What it does',
        a: `**Location:** /agents/risk_management/risk_manager.py

This is the Phase 5 final decision maker that aggregates all evaluator outputs and makes the trading decision.

---

### Class: RiskManager

**Inputs:**

- research_synthesis.json
- aggressive_eval.json
- neutral_eval.json
- conservative_eval.json

---

### Key Methods

**load_evaluations()** - Loads all 3 evaluator JSONs. Each contains:

- stance: "STRONG BUY" / "BUY" / "HOLD" / "AVOID" / "SELL"
- position_pct: 0-100
- confidence: "HIGH" / "MEDIUM" / "LOW"
- reasoning: text explanation
- red_flags: [{flag, severity, source}] (from conservative)

**count_agent_votes(synthesis, evaluations)** - Tallies votes:

- Research Manager recommendation counts as 1 vote
- Each evaluator stance counts as 1 vote
- Result: {bullish: 2, bearish: 1, neutral: 1, consensus: "LEAN_BULLISH"}

**Consensus levels:** STRONG_BULLISH (4 bullish), LEAN_BULLISH (3), SPLIT (2-2), LEAN_BEARISH (1), STRONG_BEARISH (0)

**_determine_trade_outcome(votes, lookahead)** - In historical mode with lookahead prices:

- If we would have bought and price went up - WIN
- If we would have bought and price went down - LOSS
- If we avoided and price went down - AVOIDED (good)
- If we avoided and price went up - MISSED (bad)

**_llm_calibrate_position(synthesis, evaluations, votes, base_range)** - LLM adjusts position within the base range. Considers:

- Vote consensus strength
- Evaluator confidence levels
- Red flags from conservative
- Risk/reward from synthesis

**determine_verdict(position_pct, outcome)** - Maps to final verdict:

- position_pct < 0.5% - REJECT
- position_pct >= 4.5% - APPROVE (becomes BUY)
- else - MODIFY (becomes BUY with caveats)

**set_risk_controls(position_dollars, synthesis)** - Sets:

- stop_loss: Based on bear thesis downside (typically 8-15%)
- profit_targets: [50%, 75%, 100%] of bull thesis upside
- scale_out: [33%, 33%, 34%] at each target

---

### Output: risk_decision.json

Contains:

- verdict: "BUY" / "HOLD" / "REJECT"
- final_position_pct: e.g., 0.12 (12%)
- final_position_dollars: e.g., $12,000
- confidence: "HIGH" / "MEDIUM" / "LOW"
- reasoning: why this decision
- key_factors: top 3 reasons
- agent_votes: breakdown of who voted what
- stop_loss_pct, profit_targets
- risk_consensus: summary of evaluator agreement`
      },
      {
        q: 'gt_engine.py - Tournament engine',
        a: `**Location:** /agents/game_theory/gt_engine.py

This is the main engine that runs the Strategy Arena tournament.

---

### Class: GTEngine

**Constructor params:**

- total_capital: $1,000,000 (default)
- reallocation_rate: 0.10 (10% of pool transfers per round)
- output_dir: Where to save results

**Initialization:**

- Creates 4 tournament strategies via get_tournament_strategies()
- Creates benchmark (Buy-and-Hold) via get_benchmark_strategy()
- Sets up output directory structure

---

### Key Methods

**run_ticker(ticker)** - Runs full tournament for one ticker:

1. Loads all samples via DataLoader.load_ticker_data(ticker)
2. Returns list of {date, sample_num, risk_evals, price_data}
3. Initializes GameState with 4 strategies, $250K each
4. For each of 90 rounds:
   - Creates MarketContext from that round's data
   - Each strategy calls decide_position(market, game)
   - Calculates returns based on position x market_return
   - Determines round winner (highest return)
   - Runs z-score reallocation
   - Updates GameState
5. Calculates final metrics
6. Generates visualizations
7. Saves all outputs

**_run_single_round(game, market, round_num)** - One round logic:

1. Get each strategy's position (0-100%)
2. Calculate each strategy's return: position% x market_return%
3. Convert to dollar returns based on current allocation
4. Find winner (best percentage return)
5. Call game.update_round() to record and reallocate

**_calculate_z_score_reallocation(returns)** - The game theory mechanism:

- returns = {strategy_name: percentage_return}
- mean = average of all returns
- std = standard deviation of returns
- For each strategy: z = (return - mean) / std
- new_allocation = old_allocation x (1 + z x reallocation_rate)
- Normalize so total = $1M

**_generate_dashboard(ticker, game)** - Creates PNG visualization:

- Capital progression chart (line graph, all 90 rounds)
- Win distribution bar chart
- Final standings table
- Position ranges box plot
- Regime shading (green=bull, red=bear, yellow=sideways)

---

### Output Files Per Ticker

- {ticker}_detailed.json - Full round-by-round data
- {ticker}_summary.json - Final metrics only
- {ticker}_readable.txt - Human-readable log
- {ticker}_trades.csv - For spreadsheet analysis
- {ticker}_dashboard.png - Visual summary
- {ticker}.log - Execution log

**run_all_tickers()** - Loops through all 20 tickers, calls run_ticker(), then generates combined cross-ticker analysis.`
      },
      {
        q: 'Strategy implementations',
        a: `**Location:** /agents/game_theory/strategies/

All strategies inherit from Strategy base class with method:

decide_position(market: MarketContext, game: GameState) -> float

---

### signal_follower.py - LLM Signal Following

SignalFollowerStrategy reads the 3 risk evaluator positions:

- aggressive_pct from aggressive_eval.json
- neutral_pct from neutral_eval.json
- conservative_pct from conservative_eval.json

**Position logic:**

If max(aggressive, neutral, conservative) > 10:
    Use strongest signal, scaled up 3x
    position = max(all_signals) x 3.0
Else:
    Use average, scaled up
    position = average(all_signals) x 3.0

Capped at 80%, minimum 5%.

Ignores what other strategies are doing - pure signal test.

---

### cooperator.py - Momentum/Group Following

CooperatorStrategy blends group average with LLM signals:

**Round 1-3 (warmup):** Uses LLM signal directly

**Round 4+:**

- group_avg = average position of all strategies last round
- llm_signal = average of evaluator positions x 3.0
- position = 0.70 x group_avg + 0.30 x llm_signal

Also calculates momentum from recent returns:

- if sum(last 3 returns) > 0.02: position += 10 (Bullish momentum)
- elif sum(last 3 returns) < -0.02: position -= 10 (Bearish momentum)

---

### defector.py - Mean Reversion/Contrarian

DefectorStrategy bets against recent trends:

**Round 1-2 (warmup):** Uses LLM signal

**Round 3+:**

- cumulative_return = sum of last 3 market returns
- group_avg = average position of others

If cumulative_return > 0.015 (Market overbought):
    Go bearish (low position)
    position = 100 - group_avg (Opposite of group)
Elif cumulative_return < -0.015 (Market oversold):
    Go bullish (high position)
    position = 100 - group_avg
Else:
    Neutral - use LLM
    position = llm_signal

**The key:** When everyone is bullish (high positions), Defector goes low. When everyone is bearish, Defector goes high.

---

### tit_for_tat.py - Adaptive Copying

TitForTatStrategy copies whatever worked last round:

**Round 1:** Start cooperative (position = 15%, "nice")

**Round 2+:**

- winner = game.last_winner (Who won last round)
- winner_position = game.last_positions[winner]
- winner_streak = count consecutive wins
- copy_weight = min(0.9, 0.5 + winner_streak x 0.1)
- position = copy_weight x winner_position + (1 - copy_weight) x llm_signal

If winner was on a 3+ round streak, TFT commits more heavily (90% copy). If winner just started winning, TFT is more cautious (50% copy).

---

### buy_hold.py - Passive Benchmark

BuyHoldStrategy always returns 100.0

Not in tournament capital pool - tracked separately. Used to measure: "Did active strategies beat passive investing?"`
      },
      {
        q: 'game_state.py - State tracking',
        a: `**Location:** /agents/game_theory/game_state.py

GameState tracks everything about the tournament as it runs.

---

### Class: GameState

**Constructor:**

GameState(
    strategy_names: List[str],  # ["Signal Follower", "Cooperator", ...]
    total_capital: float = 1_000_000,
    reallocation_rate: float = 0.10
)

Initializes allocations: {name: total_capital / len(names)} for each strategy.

---

### Key Attributes

**allocations: Dict[str, float]** - Current capital for each strategy. Starts at $250K each. After 90 rounds, could be anything from $0 to $1M+.

**round_num: int** - Current round number (1-90)

**rounds: List[RoundResult]** - Complete history of all rounds played.

**position_history: List[Dict[str, float]]** - What position each strategy took each round.

**last_winner: str** - Name of strategy that won last round.

**last_positions: Dict[str, float]** - Positions from last round (for TFT to copy).

**benchmark_curve: List[float]** - Buy-and-Hold portfolio value at each round.

---

### Key Methods

**update_round(positions, returns, market_return)** - Called after each round:

1. Calculates dollar returns for each strategy
2. Creates RoundResult with all data
3. Determines winner
4. Calls _reallocate_capital(returns)
5. Appends to history

**_reallocate_capital(returns)** - Z-score reallocation:

- z_scores = {name: (return - mean) / std for each}
- For each name in allocations:
    - multiplier = 1 + (z_scores[name] x reallocation_rate)
    - multiplier = clamp(multiplier, 0.8, 1.2) (Prevent extreme swings)
    - allocations[name] *= multiplier
- Normalize to maintain total:
    - scale = total_capital / sum(allocations.values())
    - allocations = {k: v x scale for k, v in allocations.items()}

**get_cooperation_rate() -> float** - How similar were positions?

- For each round, calculate std dev of positions
- Cooperation = 1 - (std / 50) where 50 = max possible std
- Return average across all rounds
- High cooperation (~0.78) means strategies took similar positions

**get_allocation_gini() -> float** - Capital concentration.

- 0 = perfectly equal ($250K each)
- 1 = one strategy has everything
- Calculated using Gini coefficient formula

---

### Class: RoundResult (dataclass)

Stores everything about one round:

- round_num, date, regime
- positions: what each strategy took
- dollar_returns: how much each made/lost
- pct_returns: percentage return
- allocations_after: capital after reallocation
- winner: who won this round
- high_variance_round: bool (was this a "defection" round?)`
      },
      {
        q: 'llm_arena.py - Model competition',
        a: `**Location:** /agents/llm_arena/llm_arena.py

This runs the LLM Arena where 7 models compete on identical trading tasks.

---

### Configuration

**ARENA_MODELS:**

- "GPT-4o-mini": {provider: "openai", model: "gpt-4o-mini"}
- "Llama-3.3-70B": {provider: "groq", model: "llama-3.3-70b-versatile"}
- "Llama-4-Maverick": {provider: "groq", model: "meta-llama/llama-4-maverick-17b-128e-instruct"}
- "Kimi-K2": {provider: "groq", model: "moonshotai/kimi-k2-instruct-0905"}
- "Qwen3-32B": {provider: "groq", model: "qwen/qwen3-32b"}
- "GPT-OSS-120B": {provider: "groq", model: "openai/gpt-oss-120b"}
- "Allam-2-7B": {provider: "groq", model: "allam-2-7b"}

**STARTING_CAPITAL** = $250,000 per model
**MAX_POSITION_PCT** = 40% (no single position > 40% of portfolio)

---

### Class: Portfolio (dataclass)

- cash: float
- positions: Dict[ticker, Position]
- trades: List[Trade]
- equity_curve: List[float]

**Methods:** execute_buy(), execute_sell(), total_value, return_pct

---

### Class: TradingArena

**Key methods:**

**get_all_rounds()** - Builds list of all rounds from:

- Workflow data (rich): 90 rounds with full pipeline output
- Market data (lite): Additional rounds with just price/indicators
- Total: ~473 rounds

**run_round(round_info, market_data) -> bool:**

For each model:
1. Build prompt with market context
2. Call LLM via call_llm()
3. Parse response for trades (JSON in <json> tags)
4. Execute trades on model's portfolio
5. Record reasoning

If ANY model fails (API error), round is aborted (no partial saves).

**call_llm(provider, model, prompt) -> str:**

Routes to correct API:
- OpenAI: api.openai.com/v1/chat/completions
- Groq: api.groq.com/openai/v1/chat/completions
- Google: generativelanguage.googleapis.com (disabled)

Includes retry logic with exponential backoff. Rate limiting between calls (2-3 seconds for Groq).

**parse_response(response) -> (trades, reasoning):**

Extracts:
- <reasoning>...</reasoning> - Model's thought process
- <json>{"trades": [...], "market_outlook": "..."}</json>

Trades format: [{ticker, action, amount_usd, reasoning}]

---

### Checkpoint System

- Saves state every 5 rounds
- Can resume from checkpoint after crash
- Command: python llm_arena.py --resume

---

### Output

- arena_results.json - Final standings, metrics
- trade_history.json - Every trade by every model
- Per-model equity curves
- Reasoning samples for analysis`
      }
    ]
  },
]

// Icons
const Icons = {
  ChevronDown: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
  Book: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </svg>
  ),
  Search: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" />
    </svg>
  ),
  ExternalLink: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" x2="21" y1="14" y2="3" />
    </svg>
  ),
  GitHub: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
    </svg>
  ),
  LinkedIn: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  )
}

// Replace your existing FAQItem component with this one
// Make sure you have: import ReactMarkdown from 'react-markdown'

const FAQItem = ({ faq, isOpen, onToggle }) => {
  return (
    <div style={{
      background: 'var(--bg-card)',
      borderRadius: '12px',
      border: '1px solid var(--border-primary)',
      overflow: 'hidden',
      marginBottom: '10px',
    }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%',
          padding: '16px 20px',
          textAlign: 'left',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: isOpen ? 'var(--bg-tertiary)' : 'none',
          border: 'none',
          cursor: 'pointer',
          gap: '12px',
          transition: 'background 0.15s'
        }}
      >
        <span style={{
          fontSize: '14px',
          fontWeight: '600',
          color: 'var(--text-primary)',
          lineHeight: '1.4'
        }}>
          {faq.q}
        </span>
        <span style={{
          color: 'var(--text-muted)',
          transform: isOpen ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
          flexShrink: 0
        }}>
          <Icons.ChevronDown size={18} />
        </span>
      </button>

      {isOpen && (
        <div style={{
          padding: '16px 20px 20px',
          fontSize: '13px',
          color: 'var(--text-secondary)',
          lineHeight: '1.7',
          borderTop: '1px solid var(--border-primary)',
        }}>
          <ReactMarkdown
            components={{
              h1: ({ children }) => <h1 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)', margin: '20px 0 12px', borderBottom: '1px solid var(--border-primary)', paddingBottom: '8px' }}>{children}</h1>,
              h2: ({ children }) => <h2 style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-primary)', margin: '18px 0 10px' }}>{children}</h2>,
              h3: ({ children }) => <h3 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', margin: '16px 0 8px' }}>{children}</h3>,
              p: ({ children }) => <p style={{ margin: '0 0 12px', lineHeight: '1.7' }}>{children}</p>,
              ul: ({ children }) => <ul style={{ margin: '8px 0 12px', paddingLeft: '20px' }}>{children}</ul>,
              ol: ({ children }) => <ol style={{ margin: '8px 0 12px', paddingLeft: '20px' }}>{children}</ol>,
              li: ({ children }) => <li style={{ margin: '4px 0', lineHeight: '1.6' }}>{children}</li>,
              code: ({ inline, children }) => inline
                ? <code style={{ background: 'var(--bg-tertiary)', padding: '2px 6px', borderRadius: '4px', fontSize: '12px', fontFamily: 'Monaco, Consolas, monospace', color: 'var(--accent-primary)' }}>{children}</code>
                : <code style={{ display: 'block', background: 'var(--bg-tertiary)', padding: '12px 16px', borderRadius: '8px', fontSize: '12px', fontFamily: 'Monaco, Consolas, monospace', overflowX: 'auto', margin: '8px 0 12px', border: '1px solid var(--border-primary)' }}>{children}</code>,
              pre: ({ children }) => <pre style={{ background: 'var(--bg-tertiary)', padding: '14px 16px', borderRadius: '8px', fontSize: '12px', fontFamily: 'Monaco, Consolas, monospace', overflowX: 'auto', margin: '12px 0', border: '1px solid var(--border-primary)', lineHeight: '1.5' }}>{children}</pre>,
              strong: ({ children }) => <strong style={{ color: 'var(--text-primary)', fontWeight: '600' }}>{children}</strong>,
              a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-primary)', textDecoration: 'none' }}>{children}</a>,
              hr: () => <hr style={{ border: 'none', borderTop: '1px solid var(--border-primary)', margin: '16px 0' }} />,
              blockquote: ({ children }) => <blockquote style={{ borderLeft: '3px solid var(--accent-primary)', paddingLeft: '16px', margin: '12px 0', color: 'var(--text-muted)', fontStyle: 'italic' }}>{children}</blockquote>,
            }}
          >
            {faq.a}
          </ReactMarkdown>
        </div>
      )}
    </div>
  )
}

// Main Component
export default function DocsPage() {
  const { isDark } = useTheme()
  const [selectedCategory, setSelectedCategory] = useState('overview')
  const [openFAQ, setOpenFAQ] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')

  const currentCategory = faqCategories.find(c => c.id === selectedCategory)

  const filteredFAQs = searchQuery
    ? faqCategories.flatMap(cat =>
      cat.faqs.filter(faq =>
        faq.q.toLowerCase().includes(searchQuery.toLowerCase()) ||
        faq.a.toLowerCase().includes(searchQuery.toLowerCase())
      ).map(faq => ({ ...faq, category: cat.name }))
    )
    : currentCategory?.faqs || []

  return (
    <div style={{
      height: '100vh',
      background: 'var(--bg-secondary)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <header style={{
        background: 'var(--bg-card)',
        borderBottom: '1px solid var(--border-primary)',
        padding: '12px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              Documentation
            </h1>
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
              FAQ & Technical Reference
            </p>
          </div>
        </div>

        {/* Search */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'var(--bg-tertiary)',
          borderRadius: '10px',
          padding: '0 12px',
          width: '300px',
          border: '1px solid var(--border-primary)'
        }}>
          <Icons.Search size={16} style={{ color: 'var(--text-muted)' }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documentation..."
            style={{
              flex: 1,
              padding: '9px 0',
              border: 'none',
              background: 'transparent',
              fontSize: '13px',
              outline: 'none',
              color: 'var(--text-primary)'
            }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                padding: '4px'
              }}
            >
              ✕
            </button>
          )}
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar */}
        <aside style={{
          width: '240px',
          padding: '20px',
          borderRight: '1px solid var(--border-primary)',
          background: 'var(--bg-card)',
          overflowY: 'auto',
          flexShrink: 0
        }}>
          <div style={{
            fontSize: '11px',
            fontWeight: '600',
            color: 'var(--text-muted)',
            marginBottom: '12px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            Categories
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {faqCategories.map(cat => (
              <button
                key={cat.id}
                onClick={() => {
                  setSelectedCategory(cat.id)
                  setSearchQuery('')
                  setOpenFAQ(null)
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: 'none',
                  background: selectedCategory === cat.id && !searchQuery
                    ? 'var(--bg-hover)'
                    : 'transparent',
                  color: selectedCategory === cat.id && !searchQuery
                    ? 'var(--text-primary)'
                    : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: '500',
                  textAlign: 'left',
                  transition: 'all 0.15s'
                }}
              >
                <span style={{ fontSize: '16px' }}>{cat.icon}</span>
                {cat.name}
              </button>
            ))}
          </div>


          {/* Quick Links */}
          <div style={{ marginTop: '24px' }}>
            <div style={{
              fontSize: '11px',
              fontWeight: '600',
              color: 'var(--text-muted)',
              marginBottom: '12px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Quick Links
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <a href="#" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)', textDecoration: 'none' }}>
                <span>📄</span> Documentation <Icons.ExternalLink size={12} />
              </a>
              <a href="https://github.com/priyam-03/Capstone" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)', textDecoration: 'none' }}>
                <Icons.GitHub size={14} /> Project Repo <Icons.ExternalLink size={12} />
              </a>
              <a href="#" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)', textDecoration: 'none' }}>
                <span>🖥️</span> UI Repo <Icons.ExternalLink size={12} />
              </a>
              <a href="#" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)', textDecoration: 'none' }}>
                <span>📊</span> Slide Deck <Icons.ExternalLink size={12} />
              </a>
            </div>
          </div>

          {/* Team */}
          <div style={{ marginTop: '24px' }}>
            <div style={{
              fontSize: '11px',
              fontWeight: '600',
              color: 'var(--text-muted)',
              marginBottom: '12px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Team
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { name: 'Priyam Deepak Choksi', photo: '/priyam.jpg', github: 'https://github.com/priyam-choksi', linkedin: 'https://www.linkedin.com/in/choksipriyam/' },
                { name: 'Vishodhan Krishnan', photo: '/vishodhan.jpg', github: 'https://github.com/vishodhan-krishnan', linkedin: 'https://www.linkedin.com/in/vishodhankrishnan/' }
              ].map((member, i) => (
                <div key={i} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '8px 10px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '8px'
                }}>
                  <img
                    src={member.photo}
                    alt={member.name}
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      objectFit: 'cover'
                    }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '12px', fontWeight: '500', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {member.name}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    <a href={member.github} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)' }}>
                      <Icons.GitHub size={14} />
                    </a>
                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)' }}>
                      <Icons.LinkedIn size={14} />
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Project Info */}
          <div style={{
            marginTop: '24px',
            padding: '14px',
            background: 'var(--bg-tertiary)',
            borderRadius: '10px',
            border: '1px solid var(--border-primary)'
          }}>
            <div style={{
              fontSize: '11px',
              fontWeight: '600',
              color: 'var(--text-muted)',
              marginBottom: '8px'
            }}>
              PROJECT INFO
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              <div><strong>University:</strong> Northeastern</div>
              <div><strong>Course:</strong> DAMG 7374 - Fall 2025</div>
              <div><strong>Professor:</strong> Kishore Aradhya </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
          <div style={{ maxWidth: '800px' }}>
            {/* Search Results Header */}
            {searchQuery ? (
              <div style={{ marginBottom: '20px' }}>
                <h2 style={{
                  margin: 0,
                  fontSize: '16px',
                  fontWeight: '600',
                  color: 'var(--text-primary)'
                }}>
                  Search Results for "{searchQuery}"
                </h2>
                <p style={{
                  margin: '4px 0 0',
                  fontSize: '13px',
                  color: 'var(--text-muted)'
                }}>
                  {filteredFAQs.length} result{filteredFAQs.length !== 1 ? 's' : ''} found
                </p>
              </div>
            ) : (
              <div style={{ marginBottom: '20px' }}>
                <h2 style={{
                  margin: 0,
                  fontSize: '20px',
                  fontWeight: '700',
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <span style={{ fontSize: '24px' }}>{currentCategory?.icon}</span>
                  {currentCategory?.name}
                </h2>
                <p style={{
                  margin: '4px 0 0',
                  fontSize: '13px',
                  color: 'var(--text-muted)'
                }}>
                  {currentCategory?.faqs.length} questions in this category
                </p>
              </div>
            )}

            {/* FAQ List */}
            <div>
              {filteredFAQs.length === 0 ? (
                <div style={{
                  textAlign: 'center',
                  padding: '40px',
                  color: 'var(--text-muted)'
                }}>
                  <p style={{ fontSize: '14px' }}>No results found for "{searchQuery}"</p>
                  <p style={{ fontSize: '12px' }}>Try different keywords or browse categories</p>
                </div>
              ) : (
                filteredFAQs.map((faq, i) => (
                  <div key={i}>
                    {searchQuery && faq.category && i === 0 ||
                      (searchQuery && filteredFAQs[i - 1]?.category !== faq.category) ? (
                      <div style={{
                        fontSize: '11px',
                        fontWeight: '600',
                        color: 'var(--text-muted)',
                        marginTop: i > 0 ? '20px' : '0',
                        marginBottom: '8px',
                        textTransform: 'uppercase'
                      }}>
                        {faq.category}
                      </div>
                    ) : null}
                    <FAQItem
                      faq={faq}
                      isOpen={openFAQ === `${selectedCategory}-${i}` || openFAQ === `search-${i}`}
                      onToggle={() => setOpenFAQ(
                        openFAQ === `${selectedCategory}-${i}` || openFAQ === `search-${i}`
                          ? null
                          : searchQuery ? `search-${i}` : `${selectedCategory}-${i}`
                      )}
                    />
                  </div>
                ))
              )}
            </div>

            {/* Help Card */}
            {!searchQuery && (
              <div style={{
                marginTop: '32px',
                padding: '20px',
                background: 'var(--bg-tertiary)',
                borderRadius: '14px',
                border: '1px solid var(--border-primary)'
              }}>
                <h3 style={{ margin: '0 0 8px', fontSize: '16px', fontWeight: '600', color: 'var(--text-primary)' }}>
                  Want to learn more?
                </h3>
                <p style={{ margin: '0 0 16px', fontSize: '13px', color: 'var(--text-muted)' }}>
                  Check out the GitHub repository for source code, sample data, and detailed documentation.
                </p>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <a
                    href="https://github.com/priyam-choksi/matsmatsmats"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      padding: '10px 16px',
                      background: 'var(--accent-primary)',
                      color: 'var(--bg-card)',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '13px',
                      fontWeight: '600',
                      cursor: 'pointer',
                      textDecoration: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                  >
                    <Icons.GitHub size={14} />
                    View on GitHub
                  </a>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}