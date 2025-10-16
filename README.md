# matsmatsmats
# Multi-Agent Trading System

A trading analysis system that uses 12 specialized AI agents working together to analyze stocks and make risk-adjusted trading decisions.

## System Workflow

```
[Market Data] → [4 Analysts] → [Discussion Hub] → [Bull & Bear Researchers]
                                                            ↓
[Final Trade Order] ← [Trader] ← [Risk Manager] ← [Research Manager] ← [3 Risk Debators]
```

The system processes information through five phases:
1. **Analysis** → 2. **Research** → 3. **Risk Evaluation** → 4. **Management** → 5. **Execution**

## Quick Start

The easiest way to run the entire system is with one command:

```bash
cd agents/orchestrators
python master_orchestrator.py AAPL --run-all --portfolio-value 100000
```

This will analyze Apple stock through all phases and produce a complete trading recommendation.

## System Phases

### Phase 1: Market Analysis

This phase uses four analyst agents to gather different types of market data, plus a discussion hub to combine their findings.

**Technical Agent**  
Analyzes price charts, patterns, and technical indicators like RSI, MACD, and moving averages.
```bash
cd agents/analyst
python technical_agent.py AAPL --days 7
```
Options:
- `--days`: Number of days to analyze (default: 7)
- `--output`: Save report to file
- `--api-key`: OpenAI API key (optional)

**News Agent**  
Checks news sources and social media for sentiment about the stock.
```bash
python news_agent.py AAPL --sources yahoo reddit --days 7
```
Options:
- `--sources`: Choose from yahoo, reddit, newsapi, finnhub, alphavantage
- `--days`: How far back to look for news
- `--setup`: Shows API setup instructions

**Fundamental Agent**  
Examines company financials including earnings, valuation ratios, and profit margins.
```bash
python fundamental_agent.py AAPL
```
This agent doesn't need many options as it pulls the latest available financial data automatically.

**Macro Agent**  
Analyzes overall market conditions, sector performance, and economic indicators.
```bash
python macro_agent.py --sector technology --days 7
```
Options:
- `--sector`: Focus on specific sector
- `--days`: Analysis period for market trends

**Discussion Hub**  
After the analysts run, the discussion hub combines all their reports into structured debate points. This is the bridge between raw analysis and research.
```bash
cd agents/orchestrators
python discussion_hub.py AAPL --run-analysts
```
This can either run all analysts automatically or process existing analyst reports.

### Phase 2: Research Deep Dive

Two researchers take opposing views to build comprehensive cases using the analyst data.

**Bull Researcher**  
Builds the bullish case by:
- Synthesizing positive signals from all analysts
- Countering bearish arguments
- Identifying upside catalysts
- Calculating risk/reward ratios

```bash
cd agents/researcher
python bull_researcher.py AAPL --discussion-file ../orchestrators/discussion_points.json
```

**Bear Researcher**  
Builds the bearish case by:
- Identifying all risk factors
- Countering bullish arguments  
- Finding downside triggers
- Suggesting hedging strategies

```bash
python bear_researcher.py AAPL --discussion-file ../orchestrators/discussion_points.json
```

Both researchers save their analysis as JSON files for the next phase to use.

### Phase 3: Risk Evaluation

Three debators with different risk tolerances evaluate the research from their perspectives.

| Debator | Max Position | Risk Tolerance | Philosophy |
|---------|-------------|----------------|------------|
| **Aggressive** | 25% of portfolio | High | "Fortune favors the bold" |
| **Neutral** | 10% of portfolio | Medium | "Balance risk and reward" |
| **Conservative** | 5% of portfolio | Low | "Preserve capital first" |

Each runs the same way but produces different recommendations:
```bash
cd agents/risk_management
python aggressive_debator.py AAPL --bull-file ../researcher/bull_thesis.json --bear-file ../researcher/bear_thesis.json
```

The debators consider:
- Their risk tolerance level
- Position sizing appropriate to their style
- Stop loss and profit targets
- Overall market conditions

### Phase 4: Management Decision

Two managers synthesize all the information and make the final decision.

**Research Manager**  
This agent combines all previous analysis to create:
- Probability assessments (Bull case %, Bear case %, Base case %)
- Consensus analysis across all agents
- Key decision factors
- Final research recommendation

```bash
cd agents/managers
python research_manager.py AAPL
```

**Risk Manager** (Has Veto Power)  
Makes the final risk-adjusted decision considering:
- Portfolio size and current positions
- Risk limits and position sizing
- Stop loss and profit targets
- Can APPROVE, MODIFY, or REJECT any trade

```bash
python risk_manager.py AAPL --portfolio-value 100000
```

The Risk Manager has absolute veto power and will reject trades that exceed risk parameters, regardless of how bullish the research is.

### Phase 5: Trade Execution

**Trader**  
Only executes if the Risk Manager approved. Creates detailed order with:
- Exact entry price and share count
- Stop loss levels (fixed or trailing)
- Three-tier profit targets
- Execution instructions
- Monitoring plan

```bash
cd agents/execution
python trader.py AAPL
```

## Master Orchestrator

Instead of running each agent individually, the master orchestrator manages the entire workflow automatically.

### Available Modes

**Quick Analysis Mode**  
Runs only the first three phases (no trading decision):
```bash
python master_orchestrator.py AAPL --quick
```

**Full Trading Mode**  
Runs all five phases with final trade decision:
```bash
python master_orchestrator.py AAPL --run-all --portfolio-value 100000
```

**Custom Phase Selection**  
Run only specific phases for debugging or partial analysis:
```bash
python master_orchestrator.py AAPL --phases phase1 phase2 phase3
```
- phase1: Analysts + Discussion Hub
- phase2: Bull & Bear Researchers
- phase3: Risk Debators
- phase4: Managers
- phase5: Trader

**Error Handling Options**
- `--stop-on-error`: Stops immediately if any agent fails
- Default behavior: Continues even if some agents fail

## Output Files

All outputs are saved to `outputs/` folder in your project root:

```
TradingAgent/outputs/
├── discussion_points.json    # Analyst consensus (Phase 1)
├── bull_thesis.json          # Bullish research (Phase 2)
├── bear_thesis.json          # Bearish research (Phase 2)
├── aggressive_eval.json      # Risk evaluations (Phase 3)
├── neutral_eval.json
├── conservative_eval.json
├── research_synthesis.json   # Combined research (Phase 4)
├── risk_decision.json        # Final decision (Phase 4)
├── final_order.json          # Trade details (Phase 5)
├── master_summary.txt        # Complete summary
└── execution_log.json        # Detailed execution log
```

## Understanding the Results

The system produces three possible decisions:

**APPROVE**
- Trade is approved as recommended
- Includes exact position size and risk controls
- Ready to execute

**MODIFY**  
- Trade approved but with reduced size
- Usually due to risk concerns
- Still executable but more conservative

**REJECT**
- Trade is too risky or outlook is bearish
- No position will be taken
- Includes reason for rejection

Each approved trade includes:
- Entry price and number of shares
- Stop loss price (usually 3-7% below entry)
- Profit targets (typically at +10%, +20%, +30%)
- Maximum holding period
- Monitoring requirements

## Common Workflows

### Just Want a Quick Opinion
```bash
python master_orchestrator.py AAPL --quick
```
Gets you analysis from analysts, researchers, and risk team without making a trading decision.

### Full Investment Decision
```bash
python master_orchestrator.py AAPL --run-all --portfolio-value 250000
```
Complete analysis with position sizing based on your actual portfolio.

### Testing Single Components
Each agent can run standalone for testing:
```bash
cd agents/analyst
python technical_agent.py AAPL --days 30
```

### Debugging Issues
Run phases one at a time to identify problems:
```bash
python master_orchestrator.py AAPL --phases phase1
# Check outputs/discussion_points.json

python master_orchestrator.py AAPL --phases phase2  
# Check outputs/bull_thesis.json and bear_thesis.json
```

## Requirements

- Python 3.7 or higher
- Required packages: `openai`, `yfinance`, `pandas`, `numpy`
- Optional: OpenAI API key (set as OPENAI_API_KEY environment variable)

All agents work without the OpenAI API key but provide more detailed analysis with it.

## Important Notes

- The system is designed to be conservative by default
- Risk Manager has veto power over all trades
- Position sizes are calculated as percentage of total portfolio
- All decisions include detailed reasoning
- Every run creates timestamped logs for review
- You can run multiple analyses and compare results