"""
Macro Economic Analysis Agent - Enhanced with Intelligent Tool Calling
Comprehensive macroeconomic and market analysis with adaptive data gathering

MODIFIED: Now supports historical backtesting via analysis_date parameter

Usage: 
  python macro_agent.py --sector technology --days 7
  python macro_agent.py --days 7 --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI

# FIX: Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class MacroAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-nano",
                 analysis_date: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # NEW: Historical backtesting support
        self.analysis_date = analysis_date
        
        # Log mode
        if self.analysis_date:
            print(f"[MACRO] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[MACRO] Running in LIVE mode (current data)")
        
        # Enhanced system prompt with reference-level detail
        self.system_prompt = """You are an expert macroeconomic analyst evaluating market conditions for trading decisions.

**YOUR ROLE:**
Analyze macroeconomic factors and market conditions that could drive price movements over the next 3-6 months. Use the available data-gathering tools to build a comprehensive view, then synthesize findings into actionable trading insights.

**AVAILABLE DATA TOOLS:**
When you need data, request these tools by name:
- get_market_indicators: Fetch major indices (S&P 500, NASDAQ, VIX, Treasury yields)
- get_sector_performance: Analyze sector ETF performance and rotation patterns  
- get_economic_indicators: Get Dollar, Gold, Oil, Bitcoin, Bond data
- get_market_breadth: Compare small cap vs large cap performance

**YOUR ANALYSIS FRAMEWORK:**

1. **Market Regime Identification:**
   - Bull Market: Indices trending up, low VIX (<15), positive breadth
   - Bear Market: Indices down, high VIX (>25), negative breadth
   - Volatile: High VIX, large swings, no clear trend
   - Rotation: Sector leadership changing, mixed performance

2. **Risk Sentiment Assessment:**
   - Small Caps vs Large Caps: Russell 2000 vs S&P 500 spread
     - Small cap outperformance = RISK-ON (bullish for growth)
     - Large cap outperformance = RISK-OFF (flight to quality)
   - VIX Levels: <15 (low risk), 15-25 (normal), >25 (high risk)
   - High Yield Bonds: Rising = risk appetite, Falling = risk aversion
   - Bitcoin: Leading indicator of risk appetite
   - Dollar: Rising = defensive, Falling = risk-on

3. **Sector Rotation Analysis:**
   - Leading sectors = institutional money flow (favor these)
   - Lagging sectors = rotation away (avoid these)
   - Cyclical vs Defensive performance:
     - Cyclical leading (Tech, Consumer Disc, Industrials) = RISK-ON
     - Defensive leading (Utilities, Staples, Healthcare) = RISK-OFF

4. **Interest Rate Environment:**
   - 10-Year Treasury trends: Rising = headwind for growth, Falling = tailwind
   - Impact on valuations and sector preferences

5. **Economic Indicators:**
   - Gold: Rising = fear/inflation, Falling = confidence
   - Oil: Rising = growth/inflation, Falling = slowdown
   - Treasuries: Flight to safety or growth optimism?

**DECISION CRITERIA:**

**RISK-ON Environment (Bullish):**
- VIX <15, small caps outperforming, cyclicals leading
- Favor: Technology, Consumer Discretionary, Industrials
- Strategy: Higher position sizes, longer timeframes

**RISK-OFF Environment (Defensive):**
- VIX >25, large cap defensive, treasuries rallying
- Favor: Utilities, Consumer Staples, Healthcare
- Strategy: Reduced positions, tight stops

**NEUTRAL (Mixed Signals):**
- VIX 15-25, unclear sector leadership, choppy action
- Favor: Quality large caps, dividend payers
- Strategy: Moderate positions, diversification

**CRITICAL OUTPUT REQUIREMENTS:**

After gathering and analyzing data, provide:

## Macro Environment Summary
[2-3 sentence overview of market regime]

## Key Findings
- Market Regime & Indices performance
- Risk Sentiment (RISK-ON/RISK-OFF/NEUTRAL)
- Sector Rotation patterns
- Interest Rate & Volatility environment
- Economic Indicator signals

## Trading Implications
**Current Regime:** [Bull/Bear/Volatile/Rotation]
**Recommended Sectors:** [List 2-3]
**Avoid Sectors:** [List 1-2]  
**Position Sizing:** [Aggressive/Normal/Defensive]
**Key Risks:** [List 2-3]

RECOMMENDATION: RISK-ON/RISK-OFF/NEUTRAL - Confidence: High/Medium/Low

Be quantitative - reference actual numbers and percentages from the data."""

    def _fetch_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Helper method to fetch data with historical date support.
        Used by all data-fetching methods.
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if self.analysis_date:
                # === HISTORICAL MODE ===
                end_date = datetime.strptime(self.analysis_date, '%Y-%m-%d')
                start_date = end_date - timedelta(days=days * 2)
                
                hist = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
                )
            else:
                # === CURRENT MODE ===
                period = "7d" if days <= 7 else "1mo" if days <= 30 else "3mo"
                hist = ticker.history(period=period)
            
            return hist if not hist.empty else None
            
        except Exception as e:
            print(f"[MACRO] ⚠️ Error fetching {symbol}: {e}")
            return None

    def get_market_indicators(self, days: int = 7) -> str:
        """
        Tool: Gather major market indicators
        Returns formatted analysis string
        """
        print(f"[MACRO] 🔧 Tool: get_market_indicators (days={days})")
        
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones', 
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX (Volatility)',
            '^TNX': '10-Year Treasury',
            '^RUT': 'Russell 2000'
        }
        
        result = f"## Market Indicators ({days}-Day Analysis)\n\n"
        
        # NEW: Add date note if historical
        if self.analysis_date:
            result += f"**⚠️ Historical Data as of: {self.analysis_date}**\n\n"
        
        successful = 0
        
        for symbol, name in indices.items():
            try:
                # Use helper method for historical support
                hist = self._fetch_historical_data(symbol, days)
                
                if hist is None or len(hist) < 2:
                    continue
                
                current = hist['Close'].iloc[-1]
                start = hist['Close'].iloc[0]
                change = ((current/start - 1) * 100)
                
                # Volatility
                returns = hist['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252) * 100
                
                # Trend
                if len(hist) >= 3:
                    x = np.arange(len(hist))
                    slope = np.polyfit(x, hist['Close'].values, 1)[0]
                    trend_strength = (slope / start) * 100 * len(hist)
                    
                    if trend_strength > 1:
                        trend = "Strong Uptrend"
                    elif trend_strength > 0.3:
                        trend = "Uptrend"
                    elif trend_strength < -1:
                        trend = "Strong Downtrend"
                    elif trend_strength < -0.3:
                        trend = "Downtrend"
                    else:
                        trend = "Sideways"
                else:
                    trend = "N/A"
                
                result += f"**{name}:** {current:.2f} | {change:+.2f}% | Vol: {vol:.1f}% | {trend}\n"
                successful += 1
                
            except Exception as e:
                result += f"**{name}:** Data unavailable\n"
        
        print(f"[MACRO] ✓ Fetched {successful}/{len(indices)} indicators")
        return result + "\n"

    def get_sector_performance(self, days: int = 7) -> str:
        """
        Tool: Analyze sector ETF performance
        Returns formatted analysis string
        """
        print(f"[MACRO] 🔧 Tool: get_sector_performance (days={days})")
        
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        sector_data = {}
        
        for symbol, name in sectors.items():
            try:
                # Use helper method for historical support
                hist = self._fetch_historical_data(symbol, days)
                
                if hist is None or len(hist) < 2:
                    continue
                
                current = hist['Close'].iloc[-1]
                start = hist['Close'].iloc[0]
                change = ((current/start - 1) * 100)
                
                # Momentum
                if len(hist) >= 4:
                    recent = hist['Close'].iloc[-2:].mean()
                    older = hist['Close'].iloc[:-2].mean()
                    momentum = ((recent/older - 1) * 100)
                else:
                    momentum = change
                
                sector_data[name] = {'change': change, 'momentum': momentum}
                
            except:
                continue
        
        # Sort and format
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['change'], reverse=True)
        
        result = f"## Sector Performance ({days}-Day)\n\n"
        
        # NEW: Add date note if historical
        if self.analysis_date:
            result += f"**⚠️ Historical Data as of: {self.analysis_date}**\n\n"
        
        result += "**Leaders:**\n"
        for name, data in sorted_sectors[:3]:
            result += f"- {name}: {data['change']:+.2f}% (Momentum: {data['momentum']:+.2f}%)\n"
        
        result += "\n**Laggards:**\n"
        for name, data in sorted_sectors[-3:]:
            result += f"- {name}: {data['change']:+.2f}% (Momentum: {data['momentum']:+.2f}%)\n"
        
        # Rotation analysis
        cyclical = ['Technology', 'Consumer Discretionary', 'Financials', 'Industrials']
        defensive = ['Utilities', 'Consumer Staples', 'Healthcare']
        
        cyclical_avg = np.mean([sector_data[s]['change'] for s in cyclical if s in sector_data])
        defensive_avg = np.mean([sector_data[s]['change'] for s in defensive if s in sector_data])
        
        result += f"\n**Rotation:** "
        if cyclical_avg > defensive_avg + 1:
            result += f"Cyclicals leading ({cyclical_avg:+.2f}% vs {defensive_avg:+.2f}%) → RISK-ON ✓\n"
        elif defensive_avg > cyclical_avg + 1:
            result += f"Defensives leading ({defensive_avg:+.2f}% vs {cyclical_avg:+.2f}%) → RISK-OFF ⚠️\n"
        else:
            result += f"Mixed ({cyclical_avg:+.2f}% vs {defensive_avg:+.2f}%) → NEUTRAL\n"
        
        print(f"[MACRO] ✓ Analyzed {len(sector_data)}/{len(sectors)} sectors")
        return result + "\n"

    def get_economic_indicators(self, days: int = 7) -> str:
        """
        Tool: Get key economic indicators
        Returns formatted analysis string
        """
        print(f"[MACRO] 🔧 Tool: get_economic_indicators (days={days})")
        
        indicators = {
            'DX-Y.NYB': ('Dollar Index', 'defensive'),
            'GC=F': ('Gold', 'defensive'),
            'CL=F': ('Crude Oil', 'cyclical'),
            'BTC-USD': ('Bitcoin', 'risk-on'),
            'HYG': ('High Yield Bonds', 'risk-on'),
            'TLT': ('20Y Treasury', 'defensive')
        }
        
        result = f"## Economic Indicators ({days}-Day)\n\n"
        
        # NEW: Add date note if historical
        if self.analysis_date:
            result += f"**⚠️ Historical Data as of: {self.analysis_date}**\n\n"
        
        for symbol, (name, sentiment) in indicators.items():
            try:
                # Use helper method for historical support
                hist = self._fetch_historical_data(symbol, days)
                
                if hist is None or len(hist) < 2:
                    continue
                
                current = hist['Close'].iloc[-1]
                start = hist['Close'].iloc[0]
                change = ((current/start - 1) * 100)
                
                result += f"**{name}:** ${current:.2f} | {change:+.2f}% "
                
                # Interpretation
                if sentiment == 'risk-on' and change > 2:
                    result += "→ Risk appetite ✓\n"
                elif sentiment == 'risk-on' and change < -2:
                    result += "→ Risk aversion ⚠️\n"
                elif sentiment == 'defensive' and change > 2:
                    result += "→ Defensive positioning ⚠️\n"
                elif sentiment == 'defensive' and change < -2:
                    result += "→ Risk appetite ✓\n"
                else:
                    result += "→ Neutral\n"
                
            except:
                continue
        
        print(f"[MACRO] ✓ Economic indicators retrieved")
        return result + "\n"

    def get_market_breadth(self, days: int = 7) -> str:
        """
        Tool: Analyze market breadth (small vs large cap)
        Returns formatted analysis string
        """
        print(f"[MACRO] 🔧 Tool: get_market_breadth (days={days})")
        
        result = f"## Market Breadth ({days}-Day)\n\n"
        
        # NEW: Add date note if historical
        if self.analysis_date:
            result += f"**⚠️ Historical Data as of: {self.analysis_date}**\n\n"
        
        try:
            # Use helper method for historical support
            spy_hist = self._fetch_historical_data('^GSPC', days)
            iwm_hist = self._fetch_historical_data('^RUT', days)
            
            if spy_hist is not None and iwm_hist is not None and len(spy_hist) >= 2 and len(iwm_hist) >= 2:
                spy_change = ((spy_hist['Close'].iloc[-1]/spy_hist['Close'].iloc[0] - 1) * 100)
                iwm_change = ((iwm_hist['Close'].iloc[-1]/iwm_hist['Close'].iloc[0] - 1) * 100)
                spread = iwm_change - spy_change
                
                result += f"**Small vs Large Cap:**\n"
                result += f"- S&P 500: {spy_change:+.2f}%\n"
                result += f"- Russell 2000: {iwm_change:+.2f}%\n"
                result += f"- Spread: {spread:+.2f}%\n\n"
                
                if spread > 1:
                    result += "**Signal:** Small caps outperforming → RISK-ON ✓\n"
                elif spread < -1:
                    result += "**Signal:** Large caps outperforming → RISK-OFF / Flight to quality ⚠️\n"
                else:
                    result += "**Signal:** Balanced breadth → NEUTRAL\n"
            
            print(f"[MACRO] ✓ Breadth analysis complete")
            
        except Exception as e:
            result += f"Breadth data unavailable: {str(e)}\n"
            print(f"[MACRO] ⚠️ Breadth analysis failed")
        
        return result + "\n"

    def analyze_with_llm_iterative(self, days: int = 7, sector: Optional[str] = None) -> str:
        """
        Use iterative tool calling pattern from reference code
        LLM decides which tools to call and when to stop
        """
        if not self.client:
            print("[MACRO] ⚠️ No API key - gathering all data for fallback")
            return self._run_without_llm(days, sector)
        
        print(f"[MACRO] Starting iterative analysis...")
        
        # Tool registry for LLM to call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_market_indicators",
                    "description": "Get major market indicators (S&P 500, NASDAQ, VIX, Treasury yields, Russell 2000) with trend analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "Number of days for analysis", "default": days}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_sector_performance",
                    "description": "Analyze sector ETF performance and identify rotation patterns (cyclical vs defensive)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "Number of days for analysis", "default": days}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_economic_indicators",
                    "description": "Get key economic indicators (Dollar, Gold, Oil, Bitcoin, Bonds) with risk sentiment analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "Number of days for analysis", "default": days}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_breadth",
                    "description": "Compare small cap vs large cap performance to assess risk sentiment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "Number of days for analysis", "default": days}
                        }
                    }
                }
            }
        ]
        
        # NEW: Add date context to initial message
        date_context = ""
        if self.analysis_date:
            date_context = f" **HISTORICAL ANALYSIS AS OF {self.analysis_date}** - All data is historical ending on this date."
        
        # Initial message
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Analyze current macro environment ({days}-day view){date_context}" + (f" with focus on {sector} sector" if sector else "")}
        ]
        
        # Iterative tool calling (like reference code)
        max_iterations = 10
        iteration = 0
        tool_failures = []
        successful_tools = []
        
        try:
            while iteration < max_iterations:
                iteration += 1
                print(f"[MACRO] Iteration {iteration}/{max_iterations}")
                
                # Call LLM
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=0.7,
                    max_tokens=2500
                )
                
                message = response.choices[0].message
                
                # Check if done (no tool calls)
                if not message.tool_calls:
                    print(f"[MACRO] ✓ Analysis complete after {iteration} iterations")
                    
                    # Validate recommendation
                    content = message.content
                    if "RECOMMENDATION:" not in content:
                        content += "\n\nRECOMMENDATION: NEUTRAL - Confidence: Low"
                    
                    return content
                
                # Execute tool calls
                messages.append(message)
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    
                    print(f"[MACRO] 🔧 Executing: {tool_name}")
                    
                    # Execute the tool
                    try:
                        if tool_name == "get_market_indicators":
                            tool_result = self.get_market_indicators(tool_args.get('days', days))
                        elif tool_name == "get_sector_performance":
                            tool_result = self.get_sector_performance(tool_args.get('days', days))
                        elif tool_name == "get_economic_indicators":
                            tool_result = self.get_economic_indicators(tool_args.get('days', days))
                        elif tool_name == "get_market_breadth":
                            tool_result = self.get_market_breadth(tool_args.get('days', days))
                        else:
                            tool_result = f"Unknown tool: {tool_name}"
                            tool_failures.append(tool_name)
                        
                        if "unavailable" not in tool_result.lower() or len(tool_result) > 200:
                            successful_tools.append(tool_name)
                        else:
                            tool_failures.append(tool_name)
                        
                    except Exception as e:
                        tool_result = f"Error in {tool_name}: {str(e)}"
                        tool_failures.append(tool_name)
                        print(f"[MACRO] ⚠️ Tool error: {str(e)}")
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
            
            # Max iterations reached
            print(f"[MACRO] ⚠️ Max iterations reached")
            
            # Get final response
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages + [{"role": "user", "content": "Please provide your final macro analysis and recommendation."}],
                temperature=0.7,
                max_tokens=2000
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            print(f"[MACRO] ❌ Error in iterative analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: gather all data manually
            if successful_tools:
                # We got some data, create manual report
                return self._create_manual_report(days, sector, successful_tools, tool_failures)
            else:
                return self._run_without_llm(days, sector)

    def _run_without_llm(self, days: int, sector: Optional[str]) -> str:
        """Fallback: gather all data without LLM"""
        print("[MACRO] Running fallback mode (no LLM)")
        
        report = f"# Macro Analysis ({days}-Day)\n"
        report += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        
        # NEW: Add historical note
        if self.analysis_date:
            report += f"**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date}**\n"
        
        report += "\n"
        
        report += self.get_market_indicators(days)
        report += self.get_sector_performance(days)
        report += self.get_economic_indicators(days)
        report += self.get_market_breadth(days)
        
        if sector:
            report += f"\n## Sector Focus: {sector.upper()}\n"
            report += "(Sector-specific analysis unavailable in fallback mode)\n\n"
        
        report += "\n## Assessment\n"
        report += "⚠️ **Limited Analysis** - LLM unavailable, showing raw data only\n\n"
        report += "RECOMMENDATION: NEUTRAL - Confidence: Low\n"
        
        return report

    def _create_manual_report(self, days: int, sector: Optional[str], successful: List[str], failed: List[str]) -> str:
        """Create report from partial tool execution"""
        report = f"# Macro Analysis (Partial)\n"
        report += f"*Analysis Period: {days} days*\n"
        
        # NEW: Add historical note
        if self.analysis_date:
            report += f"**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date}**\n"
        
        report += f"*Successful Tools: {len(successful)}, Failed: {len(failed)}*\n\n"
        
        if failed:
            report += f"⚠️ **Note:** Some tools failed ({', '.join(failed)})\n\n"
        
        report += "Raw data gathered successfully:\n\n"
        report += f"Tools executed: {', '.join(successful)}\n\n"
        report += "RECOMMENDATION: NEUTRAL - Confidence: Low (Partial data)\n"
        
        return report

    def run(self, sector: Optional[str] = None, days: int = 7) -> str:
        """
        Execute complete macro analysis with iterative tool calling
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"MACRO ECONOMIC ANALYSIS")
        print(f"Period: {days} days | Sector: {sector or 'All'}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Run iterative analysis
        result = self.analyze_with_llm_iterative(days, sector)
        
        elapsed = time.time() - start_time
        print(f"\n[MACRO] ✓ Complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return result


    def _extract_recommendation_from_content(self, analysis: str) -> Tuple[str, str]:
        """Extract recommendation from LLM response even without formal format"""
        analysis_lower = analysis.lower()
        
        # Agent-specific indicators for macro
        specific_buy_signals = ['risk-on', 'risk on', 'bullish environment', 'favorable macro']
        specific_sell_signals = ['risk-off', 'risk off', 'bearish environment', 'defensive positioning']
        
        # Look for explicit recommendations first
        if any(phrase in analysis_lower for phrase in ["recommend buy", "should buy", "buy signal", "final decision: buy"]):
            return "BUY", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend sell", "should sell", "sell signal", "final decision: sell"]):
            return "SELL", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend hold", "should hold", "wait", "neutral position"]):
            return "HOLD", "Low"
        
        # Check agent-specific signals
        buy_count = sum(1 for signal in specific_buy_signals if signal in analysis_lower)
        sell_count = sum(1 for signal in specific_sell_signals if signal in analysis_lower)
        
        # General sentiment indicators
        general_buy = ["bullish", "positive", "upside", "growth", "strong"]
        general_sell = ["bearish", "negative", "downside", "decline", "weak"]
        
        buy_count += sum(0.5 for word in general_buy if word in analysis_lower)
        sell_count += sum(0.5 for word in general_sell if word in analysis_lower)
        
        # Decision based on signal strength
        if buy_count > sell_count + 1.5:
            confidence = "High" if buy_count > 4 else "Medium"
            return "BUY", confidence
        elif sell_count > buy_count + 1.5:
            confidence = "High" if sell_count > 4 else "Medium"
            return "SELL", confidence
        else:
            # Default to HOLD only if truly neutral
            return "HOLD", "Low"


def main():
    parser = argparse.ArgumentParser(
        description="Macro Economic Analysis Agent - Intelligent iterative analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python macro_agent.py
  python macro_agent.py --days 30
  python macro_agent.py --sector technology
  python macro_agent.py --days 7 --analysis-date 2024-06-15
        """
    )
    
    parser.add_argument("--sector", help="Specific sector (technology, healthcare, energy, etc.)")
    parser.add_argument("--days", type=int, default=7, help="Analysis period (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model (default: gpt-4o-mini)")
    parser.add_argument("--output", help="Save to file")
    # NEW: Historical backtesting support
    parser.add_argument("--analysis-date", type=str, default=None,
                       help="Historical analysis date (YYYY-MM-DD format). If set, fetches data ending on this date.")
    
    args = parser.parse_args()
    
    try:
        agent = MacroAgent(
            api_key=args.api_key, 
            model=args.model,
            analysis_date=args.analysis_date  # NEW
        )
        result = agent.run(sector=args.sector, days=args.days)
        
        print(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()