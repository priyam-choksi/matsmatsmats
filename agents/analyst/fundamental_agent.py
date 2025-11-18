"""
Fundamental Analysis Agent - Enhanced Version
Provides comprehensive fundamental analysis for trading decisions

Usage: python fundamental_agent.py AAPL --output report.txt
"""

import os
import sys
import json
import argparse
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI

# FIX: Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class FundamentalAgent:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini", use_cache: bool = True):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.use_cache = use_cache
        
        # Setup cache
        self.cache_dir = Path("cache/fundamental")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced system prompt based on reference quality
        self.system_prompt = """You are an expert fundamental analyst focused on identifying fundamental catalysts and factors that could drive price movements over the next 3-6 months.

**YOUR ANALYSIS FRAMEWORK:**

1. **Valuation Assessment:**
   - Is the P/E ratio reasonable compared to growth rate (PEG ratio)?
   - How does Price/Book compare to sector averages?
   - Is the company trading at a premium or discount to historical valuations?
   - What does the EV/Revenue and EV/EBITDA tell us about market pricing?

2. **Growth Analysis:**
   - Are BOTH revenue AND earnings growing? At what rates?
   - Is growth accelerating (good) or decelerating (concerning)?
   - Are margins expanding (excellent) or contracting (warning)?
   - How does growth compare to industry peers?

3. **Profitability Metrics:**
   - Are profit margins healthy for the sector?
   - Is ROE above 15% (generally good) or below 10% (concerning)?
   - Are operating margins improving or deteriorating?
   - Is the company generating strong returns on capital?

4. **Financial Health:**
   - Is Current Ratio > 1.5 (healthy) or < 1.0 (liquidity concerns)?
   - Is Debt/Equity manageable for the industry?
   - Is Free Cash Flow positive and growing?
   - Can the company meet its obligations comfortably?

5. **Earnings Quality:**
   - What's the earnings beat/miss pattern? (3-4 beats = strong, mostly misses = weak)
   - Are earnings surprises getting larger or smaller?
   - Are estimates being revised up (bullish) or down (bearish)?
   - Any red flags in recent quarters (guidance cuts, misses, etc.)?

6. **Analyst Sentiment:**
   - What's the analyst consensus? (Strong Buy, Buy, Hold, Sell)
   - How many analysts cover the stock? (More = higher confidence)
   - What's the price target upside/downside?
   - Are analysts upgrading or downgrading recently?

7. **Catalysts & Events:**
   - Upcoming earnings date and expectations
   - Recent insider buying (bullish) or selling (bearish)?
   - Any upcoming events (product launches, FDA approvals, etc.)?

**DECISION CRITERIA:**

**BUY Signals:**
- Strong valuation (low P/E with high growth, PEG < 1.5)
- Positive earnings trend (3+ quarters of beats)
- Improving margins and growing cash flow
- Bullish analyst sentiment with price target upside > 15%
- Strong financial health (Current Ratio > 1.5, manageable debt)

**SELL Signals:**
- Deteriorating fundamentals (declining growth, shrinking margins)
- Overvaluation (high P/E without growth to justify, PEG > 2.5)
- Consistent earnings misses or negative guidance
- Bearish analyst sentiment with downgrades
- Financial stress (high debt, negative cash flow, liquidity concerns)

**HOLD Signals:**
- Fair valuation with stable fundamentals
- Mixed earnings performance (2 beats, 2 misses)
- Neutral analyst sentiment
- No clear catalysts in near term

**OUTPUT FORMAT:**

Provide your analysis in this structure:

## Fundamental Analysis Summary

**Overall Assessment:** [1-2 sentence summary of the fundamental picture]

## Key Findings

### Valuation
[Analyze P/E, PEG, P/B, EV ratios - are they attractive or expensive?]

### Growth & Profitability  
[Discuss revenue/earnings growth rates, margin trends, ROE]

### Financial Health
[Evaluate balance sheet strength, cash flow, debt levels]

### Earnings Quality
[Analyze beat/miss pattern, consistency, trends]

### Analyst View
[Summarize consensus, price targets, recent changes]

## Investment Thesis

**Bull Case:** [2-3 strongest positive factors]

**Bear Case:** [2-3 key risks or concerns]

**Critical Catalysts:** [Upcoming events or factors that could move the stock]

## Final Recommendation

**Decision:** [Clear reasoning for recommendation]

RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low

**Target Price Range:** [If enough data, suggest reasonable target]
**Time Horizon:** [3-6 months typically]
**Risk Level:** [Low/Medium/High based on volatility and uncertainty]

Be specific and quantitative. Every claim should reference actual numbers from the data provided."""

    def get_comprehensive_fundamentals(self) -> Dict[str, Any]:
        """
        Gather comprehensive fundamental data from yfinance
        Returns dict with all fundamental metrics organized by category
        """
        try:
            print(f"[FUNDAMENTALS] Fetching comprehensive data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                print(f"[FUNDAMENTALS] ⚠️  No data returned for {self.ticker}")
                return None
            
            # Helper for safe value extraction
            def safe_get(key, default="N/A", format_type=None):
                value = info.get(key)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return default
                try:
                    if format_type == "billions":
                        return f"${float(value)/1e9:.2f}B"
                    elif format_type == "millions":
                        return f"${float(value)/1e6:.2f}M"
                    elif format_type == "percentage":
                        return f"{float(value)*100:.1f}%"
                    elif format_type == "currency":
                        return f"${float(value):.2f}"
                    elif format_type == "ratio":
                        return f"{float(value):.2f}"
                    else:
                        return value
                except (ValueError, TypeError):
                    return default
            
            # Organize data into categories
            fundamentals = {
                "company_info": {
                    "name": safe_get('longName', self.ticker),
                    "sector": safe_get('sector', 'Unknown'),
                    "industry": safe_get('industry', 'Unknown'),
                },
                "valuation": {
                    "market_cap": safe_get('marketCap', format_type="billions"),
                    "enterprise_value": safe_get('enterpriseValue', format_type="billions"),
                    "trailing_pe": safe_get('trailingPE', format_type="ratio"),
                    "forward_pe": safe_get('forwardPE', format_type="ratio"),
                    "peg_ratio": safe_get('pegRatio', format_type="ratio"),
                    "price_to_book": safe_get('priceToBook', format_type="ratio"),
                    "price_to_sales": safe_get('priceToSalesTrailing12Months', format_type="ratio"),
                    "ev_to_revenue": safe_get('enterpriseToRevenue', format_type="ratio"),
                    "ev_to_ebitda": safe_get('enterpriseToEbitda', format_type="ratio"),
                },
                "growth": {
                    "revenue_growth": safe_get('revenueGrowth', format_type="percentage"),
                    "earnings_growth": safe_get('earningsGrowth', format_type="percentage"),
                    "quarterly_revenue_growth": safe_get('quarterlyRevenueGrowth', format_type="percentage"),
                    "quarterly_earnings_growth": safe_get('quarterlyEarningsGrowth', format_type="percentage"),
                },
                "profitability": {
                    "profit_margin": safe_get('profitMargins', format_type="percentage"),
                    "operating_margin": safe_get('operatingMargins', format_type="percentage"),
                    "gross_margin": safe_get('grossMargins', format_type="percentage"),
                    "ebitda_margin": safe_get('ebitdaMargins', format_type="percentage"),
                    "roe": safe_get('returnOnEquity', format_type="percentage"),
                    "roa": safe_get('returnOnAssets', format_type="percentage"),
                },
                "financial_health": {
                    "current_ratio": safe_get('currentRatio', format_type="ratio"),
                    "quick_ratio": safe_get('quickRatio', format_type="ratio"),
                    "debt_to_equity": safe_get('debtToEquity', format_type="ratio"),
                    "total_cash": safe_get('totalCash', format_type="billions"),
                    "total_debt": safe_get('totalDebt', format_type="billions"),
                    "free_cash_flow": safe_get('freeCashflow', format_type="billions"),
                    "operating_cash_flow": safe_get('operatingCashflow', format_type="billions"),
                },
                "analyst": {
                    "recommendation": safe_get('recommendationKey', 'none').upper(),
                    "num_analysts": safe_get('numberOfAnalystOpinions', 0),
                    "target_mean": safe_get('targetMeanPrice', format_type="currency"),
                    "target_high": safe_get('targetHighPrice', format_type="currency"),
                    "target_low": safe_get('targetLowPrice', format_type="currency"),
                    "current_price": safe_get('currentPrice', format_type="currency"),
                },
                "price_info": {
                    "current_price": safe_get('currentPrice', format_type="currency"),
                    "fifty_two_week_high": safe_get('fiftyTwoWeekHigh', format_type="currency"),
                    "fifty_two_week_low": safe_get('fiftyTwoWeekLow', format_type="currency"),
                }
            }
            
            # Calculate derived metrics
            if fundamentals["analyst"]["current_price"] != "N/A" and fundamentals["analyst"]["target_mean"] != "N/A":
                try:
                    current = float(fundamentals["analyst"]["current_price"].replace('$', ''))
                    target = float(fundamentals["analyst"]["target_mean"].replace('$', ''))
                    upside = ((target / current) - 1) * 100
                    fundamentals["analyst"]["implied_upside"] = f"{upside:+.1f}%"
                except:
                    fundamentals["analyst"]["implied_upside"] = "N/A"
            else:
                fundamentals["analyst"]["implied_upside"] = "N/A"
            
            print(f"[FUNDAMENTALS] ✓ Comprehensive data retrieved")
            return fundamentals
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ❌ Error fetching fundamentals: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_earnings_analysis(self) -> Dict[str, Any]:
        """
        Analyze recent earnings performance with detailed metrics
        Returns dict with earnings history and analysis
        """
        try:
            print(f"[FUNDAMENTALS] Analyzing earnings history...")
            stock = yf.Ticker(self.ticker)
            earnings = stock.earnings_history
            
            if earnings is None or len(earnings) == 0:
                print(f"[FUNDAMENTALS] ⚠️  No earnings data available")
                return {
                    "available": False,
                    "message": "No earnings history available"
                }
            
            df = pd.DataFrame(earnings)
            recent = df.tail(4)  # Last 4 quarters
            
            # Calculate metrics
            beats = 0
            misses = 0
            surprises = []
            
            quarters_data = []
            
            for idx, row in recent.iterrows():
                actual = row.get('epsActual', 0)
                estimate = row.get('epsEstimate', 0)
                period = row.get('period', 'Unknown')
                
                if estimate != 0:
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                    surprises.append(surprise_pct)
                else:
                    surprise_pct = 0
                
                beat = actual >= estimate
                if beat:
                    beats += 1
                else:
                    misses += 1
                
                quarters_data.append({
                    "period": period,
                    "actual": actual,
                    "estimate": estimate,
                    "surprise_pct": surprise_pct,
                    "beat": beat
                })
            
            # Calculate summary stats
            avg_surprise = np.mean(surprises) if surprises else 0
            beat_rate = (beats / len(recent)) * 100 if len(recent) > 0 else 0
            
            # Determine trend
            if len(surprises) >= 3:
                recent_trend = np.mean(surprises[-2:]) - np.mean(surprises[:2])
                if recent_trend > 2:
                    trend = "Improving"
                elif recent_trend < -2:
                    trend = "Deteriorating"
                else:
                    trend = "Stable"
            else:
                trend = "Unknown"
            
            # Assess quality
            if beats >= 3:
                quality = "Strong"
            elif beats == 2:
                quality = "Mixed"
            else:
                quality = "Weak"
            
            print(f"[FUNDAMENTALS] ✓ Earnings analysis complete ({beats}/4 beats)")
            
            return {
                "available": True,
                "quarters": quarters_data,
                "summary": {
                    "beats": beats,
                    "misses": misses,
                    "beat_rate": beat_rate,
                    "avg_surprise": avg_surprise,
                    "trend": trend,
                    "quality": quality
                }
            }
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ⚠️  Error analyzing earnings: {e}")
            return {
                "available": False,
                "message": f"Error: {str(e)}"
            }

    def get_sec_filing_data(self) -> Dict[str, Any]:
        """
        Fetch data from SEC 10-K filings with intelligent caching
        Returns dict with SEC filing information
        """
        # Check cache first (10-K data doesn't change frequently)
        if self.use_cache:
            cached = self._get_cached_sec_data()
            if cached:
                print(f"[FUNDAMENTALS] ✓ Using cached SEC data")
                return cached
        
        print(f"[FUNDAMENTALS] Fetching SEC 10-K filing data...")
        headers = {'User-Agent': "fundamental-agent/1.0"}
        
        try:
            # Get company CIK from SEC
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                return {"available": False, "error": "SEC API unavailable"}
            
            companies = pd.DataFrame.from_dict(response.json(), orient='index')
            ticker_match = companies[companies['ticker'] == self.ticker]
            
            if ticker_match.empty:
                return {"available": False, "error": f"{self.ticker} not in SEC database"}
            
            cik = str(ticker_match['cik_str'].values[0]).zfill(10)
            
            # Fetch assets data from latest 10-K
            assets_response = requests.get(
                f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json',
                headers=headers,
                timeout=10
            )
            
            if assets_response.status_code != 200:
                return {"available": False, "error": "Assets data unavailable"}
            
            assets_data = pd.DataFrame.from_dict(
                assets_response.json()['units']['USD']
            )
            
            # Filter for 10-K filings only
            assets_10k = assets_data[assets_data['form'] == '10-K']
            
            if assets_10k.empty:
                return {"available": False, "error": "No 10-K filings found"}
            
            # Get latest filing
            assets_10k = assets_10k.sort_values('filed', ascending=False).reset_index(drop=True)
            latest = assets_10k.iloc[0]
            
            sec_data = {
                "available": True,
                "cik": cik,
                "total_assets": latest['val'],
                "total_assets_formatted": f"${latest['val']/1e9:.2f}B",
                "filing_date": latest['filed'],
                "fiscal_year": latest.get('fy', 'Unknown'),
                "form": latest['form']
            }
            
            # Cache the result
            if self.use_cache:
                self._cache_sec_data(sec_data)
            
            print(f"[FUNDAMENTALS] ✓ SEC data retrieved (Filed: {latest['filed']})")
            return sec_data
            
        except requests.Timeout:
            print(f"[FUNDAMENTALS] ⚠️  SEC API timeout")
            return {"available": False, "error": "Request timeout"}
        except Exception as e:
            print(f"[FUNDAMENTALS] ⚠️  SEC error: {str(e)}")
            return {"available": False, "error": str(e)}

    def _get_cached_sec_data(self) -> Optional[Dict]:
        """Load cached SEC data if less than 90 days old"""
        cache_file = self.cache_dir / f"{self.ticker}_sec.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            
            cache_age = datetime.now() - cached['timestamp']
            
            # 10-K data is only updated annually, so 90 day cache is reasonable
            if cache_age < timedelta(days=90):
                data = cached['data']
                data['cache_age_days'] = cache_age.days
                return data
            
            return None
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ⚠️  Cache read error: {e}")
            return None

    def _cache_sec_data(self, data: Dict):
        """Save SEC data to cache with timestamp"""
        cache_file = self.cache_dir / f"{self.ticker}_sec.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'data': data
                }, f)
        except Exception as e:
            print(f"[FUNDAMENTALS] ⚠️  Cache write failed: {e}")

    def format_data_for_analysis(self, fundamentals: Dict, earnings: Dict, sec_data: Dict) -> str:
        """
        Format all gathered data into a structured report for LLM analysis
        Returns formatted string ready for LLM consumption
        """
        report = f"""# Fundamental Analysis Data for {self.ticker}

## Company Information
- Name: {fundamentals['company_info']['name']}
- Sector: {fundamentals['company_info']['sector']}
- Industry: {fundamentals['company_info']['industry']}

## Valuation Metrics
- Market Cap: {fundamentals['valuation']['market_cap']}
- Enterprise Value: {fundamentals['valuation']['enterprise_value']}
- P/E Ratio (Trailing): {fundamentals['valuation']['trailing_pe']}
- P/E Ratio (Forward): {fundamentals['valuation']['forward_pe']}
- PEG Ratio: {fundamentals['valuation']['peg_ratio']}
- Price/Book: {fundamentals['valuation']['price_to_book']}
- Price/Sales: {fundamentals['valuation']['price_to_sales']}
- EV/Revenue: {fundamentals['valuation']['ev_to_revenue']}
- EV/EBITDA: {fundamentals['valuation']['ev_to_ebitda']}

## Growth Metrics
- Revenue Growth (YoY): {fundamentals['growth']['revenue_growth']}
- Earnings Growth (YoY): {fundamentals['growth']['earnings_growth']}
- Quarterly Revenue Growth: {fundamentals['growth']['quarterly_revenue_growth']}
- Quarterly Earnings Growth: {fundamentals['growth']['quarterly_earnings_growth']}

## Profitability Metrics
- Profit Margin: {fundamentals['profitability']['profit_margin']}
- Operating Margin: {fundamentals['profitability']['operating_margin']}
- Gross Margin: {fundamentals['profitability']['gross_margin']}
- EBITDA Margin: {fundamentals['profitability']['ebitda_margin']}
- Return on Equity (ROE): {fundamentals['profitability']['roe']}
- Return on Assets (ROA): {fundamentals['profitability']['roa']}

## Financial Health
- Current Ratio: {fundamentals['financial_health']['current_ratio']}
- Quick Ratio: {fundamentals['financial_health']['quick_ratio']}
- Debt/Equity: {fundamentals['financial_health']['debt_to_equity']}
- Total Cash: {fundamentals['financial_health']['total_cash']}
- Total Debt: {fundamentals['financial_health']['total_debt']}
- Free Cash Flow: {fundamentals['financial_health']['free_cash_flow']}
- Operating Cash Flow: {fundamentals['financial_health']['operating_cash_flow']}

## Price Information
- Current Price: {fundamentals['price_info']['current_price']}
- 52-Week High: {fundamentals['price_info']['fifty_two_week_high']}
- 52-Week Low: {fundamentals['price_info']['fifty_two_week_low']}

## Analyst Consensus
- Recommendation: {fundamentals['analyst']['recommendation']}
- Number of Analysts: {fundamentals['analyst']['num_analysts']}
- Target Price (Mean): {fundamentals['analyst']['target_mean']}
- Target Price (High): {fundamentals['analyst']['target_high']}
- Target Price (Low): {fundamentals['analyst']['target_low']}
- Implied Upside/Downside: {fundamentals['analyst']['implied_upside']}

"""
        
        # Add earnings history
        if earnings['available']:
            report += "## Earnings History (Last 4 Quarters)\n\n"
            for q in earnings['quarters']:
                symbol = "✓" if q['beat'] else "✗"
                report += f"- {symbol} {q['period']}: Actual ${q['actual']:.2f} vs Est ${q['estimate']:.2f} ({q['surprise_pct']:+.1f}%)\n"
            
            report += f"\n**Summary:**\n"
            report += f"- Beat Rate: {earnings['summary']['beats']}/{earnings['summary']['beats'] + earnings['summary']['misses']} ({earnings['summary']['beat_rate']:.0f}%)\n"
            report += f"- Average Surprise: {earnings['summary']['avg_surprise']:+.1f}%\n"
            report += f"- Trend: {earnings['summary']['trend']}\n"
            report += f"- Quality Assessment: {earnings['summary']['quality']}\n\n"
        else:
            report += "## Earnings History\n- No earnings data available\n\n"
        
        # Add SEC filing data
        if sec_data['available']:
            report += "## SEC 10-K Filing Data\n"
            report += f"- Total Assets: {sec_data['total_assets_formatted']}\n"
            report += f"- Filing Date: {sec_data['filing_date']}\n"
            report += f"- Fiscal Year: {sec_data['fiscal_year']}\n"
            if 'cache_age_days' in sec_data:
                report += f"- Data Age: {sec_data['cache_age_days']} days (cached)\n"
            report += "\n"
        else:
            report += f"## SEC 10-K Filing Data\n- Status: {sec_data.get('error', 'Unavailable')}\n\n"
        
        return report

    def analyze_with_llm(self, formatted_data: str) -> str:
        """
        Send formatted data to LLM for comprehensive analysis
        Returns detailed fundamental analysis report
        """
        if not self.client:
            print("[FUNDAMENTALS] ⚠️  No OpenAI API key - using fallback analysis")
            return self._create_fallback_analysis(formatted_data)
        
        try:
            print(f"[FUNDAMENTALS] Generating analysis with {self.model}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Please provide a comprehensive fundamental analysis for {self.ticker} based on this data:\n\n{formatted_data}"}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            analysis = response.choices[0].message.content
            
            # Validate response has required recommendation
            if "RECOMMENDATION:" not in analysis:
                print("[FUNDAMENTALS] ⚠️  Response missing recommendation, appending...")
                analysis += "\n\nRECOMMENDATION: HOLD - Confidence: Low\n(Note: Analysis generated but recommendation format was missing)"
            
            print(f"[FUNDAMENTALS] ✓ Analysis generated ({len(analysis)} chars)")
            return analysis
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ❌ LLM analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(formatted_data)

    def _create_fallback_analysis(self, formatted_data: str) -> str:
        """
        Create basic rule-based analysis when LLM unavailable
        Not as sophisticated but provides baseline assessment
        """
        print("[FUNDAMENTALS] Creating fallback analysis...")
        
        report = f"""# Fundamental Analysis Report: {self.ticker}
*Generated using fallback analysis (LLM unavailable)*

---

{formatted_data}

---

## Automated Assessment

**Note:** This is a basic automated analysis. Full LLM analysis unavailable.

"""
        
        # Simple signal detection
        data_lower = formatted_data.lower()
        
        positive_signals = [
            "revenue growth" in data_lower and "n/a" not in formatted_data[:500],
            data_lower.count("✓") >= 3,  # 3+ earnings beats
            "strong buy" in data_lower or "buy" in data_lower,
        ]
        
        negative_signals = [
            data_lower.count("✗") >= 3,  # 3+ earnings misses
            "sell" in data_lower,
            "debt" in data_lower and "high" in data_lower,
        ]
        
        pos_count = sum(positive_signals)
        neg_count = sum(negative_signals)
        
        if pos_count > neg_count and pos_count >= 2:
            recommendation = "BUY"
            confidence = "Medium"
            reasoning = f"Detected {pos_count} positive signals vs {neg_count} negative signals"
        elif neg_count > pos_count and neg_count >= 2:
            recommendation = "SELL"
            confidence = "Medium"
            reasoning = f"Detected {neg_count} negative signals vs {pos_count} positive signals"
        else:
            recommendation = "HOLD"
            confidence = "Low"
            reasoning = f"Mixed signals ({pos_count} positive, {neg_count} negative)"
        
        report += f"**Reasoning:** {reasoning}\n\n"
        report += f"RECOMMENDATION: {recommendation} - Confidence: {confidence}\n"
        
        return report

    def run(self) -> str:
        """
        Execute complete fundamental analysis workflow
        Returns comprehensive analysis report
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"FUNDAMENTAL ANALYSIS: {self.ticker}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Step 1: Gather comprehensive fundamentals
        fundamentals = self.get_comprehensive_fundamentals()
        if not fundamentals:
            error_report = f"""# Fundamental Analysis Error

**Ticker:** {self.ticker}
**Error:** Unable to retrieve fundamental data

**Possible Causes:**
- Invalid ticker symbol
- Company not publicly traded  
- Network connectivity issues
- yfinance API temporarily unavailable

**Recommendation:** Please verify the ticker symbol and try again.

RECOMMENDATION: HOLD - Confidence: N/A
"""
            print(f"[FUNDAMENTALS] ❌ Fatal: No data available for {self.ticker}")
            return error_report
        
        # Step 2: Get earnings analysis
        earnings = self.get_earnings_analysis()
        
        # Step 3: Get SEC filing data
        sec_data = self.get_sec_filing_data()
        
        # Step 4: Format all data
        print(f"[FUNDAMENTALS] Formatting data for analysis...")
        formatted_data = self.format_data_for_analysis(fundamentals, earnings, sec_data)
        
        # Step 5: Generate LLM analysis
        analysis = self.analyze_with_llm(formatted_data)
        
        elapsed_time = time.time() - start_time
        print(f"\n[FUNDAMENTALS] ✓ Analysis complete in {elapsed_time:.2f}s")
        print(f"{'='*70}\n")
        
        return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Fundamental Analysis Agent - Comprehensive fundamental analysis for trading decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fundamental_agent.py AAPL
  python fundamental_agent.py MSFT --output msft_report.txt
  python fundamental_agent.py GOOGL --no-cache --model gpt-4
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--output", help="Save analysis report to file")
    parser.add_argument("--no-cache", action="store_true", help="Disable SEC data caching (fetch fresh data)")
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = FundamentalAgent(
            ticker=args.ticker,
            api_key=args.api_key,
            model=args.model,
            use_cache=not args.no_cache
        )
        
        # Run analysis
        result = agent.run()
        
        # Display result
        print(result)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Analysis saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()