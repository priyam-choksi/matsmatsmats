"""
Fundamental Analysis Agent - Enhanced Version
Provides comprehensive fundamental analysis for trading decisions

MODIFIED: Now supports historical backtesting via analysis_date parameter
- Calculates historical P/E, P/B, Current Ratio from quarterly financials
- Marks unavailable metrics as N/A instead of using current (invalid) values
ENHANCED: Added _get_llm_decision() for better recommendation extraction

Usage: 
  python fundamental_agent.py AAPL
  python fundamental_agent.py AAPL --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
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
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini", 
                 use_cache: bool = True, analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.use_cache = use_cache
        
        # Historical backtesting support
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[FUNDAMENTALS] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[FUNDAMENTALS] Running in LIVE mode (current data)")
        
        # Setup cache
        self.cache_dir = Path("cache/fundamental")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced system prompt
        self.system_prompt = """You are an expert fundamental analyst focused on identifying fundamental catalysts and factors that could drive price movements over the next 3-6 months.

**YOUR ANALYSIS FRAMEWORK:**

1. **Valuation Assessment:**
   - Is the P/E ratio reasonable compared to growth rate (PEG ratio)?
   - How does Price/Book compare to sector averages?
   - Is the company trading at a premium or discount to historical valuations?

2. **Growth Analysis:**
   - Are BOTH revenue AND earnings growing? At what rates?
   - Is growth accelerating (good) or decelerating (concerning)?
   - Are margins expanding (excellent) or contracting (warning)?

3. **Profitability Metrics:**
   - Are profit margins healthy for the sector?
   - Is ROE above 15% (generally good) or below 10% (concerning)?
   - Are operating margins improving or deteriorating?

4. **Financial Health:**
   - Is Current Ratio > 1.5 (healthy) or < 1.0 (liquidity concerns)?
   - Is Debt/Equity manageable for the industry?
   - Is Free Cash Flow positive and growing?

5. **Earnings Quality:**
   - What's the earnings beat/miss pattern?
   - Are earnings surprises getting larger or smaller?

**IMPORTANT FOR HISTORICAL ANALYSIS:**
- Some metrics may show "N/A (historical)" - this means we couldn't calculate them for the historical date
- Only use metrics that have actual values, ignore N/A metrics in your analysis
- Focus on the metrics that ARE available

**OUTPUT FORMAT:**

## Fundamental Analysis Summary
**Overall Assessment:** [1-2 sentence summary]

## Key Findings
### Valuation
[Analyze available P/E, P/B ratios]

### Growth & Profitability  
[Discuss revenue/earnings growth, margins]

### Financial Health
[Evaluate balance sheet strength, debt levels]

### Earnings Quality
[Analyze beat/miss pattern]

## Investment Thesis
**Bull Case:** [2-3 strongest positive factors]
**Bear Case:** [2-3 key risks or concerns]

## Final Recommendation
RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low
**Risk Level:** Low/Medium/High"""

    def get_comprehensive_fundamentals(self) -> Dict[str, Any]:
        """
        Gather comprehensive fundamental data from yfinance
        NOTE: yfinance .info is always current - we'll override with historical in calculate_historical_metrics()
        """
        try:
            print(f"[FUNDAMENTALS] Fetching comprehensive data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                print(f"[FUNDAMENTALS] ⚠️ No data returned for {self.ticker}")
                return None
            
            def safe_get(key, default="N/A", format_type=None):
                value = info.get(key)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return default
                try:
                    if format_type == "billions":
                        return f"${float(value)/1e9:.2f}B"
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
                },
                "_raw": {
                    "shares_outstanding": info.get('sharesOutstanding'),
                    "current_price": info.get('currentPrice'),
                }
            }
            
            # Calculate implied upside
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

    def calculate_historical_metrics(self, fundamentals: Dict) -> Dict:
        """
        Calculate historical fundamental metrics using quarterly financial data.
        This replaces current-only values with historical calculations.
        For metrics we CAN'T calculate historically, mark them as N/A.
        """
        if not self.analysis_date:
            return fundamentals
        
        try:
            stock = yf.Ticker(self.ticker)
            analysis_dt = datetime.strptime(self.analysis_date, '%Y-%m-%d')
            
            print(f"[FUNDAMENTALS] Calculating historical metrics for {self.analysis_date}...")
            
            calculated = []
            
            # === 1. HISTORICAL PRICE ===
            hist_price = stock.history(
                start=(analysis_dt - timedelta(days=7)).strftime('%Y-%m-%d'),
                end=(analysis_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            
            historical_close = None
            if not hist_price.empty:
                historical_close = hist_price['Close'].iloc[-1]
                fundamentals['price_info']['current_price'] = f"${historical_close:.2f}"
                fundamentals['analyst']['current_price'] = f"${historical_close:.2f}"
                calculated.append(f"Price: ${historical_close:.2f}")
            else:
                fundamentals['price_info']['current_price'] = "N/A (no historical data)"
            
            # === 2. GET QUARTERLY FINANCIALS ===
            quarterly_fin = stock.quarterly_financials
            quarterly_bs = stock.quarterly_balance_sheet
            shares = fundamentals.get('_raw', {}).get('shares_outstanding') or stock.info.get('sharesOutstanding')
            
            valid_fin_quarters = []
            valid_bs_quarters = []
            
            if quarterly_fin is not None and not quarterly_fin.empty:
                valid_fin_quarters = sorted(
                    [col for col in quarterly_fin.columns if col.to_pydatetime() <= analysis_dt],
                    reverse=True
                )
            
            if quarterly_bs is not None and not quarterly_bs.empty:
                valid_bs_quarters = sorted(
                    [col for col in quarterly_bs.columns if col.to_pydatetime() <= analysis_dt],
                    reverse=True
                )
            
            # === 3. CALCULATE HISTORICAL P/E ===
            if historical_close and valid_fin_quarters and shares:
                try:
                    ttm_earnings = 0
                    for q in valid_fin_quarters[:4]:
                        net_income = quarterly_fin.loc['Net Income', q] if 'Net Income' in quarterly_fin.index else None
                        if net_income and not pd.isna(net_income):
                            ttm_earnings += net_income
                    
                    if ttm_earnings > 0:
                        eps = ttm_earnings / shares
                        historical_pe = historical_close / eps
                        fundamentals['valuation']['trailing_pe'] = f"{historical_pe:.2f}"
                        calculated.append(f"P/E: {historical_pe:.2f}")
                    else:
                        fundamentals['valuation']['trailing_pe'] = "N/A (negative earnings)"
                except Exception as e:
                    fundamentals['valuation']['trailing_pe'] = "N/A (calculation error)"
                    print(f"[FUNDAMENTALS] ⚠️ P/E calculation error: {e}")
            else:
                fundamentals['valuation']['trailing_pe'] = "N/A (historical)"
            
            # === 4. CALCULATE HISTORICAL P/B ===
            if historical_close and valid_bs_quarters and shares:
                try:
                    latest_bs = valid_bs_quarters[0]
                    total_equity = None
                    
                    for equity_field in ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest']:
                        if equity_field in quarterly_bs.index:
                            total_equity = quarterly_bs.loc[equity_field, latest_bs]
                            if not pd.isna(total_equity):
                                break
                    
                    if total_equity and total_equity > 0:
                        bvps = total_equity / shares
                        historical_pb = historical_close / bvps
                        fundamentals['valuation']['price_to_book'] = f"{historical_pb:.2f}"
                        calculated.append(f"P/B: {historical_pb:.2f}")
                    else:
                        fundamentals['valuation']['price_to_book'] = "N/A (no equity data)"
                except Exception as e:
                    fundamentals['valuation']['price_to_book'] = "N/A (calculation error)"
                    print(f"[FUNDAMENTALS] ⚠️ P/B calculation error: {e}")
            else:
                fundamentals['valuation']['price_to_book'] = "N/A (historical)"
            
            # === 5. CALCULATE HISTORICAL CURRENT RATIO ===
            if valid_bs_quarters:
                try:
                    latest_bs = valid_bs_quarters[0]
                    current_assets = None
                    current_liab = None
                    
                    for ca_field in ['Current Assets', 'Total Current Assets']:
                        if ca_field in quarterly_bs.index:
                            current_assets = quarterly_bs.loc[ca_field, latest_bs]
                            if not pd.isna(current_assets):
                                break
                    
                    for cl_field in ['Current Liabilities', 'Total Current Liabilities']:
                        if cl_field in quarterly_bs.index:
                            current_liab = quarterly_bs.loc[cl_field, latest_bs]
                            if not pd.isna(current_liab):
                                break
                    
                    if current_assets and current_liab and current_liab > 0:
                        historical_cr = current_assets / current_liab
                        fundamentals['financial_health']['current_ratio'] = f"{historical_cr:.2f}"
                        calculated.append(f"Current Ratio: {historical_cr:.2f}")
                    else:
                        fundamentals['financial_health']['current_ratio'] = "N/A (missing data)"
                except Exception as e:
                    fundamentals['financial_health']['current_ratio'] = "N/A (calculation error)"
                    print(f"[FUNDAMENTALS] ⚠️ Current Ratio calculation error: {e}")
            else:
                fundamentals['financial_health']['current_ratio'] = "N/A (historical)"
            
            # === 6. Mark other valuation metrics as N/A for historical ===
            fundamentals['valuation']['forward_pe'] = "N/A (historical - forward-looking)"
            fundamentals['valuation']['peg_ratio'] = "N/A (historical)"
            
            print(f"[FUNDAMENTALS] ✓ Historical metrics calculated: {len(calculated)}")
            for item in calculated:
                print(f"[FUNDAMENTALS]   - {item}")
            
            return fundamentals
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ⚠️ Historical calculation error: {e}")
            import traceback
            traceback.print_exc()
            return fundamentals

    def get_earnings_analysis(self) -> Dict[str, Any]:
        """Analyze recent earnings performance"""
        try:
            print(f"[FUNDAMENTALS] Analyzing earnings history...")
            stock = yf.Ticker(self.ticker)
            earnings = stock.earnings_history
            
            if earnings is None or len(earnings) == 0:
                print(f"[FUNDAMENTALS] ⚠️ No earnings data available")
                return {"available": False, "message": "No earnings history available"}
            
            df = pd.DataFrame(earnings)
            
            # Filter earnings to before analysis_date if historical
            if self.analysis_date:
                analysis_dt = datetime.strptime(self.analysis_date, '%Y-%m-%d')
                if 'Earnings Date' in df.columns:
                    df['Earnings Date'] = pd.to_datetime(df['Earnings Date'])
                    df = df[df['Earnings Date'] <= analysis_dt]
                print(f"[FUNDAMENTALS] Filtering earnings to before {self.analysis_date}")
            
            if len(df) == 0:
                return {"available": False, "message": "No earnings data for period"}
            
            # Analyze last 4 quarters
            recent = df.head(4)
            quarters_data = []
            beats = 0
            misses = 0
            surprises = []
            
            for _, row in recent.iterrows():
                actual = row.get('Reported EPS') or row.get('epsActual')
                estimate = row.get('EPS Estimate') or row.get('epsEstimate')
                
                if pd.notna(actual) and pd.notna(estimate) and estimate != 0:
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                    beat = actual > estimate
                    
                    if beat:
                        beats += 1
                    else:
                        misses += 1
                    
                    surprises.append(surprise_pct)
                    
                    quarters_data.append({
                        'period': str(row.get('Earnings Date', 'Unknown'))[:10],
                        'actual': actual,
                        'estimate': estimate,
                        'surprise_pct': surprise_pct,
                        'beat': beat
                    })
            
            beat_rate = (beats / (beats + misses) * 100) if (beats + misses) > 0 else 0
            avg_surprise = np.mean(surprises) if surprises else 0
            
            if len(surprises) >= 2:
                if surprises[0] > surprises[-1]:
                    trend = "Improving"
                elif surprises[0] < surprises[-1]:
                    trend = "Declining"
                else:
                    trend = "Stable"
            else:
                trend = "Unknown"
            
            quality = "Strong" if beats >= 3 else "Mixed" if beats == 2 else "Weak"
            
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
            print(f"[FUNDAMENTALS] ⚠️ Error analyzing earnings: {e}")
            return {"available": False, "message": f"Error: {str(e)}"}

    def get_sec_filing_data(self) -> Dict[str, Any]:
        """Fetch data from SEC 10-K filings with historical date support"""
        if self.use_cache and not self.analysis_date:
            cached = self._get_cached_sec_data()
            if cached:
                print(f"[FUNDAMENTALS] ✓ Using cached SEC data")
                return cached
        
        print(f"[FUNDAMENTALS] Fetching SEC 10-K filing data...")
        headers = {'User-Agent': "fundamental-agent/1.0"}
        
        try:
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=headers, timeout=10
            )
            
            if response.status_code != 200:
                return {"available": False, "error": "SEC API unavailable"}
            
            companies = pd.DataFrame.from_dict(response.json(), orient='index')
            ticker_match = companies[companies['ticker'] == self.ticker]
            
            if ticker_match.empty:
                return {"available": False, "error": f"{self.ticker} not in SEC database"}
            
            cik = str(ticker_match['cik_str'].values[0]).zfill(10)
            
            assets_response = requests.get(
                f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json',
                headers=headers, timeout=10
            )
            
            if assets_response.status_code != 200:
                return {"available": False, "error": "Assets data unavailable"}
            
            assets_data = pd.DataFrame.from_dict(assets_response.json()['units']['USD'])
            assets_10k = assets_data[assets_data['form'] == '10-K']
            
            if assets_10k.empty:
                return {"available": False, "error": "No 10-K filings found"}
            
            # Filter by analysis_date if historical
            if self.analysis_date:
                assets_10k = assets_10k[assets_10k['filed'] <= self.analysis_date]
                if assets_10k.empty:
                    return {"available": False, "error": f"No 10-K filings before {self.analysis_date}"}
                print(f"[FUNDAMENTALS] Filtering SEC filings to before {self.analysis_date}")
            
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
            
            if self.use_cache and not self.analysis_date:
                self._cache_sec_data(sec_data)
            
            print(f"[FUNDAMENTALS] ✓ SEC data retrieved (Filed: {latest['filed']})")
            return sec_data
            
        except requests.Timeout:
            return {"available": False, "error": "Request timeout"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def _get_cached_sec_data(self) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{self.ticker}_sec.pkl"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            cache_age = datetime.now() - cached['timestamp']
            if cache_age < timedelta(days=90):
                data = cached['data']
                data['cache_age_days'] = cache_age.days
                return data
            return None
        except:
            return None

    def _cache_sec_data(self, data: Dict):
        cache_file = self.cache_dir / f"{self.ticker}_sec.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'timestamp': datetime.now(), 'data': data}, f)
        except:
            pass

    def format_data_for_analysis(self, fundamentals: Dict, earnings: Dict, sec_data: Dict) -> str:
        """Format all data for LLM analysis"""
        date_header = ""
        if self.analysis_date:
            date_header = f"""
**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date} ⚠️**
Metrics marked "N/A" could not be calculated historically - ignore them in your analysis.
Only analyze metrics with actual values.

"""
        
        report = f"""{date_header}# Fundamental Data for {self.ticker}

## Company Information
- **Name:** {fundamentals['company_info']['name']}
- **Sector:** {fundamentals['company_info']['sector']}
- **Industry:** {fundamentals['company_info']['industry']}

## Current Price
- **Price:** {fundamentals['price_info']['current_price']}
- **52-Week High:** {fundamentals['price_info']['fifty_two_week_high']}
- **52-Week Low:** {fundamentals['price_info']['fifty_two_week_low']}

## Valuation Metrics
- **Market Cap:** {fundamentals['valuation']['market_cap']}
- **Enterprise Value:** {fundamentals['valuation']['enterprise_value']}
- **Trailing P/E:** {fundamentals['valuation']['trailing_pe']}
- **Forward P/E:** {fundamentals['valuation']['forward_pe']}
- **PEG Ratio:** {fundamentals['valuation']['peg_ratio']}
- **Price/Book:** {fundamentals['valuation']['price_to_book']}
- **Price/Sales:** {fundamentals['valuation']['price_to_sales']}
- **EV/Revenue:** {fundamentals['valuation']['ev_to_revenue']}
- **EV/EBITDA:** {fundamentals['valuation']['ev_to_ebitda']}

## Growth Metrics
- **Revenue Growth:** {fundamentals['growth']['revenue_growth']}
- **Earnings Growth:** {fundamentals['growth']['earnings_growth']}
- **Quarterly Revenue Growth:** {fundamentals['growth']['quarterly_revenue_growth']}
- **Quarterly Earnings Growth:** {fundamentals['growth']['quarterly_earnings_growth']}

## Profitability
- **Profit Margin:** {fundamentals['profitability']['profit_margin']}
- **Operating Margin:** {fundamentals['profitability']['operating_margin']}
- **Gross Margin:** {fundamentals['profitability']['gross_margin']}
- **ROE:** {fundamentals['profitability']['roe']}
- **ROA:** {fundamentals['profitability']['roa']}

## Financial Health
- **Current Ratio:** {fundamentals['financial_health']['current_ratio']}
- **Quick Ratio:** {fundamentals['financial_health']['quick_ratio']}
- **Debt/Equity:** {fundamentals['financial_health']['debt_to_equity']}
- **Total Cash:** {fundamentals['financial_health']['total_cash']}
- **Total Debt:** {fundamentals['financial_health']['total_debt']}
- **Free Cash Flow:** {fundamentals['financial_health']['free_cash_flow']}

## Analyst Opinions
- **Recommendation:** {fundamentals['analyst']['recommendation']}
- **Number of Analysts:** {fundamentals['analyst']['num_analysts']}
- **Target (Mean):** {fundamentals['analyst']['target_mean']}
- **Target (High):** {fundamentals['analyst']['target_high']}
- **Target (Low):** {fundamentals['analyst']['target_low']}
- **Implied Upside:** {fundamentals['analyst'].get('implied_upside', 'N/A')}

"""
        
        # Add earnings data
        if earnings['available']:
            report += f"## Earnings History (Last 4 Quarters)\n"
            for q in earnings['quarters']:
                beat_miss = "✓ Beat" if q['beat'] else "✗ Miss"
                report += f"- {q['period']}: Actual ${q['actual']:.2f} vs Est ${q['estimate']:.2f} ({q['surprise_pct']:+.1f}%) {beat_miss}\n"
            report += f"\n**Summary:** Beat Rate: {earnings['summary']['beat_rate']:.0f}%, Trend: {earnings['summary']['trend']}, Quality: {earnings['summary']['quality']}\n\n"
        
        # Add SEC data
        if sec_data['available']:
            report += f"## SEC 10-K Filing Data\n"
            report += f"- **Total Assets:** {sec_data['total_assets_formatted']}\n"
            report += f"- **Filing Date:** {sec_data['filing_date']}\n\n"
        
        return report

    def _get_llm_decision(self, analysis: str) -> Tuple[str, str]:
        """
        NEW: Use LLM to extract/determine recommendation from analysis.
        This avoids the HOLD default bias problem.
        """
        if not self.client:
            return self._extract_recommendation_from_content(analysis)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a trading decision extractor. 
Given a fundamental analysis, extract or determine the final recommendation.

RULES:
1. If there's an explicit "RECOMMENDATION: X" line, extract it
2. If not explicit, analyze the content and determine the most appropriate recommendation
3. Consider: valuation metrics, growth rates, profitability, financial health, analyst targets
4. Do NOT default to HOLD - make an actual decision based on the evidence
5. FUNDAMENTAL-SPECIFIC signals:
   - Low P/E + high growth = BUY signal
   - High P/E + slowing growth = SELL signal  
   - Strong balance sheet + beat streak = BUY signal
   - High debt + margin compression = SELL signal

Respond in EXACTLY this format (no other text):
RECOMMENDATION: BUY|HOLD|SELL
CONFIDENCE: High|Medium|Low"""},
                    {"role": "user", "content": f"Extract/determine recommendation from:\n\n{analysis[:3000]}"}
                ],
                temperature=0.3,
                max_completion_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            rec = "HOLD"
            conf = "Medium"
            
            for line in result.split('\n'):
                if 'RECOMMENDATION:' in line.upper():
                    if 'BUY' in line.upper():
                        rec = "BUY"
                    elif 'SELL' in line.upper():
                        rec = "SELL"
                    else:
                        rec = "HOLD"
                elif 'CONFIDENCE:' in line.upper():
                    if 'HIGH' in line.upper():
                        conf = "High"
                    elif 'LOW' in line.upper():
                        conf = "Low"
                    else:
                        conf = "Medium"
            
            print(f"[FUNDAMENTALS] LLM Decision: {rec} ({conf})")
            return rec, conf
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ⚠️ LLM decision error: {e}, using fallback")
            return self._extract_recommendation_from_content(analysis)

    def _extract_recommendation_from_content(self, analysis: str) -> Tuple[str, str]:
        """Fallback: Extract recommendation using keyword analysis"""
        analysis_lower = analysis.lower()
        
        # Fundamental-specific signals
        specific_buy_signals = ['undervalued', 'strong growth', 'beat expectations', 'expanding margins', 'healthy balance sheet', 'positive cash flow']
        specific_sell_signals = ['overvalued', 'slowing growth', 'missed expectations', 'margin pressure', 'high debt', 'negative cash flow']
        
        if any(phrase in analysis_lower for phrase in ["recommend buy", "should buy", "buy rating", "strong buy"]):
            return "BUY", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend sell", "should sell", "sell rating", "underperform"]):
            return "SELL", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend hold", "neutral rating", "wait", "fairly valued"]):
            return "HOLD", "Low"
        
        buy_count = sum(1 for signal in specific_buy_signals if signal in analysis_lower)
        sell_count = sum(1 for signal in specific_sell_signals if signal in analysis_lower)
        
        general_buy = ["bullish", "positive", "upside", "growth", "strong", "attractive"]
        general_sell = ["bearish", "negative", "downside", "decline", "weak", "expensive"]
        
        buy_count += sum(0.5 for word in general_buy if word in analysis_lower)
        sell_count += sum(0.5 for word in general_sell if word in analysis_lower)
        
        if buy_count > sell_count + 1.5:
            confidence = "High" if buy_count > 4 else "Medium"
            return "BUY", confidence
        elif sell_count > buy_count + 1.5:
            confidence = "High" if sell_count > 4 else "Medium"
            return "SELL", confidence
        else:
            return "HOLD", "Low"

    def analyze_with_llm(self, formatted_data: str) -> str:
        """Send data to LLM for analysis"""
        if not self.client:
            print("[FUNDAMENTALS] ⚠️ No OpenAI API key - using fallback")
            return self._create_fallback_analysis(formatted_data)
        
        try:
            print(f"[FUNDAMENTALS] Generating analysis with {self.model}...")
            
            date_note = ""
            if self.analysis_date:
                date_note = f"IMPORTANT: This is a HISTORICAL analysis as of {self.analysis_date}. Ignore any metrics marked 'N/A'.\n\n"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{date_note}Analyze this fundamental data:\n\n{formatted_data}"}
                ],
                temperature=0.7,
                max_completion_tokens=2000
            )
            
            analysis = response.choices[0].message.content
            
            # Use new LLM decision extraction if no formal recommendation
            if "RECOMMENDATION:" not in analysis:
                print(f"[FUNDAMENTALS] ⚠️ Response missing formal recommendation, extracting...")
                recommendation, confidence = self._get_llm_decision(analysis)
                analysis += f"\n\nRECOMMENDATION: {recommendation} - Confidence: {confidence}"
            
            print(f"[FUNDAMENTALS] ✓ Analysis generated ({len(analysis)} chars)")
            return analysis
            
        except Exception as e:
            print(f"[FUNDAMENTALS] ❌ LLM error: {e}")
            return self._create_fallback_analysis(formatted_data)

    def _create_fallback_analysis(self, formatted_data: str) -> str:
        """Fallback analysis when LLM unavailable"""
        return f"""## Fundamental Analysis Summary
*Generated using fallback analysis (LLM unavailable)*

{formatted_data}

RECOMMENDATION: HOLD - Confidence: Low
"""

    def run(self) -> str:
        """Run complete fundamental analysis pipeline"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"FUNDAMENTAL ANALYSIS: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Step 1: Get fundamentals
        fundamentals = self.get_comprehensive_fundamentals()
        
        # Step 1b: Calculate historical metrics
        if fundamentals and self.analysis_date:
            fundamentals = self.calculate_historical_metrics(fundamentals)
        
        if not fundamentals:
            return f"RECOMMENDATION: HOLD - Confidence: N/A\n[FUNDAMENTALS] No data available for {self.ticker}"
        
        # Step 2: Earnings analysis
        earnings = self.get_earnings_analysis()
        
        # Step 3: SEC filing data
        sec_data = self.get_sec_filing_data()
        
        # Step 4: Format data
        formatted_data = self.format_data_for_analysis(fundamentals, earnings, sec_data)
        
        # Step 5: LLM analysis
        analysis = self.analyze_with_llm(formatted_data)
        
        elapsed = time.time() - start_time
        print(f"\n[FUNDAMENTALS] ✓ Analysis complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description="Fundamental Analysis Agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Save to file")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--analysis-date", help="Historical date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        agent = FundamentalAgent(
            ticker=args.ticker,
            api_key=args.api_key,
            model=args.model,
            use_cache=not args.no_cache,
            analysis_date=args.analysis_date
        )
        
        result = agent.run()
        print(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()