"""
Fundamental Analysis Agent - Standalone CLI Tool
Usage: python fundamental_agent.py AAPL --output report.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
from openai import OpenAI

class FundamentalAgent:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are an EOD trading fundamental analyst. Analyze company fundamentals for trading decisions.
Focus on: 1) Valuation 2) Growth 3) Profitability 4) Financial health 5) Analyst sentiment.
End with: RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low"""
    
    def get_fundamentals(self):
        """Fetch fundamental data"""
        stock = yf.Ticker(self.ticker)
        info = stock.info
        
        fundamentals = f"""Fundamental Data for {self.ticker}:

VALUATION:
- Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Forward P/E: {info.get('forwardPE', 'N/A')}
- PEG Ratio: {info.get('pegRatio', 'N/A')}
- Price/Book: {info.get('priceToBook', 'N/A')}

GROWTH:
- Revenue Growth: {info.get('revenueGrowth', 0)*100:.1f}%
- Earnings Growth: {info.get('earningsGrowth', 0)*100:.1f}%

PROFITABILITY:
- Profit Margin: {info.get('profitMargins', 0)*100:.1f}%
- Operating Margin: {info.get('operatingMargins', 0)*100:.1f}%
- ROE: {info.get('returnOnEquity', 0)*100:.1f}%

FINANCIAL HEALTH:
- Current Ratio: {info.get('currentRatio', 'N/A')}
- Debt/Equity: {info.get('debtToEquity', 'N/A')}
- Free Cash Flow: ${info.get('freeCashflow', 0)/1e9:.2f}B

ANALYST VIEW:
- Recommendation: {info.get('recommendationKey', 'N/A')}
- Target Price: ${info.get('targetMeanPrice', 0):.2f}
- Current Price: ${info.get('currentPrice', 0):.2f}
- Upside: {((info.get('targetMeanPrice', 0)/info.get('currentPrice', 1)-1)*100) if info.get('currentPrice') else 0:.1f}%
"""
        return fundamentals
    
    def get_earnings(self):
        """Get earnings history"""
        stock = yf.Ticker(self.ticker)
        try:
            earnings = stock.earnings_history
            if len(earnings):
                df = pd.DataFrame(earnings)
                recent = df.tail(4)
                
                earnings_summary = "\nRECENT EARNINGS (Last 4 Quarters):\n"
                beats = 0
                for _, row in recent.iterrows():
                    actual = row.get('epsActual', 0)
                    estimate = row.get('epsEstimate', 0)
                    if actual > estimate:
                        beats += 1
                        earnings_summary += f"- BEAT: Actual ${actual:.2f} vs Est ${estimate:.2f}\n"
                    else:
                        earnings_summary += f"- MISS: Actual ${actual:.2f} vs Est ${estimate:.2f}\n"
                
                earnings_summary += f"\nBeat Rate: {beats}/4 quarters ({beats*25}%)"
                return earnings_summary
        except Exception as e:
            print(f"Error fetching earnings data: {e}")
            pass
        return "\nEarnings data not available"
    
    def get_assets(self):
        "Get assets info from company annual report 10-K"
        headers = {'User-Agent': "test@gmail.com"}
        try:
            companyData = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=headers
                )

            if companyData.status_code == 200:
                companyData = pd.DataFrame.from_dict(
                    companyData.json(), orient='index' )
                cik = str(companyData[companyData['ticker'] == self.ticker]['cik_str'].values[0]).zfill(10)
            else:
                print(f"Error: Failed to retrieve company data. Status code: {companyData.status_code}")
        except Exception as e:
            raise Exception(f"Failed to retrieve company data {e}")


        companyConcept = requests.get(
            (
                f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}'
                f'/us-gaap/Assets.json'
            ),
            headers=headers
        )

        if companyConcept.status_code == 200:
            print(f"Company data for {self.ticker} retrieved successfully.")
            assetsData = pd.DataFrame.from_dict(
                companyConcept.json()['units']['USD']
            )

            assets10K = assetsData[assetsData.form == '10-K']
            assets10K = assets10K.reset_index(drop=True)
            val = str(assets10K['val'].iloc[-1])

            company_assets = f"""
            {self.ticker} assets from latest 10-K filing:
            ${val}
            """

            return company_assets
        else:
            print(f"Error fetching company data: Status code {companyConcept.status_code}")



    def analyze_with_llm(self, fundamental_data):
        """Use LLM to analyze fundamentals"""
        if not self.client:
            # Simple rule-based fallback
            if "strong buy" in fundamental_data.lower():
                return f"{fundamental_data}\n\nRECOMMENDATION: BUY - Confidence: Medium"
            elif "sell" in fundamental_data.lower():
                return f"{fundamental_data}\n\nRECOMMENDATION: SELL - Confidence: Medium"
            else:
                return f"{fundamental_data}\n\nRECOMMENDATION: HOLD - Confidence: Low"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze these fundamentals for {self.ticker}:\n{fundamental_data}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"{fundamental_data}\n\nRECOMMENDATION: HOLD - Confidence: Low"
    
    def run(self):
        """Execute full analysis"""
        fundamentals = self.get_fundamentals()
        earnings = self.get_earnings()
        assets = self.get_assets()

        full_data = fundamentals + earnings + assets
        analysis = self.analyze_with_llm(full_data)
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="Fundamental Analysis Agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    
    args = parser.parse_args()
    
    agent = FundamentalAgent(args.ticker, args.api_key, args.model)
    result = agent.run()
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()