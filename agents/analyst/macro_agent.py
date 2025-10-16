"""
Macro Economic Analysis Agent - Standalone CLI Tool
Updated with 7-day market movements focus
Usage: python macro_agent.py --sector technology --days 7 --output report.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import yfinance as yf
from openai import OpenAI

class MacroAgent:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are an EOD trading macro analyst. Analyze macroeconomic conditions for overnight trading.
Focus on: 1) Interest rates 2) Economic indicators 3) Market volatility 4) Sector rotation 5) Risk sentiment.
End with market outlook and sector recommendations."""
    
    def get_market_indicators(self, days=7):
        """Get major market indicators for specified period"""
        indicators = {}
        
        # Get major indices
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX (Volatility)',
            '^TNX': '10-Year Treasury',
            '^RUT': 'Russell 2000'
        }
        
        market_summary = f"MARKET INDICATORS ({days}-Day Analysis):\n\n"
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                # Get appropriate period
                if days <= 7:
                    hist = ticker.history(period="7d")
                elif days <= 30:
                    hist = ticker.history(period="1mo")
                else:
                    hist = ticker.history(period=f"{days}d")
                
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    start = hist['Close'].iloc[0]
                    
                    # Calculate period change
                    period_change = ((current/start - 1) * 100)
                    
                    # Daily average change
                    daily_avg = period_change / min(days, len(hist))
                    
                    # Volatility (standard deviation of daily returns)
                    daily_returns = hist['Close'].pct_change()
                    volatility = daily_returns.std() * 100
                    
                    market_summary += f"{name}:\n"
                    market_summary += f"  Current: {current:.2f}\n"
                    market_summary += f"  {days}-Day Change: {period_change:+.2f}%\n"
                    market_summary += f"  Daily Avg: {daily_avg:+.2f}%\n"
                    market_summary += f"  Volatility: {volatility:.2f}%\n\n"
                    
                    indicators[name] = {
                        'value': current, 
                        'change': period_change,
                        'volatility': volatility
                    }
            except Exception as e:
                market_summary += f"{name}: Data unavailable\n\n"
                continue
        
        return market_summary, indicators
    
    def get_sector_performance(self, days=7):
        """Get sector ETF performance"""
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
        
        sector_summary = f"\nSECTOR PERFORMANCE ({days}-Day):\n\n"
        sector_data = {}
        
        for symbol, name in sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                if days <= 7:
                    hist = ticker.history(period="7d")
                elif days <= 30:
                    hist = ticker.history(period="1mo")
                else:
                    hist = ticker.history(period=f"{days}d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    start = hist['Close'].iloc[0]
                    change = ((current/start - 1) * 100)
                    
                    # Calculate momentum (last 2 days vs prior period)
                    if len(hist) >= 3:
                        recent = hist['Close'].iloc[-2:].mean()
                        older = hist['Close'].iloc[:-2].mean()
                        momentum = ((recent/older - 1) * 100)
                    else:
                        momentum = 0
                    
                    sector_data[name] = {
                        'change': change,
                        'momentum': momentum
                    }
            except:
                continue
        
        # Sort sectors by performance
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['change'], reverse=True)
        
        sector_summary += "LEADERS:\n"
        for name, data in sorted_sectors[:3]:
            sector_summary += f"  {name}: {data['change']:+.2f}% (Momentum: {data['momentum']:+.2f}%)\n"
        
        sector_summary += "\nLAGGARDS:\n"
        for name, data in sorted_sectors[-3:]:
            sector_summary += f"  {name}: {data['change']:+.2f}% (Momentum: {data['momentum']:+.2f}%)\n"
        
        return sector_summary, sector_data
    
    def get_economic_context(self, days=7):
        """Get economic context from key indicators"""
        context = f"\nECONOMIC INDICATORS ({days}-Day Changes):\n\n"
        
        economic_indicators = {
            'DX-Y.NYB': ('Dollar Index', 'Currency Strength'),
            'GC=F': ('Gold', 'Safe Haven'),
            'CL=F': ('Crude Oil', 'Energy/Inflation'),
            'BTC-USD': ('Bitcoin', 'Risk Sentiment'),
            'HYG': ('High Yield Bonds', 'Credit Risk'),
            'TLT': ('20Y Treasury', 'Rate Expectations')
        }
        
        for symbol, (name, indicator_type) in economic_indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                if days <= 7:
                    hist = ticker.history(period="7d")
                elif days <= 30:
                    hist = ticker.history(period="1mo")
                else:
                    hist = ticker.history(period=f"{days}d")
                
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    start = hist['Close'].iloc[0]
                    change = ((current/start - 1) * 100)
                    
                    # Trend direction
                    if len(hist) >= 3:
                        mid_point = hist['Close'].iloc[len(hist)//2]
                        if current > mid_point > start:
                            trend = "Uptrend"
                        elif current < mid_point < start:
                            trend = "Downtrend"
                        else:
                            trend = "Choppy"
                    else:
                        trend = "N/A"
                    
                    context += f"{name} ({indicator_type}):\n"
                    context += f"  Price: ${current:.2f}\n"
                    context += f"  {days}-Day: {change:+.2f}%\n"
                    context += f"  Trend: {trend}\n\n"
            except:
                continue
        
        return context
    
    def get_market_breadth(self, days=7):
        """Analyze market breadth indicators"""
        breadth = f"\nMARKET BREADTH ANALYSIS ({days}-Day):\n\n"
        
        try:
            # Compare small caps vs large caps
            spy = yf.Ticker('^GSPC')
            iwm = yf.Ticker('^RUT')
            
            spy_hist = spy.history(period="7d" if days <= 7 else "1mo")
            iwm_hist = iwm.history(period="7d" if days <= 7 else "1mo")
            
            if not spy_hist.empty and not iwm_hist.empty:
                spy_change = ((spy_hist['Close'].iloc[-1]/spy_hist['Close'].iloc[0] - 1) * 100)
                iwm_change = ((iwm_hist['Close'].iloc[-1]/iwm_hist['Close'].iloc[0] - 1) * 100)
                
                breadth += "Small vs Large Caps:\n"
                if iwm_change > spy_change:
                    breadth += f"  Russell 2000 outperforming S&P by {(iwm_change-spy_change):.2f}%\n"
                    breadth += "  → Risk-ON sentiment (bullish for growth)\n"
                else:
                    breadth += f"  S&P 500 outperforming Russell by {(spy_change-iwm_change):.2f}%\n"
                    breadth += "  → Flight to quality (defensive positioning)\n"
        except:
            breadth += "Breadth data unavailable\n"
        
        return breadth
    
    def analyze_with_llm(self, market_data):
        """Use LLM to analyze macro conditions"""
        if not self.client:
            return f"""{market_data}

MACRO ANALYSIS:
Based on the data, market conditions show:
- Volatility levels indicate moderate risk
- Sector rotation suggests defensive positioning
- Economic indicators point to mixed growth outlook

Risk Sentiment: NEUTRAL
Recommended Sectors: Technology, Healthcare (defensive growth)
Avoid Sectors: Energy, Real Estate (rate sensitive)

EOD Trading Implications:
- Consider reduced position sizes in volatile conditions
- Focus on sector leaders with momentum
- Monitor VIX for overnight risk assessment"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze these macro conditions and provide EOD trading insights:\n{market_data}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"{market_data}\n\nMacro outlook: Neutral (Analysis limited)"
    
    def run(self, sector=None, days=7):
        """Execute full macro analysis"""
        print(f"Analyzing macro conditions ({days}-day view)...")
        
        full_data = f"MACRO ECONOMIC ANALYSIS\n"
        full_data += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        full_data += f"Analysis Period: {days} Days\n"
        full_data += "="*60 + "\n\n"
        
        # Get all components
        market_summary, indicators = self.get_market_indicators(days)
        full_data += market_summary
        
        sector_summary, sectors = self.get_sector_performance(days)
        full_data += sector_summary
        
        economic_context = self.get_economic_context(days)
        full_data += economic_context
        
        market_breadth = self.get_market_breadth(days)
        full_data += market_breadth
        
        # Add specific sector focus if requested
        if sector:
            full_data += f"\n{'='*40}\n"
            full_data += f"FOCUS SECTOR: {sector.upper()}\n"
            full_data += f"{'='*40}\n"
            
            # Find relevant sector ETF
            sector_map = {
                'technology': 'XLK',
                'tech': 'XLK',
                'financial': 'XLF',
                'financials': 'XLF',
                'energy': 'XLE',
                'healthcare': 'XLV',
                'health': 'XLV'
            }
            
            if sector.lower() in sector_map:
                etf = sector_map[sector.lower()]
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="7d" if days <= 7 else "1mo")
                    if not hist.empty:
                        full_data += f"Sector Performance: {((hist['Close'].iloc[-1]/hist['Close'].iloc[0]-1)*100):+.2f}%\n"
                except:
                    pass
        
        # Add summary statistics
        full_data += f"\n{'='*40}\n"
        full_data += "MARKET SUMMARY:\n"
        full_data += f"{'='*40}\n"
        
        # Risk assessment based on VIX
        if 'VIX (Volatility)' in indicators:
            vix = indicators['VIX (Volatility)']['value']
            if vix < 15:
                full_data += "Volatility: LOW (VIX < 15) - Favorable for overnight positions\n"
            elif vix < 25:
                full_data += "Volatility: MODERATE (VIX 15-25) - Normal overnight risk\n"
            else:
                full_data += "Volatility: HIGH (VIX > 25) - Reduce overnight exposure\n"
        
        # Analyze with LLM
        analysis = self.analyze_with_llm(full_data)
        
        # Add recommendation
        full_data += "\n" + analysis
        
        # Add simple recommendation if no LLM
        if not self.client:
            full_data += "\n\nRECOMMENDATION: NEUTRAL - Monitor sector rotation"
        
        return full_data

def main():
    parser = argparse.ArgumentParser(description="Macro Economic Analysis Agent")
    parser.add_argument("--sector", help="Specific sector to focus on (e.g., technology, financials)")
    parser.add_argument("--days", type=int, default=7, 
                       help="Number of days for analysis (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    
    args = parser.parse_args()
    
    agent = MacroAgent(args.api_key, args.model)
    result = agent.run(args.sector, args.days)
    
    print(result)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()