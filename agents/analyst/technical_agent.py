"""
Technical Analysis Agent - Standalone CLI Tool
Updated to properly handle --days parameter
Usage: python technical_agent.py AAPL --days 7 --output report.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI

class TechnicalAgent:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are an EOD trading technical analyst. Analyze price action and technical indicators.
Focus on: 1) Trend direction 2) Support/resistance 3) Momentum 4) Volume confirmation 5) Risk/reward setup.
Provide specific entry, target, stop-loss levels. End with: RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low"""

    def get_price_data(self, days=7):
        """Fetch OHLCV data for specified days"""
        stock = yf.Ticker(self.ticker)
        
        # For short periods, use specific day count
        if days <= 7:
            df = stock.history(period="7d")
        elif days <= 30:
            df = stock.history(period="1mo")
        elif days <= 90:
            df = stock.history(period="3mo")
        else:
            df = stock.history(period=f"{days}d")
        
        # Ensure we have at least some data
        if df.empty:
            print(f"Warning: No data found for {self.ticker}, trying alternative period...")
            df = stock.history(period="1mo")
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators (adjusted for shorter periods)"""
        if df.empty or len(df) < 2:
            return df
        
        # Adjust indicator periods for short data
        data_length = len(df)
        
        # RSI (use shorter period if needed)
        rsi_period = min(14, data_length - 1)
        if rsi_period > 1:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            df['RSI'] = 50  # Neutral if not enough data
        
        # Moving Averages (adjust periods based on available data)
        if data_length >= 5:
            df['SMA_5'] = df['Close'].rolling(5).mean()
        if data_length >= 10:
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        if data_length >= 20:
            df['SMA_20'] = df['Close'].rolling(20).mean()
        if data_length >= 50:
            df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # MACD (only if enough data)
        if data_length >= 26:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        else:
            # Simple momentum for short periods
            df['MACD'] = df['Close'].pct_change(min(3, data_length-1))
            df['Signal'] = df['MACD'].rolling(min(3, data_length-1)).mean()
        
        # Bollinger Bands (adjust period)
        bb_period = min(20, max(5, data_length - 1))
        if bb_period > 1:
            df['BB_Mid'] = df['Close'].rolling(bb_period).mean()
            bb_std = df['Close'].rolling(bb_period).std()
            df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
        
        # ATR (adjusted)
        atr_period = min(14, max(3, data_length - 1))
        if 'High' in df.columns and 'Low' in df.columns and atr_period > 1:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['ATR'] = ranges.max(axis=1).rolling(atr_period).mean()
        else:
            # Approximate ATR using close prices only
            df['ATR'] = df['Close'].pct_change().abs().rolling(atr_period).mean() * df['Close']
        
        return df
    
    def analyze_with_llm(self, data_summary):
        """Use LLM to analyze the data"""
        if not self.client:
            return self.fallback_analysis(data_summary)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this technical data for {self.ticker}:\n{data_summary}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.fallback_analysis(data_summary)
    
    def fallback_analysis(self, data_summary):
        """Rule-based fallback when no LLM available"""
        lines = data_summary.split('\n')
        
        # Try to extract RSI value
        rsi_val = 50  # Default neutral
        for line in lines:
            if 'RSI:' in line:
                try:
                    rsi_str = line.split('RSI:')[1].split()[0]
                    rsi_val = float(rsi_str)
                except:
                    pass
        
        # Simple decision based on RSI
        if rsi_val < 30:
            rec = "BUY"
            conf = "Medium"
            reason = "oversold conditions"
        elif rsi_val > 70:
            rec = "SELL"
            conf = "Medium"
            reason = "overbought conditions"
        else:
            rec = "HOLD"
            conf = "Low"
            reason = "neutral technical indicators"
        
        return f"""Technical Analysis for {self.ticker}:
{data_summary}

Based on the indicators, the stock shows {reason}.
RECOMMENDATION: {rec} - Confidence: {conf}"""
    
    def run(self, days=7):
        """Execute full analysis"""
        print(f"Analyzing {self.ticker} with {days}-day lookback...")
        
        # Get data
        df = self.get_price_data(days)
        
        if df.empty:
            return f"Error: No data available for {self.ticker}"
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Calculate changes
        if len(df) >= 5:
            week_ago = df.iloc[-5]
            week_change = ((latest['Close']/week_ago['Close']-1)*100)
        elif len(df) >= 2:
            prev = df.iloc[0]
            week_change = ((latest['Close']/prev['Close']-1)*100)
        else:
            week_change = 0
        
        # Build summary based on available indicators
        summary = f"""
Technical Analysis Summary for {self.ticker}:
Analysis Period: {days} days
Date: {latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else 'Latest'}
Data Points Available: {len(df)} days

PRICE ACTION:
- Current Price: ${latest['Close']:.2f}
- Period Change: {week_change:.2f}%
- Volume: {latest['Volume']:,.0f}
"""
        
        # Add indicators that are available
        summary += "\nINDICATORS:\n"
        
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi_val = latest['RSI']
            summary += f"- RSI({min(14, len(df)-1)}): {rsi_val:.2f} "
            if rsi_val < 30:
                summary += "(Oversold - Bullish Signal)\n"
            elif rsi_val > 70:
                summary += "(Overbought - Bearish Signal)\n"
            else:
                summary += "(Neutral)\n"
        
        if 'MACD' in df.columns and 'Signal' in df.columns:
            if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal']):
                macd_signal = "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish"
                summary += f"- MACD: {latest['MACD']:.4f} vs Signal: {latest['Signal']:.4f} ({macd_signal})\n"
        
        # Add available moving averages
        for ma in ['SMA_5', 'EMA_10', 'SMA_20', 'SMA_50']:
            if ma in df.columns and not pd.isna(latest.get(ma)):
                ma_val = latest[ma]
                position = "Above" if latest['Close'] > ma_val else "Below"
                summary += f"- Price vs {ma}: ${latest['Close']:.2f} vs ${ma_val:.2f} ({position})\n"
        
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            if not pd.isna(latest.get('BB_Upper')) and not pd.isna(latest.get('BB_Lower')):
                summary += f"- Bollinger Bands: ${latest['BB_Lower']:.2f} < ${latest['Close']:.2f} < ${latest['BB_Upper']:.2f}\n"
                
                # Add BB position analysis
                bb_range = latest['BB_Upper'] - latest['BB_Lower']
                bb_position = (latest['Close'] - latest['BB_Lower']) / bb_range if bb_range > 0 else 0.5
                if bb_position > 0.8:
                    summary += "  (Near upper band - Potential resistance)\n"
                elif bb_position < 0.2:
                    summary += "  (Near lower band - Potential support)\n"
        
        if 'ATR' in df.columns and not pd.isna(latest.get('ATR')):
            summary += f"- ATR (Volatility): ${latest['ATR']:.2f}\n"
        
        # Add data quality note for short periods
        if len(df) < 20:
            summary += f"\nNote: Limited to {len(df)} days of data. Some indicators may be less reliable.\n"
        
        # Get LLM analysis
        analysis = self.analyze_with_llm(summary)
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="Technical Analysis Agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    
    args = parser.parse_args()
    
    agent = TechnicalAgent(args.ticker, args.api_key, args.model)
    result = agent.run(args.days)
    
    print(result)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()