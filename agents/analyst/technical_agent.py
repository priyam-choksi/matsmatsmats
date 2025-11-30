"""
Technical Analysis Agent - Enhanced Version
Comprehensive technical analysis with multiple indicators and pattern recognition

MODIFIED: Now supports historical backtesting via analysis_date parameter
ENHANCED: Added _get_llm_decision() for better recommendation extraction

Usage: 
  python technical_agent.py AAPL --days 7
  python technical_agent.py AAPL --days 7 --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI

# FIX: Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class TechnicalAgent:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Historical backtesting support
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[TECHNICAL] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[TECHNICAL] Running in LIVE mode (current data)")
        
        # Enhanced system prompt
        self.system_prompt = """You are an expert technical analyst evaluating price action and momentum for trading decisions.

**YOUR ANALYSIS FRAMEWORK:**

1. **Trend Analysis:**
   - Primary Trend: Identify overall direction (uptrend/downtrend/sideways)
   - Trend Strength: Strong, Moderate, or Weak based on moving average alignment
   - Moving Averages: Price position relative to SMA_20, SMA_50
     - Bullish: Price > SMA_20 > SMA_50 (golden cross setup)
     - Bearish: Price < SMA_20 < SMA_50 (death cross setup)
   - Trend Quality: Clean trends or choppy action?

2. **Momentum Indicators:**
   - RSI (Relative Strength Index):
     - <30 = Oversold (potential bounce, especially in uptrends)
     - >70 = Overbought (potential pullback, especially in downtrends)
     - 40-60 = Neutral zone
     - Divergences: Price vs RSI disagreement (reversal signal)
   - MACD (Moving Average Convergence Divergence):
     - MACD > Signal = Bullish momentum
     - MACD < Signal = Bearish momentum
     - Crossovers = Momentum shifts
     - Histogram expansion = Strengthening trend

3. **Support & Resistance:**
   - Key Price Levels: Recent highs/lows that could act as barriers
   - Bollinger Bands:
     - Near upper band = Resistance zone (potential reversal)
     - Near lower band = Support zone (potential bounce)
     - Middle band = Dynamic support/resistance
     - Band width = Volatility (narrow = breakout potential, wide = high volatility)

4. **Volatility Assessment:**
   - ATR (Average True Range): Measure of daily volatility
   - High ATR = Large price swings, higher risk, wider stops needed
   - Low ATR = Quiet market, tighter stops possible

5. **Volume Confirmation:**
   - Volume Trends: Rising on up days = healthy, rising on down days = distribution
   - Above Average Volume: Strong conviction in price move
   - Below Average Volume: Weak move, likely to reverse

6. **Risk/Reward Setup:**
   - Entry Point: Optimal entry based on current price vs support/resistance
   - Stop Loss: Logical level below support (for longs) or above resistance (for shorts)
   - Target Levels: Based on resistance (longs) or support (shorts)
   - Risk/Reward Ratio: Minimum 2:1 for quality setups

**DECISION CRITERIA:**

**BUY Signals (Long Setup):**
- Uptrend confirmed (price > moving averages)
- RSI 30-50 (oversold recovery or healthy pullback)
- MACD bullish crossover or positive
- Price near support level with volume
- Risk/reward > 2:1

**SELL Signals (Short Setup or Exit):**
- Downtrend confirmed (price < moving averages)
- RSI 50-70 or overbought >70
- MACD bearish crossover or negative
- Price near resistance with declining volume
- Breakdown below key support

**HOLD Signals (Wait for Setup):**
- Sideways/choppy price action
- RSI 40-60 (neutral zone)
- No clear support/resistance nearby
- Mixed indicator signals
- Poor risk/reward ratio (<1.5:1)

**CRITICAL OUTPUT REQUIREMENTS:**

## Technical Analysis Summary
[2-3 sentence overview of price action and trend]

## Trend Assessment
- Primary Trend: [Uptrend/Downtrend/Sideways]
- Trend Strength: [Strong/Moderate/Weak]
- Moving Average Alignment: [Bullish/Bearish/Neutral]

## Momentum Indicators
- RSI: [Value and interpretation]
- MACD: [Signal and direction]
- Volume: [Confirmation or divergence]

## Support & Resistance Levels
- Key Support: $[price]
- Key Resistance: $[price]
- Current Position: [Near support/resistance/neutral]

## Volatility & Risk
- ATR: $[value] ([High/Normal/Low] volatility)
- Bollinger Bands: [Position and interpretation]

## Trade Setup (If Applicable)
**Entry:** $[price]
**Stop Loss:** $[price]
**Target:** $[price]
**Risk/Reward:** [ratio]

RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low

**IMPORTANT:** 
- Provide specific price levels for entry, stop, target
- Calculate actual risk/reward ratios
- Reference specific indicator values
- Note if data is limited and adjust confidence accordingly"""

    def get_price_data(self, days: int = 7) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV price data
        If analysis_date is set, fetches HISTORICAL data ending on that date
        """
        print(f"[TECHNICAL] Fetching {days}-day price data for {self.ticker}...")
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # === HISTORICAL MODE ===
            if self.analysis_date:
                end_date = datetime.strptime(self.analysis_date, '%Y-%m-%d')
                start_date = end_date - timedelta(days=days * 2 + 60)
                
                df = stock.history(
                    start=start_date.strftime('%Y-%m-%d'), 
                    end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
                )
                
                if df.empty:
                    print(f"[TECHNICAL] ⚠️ No historical data for {self.analysis_date}")
                    return None
                
                print(f"[TECHNICAL] ✓ Historical: {len(df)} days ending {self.analysis_date}")
                print(f"[TECHNICAL]   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                
            # === CURRENT MODE ===
            else:
                if days <= 7:
                    period = "7d"
                elif days <= 30:
                    period = "1mo"
                elif days <= 90:
                    period = "3mo"
                elif days <= 180:
                    period = "6mo"
                else:
                    period = "1y"
                
                df = stock.history(period=period)
                
                if df.empty:
                    print(f"[TECHNICAL] ⚠️ No data for period '{period}', trying 1mo fallback...")
                    df = stock.history(period="1mo")
            
            if df.empty:
                print(f"[TECHNICAL] ❌ No data available for {self.ticker}")
                return None
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"[TECHNICAL] ⚠️ Missing required columns")
                return None
            
            print(f"[TECHNICAL] ✓ Retrieved {len(df)} days of data")
            return df
            
        except Exception as e:
            print(f"[TECHNICAL] ❌ Error fetching data: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 2:
            print(f"[TECHNICAL] ⚠️ Insufficient data for indicators")
            return df
        
        print(f"[TECHNICAL] Calculating technical indicators...")
        data_length = len(df)
        
        # RSI
        rsi_period = min(14, data_length - 1)
        if rsi_period >= 2:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)
        else:
            df['RSI'] = 50
        
        # Moving Averages
        if data_length >= 5:
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
        if data_length >= 10:
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        if data_length >= 20:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        if data_length >= 50:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        if data_length >= 26:
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        elif data_length >= 12:
            ema_short = df['Close'].ewm(span=min(6, data_length//2), adjust=False).mean()
            ema_long = df['Close'].ewm(span=min(12, data_length-1), adjust=False).mean()
            df['MACD'] = ema_short - ema_long
            df['MACD_Signal'] = df['MACD'].ewm(span=min(5, data_length//3), adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_period = min(20, data_length - 1) if data_length > 5 else data_length - 1
        if bb_period >= 2:
            df['BB_Mid'] = df['Close'].rolling(window=bb_period).mean()
            bb_std = df['Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
            df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']) * 100
        
        # ATR
        atr_period = min(14, data_length - 1) if data_length > 2 else 1
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=atr_period).mean()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # Stochastic
        if data_length >= 14:
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['Stochastic_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        
        # Volume Analysis
        vol_period = min(20, data_length - 1) if data_length > 5 else data_length - 1
        if vol_period >= 2:
            df['Volume_SMA'] = df['Volume'].rolling(window=vol_period).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        print(f"[TECHNICAL] ✓ Indicators calculated")
        return df

    def identify_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """Identify key support and resistance levels"""
        print(f"[TECHNICAL] Identifying support/resistance levels...")
        
        try:
            lookback = min(lookback, len(df))
            recent_data = df.tail(lookback)
            
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            highs = recent_data['High'].nlargest(3).values
            lows = recent_data['Low'].nsmallest(3).values
            
            levels = {
                'resistance_strong': resistance,
                'resistance_weak': highs[1] if len(highs) > 1 else resistance,
                'support_strong': support,
                'support_weak': lows[1] if len(lows) > 1 else support,
                'current_price': df['Close'].iloc[-1]
            }
            
            print(f"[TECHNICAL] ✓ Support: ${support:.2f}, Resistance: ${resistance:.2f}")
            return levels
            
        except Exception as e:
            print(f"[TECHNICAL] ⚠️ Error identifying levels: {e}")
            return {}

    def calculate_price_targets(self, df: pd.DataFrame, levels: Dict[str, float]) -> Dict[str, Any]:
        """Calculate entry, stop, and target levels"""
        if not levels or 'current_price' not in levels:
            return {}
        
        try:
            current = levels['current_price']
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else current * 0.02
            
            if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]):
                sma_20 = df['SMA_20'].iloc[-1]
                is_uptrend = current > sma_20
            else:
                is_uptrend = True
            
            if is_uptrend:
                entry = current
                stop = current - (atr * 1.5)
                target = levels.get('resistance_strong', current * 1.05)
                risk = entry - stop
                reward = target - entry
                rr_ratio = reward / risk if risk > 0 else 0
                setup_type = "LONG"
            else:
                entry = current
                stop = current + (atr * 1.5)
                target = levels.get('support_strong', current * 0.95)
                risk = stop - entry
                reward = entry - target
                rr_ratio = reward / risk if risk > 0 else 0
                setup_type = "SHORT"
            
            return {
                'setup_type': setup_type,
                'entry': entry,
                'stop_loss': stop,
                'target': target,
                'risk_amount': abs(risk),
                'reward_amount': abs(reward),
                'risk_reward_ratio': rr_ratio
            }
            
        except Exception as e:
            print(f"[TECHNICAL] ⚠️ Error calculating targets: {e}")
            return {}

    def format_technical_summary(self, df: pd.DataFrame, levels: Dict, targets: Dict, days: int) -> str:
        """Format comprehensive technical data for LLM analysis"""
        latest = df.iloc[-1]
        data_length = len(df)
        current_price = latest['Close']
        
        if len(df) >= 5:
            period_start = df.iloc[max(0, len(df) - days)]
            period_change = ((latest['Close'] / period_start['Close']) - 1) * 100
        else:
            period_change = ((latest['Close'] / df.iloc[0]['Close']) - 1) * 100
        
        date_header = ""
        if self.analysis_date:
            date_header = f"""
**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date} ⚠️**
All data below is historical, ending on {self.analysis_date}.
Do NOT reference any price movements or events after this date.

"""
        
        summary = f"""{date_header}# Technical Analysis Data for {self.ticker}

**Analysis Period:** {days} days (Data: {data_length} days available)
**Date:** {latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else 'Latest'}

## Price Action
- **Current Price:** ${latest['Close']:.2f}
- **Period Change:** {period_change:+.2f}%
- **Day's Range:** ${latest['Low']:.2f} - ${latest['High']:.2f}
- **Volume:** {latest['Volume']:,.0f}
"""
        
        # Volume analysis
        if 'Volume_Ratio' in df.columns and not pd.isna(latest.get('Volume_Ratio')):
            vol_ratio = latest['Volume_Ratio']
            if vol_ratio > 1.5:
                vol_analysis = "HIGH (1.5x+ average) - Strong conviction"
            elif vol_ratio > 1.0:
                vol_analysis = "Above average - Good participation"
            elif vol_ratio > 0.7:
                vol_analysis = "Normal levels"
            else:
                vol_analysis = "LOW (<0.7x average) - Weak conviction"
            summary += f"- **Volume Assessment:** {vol_analysis}\n"
        
        summary += "\n## Momentum Indicators\n"
        
        # RSI
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi = latest['RSI']
            summary += f"- **RSI({min(14, data_length-1)}):** {rsi:.1f} "
            if rsi < 30:
                summary += "→ **OVERSOLD** (Potential bounce) ✓\n"
            elif rsi > 70:
                summary += "→ **OVERBOUGHT** (Potential pullback) ⚠️\n"
            elif rsi < 40:
                summary += "→ Slightly oversold\n"
            elif rsi > 60:
                summary += "→ Slightly overbought\n"
            else:
                summary += "→ Neutral zone\n"
        
        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if not pd.isna(latest.get('MACD')) and not pd.isna(latest.get('MACD_Signal')):
                macd = latest['MACD']
                signal = latest['MACD_Signal']
                hist = latest.get('MACD_Histogram', macd - signal)
                
                summary += f"- **MACD:** {macd:.3f} | Signal: {signal:.3f} | Histogram: {hist:.3f}\n"
                
                if macd > signal:
                    if hist > 0:
                        summary += "  → Bullish momentum ✓\n"
                    else:
                        summary += "  → Bullish but weakening\n"
                else:
                    if hist < 0:
                        summary += "  → Bearish momentum ⚠️\n"
                    else:
                        summary += "  → Bearish but stabilizing\n"
        
        # Stochastic
        if 'Stochastic_K' in df.columns and not pd.isna(latest.get('Stochastic_K')):
            stoch_k = latest['Stochastic_K']
            stoch_d = latest.get('Stochastic_D', stoch_k)
            summary += f"- **Stochastic:** %K={stoch_k:.1f}, %D={stoch_d:.1f}\n"
            if stoch_k < 20:
                summary += "  → Oversold territory\n"
            elif stoch_k > 80:
                summary += "  → Overbought territory\n"
        
        # Moving Averages
        summary += "\n## Moving Average Analysis\n"
        ma_alignment = []
        
        for ma_name in ['SMA_5', 'EMA_10', 'SMA_20', 'SMA_50']:
            if ma_name in df.columns and not pd.isna(latest.get(ma_name)):
                ma_val = latest[ma_name]
                above = current_price > ma_val
                ma_alignment.append(above)
                status = "Above ✓" if above else "Below ⚠️"
                summary += f"- **{ma_name}:** ${ma_val:.2f} ({status})\n"
        
        # Trend assessment
        if ma_alignment:
            bullish_count = sum(ma_alignment)
            if bullish_count == len(ma_alignment):
                summary += "\n**Trend:** Strong uptrend (price above all MAs) ✓\n"
            elif bullish_count > len(ma_alignment) / 2:
                summary += "\n**Trend:** Uptrend (price above most MAs)\n"
            elif bullish_count == 0:
                summary += "\n**Trend:** Strong downtrend (price below all MAs) ⚠️\n"
            else:
                summary += "\n**Trend:** Mixed/Choppy (price near MAs)\n"
        
        summary += "\n## Volatility & Range\n"
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Mid']):
            if not any(pd.isna(latest.get(col)) for col in ['BB_Upper', 'BB_Lower', 'BB_Mid']):
                bb_upper = latest['BB_Upper']
                bb_lower = latest['BB_Lower']
                
                summary += f"- **Bollinger Bands:** ${bb_lower:.2f} < ${current_price:.2f} < ${bb_upper:.2f}\n"
                
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = (current_price - bb_lower) / bb_range
                    if bb_position > 0.9:
                        summary += "  → Near upper band (Resistance zone) ⚠️\n"
                    elif bb_position < 0.1:
                        summary += "  → Near lower band (Support zone) ✓\n"
                    else:
                        summary += f"  → Middle range ({bb_position*100:.0f}% position)\n"
                
                if 'BB_Width' in df.columns and not pd.isna(latest.get('BB_Width')):
                    bb_width = latest['BB_Width']
                    if bb_width < 5:
                        summary += f"  → Narrow bands ({bb_width:.1f}%) - Breakout potential\n"
                    elif bb_width > 15:
                        summary += f"  → Wide bands ({bb_width:.1f}%) - High volatility\n"
        
        # ATR
        if 'ATR' in df.columns and not pd.isna(latest.get('ATR')):
            atr = latest['ATR']
            atr_pct = latest.get('ATR_Pct', (atr / current_price) * 100)
            summary += f"- **ATR:** ${atr:.2f} ({atr_pct:.2f}% of price)\n"
            if atr_pct > 4:
                summary += "  → High volatility (wider stops needed) ⚠️\n"
            elif atr_pct < 2:
                summary += "  → Low volatility (tight stops possible) ✓\n"
            else:
                summary += "  → Normal volatility\n"
        
        # Support/Resistance
        if levels:
            summary += "\n## Support & Resistance\n"
            if 'resistance_strong' in levels:
                summary += f"- **Key Resistance:** ${levels['resistance_strong']:.2f}\n"
            if 'resistance_weak' in levels:
                summary += f"- **Secondary Resistance:** ${levels['resistance_weak']:.2f}\n"
            if 'support_strong' in levels:
                summary += f"- **Key Support:** ${levels['support_strong']:.2f}\n"
            if 'support_weak' in levels:
                summary += f"- **Secondary Support:** ${levels['support_weak']:.2f}\n"
            
            if 'resistance_strong' in levels:
                dist_to_resistance = ((levels['resistance_strong'] / current_price) - 1) * 100
                summary += f"\n- **Distance to Resistance:** {dist_to_resistance:+.1f}%\n"
            if 'support_strong' in levels:
                dist_to_support = ((current_price / levels['support_strong']) - 1) * 100
                summary += f"- **Distance to Support:** {dist_to_support:+.1f}%\n"
        
        # Trade setup
        if targets:
            summary += "\n## Potential Trade Setup\n"
            summary += f"- **Setup Type:** {targets.get('setup_type', 'N/A')}\n"
            summary += f"- **Entry:** ${targets.get('entry', 0):.2f}\n"
            summary += f"- **Stop Loss:** ${targets.get('stop_loss', 0):.2f} (Risk: ${targets.get('risk_amount', 0):.2f})\n"
            summary += f"- **Target:** ${targets.get('target', 0):.2f} (Reward: ${targets.get('reward_amount', 0):.2f})\n"
            summary += f"- **Risk/Reward Ratio:** {targets.get('risk_reward_ratio', 0):.2f}:1\n"
            
            rr_ratio = targets.get('risk_reward_ratio', 0)
            if rr_ratio >= 3:
                summary += "  → **Excellent** risk/reward ✓\n"
            elif rr_ratio >= 2:
                summary += "  → Good risk/reward ✓\n"
            elif rr_ratio >= 1:
                summary += "  → Acceptable risk/reward\n"
            else:
                summary += "  → Poor risk/reward ⚠️\n"
        
        if data_length < 20:
            summary += f"\n⚠️ **Note:** Limited data ({data_length} days) - indicators may be less reliable\n"
        
        return summary

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
Given a technical analysis, extract or determine the final recommendation.

RULES:
1. If there's an explicit "RECOMMENDATION: X" line, extract it
2. If not explicit, analyze the content and determine the most appropriate recommendation
3. Consider: trend direction, RSI levels, MACD signals, support/resistance proximity
4. Do NOT default to HOLD - make an actual decision based on the evidence

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
            
            print(f"[TECHNICAL] LLM Decision: {rec} ({conf})")
            return rec, conf
            
        except Exception as e:
            print(f"[TECHNICAL] ⚠️ LLM decision error: {e}, using fallback")
            return self._extract_recommendation_from_content(analysis)

    def _extract_recommendation_from_content(self, analysis: str) -> Tuple[str, str]:
        """Fallback: Extract recommendation using keyword analysis"""
        analysis_lower = analysis.lower()
        
        specific_buy_signals = ['uptrend', 'oversold', 'support level', 'bullish crossover', 'breakout']
        specific_sell_signals = ['downtrend', 'overbought', 'resistance level', 'bearish crossover', 'breakdown']
        
        if any(phrase in analysis_lower for phrase in ["recommend buy", "should buy", "buy signal", "final decision: buy"]):
            return "BUY", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend sell", "should sell", "sell signal", "final decision: sell"]):
            return "SELL", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend hold", "should hold", "wait", "neutral position"]):
            return "HOLD", "Low"
        
        buy_count = sum(1 for signal in specific_buy_signals if signal in analysis_lower)
        sell_count = sum(1 for signal in specific_sell_signals if signal in analysis_lower)
        
        general_buy = ["bullish", "positive", "upside", "growth", "strong"]
        general_sell = ["bearish", "negative", "downside", "decline", "weak"]
        
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

    def analyze_with_llm(self, technical_summary: str) -> str:
        """Send technical data to LLM for comprehensive analysis"""
        if not self.client:
            print("[TECHNICAL] ⚠️ No API key - using fallback analysis")
            return self._create_fallback_analysis(technical_summary)
        
        try:
            print(f"[TECHNICAL] Generating analysis with {self.model}...")
            
            date_note = ""
            if self.analysis_date:
                date_note = f"""
**⚠️ HISTORICAL ANALYSIS: {self.analysis_date} ⚠️**
The technical data below is historical, ending on {self.analysis_date}.
Analyze price action and indicators AS OF this date.
Do NOT reference any price movements or events after {self.analysis_date}.
Pretend today IS {self.analysis_date}.

"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{date_note}Provide comprehensive technical analysis for {self.ticker}:\n\n{technical_summary}"}
                ],
                temperature=0.7,
                max_completion_tokens=2000
            )
            
            analysis = response.choices[0].message.content
            
            # Use new LLM decision extraction if no formal recommendation
            if "RECOMMENDATION:" not in analysis:
                print(f"[TECHNICAL] ⚠️ Response missing formal recommendation, extracting...")
                recommendation, confidence = self._get_llm_decision(analysis)
                analysis += f"\n\nRECOMMENDATION: {recommendation} - Confidence: {confidence}"
            
            print(f"[TECHNICAL] ✓ Analysis generated ({len(analysis)} chars)")
            return analysis
            
        except Exception as e:
            print(f"[TECHNICAL] ❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(technical_summary)

    def _create_fallback_analysis(self, technical_summary: str) -> str:
        """Rule-based technical analysis fallback"""
        print("[TECHNICAL] Creating fallback analysis...")
        
        analysis = f"""## Technical Analysis
*Generated using fallback analysis (LLM unavailable)*

{technical_summary}

---

## Automated Assessment

"""
        
        rsi_value = None
        for line in technical_summary.split('\n'):
            if 'RSI' in line and ':' in line:
                try:
                    rsi_str = line.split(':')[1].split()[0]
                    rsi_value = float(rsi_str)
                except:
                    pass
        
        signals = []
        
        if rsi_value:
            if rsi_value < 30:
                signals.append(('BUY', 'RSI oversold'))
            elif rsi_value > 70:
                signals.append(('SELL', 'RSI overbought'))
        
        if 'Strong uptrend' in technical_summary:
            signals.append(('BUY', 'Strong uptrend'))
        elif 'Strong downtrend' in technical_summary:
            signals.append(('SELL', 'Strong downtrend'))
        
        if 'Above ✓' in technical_summary:
            ma_count = technical_summary.count('Above ✓')
            if ma_count >= 3:
                signals.append(('BUY', 'Price above MAs'))
        
        buy_signals = sum(1 for s in signals if s[0] == 'BUY')
        sell_signals = sum(1 for s in signals if s[0] == 'SELL')
        
        if buy_signals > sell_signals and buy_signals >= 2:
            recommendation = "BUY"
            confidence = "Medium"
            reasoning = f"Multiple bullish signals: {', '.join([s[1] for s in signals if s[0] == 'BUY'])}"
        elif sell_signals > buy_signals and sell_signals >= 2:
            recommendation = "SELL"
            confidence = "Medium"
            reasoning = f"Multiple bearish signals: {', '.join([s[1] for s in signals if s[0] == 'SELL'])}"
        else:
            recommendation = "HOLD"
            confidence = "Low"
            reasoning = "Mixed or weak signals"
        
        analysis += f"**Signals Detected:** {buy_signals} bullish, {sell_signals} bearish\n"
        analysis += f"**Reasoning:** {reasoning}\n\n"
        analysis += f"RECOMMENDATION: {recommendation} - Confidence: {confidence}\n"
        
        return analysis

    def run(self, days: int = 7) -> str:
        """Execute complete technical analysis workflow"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"TECHNICAL ANALYSIS: {self.ticker}")
        print(f"Analysis Period: {days} days")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        df = self.get_price_data(days)
        
        if df is None or df.empty:
            error_report = f"""# Technical Analysis Error

**Ticker:** {self.ticker}
**Error:** Unable to retrieve price data
{"**Analysis Date:** " + self.analysis_date if self.analysis_date else ""}

**Possible Causes:**
- Invalid ticker symbol
- Market closed / no recent trading
- Network connectivity issues
- yfinance API issues
{f"- No data available for historical date {self.analysis_date}" if self.analysis_date else ""}

RECOMMENDATION: HOLD - Confidence: N/A
"""
            print(f"[TECHNICAL] ❌ No data for {self.ticker}")
            return error_report
        
        df = self.calculate_indicators(df)
        levels = self.identify_support_resistance(df)
        targets = self.calculate_price_targets(df, levels)
        
        print(f"[TECHNICAL] Formatting technical summary...")
        technical_summary = self.format_technical_summary(df, levels, targets, days)
        
        analysis = self.analyze_with_llm(technical_summary)
        
        elapsed = time.time() - start_time
        print(f"\n[TECHNICAL] ✓ Analysis complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Technical Analysis Agent - Comprehensive price action and indicator analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python technical_agent.py AAPL
  python technical_agent.py MSFT --days 30
  python technical_agent.py GOOGL --days 7 --output tech_report.txt
  python technical_agent.py AAPL --days 7 --analysis-date 2024-06-15
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, MSFT)")
    parser.add_argument("--days", type=int, default=7, help="Analysis period in days (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--output", help="Save analysis to file")
    parser.add_argument("--analysis-date", type=str, default=None,
                       help="Historical analysis date (YYYY-MM-DD format)")
    
    args = parser.parse_args()
    
    try:
        agent = TechnicalAgent(
            ticker=args.ticker, 
            api_key=args.api_key, 
            model=args.model,
            analysis_date=args.analysis_date
        )
        result = agent.run(days=args.days)
        
        print(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Analysis saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()