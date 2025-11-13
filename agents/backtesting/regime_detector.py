# agents/backtesting/regime_detector.py

"""
Detects market regime (bull/bear/volatile/sideways/momentum/mean_reverting).
Uses technical indicators to classify current market conditions.

Example usage:
    from backtesting.data_fetcher import DataFetcher
    from backtesting.regime_detector import RegimeDetector
    
    fetcher = DataFetcher()
    detector = RegimeDetector()
    
    data = fetcher.get_price_data("AAPL", days=60)
    regime = detector.detect_regime(data)
    print(f"Market Regime: {regime}")
"""

import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class RegimeAnalysis:
    """
    Complete regime analysis with confidence scores.
    
    Attributes:
        regime: Primary regime classification
        confidence: 0.0 to 1.0 confidence in classification
        trend_direction: 'up', 'down', or 'neutral'
        volatility_level: 'low', 'medium', or 'high'
        momentum_strength: 0.0 to 1.0
        scores: Dict with scores for each regime type
    """
    regime: str
    confidence: float
    trend_direction: str
    volatility_level: str
    momentum_strength: float
    scores: Dict[str, float]
    
    def __str__(self):
        return (f"Regime: {self.regime} (confidence: {self.confidence:.1%})\n"
                f"Trend: {self.trend_direction}, "
                f"Volatility: {self.volatility_level}, "
                f"Momentum: {self.momentum_strength:.2f}")


class RegimeDetector:
    """
    Detects market regime using technical analysis.
    
    Classifies markets into 6 regimes:
    1. bull_trend - Strong uptrend, low volatility
    2. bear_trend - Strong downtrend, low volatility
    3. high_volatility - Large swings, no clear direction
    4. sideways - Range-bound, choppy
    5. momentum - Strong directional move with volume
    6. mean_reverting - Oversold/overbought, likely to reverse
    """
    
    def __init__(self):
        pass
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """
        Detect market regime from price data.
        
        Args:
            price_data: DataFrame with OHLCV data (from DataFetcher)
            
        Returns:
            String regime name: 'bull_trend', 'bear_trend', 'high_volatility',
                                'sideways', 'momentum', 'mean_reverting'
        """
        analysis = self.analyze_regime(price_data)
        return analysis.regime
    
    def analyze_regime(self, price_data: pd.DataFrame) -> RegimeAnalysis:
        """
        Perform full regime analysis with confidence scores.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            RegimeAnalysis object with detailed classification
        """
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(price_data)
        
        # Analyze trend
        trend_direction, trend_strength = self._analyze_trend(indicators)
        
        # Analyze volatility
        volatility_level, volatility_score = self._analyze_volatility(indicators)
        
        # Analyze momentum
        momentum_strength = self._analyze_momentum(indicators)
        
        # Analyze mean reversion potential
        mean_reversion_score = self._analyze_mean_reversion(indicators)
        
        # Calculate scores for each regime
        scores = self._calculate_regime_scores(
            trend_direction, trend_strength,
            volatility_level, volatility_score,
            momentum_strength, mean_reversion_score
        )
        
        # Determine primary regime
        regime = max(scores, key=scores.get)
        confidence = scores[regime]
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            momentum_strength=momentum_strength,
            scores=scores
        )
    
    # ========== Internal Methods ==========
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from price data"""
        
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Price position relative to SMAs
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        # Volatility (20-day)
        df['Returns'] = df['Close'].pct_change()
        volatility = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        current_volatility = volatility.iloc[-1]
        
        # ATR (Average True Range) for volatility
        df['HL'] = df['High'] - df['Low']
        df['HC'] = abs(df['High'] - df['Close'].shift(1))
        df['LC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        current_atr = df['ATR'].iloc[-1]
        atr_pct = (current_atr / current_price) * 100  # ATR as % of price
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        current_rsi = df['RSI'].iloc[-1]
        
        # Price change over different periods
        price_change_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        price_change_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
        
        # Volume trend
        avg_volume_20d = df['Volume'].rolling(window=20).mean().iloc[-1]
        recent_volume = df['Volume'].iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
        
        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'price_vs_sma20': (current_price / sma_20 - 1) * 100,
            'price_vs_sma50': (current_price / sma_50 - 1) * 100,
            'sma20_vs_sma50': (sma_20 / sma_50 - 1) * 100,
            'volatility': current_volatility,
            'atr_pct': atr_pct,
            'rsi': current_rsi,
            'price_change_5d': price_change_5d,
            'price_change_20d': price_change_20d,
            'volume_ratio': volume_ratio
        }
    
    def _analyze_trend(self, indicators: Dict) -> tuple:
        """
        Analyze trend direction and strength.
        Returns: (direction, strength) where direction is 'up'/'down'/'neutral'
                 and strength is 0.0 to 1.0
        """
        
        # Check SMA alignment
        sma_20_vs_50 = indicators['sma20_vs_sma50']
        price_vs_sma20 = indicators['price_vs_sma20']
        price_change_20d = indicators['price_change_20d']
        
        # Determine direction
        if sma_20_vs_50 > 2 and price_vs_sma20 > 0:
            direction = 'up'
            strength = min(abs(price_change_20d) / 20.0, 1.0)  # Normalize
        elif sma_20_vs_50 < -2 and price_vs_sma20 < 0:
            direction = 'down'
            strength = min(abs(price_change_20d) / 20.0, 1.0)
        else:
            direction = 'neutral'
            strength = 0.3
        
        return direction, strength
    
    def _analyze_volatility(self, indicators: Dict) -> tuple:
        """
        Analyze volatility level.
        Returns: (level, score) where level is 'low'/'medium'/'high'
                 and score is 0.0 to 1.0
        """
        
        vol = indicators['volatility']
        atr_pct = indicators['atr_pct']
        
        # Classify volatility
        if vol < 0.20 and atr_pct < 2.0:
            level = 'low'
            score = 0.2
        elif vol > 0.40 or atr_pct > 4.0:
            level = 'high'
            score = 0.9
        else:
            level = 'medium'
            score = 0.5
        
        return level, score
    
    def _analyze_momentum(self, indicators: Dict) -> float:
        """
        Analyze momentum strength (0.0 to 1.0).
        """
        
        price_change_5d = abs(indicators['price_change_5d'])
        volume_ratio = indicators['volume_ratio']
        
        # Strong momentum = big price move + high volume
        momentum = (price_change_5d / 10.0) * min(volume_ratio, 2.0)
        
        return min(momentum, 1.0)
    
    def _analyze_mean_reversion(self, indicators: Dict) -> float:
        """
        Analyze mean reversion potential (0.0 to 1.0).
        Higher score = more likely to revert
        """
        
        rsi = indicators['rsi']
        price_vs_sma20 = indicators['price_vs_sma20']
        
        # Extreme RSI or price far from SMA suggests reversion
        rsi_extreme = 0.0
        if rsi > 70:
            rsi_extreme = (rsi - 70) / 30.0  # 0 to 1 as RSI goes 70 to 100
        elif rsi < 30:
            rsi_extreme = (30 - rsi) / 30.0  # 0 to 1 as RSI goes 30 to 0
        
        price_extreme = min(abs(price_vs_sma20) / 10.0, 1.0)
        
        return (rsi_extreme + price_extreme) / 2.0
    
    def _calculate_regime_scores(self, 
                                  trend_dir: str, trend_strength: float,
                                  vol_level: str, vol_score: float,
                                  momentum: float, mean_reversion: float) -> Dict[str, float]:
        """
        Calculate scores for each regime type.
        Returns dict with scores 0.0 to 1.0 for each regime.
        """
        
        scores = {}
        
        # Bull Trend: up trend + low volatility
        if trend_dir == 'up' and vol_level == 'low':
            scores['bull_trend'] = trend_strength * 0.9
        else:
            scores['bull_trend'] = 0.1
        
        # Bear Trend: down trend + low/medium volatility
        if trend_dir == 'down' and vol_level in ['low', 'medium']:
            scores['bear_trend'] = trend_strength * 0.9
        else:
            scores['bear_trend'] = 0.1
        
        # High Volatility: high volatility + no clear trend
        if vol_level == 'high' and trend_dir == 'neutral':
            scores['high_volatility'] = vol_score
        elif vol_level == 'high':
            scores['high_volatility'] = vol_score * 0.7
        else:
            scores['high_volatility'] = 0.1
        
        # Sideways: neutral trend + low/medium volatility
        if trend_dir == 'neutral' and vol_level in ['low', 'medium']:
            scores['sideways'] = 0.8
        else:
            scores['sideways'] = 0.2
        
        # Momentum: strong directional move + high volume
        if trend_dir in ['up', 'down'] and momentum > 0.6:
            scores['momentum'] = momentum * trend_strength
        else:
            scores['momentum'] = momentum * 0.3
        
        # Mean Reverting: extreme price + potential for reversal
        scores['mean_reverting'] = mean_reversion
        
        return scores


# Convenience function
def quick_detect(symbol: str, days: int = 60) -> str:
    """Quick regime detection without creating objects"""
    from .data_fetcher import DataFetcher
    fetcher = DataFetcher()
    detector = RegimeDetector()
    data = fetcher.get_price_data(symbol, days)
    return detector.detect_regime(data)