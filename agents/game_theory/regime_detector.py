"""
regime_detector.py - Detect Market Regime from Price Returns

Location: agents/game_theory/regime_detector.py

This module classifies the current market regime as:
- "bull"     : Strong upward momentum (cumulative return > +3%)
- "bear"     : Strong downward momentum (cumulative return < -3%)
- "sideways" : Ranging/neutral market (between -3% and +3%)

The regime is used by strategies to adjust their behavior:
- Aggressive strategies might increase position in bull markets
- Defensive strategies might reduce exposure in bear markets
- Adaptive strategies (Tit-for-Tat) might change who they mirror

Usage:
    from game_theory.regime_detector import RegimeDetector
    
    detector = RegimeDetector()
    
    # Detect regime from list of daily returns
    returns = [0.01, 0.02, -0.005, 0.015, 0.008]  # 1%, 2%, -0.5%, etc.
    regime = detector.detect(returns)
    print(f"Current regime: {regime}")  # "bull"
    
    # Custom thresholds
    detector = RegimeDetector(
        lookback=20,           # Use last 20 returns
        bull_threshold=0.05,   # +5% for bull
        bear_threshold=-0.05   # -5% for bear
    )
"""

from typing import List, Optional
import numpy as np


class RegimeDetector:
    """
    Detect market regime based on recent cumulative returns.
    
    Uses a simple but effective approach:
    1. Look at the last N returns (default 10)
    2. Sum them to get cumulative return
    3. Classify based on thresholds
    
    This approach is:
    - Fast to compute (no complex indicators)
    - Intuitive to understand
    - Robust to noise (uses cumulative, not single-day)
    
    Attributes:
        lookback: Number of recent returns to consider
        bull_threshold: Cumulative return threshold for bull market
        bear_threshold: Cumulative return threshold for bear market
    """
    
    def __init__(
        self, 
        lookback: int = 10, 
        bull_threshold: float = 0.03,
        bear_threshold: float = -0.03
    ):
        """
        Initialize RegimeDetector.
        
        Args:
            lookback: Number of recent returns to use (default 10)
            bull_threshold: Cumulative return above this = bull (default +3%)
            bear_threshold: Cumulative return below this = bear (default -3%)
            
        Example thresholds:
            Conservative: bull=0.05, bear=-0.05 (less sensitive)
            Aggressive: bull=0.02, bear=-0.02 (more sensitive)
        """
        self.lookback = lookback
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
    
    def detect(self, returns: List[float]) -> str:
        """
        Classify market regime based on recent returns.
        
        Args:
            returns: List of daily returns as decimals 
                    (e.g., 0.01 = 1%, -0.02 = -2%)
                    Most recent return should be LAST in the list.
        
        Returns:
            "bull", "bear", or "sideways"
            
        Examples:
            >>> detector = RegimeDetector()
            >>> detector.detect([0.01, 0.02, 0.015, 0.01, 0.005])
            'bull'  # Cumulative ~6% > 3%
            
            >>> detector.detect([-0.02, -0.015, -0.01, 0.005, -0.01])
            'bear'  # Cumulative ~-5% < -3%
            
            >>> detector.detect([0.01, -0.01, 0.005, -0.005, 0.002])
            'sideways'  # Cumulative ~0.2%, between -3% and +3%
        """
        # Need at least a few data points for meaningful detection
        if not returns or len(returns) < 3:
            return "sideways"
        
        # Get recent returns (up to lookback period)
        recent = returns[-self.lookback:] if len(returns) >= self.lookback else returns
        
        # Calculate cumulative return
        cumulative = np.sum(recent)
        
        # Classify regime
        if cumulative > self.bull_threshold:
            return "bull"
        elif cumulative < self.bear_threshold:
            return "bear"
        else:
            return "sideways"
    
    def detect_with_details(self, returns: List[float]) -> dict:
        """
        Detect regime and return detailed information.
        
        Args:
            returns: List of daily returns as decimals
            
        Returns:
            Dictionary with:
                - regime: "bull", "bear", or "sideways"
                - cumulative_return: Sum of recent returns
                - lookback_used: How many returns were used
                - volatility: Std dev of recent returns
                - trend_strength: How far from sideways threshold
        """
        if not returns or len(returns) < 3:
            return {
                "regime": "sideways",
                "cumulative_return": 0.0,
                "lookback_used": 0,
                "volatility": 0.0,
                "trend_strength": 0.0
            }
        
        recent = returns[-self.lookback:] if len(returns) >= self.lookback else returns
        cumulative = float(np.sum(recent))
        volatility = float(np.std(recent))
        
        # Determine regime
        regime = self.detect(returns)
        
        # Calculate trend strength (distance from neutral zone)
        if cumulative > self.bull_threshold:
            trend_strength = (cumulative - self.bull_threshold) / abs(self.bull_threshold)
        elif cumulative < self.bear_threshold:
            trend_strength = (cumulative - self.bear_threshold) / abs(self.bear_threshold)
        else:
            # In sideways, strength is how close to breaking out
            dist_to_bull = self.bull_threshold - cumulative
            dist_to_bear = cumulative - self.bear_threshold
            trend_strength = -min(dist_to_bull, dist_to_bear) / abs(self.bull_threshold)
        
        return {
            "regime": regime,
            "cumulative_return": round(cumulative, 4),
            "lookback_used": len(recent),
            "volatility": round(volatility, 4),
            "trend_strength": round(trend_strength, 2)
        }
    
    def get_regime_history(self, returns: List[float]) -> List[str]:
        """
        Get regime for each point in the return series.
        
        Useful for analyzing how regime changed over time.
        
        Args:
            returns: Full list of daily returns
            
        Returns:
            List of regime strings, same length as returns
        """
        regimes = []
        
        for i in range(len(returns)):
            # Use returns up to this point
            returns_so_far = returns[:i+1]
            regime = self.detect(returns_so_far)
            regimes.append(regime)
        
        return regimes
    
    def get_regime_counts(self, returns: List[float]) -> dict:
        """
        Count how many periods were in each regime.
        
        Args:
            returns: Full list of daily returns
            
        Returns:
            Dictionary with counts for each regime
        """
        regimes = self.get_regime_history(returns)
        
        return {
            "bull": regimes.count("bull"),
            "bear": regimes.count("bear"),
            "sideways": regimes.count("sideways"),
            "total": len(regimes)
        }


# === Alternative Regime Detectors ===

class VolatilityRegimeDetector:
    """
    Alternative detector that also considers volatility.
    
    Classifies into 6 regimes:
    - bull_low_vol, bull_high_vol
    - bear_low_vol, bear_high_vol  
    - sideways_low_vol, sideways_high_vol
    """
    
    def __init__(
        self,
        lookback: int = 10,
        bull_threshold: float = 0.03,
        bear_threshold: float = -0.03,
        vol_threshold: float = 0.015
    ):
        self.lookback = lookback
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.vol_threshold = vol_threshold
    
    def detect(self, returns: List[float]) -> str:
        """Detect regime including volatility component."""
        if not returns or len(returns) < 3:
            return "sideways_low_vol"
        
        recent = returns[-self.lookback:] if len(returns) >= self.lookback else returns
        cumulative = np.sum(recent)
        volatility = np.std(recent)
        
        # Trend direction
        if cumulative > self.bull_threshold:
            trend = "bull"
        elif cumulative < self.bear_threshold:
            trend = "bear"
        else:
            trend = "sideways"
        
        # Volatility level
        vol_level = "high_vol" if volatility > self.vol_threshold else "low_vol"
        
        return f"{trend}_{vol_level}"


class MomentumRegimeDetector:
    """
    Alternative detector using momentum (rate of change).
    
    Looks at whether returns are accelerating or decelerating.
    """
    
    def __init__(self, short_lookback: int = 5, long_lookback: int = 15):
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
    
    def detect(self, returns: List[float]) -> str:
        """Detect regime based on momentum."""
        if not returns or len(returns) < self.long_lookback:
            return "sideways"
        
        short_avg = np.mean(returns[-self.short_lookback:])
        long_avg = np.mean(returns[-self.long_lookback:])
        
        # Compare short-term to long-term
        if short_avg > long_avg * 1.5 and short_avg > 0:
            return "bull"  # Accelerating up
        elif short_avg < long_avg * 1.5 and short_avg < 0:
            return "bear"  # Accelerating down
        else:
            return "sideways"


# === Quick test when run directly ===

if __name__ == "__main__":
    print("Testing RegimeDetector...")
    print("=" * 50)
    
    detector = RegimeDetector()
    
    # Test cases
    test_cases = [
        ("Bull market", [0.01, 0.015, 0.02, 0.01, 0.008, 0.012]),
        ("Bear market", [-0.02, -0.015, -0.01, -0.008, -0.012, -0.01]),
        ("Sideways", [0.005, -0.003, 0.002, -0.004, 0.001, -0.002]),
        ("Recovery", [-0.02, -0.01, 0.005, 0.01, 0.015, 0.02]),
        ("Selloff", [0.02, 0.01, -0.005, -0.015, -0.02, -0.01]),
    ]
    
    for name, returns in test_cases:
        result = detector.detect_with_details(returns)
        print(f"\n{name}:")
        print(f"  Returns: {[f'{r:.1%}' for r in returns]}")
        print(f"  Regime: {result['regime']}")
        print(f"  Cumulative: {result['cumulative_return']:.2%}")
        print(f"  Volatility: {result['volatility']:.2%}")
    
    print("\n" + "=" * 50)
    print("RegimeDetector test complete!")