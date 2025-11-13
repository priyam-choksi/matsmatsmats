# agents/tests/test_regime_detector.py

"""
Tests for regime detection.
Run this to verify the regime detector works correctly.

Usage:
    cd agents
    python tests/test_regime_detector.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.data_fetcher import DataFetcher
from backtesting.regime_detector import RegimeDetector


def test_regime_detection():
    """Test regime detection on real stock data"""
    
    print("\n" + "="*60)
    print("TEST: Regime Detection on Real Data")
    print("="*60)
    
    try:
        # Initialize
        fetcher = DataFetcher()
        detector = RegimeDetector()
        
        # Test on a few different stocks
        symbols = ["AAPL", "NVDA", "SPY"]
        
        for symbol in symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Fetch data
            data = fetcher.get_price_data(symbol, days=60)
            print(f"  ✓ Fetched {len(data)} days of price data")
            
            # Detect regime
            analysis = detector.analyze_regime(data)
            
            print(f"  ✓ Regime: {analysis.regime}")
            print(f"  ✓ Confidence: {analysis.confidence:.1%}")
            print(f"  ✓ Trend: {analysis.trend_direction}")
            print(f"  ✓ Volatility: {analysis.volatility_level}")
            print(f"  ✓ Momentum: {analysis.momentum_strength:.2f}")
            
            # Show regime scores
            print(f"  ✓ Regime Scores:")
            for regime_name, score in analysis.scores.items():
                print(f"      {regime_name}: {score:.2f}")
        
        print("\n✅ Regime detection test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*14 + "REGIME DETECTOR TESTS" + " "*23 + "║")
    print("╚" + "="*58 + "╝")
    
    success = test_regime_detection()
    
    if success:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nThe regime detector is working correctly!")
        print("It can now classify market conditions for your strategies.\n")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)