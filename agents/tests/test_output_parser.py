# agents/tests/test_output_parser.py

"""
Tests for the output parser.
Run this to verify the parser works correctly.

Usage:
    cd agents
    python tests/test_output_parser.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import game_theory
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_theory.output_parser import OutputParser


def test_decision_parsing():
    """Test parsing Risk Manager decisions"""
    
    print("\n" + "="*60)
    print("TEST 1: Parsing Risk Manager Decision")
    print("="*60)
    
    parser = OutputParser()
    
    # Sample text that looks like your Risk Manager output
    sample_text = """
    After careful analysis of all risk factors and market conditions, I recommend 
    BUY with a position size of $7,500 (7.5% of portfolio) with 73% confidence.
    
    The technical setup is strong with multiple analysts agreeing on bullish signals.
    However, we should remain cautious of potential volatility.
    
    FINAL TRANSACTION PROPOSAL: **BUY**
    """
    
    decision = parser.parse_final_decision(sample_text)
    
    print(f"✓ Extracted Action: {decision.action}")
    print(f"✓ Extracted Confidence: {decision.confidence:.1%}")
    print(f"✓ Extracted Position ($): ${decision.position_size_dollars:,.2f}")
    print(f"✓ Extracted Position (%): {decision.position_size_pct}%")
    print(f"✓ Extracted Reasoning: {decision.reasoning[:100]}...")
    
    # Verify correctness
    assert decision.action == "BUY", "Action should be BUY"
    assert decision.confidence == 0.73, "Confidence should be 0.73"
    assert decision.position_size_dollars == 7500.0, "Position should be $7500"
    assert decision.position_size_pct == 7.5, "Position % should be 7.5"
    
    print("\n✅ All decision parsing tests passed!")


def test_analyst_consensus():
    """Test parsing analyst recommendations and calculating consensus"""
    
    print("\n" + "="*60)
    print("TEST 2: Parsing Analyst Reports & Consensus")
    print("="*60)
    
    parser = OutputParser()
    
    # Sample analyst reports (3 BUY, 1 HOLD)
    market_report = """
    Strong technical setup with price above key moving averages.
    FINAL TRANSACTION PROPOSAL: **BUY**
    """
    
    sentiment_report = """
    Social media sentiment is overwhelmingly positive.
    FINAL TRANSACTION PROPOSAL: **BUY**
    """
    
    news_report = """
    Recent news is mixed with both positive and negative factors.
    FINAL TRANSACTION PROPOSAL: **HOLD**
    """
    
    fundamentals_report = """
    Strong earnings growth and healthy balance sheet.
    FINAL TRANSACTION PROPOSAL: **BUY**
    """
    
    consensus = parser.parse_analyst_reports(
        market_report=market_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report
    )
    
    print(f"✓ Market Analyst: {consensus.market_rec}")
    print(f"✓ Sentiment Analyst: {consensus.sentiment_rec}")
    print(f"✓ News Analyst: {consensus.news_rec}")
    print(f"✓ Fundamentals Analyst: {consensus.fundamentals_rec}")
    print(f"\n✓ Consensus Score: {consensus.consensus_score:.1%}")
    print(f"✓ Majority Recommendation: {consensus.majority_rec}")
    
    # Verify correctness
    assert consensus.market_rec == "BUY"
    assert consensus.sentiment_rec == "BUY"
    assert consensus.news_rec == "HOLD"
    assert consensus.fundamentals_rec == "BUY"
    assert consensus.consensus_score == 0.75, "3 out of 4 = 75% consensus"
    assert consensus.majority_rec == "BUY"
    
    print("\n✅ All analyst consensus tests passed!")


def test_edge_cases():
    """Test edge cases and different formats"""
    
    print("\n" + "="*60)
    print("TEST 3: Edge Cases & Different Formats")
    print("="*60)
    
    parser = OutputParser()
    
    # Test 1: SELL decision with no confidence mentioned
    text1 = "I recommend SELL. The risk is too high."
    decision1 = parser.parse_final_decision(text1)
    print(f"✓ SELL with no confidence: {decision1.action}, {decision1.confidence:.1%}")
    assert decision1.action == "SELL"
    assert 0.6 <= decision1.confidence <= 0.7  # Should default to moderate
    
    # Test 2: LONG format (trading mode)
    text2 = "FINAL TRANSACTION PROPOSAL: **LONG**"
    decision2 = parser.parse_final_decision(text2)
    print(f"✓ LONG normalized to: {decision2.action}")
    assert decision2.action == "BUY"  # LONG should map to BUY
    
    # Test 3: Position as percentage only
    text3 = "Recommend BUY at 10% position size with 80% confidence"
    decision3 = parser.parse_final_decision(text3)
    print(f"✓ Percentage only: {decision3.position_size_pct}%")
    assert decision3.position_size_pct == 10.0
    
    print("\n✅ All edge case tests passed!")


def run_all_tests():
    """Run all tests"""
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "OUTPUT PARSER TESTS" + " "*24 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        test_decision_parsing()
        test_analyst_consensus()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nThe output parser is working correctly!")
        print("You can now use it to parse your agent outputs.\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)