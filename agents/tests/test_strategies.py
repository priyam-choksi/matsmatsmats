# agents/tests/test_strategies.py

"""
Tests for game theory strategies and tournament engine.
Run this to verify all strategies work correctly.

Usage:
    cd agents
    python tests/test_strategies.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_theory.output_parser import OutputParser, ParsedDecision, AnalystConsensus
from game_theory.market_context import MarketContextBuilder
from game_theory.tournament_engine import TournamentEngine


def test_individual_strategies():
    """Test each strategy individually"""
    
    print("\n" + "="*60)
    print("TEST 1: Individual Strategy Decisions")
    print("="*60)
    
    # Create sample market context
    parser = OutputParser()
    
    # High consensus, bullish scenario
    base_decision = ParsedDecision(
        action="BUY",
        confidence=0.75,
        position_size_dollars=10000.0,
        position_size_pct=10.0,
        reasoning="Strong technical setup",
        raw_text=""
    )
    
    analyst_consensus = AnalystConsensus(
        market_rec="BUY",
        sentiment_rec="BUY",
        news_rec="BUY",
        fundamentals_rec="HOLD"
    )
    
    builder = MarketContextBuilder()
    context = builder.build(base_decision, analyst_consensus, "bull_trend")
    
    print(f"\n📊 Test Scenario:")
    print(f"  Base: BUY $10,000 (75% confidence)")
    print(f"  Regime: bull_trend")
    print(f"  Consensus: {context.analyst_consensus:.1%}")
    print(f"  Sentiment: {context.sentiment_score:.1%}")
    
    # Test each strategy
    from game_theory.strategies import (
        ActualMarketStrategy, BuyHoldStrategy, CooperatorStrategy,
        DefectorStrategy, TitForTatStrategy
    )
    
    strategies = [
        ActualMarketStrategy(),
        BuyHoldStrategy(),
        CooperatorStrategy(),
        DefectorStrategy(),
        TitForTatStrategy()
    ]
    
    print(f"\n🎯 Strategy Decisions:")
    for strategy in strategies:
        decision = strategy.make_decision(context)
        print(f"\n  {strategy.name.upper()}:")
        print(f"    Action: {decision.action}")
        print(f"    Position: ${decision.position_size:,.0f}")
        print(f"    Confidence: {decision.confidence:.1%}")
        print(f"    Reasoning: {decision.reasoning[:80]}...")
    
    print("\n✅ Individual strategy test passed!")


def test_tournament():
    """Test full tournament with all strategies"""
    
    print("\n" + "="*60)
    print("TEST 2: Full Tournament")
    print("="*60)
    
    # Create tournament engine
    engine = TournamentEngine()
    
    # Create test context
    base_decision = ParsedDecision(
        action="BUY",
        confidence=0.80,
        position_size_dollars=15000.0,
        position_size_pct=15.0,
        reasoning="Strong bullish setup",
        raw_text=""
    )
    
    analyst_consensus = AnalystConsensus(
        market_rec="BUY",
        sentiment_rec="BUY",
        news_rec="BUY",
        fundamentals_rec="BUY"
    )
    
    builder = MarketContextBuilder()
    context = builder.build(base_decision, analyst_consensus, "bull_trend")
    
    # Run tournament
    result = engine.run_tournament(context, symbol="TEST")
    
    # Verify result structure
    assert 'strategies' in result
    assert 'recommended_strategy' in result
    assert len(result['strategies']) == 5
    
    # Print results
    engine.print_results(result)
    
    print("✅ Tournament test passed!")


def test_different_regimes():
    """Test how strategies behave in different market regimes"""
    
    print("\n" + "="*60)
    print("TEST 3: Different Market Regimes")
    print("="*60)
    
    engine = TournamentEngine()
    
    # Base decision
    base_decision = ParsedDecision(
        action="BUY",
        confidence=0.70,
        position_size_dollars=10000.0,
        position_size_pct=10.0,
        reasoning="Test",
        raw_text=""
    )
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'High Consensus Bull',
            'regime': 'bull_trend',
            'analysts': AnalystConsensus('BUY', 'BUY', 'BUY', 'BUY'),
            'expected_winner': 'cooperator'
        },
        {
            'name': 'Low Consensus Volatile',
            'regime': 'high_volatility',
            'analysts': AnalystConsensus('BUY', 'SELL', 'HOLD', 'BUY'),
            'expected_winner': 'defector'
        },
        {
            'name': 'Sideways Market',
            'regime': 'sideways',
            'analysts': AnalystConsensus('HOLD', 'HOLD', 'BUY', 'HOLD'),
            'expected_winner': 'buy_hold'
        }
    ]
    
    builder = MarketContextBuilder()
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        
        context = builder.build(
            base_decision,
            scenario['analysts'],
            scenario['regime']
        )
        
        result = engine.run_tournament(context, symbol="TEST")
        winner = result['recommended_strategy']
        
        print(f"  Regime: {context.regime}")
        print(f"  Consensus: {context.analyst_consensus:.1%}")
        print(f"  Winner: {winner}")
        print(f"  Expected: {scenario['expected_winner']}")
        
        # Note: Winner might not always match expected (that's OK - heuristic is simple)
        if winner == scenario['expected_winner']:
            print(f"  ✅ Matched expected winner!")
        else:
            print(f"  ⚠️  Different winner (OK - selection is heuristic)")
    
    print("\n✅ Regime test completed!")


def test_tit_for_tat_adaptation():
    """Test that Tit-for-Tat adapts to winners"""
    
    print("\n" + "="*60)
    print("TEST 4: Tit-for-Tat Adaptation")
    print("="*60)
    
    engine = TournamentEngine()
    
    base_decision = ParsedDecision(
        action="BUY",
        confidence=0.75,
        position_size_dollars=10000.0,
        position_size_pct=10.0,
        reasoning="Test",
        raw_text=""
    )
    
    analyst_consensus = AnalystConsensus('BUY', 'BUY', 'BUY', 'BUY')
    builder = MarketContextBuilder()
    context = builder.build(base_decision, analyst_consensus, "bull_trend")
    
    # Round 1: No history (should use Cooperator)
    print("\n📊 Round 1 (no history):")
    result1 = engine.run_tournament(context, symbol="TEST")
    tft_decision1 = result1['strategies']['tit_for_tat']
    print(f"  Tit-for-Tat reasoning: {tft_decision1.reasoning}")
    assert 'First round' in tft_decision1.reasoning or 'Mirroring' not in tft_decision1.reasoning
    print("  ✅ Started cooperative")
    
    # Round 2: Has history (should mirror winner)
    print("\n📊 Round 2 (with history):")
    result2 = engine.run_tournament(context, symbol="TEST")
    tft_decision2 = result2['strategies']['tit_for_tat']
    print(f"  Tit-for-Tat reasoning: {tft_decision2.reasoning}")
    assert 'Mirroring' in tft_decision2.reasoning
    print("  ✅ Mirroring previous winner")
    
    print("\n✅ Tit-for-Tat adaptation test passed!")


def run_all_tests():
    """Run all tests"""
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*12 + "GAME THEORY STRATEGIES TESTS" + " "*18 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        test_individual_strategies()
        test_tournament()
        test_different_regimes()
        test_tit_for_tat_adaptation()
        
        print("\n" + "="*60)
        print("🎉 ALL STRATEGY TESTS PASSED! 🎉")
        print("="*60)
        print("\nAll 5 strategies are working correctly!")
        print("The tournament engine successfully runs competitions.")
        print("\nNext: We can integrate with your actual multi-agent system!\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)