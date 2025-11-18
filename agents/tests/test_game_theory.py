# agents/tests/test_game_theory.py
"""
Quick diagnostic test for game theory integration
Tests if all components can import and basic functionality works
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_imports():
    """Test if all game theory modules can import"""
    print("="*70)
    print("TESTING GAME THEORY IMPORTS")
    print("="*70 + "\n")
    
    try:
        print("1. Testing output_parser import...")
        from agents.game_theory.output_parser import OutputParser, ParsedDecision, AnalystConsensus
        print("   ✅ output_parser imported successfully\n")
        
        print("2. Testing market_context import...")
        from agents.game_theory.market_context import MarketContext, MarketContextBuilder
        print("   ✅ market_context imported successfully\n")
        
        print("3. Testing strategy_interface import...")
        from agents.game_theory.strategy_interface import TradingStrategy, StrategyDecision
        print("   ✅ strategy_interface imported successfully\n")
        
        print("4. Testing strategies import...")
        from agents.game_theory.strategies import (
            ActualMarketStrategy,
            BuyHoldStrategy,
            CooperatorStrategy,
            DefectorStrategy,
            TitForTatStrategy
        )
        print("   ✅ All 5 strategies imported successfully\n")
        
        print("5. Testing tournament_engine import...")
        from agents.game_theory.tournament_engine import TournamentEngine
        print("   ✅ tournament_engine imported successfully\n")
        
        print("6. Testing backtesting imports...")
        from agents.backtesting.data_fetcher import DataFetcher
        from agents.backtesting.regime_detector import RegimeDetector
        print("   ✅ backtesting modules imported successfully\n")
        
        print("="*70)
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("="*70 + "\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test if basic game theory workflow works"""
    print("\n" + "="*70)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*70 + "\n")
    
    try:
        from agents.game_theory.output_parser import ParsedDecision, AnalystConsensus
        from agents.game_theory.market_context import MarketContext, MarketContextBuilder
        from agents.game_theory.tournament_engine import TournamentEngine
        
        # Create mock data
        print("1. Creating mock decision...")
        mock_decision = ParsedDecision(
            action='BUY',
            confidence=0.75,
            position_size_dollars=5000,
            position_size_pct=5.0,
            reasoning="Test decision",
            verdict='APPROVE'
        )
        print("   ✅ Mock decision created\n")
        
        print("2. Creating mock analyst consensus...")
        mock_consensus = AnalystConsensus(
            technical_rec='BUY',
            fundamental_rec='BUY',
            news_rec='HOLD',
            macro_rec='BUY'
        )
        print(f"   ✅ Consensus score: {mock_consensus.consensus_score:.1%}\n")
        
        print("3. Building market context...")
        builder = MarketContextBuilder()
        context = builder.build(
            base_decision=mock_decision,
            analyst_consensus=mock_consensus,
            regime='bull_trend'
        )
        print(f"   ✅ Context created:")
        print(f"      Action: {context.base_decision.action}")
        print(f"      Regime: {context.regime}")
        print(f"      Consensus: {context.analyst_consensus:.1%}")
        print(f"      Alignment: {context.alignment_score():.2f}\n")
        
        print("4. Initializing tournament engine...")
        tournament = TournamentEngine()
        print(f"   ✅ Tournament has {len(tournament.strategies)} strategies\n")
        
        print("5. Running tournament...")
        result = tournament.run_tournament(context, symbol="TEST")
        print(f"   ✅ Tournament complete!")
        print(f"      Recommended: {result['recommended_strategy']}\n")
        
        print("6. Checking all strategy decisions...")
        for name, decision in result['strategies'].items():
            print(f"      {name}: {decision.action} ${decision.position_size:,.0f}")
        print()
        
        print("="*70)
        print("✅ ALL FUNCTIONALITY TESTS PASSED!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FUNCTIONALITY ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("\n" + "🎯" * 35)
    print("GAME THEORY DIAGNOSTIC TEST")
    print("🎯" * 35 + "\n")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Cannot proceed - fix import errors first\n")
        return False
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    if not functionality_ok:
        print("\n❌ Functionality tests failed\n")
        return False
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED - GAME THEORY READY!")
    print("="*70)
    print("\nNext step: Run the full pipeline:")
    print("  python agents/orchestrators/master_orchestrator.py AAPL --run-all\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)