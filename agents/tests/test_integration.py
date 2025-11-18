# agents/tests/test_integration.py

"""
Integration test for complete game theory system.
Tests the full pipeline from agent outputs to tournament results.

Usage:
    cd agents
    python tests/test_integration.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrators.game_theory_orchestrator import GameTheoryOrchestrator


def test_complete_pipeline():
    """Test the complete pipeline with mock data"""
    
    print("\n" + "="*60)
    print("END-TO-END INTEGRATION TEST")
    print("="*60)
    
    # Create orchestrator
    orchestrator = GameTheoryOrchestrator(portfolio_value=100000)
    
    # Mock agent state (simulates output from your master_orchestrator)
    mock_agent_state = {
        'company_of_interest': 'AAPL',
        'final_trade_decision': """
After careful evaluation of all factors, I recommend BUY with a position size 
of $8,500 (8.5% of portfolio) with 78% confidence.

Technical analysis shows strong momentum, sentiment is bullish, and fundamentals 
are solid. Risk is manageable with proper stop-loss placement.

FINAL TRANSACTION PROPOSAL: **BUY**
        """,
        'market_report': """
Strong technical setup with price above all moving averages. RSI at 65 indicates 
healthy momentum without being overbought. Volume confirms the move.
FINAL TRANSACTION PROPOSAL: **BUY**
        """,
        'sentiment_report': """
Social media sentiment is 75% bullish with increasing engagement. Retail investors 
are showing strong interest.
FINAL TRANSACTION PROPOSAL: **BUY**
        """,
        'news_report': """
Recent product launch received positive reviews. Some concerns about supply chain 
but overall outlook remains positive.
FINAL TRANSACTION PROPOSAL: **HOLD**
        """,
        'fundamentals_report': """
Q4 earnings beat expectations by 12%. Revenue growth of 18% YoY. Strong balance 
sheet with low debt levels.
FINAL TRANSACTION PROPOSAL: **BUY**
        """
    }
    
    # Run analysis
    try:
        result = orchestrator.analyze('AAPL', mock_agent_state)
        
        # Verify result structure
        assert 'symbol' in result
        assert 'base_decision' in result
        assert 'tournament_result' in result
        assert 'recommended_strategy' in result
        assert 'final_decision' in result
        
        print("\n✅ Pipeline test passed!")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Base Action: {result['base_decision'].action}")
        print(f"   Recommended Strategy: {result['recommended_strategy']}")
        print(f"   Final Action: {result['final_decision'].action}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*16 + "INTEGRATION TESTS" + " "*25 + "║")
    print("╚" + "="*58 + "╝")
    
    success = test_complete_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("🎉 INTEGRATION TEST PASSED! 🎉")
        print("="*60)
        print("\nThe complete system works end-to-end!")
        print("You can now integrate this with your actual multi-agent system.\n")
        
        print("📚 HOW TO INTEGRATE WITH YOUR REAL SYSTEM:")
        print("="*60)
        print("""
1. In your master_orchestrator.py, after running all 5 phases:

   from orchestrators.game_theory_orchestrator import GameTheoryOrchestrator
   
   # After your agents finish
   final_state = self.run_all(symbol)
   
   # Add game theory layer
   gt_orchestrator = GameTheoryOrchestrator(portfolio_value=100000)
   gt_result = gt_orchestrator.analyze(symbol, final_state)
   
   # Use gt_result['final_decision'] for trading

2. Or create a new CLI command:

   python orchestrators/game_theory_orchestrator.py AAPL

3. The system will:
   ✅ Parse your agent outputs
   ✅ Detect market regime
   ✅ Run 5 strategies in tournament
   ✅ Recommend optimal strategy
        """)
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)