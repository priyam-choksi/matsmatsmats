"""
Game Theory Orchestrator - Updated for JSON Inputs
Runs tournament on risk manager's decision

Usage: python game_theory_orchestrator.py AAPL
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from game_theory.output_parser import OutputParser, load_and_parse_risk_decision, load_and_parse_discussion_points
from game_theory.market_context import MarketContextBuilder
from game_theory.tournament_engine import TournamentEngine
from backtesting.data_fetcher import DataFetcher
from backtesting.regime_detector import RegimeDetector


class GameTheoryOrchestrator:
    """Runs game theory tournament on existing trading decision"""
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        
        if not self.outputs_dir.exists():
            raise FileNotFoundError(f"Outputs directory not found: {self.outputs_dir}")
        
        # Initialize components
        self.parser = OutputParser()
        self.context_builder = MarketContextBuilder()
        self.tournament = TournamentEngine()
        self.data_fetcher = DataFetcher()
        self.regime_detector = RegimeDetector()
    
    def analyze(self, symbol: str) -> dict:
        """Run tournament on risk manager's decision"""
        
        print(f"\n{'='*70}")
        print(f"🎮 GAME THEORY TOURNAMENT: {symbol}")
        print(f"{'='*70}\n")
        
        # Step 1: Load risk decision (JSON format)
        print("📂 Loading risk decision...")
        parsed_decision = load_and_parse_risk_decision(
            str(self.outputs_dir / "risk_decision.json")
        )
        print(f"   ✅ Decision: {parsed_decision.action} ${parsed_decision.position_size_dollars:,.0f}\n")
        
        # Step 2: Load analyst consensus (JSON format)
        print("📂 Loading analyst consensus...")
        analyst_consensus = load_and_parse_discussion_points(
            str(self.outputs_dir / "discussion_points.json")
        )
        print(f"   ✅ Consensus: {analyst_consensus.consensus_score:.1%} ({analyst_consensus.majority_rec})\n")
        
        # Step 3: Detect market regime
        print("📈 Detecting market regime...")
        price_data = self.data_fetcher.get_price_data(symbol, days=60)
        regime_analysis = self.regime_detector.analyze_regime(price_data)
        print(f"   ✅ Regime: {regime_analysis.regime}\n")
        
        # Step 4: Build context
        context = self.context_builder.build(
            base_decision=parsed_decision,
            analyst_consensus=analyst_consensus,
            regime=regime_analysis.regime
        )
        
        # Step 5: Run tournament
        print("🏆 Running tournament with 5 strategies...\n")
        tournament_result = self.tournament.run_tournament(context, symbol)
        
        # Step 6: Display results
        self._display_results(parsed_decision, tournament_result, regime_analysis)
        
        # Step 7: Save results
        self._save_results(tournament_result)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'base_decision': parsed_decision,
            'regime': regime_analysis,
            'tournament_result': tournament_result
        }
    
    def _display_results(self, base_decision, tournament_result, regime_analysis):
        """Display tournament results"""
        
        print(f"{'='*70}")
        print("📊 YOUR SYSTEM vs 5 GAME THEORY STRATEGIES")
        print(f"{'='*70}\n")
        
        print(f"🌍 Market Regime: {regime_analysis.regime}")
        print(f"   Trend: {regime_analysis.trend_direction}, Volatility: {regime_analysis.volatility_level}\n")
        
        print(f"💼 YOUR RISK MANAGER DECISION:")
        print(f"   {base_decision.action} ${base_decision.position_size_dollars:,.0f}")
        print(f"   ({base_decision.confidence:.1%} confidence)\n")
        
        print(f"🎭 WHAT EACH STRATEGY WOULD DO:\n")
        
        strategies = tournament_result['strategies']
        recommended = tournament_result['recommended_strategy']
        
        for name, decision in strategies.items():
            marker = "⭐" if name == recommended else "  "
            diff = decision.position_size - base_decision.position_size_dollars
            diff_pct = (diff / base_decision.position_size_dollars * 100) if base_decision.position_size_dollars else 0
            
            print(f"{marker} {name.upper()}:")
            print(f"     {decision.action} ${decision.position_size:,.0f} ({diff_pct:+.1f}% vs yours)")
            print(f"     Why: {decision.reasoning[:80]}...\n")
        
        print(f"{'='*70}")
        print(f"🏆 RECOMMENDED STRATEGY: {recommended.upper()}")
        print(f"{'='*70}\n")
    
    def _save_results(self, tournament_result: dict):
        """Save tournament results"""
        output_file = self.outputs_dir / "game_theory_tournament.json"
        
        # Convert to serializable
        serializable = {
            'symbol': tournament_result['symbol'],
            'timestamp': tournament_result['timestamp'],
            'recommended_strategy': tournament_result['recommended_strategy'],
            'regime': tournament_result['context'].regime,
            'strategies': {
                name: {
                    'action': decision.action,
                    'position_size': decision.position_size,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning
                }
                for name, decision in tournament_result['strategies'].items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"💾 Tournament results saved: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Game Theory Tournament - Strategy comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  1. First run the full pipeline:
     python master_cli.py AAPL --mode quick
  
  2. Then run game theory tournament:
     python game_theory_orchestrator.py AAPL

  This compares your system's decision against 5 game theory strategies.
        """
    )
    
    parser.add_argument('symbol', type=str, help='Stock ticker')
    parser.add_argument('--outputs-dir', type=str, default='outputs',
                       help='Outputs directory (default: outputs)')
    
    args = parser.parse_args()
    
    try:
        orchestrator = GameTheoryOrchestrator(outputs_dir=args.outputs_dir)
        result = orchestrator.analyze(args.symbol)
        
        print("✅ Game theory analysis complete!")
        print(f"   View detailed results: outputs/game_theory_tournament.json\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Make sure you've run the pipeline first:")
        print(f"   python master_cli.py {args.symbol} --mode quick\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()