# agents/game_theory/market_context.py

"""
Builds market context from parsed agent outputs and regime detection.
Provides clean interface for game theory strategies.

Example usage:
    from game_theory.output_parser import OutputParser
    from game_theory.market_context import MarketContextBuilder
    from backtesting.regime_detector import RegimeDetector
    from backtesting.data_fetcher import DataFetcher
    
    # Parse your agent outputs
    parser = OutputParser()
    decision = parser.parse_final_decision(risk_manager_text)
    analysts = parser.parse_analyst_reports(market, sentiment, news, fundamentals)
    
    # Detect regime
    fetcher = DataFetcher()
    detector = RegimeDetector()
    data = fetcher.get_price_data("AAPL", days=60)
    regime = detector.detect_regime(data)
    
    # Build context
    builder = MarketContextBuilder()
    context = builder.build(decision, analysts, regime)
    
    print(f"Consensus: {context.analyst_consensus:.1%}")
    print(f"Regime: {context.regime}")
"""

from dataclasses import dataclass
from typing import Optional
from .output_parser import ParsedDecision, AnalystConsensus


@dataclass
class MarketContext:
    """
    Complete market context for strategy decision-making.
    
    This is what game theory strategies receive as input.
    
    Attributes:
        base_decision: The Risk Manager's original decision
        regime: Market regime (bull_trend, bear_trend, etc.)
        analyst_consensus: Score 0.0-1.0 (how much analysts agree)
        majority_recommendation: Most common analyst recommendation
        sentiment_score: 0.0-1.0 (0=bearish, 0.5=neutral, 1.0=bullish)
        confidence: Risk Manager's confidence 0.0-1.0
    """
    base_decision: ParsedDecision
    regime: str
    analyst_consensus: float
    majority_recommendation: Optional[str]
    sentiment_score: float
    confidence: float
    
    def is_high_consensus(self, threshold: float = 0.75) -> bool:
        """Check if analyst consensus is high"""
        return self.analyst_consensus >= threshold
    
    def is_bullish_sentiment(self) -> bool:
        """Check if overall sentiment is bullish"""
        return self.sentiment_score > 0.6
    
    def is_bearish_sentiment(self) -> bool:
        """Check if overall sentiment is bearish"""
        return self.sentiment_score < 0.4
    
    def alignment_score(self) -> float:
        """Calculate alignment (-1.0 to +1.0)"""
        action = self.base_decision.action
        score = 0.0
        
        # Regime alignment
        if action == 'BUY' and self.regime in ['bull_trend', 'momentum']:
            score += 0.5
        elif action == 'SELL' and self.regime in ['bear_trend', 'high_volatility']:
            score += 0.5
        elif action == 'HOLD' and self.regime == 'sideways':
            score += 0.5
        else:
            score -= 0.3
        
        # Sentiment alignment
        if self.sentiment_score > 0.6 and action == 'BUY':
            score += 0.3
        elif self.sentiment_score < 0.4 and action == 'SELL':
            score += 0.3
        
        # Weight by confidence
        score *= self.confidence
        
        return max(-1.0, min(1.0, score))
    
    def __str__(self):
        return (f"MarketContext(\n"
                f"  Action: {self.base_decision.action}\n"
                f"  Regime: {self.regime}\n"
                f"  Consensus: {self.analyst_consensus:.1%}\n"
                f"  Sentiment: {self.sentiment_score:.1%}\n"
                f"  Alignment: {self.alignment_score():.2f}\n"
                f")")


class MarketContextBuilder:
    """
    Builds MarketContext from parsed data and regime detection.
    """
    
    def __init__(self):
        pass
    
    def build(self,
              base_decision: ParsedDecision,
              analyst_consensus: AnalystConsensus,
              regime: str) -> MarketContext:
        """
        Build complete market context.
        
        Args:
            base_decision: Parsed Risk Manager decision
            analyst_consensus: Parsed analyst recommendations
            regime: Detected market regime
            
        Returns:
            MarketContext object
        """
        
        # Calculate sentiment score from analyst recommendations
        sentiment_score = self._calculate_sentiment_score(analyst_consensus)
        
        return MarketContext(
            base_decision=base_decision,
            regime=regime,
            analyst_consensus=analyst_consensus.consensus_score,
            majority_recommendation=analyst_consensus.majority_rec,
            sentiment_score=sentiment_score,
            confidence=base_decision.confidence
        )
    
    def _calculate_sentiment_score(self, analyst_consensus: AnalystConsensus) -> float:
        """
        Calculate overall sentiment score from analyst recommendations.
        
        Returns:
            0.0 = very bearish (all SELL)
            0.5 = neutral (mixed or all HOLD)
            1.0 = very bullish (all BUY)
        """
        
        recs = [
            analyst_consensus.technical_rec,      # ✅ FIXED: was market_rec
            analyst_consensus.fundamental_rec,    # ✅ FIXED: was sentiment_rec  
            analyst_consensus.news_rec,           # ✅ Already correct
            analyst_consensus.macro_rec           # ✅ FIXED: was fundamentals_rec
        ]
        
        # Count each type
        buy_count = sum(1 for r in recs if r == 'BUY')
        sell_count = sum(1 for r in recs if r == 'SELL')
        total = len([r for r in recs if r is not None])
        
        if total == 0:
            return 0.5  # Neutral if no data
        
        # Calculate score
        # 4 BUYs = 1.0, 2 BUYs 2 HOLDs = 0.5, 4 SELLs = 0.0
        score = (buy_count - sell_count) / total
        
        # Convert from [-1, 1] to [0, 1]
        return (score + 1) / 2.0