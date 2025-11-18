"""
Output Parser - Enhanced for JSON and Text Formats
Parses both structured JSON and text outputs from trading system

Usage:
    parser = OutputParser()
    decision = parser.parse_risk_decision(risk_decision_data)
    analysts = parser.parse_discussion_points(discussion_data)
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from collections import Counter


# ========== Data Structures ==========

@dataclass
class ParsedDecision:
    """
    Structured trading decision
    
    Attributes:
        action: BUY, SELL, or HOLD
        confidence: 0.0 to 1.0 (0% to 100%)
        position_size_dollars: Dollar amount
        position_size_pct: Percentage
        reasoning: Brief explanation
        verdict: APPROVE, MODIFY, or REJECT (from risk manager)
    """
    action: str
    confidence: float
    position_size_dollars: Optional[float] = None
    position_size_pct: Optional[float] = None
    reasoning: str = ""
    verdict: Optional[str] = None
    
    def __str__(self):
        return (f"Action: {self.action}, "
                f"Confidence: {self.confidence:.1%}, "
                f"Position: ${self.position_size_dollars:,.0f}")


@dataclass
class AnalystConsensus:
    """
    Consensus from 4 analysts
    
    Attributes:
        technical_rec: Technical analyst recommendation
        fundamental_rec: Fundamental analyst recommendation
        news_rec: News analyst recommendation
        macro_rec: Macro analyst recommendation
        consensus_score: 0.0 to 1.0 (agreement level)
        majority_rec: Most common recommendation
    """
    technical_rec: Optional[str] = None
    fundamental_rec: Optional[str] = None
    news_rec: Optional[str] = None
    macro_rec: Optional[str] = None
    
    @property
    def consensus_score(self) -> float:
        """Calculate agreement level"""
        recs = [self.technical_rec, self.fundamental_rec, self.news_rec, self.macro_rec]
        valid_recs = [r for r in recs if r is not None]
        
        if len(valid_recs) == 0:
            return 0.5
        
        counts = Counter(valid_recs)
        most_common_count = counts.most_common(1)[0][1]
        
        return most_common_count / len(valid_recs)
    
    @property
    def majority_rec(self) -> Optional[str]:
        """Get most common recommendation"""
        recs = [self.technical_rec, self.fundamental_rec, self.news_rec, self.macro_rec]
        valid_recs = [r for r in recs if r is not None]
        
        if not valid_recs:
            return None
        
        return Counter(valid_recs).most_common(1)[0][0]
    
    def __str__(self):
        return (f"Consensus: {self.consensus_score:.1%}, "
                f"Majority: {self.majority_rec}")


# ========== Main Parser Class ==========

class OutputParser:
    """
    Parses both JSON and text outputs from trading system
    """
    
    def __init__(self):
        pass
    
    # ========== NEW: JSON Parsing Methods ==========
    
    def parse_risk_decision_json(self, decision_data: Dict[str, Any]) -> ParsedDecision:
        """
        Parse risk_decision.json (new JSON format)
        
        Args:
            decision_data: Dict loaded from risk_decision.json
            
        Returns:
            ParsedDecision object
        """
        # Map verdict to action
        verdict = decision_data.get('verdict', 'HOLD')
        
        if verdict == 'APPROVE':
            action = 'BUY'
        elif verdict == 'REJECT':
            action = 'HOLD'
        elif verdict == 'MODIFY':
            action = 'BUY'  # Modified approval
        else:
            action = 'HOLD'
        
        # Parse confidence
        confidence_str = decision_data.get('confidence', 'MEDIUM')
        confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.65, 'LOW': 0.4}
        confidence = confidence_map.get(confidence_str, 0.5)
        
        # Get position size
        position_dollars = decision_data.get('final_position_dollars', 0)
        position_pct = decision_data.get('final_position_pct', 0)
        
        # Get reasoning
        reasoning_list = decision_data.get('reasoning', [])
        reasoning = "; ".join(reasoning_list) if reasoning_list else ""
        
        return ParsedDecision(
            action=action,
            confidence=confidence,
            position_size_dollars=position_dollars,
            position_size_pct=position_pct,
            reasoning=reasoning,
            verdict=verdict
        )
    
    def parse_discussion_points_json(self, discussion_data: Dict[str, Any]) -> AnalystConsensus:
        """
        Parse discussion_points.json (new JSON format)
        
        Args:
            discussion_data: Dict loaded from discussion_points.json
            
        Returns:
            AnalystConsensus object
        """
        recommendations = discussion_data.get('summary', {}).get('recommendations', {})
        
        return AnalystConsensus(
            technical_rec=recommendations.get('technical'),
            fundamental_rec=recommendations.get('fundamental'),
            news_rec=recommendations.get('news'),
            macro_rec=recommendations.get('macro')
        )
    
    # ========== LEGACY: Text Parsing Methods (for backward compatibility) ==========
    
    def parse_final_decision(self, text: str) -> ParsedDecision:
        """
        Parse text-based decision (legacy format)
        
        Args:
            text: Text report from risk manager
            
        Returns:
            ParsedDecision object
        """
        action = self._extract_action(text)
        confidence = self._extract_confidence(text)
        position_dollars, position_pct = self._extract_position_size(text)
        reasoning = self._extract_reasoning(text)
        
        return ParsedDecision(
            action=action,
            confidence=confidence,
            position_size_dollars=position_dollars,
            position_size_pct=position_pct,
            reasoning=reasoning
        )
    
    def parse_analyst_reports(self,
                             technical_report: str,
                             fundamental_report: str,
                             news_report: str,
                             macro_report: str) -> AnalystConsensus:
        """
        Parse text-based analyst reports (legacy format)
        
        Args:
            Reports as text strings
            
        Returns:
            AnalystConsensus object
        """
        return AnalystConsensus(
            technical_rec=self._extract_action(technical_report),
            fundamental_rec=self._extract_action(fundamental_report),
            news_rec=self._extract_action(news_report),
            macro_rec=self._extract_action(macro_report)
        )
    
    # ========== Helper Methods ==========
    
    def _extract_action(self, text: str) -> str:
        """Extract BUY/SELL/HOLD from text"""
        # Try various patterns
        patterns = [
            r'RECOMMENDATION:\s*(\w+)',
            r'FINAL DECISION:\s*\*\*(\w+)\*\*',
            r'VERDICT:\s*(\w+)',
            r'STANCE:\s*(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                action = match.group(1).upper()
                
                # Normalize
                if 'BUY' in action:
                    return 'BUY'
                elif 'SELL' in action or 'AVOID' in action:
                    return 'SELL'
                elif 'HOLD' in action:
                    return 'HOLD'
        
        return 'HOLD'
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence as 0.0-1.0"""
        # Pattern: "Confidence: High/Medium/Low"
        match = re.search(r'Confidence:\s*(\w+)', text, re.IGNORECASE)
        
        if match:
            conf_str = match.group(1).upper()
            conf_map = {'HIGH': 0.9, 'MEDIUM': 0.65, 'LOW': 0.4}
            return conf_map.get(conf_str, 0.5)
        
        return 0.5
    
    def _extract_position_size(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract position size"""
        dollars = None
        pct = None
        
        # Extract dollars
        match = re.search(r'\$([0-9,]+(?:\.\d+)?)', text)
        if match:
            dollars_str = match.group(1).replace(',', '')
            dollars = float(dollars_str)
        
        # Extract percentage
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            pct = float(match.group(1))
        
        return dollars, pct
    
    def _extract_reasoning(self, text: str, max_length: int = 200) -> str:
        """Extract reasoning summary"""
        sentences = re.split(r'[.!?]\s+', text)
        reasoning = '. '.join(sentences[:2])
        
        if len(reasoning) > max_length:
            reasoning = reasoning[:max_length] + "..."
        
        return reasoning.strip()


# ========== Convenience Functions ==========

def load_and_parse_risk_decision(filepath: str = "outputs/risk_decision.json") -> ParsedDecision:
    """Load and parse risk decision from JSON file"""
    parser = OutputParser()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return parser.parse_risk_decision_json(data)
    except Exception as e:
        print(f"Error loading risk decision: {e}")
        return ParsedDecision(action='HOLD', confidence=0.5)


def load_and_parse_discussion_points(filepath: str = "outputs/discussion_points.json") -> AnalystConsensus:
    """Load and parse discussion points from JSON file"""
    parser = OutputParser()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return parser.parse_discussion_points_json(data)
    except Exception as e:
        print(f"Error loading discussion points: {e}")
        return AnalystConsensus()