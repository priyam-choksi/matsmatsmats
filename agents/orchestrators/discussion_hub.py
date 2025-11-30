"""
Discussion Hub - Enhanced with Full Report Preservation
Aggregates analyst reports while preserving complete context for downstream agents

MODIFIED: Now supports historical backtesting via analysis_date parameter
ENHANCED: Better RISK-ON/RISK-OFF handling and synthesis direction detection

FIXES APPLIED:
  - Fix 1: Synthesis direction extraction now properly handles markdown formatting (**NEUTRAL**)
  - Fix 2: Article count now totals from ALL sources (Yahoo + Finnhub), not just Yahoo

Usage: 
  python discussion_hub.py AAPL --run-analysts --output discussion_points.json
  python discussion_hub.py AAPL --run-analysts --analysis-date 2024-06-15
"""

import os
import sys
import json
import re
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from openai import OpenAI

# Force UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class DiscussionHub:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Historical backtesting support
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[HUB] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[HUB] Running in LIVE mode (current data)")
        
        # Enhanced system prompt for synthesis
        self.system_prompt = """You are a research coordinator synthesizing multiple analyst perspectives.

**YOUR ROLE:**
You have access to complete reports from 4 analysts (technical, fundamental, news, macro). Create a synthesis that:

1. **Executive Summary (1-2 paragraphs):**
   - What's the overall picture? Bullish, bearish, or mixed?
   - What are the 2-3 most important factors driving the decision?

2. **Bull Case (3-5 key points):**
   - Strongest reasons to buy from ALL analysts
   - Reference which analyst provided each point

3. **Bear Case (3-5 key points):**
   - Biggest risks or reasons to sell from ALL analysts
   - Reference which analyst flagged each concern

4. **Key Conflicts:**
   - Where do analysts disagree and why?
   - Which conflict is most important to resolve?

5. **Research Priorities:**
   - What needs deeper investigation?
   - Any time-sensitive catalysts?

6. **Overall Direction:**
   At the end, you MUST provide a clear directional assessment:
   SYNTHESIS DIRECTION: BULLISH/BEARISH/NEUTRAL - Confidence: High/Medium/Low

Be balanced and objective. Stick to a formal and professional tone. Ensure all technical details and perspectives are captured in the synthesis.
Your synthesis will guide the bull and bear researchers."""
        
        # Storage
        self.analyst_reports = {}
        self.recommendations = {}
        self.confidence_levels = {}
        
        # Setup paths
        self.setup_paths()
    
    def setup_paths(self):
        """Detect analyst agent locations"""
        self.analyst_configs = {
            'technical': '../analyst/technical_agent.py',
            'news': '../analyst/news_agent.py',
            'fundamental': '../analyst/fundamental_agent.py',
            'macro': '../analyst/macro_agent.py'
        }
        
        # Validate
        found = sum(1 for path in self.analyst_configs.values() if os.path.exists(path))
        print(f"[HUB] Found {found}/4 analyst agents")
    
    def run_analyst(self, agent_name: str, agent_script: str) -> str:
        """Run individual analyst via subprocess - Updated error handling"""
        print(f"[HUB] Running {agent_name}...")
        
        cmd = ["python", agent_script]
        
        if agent_name != "macro":
            cmd.append(self.ticker)
        
        # Standard parameters
        if agent_name == "technical":
            cmd.extend(["--days", "7"])
        elif agent_name == "news":
            cmd.extend(["--sources", "yahoo", "finnhub", "--days", "7"])
        elif agent_name == "macro":
            cmd.extend(["--days", "7"])
        
        # Pass analysis date to ALL analysts
        if self.analysis_date:
            cmd.extend(["--analysis-date", self.analysis_date])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
                cwd=Path(agent_script).parent,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                print(f"[HUB] ✓ {agent_name} completed ({len(result.stdout)} chars)")
                return result.stdout
            else:
                print(f"[HUB] ⚠ {agent_name} error: {result.stderr[:100]}")
                return f"# {agent_name.upper()} ANALYSIS ERROR\n\n{result.stderr}\n\nRECOMMENDATION: UNDETERMINED - Confidence: N/A"
                
        except subprocess.TimeoutExpired:
            print(f"[HUB] ⚠ {agent_name} timeout")
            return f"""# {agent_name.upper()} ANALYSIS TIMEOUT

**Status:** Agent timed out after 90 seconds

Unable to complete analysis. This should not count as a HOLD recommendation.

RECOMMENDATION: UNDETERMINED - Confidence: N/A
DATA_QUALITY: TIMEOUT"""
            
        except FileNotFoundError:
            print(f"[HUB] ✗ {agent_script} not found")
            return f"""# {agent_name.upper()} NOT FOUND

**Status:** Agent script not found at {agent_script}

Unable to complete analysis. This should not count as a HOLD recommendation.

RECOMMENDATION: UNDETERMINED - Confidence: N/A
DATA_QUALITY: MISSING"""
            
        except Exception as e:
            print(f"[HUB] ✗ {agent_name} error: {str(e)}")
            return f"""# {agent_name.upper()} ERROR

**Status:** Unexpected error
**Error:** {str(e)}

Unable to complete analysis. This should not count as a HOLD recommendation.

RECOMMENDATION: UNDETERMINED - Confidence: N/A
DATA_QUALITY: ERROR"""

    def extract_recommendation(self, report: str) -> Tuple[str, str]:
        """
        Extract recommendation and confidence from report
        ENHANCED: Handles markdown formatting like **HOLD** and RISK-ON/RISK-OFF
        """
        recommendation = "UNDETERMINED"
        confidence = "Low"
        
        # Normalize the report text for matching
        report_upper = report.upper()
        
        # =========================================================================
        # FIX: Strip markdown formatting (**, *, __, etc.) before matching
        # This handles cases like "RECOMMENDATION: **HOLD**"
        # =========================================================================
        report_clean = re.sub(r'\*\*|\*|__|_', '', report_upper)
        
        # Pattern to find RECOMMENDATION line (now works on cleaned text)
        rec_pattern = r"RECOMMENDATION:\s*([A-Z][A-Z0-9\-_\s]*?)(?:\s*[-–—]\s*|\s+)(?:CONFIDENCE|$)"
        conf_pattern = r"CONFIDENCE:\s*(\w+)"
        
        rec_match = re.search(rec_pattern, report_clean)
        if rec_match:
            rec = rec_match.group(1).strip()
            
            # Normalize: remove spaces, underscores, hyphens for comparison
            rec_normalized = re.sub(r'[\s\-_]', '', rec)
            
            if rec_normalized == "BUY" or "BUY" in rec:
                recommendation = "BUY"
            elif rec_normalized == "SELL" or "SELL" in rec:
                recommendation = "SELL"
            elif rec_normalized == "HOLD" or "HOLD" in rec:
                recommendation = "HOLD"
            # RISK-ON variations (macro agent bullish signal)
            elif rec_normalized == "RISKON" or "RISKON" in rec_normalized:
                recommendation = "BUY"
            elif "RISK" in rec and "ON" in rec:
                recommendation = "BUY"
            # RISK-OFF variations (macro agent bearish signal)
            elif rec_normalized == "RISKOFF" or "RISKOFF" in rec_normalized:
                recommendation = "SELL"
            elif "RISK" in rec and "OFF" in rec:
                recommendation = "SELL"
            # NEUTRAL variations
            elif rec_normalized == "NEUTRAL" or "NEUTRAL" in rec:
                recommendation = "HOLD"
            elif "UNDETERMINED" in rec:
                recommendation = "UNDETERMINED"
            else:
                recommendation = "UNDETERMINED"
        
        # Backup detection if still UNDETERMINED
        if recommendation == "UNDETERMINED":
            # Look for clear directional statements (also on cleaned text)
            if re.search(r'\bRISK[\s\-_]?ON\b', report_clean):
                recommendation = "BUY"
            elif re.search(r'\bRISK[\s\-_]?OFF\b', report_clean):
                recommendation = "SELL"
            # NEW: Direct keyword backup for markdown-formatted recommendations
            elif re.search(r'RECOMMENDATION:.*\bBUY\b', report_clean):
                recommendation = "BUY"
            elif re.search(r'RECOMMENDATION:.*\bSELL\b', report_clean):
                recommendation = "SELL"
            elif re.search(r'RECOMMENDATION:.*\bHOLD\b', report_clean):
                recommendation = "HOLD"
        
        # Extract confidence (also from cleaned text)
        conf_match = re.search(conf_pattern, report_clean)
        if conf_match:
            conf = conf_match.group(1).upper()
            if "HIGH" in conf:
                confidence = "High"
            elif "MEDIUM" in conf or "MED" in conf:
                confidence = "Medium"
            elif "LOW" in conf:
                confidence = "Low"
            elif "N/A" in conf or "NA" in conf:
                confidence = "N/A"
        
        return recommendation, confidence

    def create_analyst_summary(self, report: str, analyst_type: str) -> str:
        """
        Create concise summary - Updated to handle UNDETERMINED
        FIX 2: For news analyst, count articles from ALL sources (Yahoo + Finnhub)
        """
        rec, conf = self.extract_recommendation(report)
        
        # Check for data quality issues
        if "DATA_QUALITY:" in report:
            quality_match = re.search(r"DATA_QUALITY:\s*(\w+)", report)
            if quality_match:
                quality = quality_match.group(1)
                if quality in ["ERROR", "TIMEOUT", "MISSING"]:
                    return f"⚠ {analyst_type.upper()}: Analysis failed ({quality}) - Recommendation: {rec}"
        
        # Extract key findings based on analyst type
        key_finding = ""
        
        if analyst_type == "technical":
            if "uptrend" in report.lower():
                key_finding = "Uptrend confirmed"
            elif "downtrend" in report.lower():
                key_finding = "Downtrend confirmed"
            elif "sideways" in report.lower() or "consolidat" in report.lower():
                key_finding = "Sideways/consolidation"
            else:
                key_finding = "Mixed technicals"
            
            rsi_match = re.search(r"RSI.*?(\d+\.?\d*)", report)
            if rsi_match:
                rsi = float(rsi_match.group(1))
                key_finding += f", RSI {rsi:.0f}"
                
        elif analyst_type == "fundamental":
            if "undervalued" in report.lower():
                key_finding = "Undervalued"
            elif "overvalued" in report.lower():
                key_finding = "Overvalued"
            elif "fairly valued" in report.lower() or "fair value" in report.lower():
                key_finding = "Fairly valued"
            else:
                key_finding = "Mixed fundamentals"
                
        elif analyst_type == "news":
            # =====================================================================
            # FIX 2: Count TOTAL articles from ALL sources, not just Yahoo
            # =====================================================================
            total_articles = 0
            sentiment = "neutral"
            
            # Count Yahoo articles
            yahoo_match = re.search(r"Yahoo.*?(\d+)\s*(?:articles?|items?|stories?)", report, re.IGNORECASE)
            if yahoo_match:
                total_articles += int(yahoo_match.group(1))
            
            # Count Finnhub articles - look for "Found X articles" pattern
            finnhub_match = re.search(r"(?:Finnhub|Found)\s*(\d+)\s*articles?", report, re.IGNORECASE)
            if finnhub_match:
                total_articles += int(finnhub_match.group(1))
            
            # Alternative patterns for article counts
            if total_articles == 0:
                # Try generic "X articles" pattern
                generic_match = re.search(r"(\d+)\s+(?:total\s+)?articles?", report, re.IGNORECASE)
                if generic_match:
                    total_articles = int(generic_match.group(1))
            
            # Determine sentiment
            if "bullish" in report.lower() or "positive" in report.lower():
                sentiment = "Bullish"
            elif "bearish" in report.lower() or "negative" in report.lower():
                sentiment = "Bearish"
            else:
                sentiment = "Mixed"
            
            key_finding = f"{sentiment} sentiment ({total_articles} articles)"
            
        elif analyst_type == "macro":
            if "risk-on" in report.lower() or "risk on" in report.lower():
                key_finding = "Risk-on environment"
            elif "risk-off" in report.lower() or "risk off" in report.lower():
                key_finding = "Risk-off environment"
            else:
                key_finding = "Mixed macro"
            
            vix_match = re.search(r"VIX.*?(\d+\.?\d*)", report)
            if vix_match:
                vix = float(vix_match.group(1))
                key_finding += f", VIX {vix:.1f}"
        
        # Handle UNDETERMINED gracefully
        if rec == "UNDETERMINED":
            return f"⚠ {analyst_type.upper()}: {key_finding} - Unable to determine recommendation"
        
        return f"{analyst_type.upper()}: {key_finding} → {rec} ({conf})"
    
    def extract_key_points(self, report: str, analyst_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract bullish and bearish signals with scoring"""
        bullish_signals = []
        bearish_signals = []
        
        bullish_keywords = [
            'bullish', 'buy', 'upside', 'growth', 'strong', 'positive',
            'outperform', 'upgrade', 'momentum', 'breakout', 'support',
            'oversold', 'undervalued', 'beat', 'exceed', 'improving',
            'expansion', 'accelerating', 'strength', 'risk-on', 'risk on'
        ]
        
        bearish_keywords = [
            'bearish', 'sell', 'downside', 'risk', 'weak', 'negative',
            'underperform', 'downgrade', 'resistance', 'overbought',
            'overvalued', 'miss', 'concern', 'deteriorat', 'decline',
            'contraction', 'slowing', 'weakness', 'risk-off', 'risk off'
        ]
        
        sentences = report.replace('\n', '. ').split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            cleaned = sentence.strip()
            
            if len(cleaned) < 20 or len(cleaned) > 300:
                continue
            
            bullish_count = sum(1 for kw in bullish_keywords if kw in sentence_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in sentence_lower)
            
            if bullish_count > bearish_count and bullish_count >= 1:
                bullish_signals.append({
                    'source': analyst_type,
                    'signal': cleaned,
                    'strength': bullish_count,
                    'keywords_matched': bullish_count
                })
            elif bearish_count > bullish_count and bearish_count >= 1:
                bearish_signals.append({
                    'source': analyst_type,
                    'signal': cleaned,
                    'strength': bearish_count,
                    'keywords_matched': bearish_count
                })
        
        bullish_signals.sort(key=lambda x: x['strength'], reverse=True)
        bearish_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        return bullish_signals[:20], bearish_signals[:20]
    
    def identify_conflicts(self) -> List[Dict]:
        """Identify and categorize conflicts"""
        conflicts = []
        
        rec_list = list(self.recommendations.values())
        
        if len(set(rec_list)) > 1:
            for analyst1, rec1 in self.recommendations.items():
                for analyst2, rec2 in self.recommendations.items():
                    if analyst1 < analyst2 and rec1 != rec2:
                        # Skip if either is UNDETERMINED
                        if rec1 == 'UNDETERMINED' or rec2 == 'UNDETERMINED':
                            continue
                        
                        if (rec1 == 'BUY' and rec2 == 'SELL') or (rec1 == 'SELL' and rec2 == 'BUY'):
                            severity = 'CRITICAL'
                        elif 'HOLD' in [rec1, rec2]:
                            severity = 'MINOR'
                        else:
                            severity = 'MODERATE'
                        
                        conflicts.append({
                            'type': 'recommendation_conflict',
                            'severity': severity,
                            'analysts': [analyst1, analyst2],
                            'positions': {analyst1: rec1, analyst2: rec2},
                            'description': f"{analyst1} says {rec1} while {analyst2} says {rec2}"
                        })
        
        severity_order = {'CRITICAL': 0, 'MODERATE': 1, 'MINOR': 2}
        conflicts.sort(key=lambda x: severity_order.get(x['severity'], 99))
        
        return conflicts
    
    def find_consensus(self) -> List[Dict]:
        """Find agreement points"""
        consensus_points = []
        
        # Filter out UNDETERMINED for consensus calculation
        valid_recs = {k: v for k, v in self.recommendations.items() if v != 'UNDETERMINED'}
        
        rec_counts = {}
        for rec in valid_recs.values():
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        # Check for majority consensus
        for rec, count in rec_counts.items():
            if count >= 3:
                consensus_points.append({
                    'type': 'strong_consensus',
                    'recommendation': rec,
                    'count': count,
                    'description': f"Strong consensus: {count}/4 analysts say {rec}"
                })
            elif count == 2:
                analysts = [a for a, r in valid_recs.items() if r == rec]
                consensus_points.append({
                    'type': 'partial_consensus',
                    'recommendation': rec,
                    'count': count,
                    'analysts': analysts,
                    'description': f"Partial consensus: {analysts[0]} and {analysts[1]} both say {rec}"
                })
        
        # Check for high-confidence agreements
        high_conf = [a for a, c in self.confidence_levels.items() 
                     if c == 'High' and self.recommendations.get(a) != 'UNDETERMINED']
        if len(high_conf) >= 2:
            high_recs = [self.recommendations[a] for a in high_conf]
            if len(set(high_recs)) == 1:
                consensus_points.append({
                    'type': 'high_confidence_agreement',
                    'analysts': high_conf,
                    'recommendation': high_recs[0],
                    'description': f"High-confidence agreement: {', '.join(high_conf)} → {high_recs[0]}"
                })
        
        return consensus_points
    
    def identify_research_priorities(self, bullish: List, bearish: List, conflicts: List) -> List[Dict]:
        """Determine research priorities"""
        priorities = []
        
        critical_conflicts = [c for c in conflicts if c.get('severity') == 'CRITICAL']
        if critical_conflicts:
            priorities.append({
                'priority': 'CRITICAL',
                'focus': 'resolve_major_conflict',
                'description': f"BUY vs SELL conflict - {critical_conflicts[0]['description']}",
                'action': 'Deep dive needed to determine which analyst is correct'
            })
        
        bull_count = len(bullish)
        bear_count = len(bearish)
        
        if bull_count > bear_count * 2.5:
            priorities.append({
                'priority': 'HIGH',
                'focus': 'validate_bull_thesis',
                'description': f'Strong bullish bias ({bull_count} vs {bear_count} signals)',
                'action': 'Validate assumptions - could be herd mentality or overlooked risks'
            })
        elif bear_count > bull_count * 2.5:
            priorities.append({
                'priority': 'HIGH',
                'focus': 'validate_bear_thesis',
                'description': f'Strong bearish bias ({bear_count} vs {bull_count} signals)',
                'action': 'Validate assumptions - justified fear or oversold opportunity?'
            })
        elif abs(bull_count - bear_count) <= 5:
            priorities.append({
                'priority': 'MEDIUM',
                'focus': 'break_tie',
                'description': f'Balanced signals ({bull_count} vs {bear_count})',
                'action': 'Need deeper analysis to determine edge'
            })
        
        time_keywords = ['earnings', 'announcement', 'tomorrow', 'today', 'imminent', 'upcoming', 'next week']
        
        for signal in bullish + bearish:
            signal_text = signal.get('signal', '').lower()
            if any(kw in signal_text for kw in time_keywords):
                priorities.append({
                    'priority': 'URGENT',
                    'focus': 'time_sensitive_event',
                    'description': 'Upcoming catalyst detected',
                    'signal': signal.get('signal', ''),
                    'source': signal.get('source', ''),
                    'action': 'Research before event occurs'
                })
                break
        
        return priorities[:5]

    def _extract_synthesis_direction(self, synthesis: str) -> Tuple[str, str]:
        """
        FIX 1: Extract or determine direction from LLM synthesis
        IMPROVED: Now properly handles markdown formatting like **NEUTRAL**
        Returns (direction, confidence)
        """
        # =========================================================================
        # FIX 1: Strip ALL markdown formatting BEFORE attempting to match
        # This handles cases like "SYNTHESIS DIRECTION: **NEUTRAL**"
        # =========================================================================
        
        # Strip markdown bold/italic markers
        synthesis_clean = re.sub(r'\*\*|\*|__|_', '', synthesis)
        synthesis_upper = synthesis_clean.upper()
        
        # Look for explicit SYNTHESIS DIRECTION line (on cleaned text)
        direction_pattern = r"SYNTHESIS\s*DIRECTION:\s*(\w+)"
        dir_match = re.search(direction_pattern, synthesis_upper)
        
        if dir_match:
            direction = dir_match.group(1).strip()
            
            # Also try to extract confidence from the same line
            conf_pattern = r"SYNTHESIS\s*DIRECTION:\s*\w+\s*[-–—]?\s*CONFIDENCE:\s*(\w+)"
            conf_match = re.search(conf_pattern, synthesis_upper)
            confidence = "Medium"  # default
            
            if conf_match:
                conf = conf_match.group(1)
                if "HIGH" in conf:
                    confidence = "High"
                elif "LOW" in conf:
                    confidence = "Low"
                else:
                    confidence = "Medium"
            
            # Normalize direction
            if "BULLISH" in direction or "BUY" in direction:
                return "BULLISH", confidence
            elif "BEARISH" in direction or "SELL" in direction:
                return "BEARISH", confidence
            elif "NEUTRAL" in direction or "HOLD" in direction:
                return "NEUTRAL", confidence
        
        # Fallback: analyze synthesis content (also on cleaned text)
        bullish_indicators = ['bullish', 'buy', 'upside', 'positive', 'strong', 'outperform', 'growth']
        bearish_indicators = ['bearish', 'sell', 'downside', 'negative', 'weak', 'underperform', 'risk']
        
        synthesis_lower = synthesis_clean.lower()
        bull_score = sum(synthesis_lower.count(word) for word in bullish_indicators)
        bear_score = sum(synthesis_lower.count(word) for word in bearish_indicators)
        
        if bull_score > bear_score * 1.5:
            return "BULLISH", "Low"
        elif bear_score > bull_score * 1.5:
            return "BEARISH", "Low"
        else:
            return "NEUTRAL", "Low"

    def _ensure_synthesis_direction(self, synthesis: str, discussion_data: Dict) -> str:
        """
        Ensure synthesis has a clear directional statement
        If missing, add one based on analyst consensus or content analysis
        """
        # Check if synthesis already has direction (check cleaned version)
        synthesis_clean = re.sub(r'\*\*|\*|__|_', '', synthesis)
        if re.search(r"SYNTHESIS\s*DIRECTION:", synthesis_clean.upper()):
            return synthesis
        
        # Determine direction from available data
        valid_recs = discussion_data['summary'].get('valid_recommendations', {})
        
        if valid_recs:
            buy_count = sum(1 for r in valid_recs.values() if r == 'BUY')
            sell_count = sum(1 for r in valid_recs.values() if r == 'SELL')
            hold_count = sum(1 for r in valid_recs.values() if r == 'HOLD')
            
            if buy_count > sell_count and buy_count >= hold_count:
                direction = "BULLISH"
                confidence = "High" if buy_count >= 3 else "Medium"
            elif sell_count > buy_count and sell_count >= hold_count:
                direction = "BEARISH"
                confidence = "High" if sell_count >= 3 else "Medium"
            else:
                direction = "NEUTRAL"
                confidence = "Medium"
        else:
            # Fallback to content analysis
            direction, confidence = self._extract_synthesis_direction(synthesis)
        
        # Append direction to synthesis
        direction_statement = f"\n\n---\n\n**SYNTHESIS DIRECTION: {direction}** - Confidence: {confidence}"
        
        return synthesis + direction_statement
    
    def synthesize_with_llm(self, discussion_data: Dict) -> Dict:
        """
        Create intelligent synthesis using FULL analyst reports
        ENHANCED: Ensures synthesis has clear directional statement
        """
        if not self.client:
            print("[HUB] ⚠ No API key - skipping synthesis")
            discussion_data['llm_synthesis'] = "LLM synthesis unavailable (no API key)"
            discussion_data['synthesis_direction'] = "UNDETERMINED"
            discussion_data['synthesis_confidence'] = "N/A"
            return discussion_data
        
        try:
            print(f"[HUB] Generating synthesis from full reports...")
            
            date_context = ""
            if self.analysis_date:
                date_context = f"""
**⚠ HISTORICAL ANALYSIS MODE ⚠**
All analyst reports are based on data AS OF {self.analysis_date}.
Synthesize as if making a decision ON {self.analysis_date}.

"""
            
            full_context = f"""{date_context}# Complete Analyst Reports for {self.ticker}

## Quick Summary
- Recommendations: {discussion_data['summary']['recommendations']}
- Valid Recommendations: {discussion_data['summary'].get('valid_recommendations', {})}
- Bullish signals: {discussion_data['summary']['bull_signal_count']}
- Bearish signals: {discussion_data['summary']['bear_signal_count']}

## Full Analyst Reports

### Technical Analysis
{discussion_data['full_analyst_reports'].get('technical', 'Not available')}

---

### Fundamental Analysis
{discussion_data['full_analyst_reports'].get('fundamental', 'Not available')}

---

### News & Sentiment
{discussion_data['full_analyst_reports'].get('news', 'Not available')}

---

### Macro Environment
{discussion_data['full_analyst_reports'].get('macro', 'Not available')}

---

## Identified Conflicts
{json.dumps(discussion_data['key_conflicts'], indent=2)}

## Consensus Points
{json.dumps(discussion_data['consensus_points'], indent=2)}

IMPORTANT: End your synthesis with a clear directional statement:
SYNTHESIS DIRECTION: BULLISH/BEARISH/NEUTRAL - Confidence: High/Medium/Low
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Synthesize these complete analyst reports:\n\n{full_context}"}
                ],
                temperature=0.7,
                max_completion_tokens=2500
            )
            
            synthesis = response.choices[0].message.content
            
            # Ensure synthesis has direction
            synthesis = self._ensure_synthesis_direction(synthesis, discussion_data)
            
            # Extract direction for structured data
            direction, conf = self._extract_synthesis_direction(synthesis)
            
            discussion_data['llm_synthesis'] = synthesis
            discussion_data['synthesis_direction'] = direction
            discussion_data['synthesis_confidence'] = conf
            
            print(f"[HUB] ✓ Synthesis complete ({len(synthesis)} chars)")
            print(f"[HUB] ✓ Synthesis direction: {direction} ({conf})")
            
        except Exception as e:
            print(f"[HUB] ⚠ Synthesis error: {e}")
            discussion_data['llm_synthesis'] = f"Synthesis error: {str(e)}"
            discussion_data['synthesis_direction'] = "UNDETERMINED"
            discussion_data['synthesis_confidence'] = "N/A"
        
        return discussion_data
    
    def aggregate_reports(self, reports: Optional[Dict] = None, run_analysts: bool = False) -> Dict:
        """
        Main aggregation workflow - NOW PRESERVES FULL REPORTS
        UPDATED: Handles UNDETERMINED recommendations properly
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"DISCUSSION HUB: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"{'='*70}\n")
        
        # Step 1: Get analyst reports
        if run_analysts:
            print(f"[HUB] Running all analysts...\n")
            
            for name, script in self.analyst_configs.items():
                if os.path.exists(script):
                    report = self.run_analyst(name, script)
                    self.analyst_reports[name] = report
                    
                    rec, conf = self.extract_recommendation(report)
                    self.recommendations[name] = rec
                    self.confidence_levels[name] = conf
                else:
                    print(f"[HUB] ⚠ Skipping {name} (not found)")
            
            print()
            
        elif reports:
            print(f"[HUB] Using provided reports...")
            self.analyst_reports = reports
            for name, report in reports.items():
                rec, conf = self.extract_recommendation(report)
                self.recommendations[name] = rec
                self.confidence_levels[name] = conf
        else:
            print(f"[HUB] ✗ No reports available")
            return {
                'ticker': self.ticker,
                'error': 'No analyst reports',
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: Create concise summaries
        print(f"[HUB] Creating analyst summaries...")
        analyst_summaries = {}
        for name, report in self.analyst_reports.items():
            analyst_summaries[name] = self.create_analyst_summary(report, name)
        print(f"[HUB] ✓ Summaries created\n")
        
        # Step 3: Extract signals
        print(f"[HUB] Extracting signals...")
        all_bullish = []
        all_bearish = []
        
        for analyst_type, report in self.analyst_reports.items():
            bullish, bearish = self.extract_key_points(report, analyst_type)
            all_bullish.extend(bullish)
            all_bearish.extend(bearish)
            print(f"[HUB]   {analyst_type}: {len(bullish)} bull, {len(bearish)} bear")
        
        print(f"[HUB] ✓ Total: {len(all_bullish)} bullish, {len(all_bearish)} bearish\n")
        
        # Step 4: Conflicts and consensus
        print(f"[HUB] Analyzing consensus...")
        conflicts = self.identify_conflicts()
        consensus = self.find_consensus()
        print(f"[HUB] ✓ {len(conflicts)} conflicts, {len(consensus)} consensus points\n")
        
        # Step 5: Research priorities
        print(f"[HUB] Identifying priorities...")
        priorities = self.identify_research_priorities(all_bullish, all_bearish, conflicts)
        print(f"[HUB] ✓ {len(priorities)} priorities identified\n")
        
        # Filter out UNDETERMINED for accurate counting
        valid_recommendations = {
            agent: rec for agent, rec in self.recommendations.items() 
            if rec != "UNDETERMINED"
        }
        
        bull_rec_count = sum(1 for rec in valid_recommendations.values() if rec == "BUY")
        bear_rec_count = sum(1 for rec in valid_recommendations.values() if rec == "SELL")
        hold_rec_count = sum(1 for rec in valid_recommendations.values() if rec == "HOLD")
        undetermined_count = len(self.recommendations) - len(valid_recommendations)
        
        if len(valid_recommendations) == 0:
            net_sentiment = 'UNDETERMINED'
        elif bull_rec_count > bear_rec_count:
            net_sentiment = 'BULLISH'
        elif bear_rec_count > bull_rec_count:
            net_sentiment = 'BEARISH'
        else:
            net_sentiment = 'NEUTRAL'
        
        # Step 6: Structure output
        discussion_points = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'analysis_date': self.analysis_date,
            'historical_mode': self.analysis_date is not None,
            
            'summary': {
                'recommendations': self.recommendations,
                'valid_recommendations': valid_recommendations,
                'confidence_levels': self.confidence_levels,
                'bull_signal_count': len(all_bullish),
                'bear_signal_count': len(all_bearish),
                'bull_rec_count': bull_rec_count,
                'bear_rec_count': bear_rec_count,
                'hold_rec_count': hold_rec_count,
                'undetermined_count': undetermined_count,
                'analyst_count': len(self.analyst_reports),
                'valid_analyst_count': len(valid_recommendations),
                'net_sentiment': net_sentiment
            },
            
            'analyst_summaries': analyst_summaries,
            'full_analyst_reports': self.analyst_reports,
            'bull_evidence': all_bullish[:20],
            'bear_evidence': all_bearish[:20],
            'key_conflicts': conflicts,
            'consensus_points': consensus,
            'research_priorities': priorities,
            
            'analyst_reports_summary': {
                name: {
                    'recommendation': self.recommendations.get(name, 'N/A'),
                    'confidence': self.confidence_levels.get(name, 'N/A'),
                    'report_length': len(report),
                    'has_data': len(report) > 100,
                    'completed_successfully': self.recommendations.get(name, 'UNDETERMINED') != 'UNDETERMINED'
                }
                for name, report in self.analyst_reports.items()
            },
            
            'data_quality': {
                'status': 'COMPLETE' if undetermined_count == 0 else 'PARTIAL',
                'total_analysts': len(self.analyst_reports),
                'valid_analysts': len(valid_recommendations),
                'failed_analysts': undetermined_count,
                'failed_list': [
                    name for name, rec in self.recommendations.items() 
                    if rec == 'UNDETERMINED'
                ],
                'note': None if undetermined_count == 0 else f"{undetermined_count} analyst(s) failed to produce valid recommendations"
            }
        }
        
        if undetermined_count > 0:
            print(f"[HUB] ⚠ Data Quality: PARTIAL - {undetermined_count} analyst(s) returned UNDETERMINED")
            for name, rec in self.recommendations.items():
                if rec == 'UNDETERMINED':
                    print(f"[HUB]    - {name}: UNDETERMINED")
        else:
            print(f"[HUB] ✓ Data Quality: COMPLETE - all analysts produced valid recommendations")
        
        # Step 7: LLM synthesis (with direction detection)
        print(f"[HUB] Generating comprehensive synthesis...")
        discussion_points = self.synthesize_with_llm(discussion_points)
        
        elapsed = time.time() - start_time
        print(f"\n[HUB] ✓ Aggregation complete in {elapsed:.2f}s")
        
        total_chars = sum(len(r) for r in self.analyst_reports.values())
        print(f"[HUB] 📊 Captured {total_chars:,} total characters from analysts")
        print(f"[HUB] 📊 Recommendations: {bull_rec_count} BUY, {bear_rec_count} SELL, {hold_rec_count} HOLD, {undetermined_count} UNDETERMINED")
        print(f"[HUB] 📊 Synthesis Direction: {discussion_points.get('synthesis_direction', 'N/A')}")
        print(f"{'='*70}\n")
        
        return discussion_points
    
    def format_report(self, discussion_points: Dict) -> str:
        """Format as readable text report"""
        dp = discussion_points
        
        report = f"""
{'='*80}
                        DISCUSSION HUB ANALYSIS
{'='*80}
Ticker: {dp['ticker']}
Timestamp: {dp['timestamp']}
"""
        if dp.get('analysis_date'):
            report += f"*** HISTORICAL ANALYSIS AS OF {dp['analysis_date']} ***\n"
        
        report += f"""Net Sentiment: {dp['summary']['net_sentiment']}
Synthesis Direction: {dp.get('synthesis_direction', 'N/A')} ({dp.get('synthesis_confidence', 'N/A')})

ANALYST QUICK SUMMARIES
{'-'*80}
"""
        for analyst, summary in dp.get('analyst_summaries', {}).items():
            rec = dp['summary']['recommendations'].get(analyst, 'N/A')
            icon = "✓" if rec in ['BUY', 'SELL'] else "○" if rec == 'HOLD' else "⚠"
            report += f"  {icon} {analyst:12} | {summary}\n"
        
        report += f"""
CONSENSUS & CONFLICTS
{'-'*80}
"""
        if dp['consensus_points']:
            report += "Consensus:\n"
            for consensus in dp['consensus_points']:
                report += f"  ✓ {consensus['description']}\n"
        
        if dp['key_conflicts']:
            report += "\nConflicts:\n"
            for conflict in dp['key_conflicts']:
                severity_icon = "🔴" if conflict['severity'] == 'CRITICAL' else "🟡" if conflict['severity'] == 'MODERATE' else "⚪"
                report += f"  {severity_icon} {conflict['description']}\n"
        
        report += f"""
SIGNAL BREAKDOWN
{'-'*80}
  Bullish Signals: {dp['summary']['bull_signal_count']}
  Bearish Signals: {dp['summary']['bear_signal_count']}

RESEARCH PRIORITIES
{'-'*80}
"""
        for priority in dp.get('research_priorities', []):
            priority_icon = "🔥" if priority['priority'] == 'URGENT' else "⚠" if priority['priority'] == 'CRITICAL' else "📊"
            report += f"  {priority_icon} [{priority['priority']}] {priority['description']}\n"
            if 'action' in priority:
                report += f"      → {priority['action']}\n"
        
        if 'llm_synthesis' in dp and 'unavailable' not in dp['llm_synthesis'].lower():
            report += f"""
AI SYNTHESIS (Based on Full Reports)
{'-'*80}
{dp['llm_synthesis']}
"""
        
        report += f"""
DATA QUALITY
{'-'*80}
  Status: {dp['data_quality']['status']}
  Valid Analysts: {dp['data_quality']['valid_analysts']}/{dp['data_quality']['total_analysts']}
"""
        if dp['data_quality']['failed_list']:
            report += f"  Failed: {', '.join(dp['data_quality']['failed_list'])}\n"

        report += f"""
DATA CAPTURED
{'-'*80}
  Full Reports: {len(dp.get('full_analyst_reports', {}))} analysts
  Total Size: {sum(len(r) for r in dp.get('full_analyst_reports', {}).values()):,} characters
  Extracted Signals: {len(dp.get('bull_evidence', []))} bull + {len(dp.get('bear_evidence', []))} bear

{'='*80}
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Discussion Hub - Aggregates analyst reports with full context preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discussion_hub.py AAPL --run-analysts
  python discussion_hub.py MSFT --run-analysts --output discussion.json
  python discussion_hub.py GOOGL --run-analysts --format text --output report.txt
  
  # HISTORICAL BACKTESTING:
  python discussion_hub.py AAPL --run-analysts --analysis-date 2024-06-15

Output Format:
  The JSON output contains:
  - analyst_summaries: One-line summaries for quick scanning
  - full_analyst_reports: Complete original reports
  - bull_evidence/bear_evidence: Extracted signals
  - llm_synthesis: AI synthesis using full context
  - synthesis_direction: Overall direction (BULLISH/BEARISH/NEUTRAL)
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--run-analysts", action="store_true",
                       help="Run all 4 analyst agents")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--format", choices=['json', 'text'], default='text',
                       help="Output format")
    parser.add_argument("--analysis-date",
                       type=str,
                       default=None,
                       help="Historical analysis date (YYYY-MM-DD format)")
    
    args = parser.parse_args()
    
    try:
        hub = DiscussionHub(
            ticker=args.ticker, 
            api_key=args.api_key, 
            model=args.model,
            analysis_date=args.analysis_date
        )
        
        discussion_points = hub.aggregate_reports(run_analysts=args.run_analysts)
        
        if args.format == 'json' or (args.output and args.output.endswith('.json')):
            output = json.dumps(discussion_points, indent=2)
        else:
            output = hub.format_report(discussion_points)
        
        print(output)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.output.endswith('.json'):
                    json.dump(discussion_points, f, indent=2)
                else:
                    f.write(output)
            print(f"\n✓ Saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()