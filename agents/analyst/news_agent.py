"""
News Sentiment Analysis Agent - Enhanced Version
Comprehensive news and sentiment analysis from multiple sources

MODIFIED: Now supports historical backtesting via analysis_date parameter
ENHANCED: Added _get_llm_decision() for better recommendation extraction

Supports: Yahoo Finance, Reddit (PRAW), NewsAPI, Finnhub, Alpha Vantage
Usage: 
  python news_agent.py AAPL --sources yahoo finnhub --days 7
  python news_agent.py AAPL --sources yahoo finnhub --days 7 --analysis-date 2024-06-15
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
import yfinance as yf
from openai import OpenAI

# FIX: Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Optional imports
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False


class NewsAgent:
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 analysis_date: Optional[str] = None):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Historical backtesting support
        self.analysis_date = analysis_date
        
        if self.analysis_date:
            print(f"[NEWS] *** HISTORICAL MODE: Analyzing as of {self.analysis_date} ***")
        else:
            print(f"[NEWS] Running in LIVE mode (current data)")
        
        # Load API keys from environment
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.finnhub_key = os.getenv("FINNHUB_KEY")
        self.alphavantage_key = os.getenv("ALPHAVANTAGE_KEY")
        
        # Reddit credentials
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "NewsAgent/1.0")
        
        # Enhanced system prompt
        self.system_prompt = """You are an expert news and sentiment analyst evaluating market-moving information for trading decisions.

**YOUR ANALYSIS FRAMEWORK:**

1. **Catalyst Identification:**
   - Breaking News: Major announcements, earnings surprises, product launches
   - Material Events: M&A activity, regulatory changes, executive changes
   - Market Reactions: How did similar news impact stock historically?
   - Timing: Is this news recent (actionable) or old (already priced in)?

2. **Sentiment Analysis:**
   - Overall Tone: Bullish, Bearish, or Neutral across sources
   - Sentiment Shifts: Has sentiment changed recently?
   - Source Credibility: Weight mainstream financial news higher
   - Social Sentiment: Reddit/Twitter buzz (contrarian indicator if extreme)

3. **News Impact Assessment:**
   - Price Impact Potential: High (earnings, M&A), Medium (analyst ratings), Low (general coverage)
   - Time Horizon: Immediate catalyst or slow-burn factor?
   - Already Priced In: Has the stock already moved on this news?

4. **Risk Factors from News:**
   - Regulatory/Legal risks mentioned
   - Competitive threats covered
   - Management concerns raised
   - Sector-wide issues affecting company

**OUTPUT FORMAT:**

## News & Sentiment Summary
[2-3 sentence overview of news environment]

## Key Headlines
[Top 3-5 most important stories with impact assessment]

## Sentiment Assessment
- **Overall Sentiment:** Bullish/Bearish/Neutral
- **Sentiment Trend:** Improving/Stable/Deteriorating
- **News Volume:** High/Normal/Low

## Catalysts Identified
**Bullish Catalysts:** [List]
**Bearish Catalysts:** [List]
**Upcoming Events:** [If any mentioned]

## Trading Implications
[How should this news inform trading decisions?]

RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low

Distinguish between actionable breaking news and old news already priced in."""

    def _get_reference_date(self) -> datetime:
        """Get the reference date for analysis (historical or current)"""
        if self.analysis_date:
            return datetime.strptime(self.analysis_date, '%Y-%m-%d')
        return datetime.now()

    def get_yahoo_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """
        Fetch news from Yahoo Finance (free, no API key needed)
        NOTE: Yahoo Finance news is always current - limited historical support
        """
        print(f"[NEWS] 🔧 Fetching Yahoo Finance news...")
        
        if self.analysis_date:
            print(f"[NEWS] ⚠️ Yahoo Finance has limited historical news support")
        
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news[:20] if stock.news else []
            
            reference_date = self._get_reference_date()
            cutoff_date = reference_date - timedelta(days=days)
            relevant_news = []
            
            for item in news:
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if self.analysis_date:
                    if pub_time > reference_date:
                        continue
                
                if pub_time > cutoff_date:
                    time_ago = reference_date - pub_time
                    
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days}d ago"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600}h ago"
                    else:
                        time_str = f"{time_ago.seconds // 60}m ago"
                    
                    relevant_news.append({
                        'title': item.get('title', 'No title'),
                        'publisher': item.get('publisher', 'Unknown'),
                        'link': item.get('link', ''),
                        'time_str': time_str,
                        'pub_time': pub_time,
                        'age_hours': time_ago.total_seconds() / 3600
                    })
            
            relevant_news.sort(key=lambda x: x['pub_time'], reverse=True)
            
            result = f"## Yahoo Finance News (Last {days} Days)\n"
            if self.analysis_date:
                result = f"## Yahoo Finance News ({days} Days before {self.analysis_date})\n"
            result += "\n"
            
            if relevant_news:
                result += f"**Found {len(relevant_news)} articles**\n\n"
                
                for i, item in enumerate(relevant_news[:10], 1):
                    result += f"{i}. **{item['title']}**\n"
                    result += f"   - Source: {item['publisher']}\n"
                    result += f"   - Published: {item['time_str']}\n\n"
                
                recent_count = sum(1 for n in relevant_news if n['age_hours'] < 24)
                if recent_count > 5:
                    result += f"📊 High news volume: {recent_count} articles in last 24 hours\n"
            else:
                result += f"No news found in the specified period\n"
                if self.analysis_date:
                    result += f"(Historical mode: news before {self.analysis_date})\n"
            
            print(f"[NEWS] ✓ Yahoo Finance: {len(relevant_news)} articles")
            
            return result, {
                'source': 'yahoo',
                'count': len(relevant_news),
                'articles': relevant_news[:10]
            }
            
        except Exception as e:
            print(f"[NEWS] ⚠️ Yahoo Finance error: {str(e)}")
            return f"## Yahoo Finance News\n**Error:** {str(e)}\n\n", {'source': 'yahoo', 'error': str(e)}

    def get_reddit_sentiment(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """Get Reddit sentiment using PRAW"""
        print(f"[NEWS] 🔧 Analyzing Reddit sentiment...")
        
        if not PRAW_AVAILABLE:
            print(f"[NEWS] ⚠️ PRAW not installed")
            return "## Reddit Sentiment\n**Status:** PRAW library not installed (`pip install praw`)\n\n", {'source': 'reddit', 'error': 'not_installed'}
        
        if not self.reddit_client_id or not self.reddit_client_secret:
            print(f"[NEWS] ⚠️ Reddit credentials missing")
            return "## Reddit Sentiment\n**Status:** Reddit credentials not configured\n\n", {'source': 'reddit', 'error': 'no_credentials'}
        
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            reference_date = self._get_reference_date()
            cutoff = reference_date - timedelta(days=days)
            
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'options']
            mentions = []
            
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.search(self.ticker, limit=20, sort='new'):
                        post_time = datetime.fromtimestamp(post.created_utc)
                        
                        if self.analysis_date and post_time > reference_date:
                            continue
                        
                        if post_time > cutoff:
                            mentions.append({
                                'title': post.title,
                                'subreddit': sub_name,
                                'score': post.score,
                                'comments': post.num_comments,
                                'time': post_time
                            })
                except:
                    continue
            
            result = f"## Reddit Sentiment (Last {days} Days)\n\n"
            
            if mentions:
                mentions.sort(key=lambda x: x['score'], reverse=True)
                top_mentions = mentions[:5]
                
                result += f"**Found {len(mentions)} mentions across Reddit**\n\n"
                
                for m in top_mentions:
                    result += f"- **r/{m['subreddit']}**: {m['title'][:60]}... (⬆️{m['score']})\n"
                
                bullish = sum(1 for m in mentions if any(w in m['title'].lower() for w in ['bull', 'buy', 'moon', 'calls', 'long']))
                bearish = sum(1 for m in mentions if any(w in m['title'].lower() for w in ['bear', 'sell', 'puts', 'short', 'crash']))
                
                if bullish > bearish * 1.5:
                    sentiment = "bullish"
                elif bearish > bullish * 1.5:
                    sentiment = "bearish"
                else:
                    sentiment = "mixed"
                
                result += f"\n**Sentiment:** {sentiment.upper()} ({bullish} bullish, {bearish} bearish mentions)\n"
                
                if len(mentions) > 20:
                    result += f"- Volume: **HIGH** - Strong social interest\n"
                elif len(mentions) > 5:
                    result += f"- Volume: Moderate social interest\n"
                else:
                    result += f"- Volume: Low social interest\n"
            else:
                result += f"No Reddit mentions found for {self.ticker} in specified period\n"
                sentiment = "none"
            
            print(f"[NEWS] ✓ Reddit: {len(mentions)} mentions, sentiment={sentiment}")
            
            return result + "\n", {
                'source': 'reddit',
                'count': len(mentions),
                'sentiment': sentiment,
                'mentions': top_mentions if mentions else []
            }
            
        except Exception as e:
            print(f"[NEWS] ⚠️ Reddit error: {str(e)}")
            return f"## Reddit Sentiment\n**Error:** {str(e)}\n\n", {'source': 'reddit', 'error': str(e)}

    def get_newsapi_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """Get news from NewsAPI - GOOD HISTORICAL SUPPORT"""
        print(f"[NEWS] 🔧 Fetching NewsAPI articles...")
        
        if not self.newsapi_key:
            print(f"[NEWS] ⚠️ NewsAPI key missing")
            return "## NewsAPI\n**Status:** No API key (get free at newsapi.org)\n\n", {'source': 'newsapi', 'error': 'no_key'}
        
        try:
            url = "https://newsapi.org/v2/everything"
            
            reference_date = self._get_reference_date()
            from_date = (reference_date - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = reference_date.strftime('%Y-%m-%d')
            
            params = {
                'q': self.ticker,
                'apiKey': self.newsapi_key,
                'from': from_date,
                'to': to_date,
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'language': 'en'
            }
            
            if self.analysis_date:
                print(f"[NEWS] NewsAPI: Fetching news from {from_date} to {to_date}")
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = f"## NewsAPI (Last {days} Days)\n"
            if self.analysis_date:
                result = f"## NewsAPI ({from_date} to {to_date})\n"
            result += "\n"
            
            if data.get('status') == 'ok' and data.get('articles'):
                articles = data['articles']
                result += f"**Found {len(articles)} articles**\n\n"
                
                for i, article in enumerate(articles[:8], 1):
                    pub_date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d') if article.get('publishedAt') else reference_date
                    days_ago = (reference_date - pub_date).days
                    time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                    
                    result += f"{i}. **{article.get('title', 'No title')}**\n"
                    result += f"   - Source: {article.get('source', {}).get('name', 'Unknown')}\n"
                    result += f"   - Published: {time_str}\n"
                    
                    if article.get('description'):
                        result += f"   - {article['description'][:150]}...\n"
                    result += "\n"
                
                print(f"[NEWS] ✓ NewsAPI: {len(articles)} articles")
                
                return result, {
                    'source': 'newsapi',
                    'count': len(articles),
                    'articles': articles[:8]
                }
            else:
                error_msg = data.get('message', 'No articles found')
                result += f"**Status:** {error_msg}\n\n"
                print(f"[NEWS] ⚠️ NewsAPI: {error_msg}")
                return result, {'source': 'newsapi', 'error': error_msg}
            
        except requests.Timeout:
            print(f"[NEWS] ⚠️ NewsAPI timeout")
            return "## NewsAPI\n**Error:** Request timeout\n\n", {'source': 'newsapi', 'error': 'timeout'}
        except Exception as e:
            print(f"[NEWS] ⚠️ NewsAPI error: {str(e)}")
            return f"## NewsAPI\n**Error:** {str(e)}\n\n", {'source': 'newsapi', 'error': str(e)}

    def get_finnhub_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """Get news from Finnhub - EXCELLENT HISTORICAL SUPPORT"""
        print(f"[NEWS] 🔧 Fetching Finnhub news...")
        
        if not self.finnhub_key:
            print(f"[NEWS] ⚠️ Finnhub key missing")
            return "## Finnhub News\n**Status:** No API key (get free at finnhub.io)\n\n", {'source': 'finnhub', 'error': 'no_key'}
        
        try:
            url = "https://finnhub.io/api/v1/company-news"
            
            reference_date = self._get_reference_date()
            from_date = (reference_date - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = reference_date.strftime('%Y-%m-%d')
            
            params = {
                'symbol': self.ticker,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_key
            }
            
            if self.analysis_date:
                print(f"[NEWS] Finnhub: Fetching news from {from_date} to {to_date}")
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = f"## Finnhub News (Last {days} Days)\n"
            if self.analysis_date:
                result = f"## Finnhub News ({from_date} to {to_date})\n"
            result += "\n"
            
            if data and isinstance(data, list) and len(data) > 0:
                result += f"**Found {len(data)} articles**\n\n"
                
                for i, article in enumerate(data[:8], 1):
                    pub_date = datetime.fromtimestamp(article.get('datetime', 0))
                    days_ago = (reference_date - pub_date).days
                    time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                    
                    result += f"{i}. **{article.get('headline', 'No title')}**\n"
                    result += f"   - Source: {article.get('source', 'Unknown')}\n"
                    result += f"   - Published: {time_str}\n"
                    
                    if article.get('summary'):
                        result += f"   - {article['summary'][:150]}...\n"
                    result += "\n"
                
                print(f"[NEWS] ✓ Finnhub: {len(data)} articles")
                
                return result, {
                    'source': 'finnhub',
                    'count': len(data),
                    'articles': data[:8]
                }
            else:
                result += "No news found\n\n"
                print(f"[NEWS] ⚠️ Finnhub: No news")
                return result, {'source': 'finnhub', 'count': 0}
            
        except requests.Timeout:
            print(f"[NEWS] ⚠️ Finnhub timeout")
            return "## Finnhub News\n**Error:** Request timeout\n\n", {'source': 'finnhub', 'error': 'timeout'}
        except Exception as e:
            print(f"[NEWS] ⚠️ Finnhub error: {str(e)}")
            return f"## Finnhub News\n**Error:** {str(e)}\n\n", {'source': 'finnhub', 'error': str(e)}

    def get_alphavantage_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """Get news from Alpha Vantage with sentiment scores - GOOD HISTORICAL SUPPORT"""
        print(f"[NEWS] 🔧 Fetching Alpha Vantage news...")
        
        if not self.alphavantage_key:
            print(f"[NEWS] ⚠️ Alpha Vantage key missing")
            return "## Alpha Vantage News\n**Status:** No API key (get free at alphavantage.co)\n\n", {'source': 'alphavantage', 'error': 'no_key'}
        
        try:
            url = "https://www.alphavantage.co/query"
            
            reference_date = self._get_reference_date()
            time_from = (reference_date - timedelta(days=days)).strftime('%Y%m%dT0000')
            time_to = reference_date.strftime('%Y%m%dT2359')
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': self.ticker,
                'apikey': self.alphavantage_key,
                'time_from': time_from,
                'time_to': time_to,
                'limit': 50
            }
            
            if self.analysis_date:
                print(f"[NEWS] Alpha Vantage: Fetching news from {time_from} to {time_to}")
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = f"## Alpha Vantage News (Last {days} Days)\n"
            if self.analysis_date:
                result = f"## Alpha Vantage News ({days} Days before {self.analysis_date})\n"
            result += "\n"
            
            cutoff_date = reference_date - timedelta(days=days)
            
            if 'feed' in data:
                relevant_articles = []
                
                for article in data['feed']:
                    try:
                        pub_date = datetime.strptime(article['time_published'][:8], '%Y%m%d')
                        
                        if self.analysis_date and pub_date > reference_date:
                            continue
                        
                        if pub_date >= cutoff_date:
                            sentiment_score = 0
                            for ticker_sent in article.get('ticker_sentiment', []):
                                if ticker_sent.get('ticker') == self.ticker:
                                    sentiment_score = float(ticker_sent.get('ticker_sentiment_score', 0))
                                    break
                            
                            if sentiment_score > 0.15:
                                sentiment = "Bullish"
                            elif sentiment_score < -0.15:
                                sentiment = "Bearish"
                            else:
                                sentiment = "Neutral"
                            
                            relevant_articles.append({
                                'title': article.get('title', 'No title'),
                                'source': article.get('source', 'Unknown'),
                                'sentiment': sentiment,
                                'sentiment_score': sentiment_score,
                                'pub_date': pub_date,
                                'summary': article.get('summary', '')
                            })
                    except:
                        continue
                
                if relevant_articles:
                    result += f"**Found {len(relevant_articles)} articles with sentiment**\n\n"
                    
                    for i, article in enumerate(relevant_articles[:8], 1):
                        days_ago = (reference_date - article['pub_date']).days
                        time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                        
                        sentiment_emoji = "🟢" if article['sentiment'] == "Bullish" else "🔴" if article['sentiment'] == "Bearish" else "⚪"
                        
                        result += f"{i}. **{article['title']}**\n"
                        result += f"   - Sentiment: {sentiment_emoji} {article['sentiment']}\n"
                        result += f"   - Published: {time_str}\n"
                        
                        if article['summary']:
                            result += f"   - {article['summary'][:150]}...\n"
                        result += "\n"
                    
                    bullish_count = sum(1 for a in relevant_articles if a['sentiment'] == 'Bullish')
                    bearish_count = sum(1 for a in relevant_articles if a['sentiment'] == 'Bearish')
                    
                    result += f"**Sentiment Breakdown:** {bullish_count} Bullish, {bearish_count} Bearish\n"
                    
                    print(f"[NEWS] ✓ Alpha Vantage: {len(relevant_articles)} articles")
                    
                    return result, {
                        'source': 'alphavantage',
                        'count': len(relevant_articles),
                        'bullish': bullish_count,
                        'bearish': bearish_count,
                        'articles': relevant_articles[:8]
                    }
                else:
                    result += f"No news found in specified period\n\n"
                    print(f"[NEWS] ⚠️ Alpha Vantage: No recent news")
                    return result, {'source': 'alphavantage', 'count': 0}
            else:
                result += "No news available\n\n"
                print(f"[NEWS] ⚠️ Alpha Vantage: No feed data")
                return result, {'source': 'alphavantage', 'error': 'no_feed'}
            
        except requests.Timeout:
            print(f"[NEWS] ⚠️ Alpha Vantage timeout")
            return "## Alpha Vantage News\n**Error:** Request timeout\n\n", {'source': 'alphavantage', 'error': 'timeout'}
        except Exception as e:
            print(f"[NEWS] ⚠️ Alpha Vantage error: {str(e)}")
            return f"## Alpha Vantage News\n**Error:** {str(e)}\n\n", {'source': 'alphavantage', 'error': str(e)}

    def _get_llm_decision(self, analysis: str) -> Tuple[str, str]:
        """
        NEW: Use LLM to extract/determine recommendation from analysis.
        This avoids the HOLD default bias problem.
        """
        if not self.client:
            return self._extract_recommendation_from_content(analysis)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a trading decision extractor. 
Given a news/sentiment analysis, extract or determine the final recommendation.

RULES:
1. If there's an explicit "RECOMMENDATION: X" line, extract it
2. If not explicit, analyze the content and determine the most appropriate recommendation
3. Consider: sentiment tone, news catalysts, volume of coverage, bullish/bearish signals
4. Do NOT default to HOLD - make an actual decision based on the evidence
5. NEWS-SPECIFIC signals:
   - Multiple bullish headlines + upgrades = BUY signal
   - Negative news + downgrades + high volume = SELL signal
   - Breaking positive catalyst = BUY signal
   - Breaking negative catalyst = SELL signal

Respond in EXACTLY this format (no other text):
RECOMMENDATION: BUY|HOLD|SELL
CONFIDENCE: High|Medium|Low"""},
                    {"role": "user", "content": f"Extract/determine recommendation from:\n\n{analysis[:3000]}"}
                ],
                temperature=0.3,
                max_completion_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            rec = "HOLD"
            conf = "Medium"
            
            for line in result.split('\n'):
                if 'RECOMMENDATION:' in line.upper():
                    if 'BUY' in line.upper():
                        rec = "BUY"
                    elif 'SELL' in line.upper():
                        rec = "SELL"
                    else:
                        rec = "HOLD"
                elif 'CONFIDENCE:' in line.upper():
                    if 'HIGH' in line.upper():
                        conf = "High"
                    elif 'LOW' in line.upper():
                        conf = "Low"
                    else:
                        conf = "Medium"
            
            print(f"[NEWS] LLM Decision: {rec} ({conf})")
            return rec, conf
            
        except Exception as e:
            print(f"[NEWS] ⚠️ LLM decision error: {e}, using fallback")
            return self._extract_recommendation_from_content(analysis)

    def _extract_recommendation_from_content(self, analysis: str) -> Tuple[str, str]:
        """Fallback: Extract recommendation using keyword analysis"""
        analysis_lower = analysis.lower()
        
        specific_buy_signals = ['upgrade', 'bullish', 'positive catalyst', 'beat expectations', 'strong buy']
        specific_sell_signals = ['downgrade', 'bearish', 'negative catalyst', 'missed expectations', 'sell rating']
        
        if any(phrase in analysis_lower for phrase in ["recommend buy", "should buy", "buy signal"]):
            return "BUY", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend sell", "should sell", "sell signal"]):
            return "SELL", "Medium"
        elif any(phrase in analysis_lower for phrase in ["recommend hold", "should hold", "wait", "neutral"]):
            return "HOLD", "Low"
        
        buy_count = sum(1 for signal in specific_buy_signals if signal in analysis_lower)
        sell_count = sum(1 for signal in specific_sell_signals if signal in analysis_lower)
        
        general_buy = ["bullish", "positive", "upside", "growth", "strong"]
        general_sell = ["bearish", "negative", "downside", "decline", "weak"]
        
        buy_count += sum(0.5 for word in general_buy if word in analysis_lower)
        sell_count += sum(0.5 for word in general_sell if word in analysis_lower)
        
        if buy_count > sell_count + 1.5:
            confidence = "High" if buy_count > 4 else "Medium"
            return "BUY", confidence
        elif sell_count > buy_count + 1.5:
            confidence = "High" if sell_count > 4 else "Medium"
            return "SELL", confidence
        else:
            return "HOLD", "Low"

    def analyze_with_llm(self, all_news: str, news_data: List[Dict]) -> str:
        """Analyze all gathered news with LLM"""
        if not self.client:
            print("[NEWS] ⚠️ No API key - using fallback analysis")
            return self._create_fallback_analysis(all_news, news_data)
        
        try:
            print(f"[NEWS] Analyzing with {self.model}...")
            
            date_context = ""
            if self.analysis_date:
                date_context = f"""
**Analysis Date: {self.analysis_date}**
Analyze all provided information as of this date.
Do NOT reference any events after {self.analysis_date}.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{date_context}\nAnalyze this news and market research for {self.ticker}:\n\n{all_news}"}
                ],
                temperature=0.7,
                max_completion_tokens=2000
            )
            
            analysis = response.choices[0].message.content
            
            # Use LLM decision extraction if no formal recommendation
            if "RECOMMENDATION:" not in analysis:
                print(f"[NEWS] ⚠️ Response missing formal recommendation, extracting...")
                recommendation, confidence = self._get_llm_decision(analysis)
                analysis += f"\n\nRECOMMENDATION: {recommendation} - Confidence: {confidence}"
                
            print(f"[NEWS] ✓ Analysis complete ({len(analysis)} chars)")
            return analysis
                
        except Exception as e:
            print(f"[NEWS] ❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(all_news, news_data)

    def _generate_llm_historical_context(self, days: int, existing_news_count: int = 0) -> Tuple[str, Dict[str, Any]]:
            """
            Generate historical market context using LLM knowledge.
            
            SAFETY FEATURES:
            1. Disables LLM context for dates past knowledge cutoff
            2. Uses conservative prompts to prevent hallucination
            3. Only states facts LLM is highly confident about
            """
            if not self.client or not self.analysis_date:
                return "", {'source': 'background_research', 'count': 0, 'error': 'not_available'}
            
            LLM_KNOWLEDGE_CUTOFF = datetime(2024, 1, 1)  # Conservative estimate
            
            try:
                analysis_dt = datetime.strptime(self.analysis_date, '%Y-%m-%d')
            except ValueError:
                print(f"[NEWS] ⚠️ Invalid date format: {self.analysis_date}")
                return "", {'source': 'background_research', 'count': 0, 'error': 'invalid_date'}
            
            # If date is past cutoff, don't use LLM context (high hallucination risk)
            if analysis_dt > LLM_KNOWLEDGE_CUTOFF:
                print(f"[NEWS] ⚠️ Date {self.analysis_date} is past LLM knowledge cutoff ({LLM_KNOWLEDGE_CUTOFF.strftime('%Y-%m-%d')})")
                print(f"[NEWS] ⚠️ Skipping LLM context to prevent hallucination")
                
                # Return a simple disclaimer instead of hallucinated content
                if existing_news_count == 0:
                    result = f"""## Market Research Notes

    **Note:** No news articles were found for this historical period, and the analysis date ({self.analysis_date}) is beyond the AI's knowledge cutoff date. Unable to provide historical context without risk of inaccuracy.

    For dates after {LLM_KNOWLEDGE_CUTOFF.strftime('%B %Y')}, please rely on actual news sources or archived data.

    """
                    return result, {
                        'source': 'background_research',
                        'count': 0,
                        'skipped_reason': 'date_past_cutoff',
                        'date': self.analysis_date
                    }
                else:
                    # We have some real news, no need to add LLM context
                    return "", {'source': 'background_research', 'count': 0, 'skipped_reason': 'date_past_cutoff'}
            
            print(f"[NEWS] 📋 Gathering background market context...")
            
            try:
                # Get company name for context
                try:
                    stock = yf.Ticker(self.ticker)
                    company_name = stock.info.get('shortName', self.ticker)
                except:
                    company_name = self.ticker
                
                system_prompt = f"""You are a financial analyst writing background notes dated {self.analysis_date}.

    CRITICAL RULES TO PREVENT HALLUCINATION:
    1. ONLY state facts you are 100% CERTAIN occurred BEFORE {self.analysis_date}
    2. DO NOT invent specific partnerships, deals, announcements, or news events
    3. DO NOT make up specific dates, numbers, percentages, or quotes
    4. DO NOT guess what "probably" or "likely" happened
    5. If you are not ABSOLUTELY CERTAIN about something, DO NOT include it
    6. Use hedging language ("typically", "generally", "historically") for patterns
    7. It is MUCH better to provide less information than to fabricate anything

    SAFE TO INCLUDE (established facts):
    - Company's main products and business model
    - General market position and reputation
    - Typical business cycle (when earnings usually occur)
    - Major annual events (WWDC for Apple, etc.) - but only mention they "typically occur", not specific announcements
    - General sector trends that were established before {self.analysis_date}

    DO NOT INCLUDE (high hallucination risk):
    - Specific news from the weeks before {self.analysis_date}
    - Specific product announcements or launches
    - Specific partnerships or deals
    - Specific analyst ratings or price target changes
    - Specific executive statements or quotes
    - Any specific numbers you're not 100% sure about

    Write factually and conservatively. When in doubt, leave it out."""

                if existing_news_count > 0:
                    # We have real news - just add brief, safe context
                    prompt = f"""Provide VERY BRIEF background context for {company_name} ({self.ticker}).

    ONLY include information you are 100% CERTAIN about:
    1. What the company does (1-2 sentences)
    2. Its general market position (1 sentence)

    Keep it to 1 short paragraph. DO NOT mention any specific recent events or news.
    If you're not certain about something, DO NOT include it."""

                else:
                    # No news found - provide general company background only
                    prompt = f"""Provide general background information for {company_name} ({self.ticker}).

    ONLY include established facts you are 100% CERTAIN about:
    1. **Company Overview** - What the company does, main products (2-3 sentences)
    2. **Business Model** - How they make money (1-2 sentences)
    3. **Market Position** - General reputation and competitive position (1-2 sentences)
    4. **Typical Calendar** - When they usually report earnings, any major annual events (1 sentence)

    IMPORTANT RULES:
    - DO NOT mention any specific news, announcements, or events
    - DO NOT guess what might have happened around {self.analysis_date}
    - DO NOT include specific numbers unless you are 100% certain
    - Keep it general and factual - this is just background context
    - 3-4 short paragraphs maximum

    If you cannot provide accurate information, just write a brief company description."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Very low temperature = more conservative/factual
                    max_completion_tokens=800  # Limit length to reduce hallucination surface
                )
                
                content = response.choices[0].message.content
                
                # Format as natural "background" section
                if existing_news_count > 0:
                    result = f"## Background Context\n\n{content}\n\n"
                else:
                    result = f"## Market Research Notes\n\n{content}\n\n"
                
                print(f"[NEWS] ✓ Background context compiled ({len(content)} chars)")
                
                return result, {
                    'source': 'background_research',
                    'count': 1,
                    'articles': [{'title': 'Analyst Background Research', 'type': 'context'}],
                    'date': self.analysis_date,
                    'within_cutoff': True
                }
                
            except Exception as e:
                print(f"[NEWS] ⚠️ Background context error: {e}")
                return "", {'source': 'background_research', 'count': 0, 'error': str(e)}

    def _check_news_data_empty(self, news_data: List[Dict]) -> bool:
        """Check if all news sources returned empty/zero articles"""
        total_articles = sum(d.get('count', 0) for d in news_data if 'count' in d)
        return total_articles == 0

    def _create_fallback_analysis(self, all_news: str, news_data: List[Dict]) -> str:
        """Rule-based fallback analysis"""
        print("[NEWS] Creating fallback analysis...")
        
        analysis = f"## News & Sentiment Analysis\n"
        analysis += "*Generated using fallback analysis (LLM unavailable)*\n"
        
        if self.analysis_date:
            analysis += f"*Historical analysis as of {self.analysis_date}*\n"
        
        analysis += "\n"
        
        successful_sources = [d['source'] for d in news_data if 'error' not in d]
        failed_sources = [d['source'] for d in news_data if 'error' in d]
        
        analysis += f"**Data Sources:** {len(successful_sources)} successful, {len(failed_sources)} failed\n\n"
        
        text_lower = all_news.lower()
        
        bullish_signals = sum([
            text_lower.count('buy'),
            text_lower.count('upgrade'),
            text_lower.count('bullish'),
            text_lower.count('positive'),
            text_lower.count('growth')
        ])
        
        bearish_signals = sum([
            text_lower.count('sell'),
            text_lower.count('downgrade'),
            text_lower.count('bearish'),
            text_lower.count('negative'),
            text_lower.count('concern')
        ])
        
        if bullish_signals > bearish_signals * 1.3:
            sentiment = "BULLISH"
            recommendation = "BUY"
        elif bearish_signals > bullish_signals * 1.3:
            sentiment = "BEARISH"
            recommendation = "SELL"
        else:
            sentiment = "NEUTRAL"
            recommendation = "HOLD"
        
        analysis += f"**Sentiment:** {sentiment}\n"
        analysis += f"- Bullish signals: {bullish_signals}\n"
        analysis += f"- Bearish signals: {bearish_signals}\n\n"
        
        total_articles = sum(d.get('count', 0) for d in news_data if 'count' in d)
        analysis += f"**Total Articles:** {total_articles}\n\n"
        
        analysis += f"RECOMMENDATION: {recommendation} - Confidence: Low\n"
        
        return analysis

    def run(self, sources: Optional[List[str]] = None, days: int = 7) -> str:
        """Execute comprehensive news analysis"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"NEWS & SENTIMENT ANALYSIS: {self.ticker}")
        if self.analysis_date:
            print(f"*** HISTORICAL MODE: As of {self.analysis_date} ***")
        print(f"Period: Last {days} days | Sources: {sources or ['yahoo']}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        if not sources:
            sources = ['yahoo']
        
        all_news = f"# News & Sentiment Analysis: {self.ticker}\n"
        if self.analysis_date:
            all_news += f"**⚠️ HISTORICAL ANALYSIS AS OF {self.analysis_date} ⚠️**\n"
        all_news += f"**Analysis Period:** Last {days} Days\n"
        all_news += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        all_news += "="*70 + "\n\n"
        
        news_data = []
        
        if 'yahoo' in sources:
            yahoo_text, yahoo_data = self.get_yahoo_news(days)
            all_news += yahoo_text
            news_data.append(yahoo_data)
        
        if 'reddit' in sources:
            reddit_text, reddit_data = self.get_reddit_sentiment(days)
            all_news += reddit_text
            news_data.append(reddit_data)
        
        if 'newsapi' in sources:
            newsapi_text, newsapi_data = self.get_newsapi_news(days)
            all_news += newsapi_text
            news_data.append(newsapi_data)
        
        if 'finnhub' in sources:
            finnhub_text, finnhub_data = self.get_finnhub_news(days)
            all_news += finnhub_text
            news_data.append(finnhub_data)
        
        if 'alphavantage' in sources:
            alpha_text, alpha_data = self.get_alphavantage_news(days)
            all_news += alpha_text
            news_data.append(alpha_data)
        
        # =====================================================================
        # HYBRID APPROACH: Always add background context in historical mode
        # - If news found: Add brief supplementary context
        # - If no news: Add fuller market research notes
        # This simulates what a well-informed analyst would have known
        # =====================================================================
        if self.analysis_date:
            existing_count = sum(d.get('count', 0) for d in news_data if 'count' in d)
            
            if existing_count == 0:
                print(f"[NEWS] 📋 No historical news found - compiling market research")
            else:
                print(f"[NEWS] 📋 Enhancing {existing_count} articles with background context")
            
            background_text, background_data = self._generate_llm_historical_context(days, existing_count)
            if background_text:
                all_news += background_text
                news_data.append(background_data)
        
        all_news += "\n" + "="*70 + "\n\n"
        
        analysis = self.analyze_with_llm(all_news, news_data)
        
        final_report = all_news + analysis
        
        elapsed = time.time() - start_time
        print(f"\n[NEWS] ✓ Analysis complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return final_report


def main():
    parser = argparse.ArgumentParser(
        description="News & Sentiment Analysis Agent - Multi-source news aggregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python news_agent.py AAPL
  python news_agent.py MSFT --sources yahoo reddit newsapi
  python news_agent.py GOOGL --days 3 --output news_report.txt
  
  # HISTORICAL BACKTESTING:
  python news_agent.py AAPL --sources yahoo finnhub --analysis-date 2024-06-15

Available Sources:
  yahoo        - Yahoo Finance (limited historical support)
  reddit       - Reddit (requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
  newsapi      - NewsAPI (requires NEWSAPI_KEY) - GOOD historical support
  finnhub      - Finnhub (requires FINNHUB_KEY) - EXCELLENT historical support
  alphavantage - Alpha Vantage (requires ALPHAVANTAGE_KEY) - GOOD historical support
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--sources", nargs="+",
                       choices=["yahoo", "reddit", "newsapi", "finnhub", "alphavantage"],
                       default=["yahoo", "finnhub"],
                       help="News sources (default: yahoo finnhub)")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Save report to file")
    parser.add_argument("--analysis-date", help="Historical date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        agent = NewsAgent(
            ticker=args.ticker, 
            api_key=args.api_key, 
            model=args.model,
            analysis_date=args.analysis_date
        )
        result = agent.run(sources=args.sources, days=args.days)
        
        print(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Report saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()