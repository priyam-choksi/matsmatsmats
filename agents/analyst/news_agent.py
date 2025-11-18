"""
News Sentiment Analysis Agent - Enhanced Version
Comprehensive news and sentiment analysis from multiple sources

Supports: Yahoo Finance, Reddit (PRAW), NewsAPI, Finnhub, Alpha Vantage
Usage: python news_agent.py AAPL --sources yahoo reddit --days 7 --output report.txt
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
    def __init__(self, ticker: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.ticker = ticker.upper()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
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
   - Sentiment Shifts: Has sentiment changed recently? (Getting better/worse)
   - Source Quality: Weigh reputable sources higher than social media
   - Consensus: Do multiple sources agree or is sentiment mixed?

3. **Social Media Signals:**
   - Reddit Activity: Volume of mentions, upvotes, comment engagement
   - Retail Sentiment: Bullish or bearish positioning from retail traders
   - Meme Stock Risk: Is excessive hype a contrarian signal?
   - Momentum: Is social buzz increasing or fading?

4. **Upcoming Catalysts:**
   - Scheduled Events: Earnings dates, product reveals, conferences
   - Regulatory Deadlines: FDA decisions, legal rulings
   - Market Events: Ex-dividend dates, option expiration
   - Timeline: How soon? (Imminent vs distant)

5. **Risk Assessment:**
   - Negative Headlines: Lawsuits, investigations, controversies
   - Competitive Threats: Market share loss, new competitors
   - Operational Issues: Supply chain, production problems
   - Sentiment Deterioration: Previously positive now turning negative

**DECISION CRITERIA:**

**BUY Signals:**
- Positive breaking news (earnings beat, major contract, innovation)
- Bullish sentiment shift (was negative, now turning positive)
- Upcoming positive catalyst (product launch, FDA approval expected)
- Strong social momentum with institutional news support

**SELL Signals:**
- Negative breaking news (earnings miss, guidance cut, scandal)
- Bearish sentiment shift (was positive, now deteriorating)
- Risk events materializing (lawsuits, regulatory action)
- Social hype reaching extreme levels (potential reversal)

**HOLD Signals:**
- Mixed sentiment across sources (no clear direction)
- Old news already priced in (no new catalysts)
- Low news volume (lack of information)
- Neutral social media activity

**OUTPUT FORMAT:**

## News & Sentiment Summary
[2-3 sentence overview of key findings]

## Key Headlines & Catalysts
[List 3-5 most important news items with dates]

## Sentiment Analysis
- Overall Sentiment: [Bullish/Bearish/Neutral]
- Sentiment Trend: [Improving/Deteriorating/Stable]
- Source Consensus: [Strong/Moderate/Weak agreement]

## Social Media Analysis
[Reddit/community sentiment if available]

## Upcoming Catalysts
[List scheduled events with dates]

## Risk Factors
[Key concerns or red flags]

## Trading Implications
**News Impact:** [High/Medium/Low]
**Actionability:** [Immediate/Near-term/Watch]
**Confidence Level:** [High/Medium/Low]

RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low

Be specific about dates, sources, and sentiment direction. Distinguish between actionable breaking news and old news already priced in."""

    def get_yahoo_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """
        Fetch news from Yahoo Finance (free, no API key needed)
        Returns formatted string and structured data
        """
        print(f"[NEWS] 🔧 Fetching Yahoo Finance news...")
        
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news[:20] if stock.news else []
            
            cutoff_date = datetime.now() - timedelta(days=days)
            relevant_news = []
            
            for item in news:
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if pub_time > cutoff_date:
                    time_ago = datetime.now() - pub_time
                    
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
            
            # Sort by recency
            relevant_news.sort(key=lambda x: x['pub_time'], reverse=True)
            
            # Format output
            result = f"## Yahoo Finance News (Last {days} Days)\n\n"
            
            if relevant_news:
                result += f"**Found {len(relevant_news)} articles**\n\n"
                
                for i, item in enumerate(relevant_news[:10], 1):
                    result += f"{i}. **{item['title']}**\n"
                    result += f"   - Source: {item['publisher']}\n"
                    result += f"   - Published: {item['time_str']}\n\n"
                
                # Analyze recency
                recent_count = sum(1 for n in relevant_news if n['age_hours'] < 24)
                if recent_count > 5:
                    result += f"📊 High news volume: {recent_count} articles in last 24 hours\n"
            else:
                result += f"No news found in the last {days} days\n"
            
            print(f"[NEWS] ✓ Yahoo Finance: {len(relevant_news)} articles")
            
            return result, {
                'source': 'yahoo',
                'count': len(relevant_news),
                'articles': relevant_news[:10]
            }
            
        except Exception as e:
            print(f"[NEWS] ⚠️  Yahoo Finance error: {str(e)}")
            return f"## Yahoo Finance News\n**Error:** {str(e)}\n\n", {'source': 'yahoo', 'error': str(e)}

    def get_reddit_sentiment(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """
        Get Reddit sentiment using PRAW
        Returns formatted string and structured data
        """
        print(f"[NEWS] 🔧 Analyzing Reddit sentiment...")
        
        if not PRAW_AVAILABLE:
            print(f"[NEWS] ⚠️  PRAW not installed")
            return "## Reddit Sentiment\n**Status:** PRAW library not installed\n\n", {'source': 'reddit', 'error': 'praw_missing'}
        
        if not (self.reddit_client_id and self.reddit_client_secret):
            print(f"[NEWS] ⚠️  Reddit credentials missing")
            return "## Reddit Sentiment\n**Status:** No credentials (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)\n\n", {'source': 'reddit', 'error': 'no_credentials'}
        
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
            mentions = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    time_filter = 'week' if days <= 7 else 'month'
                    
                    for submission in subreddit.search(self.ticker, time_filter=time_filter, limit=10):
                        created_time = datetime.fromtimestamp(submission.created_utc)
                        
                        if created_time > cutoff_date:
                            mentions.append({
                                'title': submission.title,
                                'score': submission.score,
                                'comments': submission.num_comments,
                                'subreddit': sub_name,
                                'created': created_time,
                                'url': f"https://reddit.com{submission.permalink}"
                            })
                except Exception as e:
                    print(f"[NEWS] ⚠️  Error in r/{sub_name}: {str(e)}")
                    continue
            
            # Sort by score (engagement)
            mentions.sort(key=lambda x: x['score'], reverse=True)
            top_mentions = mentions[:10]
            
            # Format output
            result = f"## Reddit Sentiment (Last {days} Days)\n\n"
            
            if top_mentions:
                result += f"**Found {len(mentions)} mentions across Reddit**\n\n"
                
                # Top posts
                for i, m in enumerate(top_mentions[:5], 1):
                    days_ago = (datetime.now() - m['created']).days
                    time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                    
                    result += f"{i}. **{m['title']}**\n"
                    result += f"   - r/{m['subreddit']} | ⬆️ {m['score']} | 💬 {m['comments']}\n"
                    result += f"   - Posted: {time_str}\n\n"
                
                # Sentiment analysis
                bullish_keywords = ['moon', 'buy', 'calls', 'bullish', 'long', 'squeeze', 'rocket', '🚀']
                bearish_keywords = ['puts', 'sell', 'bearish', 'short', 'dump', 'crash', 'rip']
                
                bull_count = sum(1 for m in mentions if any(kw in m['title'].lower() for kw in bullish_keywords))
                bear_count = sum(1 for m in mentions if any(kw in m['title'].lower() for kw in bearish_keywords))
                
                result += "**Sentiment Analysis:**\n"
                if bull_count > bear_count * 1.5:
                    result += f"- Overall: **BULLISH** ({bull_count} bullish vs {bear_count} bearish signals)\n"
                    sentiment = "bullish"
                elif bear_count > bull_count * 1.5:
                    result += f"- Overall: **BEARISH** ({bear_count} bearish vs {bull_count} bullish signals)\n"
                    sentiment = "bearish"
                else:
                    result += f"- Overall: **MIXED** ({bull_count} bullish vs {bear_count} bearish)\n"
                    sentiment = "neutral"
                
                # Volume assessment
                total_engagement = sum(m['score'] + m['comments'] for m in mentions)
                result += f"- Engagement: {total_engagement:,} total (scores + comments)\n"
                
                if len(mentions) > 15:
                    result += f"- Volume: **HIGH** - Strong social interest\n"
                elif len(mentions) > 5:
                    result += f"- Volume: Moderate social interest\n"
                else:
                    result += f"- Volume: Low social interest\n"
            else:
                result += f"No Reddit mentions found for {self.ticker} in last {days} days\n"
                sentiment = "none"
            
            print(f"[NEWS] ✓ Reddit: {len(mentions)} mentions, sentiment={sentiment}")
            
            return result + "\n", {
                'source': 'reddit',
                'count': len(mentions),
                'sentiment': sentiment,
                'mentions': top_mentions
            }
            
        except Exception as e:
            print(f"[NEWS] ⚠️  Reddit error: {str(e)}")
            return f"## Reddit Sentiment\n**Error:** {str(e)}\n\n", {'source': 'reddit', 'error': str(e)}

    def get_newsapi_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """
        Get news from NewsAPI (free tier: 100 requests/day)
        Returns formatted string and structured data
        """
        print(f"[NEWS] 🔧 Fetching NewsAPI articles...")
        
        if not self.newsapi_key:
            print(f"[NEWS] ⚠️  NewsAPI key missing")
            return "## NewsAPI\n**Status:** No API key (get free at newsapi.org)\n\n", {'source': 'newsapi', 'error': 'no_key'}
        
        try:
            url = "https://newsapi.org/v2/everything"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': self.ticker,
                'apiKey': self.newsapi_key,
                'from': from_date,
                'sortBy': 'publishedAt',  # Sort by date
                'pageSize': 20,
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = f"## NewsAPI (Last {days} Days)\n\n"
            
            if data.get('status') == 'ok' and data.get('articles'):
                articles = data['articles']
                result += f"**Found {len(articles)} articles**\n\n"
                
                for i, article in enumerate(articles[:8], 1):
                    pub_date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d') if article.get('publishedAt') else datetime.now()
                    days_ago = (datetime.now() - pub_date).days
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
                print(f"[NEWS] ⚠️  NewsAPI: {error_msg}")
                
                return result, {'source': 'newsapi', 'error': error_msg}
            
        except requests.Timeout:
            print(f"[NEWS] ⚠️  NewsAPI timeout")
            return "## NewsAPI\n**Error:** Request timeout\n\n", {'source': 'newsapi', 'error': 'timeout'}
        except Exception as e:
            print(f"[NEWS] ⚠️  NewsAPI error: {str(e)}")
            return f"## NewsAPI\n**Error:** {str(e)}\n\n", {'source': 'newsapi', 'error': str(e)}

    def get_finnhub_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """
        Get news from Finnhub (free tier available)
        Returns formatted string and structured data
        """
        print(f"[NEWS] 🔧 Fetching Finnhub news...")
        
        if not self.finnhub_key:
            print(f"[NEWS] ⚠️  Finnhub key missing")
            return "## Finnhub News\n**Status:** No API key (get free at finnhub.io)\n\n", {'source': 'finnhub', 'error': 'no_key'}
        
        try:
            url = "https://finnhub.io/api/v1/company-news"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'symbol': self.ticker,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = f"## Finnhub News (Last {days} Days)\n\n"
            
            if data and isinstance(data, list) and len(data) > 0:
                result += f"**Found {len(data)} articles**\n\n"
                
                for i, article in enumerate(data[:8], 1):
                    pub_date = datetime.fromtimestamp(article.get('datetime', 0))
                    days_ago = (datetime.now() - pub_date).days
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
                print(f"[NEWS] ⚠️  Finnhub: No news")
                return result, {'source': 'finnhub', 'count': 0}
            
        except requests.Timeout:
            print(f"[NEWS] ⚠️  Finnhub timeout")
            return "## Finnhub News\n**Error:** Request timeout\n\n", {'source': 'finnhub', 'error': 'timeout'}
        except Exception as e:
            print(f"[NEWS] ⚠️  Finnhub error: {str(e)}")
            return f"## Finnhub News\n**Error:** {str(e)}\n\n", {'source': 'finnhub', 'error': str(e)}

    def get_alphavantage_news(self, days: int = 7) -> Tuple[str, Dict[str, Any]]:
        """
        Get news from Alpha Vantage with sentiment scores
        Returns formatted string and structured data
        """
        print(f"[NEWS] 🔧 Fetching Alpha Vantage news...")
        
        if not self.alphavantage_key:
            print(f"[NEWS] ⚠️  Alpha Vantage key missing")
            return "## Alpha Vantage News\n**Status:** No API key (get free at alphavantage.co)\n\n", {'source': 'alphavantage', 'error': 'no_key'}
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': self.ticker,
                'apikey': self.alphavantage_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            result = f"## Alpha Vantage News (Last {days} Days)\n\n"
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if 'feed' in data:
                relevant_articles = []
                
                for article in data['feed']:
                    try:
                        pub_date = datetime.strptime(article.get('time_published', '')[:8], '%Y%m%d')
                        
                        if pub_date > cutoff_date:
                            # Extract ticker-specific sentiment
                            ticker_sentiment = None
                            for ts in article.get('ticker_sentiment', []):
                                if ts.get('ticker') == self.ticker:
                                    ticker_sentiment = ts.get('ticker_sentiment_label')
                                    break
                            
                            relevant_articles.append({
                                'title': article.get('title', 'No title'),
                                'sentiment': ticker_sentiment or 'Neutral',
                                'summary': article.get('summary', ''),
                                'pub_date': pub_date
                            })
                    except:
                        continue
                
                if relevant_articles:
                    result += f"**Found {len(relevant_articles)} articles with sentiment**\n\n"
                    
                    for i, article in enumerate(relevant_articles[:8], 1):
                        days_ago = (datetime.now() - article['pub_date']).days
                        time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                        
                        sentiment_emoji = "📈" if "Bullish" in article['sentiment'] else "📉" if "Bearish" in article['sentiment'] else "➡️"
                        
                        result += f"{i}. **{article['title']}**\n"
                        result += f"   - Sentiment: {sentiment_emoji} {article['sentiment']}\n"
                        result += f"   - Published: {time_str}\n"
                        
                        if article['summary']:
                            result += f"   - {article['summary'][:150]}...\n"
                        result += "\n"
                    
                    # Aggregate sentiment
                    bullish_count = sum(1 for a in relevant_articles if 'Bullish' in a['sentiment'])
                    bearish_count = sum(1 for a in relevant_articles if 'Bearish' in a['sentiment'])
                    
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
                    result += f"No news found in last {days} days\n\n"
                    print(f"[NEWS] ⚠️  Alpha Vantage: No recent news")
                    return result, {'source': 'alphavantage', 'count': 0}
            else:
                result += "No news available\n\n"
                print(f"[NEWS] ⚠️  Alpha Vantage: No feed data")
                return result, {'source': 'alphavantage', 'error': 'no_feed'}
            
        except requests.Timeout:
            print(f"[NEWS] ⚠️  Alpha Vantage timeout")
            return "## Alpha Vantage News\n**Error:** Request timeout\n\n", {'source': 'alphavantage', 'error': 'timeout'}
        except Exception as e:
            print(f"[NEWS] ⚠️  Alpha Vantage error: {str(e)}")
            return f"## Alpha Vantage News\n**Error:** {str(e)}\n\n", {'source': 'alphavantage', 'error': str(e)}

    def analyze_with_llm(self, all_news: str, news_data: List[Dict]) -> str:
        """
        Analyze all gathered news with LLM
        Returns comprehensive analysis
        """
        if not self.client:
            print("[NEWS] ⚠️  No API key - using fallback analysis")
            return self._create_fallback_analysis(all_news, news_data)
        
        try:
            print(f"[NEWS] Analyzing with {self.model}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this news and sentiment data for {self.ticker}:\n\n{all_news}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            analysis = response.choices[0].message.content
            
            # Validate recommendation
            if "RECOMMENDATION:" not in analysis:
                print("[NEWS] ⚠️  Missing recommendation, appending...")
                analysis += "\n\nRECOMMENDATION: HOLD - Confidence: Low"
            
            print(f"[NEWS] ✓ Analysis complete ({len(analysis)} chars)")
            return analysis
            
        except Exception as e:
            print(f"[NEWS] ❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(all_news, news_data)

    def _create_fallback_analysis(self, all_news: str, news_data: List[Dict]) -> str:
        """
        Rule-based fallback analysis
        """
        print("[NEWS] Creating fallback analysis...")
        
        analysis = f"## News & Sentiment Analysis\n"
        analysis += "*Generated using fallback analysis (LLM unavailable)*\n\n"
        
        # Count sources
        successful_sources = [d['source'] for d in news_data if 'error' not in d]
        failed_sources = [d['source'] for d in news_data if 'error' in d]
        
        analysis += f"**Data Sources:** {len(successful_sources)} successful, {len(failed_sources)} failed\n\n"
        
        # Simple sentiment counting
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
        
        # Determine sentiment
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
        
        # Count total articles
        total_articles = sum(d.get('count', 0) for d in news_data if 'count' in d)
        analysis += f"**Total Articles:** {total_articles}\n\n"
        
        analysis += f"RECOMMENDATION: {recommendation} - Confidence: Low\n"
        
        return analysis

    def run(self, sources: Optional[List[str]] = None, days: int = 7) -> str:
        """
        Execute comprehensive news analysis
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"NEWS & SENTIMENT ANALYSIS: {self.ticker}")
        print(f"Period: Last {days} days | Sources: {sources or ['yahoo']}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        if not sources:
            sources = ['yahoo']
        
        all_news = f"# News & Sentiment Analysis: {self.ticker}\n"
        all_news += f"**Analysis Period:** Last {days} Days\n"
        all_news += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        all_news += "="*70 + "\n\n"
        
        news_data = []
        
        # Gather from each source
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
        
        all_news += "\n" + "="*70 + "\n\n"
        
        # Analyze with LLM
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

Available Sources:
  yahoo        - Yahoo Finance (no API key required)
  reddit       - Reddit (requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
  newsapi      - NewsAPI (requires NEWSAPI_KEY)
  finnhub      - Finnhub (requires FINNHUB_KEY)
  alphavantage - Alpha Vantage (requires ALPHAVANTAGE_KEY)
        """
    )
    
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, MSFT)")
    parser.add_argument("--sources", nargs="+",
                       choices=["yahoo", "reddit", "newsapi", "finnhub", "alphavantage"],
                       default=["yahoo"],
                       help="News sources to use (default: yahoo)")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to analyze (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--output", help="Save report to file")
    
    args = parser.parse_args()
    
    try:
        agent = NewsAgent(ticker=args.ticker, api_key=args.api_key, model=args.model)
        result = agent.run(sources=args.sources, days=args.days)
        
        print(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Report saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()