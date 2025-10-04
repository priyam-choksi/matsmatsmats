"""
Enhanced News Sentiment Analysis Agent with 7-Day Limit
Supports: Reddit (PRAW), NewsAPI, Finnhub, Alpha Vantage, Yahoo Finance
Usage: python news_agent.py AAPL --sources reddit newsapi yahoo --days 7 --output report.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import requests
import yfinance as yf
from openai import OpenAI

# Optional imports
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    print("Note: PRAW not installed. Run 'pip install praw' for Reddit support")

class NewsAgent:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Load various API keys from environment
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.finnhub_key = os.getenv("FINNHUB_KEY")
        self.alphavantage_key = os.getenv("ALPHAVANTAGE_KEY")
        
        # Reddit credentials
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "Trading Bot 1.0")
        
        self.system_prompt = """You are an EOD trading news analyst. Analyze news and sentiment for overnight trading.
Focus on: 1) Recent catalysts 2) Sentiment shifts 3) Upcoming events 4) Risk factors.
End with: RECOMMENDATION: BUY/HOLD/SELL - Confidence: High/Medium/Low"""
    
    def get_yahoo_news(self, days=7):
        """Fetch news from Yahoo Finance (No API key needed)"""
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news[:20] if stock.news else []  # Get more initially
            
            news_summary = f"\n=== YAHOO FINANCE NEWS (Last {days} Days) ===\n"
            cutoff_date = datetime.now() - timedelta(days=days)
            relevant_news = []
            
            for item in news:
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                # Only include news from last N days
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
                        'time_str': time_str,
                        'pub_time': pub_time
                    })
            
            if relevant_news:
                for i, item in enumerate(relevant_news[:10], 1):  # Limit to 10 most recent
                    news_summary += f"{i}. {item['title']}\n"
                    news_summary += f"   Source: {item['publisher']} | {item['time_str']}\n\n"
            else:
                news_summary += f"No news found in the last {days} days\n"
            
            return news_summary
        except Exception as e:
            return f"Yahoo News Error: {str(e)}\n"
    
    def get_reddit_sentiment(self, days=7):
        """Get Reddit sentiment using PRAW (limited to last N days)"""
        if not PRAW_AVAILABLE:
            return "Reddit: PRAW not installed\n"
        
        if not (self.reddit_client_id and self.reddit_client_secret):
            return "Reddit: No credentials (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)\n"
        
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
            mentions = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            reddit_summary = f"\n=== REDDIT SENTIMENT (Last {days} Days) ===\n"
            
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    # Search for ticker in each subreddit (week filter for 7 days)
                    time_filter = 'week' if days <= 7 else 'month'
                    
                    for submission in subreddit.search(self.ticker, time_filter=time_filter, limit=10):
                        created_time = datetime.fromtimestamp(submission.created_utc)
                        
                        # Only include posts from last N days
                        if created_time > cutoff_date:
                            mentions.append({
                                'title': submission.title,
                                'score': submission.score,
                                'comments': submission.num_comments,
                                'subreddit': sub_name,
                                'created': created_time
                            })
                except:
                    continue
            
            # Sort by score
            mentions = sorted(mentions, key=lambda x: x['score'], reverse=True)[:10]
            
            if mentions:
                reddit_summary += f"Found {len(mentions)} mentions across Reddit:\n\n"
                for m in mentions[:5]:  # Top 5
                    days_ago = (datetime.now() - m['created']).days
                    time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                    reddit_summary += f"- {m['title']}\n"
                    reddit_summary += f"  r/{m['subreddit']} | Score: {m['score']} | Comments: {m['comments']} | {time_str}\n\n"
                
                # Calculate sentiment
                bullish_keywords = ['moon', 'buy', 'calls', 'bullish', 'long', 'squeeze']
                bearish_keywords = ['puts', 'sell', 'bearish', 'short', 'dump', 'crash']
                
                bull_count = sum(1 for m in mentions if any(kw in m['title'].lower() for kw in bullish_keywords))
                bear_count = sum(1 for m in mentions if any(kw in m['title'].lower() for kw in bearish_keywords))
                
                if bull_count > bear_count:
                    reddit_summary += f"Overall Sentiment: BULLISH ({bull_count} bull vs {bear_count} bear signals)\n"
                elif bear_count > bull_count:
                    reddit_summary += f"Overall Sentiment: BEARISH ({bear_count} bear vs {bull_count} bull signals)\n"
                else:
                    reddit_summary += f"Overall Sentiment: NEUTRAL\n"
            else:
                reddit_summary += f"No recent Reddit mentions found for {self.ticker} in last {days} days\n"
            
            return reddit_summary
            
        except Exception as e:
            return f"Reddit Error: {str(e)}\n"
    
    def get_newsapi_news(self, days=7):
        """Get news from NewsAPI (Free tier: 100 requests/day)"""
        if not self.newsapi_key:
            return "NewsAPI: No API key (get free at newsapi.org)\n"
        
        try:
            url = "https://newsapi.org/v2/everything"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': self.ticker,
                'apiKey': self.newsapi_key,
                'from': from_date,
                'sortBy': 'relevancy',
                'pageSize': 10,
                'language': 'en'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            news_summary = f"\n=== NEWSAPI NEWS (Last {days} Days) ===\n"
            
            if data.get('status') == 'ok' and data.get('articles'):
                for i, article in enumerate(data['articles'][:5], 1):
                    pub_date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d') if article.get('publishedAt') else datetime.now()
                    days_ago = (datetime.now() - pub_date).days
                    time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                    
                    news_summary += f"{i}. {article.get('title', 'No title')}\n"
                    news_summary += f"   Source: {article.get('source', {}).get('name', 'Unknown')} | {time_str}\n"
                    news_summary += f"   {article.get('description', '')[:100]}...\n\n"
            else:
                news_summary += f"No news found or API error: {data.get('message', 'Unknown error')}\n"
            
            return news_summary
            
        except Exception as e:
            return f"NewsAPI Error: {str(e)}\n"
    
    def get_finnhub_news(self, days=7):
        """Get news from Finnhub (Free tier available)"""
        if not self.finnhub_key:
            return "Finnhub: No API key (get free at finnhub.io)\n"
        
        try:
            url = f"https://finnhub.io/api/v1/company-news"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'symbol': self.ticker,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            news_summary = f"\n=== FINNHUB NEWS (Last {days} Days) ===\n"
            
            if data and isinstance(data, list):
                for i, article in enumerate(data[:5], 1):
                    pub_date = datetime.fromtimestamp(article.get('datetime', 0))
                    days_ago = (datetime.now() - pub_date).days
                    time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                    
                    news_summary += f"{i}. {article.get('headline', 'No title')}\n"
                    news_summary += f"   Source: {article.get('source', 'Unknown')} | {time_str}\n"
                    news_summary += f"   {article.get('summary', '')[:100]}...\n\n"
            else:
                news_summary += "No Finnhub news found\n"
            
            return news_summary
            
        except Exception as e:
            return f"Finnhub Error: {str(e)}\n"
    
    def get_alphavantage_news(self, days=7):
        """Get news from Alpha Vantage (Free tier: 25 requests/day)"""
        if not self.alphavantage_key:
            return "Alpha Vantage: No API key (get free at alphavantage.co)\n"
        
        try:
            url = "https://www.alphavantage.co/query"
            # Alpha Vantage doesn't support date filtering in free tier
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': self.ticker,
                'apikey': self.alphavantage_key,
                'limit': 20  # Get more to filter by date
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            news_summary = f"\n=== ALPHA VANTAGE NEWS (Last {days} Days) ===\n"
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if 'feed' in data:
                relevant_count = 0
                for article in data['feed']:
                    # Parse date from Alpha Vantage format
                    try:
                        pub_date = datetime.strptime(article.get('time_published', '')[:8], '%Y%m%d')
                        
                        if pub_date > cutoff_date and relevant_count < 5:
                            relevant_count += 1
                            days_ago = (datetime.now() - pub_date).days
                            time_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                            
                            news_summary += f"{relevant_count}. {article.get('title', 'No title')}\n"
                            
                            # Get ticker-specific sentiment
                            ticker_sentiment = None
                            for ts in article.get('ticker_sentiment', []):
                                if ts.get('ticker') == self.ticker:
                                    ticker_sentiment = ts.get('ticker_sentiment_label')
                                    break
                            
                            if ticker_sentiment:
                                news_summary += f"   Sentiment: {ticker_sentiment} | {time_str}\n"
                            else:
                                news_summary += f"   {time_str}\n"
                            news_summary += f"   {article.get('summary', '')[:100]}...\n\n"
                    except:
                        continue
                        
                if relevant_count == 0:
                    news_summary += f"No news found in last {days} days\n"
            else:
                news_summary += "No Alpha Vantage news found\n"
            
            return news_summary
            
        except Exception as e:
            return f"Alpha Vantage Error: {str(e)}\n"
    
    def analyze_with_llm(self, all_news):
        """Use LLM to analyze all news sources"""
        if not self.client:
            # Simple sentiment counting
            text_lower = all_news.lower()
            bullish = text_lower.count('buy') + text_lower.count('upgrade') + text_lower.count('bullish')
            bearish = text_lower.count('sell') + text_lower.count('downgrade') + text_lower.count('bearish')
            
            if bullish > bearish:
                return f"{all_news}\n\nOverall: Bullish sentiment\nRECOMMENDATION: BUY - Confidence: Low"
            elif bearish > bullish:
                return f"{all_news}\n\nOverall: Bearish sentiment\nRECOMMENDATION: SELL - Confidence: Low"
            else:
                return f"{all_news}\n\nOverall: Neutral sentiment\nRECOMMENDATION: HOLD - Confidence: Low"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this news for {self.ticker}:\n{all_news}"}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}\n\nRECOMMENDATION: HOLD - Confidence: Low"
    
    def run(self, sources=None, days=7):
        """Execute full analysis with specified lookback period"""
        if not sources:
            sources = ['yahoo']  # Default to Yahoo if no sources specified
        
        all_news = f"NEWS ANALYSIS FOR {self.ticker}\n"
        all_news += f"Analysis Period: Last {days} Days\n"
        all_news += "="*50 + "\n"
        
        # Gather news from requested sources
        if 'yahoo' in sources:
            all_news += self.get_yahoo_news(days)
        
        if 'reddit' in sources:
            all_news += self.get_reddit_sentiment(days)
        
        if 'newsapi' in sources:
            all_news += self.get_newsapi_news(days)
        
        if 'finnhub' in sources:
            all_news += self.get_finnhub_news(days)
        
        if 'alphavantage' in sources:
            all_news += self.get_alphavantage_news(days)
        
        # Analyze with LLM
        analysis = self.analyze_with_llm(all_news)
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="News Sentiment Analysis Agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--sources", nargs="+", 
                       choices=["yahoo", "reddit", "newsapi", "finnhub", "alphavantage"],
                       default=["yahoo"],
                       help="News sources to use")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to look back (default: 7)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--setup", action="store_true", help="Show API setup instructions")
    
    args = parser.parse_args()
    
    if args.setup:
        print("""
add apis dumbass
        """)
        return
    
    agent = NewsAgent(args.ticker, args.api_key, args.model)
    result = agent.run(args.sources, args.days)
    
    print(result)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()