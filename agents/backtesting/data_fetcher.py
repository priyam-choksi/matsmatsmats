# agents/backtesting/data_fetcher.py

"""
Fetches historical market data using yfinance.
Provides clean price data for regime detection and backtesting.

Example usage:
    fetcher = DataFetcher()
    data = fetcher.get_price_data("AAPL", days=60)
    print(data.tail())  # Shows recent prices
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class DataFetcher:
    """
    Fetches historical price data from Yahoo Finance.
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Args:
            cache_enabled: If True, caches downloaded data to avoid re-downloading
        """
        self.cache_enabled = cache_enabled
        self._cache = {}
    
    def get_price_data(self, 
                       symbol: str,
                       days: int = 60,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL", "NVDA")
            days: Number of days of history to fetch
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index is DatetimeIndex
            
        Example:
            >>> fetcher = DataFetcher()
            >>> data = fetcher.get_price_data("AAPL", days=30)
            >>> print(data.tail(3))
            #                 Open   High    Low  Close     Volume
            # Date                                                  
            # 2024-01-15  180.00  182.50  179.00  181.50  50000000
            # 2024-01-16  181.00  183.00  180.50  182.75  48000000
            # 2024-01-17  182.50  184.00  181.00  183.25  52000000
        """
        
        # Set end date
        if end_date is None:
            end_date = datetime.now()
        
        # Calculate start date
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer
        
        # Check cache
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        try:
            # Download data from yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Clean up
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Keep only OHLCV columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Latest closing price
        """
        data = self.get_price_data(symbol, days=5)
        return float(data['Close'].iloc[-1])
    
    def clear_cache(self):
        """Clear the price data cache"""
        self._cache.clear()


# Convenience function
def quick_fetch(symbol: str, days: int = 60) -> pd.DataFrame:
    """Quick way to fetch data without creating object"""
    fetcher = DataFetcher()
    return fetcher.get_price_data(symbol, days)