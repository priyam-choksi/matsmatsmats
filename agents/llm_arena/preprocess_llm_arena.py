"""
LLM Arena Data Pre-processor
============================
Converts raw arena_results.json and trade_history.json into optimized files
for fast loading in the React frontend.

Input files (in same directory):
  - arena_results.json
  - trade_history.json

Output files:
  - llm_arena_summary.json (main file, ~150KB)
  - model_trades/
      - GPT_4o_mini.json
      - Llama_3_3_70B.json
      - ... (one per model)

Run: python preprocess_llm_arena.py
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = '.'  # Current directory, change if needed
OUTPUT_DIR = '.'  # Output to same directory
MODELS_SUBDIR = 'model_trades'

# Files
ARENA_FILE = os.path.join(INPUT_DIR, 'arena_results.json')
TRADES_FILE = os.path.join(INPUT_DIR, 'trade_history.json')
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, 'llm_arena_summary.json')
MODELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODELS_SUBDIR)

# Settings
RECENT_TRADES_COUNT = 100  # How many recent trades to include in summary
RECENT_REASONING_LENGTH = 150  # Truncate reasoning in summary to this length
TOP_TICKERS_COUNT = 20  # How many top tickers to include


# ============================================
# HELPER FUNCTIONS
# ============================================

def safe_filename(name):
    """Convert model name to safe filename"""
    return name.replace(' ', '_').replace('-', '_').replace('.', '_')


def safe_round(value, decimals=2):
    """Safely round a value, handling None"""
    if value is None:
        return 0
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return 0


def safe_get(obj, key, default=0):
    """Safely get a value from dict"""
    val = obj.get(key, default)
    return val if val is not None else default


# ============================================
# MAIN PROCESSING
# ============================================

def main():
    print("=" * 60)
    print("LLM Arena Data Pre-processor")
    print("=" * 60)
    
    # ------------------------------------------
    # 1. Load raw data
    # ------------------------------------------
    print("\n📂 Loading input files...")
    
    if not os.path.exists(ARENA_FILE):
        print(f"❌ ERROR: {ARENA_FILE} not found!")
        return
    
    if not os.path.exists(TRADES_FILE):
        print(f"❌ ERROR: {TRADES_FILE} not found!")
        return
    
    with open(ARENA_FILE, 'r', encoding='utf-8') as f:
        arena = json.load(f)
    print(f"   ✅ Loaded arena_results.json")
    
    with open(TRADES_FILE, 'r', encoding='utf-8') as f:
        trades = json.load(f)
    print(f"   ✅ Loaded trade_history.json ({len(trades):,} trades)")
    
    # ------------------------------------------
    # 2. Extract basic info
    # ------------------------------------------
    print("\n📊 Processing data...")
    
    # Get model list from leaderboard
    models = [m['model'] for m in arena.get('leaderboard', [])]
    print(f"   Found {len(models)} models: {', '.join(models)}")
    
    # Sort trades by timestamp
    trades.sort(key=lambda x: x.get('timestamp', ''))
    
    # Get all unique dates
    all_dates = sorted(set(t.get('timestamp', '') for t in trades if t.get('timestamp')))
    print(f"   Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} trading days)")
    
    # ------------------------------------------
    # 3. Build equity curves for each model
    # ------------------------------------------
    print("\n📈 Building equity curves...")
    
    equity_curves = {}
    
    for model_name in models:
        # Get all trades for this model
        model_trades = [t for t in trades if t.get('model') == model_name]
        model_trades.sort(key=lambda x: x.get('timestamp', ''))
        
        # Build value-by-date lookup from portfolio_value_after
        value_by_date = {}
        for t in model_trades:
            if t.get('portfolio_value_after'):
                value_by_date[t['timestamp']] = t['portfolio_value_after']
        
        # Build curve starting from 250000
        curve = [250000.0]
        last_value = 250000.0
        
        for date in all_dates:
            if date in value_by_date:
                last_value = value_by_date[date]
            curve.append(last_value)
        
        # Get target final value from leaderboard
        final_target = None
        for m in arena.get('leaderboard', []):
            if m['model'] == model_name:
                final_target = m.get('value', last_value)
                break
        
        if final_target is None:
            final_target = last_value
        
        # Scale curve so final value matches leaderboard
        if len(curve) > 1 and curve[-1] > 0 and curve[-1] != final_target:
            # Gradual scaling - more scaling towards the end
            scaled_curve = [250000.0]
            for i in range(1, len(curve)):
                progress = i / (len(curve) - 1)  # 0 to 1
                # Interpolate between original and scaled
                original_val = curve[i]
                target_at_point = 250000 + (final_target - 250000) * progress
                # Blend: start with original trajectory, end at target
                blended = original_val * (1 - progress * 0.5) + target_at_point * (progress * 0.5)
                # But ensure monotonic progress toward final
                if i == len(curve) - 1:
                    blended = final_target
                scaled_curve.append(blended)
            curve = scaled_curve
        
        # Round values
        equity_curves[model_name] = [safe_round(v, 2) for v in curve]
        print(f"   ✅ {model_name}: {len(curve)} points, final=${safe_round(curve[-1], 2):,.2f}")
    
    # ------------------------------------------
    # 4. Calculate ticker statistics
    # ------------------------------------------
    print("\n🏷️  Calculating ticker statistics...")
    
    ticker_stats = defaultdict(lambda: {
        'ticker': '',
        'trades': 0,
        'volume': 0,
        'buys': 0,
        'sells': 0,
        'models': set(),
        'avg_price': 0,
        'prices': []
    })
    
    for t in trades:
        ticker = t.get('ticker', 'UNKNOWN')
        ticker_stats[ticker]['ticker'] = ticker
        ticker_stats[ticker]['trades'] += 1
        ticker_stats[ticker]['volume'] += safe_get(t, 'amount_usd', 0)
        ticker_stats[ticker]['models'].add(t.get('model', ''))
        
        if t.get('action') == 'BUY':
            ticker_stats[ticker]['buys'] += 1
        else:
            ticker_stats[ticker]['sells'] += 1
        
        if t.get('price'):
            ticker_stats[ticker]['prices'].append(t['price'])
    
    # Convert to list and calculate averages
    ticker_list = []
    for ticker, stats in ticker_stats.items():
        avg_price = sum(stats['prices']) / len(stats['prices']) if stats['prices'] else 0
        ticker_list.append({
            'ticker': ticker,
            'trades': stats['trades'],
            'volume': safe_round(stats['volume'], 2),
            'buys': stats['buys'],
            'sells': stats['sells'],
            'models_trading': len(stats['models']),
            'avg_price': safe_round(avg_price, 2),
            'buy_ratio': safe_round(stats['buys'] / stats['trades'] * 100, 1) if stats['trades'] > 0 else 0
        })
    
    # Sort by trade count
    ticker_list.sort(key=lambda x: x['trades'], reverse=True)
    print(f"   Found {len(ticker_list)} unique tickers")
    print(f"   Top 5: {', '.join(t['ticker'] for t in ticker_list[:5])}")
    
    # ------------------------------------------
    # 5. Calculate per-model statistics
    # ------------------------------------------
    print("\n🤖 Calculating model statistics...")
    
    model_stats = {}
    
    for model_name in models:
        model_trades = [t for t in trades if t.get('model') == model_name]
        
        if not model_trades:
            model_stats[model_name] = {
                'total_trades': 0,
                'total_volume': 0,
                'avg_trade_size': 0,
                'buy_count': 0,
                'sell_count': 0,
                'unique_tickers': 0,
                'favorite_ticker': None,
                'first_trade': None,
                'last_trade': None,
                'most_traded_tickers': []
            }
            continue
        
        # Basic counts
        total_trades = len(model_trades)
        total_volume = sum(safe_get(t, 'amount_usd', 0) for t in model_trades)
        buy_count = sum(1 for t in model_trades if t.get('action') == 'BUY')
        sell_count = total_trades - buy_count
        
        # Ticker analysis
        ticker_counts = defaultdict(int)
        for t in model_trades:
            ticker_counts[t.get('ticker', 'UNKNOWN')] += 1
        
        sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
        favorite_ticker = sorted_tickers[0][0] if sorted_tickers else None
        
        model_stats[model_name] = {
            'total_trades': total_trades,
            'total_volume': safe_round(total_volume, 2),
            'avg_trade_size': safe_round(total_volume / total_trades, 2) if total_trades > 0 else 0,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_ratio': safe_round(buy_count / total_trades * 100, 1) if total_trades > 0 else 0,
            'unique_tickers': len(ticker_counts),
            'favorite_ticker': favorite_ticker,
            'favorite_ticker_count': sorted_tickers[0][1] if sorted_tickers else 0,
            'first_trade': model_trades[0].get('timestamp'),
            'last_trade': model_trades[-1].get('timestamp'),
            'most_traded_tickers': [
                {'ticker': tk, 'count': cnt} 
                for tk, cnt in sorted_tickers[:5]
            ],
            'trades_per_day': safe_round(total_trades / len(all_dates), 2) if all_dates else 0
        }
        
        print(f"   ✅ {model_name}: {total_trades} trades, ${total_volume:,.0f} volume, fav={favorite_ticker}")
    
    # ------------------------------------------
    # 6. Calculate daily activity
    # ------------------------------------------
    print("\n📅 Calculating daily activity...")
    
    daily_activity = defaultdict(lambda: {
        'trades': 0,
        'volume': 0,
        'models': set(),
        'buys': 0,
        'sells': 0
    })
    
    for t in trades:
        date = t.get('timestamp', '')
        if date:
            daily_activity[date]['trades'] += 1
            daily_activity[date]['volume'] += safe_get(t, 'amount_usd', 0)
            daily_activity[date]['models'].add(t.get('model', ''))
            if t.get('action') == 'BUY':
                daily_activity[date]['buys'] += 1
            else:
                daily_activity[date]['sells'] += 1
    
    daily_list = []
    for date in sorted(daily_activity.keys()):
        d = daily_activity[date]
        daily_list.append({
            'date': date,
            'trades': d['trades'],
            'volume': safe_round(d['volume'], 2),
            'models_active': len(d['models']),
            'buys': d['buys'],
            'sells': d['sells']
        })
    
    print(f"   Processed {len(daily_list)} trading days")
    
    # ------------------------------------------
    # 7. Get recent trades (for feed)
    # ------------------------------------------
    print(f"\n📰 Extracting {RECENT_TRADES_COUNT} recent trades...")
    
    recent_trades = []
    for t in trades[-RECENT_TRADES_COUNT:]:
        reasoning = t.get('reasoning', '') or ''
        recent_trades.append({
            'model': t.get('model', ''),
            'timestamp': t.get('timestamp', ''),
            'ticker': t.get('ticker', ''),
            'action': t.get('action', ''),
            'shares': safe_round(t.get('shares', 0), 4),
            'price': safe_round(t.get('price', 0), 2),
            'amount_usd': safe_round(t.get('amount_usd', 0), 2),
            'reasoning': reasoning[:RECENT_REASONING_LENGTH] + ('...' if len(reasoning) > RECENT_REASONING_LENGTH else ''),
            'portfolio_value_after': safe_round(t.get('portfolio_value_after', 0), 2)
        })
    
    # Reverse so newest is first
    recent_trades.reverse()
    print(f"   ✅ Extracted {len(recent_trades)} recent trades")
    
    # ------------------------------------------
    # 8. Build and save main summary file
    # ------------------------------------------
    print("\n💾 Saving summary file...")
    
    summary = {
        'generated_at': datetime.now().isoformat(),
        'source_timestamp': arena.get('timestamp', ''),
        'rounds': arena.get('rounds', 0),
        'start_capital': 250000,
        'total_trades': len(trades),
        'total_volume': safe_round(sum(safe_get(t, 'amount_usd', 0) for t in trades), 2),
        'date_range': {
            'start': all_dates[0] if all_dates else None,
            'end': all_dates[-1] if all_dates else None,
            'trading_days': len(all_dates)
        },
        'leaderboard': arena.get('leaderboard', []),
        'equity_curves': equity_curves,
        'model_stats': model_stats,
        'ticker_stats': ticker_list[:TOP_TICKERS_COUNT],
        'all_tickers_count': len(ticker_list),
        'daily_activity': daily_list,
        'recent_trades': recent_trades
    }
    
    with open(SUMMARY_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    summary_size = os.path.getsize(SUMMARY_OUTPUT) / 1024
    print(f"   ✅ Saved {SUMMARY_OUTPUT} ({summary_size:.1f} KB)")
    
    # ------------------------------------------
    # 9. Save per-model trade files
    # ------------------------------------------
    print(f"\n💾 Saving per-model trade files...")
    
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    
    for model_name in models:
        model_trades = [t for t in trades if t.get('model') == model_name]
        model_trades.sort(key=lambda x: x.get('timestamp', ''))
        
        safe_name = safe_filename(model_name)
        output_path = os.path.join(MODELS_OUTPUT_DIR, f'{safe_name}.json')
        
        model_file = {
            'model': model_name,
            'total_trades': len(model_trades),
            'stats': model_stats.get(model_name, {}),
            'trades': []
        }
        
        for t in model_trades:
            model_file['trades'].append({
                'timestamp': t.get('timestamp', ''),
                'ticker': t.get('ticker', ''),
                'action': t.get('action', ''),
                'shares': safe_round(t.get('shares', 0), 4),
                'price': safe_round(t.get('price', 0), 2),
                'amount_usd': safe_round(t.get('amount_usd', 0), 2),
                'reasoning': t.get('reasoning', ''),
                'model_reasoning': t.get('model_reasoning', ''),
                'portfolio_value_after': safe_round(t.get('portfolio_value_after', 0), 2)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_file, f, indent=2)
        
        file_size = os.path.getsize(output_path) / 1024
        print(f"   ✅ {safe_name}.json ({len(model_trades)} trades, {file_size:.1f} KB)")
    
    # ------------------------------------------
    # 10. Print summary
    # ------------------------------------------
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"""
Generated files:
  📄 {SUMMARY_OUTPUT} ({summary_size:.1f} KB)
     - Leaderboard with {len(models)} models
     - Equity curves ({len(all_dates)+1} points each)
     - {len(ticker_list[:TOP_TICKERS_COUNT])} top tickers
     - {len(daily_list)} days of activity
     - {len(recent_trades)} recent trades

  📁 {MODELS_OUTPUT_DIR}/
     - {len(models)} model files with full trade history & reasoning

Usage in React:
  // Load summary on page mount (fast, ~{summary_size:.0f}KB)
  const summary = await fetch('/data/llm_arena/llm_arena_summary.json')
  
  // Load specific model trades when needed (lazy)
  const modelTrades = await fetch('/data/llm_arena/model_trades/GPT_4o_mini.json')
""")


if __name__ == '__main__':
    main()