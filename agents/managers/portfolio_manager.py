"""
Portfolio Manager - Smart Trade Execution, Stop Loss Tracking & Backtest Runner
Single script that handles everything for backtesting and TradingView export

Usage:
    # Process single decision
    python portfolio_manager.py --decision outputs/AAPL/risk_decision.json
    
    # Run full backtest on all decisions in a folder
    python portfolio_manager.py --backtest --outputs-dir outputs/
    
    # Check portfolio status
    python portfolio_manager.py --status
    
    # Export to TradingView CSV
    python portfolio_manager.py --export
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import yfinance as yf
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install yfinance pandas")
    sys.exit(1)


class PortfolioManager:
    """
    Smart Portfolio Manager that:
    - Processes risk_decisions and executes trades
    - Tracks stop losses and profit targets using real price data
    - Checks if stops/targets were hit BETWEEN analysis dates
    - Runs full backtests across multiple tickers chronologically
    - Outputs TradingView-compatible CSV with accurate dates
    """
    
    def __init__(
        self,
        portfolio_file: str = "portfolio_state.json",
        trades_csv: str = "tradingview_trades.csv",
        detailed_log: str = "detailed_trades.json",
        initial_capital: float = 1000000.0
    ):
        self.portfolio_file = portfolio_file
        self.trades_csv = trades_csv
        self.detailed_log = detailed_log
        self.initial_capital = initial_capital
        
        # Price cache to avoid repeated API calls
        self.price_cache: Dict[str, pd.DataFrame] = {}
        
        # Load or initialize portfolio
        self.portfolio = self._load_portfolio()
        
        # Configuration
        self.config = {
            "default_stop_loss_pct": 0.10,
            "default_profit_targets": [10, 20, 30],
            "default_scale_out": [0.33, 0.33, 0.34],
            "max_position_pct": 0.25,
            "min_cash_reserve_pct": 0.05,
            "confidence_multipliers": {"HIGH": 1.2, "MEDIUM": 1.0, "LOW": 0.7},
            "systemic_risk_keywords": [
                "market crash", "recession", "systemic", "market-wide",
                "sector collapse", "contagion", "liquidity crisis", "bear market"
            ]
        }
    
    # ==================== Portfolio State Management ====================
    
    def _load_portfolio(self) -> Dict:
        """Load or initialize portfolio state."""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        return self._create_new_portfolio()
    
    def _create_new_portfolio(self) -> Dict:
        """Create fresh portfolio state."""
        return {
            "created_at": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "cash": self.initial_capital,
            "positions": {},
            "trade_history": [],
            "metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_realized_pnl": 0.0,
                "stops_triggered": 0,
                "targets_hit": 0
            }
        }
    
    def _save_portfolio(self):
        """Save portfolio state."""
        self.portfolio["last_updated"] = datetime.now().isoformat()
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.portfolio = self._create_new_portfolio()
        self._save_portfolio()
        
        # Clear CSV file
        if os.path.exists(self.trades_csv):
            os.remove(self.trades_csv)
        if os.path.exists(self.detailed_log):
            os.remove(self.detailed_log)
        
        print(f"[PM] Portfolio reset to ${self.initial_capital:,.0f}")
    
    # ==================== Price Data ====================
    
    def _get_price_history(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get OHLC price history from yfinance with caching."""
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            # Add buffer days for safety
            start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5)
            end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=5)
            
            hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            
            if not hist.empty:
                self.price_cache[cache_key] = hist
                return hist
            return None
        except Exception as e:
            print(f"[PM] Error fetching {ticker} prices: {e}")
            return None
    
    def _get_price_on_date(self, ticker: str, date: str) -> Optional[float]:
        """Get closing price on specific date."""
        hist = self._get_price_history(ticker, date, date)
        if hist is not None and not hist.empty:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            # Find closest date
            for idx in hist.index:
                if idx.date() <= target_date.date():
                    return float(hist.loc[idx, 'Close'])
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
        return None
    
    def _check_stop_hit_between_dates(
        self, 
        ticker: str, 
        stop_price: float, 
        start_date: str, 
        end_date: str
    ) -> Optional[Dict]:
        """Check if stop loss was hit between two dates using daily lows."""
        hist = self._get_price_history(ticker, start_date, end_date)
        if hist is None or hist.empty:
            return None
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        for idx, row in hist.iterrows():
            row_date = idx.date() if hasattr(idx, 'date') else idx
            if isinstance(row_date, datetime):
                row_date = row_date.date()
            
            # Skip dates outside range
            if row_date <= start_dt.date() or row_date > end_dt.date():
                continue
            
            # Check if low hit stop
            if row['Low'] <= stop_price:
                return {
                    "hit": True,
                    "date": row_date.strftime("%Y-%m-%d"),
                    "trigger_price": stop_price,
                    "low_price": row['Low'],
                    "close_price": row['Close']
                }
        
        return None
    
    def _check_target_hit_between_dates(
        self, 
        ticker: str, 
        target_price: float, 
        start_date: str, 
        end_date: str
    ) -> Optional[Dict]:
        """Check if profit target was hit between two dates using daily highs."""
        hist = self._get_price_history(ticker, start_date, end_date)
        if hist is None or hist.empty:
            return None
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        for idx, row in hist.iterrows():
            row_date = idx.date() if hasattr(idx, 'date') else idx
            if isinstance(row_date, datetime):
                row_date = row_date.date()
            
            if row_date <= start_dt.date() or row_date > end_dt.date():
                continue
            
            # Check if high hit target
            if row['High'] >= target_price:
                return {
                    "hit": True,
                    "date": row_date.strftime("%Y-%m-%d"),
                    "trigger_price": target_price,
                    "high_price": row['High'],
                    "close_price": row['Close']
                }
        
        return None
    
    # ==================== Stop Loss & Target Checking ====================
    
    def check_all_positions(self, current_date: str) -> List[Dict]:
        """
        Check ALL open positions for stop loss or target hits up to current_date.
        Returns list of triggered actions.
        """
        triggered_actions = []
        
        for ticker, position in list(self.portfolio["positions"].items()):
            last_check_date = position.get("last_checked_date", position["entry_date"])
            
            # Skip if current date is not after last check
            if current_date <= last_check_date:
                continue
            
            entry_price = position["avg_entry_price"]
            stop_price = entry_price * (1 - position.get("stop_loss_pct", self.config["default_stop_loss_pct"]))
            
            # Check stop loss first
            stop_hit = self._check_stop_hit_between_dates(ticker, stop_price, last_check_date, current_date)
            
            if stop_hit:
                triggered_actions.append({
                    "ticker": ticker,
                    "action": "STOP_LOSS",
                    "date": stop_hit["date"],
                    "price": stop_hit["trigger_price"],
                    "shares": position["shares"],
                    "reason": f"Stop loss hit: Low ${stop_hit['low_price']:.2f} breached stop ${stop_price:.2f}"
                })
                continue  # Don't check targets if stopped out
            
            # Check profit targets
            targets = position.get("profit_targets", self.config["default_profit_targets"])
            targets_hit = position.get("targets_hit", 0)
            scale_out = position.get("scale_out", self.config["default_scale_out"])
            
            for i, target_pct in enumerate(targets):
                if i < targets_hit:
                    continue  # Already hit this target
                
                target_price = entry_price * (1 + target_pct / 100)
                target_hit = self._check_target_hit_between_dates(ticker, target_price, last_check_date, current_date)
                
                if target_hit:
                    sell_pct = scale_out[min(i, len(scale_out) - 1)]
                    shares_to_sell = int(position["shares"] * sell_pct)
                    
                    if shares_to_sell > 0:
                        triggered_actions.append({
                            "ticker": ticker,
                            "action": "TAKE_PROFIT",
                            "date": target_hit["date"],
                            "price": target_hit["trigger_price"],
                            "shares": shares_to_sell,
                            "target_level": i + 1,
                            "reason": f"Target {i+1} hit: High ${target_hit['high_price']:.2f} reached +{target_pct}%"
                        })
                        
                        # Update targets hit (will be saved after execution)
                        position["targets_hit"] = i + 1
                        
                        # Move stop to breakeven after first target
                        if i == 0:
                            position["stop_loss_pct"] = 0.0
                    break  # Only one target per check cycle
            
            # Update last checked date
            position["last_checked_date"] = current_date
        
        self._save_portfolio()
        return triggered_actions
    
    # ==================== Trade Execution ====================
    
    def _execute_buy(
        self,
        ticker: str,
        shares: int,
        price: float,
        date: str,
        reasoning: str,
        risk_decision: Dict
    ) -> Dict:
        """Execute buy order and record to portfolio."""
        cost = shares * price
        
        if cost > self.portfolio["cash"]:
            # Adjust shares to what we can afford
            shares = int(self.portfolio["cash"] * 0.95 / price)
            cost = shares * price
            if shares <= 0:
                return {"success": False, "error": "Insufficient cash"}
        
        self.portfolio["cash"] -= cost
        
        # Extract stop loss from risk_decision
        stop_loss_pct = risk_decision.get("stop_loss_pct", 10) / 100
        risk_controls = risk_decision.get("risk_controls", {})
        if risk_controls.get("stop_loss", {}).get("percentage"):
            stop_loss_pct = risk_controls["stop_loss"]["percentage"] / 100
        
        profit_targets = risk_decision.get("profit_targets", self.config["default_profit_targets"])
        scale_out = risk_controls.get("take_profit", {}).get("scale_out", self.config["default_scale_out"])
        
        # Create or update position
        if ticker in self.portfolio["positions"]:
            pos = self.portfolio["positions"][ticker]
            total_shares = pos["shares"] + shares
            total_cost = (pos["shares"] * pos["avg_entry_price"]) + cost
            pos["avg_entry_price"] = total_cost / total_shares
            pos["shares"] = total_shares
            pos["last_updated"] = date
        else:
            self.portfolio["positions"][ticker] = {
                "shares": shares,
                "avg_entry_price": price,
                "entry_date": date,
                "last_updated": date,
                "last_checked_date": date,
                "stop_loss_pct": stop_loss_pct,
                "profit_targets": profit_targets,
                "scale_out": scale_out,
                "targets_hit": 0
            }
        
        # Record trade
        trade = {
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "total": cost,
            "date": date,
            "reasoning": reasoning[:200],
            "stop_loss_pct": stop_loss_pct,
            "profit_targets": profit_targets
        }
        self.portfolio["trade_history"].append(trade)
        self.portfolio["metrics"]["total_trades"] += 1
        
        self._append_to_csv(ticker, "buy", shares, price, date)
        self._save_portfolio()
        
        return {"success": True, "trade": trade}
    
    def _execute_sell(
        self,
        ticker: str,
        shares: int,
        price: float,
        date: str,
        reasoning: str,
        action_type: str = "SELL"
    ) -> Dict:
        """Execute sell order and record to portfolio."""
        if ticker not in self.portfolio["positions"]:
            return {"success": False, "error": f"No position in {ticker}"}
        
        pos = self.portfolio["positions"][ticker]
        shares_to_sell = min(shares, pos["shares"])
        proceeds = shares_to_sell * price
        
        # Calculate P&L
        cost_basis = shares_to_sell * pos["avg_entry_price"]
        pnl = proceeds - cost_basis
        pnl_pct = (price - pos["avg_entry_price"]) / pos["avg_entry_price"]
        
        self.portfolio["cash"] += proceeds
        
        # Update position
        pos["shares"] -= shares_to_sell
        if pos["shares"] <= 0:
            del self.portfolio["positions"][ticker]
        else:
            pos["last_updated"] = date
        
        # Update metrics
        self.portfolio["metrics"]["total_trades"] += 1
        self.portfolio["metrics"]["total_realized_pnl"] += pnl
        if pnl > 0:
            self.portfolio["metrics"]["winning_trades"] += 1
        else:
            self.portfolio["metrics"]["losing_trades"] += 1
        
        if action_type == "STOP_LOSS":
            self.portfolio["metrics"]["stops_triggered"] += 1
        elif action_type == "TAKE_PROFIT":
            self.portfolio["metrics"]["targets_hit"] += 1
        
        # Record trade
        trade = {
            "ticker": ticker,
            "action": action_type,
            "shares": shares_to_sell,
            "price": price,
            "total": proceeds,
            "date": date,
            "reasoning": reasoning[:200],
            "pnl": pnl,
            "pnl_pct": pnl_pct
        }
        self.portfolio["trade_history"].append(trade)
        
        self._append_to_csv(ticker, "sell", shares_to_sell, price, date)
        self._save_portfolio()
        
        return {"success": True, "trade": trade, "pnl": pnl, "pnl_pct": pnl_pct}
    
    def _append_to_csv(self, ticker: str, side: str, qty: int, price: float, date: str):
        """Append trade to TradingView CSV."""
        # Determine exchange
        exchange = "NASDAQ"  # Default
        
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, 'w') as f:
                f.write("Symbol,Side,Qty,Fill Price,Commission,Closing Time\n")
        
        with open(self.trades_csv, 'a') as f:
            f.write(f"{exchange}:{ticker},{side},{qty},{price:.2f},0,{date} 10:00:00\n")
    
    # ==================== Decision Processing ====================
    
    def _calculate_position_size(self, risk_decision: Dict, price: float) -> Tuple[int, str]:
        """Calculate position size based on risk decision and portfolio state."""
        confidence = risk_decision.get("confidence", "MEDIUM")
        recommended_pct = risk_decision.get("final_position_pct", 5) / 100
        
        multiplier = self.config["confidence_multipliers"].get(confidence, 1.0)
        adjusted_pct = min(recommended_pct * multiplier, self.config["max_position_pct"])
        
        available_cash = self.portfolio["cash"] * (1 - self.config["min_cash_reserve_pct"])
        portfolio_value = self._calculate_portfolio_value()
        
        position_dollars = min(portfolio_value * adjusted_pct, available_cash)
        shares = int(position_dollars / price) if price > 0 else 0
        
        reasoning = f"Confidence={confidence}, Size={adjusted_pct*100:.1f}%, Shares={shares}"
        return shares, reasoning
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total = self.portfolio["cash"]
        for ticker, pos in self.portfolio["positions"].items():
            price = self._get_price_on_date(ticker, datetime.now().strftime("%Y-%m-%d"))
            if price:
                total += pos["shares"] * price
            else:
                total += pos["shares"] * pos["avg_entry_price"]
        return total
    
    def _detect_systemic_risk(self, risk_decision: Dict, research_synthesis: Optional[Dict]) -> Tuple[bool, str]:
        """Detect if decision indicates systemic/market-wide risk."""
        text = risk_decision.get("reasoning", "").lower()
        text += " " + " ".join(str(f) for f in risk_decision.get("key_factors", []))
        
        if research_synthesis:
            text += " " + research_synthesis.get("full_synthesis", "").lower()
            bear = research_synthesis.get("bear_thesis", {})
            text += " " + bear.get("core_thesis", "").lower()
        
        found = [kw for kw in self.config["systemic_risk_keywords"] if kw in text]
        return len(found) >= 2, f"Systemic risk: {', '.join(found)}" if found else ""
    
    def process_decision(
        self,
        risk_decision_file: str,
        research_synthesis_file: Optional[str] = None
    ) -> Dict:
        """Process a single risk decision."""
        with open(risk_decision_file, 'r') as f:
            risk_decision = json.load(f)
        
        ticker = risk_decision.get("ticker")
        analysis_date = risk_decision.get("analysis_date", datetime.now().strftime("%Y-%m-%d"))
        verdict = risk_decision.get("verdict", "").upper()
        
        # Load research synthesis if available
        research_synthesis = None
        if research_synthesis_file and os.path.exists(research_synthesis_file):
            with open(research_synthesis_file, 'r') as f:
                research_synthesis = json.load(f)
        
        print(f"\n[PM] Processing {ticker} on {analysis_date} - Verdict: {verdict}")
        
        # FIRST: Check all positions for stop/target hits up to this date
        triggered = self.check_all_positions(analysis_date)
        for action in triggered:
            print(f"   ⚡ {action['action']}: {action['ticker']} @ ${action['price']:.2f} on {action['date']}")
            self._execute_sell(
                action["ticker"],
                action["shares"],
                action["price"],
                action["date"],
                action["reason"],
                action["action"]
            )
        
        # Get current price
        price = self._get_price_on_date(ticker, analysis_date)
        if not price:
            print(f"   ✗ Could not get price for {ticker}")
            return {"success": False, "error": "No price data"}
        
        result = {"ticker": ticker, "date": analysis_date, "verdict": verdict, "actions": []}
        
        # Process based on verdict
        if verdict in ["APPROVE", "BUY", "STRONG_BUY"]:
            shares, reasoning = self._calculate_position_size(risk_decision, price)
            if shares > 0:
                print(f"   📈 BUY {shares} shares @ ${price:.2f}")
                buy_result = self._execute_buy(ticker, shares, price, analysis_date, reasoning, risk_decision)
                result["actions"].append({"type": "BUY", "result": buy_result})
            else:
                print(f"   ⏸️ No buy - insufficient size")
        
        elif verdict in ["REJECT", "SELL", "AVOID"]:
            print(f"   🚫 {verdict} - No new position")
            
            # Check if we should exit existing position
            if ticker in self.portfolio["positions"]:
                is_systemic, reason = self._detect_systemic_risk(risk_decision, research_synthesis)
                if is_systemic or "company" in risk_decision.get("reasoning", "").lower():
                    pos = self.portfolio["positions"][ticker]
                    print(f"   ⚠️ Exiting existing position due to risk")
                    self._execute_sell(ticker, pos["shares"], price, analysis_date, reason or verdict, "RISK_EXIT")
        
        elif verdict == "HOLD":
            print(f"   ⏸️ HOLD")
        
        # Check for systemic risk affecting other positions
        is_systemic, systemic_reason = self._detect_systemic_risk(risk_decision, research_synthesis)
        if is_systemic:
            print(f"   🌍 SYSTEMIC RISK DETECTED - Reducing all positions")
            for pos_ticker, pos in list(self.portfolio["positions"].items()):
                if pos_ticker != ticker:
                    pos_price = self._get_price_on_date(pos_ticker, analysis_date)
                    if pos_price:
                        shares_to_sell = int(pos["shares"] * 0.3)
                        if shares_to_sell > 0:
                            print(f"      Reducing {pos_ticker}: {shares_to_sell} shares")
                            self._execute_sell(pos_ticker, shares_to_sell, pos_price, analysis_date, 
                                             systemic_reason, "SYSTEMIC_REDUCTION")
        
        self._save_detailed_log(result, risk_decision, triggered)
        return result
    
    def _save_detailed_log(self, result: Dict, risk_decision: Dict, triggered: List):
        """Save detailed decision log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "ticker": result["ticker"],
            "date": result["date"],
            "verdict": result["verdict"],
            "triggered_stops_targets": triggered,
            "portfolio_cash": self.portfolio["cash"],
            "open_positions": len(self.portfolio["positions"])
        }
        
        if os.path.exists(self.detailed_log):
            with open(self.detailed_log, 'r') as f:
                log = json.load(f)
        else:
            log = {"decisions": []}
        
        log["decisions"].append(entry)
        with open(self.detailed_log, 'w') as f:
            json.dump(log, f, indent=2)
    
    # ==================== Batch Backtest ====================
    
    def run_backtest(self, outputs_dir: str, reset: bool = True) -> Dict:
        """
        Run full backtest on all risk_decisions in outputs directory.
        Processes all tickers chronologically, checking stops/targets between dates.
        """
        print(f"\n{'='*60}")
        print("PORTFOLIO MANAGER - FULL BACKTEST")
        print('='*60)
        
        if reset:
            self.reset_portfolio()
        
        # Find all risk_decision.json files
        pattern = os.path.join(outputs_dir, "**", "risk_decision.json")
        decision_files = glob.glob(pattern, recursive=True)
        
        if not decision_files:
            print(f"No risk_decision.json files found in {outputs_dir}")
            return {"success": False, "error": "No decisions found"}
        
        print(f"Found {len(decision_files)} risk decisions")
        
        # Extract date from each decision and sort chronologically
        decisions = []
        for filepath in decision_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                decisions.append({
                    "file": filepath,
                    "ticker": data.get("ticker"),
                    "date": data.get("analysis_date", "2099-12-31"),
                    "verdict": data.get("verdict")
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        # Sort by date
        decisions.sort(key=lambda x: x["date"])
        
        print(f"Processing {len(decisions)} decisions from {decisions[0]['date']} to {decisions[-1]['date']}")
        print("-" * 60)
        
        # Process each decision chronologically
        for i, dec in enumerate(decisions):
            # Find corresponding research_synthesis
            dec_dir = os.path.dirname(dec["file"])
            synthesis_file = os.path.join(dec_dir, "research_synthesis.json")
            
            self.process_decision(
                dec["file"],
                synthesis_file if os.path.exists(synthesis_file) else None
            )
        
        # Final check for any remaining stops/targets
        final_date = (datetime.strptime(decisions[-1]["date"], "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        print(f"\n[PM] Final position check up to {final_date}...")
        triggered = self.check_all_positions(final_date)
        for action in triggered:
            print(f"   ⚡ {action['action']}: {action['ticker']} @ ${action['price']:.2f} on {action['date']}")
            self._execute_sell(action["ticker"], action["shares"], action["price"], 
                             action["date"], action["reason"], action["action"])
        
        # Print summary
        self.print_summary()
        
        return {
            "success": True,
            "total_decisions": len(decisions),
            "trades_csv": self.trades_csv,
            "final_value": self._calculate_portfolio_value()
        }
    
    # ==================== Status & Summary ====================
    
    def print_status(self):
        """Print current portfolio status."""
        print(f"\n{'='*60}")
        print("PORTFOLIO STATUS")
        print('='*60)
        print(f"Cash: ${self.portfolio['cash']:,.2f}")
        print(f"Initial Capital: ${self.portfolio['initial_capital']:,.2f}")
        
        total_value = self.portfolio["cash"]
        
        if self.portfolio["positions"]:
            print(f"\n--- Open Positions ({len(self.portfolio['positions'])}) ---")
            for ticker, pos in self.portfolio["positions"].items():
                price = self._get_price_on_date(ticker, datetime.now().strftime("%Y-%m-%d"))
                if price:
                    value = pos["shares"] * price
                    pnl = (price - pos["avg_entry_price"]) / pos["avg_entry_price"] * 100
                    total_value += value
                    print(f"  {ticker}: {pos['shares']} @ ${pos['avg_entry_price']:.2f} → ${price:.2f} ({pnl:+.1f}%)")
                    print(f"          Stop: -{pos.get('stop_loss_pct', 0.1)*100:.0f}% | Targets hit: {pos.get('targets_hit', 0)}")
        
        print(f"\nTotal Portfolio Value: ${total_value:,.2f}")
        print(f"Total Return: {(total_value / self.portfolio['initial_capital'] - 1) * 100:+.2f}%")
    
    def print_summary(self):
        """Print backtest summary with key metrics."""
        m = self.portfolio["metrics"]
        total_value = self._calculate_portfolio_value()
        total_return = (total_value / self.portfolio["initial_capital"] - 1) * 100
        
        total_closed = m["winning_trades"] + m["losing_trades"]
        win_rate = (m["winning_trades"] / total_closed * 100) if total_closed > 0 else 0
        
        print(f"\n{'='*60}")
        print("BACKTEST SUMMARY")
        print('='*60)
        print(f"Initial Capital:    ${self.portfolio['initial_capital']:,.2f}")
        print(f"Final Value:        ${total_value:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"Realized P&L:       ${m['total_realized_pnl']:,.2f}")
        print("-" * 40)
        print(f"Total Trades:       {m['total_trades']}")
        print(f"Winning Trades:     {m['winning_trades']}")
        print(f"Losing Trades:      {m['losing_trades']}")
        print(f"Win Rate:           {win_rate:.1f}%")
        print("-" * 40)
        print(f"Stops Triggered:    {m['stops_triggered']}")
        print(f"Targets Hit:        {m['targets_hit']}")
        print("-" * 40)
        print(f"Open Positions:     {len(self.portfolio['positions'])}")
        print(f"Cash Remaining:     ${self.portfolio['cash']:,.2f}")
        print(f"\n✓ TradingView CSV: {self.trades_csv}")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Manager - Backtest & TradingView Export")
    parser.add_argument("--decision", help="Process single risk_decision.json")
    parser.add_argument("--synthesis", help="Path to research_synthesis.json")
    parser.add_argument("--backtest", action="store_true", help="Run full backtest")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory with risk decisions")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio before backtest")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--portfolio-file", default="portfolio_state.json")
    parser.add_argument("--csv-file", default="tradingview_trades.csv")
    
    args = parser.parse_args()
    
    pm = PortfolioManager(
        portfolio_file=args.portfolio_file,
        trades_csv=args.csv_file,
        initial_capital=args.capital
    )
    
    if args.status:
        pm.print_status()
    elif args.backtest:
        pm.run_backtest(args.outputs_dir, reset=args.reset or True)
    elif args.decision:
        pm.process_decision(args.decision, args.synthesis)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()