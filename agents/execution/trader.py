"""
Trader - Final execution agent that creates trade orders based on risk decision
Usage: python trader.py AAPL --risk-decision ../managers/risk_decision.json --current-price 195.50
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Any, Optional
from openai import OpenAI

class Trader:
    def __init__(self, ticker, api_key=None, model="gpt-4o-mini"):
        self.ticker = ticker
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = """You are the Head Trader executing final trading decisions.
        Your responsibilities:
        1. Translate risk-approved decisions into executable orders
        2. Determine optimal entry points and timing
        3. Set specific price levels for all orders
        4. Create detailed execution instructions
        5. Provide clear tracking and monitoring plan
        
        Only execute trades that have been APPROVED or MODIFIED by Risk Management.
        End with: TRADE ORDER: EXECUTE/PENDING/CANCELLED - Entry: $X - Size: Y shares - Status: Ready/Waiting"""
        
        # Order types and execution preferences
        self.execution_params = {
            'order_types': ['MARKET', 'LIMIT', 'STOP_LIMIT'],
            'time_in_force': ['DAY', 'GTC', 'IOC', 'FOK'],
            'execution_algos': ['TWAP', 'VWAP', 'POV', 'IMPLEMENTATION_SHORTFALL'],
            'preferred_venues': ['PRIMARY', 'DARK_POOL', 'ALL']
        }
    
    def get_current_price(self, provided_price=None):
        """Get current market price"""
        if provided_price:
            return provided_price
        
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                info = stock.info
                return float(info.get('currentPrice', 0))
        except Exception as e:
            print(f"Error fetching price: {e}")
            return 0
    
    def load_risk_decision(self, decision_file):
        """Load risk manager's decision"""
        if os.path.exists(decision_file):
            with open(decision_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Risk decision file not found: {decision_file}")
            # Try to run risk manager
            return self.run_risk_manager()
    
    def run_risk_manager(self):
        """Run risk manager if decision not available"""
        manager_script = "../managers/risk_manager.py"
        if not os.path.exists(manager_script):
            print("Risk manager not found")
            return None
        
        cmd = ["python", manager_script, self.ticker, "--save-decision", "temp_decision.json"]
        
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists("temp_decision.json"):
                with open("temp_decision.json", 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error running risk manager: {e}")
        
        return None
    
    def create_trade_order(self, risk_decision, current_price):
        """Create executable trade order from risk decision"""
        order = {
            'ticker': self.ticker,
            'action': 'NONE',
            'order_status': 'PENDING',
            'entry_price': 0,
            'shares': 0,
            'total_value': 0,
            'order_type': 'MARKET',
            'time_in_force': 'DAY',
            'stop_loss': {},
            'take_profit': [],
            'execution_instructions': [],
            'monitoring_plan': {}
        }
        
        # Check if trade is approved
        verdict = risk_decision.get('verdict', 'REJECT')
        if verdict == 'REJECT':
            order['action'] = 'NO_TRADE'
            order['order_status'] = 'CANCELLED'
            order['reason'] = 'Risk Management rejected trade'
            return order
        
        # Set action based on research
        position_value = risk_decision.get('final_position', 0)
        if position_value > 0:
            order['action'] = 'BUY'
        else:
            order['action'] = 'HOLD'
            order['order_status'] = 'CANCELLED'
            order['reason'] = 'Zero position size'
            return order
        
        # Calculate shares
        if current_price > 0:
            order['shares'] = int(position_value / current_price)
            order['total_value'] = order['shares'] * current_price
        else:
            order['order_status'] = 'ERROR'
            order['reason'] = 'Unable to get current price'
            return order
        
        # Set entry strategy
        order['entry_price'] = current_price
        order['order_type'] = self.determine_order_type(current_price)
        order['time_in_force'] = 'GTC'  # Good till cancelled
        
        # Add execution instructions based on position size
        if order['shares'] > 1000:
            order['execution_instructions'].append("Split into 3-5 smaller orders")
            order['execution_instructions'].append("Use VWAP or TWAP algorithm")
            order['execution_instructions'].append("Consider dark pool for large blocks")
        elif order['shares'] > 500:
            order['execution_instructions'].append("Split into 2-3 orders")
            order['execution_instructions'].append("Space orders 5-10 minutes apart")
        else:
            order['execution_instructions'].append("Single order execution acceptable")
        
        order['order_status'] = 'READY'
        
        return order
    
    def determine_order_type(self, current_price):
        """Determine best order type based on conditions"""
        # In real implementation, would check:
        # - Market hours
        # - Bid-ask spread
        # - Volume
        # - Volatility
        
        # For now, simple logic
        import datetime
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30)
        market_close = now.replace(hour=16, minute=0)
        
        if market_open <= now <= market_close:
            return 'MARKET'  # During market hours
        else:
            return 'LIMIT'  # After hours
    
    def set_execution_levels(self, order, risk_decision, current_price):
        """Set stop loss and take profit levels"""
        # Extract risk controls
        controls = risk_decision.get('risk_controls', {})
        
        # Set stop loss
        stop_info = controls.get('stop_loss', {})
        stop_pct = stop_info.get('percentage', 5) / 100
        order['stop_loss'] = {
            'type': stop_info.get('type', 'FIXED'),
            'price': current_price * (1 - stop_pct),
            'percentage': stop_info.get('percentage', 5),
            'instructions': f"Set stop at ${current_price * (1 - stop_pct):.2f}"
        }
        
        # Set take profit levels
        tp_levels = controls.get('take_profit', [])
        for i, tp in enumerate(tp_levels, 1):
            target_pct = tp.get('target_pct', 10 * i) / 100
            exit_pct = tp.get('exit_pct', 33)
            shares_to_sell = int(order['shares'] * exit_pct / 100)
            
            order['take_profit'].append({
                'level': i,
                'price': current_price * (1 + target_pct),
                'shares': shares_to_sell,
                'percentage': exit_pct,
                'instructions': f"Sell {shares_to_sell} shares at ${current_price * (1 + target_pct):.2f}"
            })
        
        return order
    
    def create_monitoring_plan(self, order, risk_decision):
        """Create detailed monitoring and adjustment plan"""
        monitoring = {
            'immediate_actions': [],
            'daily_checks': [],
            'weekly_reviews': [],
            'adjustment_triggers': [],
            'exit_conditions': []
        }
        
        # Immediate actions after entry
        monitoring['immediate_actions'] = [
            f"Confirm order filled at ~${order['entry_price']:.2f}",
            f"Set stop loss at ${order['stop_loss']['price']:.2f}",
            "Set take profit orders as specified",
            "Update position tracker"
        ]
        
        # Daily monitoring
        monitoring['daily_checks'] = [
            "Check price vs stop/target levels",
            "Monitor volume for unusual activity",
            "Scan news for material changes",
            "Review technical indicators"
        ]
        
        # Weekly reviews
        monitoring['weekly_reviews'] = [
            "Assess thesis validity",
            "Review risk/reward ratio",
            "Consider stop adjustment if profitable",
            "Evaluate partial profit taking"
        ]
        
        # Adjustment triggers
        if order['stop_loss']['type'] == 'TRAILING':
            monitoring['adjustment_triggers'].append(
                f"Move stop up after +10% gain"
            )
        
        # Exit conditions from risk decision
        conditions = risk_decision.get('conditions', [])
        monitoring['exit_conditions'].extend(conditions)
        
        return monitoring
    
    def generate_execution_summary(self, order, risk_decision):
        """Generate comprehensive execution summary"""
        summary = {
            'trade_id': f"{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'ticker': self.ticker,
            'decision_source': 'Risk Management Approved',
            'order_details': order,
            'risk_parameters': {
                'max_loss': 0,
                'max_gain': 0,
                'risk_reward_ratio': 0
            },
            'execution_checklist': [],
            'tracking': {}
        }
        
        # Only calculate risk parameters if order is ready
        if order['order_status'] == 'READY' and 'price' in order.get('stop_loss', {}):
            summary['risk_parameters']['max_loss'] = order['stop_loss']['price'] * order['shares'] - order['total_value']
            summary['risk_parameters']['max_gain'] = (order['take_profit'][-1]['price'] if order['take_profit'] else order['entry_price'] * 1.3) * order['shares'] - order['total_value']
        
        # Calculate risk/reward ratio
        if summary['risk_parameters']['max_loss'] != 0:
            summary['risk_parameters']['risk_reward_ratio'] = abs(
                summary['risk_parameters']['max_gain'] / summary['risk_parameters']['max_loss']
            )
        
        # Create execution checklist
        summary['execution_checklist'] = [
            {'task': 'Verify available capital', 'status': 'PENDING'},
            {'task': 'Check market conditions', 'status': 'PENDING'},
            {'task': 'Place order', 'status': 'PENDING'},
            {'task': 'Set stop loss', 'status': 'PENDING'},
            {'task': 'Set profit targets', 'status': 'PENDING'},
            {'task': 'Log trade in journal', 'status': 'PENDING'}
        ]
        
        # Tracking information
        summary['tracking'] = {
            'entry_time': None,
            'entry_price_actual': None,
            'current_pnl': 0,
            'current_pnl_pct': 0,
            'days_held': 0,
            'status': 'PENDING'
        }
        
        return summary
    
    def synthesize_with_llm(self, trade_data):
        """Use LLM for final trade synthesis"""
        if not self.client:
            return self.create_fallback_report(trade_data)
        
        try:
            prompt = f"""As the Head Trader, create final execution plan for {self.ticker}:

Risk Decision: {trade_data['risk_decision']['verdict']}
Order Details: {json.dumps(trade_data['order'], indent=2)}
Monitoring Plan: {json.dumps(trade_data['monitoring'], indent=2)}
Execution Summary: {json.dumps(trade_data['summary'], indent=2)}

Create a clear, actionable trade execution plan that includes:
1. Specific order instructions
2. Entry tactics and timing
3. Risk management setup
4. Monitoring requirements
5. Success metrics

Be precise with prices, quantities, and instructions."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Low temp for execution precision
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self.create_fallback_report(trade_data)
    
    def create_fallback_report(self, trade_data):
        """Create report without LLM"""
        order = trade_data['order']
        monitoring = trade_data['monitoring']
        summary = trade_data['summary']
        
        report = f"""
TRADE EXECUTION PLAN - {self.ticker}
{'='*60}
Trade ID: {summary['trade_id']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ORDER STATUS: {order['order_status']}
=====================================
"""
        
        if order['order_status'] == 'READY':
            report += f"""
ACTION: {order['action']}
SHARES: {order['shares']:,}
ENTRY PRICE: ${order['entry_price']:.2f}
TOTAL VALUE: ${order['total_value']:,.2f}

ORDER SPECIFICATIONS:
---------------------
Order Type: {order['order_type']}
Time in Force: {order['time_in_force']}

RISK MANAGEMENT:
----------------
Stop Loss: ${order['stop_loss']['price']:.2f} ({order['stop_loss']['percentage']:.0f}% loss)
Type: {order['stop_loss']['type']}
Max Loss: ${abs(summary['risk_parameters']['max_loss']):,.2f}

PROFIT TARGETS:
---------------"""
            for tp in order['take_profit']:
                report += f"""
Target {tp['level']}: ${tp['price']:.2f} (+{((tp['price']/order['entry_price'])-1)*100:.0f}%)
  → Sell {tp['shares']} shares ({tp['percentage']}% of position)"""
            
            report += f"""

Risk/Reward Ratio: {summary['risk_parameters']['risk_reward_ratio']:.1f}x

EXECUTION INSTRUCTIONS:
-----------------------"""
            for instruction in order['execution_instructions']:
                report += f"\n• {instruction}"
            
            report += f"""

IMMEDIATE ACTIONS:
------------------"""
            for action in monitoring['immediate_actions']:
                report += f"\n□ {action}"
            
            report += f"""

MONITORING PLAN:
----------------
Daily:"""
            for check in monitoring['daily_checks']:
                report += f"\n  • {check}"
            
            report += "\n\nWeekly:"
            for review in monitoring['weekly_reviews']:
                report += f"\n  • {review}"
            
            if monitoring['exit_conditions']:
                report += "\n\nExit Conditions:"
                for condition in monitoring['exit_conditions']:
                    report += f"\n  ⚠️ {condition}"
            
            report += f"""

EXECUTION CHECKLIST:
--------------------"""
            for item in summary['execution_checklist']:
                report += f"\n□ {item['task']}"
            
            report += f"""

{'='*60}
TRADE ORDER: EXECUTE - Entry: ${order['entry_price']:.2f} - Size: {order['shares']} shares - Status: {order['order_status']}
"""
        
        elif order['order_status'] == 'CANCELLED':
            report += f"""
ACTION: NO TRADE
REASON: {order.get('reason', 'Risk Management Rejection')}

No position will be taken at this time.
Continue monitoring for better setup or catalyst.
"""
        
        else:  # ERROR
            report += f"""
ACTION: ERROR
REASON: {order.get('reason', 'Unknown error')}

Unable to create trade order. Please review inputs and try again.
"""
        
        return report
    
    def execute(self, risk_decision=None, decision_file=None, current_price=None):
        """Main execution function"""
        # Load risk decision if not provided
        if not risk_decision:
            if not decision_file:
                decision_file = "../managers/risk_decision.json"
            risk_decision = self.load_risk_decision(decision_file)
        
        if not risk_decision:
            return "Error: No risk decision available", None
        
        print(f"Trader preparing execution for {self.ticker}...")
        
        # Get current price
        if not current_price:
            print("Fetching current price...")
            current_price = self.get_current_price()
        print(f"Current Price: ${current_price:.2f}")
        
        # Create trade order
        print("Creating trade order...")
        order = self.create_trade_order(risk_decision, current_price)
        
        # Set execution levels
        if order['order_status'] == 'READY':
            print("Setting execution levels...")
            order = self.set_execution_levels(order, risk_decision, current_price)
            
            # Create monitoring plan
            print("Creating monitoring plan...")
            monitoring = self.create_monitoring_plan(order, risk_decision)
        else:
            monitoring = {'status': 'N/A - Trade not executed'}
        
        # Generate execution summary
        print("Generating execution summary...")
        summary = self.generate_execution_summary(order, risk_decision)
        
        # Compile all data
        trade_data = {
            'risk_decision': risk_decision,
            'order': order,
            'monitoring': monitoring,
            'summary': summary
        }
        
        # Generate report
        report = self.synthesize_with_llm(trade_data)
        
        return report, order

def main():
    parser = argparse.ArgumentParser(description="Trader - Final execution agent")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--risk-decision", 
                       default="../managers/risk_decision.json",
                       help="Path to risk decision JSON")
    parser.add_argument("--current-price", type=float,
                       help="Current market price (will fetch if not provided)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--save-order", help="Save order details as JSON")
    
    args = parser.parse_args()
    
    trader = Trader(args.ticker, args.api_key, args.model)
    
    # Run execution
    report, order = trader.execute(
        decision_file=args.risk_decision,
        current_price=args.current_price
    )
    
    print(report)
    
    # Save outputs
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSaved report to {args.output}")
    
    if args.save_order and order:
        with open(args.save_order, 'w') as f:
            json.dump(order, f, indent=2)
        print(f"Saved order to {args.save_order}")

if __name__ == "__main__":
    main()