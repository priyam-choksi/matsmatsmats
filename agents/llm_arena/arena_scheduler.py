"""
Arena Scheduler
Automates data collection and arena runs
Can be run as a cron job or daemon
"""

import schedule
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/llm_arena/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_INTERVAL_DAYS = 3
ARENA_RUN_AFTER_COLLECTION = True


def collect_latest_data():
    """Collect latest market data"""
    logger.info("Starting data collection...")
    
    try:
        result = subprocess.run(
            ["python", "data_collector.py", "--latest"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Data collection completed successfully")
            logger.debug(result.stdout)
        else:
            logger.error(f"Data collection failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Data collection timed out")
    except Exception as e:
        logger.error(f"Data collection error: {e}")


def run_arena_round():
    """Run one round of the arena with latest data"""
    logger.info("Running arena round...")
    
    try:
        result = subprocess.run(
            ["python", "llm_arena.py", "--run", "--rounds", "1"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Arena round completed successfully")
            logger.debug(result.stdout)
        else:
            logger.error(f"Arena round failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Arena round timed out")
    except Exception as e:
        logger.error(f"Arena round error: {e}")


def scheduled_job():
    """Main scheduled job - collect data and optionally run arena"""
    logger.info("=" * 50)
    logger.info("SCHEDULED JOB STARTED")
    logger.info("=" * 50)
    
    collect_latest_data()
    
    if ARENA_RUN_AFTER_COLLECTION:
        time.sleep(5)  # Brief pause between tasks
        run_arena_round()
    
    logger.info("Scheduled job completed")


def is_trading_day() -> bool:
    """Check if today is a trading day (weekday)"""
    return datetime.now().weekday() < 5


def run_scheduler():
    """Run the scheduler daemon"""
    logger.info("Arena Scheduler Started")
    logger.info(f"Collection interval: every {COLLECTION_INTERVAL_DAYS} days")
    logger.info(f"Auto-run arena: {ARENA_RUN_AFTER_COLLECTION}")
    
    # Schedule job every N days at 6 PM (after market close)
    schedule.every(COLLECTION_INTERVAL_DAYS).days.at("18:00").do(
        lambda: scheduled_job() if is_trading_day() else logger.info("Skipping - not a trading day")
    )
    
    # Also run immediately on start if it's a trading day
    if is_trading_day():
        logger.info("Running initial job...")
        scheduled_job()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def run_once():
    """Run collection and arena once (for testing or manual runs)"""
    logger.info("Running one-time collection and arena...")
    scheduled_job()


# =============================================================================
# CRON SETUP HELPER
# =============================================================================

def print_cron_setup():
    """Print instructions for setting up as cron job"""
    script_path = Path(__file__).resolve()
    python_path = "python3"  # Adjust if needed
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    CRON SETUP INSTRUCTIONS                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  To run this automatically every 3 days, add to your crontab:    ║
║                                                                   ║
║  1. Open crontab:                                                 ║
║     $ crontab -e                                                  ║
║                                                                   ║
║  2. Add this line (runs at 6 PM every 3rd day):                  ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
    print(f"0 18 */3 * * cd {script_path.parent} && {python_path} {script_path.name} --once")
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║  Alternative: Run as daemon (stays running):                      ║
║     $ python arena_scheduler.py --daemon                          ║
║                                                                   ║
║  Or with nohup:                                                   ║
║     $ nohup python arena_scheduler.py --daemon > scheduler.out &  ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arena Scheduler")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (continuous)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--cron", action="store_true", help="Show cron setup instructions")
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, don't run arena")
    
    args = parser.parse_args()
    
    if args.cron:
        print_cron_setup()
    elif args.daemon:
        run_scheduler()
    elif args.once:
        run_once()
    elif args.collect_only:
        collect_latest_data()
    else:
        print("Arena Scheduler")
        print("=" * 40)
        print("\nUsage:")
        print("  python arena_scheduler.py --once          # Run once now")
        print("  python arena_scheduler.py --daemon        # Run continuously")
        print("  python arena_scheduler.py --cron          # Show cron setup")
        print("  python arena_scheduler.py --collect-only  # Just collect data")