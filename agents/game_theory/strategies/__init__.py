# agents/game_theory/strategies/__init__.py

"""
Game Theory Trading Strategies

Five strategies representing different trading personalities:
1. Actual Market - Trusts the base system completely
2. Buy & Hold - Patient long-term investor
3. Cooperator - Amplifies signals when market agrees
4. Defector - Contrarian, bets against herd
5. Tit-for-Tat - Adaptive learner, mirrors winners
"""

from .actual_market import ActualMarketStrategy
from .buy_hold import BuyHoldStrategy
from .cooperator import CooperatorStrategy
from .defector import DefectorStrategy
from .tit_for_tat import TitForTatStrategy

__all__ = [
    'ActualMarketStrategy',
    'BuyHoldStrategy',
    'CooperatorStrategy',
    'DefectorStrategy',
    'TitForTatStrategy'
]