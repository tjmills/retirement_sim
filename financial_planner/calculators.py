"""
calculators.py

Provides the core calculation logic for:
- total_portfolio_value for a naive compound approach
- years_to_deplete for retirement
"""

from typing import List
from models import BaseInvestment

def total_portfolio_value(investments: List[BaseInvestment], years: int) -> float:
    """
    Sums the projected (naive) future value of all individual investments after 'years'.
    """
    total = 0
    for investment in investments:
        total += investment.project_growth(years)
    return total


def years_to_deplete(portfolio_value: float, annual_withdrawal: float, annual_growth_rate: float) -> int:
    """
    Calculates how many years it would take to deplete the portfolio
    given a fixed annual withdrawal amount, assuming a constant annual growth rate.
    """
    years = 0
    current_value = portfolio_value
    
    while current_value > 0:
        current_value *= (1 + annual_growth_rate)  # growth
        current_value -= annual_withdrawal         # withdrawal
        years += 1
        if current_value <= 0:
            break
    
    return years
