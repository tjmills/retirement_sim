"""
scenario.py

Encapsulates scenario creation and execution:
- invests in multiple assets
- includes one-time costs
- naive total growth calculation
- year-by-year simulation for charting
- retirement duration calculation
"""

from typing import List
from models import BaseInvestment, Cost
import calculators

class ScenarioManager:
    """
    Manages different investment scenarios using the provided models and calculators.
    """

    def __init__(self):
        self.investments: List[BaseInvestment] = []
        self.costs: List[Cost] = []

    def add_investment(self, investment: BaseInvestment):
        self.investments.append(investment)

    def add_cost(self, cost: Cost):
        self.costs.append(cost)

    def total_value_after_years(self, years: int) -> float:
        """
        (Naive approach):
        1) Sum final values of each investment using their compound formulas.
        2) Subtract costs (also in a naive, discounted manner if you want).
        Here, we'll do a simpler approach: we sum up each investment, then subtract
        the 'face value' of costs that occur up to 'years' from the total.
        """
        total = calculators.total_portfolio_value(self.investments, years)
        # Subtract any costs that occur on or before 'years'
        # (This is naive because it doesn't factor lost growth from the removed principal.)
        for cost_obj in self.costs:
            if cost_obj.year <= years:
                total -= cost_obj.amount
        return max(total, 0.0)  # Avoid negative if costs exceed growth

    def retirement_duration(self, years_before_retirement: int, annual_withdrawal: float, average_growth_rate: float) -> int:
        """
        1) Calculate total portfolio value after 'years_before_retirement'
        2) Estimate how many years that value would last with fixed withdrawal + growth
        """
        initial_value = self.total_value_after_years(years_before_retirement)
        return calculators.years_to_deplete(initial_value, annual_withdrawal, average_growth_rate)

    def simulate_growth(self, years: int) -> List[float]:
        """
        Return a list of the portfolio's total value for each year from 0..years,
        performing a more accurate year-by-year simulation:
          - We track each investment's principal individually.
          - Each year, each principal grows by its annual_rate.
          - If a one-time cost occurs that year, it's subtracted proportionally from the total.
        """
        # 1) Set up an array of principals for each investment
        principals = [inv.initial_amount for inv in self.investments]
        
        # We'll store the total each year (including year 0)
        total_values = []

        # Record year 0
        total_values.append(sum(principals))

        # 2) Loop through each year, applying growth and costs
        for year in range(1, years + 1):
            # Grow each investment
            for i, inv in enumerate(self.investments):
                principals[i] *= (1 + inv.annual_rate)

            # Sum up
            total_portfolio = sum(principals)

            # Check if there's a cost in this year, subtract proportionally
            costs_for_this_year = [c for c in self.costs if c.year == year]
            if costs_for_this_year:
                total_cost = sum(c.amount for c in costs_for_this_year)
                # If total_portfolio is 0, can't subtract anything
                if total_portfolio > 0:
                    # We remove the total cost proportionally from each principal
                    ratio = total_cost / total_portfolio
                    for i in range(len(principals)):
                        principals[i] -= principals[i] * ratio

                    # Re-sum after cost
                    total_portfolio = sum(principals)

            total_values.append(total_portfolio)

        return total_values
