"""
models.py

Contains model classes for different investment types:
- StockPortfolio
- RentalPortfolio
- LandInvestment
- Cost (one-time big cost, e.g., building on land)
"""

from abc import ABC, abstractmethod

class BaseInvestment(ABC):
    """
    Abstract base class for all investment types.
    """

    @abstractmethod
    def project_growth(self, years: int) -> float:
        """
        Return the projected value of this investment after a given number of years (simple compound approach).
        """
        pass

    @property
    @abstractmethod
    def initial_amount(self) -> float:
        """
        Return the initial principal for the investment.
        """
        pass

    @property
    @abstractmethod
    def annual_rate(self) -> float:
        """
        Return the annual rate (growth, return, etc.) for the investment.
        """
        pass


class StockPortfolio(BaseInvestment):
    """
    Represents a stock investment with an initial amount and an annual growth rate.
    """

    def __init__(self, initial_amount: float, annual_growth_rate: float):
        self._initial_amount = initial_amount
        self._annual_growth_rate = annual_growth_rate

    def project_growth(self, years: int) -> float:
        return self._initial_amount * ((1 + self._annual_growth_rate) ** years)

    @property
    def initial_amount(self) -> float:
        return self._initial_amount

    @property
    def annual_rate(self) -> float:
        return self._annual_growth_rate


class RentalPortfolio(BaseInvestment):
    """
    Represents a rental property portfolio with initial equity and annual return rate.
    """

    def __init__(self, initial_equity: float, annual_return_rate: float):
        self._initial_equity = initial_equity
        self._annual_return_rate = annual_return_rate

    def project_growth(self, years: int) -> float:
        return self._initial_equity * ((1 + self._annual_return_rate) ** years)

    @property
    def initial_amount(self) -> float:
        return self._initial_equity

    @property
    def annual_rate(self) -> float:
        return self._annual_return_rate


class LandInvestment(BaseInvestment):
    """
    Represents a land investment with an initial amount and an annual appreciation rate.
    """

    def __init__(self, initial_amount: float, annual_appreciation_rate: float):
        self._initial_amount = initial_amount
        self._annual_appreciation_rate = annual_appreciation_rate

    def project_growth(self, years: int) -> float:
        return self._initial_amount * ((1 + self._annual_appreciation_rate) ** years)

    @property
    def initial_amount(self) -> float:
        return self._initial_amount

    @property
    def annual_rate(self) -> float:
        return self._annual_appreciation_rate


class Cost:
    """
    One-time cost that occurs at a specified year.
    For example, a big outlay in year 10.
    """

    def __init__(self, amount: float, year: int):
        """
        :param amount: The cost amount
        :param year: The year (relative to start) in which this cost occurs
        """
        self.amount = amount
        self.year = year
