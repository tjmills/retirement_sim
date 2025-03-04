import pytest
import numpy as np
import pandas as pd

# Import functions from your simulation module.
# Adjust the module name "app" if your code is located elsewhere.
from app import (
    standard_mortgage_payment,
    update_rental_property_monthly,
    simulate_stock,
    calculate_social_security,
    estimate_healthcare_costs,
    calculate_rmd,
    analyze_social_security_strategy,
    calculate_tax_brackets,
    analyze_asset_allocations,
    run_simulation
)

# =====================================================
# Tests for standard_mortgage_payment
# =====================================================
def test_standard_mortgage_payment_normal():
    principal = 100000
    annual_interest_rate = 0.05
    mortgage_years = 30
    payment = standard_mortgage_payment(principal, annual_interest_rate, mortgage_years)
    # Expected monthly payment is approximately 536.82 for these parameters.
    assert np.isclose(payment, 536.82, atol=1.0)

def test_standard_mortgage_payment_zero_principal():
    assert standard_mortgage_payment(0, 0.05, 30) == 0.0

def test_standard_mortgage_payment_zero_interest():
    principal = 100000
    years = 30
    expected_payment = principal / (years * 12)  # ~277.78
    payment = standard_mortgage_payment(principal, 0, years)
    assert np.isclose(payment, expected_payment, atol=0.01)

def test_standard_mortgage_payment_zero_years():
    assert standard_mortgage_payment(100000, 0.05, 0) == 0.0

@pytest.mark.parametrize("principal,rate,years,expected", [
    (0, 0.05, 30, 0.0),
    (100000, 0, 30, 100000/(30*12)),
    (100000, 0.05, 0, 0.0)
])
def test_standard_mortgage_payment_parametrized(principal, rate, years, expected):
    payment = standard_mortgage_payment(principal, rate, years)
    assert np.isclose(payment, expected, atol=0.01)

# =====================================================
# Tests for update_rental_property_monthly
# =====================================================
def test_update_rental_property_monthly_basic():
    np.random.seed(0)
    rental = {
        "property_value": 200000,
        "mortgage_balance": 150000,
        "cost_basis": 180000,
        "mortgage_interest_rate": 0.04,
        "monthly_payment": 800,
        "monthly_rent_base": 1500,
        "vacancy_rate": 0.05,
        "property_tax_rate": 0.01,
        "maintenance_rate": 0.01,
        "rental_income_tax_rate": 0.2,
        "annual_depreciation": 180000 / 27.5,
        "app_mean": 0.03,
        "app_std": 0.01,
        "rent_inflation": 0.02,
        "sale_year": 0,
        "rental_cap_gains_tax": 0.15
    }
    updated_rental, net_cf = update_rental_property_monthly(rental, year_index=0)
    for key in ["property_value", "mortgage_balance", "cost_basis"]:
        assert key in updated_rental
    assert isinstance(net_cf, float)
    assert -50000 < net_cf < 50000

def test_update_rental_property_monthly_consistency():
    np.random.seed(42)
    rental = {
        "property_value": 250000,
        "mortgage_balance": 200000,
        "cost_basis": 230000,
        "mortgage_interest_rate": 0.045,
        "monthly_payment": 900,
        "monthly_rent_base": 1800,
        "vacancy_rate": 0.08,
        "property_tax_rate": 0.012,
        "maintenance_rate": 0.01,
        "rental_income_tax_rate": 0.22,
        "annual_depreciation": 230000 / 27.5,
        "app_mean": 0.04,
        "app_std": 0.015,
        "rent_inflation": 0.025,
        "sale_year": 0,
        "rental_cap_gains_tax": 0.15
    }
    result1, net_cf1 = update_rental_property_monthly(rental, year_index=1)
    np.random.seed(42)
    result2, net_cf2 = update_rental_property_monthly(rental, year_index=1)
    pd.testing.assert_series_equal(pd.Series(result1), pd.Series(result2))
    assert np.isclose(net_cf1, net_cf2)

# =====================================================
# Tests for simulate_stock
# =====================================================
def test_simulate_stock_normal():
    np.random.seed(1)
    current_stock_value = 50000
    stock_annual_contribution = 10000
    expected_return = 0.07
    volatility = 0.15
    cap_gains_tax_rate = 0.15
    dividend_yield = 0.02
    dividend_tax_rate = 0.15
    withdrawal = 10000
    new_value, actual_withdrawal = simulate_stock(
        current_stock_value,
        stock_annual_contribution,
        expected_return,
        volatility,
        cap_gains_tax_rate,
        dividend_yield,
        dividend_tax_rate,
        withdrawal=withdrawal,
        inflation_adjusted_withdrawal=False,
        current_year=0,
        inflation_rate=0.03
    )
    assert actual_withdrawal == withdrawal
    assert new_value >= 0

def test_simulate_stock_with_withdrawal_exceeds():
    np.random.seed(2)
    current_stock_value = 30000
    withdrawal = 50000
    new_value, actual_withdrawal = simulate_stock(
        current_stock_value,
        0,
        0.07,
        0.15,
        0.15,
        0.02,
        0.15,
        withdrawal,
        inflation_adjusted_withdrawal=False,
        current_year=0,
        inflation_rate=0.03
    )
    # Should withdraw all available funds.
    assert actual_withdrawal == current_stock_value
    assert new_value >= 0

# =====================================================
# Tests for calculate_social_security
# =====================================================
def test_calculate_social_security_under_full():
    benefit = calculate_social_security(1970, 65, 80000)
    assert 40000 < benefit < 42000

def test_calculate_social_security_full():
    benefit_full = calculate_social_security(1970, 67, 80000)
    benefit_reduced = calculate_social_security(1970, 65, 80000)
    assert benefit_full > benefit_reduced

# =====================================================
# Tests for estimate_healthcare_costs
# =====================================================
def test_estimate_healthcare_costs_length():
    costs = estimate_healthcare_costs(50, 65, 90)
    assert len(costs) == 90 - 65 + 1

def test_estimate_healthcare_costs_positive():
    costs = estimate_healthcare_costs(50, 65, 90)
    assert all(c > 0 for c in costs)

# =====================================================
# Tests for calculate_rmd
# =====================================================
def test_calculate_rmd_value():
    rmd = calculate_rmd(72, 100000)
    assert np.isclose(rmd, 100000/25.6, atol=0.1)

# =====================================================
# Tests for analyze_social_security_strategy
# =====================================================
def test_analyze_social_security_strategy_format():
    df = analyze_social_security_strategy(1970, 80000)
    expected_columns = ['Claim Age', 'Monthly Benefit', 'Total by Age 80', 'Total by Age 85', 'Total by Age 90']
    for col in expected_columns:
        assert col in df.columns

# =====================================================
# Tests for calculate_tax_brackets
# =====================================================
def test_calculate_tax_brackets_correctness():
    df, total_tax = calculate_tax_brackets(50000)
    # Expected: 22000 at 10% + (50000-22000)*12% ≈ 2200 + 3360 = 5560
    assert np.isclose(total_tax, 5560, atol=50)
    for col in ['Bracket', 'Income in Bracket', 'Tax in Bracket']:
        assert col in df.columns

# =====================================================
# Tests for analyze_asset_allocations
# =====================================================
def test_analyze_asset_allocations_structure():
    df = analyze_asset_allocations(50000, 30, n_sims=100)
    expected_columns = ['Allocation', 'Median', 'Worst 5%', 'Best 5%']
    for col in expected_columns:
        assert col in df.columns
    assert all(df['Median'] >= 0)

# =====================================================
# Integration Tests for run_simulation
# =====================================================
@pytest.fixture
def minimal_simulation_params():
    # Minimal parameters for running the simulation
    return {
        "birth_year": 1970,
        "retirement_age": 65,
        "life_expectancy": 85,
        "annual_income": 80000,
        "annual_budget": 30000,
        "tax_deferred_balance": 100000,
        "roth_balance": 50000,
        "n_sims": 5,
        "stock_initial": 50000,
        "stock_annual_contribution": 10000,
        "stock_expected_return": 0.07,
        "stock_volatility": 0.15,
        "cap_gains_tax_rate": 0.15,
        "dividend_yield": 0.02,
        "dividend_tax_rate": 0.15,
        "withdrawal": 30000,
        # One rental purchased in simulation year 5:
        "rentals_data": [{
            "description": "Rental 1",
            "purchase_year": 5,
            "property_value": 200000,
            "downpayment": 40000,
            "cost_basis": 180000,
            "mortgage_balance": 120000,
            "mortgage_interest_rate": 0.04,
            "monthly_payment": standard_mortgage_payment(120000, 0.04, 30),
            "monthly_rent_base": 1500,
            "sale_year": 0,
            "rental_cap_gains_tax": 0.15,
            "property_tax_rate": 0.01,
            "maintenance_rate": 0.01,
            "vacancy_rate": 0.05,
            "rental_income_tax_rate": 0.2,
            "rent_inflation": 0.02,
            "app_mean": 0.03,
            "app_std": 0.01,
            "annual_depreciation": 180000 / 27.5
        }],
        "primary_residence_value": 300000,
        "primary_mortgage_balance": 150000,
        "primary_mortgage_interest_rate": 0.04,
        "primary_mortgage_years_left": 25,
        "primary_appreciation_mean": 0.03,
        "primary_appreciation_std": 0.01,
        "luxury_expenses": [],
        "inflation_rate": 0.03,
        "healthcare_inflation": 0.05
    }

def test_run_simulation_structure(minimal_simulation_params):
    result = run_simulation(
        minimal_simulation_params["birth_year"],
        minimal_simulation_params["retirement_age"],
        minimal_simulation_params["life_expectancy"],
        minimal_simulation_params["annual_income"],
        minimal_simulation_params["annual_budget"],
        minimal_simulation_params["tax_deferred_balance"],
        minimal_simulation_params["roth_balance"],
        minimal_simulation_params["n_sims"],
        minimal_simulation_params["stock_initial"],
        minimal_simulation_params["stock_annual_contribution"],
        minimal_simulation_params["stock_expected_return"],
        minimal_simulation_params["stock_volatility"],
        minimal_simulation_params["cap_gains_tax_rate"],
        minimal_simulation_params["dividend_yield"],
        minimal_simulation_params["dividend_tax_rate"],
        minimal_simulation_params["withdrawal"],
        minimal_simulation_params["rentals_data"],
        minimal_simulation_params["primary_residence_value"],
        minimal_simulation_params["primary_mortgage_balance"],
        minimal_simulation_params["primary_mortgage_interest_rate"],
        minimal_simulation_params["primary_mortgage_years_left"],
        minimal_simulation_params["primary_appreciation_mean"],
        minimal_simulation_params["primary_appreciation_std"],
        minimal_simulation_params["luxury_expenses"],
        minimal_simulation_params["inflation_rate"],
        minimal_simulation_params["healthcare_inflation"]
    )
    for key in ["all_sims", "combined", "summary", "failure_rate"]:
        assert key in result
    summary_df = result["summary"]
    for col in ["Year", "Taxable Stock Value Median", "Roth Balance Median"]:
        assert col in summary_df.columns

def test_run_simulation_downpayment_effect(minimal_simulation_params):
    # Run simulation with rental purchase.
    np.random.seed(42)
    result_with_rental = run_simulation(
        minimal_simulation_params["birth_year"],
        minimal_simulation_params["retirement_age"],
        minimal_simulation_params["life_expectancy"],
        minimal_simulation_params["annual_income"],
        minimal_simulation_params["annual_budget"],
        minimal_simulation_params["tax_deferred_balance"],
        minimal_simulation_params["roth_balance"],
        minimal_simulation_params["n_sims"],
        minimal_simulation_params["stock_initial"],
        minimal_simulation_params["stock_annual_contribution"],
        minimal_simulation_params["stock_expected_return"],
        minimal_simulation_params["stock_volatility"],
        minimal_simulation_params["cap_gains_tax_rate"],
        minimal_simulation_params["dividend_yield"],
        minimal_simulation_params["dividend_tax_rate"],
        minimal_simulation_params["withdrawal"],
        minimal_simulation_params["rentals_data"],
        minimal_simulation_params["primary_residence_value"],
        minimal_simulation_params["primary_mortgage_balance"],
        minimal_simulation_params["primary_mortgage_interest_rate"],
        minimal_simulation_params["primary_mortgage_years_left"],
        minimal_simulation_params["primary_appreciation_mean"],
        minimal_simulation_params["primary_appreciation_std"],
        minimal_simulation_params["luxury_expenses"],
        minimal_simulation_params["inflation_rate"],
        minimal_simulation_params["healthcare_inflation"]
    )
    summary_with = result_with_rental["summary"]

    # Run simulation without any rental purchase.
    params_no_rental = minimal_simulation_params.copy()
    params_no_rental["rentals_data"] = []
    np.random.seed(42)
    result_without_rental = run_simulation(
        params_no_rental["birth_year"],
        params_no_rental["retirement_age"],
        params_no_rental["life_expectancy"],
        params_no_rental["annual_income"],
        params_no_rental["annual_budget"],
        params_no_rental["tax_deferred_balance"],
        params_no_rental["roth_balance"],
        params_no_rental["n_sims"],
        params_no_rental["stock_initial"],
        params_no_rental["stock_annual_contribution"],
        params_no_rental["stock_expected_return"],
        params_no_rental["stock_volatility"],
        params_no_rental["cap_gains_tax_rate"],
        params_no_rental["dividend_yield"],
        params_no_rental["dividend_tax_rate"],
        params_no_rental["withdrawal"],
        params_no_rental["rentals_data"],
        params_no_rental["primary_residence_value"],
        params_no_rental["primary_mortgage_balance"],
        params_no_rental["primary_mortgage_interest_rate"],
        params_no_rental["primary_mortgage_years_left"],
        params_no_rental["primary_appreciation_mean"],
        params_no_rental["primary_appreciation_std"],
        params_no_rental["luxury_expenses"],
        params_no_rental["inflation_rate"],
        params_no_rental["healthcare_inflation"]
    )
    summary_without = result_without_rental["summary"]

    # Compare taxable stock values in year 5.
    stock_with = summary_with.loc[summary_with["Year"] == 5, "Taxable Stock Value Median"].values[0]
    stock_without = summary_without.loc[summary_without["Year"] == 5, "Taxable Stock Value Median"].values[0]
    # The simulation with rental (which deducts a 40K downpayment from taxable stocks) should have a lower value.
    assert stock_with < stock_without

# =====================================================
# Additional Edge Case Tests
# =====================================================
def test_simulate_stock_no_contribution_no_withdrawal():
    np.random.seed(3)
    current_stock = 50000
    new_value, withdrawal = simulate_stock(
        current_stock, 0, 0.07, 0.15, 0.15, 0.02, 0.15,
        withdrawal=0, inflation_adjusted_withdrawal=False, current_year=0, inflation_rate=0.03
    )
    assert new_value >= 0

# =====================================================
# Run pytest if this file is executed directly
# =====================================================
if __name__ == "__main__":
    pytest.main()
