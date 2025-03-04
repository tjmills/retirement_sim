import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date
from jinja2 import Template
import base64
from io import BytesIO
from plotly import graph_objects as go

# -----------------------------------------------
# Helper: Safe Formatter for Pandas Styler
# -----------------------------------------------
def safe_formatter(fmt):
    def formatter(x):
        if x is None or pd.isna(x):
            return ""
        return fmt.format(x)
    return formatter

# -----------------------------------------------
# Mortgage Payment Formula
# -----------------------------------------------
def standard_mortgage_payment(principal, annual_interest_rate, mortgage_years):
    """
    Computes the monthly payment for a standard amortizing mortgage.
    Returns 0 if principal, mortgage_years, or a negative rate is provided.
    In a zero-interest scenario, returns principal divided by the number of months.
    """
    if principal <= 0 or mortgage_years <= 0 or annual_interest_rate < 0:
        return 0.0

    monthly_rate = annual_interest_rate / 12.0
    n_months = mortgage_years * 12

    if monthly_rate == 0:
        return principal / n_months

    payment = principal * (monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1)
    return payment

# -----------------------------------------------
# Rental Update (Monthly)
# -----------------------------------------------
def update_rental_property_monthly(rental, year_index=0):
    """
    One-year update for a single rental property using monthly amortization.
    Returns:
      updated_rental (dict with updated property_value, mortgage_balance, cost_basis, etc.)
      net_rental_cash_flow (after paying mortgage interest, property tax, maintenance, and rental tax)
    """
    # Unpack rental params
    property_value = rental["property_value"]
    mortgage_balance = rental["mortgage_balance"]
    cost_basis = rental["cost_basis"]
    annual_interest_rate = rental["mortgage_interest_rate"]
    monthly_payment = rental["monthly_payment"]
    monthly_rent_base = rental["monthly_rent_base"]
    vacancy_rate = rental["vacancy_rate"]
    property_tax_rate = rental["property_tax_rate"]
    maintenance_rate = rental["maintenance_rate"]
    rental_income_tax_rate = rental["rental_income_tax_rate"]
    annual_depreciation = rental["annual_depreciation"]
    app_mean = rental["app_mean"]
    app_std = rental["app_std"]
    rent_inflation = rental["rent_inflation"]

    # 1) Property appreciation
    annual_appreciation = np.random.normal(loc=app_mean, scale=app_std)
    property_value *= (1 + annual_appreciation)

    # 2) Gross rent for the year
    current_monthly_rent = monthly_rent_base * ((1 + rent_inflation) ** year_index)
    gross_annual_rent = current_monthly_rent * 12 * (1 - vacancy_rate)

    # 3) Property tax + maintenance
    annual_property_tax = property_value * property_tax_rate
    annual_maintenance = property_value * maintenance_rate

    # 4) Monthly mortgage loop
    monthly_interest_rate = annual_interest_rate / 12.0
    new_mortgage_balance = mortgage_balance
    total_interest_paid = 0.0

    for _ in range(12):
        interest_for_month = new_mortgage_balance * monthly_interest_rate
        principal_for_month = monthly_payment - interest_for_month
        if principal_for_month < 0:
            principal_for_month = 0
            interest_for_month = monthly_payment
        if principal_for_month > new_mortgage_balance:
            principal_for_month = new_mortgage_balance
        new_mortgage_balance -= principal_for_month
        total_interest_paid += interest_for_month

    # 5) Net operating income (before mortgage principal)
    net_operating_income = gross_annual_rent - annual_property_tax - annual_maintenance - total_interest_paid

    # 6) Depreciation reduces taxable income
    taxable_income = net_operating_income - annual_depreciation
    if taxable_income < 0:
        taxable_income = 0.0

    # 7) Pay rental income tax
    rental_income_tax = taxable_income * rental_income_tax_rate

    # 8) Net rental cash flow AFTER tax
    net_rental_cash_flow = net_operating_income - rental_income_tax

    # 9) Update cost basis
    new_cost_basis = cost_basis - annual_depreciation
    if new_cost_basis < 0:
        new_cost_basis = 0.0

    updated_rental = rental.copy()
    updated_rental["property_value"] = property_value
    updated_rental["mortgage_balance"] = new_mortgage_balance
    updated_rental["cost_basis"] = new_cost_basis

    return updated_rental, net_rental_cash_flow

# -----------------------------------------------
# Detailed Rental Cash Flow Breakdown Function
# -----------------------------------------------
def calculate_rental_cashflow_breakdown(rental, year_index=0):
    """
    Calculates a detailed breakdown of rental property cash flow for a given year.
    Returns a dictionary with the breakdown.
    """
    # Unpack parameters
    property_value = rental["property_value"]
    mortgage_balance = rental["mortgage_balance"]
    annual_interest_rate = rental["mortgage_interest_rate"]
    monthly_payment = rental["monthly_payment"]
    monthly_rent_base = rental["monthly_rent_base"]
    vacancy_rate = rental["vacancy_rate"]
    property_tax_rate = rental["property_tax_rate"]
    maintenance_rate = rental["maintenance_rate"]
    rental_income_tax_rate = rental["rental_income_tax_rate"]
    annual_depreciation = rental["annual_depreciation"]
    rent_inflation = rental["rent_inflation"]

    # 1) Calculate current monthly rent and gross annual rent
    current_monthly_rent = monthly_rent_base * ((1 + rent_inflation) ** year_index)
    gross_annual_rent = current_monthly_rent * 12 * (1 - vacancy_rate)

    # 2) Calculate property tax and maintenance
    annual_property_tax = property_value * property_tax_rate
    annual_maintenance = property_value * maintenance_rate

    # 3) Calculate total mortgage interest over 12 months
    monthly_interest_rate = annual_interest_rate / 12.0
    balance = mortgage_balance
    total_interest_paid = 0.0
    for _ in range(12):
        interest_for_month = balance * monthly_interest_rate
        principal_for_month = monthly_payment - interest_for_month
        if principal_for_month < 0:
            principal_for_month = 0
            interest_for_month = monthly_payment
        if principal_for_month > balance:
            principal_for_month = balance
        balance -= principal_for_month
        total_interest_paid += interest_for_month

    # 4) Compute Net Operating Income (NOI)
    net_operating_income = gross_annual_rent - annual_property_tax - annual_maintenance - total_interest_paid

    # 5) Calculate depreciation and taxable income
    taxable_income = net_operating_income - annual_depreciation
    if taxable_income < 0:
        taxable_income = 0.0

    # 6) Rental income tax and net cash flow
    rental_income_tax = taxable_income * rental_income_tax_rate
    net_rental_cash_flow = net_operating_income - rental_income_tax

    return {
        "Gross Annual Rent": gross_annual_rent,
        "Property Tax": annual_property_tax,
        "Maintenance": annual_maintenance,
        "Total Mortgage Interest": total_interest_paid,
        "Net Operating Income": net_operating_income,
        "Depreciation": annual_depreciation,
        "Taxable Income": taxable_income,
        "Rental Income Tax": rental_income_tax,
        "Net Rental Cash Flow": net_rental_cash_flow
    }

# -----------------------------------------------
# Stock Update (1 year)
# -----------------------------------------------
def simulate_stock(
    current_stock_value,
    stock_annual_contribution,
    expected_return,
    stock_volatility,
    cap_gains_tax_rate,
    dividend_yield,
    dividend_tax_rate,
    withdrawal=0.0,
    inflation_adjusted_withdrawal=True,
    current_year=0,
    inflation_rate=0.03
):
    """
    One-year update for taxable stock portfolio with optional inflation-adjusted withdrawals.
    """
    if inflation_adjusted_withdrawal and current_year > 0:
        withdrawal *= (1 + inflation_rate) ** current_year

    if withdrawal > current_stock_value:
        withdrawal = current_stock_value
    current_stock_value -= withdrawal
    current_stock_value += stock_annual_contribution

    dividends = current_stock_value * dividend_yield
    dividends_after_tax = dividends * (1 - dividend_tax_rate)
    current_stock_value += dividends_after_tax

    annual_return = np.random.normal(loc=expected_return, scale=stock_volatility)
    current_stock_value *= (1 + annual_return)

    if current_stock_value < 0:
        current_stock_value = 0

    return current_stock_value, withdrawal

# -----------------------------------------------
# Social Security Calculation
# -----------------------------------------------
def calculate_social_security(birth_year, retirement_age, annual_income):
    """
    Rough estimation of social security benefits based on retirement age and income.
    """
    full_retirement_age = 67 if birth_year >= 1960 else 66
    monthly_benefit_at_full = min(annual_income * 0.4, 3895)
    
    if retirement_age < full_retirement_age:
        reduction = 0.0625 * (full_retirement_age - retirement_age)
        monthly_benefit = monthly_benefit_at_full * (1 - reduction)
    else:
        increase = 0.08 * (retirement_age - full_retirement_age)
        monthly_benefit = monthly_benefit_at_full * (1 + increase)
    
    return monthly_benefit * 12

# -----------------------------------------------
# Healthcare Costs Estimation
# -----------------------------------------------
def estimate_healthcare_costs(current_age, retirement_age, life_expectancy):
    """
    Estimates healthcare costs in retirement including Medicare premiums and out-of-pocket.
    """
    base_annual_cost = 12000
    inflation_rate = 0.05
    costs = []
    for age in range(retirement_age, life_expectancy + 1):
        year = age - retirement_age
        if age < 65:
            annual_cost = base_annual_cost * 1.5 * (1 + inflation_rate) ** year
        else:
            annual_cost = base_annual_cost * (1 + inflation_rate) ** year
        costs.append(annual_cost)
    return costs

# -----------------------------------------------
# Required Minimum Distribution (RMD) Calculation
# -----------------------------------------------
def calculate_rmd(age, account_balance):
    """
    Calculate Required Minimum Distribution for tax-deferred accounts.
    """
    rmd_factors = {72: 25.6, 75: 22.9, 80: 18.7, 85: 14.8, 90: 11.4, 95: 8.6}
    closest_age = min(rmd_factors.keys(), key=lambda x: abs(x - age))
    return account_balance / rmd_factors[closest_age]

# -----------------------------------------------
# Social Security Strategy Analysis
# -----------------------------------------------
def analyze_social_security_strategy(birth_year, annual_income):
    """
    Analyzes different Social Security claiming strategies.
    """
    strategies = []
    for claim_age in range(62, 71):
        monthly_benefit = calculate_social_security(birth_year, claim_age, annual_income)
        total_by_80 = monthly_benefit * 12 * (80 - claim_age)
        total_by_85 = monthly_benefit * 12 * (85 - claim_age)
        total_by_90 = monthly_benefit * 12 * (90 - claim_age)
        strategies.append({
            'Claim Age': claim_age,
            'Monthly Benefit': monthly_benefit,
            'Total by Age 80': total_by_80,
            'Total by Age 85': total_by_85,
            'Total by Age 90': total_by_90
        })
    return pd.DataFrame(strategies)

# -----------------------------------------------
# Tax Bracket Calculation
# -----------------------------------------------
def calculate_tax_brackets(retirement_income):
    """
    Calculates estimated tax brackets in retirement.
    """
    brackets = [
        (0, 22000, 0.10),
        (22000, 89450, 0.12),
        (89450, 190750, 0.22),
        (190750, 364200, 0.24),
        (364200, 462500, 0.32),
        (462500, 693750, 0.35),
        (693750, float('inf'), 0.37)
    ]
    
    total_tax = 0
    remaining_income = retirement_income
    breakdown = []
    
    for min_income, max_income, rate in brackets:
        if remaining_income <= 0:
            break
        taxable_in_bracket = min(remaining_income, max_income - min_income)
        tax_in_bracket = taxable_in_bracket * rate
        breakdown.append({
            'Bracket': f"{rate*100:.1f}%",
            'Income in Bracket': taxable_in_bracket,
            'Tax in Bracket': tax_in_bracket
        })
        total_tax += tax_in_bracket
        remaining_income -= taxable_in_bracket
    
    return pd.DataFrame(breakdown), total_tax

# -----------------------------------------------
# Asset Allocation Analysis (Monte Carlo)
# -----------------------------------------------
def analyze_asset_allocations(initial_balance, years, n_sims=1000):
    """
    Analyzes different stock/bond allocations using Monte Carlo simulation.
    """
    allocations = [
        (100, 0),
        (80, 20),
        (60, 40),
        (40, 60),
    ]
    
    results = []
    
    for stocks, bonds in allocations:
        stock_pct = stocks / 100
        bond_pct = bonds / 100
        
        stock_return = 0.07
        stock_vol = 0.15
        bond_return = 0.03
        bond_vol = 0.05
        
        portfolio_return = (stock_return * stock_pct) + (bond_return * bond_pct)
        portfolio_vol = np.sqrt((stock_vol**2 * stock_pct**2) + (bond_vol**2 * bond_pct**2))
        
        final_values = []
        for _ in range(n_sims):
            value = initial_balance
            for _ in range(years):
                r = np.random.normal(portfolio_return, portfolio_vol)
                value *= (1 + r)
            final_values.append(value)
            
        results.append({
            'Allocation': f"{stocks}/{bonds}",
            'Median': np.median(final_values),
            'Worst 5%': np.percentile(final_values, 5),
            'Best 5%': np.percentile(final_values, 95)
        })
    
    return pd.DataFrame(results)

# -----------------------------------------------
# Future Expenses Validation
# -----------------------------------------------
def validate_future_expenses(stock_initial, stock_annual_contribution, luxury_expenses, rentals_data, current_year, years_to_retirement, total_years, assumed_growth=0.05):
    """
    Validates that future expenses (luxury expenses and rental downpayments) do not exceed the funds 
    available when accounting for annual contributions and a simple growth rate.
    Purchase years are relative to the simulation timeline where:
    Year 1 = first year of simulation.
    """
    expenses_by_year = {}
    
    # Add luxury expenses
    for expense in luxury_expenses:
        if expense['purchase_year'] > 0:  # Only include future purchases
            simulation_year = expense['purchase_year']  # Year is relative to simulation
            expenses_by_year.setdefault(simulation_year, 0)
            expenses_by_year[simulation_year] += expense['amount']
    
    # Add rental purchases (downpayments)
    for rental in rentals_data:
        if rental['purchase_year'] > 0:  # Only include future purchases
            simulation_year = rental['purchase_year']
            expenses_by_year.setdefault(simulation_year, 0)
            expenses_by_year[simulation_year] += rental['downpayment']
    
    balance = stock_initial
    # Simulate year by year until the end of the simulation.
    for year in range(1, total_years + 1):
        # Grow the balance (simplified with an assumed growth rate)
        balance *= (1 + assumed_growth)
        
        # Add contributions in accumulation phase.
        if year <= years_to_retirement:
            balance += stock_annual_contribution
        
        # Deduct expenses for this simulation year, if any.
        if year in expenses_by_year:
            if expenses_by_year[year] > balance:
                return False, (
                    f"Insufficient funds in simulation year {year} (Calendar Year: {current_year + year - 1}) "
                    f"for expenses. Need ${expenses_by_year[year]:,.2f} but only ${balance:,.2f} available."
                )
            balance -= expenses_by_year[year]
    
    return True, ""

# -----------------------------------------------
# Main Simulation Function
# -----------------------------------------------
def run_simulation(
    birth_year,
    retirement_age,
    life_expectancy,
    annual_income,
    annual_budget,
    tax_deferred_balance,
    roth_balance,
    n_sims,
    stock_initial,
    stock_annual_contribution,
    stock_expected_return,
    stock_volatility,
    cap_gains_tax_rate,
    dividend_yield,
    dividend_tax_rate,
    withdrawal,
    rentals_data,
    primary_residence_value,
    primary_mortgage_balance,
    primary_mortgage_interest_rate,
    primary_mortgage_years_left,
    primary_appreciation_mean,
    primary_appreciation_std,
    luxury_expenses,
    inflation_rate=0.03,
    healthcare_inflation=0.05,
    random_seed=None
):
    """
    Enhanced simulation including Social Security, healthcare, and simulation of
    tax-advantaged accounts.
    """
    # Set random seed if provided for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    current_year = date.today().year
    current_age = current_year - birth_year
    years_to_retirement = retirement_age - current_age
    years_in_retirement = life_expectancy - retirement_age
    total_years = years_to_retirement + years_in_retirement
    
    all_sims = []
    failure_count = 0
    num_rentals = len(rentals_data)
    
    annual_ss_benefit = calculate_social_security(birth_year, retirement_age, annual_income)
    healthcare_costs = estimate_healthcare_costs(current_age, retirement_age, life_expectancy)
    
    phase_labels = ['Accumulation' if year < years_to_retirement else 'Retirement' for year in range(total_years)]
    
    # For rentals with purchase_year == 0, assume they are already owned.
    my_rentals = [rd.copy() for rd in rentals_data if rd["purchase_year"] == 0]
    
    for sim_i in range(n_sims):
        current_stock_value = stock_initial
        total_contributions = 0.0
        current_tax_deferred = tax_deferred_balance
        current_roth = roth_balance

        # Create local copies for primary residence variables
        current_primary_mortgage_balance = primary_mortgage_balance
        current_primary_value = primary_residence_value

        primary_monthly_payment = standard_mortgage_payment(
            principal=primary_mortgage_balance,
            annual_interest_rate=primary_mortgage_interest_rate,
            mortgage_years=primary_mortgage_years_left
        )
        
        # Start with already-owned rentals
        sim_rentals = my_rentals.copy()
        
        year_list = []
        stock_vals = []
        tax_deferred_vals = []
        roth_vals = []
        total_net_worth_list = []
        primary_residence_equity_list = []
        stock_contributions_list = []
        stock_appreciation_list = []
        
        rentals_equity_lists = [[] for _ in range(len(rentals_data))]
        rentals_mortgage_lists = [[] for _ in range(len(rentals_data))]
        rentals_cashflow_lists = [[] for _ in range(len(rentals_data))]
        
        rental_index_map = {rd["description"]: i for i, rd in enumerate(rentals_data)}
        
        actual_withdrawals = []
        portfolio_failed = False

        for year in range(total_years):
            current_age = (current_year - birth_year) + year
            year_label = year + 1
            is_retirement = (year >= years_to_retirement)
            
            # Check for new rental purchases this year (purchase_year > 0)
            for rental in rentals_data:
                if rental["purchase_year"] == year_label:
                    downpayment = rental["downpayment"]
                    # Deduct the downpayment from the stock portfolio
                    current_stock_value -= downpayment
                    new_rental = rental.copy()
                    new_rental["mortgage_balance"] = rental["cost_basis"] - downpayment
                    new_rental["property_value"] = rental["cost_basis"]
                    sim_rentals.append(new_rental)
            
            # Handle luxury expenses
            for expense in luxury_expenses:
                if expense["purchase_year"] == year_label:
                    expense_amount = expense["amount"]
                    current_stock_value -= expense_amount
            
            for r_i in range(len(rentals_data)):
                rentals_equity_lists[r_i].append(0)
                rentals_mortgage_lists[r_i].append(0)
                rentals_cashflow_lists[r_i].append(0)
            
            for rental in sim_rentals:
                if rental["property_value"] <= 0:
                    continue
                r_i = rental_index_map[rental["description"]]
                updated_r, net_cf = update_rental_property_monthly(rental, year_index=year)
                if net_cf > 0:
                    current_stock_value += net_cf
                sale_year = updated_r["sale_year"]
                if sale_year == year_label:
                    gains = max(updated_r["property_value"] - updated_r["cost_basis"], 0)
                    cap_gains_tax = gains * updated_r["rental_cap_gains_tax"]
                    net_sale_proceeds = max((updated_r["property_value"] - updated_r["mortgage_balance"]) - cap_gains_tax, 0)
                    current_stock_value += net_sale_proceeds
                    updated_r["property_value"] = 0
                    updated_r["mortgage_balance"] = 0
                    updated_r["cost_basis"] = 0
                eq = max(updated_r["property_value"] - updated_r["mortgage_balance"], 0)
                rentals_equity_lists[r_i][-1] = eq
                rentals_mortgage_lists[r_i][-1] = updated_r["mortgage_balance"]
                rentals_cashflow_lists[r_i][-1] = net_cf
            
            if is_retirement:
                inflation_adjusted_budget = annual_budget * ((1 + inflation_rate) ** (year - years_to_retirement))
                ss_income = annual_ss_benefit * ((1 + inflation_rate) ** (year - years_to_retirement))
                current_stock_value += ss_income
                if (year - years_to_retirement) < len(healthcare_costs):
                    healthcare_cost = healthcare_costs[year - years_to_retirement]
                    current_stock_value -= healthcare_cost
                if current_age >= 72 and current_tax_deferred > 0:
                    rmd = calculate_rmd(current_age, current_tax_deferred)
                    current_tax_deferred -= rmd
                    current_stock_value += rmd * 0.8
            stock_contribution = stock_annual_contribution if not is_retirement else 0
            stock_withdrawal = inflation_adjusted_budget if is_retirement else 0
            current_stock_value, actual_withdrawal = simulate_stock(
                current_stock_value,
                stock_contribution,
                stock_expected_return,
                stock_volatility,
                cap_gains_tax_rate,
                dividend_yield,
                dividend_tax_rate,
                withdrawal=stock_withdrawal,
                inflation_adjusted_withdrawal=False,
                current_year=year,
                inflation_rate=inflation_rate
            )
            if is_retirement and actual_withdrawal < stock_withdrawal:
                portfolio_failed = True
            actual_withdrawals.append(actual_withdrawal)
            total_contributions += stock_contribution
            total_appreciation = current_stock_value - total_contributions
            
            r_return = np.random.normal(stock_expected_return, stock_volatility)
            current_tax_deferred *= (1 + r_return)
            current_roth *= (1 + r_return)
            
            # Update primary residence (compounding value and amortizing mortgage)
            annual_appreciation = np.random.normal(loc=primary_appreciation_mean, scale=primary_appreciation_std)
            current_primary_value *= (1 + annual_appreciation)
            monthly_interest_rate = primary_mortgage_interest_rate / 12.0
            for _ in range(12):
                interest_for_month = current_primary_mortgage_balance * monthly_interest_rate
                principal_for_month = primary_monthly_payment - interest_for_month
                if principal_for_month < 0:
                    principal_for_month = 0
                    interest_for_month = primary_monthly_payment
                if principal_for_month > current_primary_mortgage_balance:
                    principal_for_month = current_primary_mortgage_balance
                current_primary_mortgage_balance -= principal_for_month
            
            primary_equity = max(current_primary_value - current_primary_mortgage_balance, 0)
            rental_equity = sum(max(r["property_value"] - r["mortgage_balance"], 0) for r in sim_rentals)
            total_net_worth = current_stock_value + current_tax_deferred + current_roth + rental_equity + primary_equity
            
            year_list.append(year_label)
            stock_vals.append(current_stock_value)
            tax_deferred_vals.append(current_tax_deferred)
            roth_vals.append(current_roth)
            total_net_worth_list.append(total_net_worth)
            primary_residence_equity_list.append(primary_equity)
            stock_contributions_list.append(total_contributions)
            stock_appreciation_list.append(total_appreciation)
        
        sim_data = {
            "Year": year_list,
            "Taxable Stock Value": stock_vals,
            "Tax-Deferred Balance": tax_deferred_vals,
            "Roth Balance": roth_vals,
            "Total Net Worth": total_net_worth_list,
            "Primary Residence Equity": primary_residence_equity_list,
            "Stock Contributions": stock_contributions_list,
            "Stock Appreciation": stock_appreciation_list
        }
        for r_i in range(num_rentals):
            sim_data[f"Rental{r_i+1} Equity"] = rentals_equity_lists[r_i]
            sim_data[f"Rental{r_i+1} Mortgage"] = rentals_mortgage_lists[r_i]
            sim_data[f"Rental{r_i+1} CashFlow"] = rentals_cashflow_lists[r_i]
        
        sim_df = pd.DataFrame(sim_data)
        all_sims.append(sim_df)
        if portfolio_failed:
            failure_count += 1
    
    failure_rate = failure_count / n_sims * 100

    combined_df = None
    for i, df_i in enumerate(all_sims):
        rename_map = {col: f"{col} {i}" for col in df_i.columns if col != "Year"}
        temp = df_i.rename(columns=rename_map)
        combined_df = temp if combined_df is None else pd.merge(combined_df, temp, on="Year", how="inner")
    
    summary_df = pd.DataFrame({"Year": combined_df["Year"]})
    col_groups = {}
    for col in all_sims[0].columns:
        if col == "Year":
            continue
        matching = [c for c in combined_df.columns if c.startswith(col)]
        col_groups[col] = matching
    
    for group_name, cols in col_groups.items():
        summary_df[f"{group_name} Mean"]   = combined_df[cols].mean(axis=1)
        summary_df[f"{group_name} Median"] = combined_df[cols].median(axis=1)
        summary_df[f"{group_name} 10th"]   = combined_df[cols].quantile(0.1, axis=1)
        summary_df[f"{group_name} 90th"]   = combined_df[cols].quantile(0.9, axis=1)
    
    summary_df['Phase'] = phase_labels

    return {
        "all_sims": all_sims,
        "combined": combined_df,
        "summary": summary_df,
        "failure_rate": failure_rate
    }

# -----------------------------------------------
# Figure Preparation for Report
# -----------------------------------------------
def prepare_figure_for_report(fig, title):
    """
    Prepare a Plotly figure for the HTML report.
    """
    colors = px.colors.qualitative.Set3
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title={'font': {'size': 24}},
        showlegend=True
    )
    if isinstance(fig, go.Figure):
        if 'pie' in [trace.type for trace in fig.data]:
            fig.update_traces(
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='inside'
            )
    else:
        if hasattr(fig, 'data'):
            for i, trace in enumerate(fig.data):
                trace.update(marker_color=colors[i % len(colors)])
    return fig

# -----------------------------------------------
# HTML Report Generation
# -----------------------------------------------
def generate_html_report(results, params, figures, monthly_expenses, ss_analysis=None, tax_analysis=None, allocation_analysis=None):
    """
    Generate an HTML report from simulation results and figures.
    """
    html_template = """
    <html>
    <head>
        <title>Retirement Portfolio Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #eee; border-radius: 5px; }
            .metric { margin: 10px 0; }
            .plot { margin: 20px 0; text-align: center; }
            .parameters { margin: 20px 0; }
            .alert { padding: 15px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; }
            .info { padding: 15px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot img {
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border: 1px solid #eee;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .plot p {
                font-size: 1.2em;
                font-weight: bold;
                color: #333;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Retirement Portfolio Analysis Report</h1>
            <p>Generated on {{ params.generation_date }}</p>
        </div>
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="alert">Portfolio Failure Rate: {{ params.failure_rate }}%</div>
            <p>A failure rate above 5% suggests the plan may be too risky.</p>
            <h3>Key Metrics</h3>
            <div class="metric">Initial Portfolio: ${{ '{:,.2f}'.format(params.initial_portfolio) }}</div>
            <div class="metric">Annual Contribution: ${{ '{:,.2f}'.format(params.annual_contribution) }}</div>
            <div class="metric">Initial Withdrawal: ${{ '{:,.2f}'.format(params.initial_withdrawal) }}</div>
            <div class="metric">Final Year Withdrawal: ${{ '{:,.2f}'.format(params.final_withdrawal) }}</div>
        </div>
        <div class="section">
            <h2>Timeline</h2>
            <div class="metric">Current Age: {{ params.current_age }}</div>
            <div class="metric">Retirement Age: {{ params.retirement_age }}</div>
            <div class="metric">Years to Retirement: {{ params.years_to_retirement }}</div>
            <div class="metric">Years in Retirement: {{ params.years_in_retirement }}</div>
        </div>
        <div class="section">
            <h2>Monthly Budget Breakdown</h2>
            <table>
                <tr><th>Category</th><th>Amount</th></tr>
                {% for category, amount in params.monthly_expenses.items() %}
                <tr><td>{{ category }}</td><td>${{ '{:,.2f}'.format(amount) }}</td></tr>
                {% endfor %}
                <tr><th>Total Monthly</th><th>${{ '{:,.2f}'.format(params.total_monthly) }}</th></tr>
                <tr><th>Total Annual</th><th>${{ '{:,.2f}'.format(params.total_monthly * 12) }}</th></tr>
            </table>
        </div>
        <div class="section">
            <h2>Social Security Analysis</h2>
            <div class="info">Estimated Annual Benefit at Age {{ params.retirement_age }}: ${{ '{:,.2f}'.format(params.ss_benefit) }}</div>
            {% if ss_analysis %}
            <h3>Claiming Strategy Analysis</h3>
            <table>
                <tr>
                    <th>Claim Age</th>
                    <th>Monthly Benefit</th>
                    <th>Total by Age 80</th>
                    <th>Total by Age 85</th>
                    <th>Total by Age 90</th>
                </tr>
                {% for row in ss_analysis %}
                <tr>
                    <td>{{ row['Claim Age'] }}</td>
                    <td>${{ '{:,.2f}'.format(row['Monthly Benefit']) }}</td>
                    <td>${{ '{:,.0f}'.format(row['Total by Age 80']) }}</td>
                    <td>${{ '{:,.0f}'.format(row['Total by Age 85']) }}</td>
                    <td>${{ '{:,.0f}'.format(row['Total by Age 90']) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>
        {% if tax_analysis %}
        <div class="section">
            <h2>Tax Analysis</h2>
            <div class="metric">Total Retirement Income: ${{ '{:,.2f}'.format(tax_analysis.income) }}</div>
            <div class="metric">Annual Tax: ${{ '{:,.2f}'.format(tax_analysis.tax) }}</div>
            <div class="metric">Effective Tax Rate: {{ '{:.1f}'.format(tax_analysis.effective_rate) }}%</div>
            <h3>Tax Bracket Breakdown</h3>
            <table>
                <tr><th>Bracket</th><th>Income in Bracket</th><th>Tax in Bracket</th></tr>
                {% for row in tax_analysis.breakdown %}
                <tr>
                    <td>{{ row['Bracket'] }}</td>
                    <td>${{ '{:,.2f}'.format(row['Income in Bracket']) }}</td>
                    <td>${{ '{:,.2f}'.format(row['Tax in Bracket']) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        {% if allocation_analysis %}
        <div class="section">
            <h2>Asset Allocation Analysis</h2>
            <table>
                <tr>
                    <th>Allocation</th>
                    <th>Median</th>
                    <th>Worst 5%</th>
                    <th>Best 5%</th>
                </tr>
                {% for row in allocation_analysis %}
                <tr>
                    <td>{{ row['Allocation'] }}</td>
                    <td>${{ '{:,.0f}'.format(row['Median']) }}</td>
                    <td>${{ '{:,.0f}'.format(row['Worst 5%']) }}</td>
                    <td>${{ '{:,.0f}'.format(row['Best 5%']) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        <div class="section">
            <h2>Key Visualizations</h2>
            {% for fig in figures %}
            <div class="plot">
                <img src="data:image/png;base64,{{ fig.image }}" alt="{{ fig.title }}">
                <p>{{ fig.title }}</p>
            </div>
            {% endfor %}
        </div>
        <div class="section">
            <h2>Simulation Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {% for key, value in params.assumptions.items() %}
                <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
                {% endfor %}
            </table>
        </div>
        <div class="section">
            <h2>Key Assumptions and Notes</h2>
            <ul>
                <li>Stock returns include both capital appreciation and dividends</li>
                <li>Property values are adjusted for mortgage paydown and appreciation</li>
                <li>All withdrawals are adjusted for inflation</li>
                <li>Rental income (positive cash flow) is reinvested in stocks</li>
                <li>Tax-deferred and Roth accounts grow at market rates; RMDs are applied to the tax-deferred account in retirement</li>
                <li>The simulation accounts for taxes on dividends, rental income, and capital gains</li>
            </ul>
        </div>
    </body>
    </html>
    """
    figure_data = []
    for fig_dict in figures:
        buf = BytesIO()
        fig_dict['figure'].write_image(buf, format='png')
        buf.seek(0)
        fig_dict['image'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        figure_data.append({
            'image': fig_dict['image'],
            'title': fig_dict['title']
        })
    template = Template(html_template)
    html = template.render(params=params, figures=figure_data)
    return html

# -----------------------------------------------
# STREAMLIT APP
# -----------------------------------------------
def main():
    st.set_page_config(page_title="Retirement Portfolio Simulator", layout="wide")
    st.title("Retirement Portfolio Simulator")
    st.subheader("Comprehensive Analysis of Stocks, Real Estate, and Retirement Planning")
    
    st.markdown("""
    This simulator helps you model your retirement portfolio including:
    - 📈 Stock investments with dividends and volatility
    - 🏠 Primary residence with mortgage amortization
    - 🏢 Multiple rental properties with cash flow analysis
    - 💰 Retirement withdrawals with inflation adjustment
    - 🏦 Simulation of tax-deferred and Roth account growth
    """)

    # ---------- SIDEBAR ----------
    st.sidebar.title("Your Information")
    
    st.sidebar.header("1️⃣ Personal Details")
    birth_year = st.sidebar.number_input("Birth Year", 1940, 2000, 1970, help="Used to calculate Social Security benefits and RMDs")
    retirement_age = st.sidebar.number_input("Planned Retirement Age", 55, 75, 65, help="Age at which you plan to retire")
    life_expectancy = st.sidebar.number_input("Life Expectancy", retirement_age, 100, 90, help="Plan through this age for safety")
    annual_income = st.sidebar.number_input("Current Annual Income", 0, 500_000, 80_000, help="Used for Social Security calculation")
    
    current_year = date.today().year
    current_age = current_year - birth_year
    years_to_retirement = retirement_age - current_age
    years_in_retirement = life_expectancy - retirement_age
    total_years = years_to_retirement + years_in_retirement
    
    st.sidebar.info(f"""
    **Your Timeline:**
    🎂 Current Age: {current_age}
    ⏳ Years until retirement: {years_to_retirement}
    🌅 Years in retirement: {years_in_retirement}
    """)
    
    st.sidebar.header("2️⃣ Monthly Expenses")
    with st.sidebar.expander("Enter Monthly Budget", expanded=False):
        monthly_expenses = {
            "Housing (non-mortgage)": st.number_input("Monthly Housing Expenses", 0, 10000, 500),
            "Utilities": st.number_input("Monthly Utilities", 0, 5000, 200),
            "Food": st.number_input("Monthly Food Budget", 0, 5000, 600),
            "Transportation": st.number_input("Monthly Transportation", 0, 5000, 400),
            "Healthcare": st.number_input("Monthly Healthcare", 0, 5000, 400),
            "Entertainment": st.number_input("Monthly Entertainment", 0, 5000, 300),
            "Other": st.number_input("Other Monthly Expenses", 0, 10000, 500)
        }
        total_monthly = sum(monthly_expenses.values())
        st.info(f"Total Monthly Expenses: ${total_monthly:,.2f}")
        st.info(f"Annual Expenses: ${total_monthly*12:,.2f}")
    
    st.sidebar.header("3️⃣ Current Assets")
    with st.sidebar.expander("Retirement Accounts", expanded=True):
        tax_deferred_balance = st.number_input("Tax-Deferred Balance (401k, IRA)", 0, 10_000_000, 100_000, help="Current balance in traditional retirement accounts")
        roth_balance = st.number_input("Roth Balance", 0, 10_000_000, 50_000, help="Current balance in Roth accounts")
        stock_initial = st.number_input("Taxable Investment Balance", 0, 10_000_000, 50_000, step=1_000, help="Current balance in taxable investment accounts")
    
    with st.sidebar.expander("Primary Residence", expanded=True):
        primary_residence_value = st.number_input("Home Value", 0, 10_000_000, 300_000, help="Current market value of your primary residence")
        primary_mortgage_balance = st.number_input("Mortgage Balance", 0, 10_000_000, 150_000, help="Remaining balance on your mortgage")
        primary_mortgage_interest_rate = st.slider("Mortgage Rate", 0.0, 0.2, 0.04, 0.005, help="Annual interest rate on your mortgage")
        primary_mortgage_years_left = st.number_input("Years Left on Mortgage", 0, 40, 25, help="Remaining years on your mortgage term")
        primary_appreciation_mean = st.slider("Home Appreciation Rate", 0.0, 0.2, 0.03, 0.01, help="Expected annual appreciation rate for your home")
        primary_appreciation_std = st.slider("Home Appreciation Volatility", 0.0, 0.5, 0.1, 0.01, help="Standard deviation of annual home appreciation")
    
    with st.sidebar.expander("Rental Properties", expanded=True):
        num_rentals = st.number_input("Number of Rental Properties", 0, 5, 0, help="How many rental properties do you own?")
        rentals_data = []
        for rental_idx in range(num_rentals):
            st.markdown(f"**Rental Property #{rental_idx+1}**")
            rental_data = {
                "description": st.text_input(f"Property Description {rental_idx+1}", value=f"Rental {rental_idx+1}",
                                           help="Description or address of the property", key=f"rental_desc_{rental_idx}"),
                "purchase_year": st.number_input(f"Purchase Year {rental_idx+1}", 0, total_years, 0,
                                               help="Year to purchase (0 = already owned)", key=f"rental_purchase_year_{rental_idx}"),
                "property_value": st.number_input(f"Property Value (Rental {rental_idx+1})", 0, 10_000_000, 200_000, step=1_000,
                                                help="Current market value of the rental property", key=f"rental_value_{rental_idx}"),
                "downpayment": st.number_input(f"Downpayment (Rental {rental_idx+1})", 0, 2_000_000, 40_000, step=1_000,
                                             help="Downpayment amount (will be deducted from stocks if purchased in future)", key=f"rental_downpayment_{rental_idx}"),
                "cost_basis": st.number_input(f"Purchase Price (Rental {rental_idx+1})", 0, 2_000_000, 180_000, step=1_000,
                                            help="Total purchase price of the property", key=f"rental_basis_{rental_idx}"),
                "mortgage_balance": st.number_input(f"Mortgage Balance (Rental {rental_idx+1})", 0, 10_000_000, 120_000, step=1_000,
                                                  help="Remaining balance on the mortgage", key=f"rental_mortgage_{rental_idx}"),
                "mortgage_interest_rate": st.slider(f"Mortgage Rate (Rental {rental_idx+1})", 0.0, 0.2, 0.04, 0.005,
                                                  help="Annual interest rate on the mortgage", key=f"rental_rate_{rental_idx}"),
                "monthly_rent_base": st.number_input(f"Monthly Rent (Rental {rental_idx+1})", 0, 50_000, 1500, step=100,
                                                   help="Current monthly rent charged", key=f"rental_rent_{rental_idx}"),
                "sale_year": st.number_input(f"Planned Sale Year (Rental {rental_idx+1})", 0, 100, 0, step=1,
                                           help="Year you plan to sell (0 = no plan)", key=f"rental_sale_{rental_idx}"),
                "rental_cap_gains_tax": st.slider(f"Capital Gains Tax Rate (Rental {rental_idx+1})", 0.0, 0.5, 0.15, 0.01,
                                                help="Expected tax rate on sale profits", key=f"rental_capgains_{rental_idx}"),
                "property_tax_rate": st.slider(f"Property Tax Rate (Rental {rental_idx+1})", 0.0, 0.05, 0.01, 0.001,
                                             help="Annual property tax as % of value", key=f"rental_proptax_{rental_idx}"),
                "maintenance_rate": st.slider(f"Maintenance Rate (Rental {rental_idx+1})", 0.0, 0.05, 0.01, 0.001,
                                            help="Annual maintenance as % of value", key=f"rental_maint_{rental_idx}"),
                "vacancy_rate": st.slider(f"Vacancy Rate (Rental {rental_idx+1})", 0.0, 1.0, 0.05, 0.01,
                                        help="Expected vacancy rate", key=f"rental_vacancy_{rental_idx}"),
                "rental_income_tax_rate": st.slider(f"Rental Income Tax Rate (Rental {rental_idx+1})", 0.0, 0.5, 0.2, 0.01,
                                                  help="Tax rate on rental income", key=f"rental_tax_{rental_idx}"),
                "rent_inflation": st.slider(f"Rent Inflation (Rental {rental_idx+1})", 0.0, 0.1, 0.02, 0.01,
                                          help="Expected annual increase in rent", key=f"rental_inflation_{rental_idx}"),
                "app_mean": st.slider(f"Appreciation Rate (Rental {rental_idx+1})", 0.0, 0.2, 0.03, 0.01,
                                    help="Expected annual property appreciation", key=f"rental_appmean_{rental_idx}"),
                "app_std": st.slider(f"Appreciation Volatility (Rental {rental_idx+1})", 0.0, 0.5, 0.1, 0.01,
                                   help="Standard deviation of appreciation", key=f"rental_appstd_{rental_idx}")
            }
            monthly_payment = standard_mortgage_payment(
                principal=rental_data["mortgage_balance"],
                annual_interest_rate=rental_data["mortgage_interest_rate"],
                mortgage_years=30
            )
            rental_data["monthly_payment"] = monthly_payment
            rental_data["annual_depreciation"] = rental_data["cost_basis"] / 27.5
            rentals_data.append(rental_data)
    
    st.sidebar.header("4️⃣ Investment Strategy")
    with st.sidebar.expander("Contributions & Withdrawals", expanded=True):
        stock_annual_contribution = st.number_input("Annual Investment Contribution", 0, 1_000_000, 10_000, step=1_000,
                                                    help="Amount you plan to invest annually during working years")
        withdrawal_adjustment = st.slider("Retirement Budget Adjustment", 0.5, 1.5, 1.0, 0.05,
                                          help="Adjust retirement spending relative to current budget (1.0 = same as current)")
        monthly_budget = sum(monthly_expenses.values())
        annual_withdrawal_stocks = monthly_budget * 12 * withdrawal_adjustment
        st.info(f"""
        **Planned Annual Withdrawal: ${annual_withdrawal_stocks:,.2f}**
        - Based on current monthly expenses: ${monthly_budget:,.2f}
        - Adjusted by factor: {withdrawal_adjustment:.2f}x
        - Withdrawal Rate: {(annual_withdrawal_stocks / stock_initial * 100):.1f}% of current portfolio
        """)
    
    with st.sidebar.expander("Market Assumptions", expanded=False):
        stock_expected_return = st.slider("Expected Return (mean)", 0.0, 0.2, 0.07, 0.01,
                                          help="Average annual investment return before inflation")
        stock_volatility = st.slider("Volatility (std dev)", 0.0, 0.5, 0.15, 0.01,
                                     help="Standard deviation of annual returns")
        dividend_yield = st.slider("Dividend Yield", 0.0, 0.1, 0.02, 0.01,
                                   help="Expected annual dividend yield")
    
    st.sidebar.header("5️⃣ Tax & Economic Factors")
    with st.sidebar.expander("Tax & Inflation Settings", expanded=False):
        cap_gains_tax_rate = st.slider("Capital Gains Tax Rate", 0.0, 0.5, 0.15, 0.01,
                                       help="Long-term capital gains tax rate")
        dividend_tax_rate = st.slider("Dividend Tax Rate", 0.0, 0.5, 0.15, 0.01,
                                      help="Tax rate on dividend income")
        inflation_rate = st.slider("Inflation Rate", 0.01, 0.10, 0.03, 0.001,
                                   help="Expected annual inflation rate")
    
    st.sidebar.header("6️⃣ Optional Planning")
    with st.sidebar.expander("Future Expenses", expanded=False):
        num_luxury_expenses = st.number_input("Number of One-time Expenses", 0, 10, 0, help="How many large future expenses to plan for")
        luxury_expenses = []
        for expense_idx in range(num_luxury_expenses):
            st.markdown(f"**One-time Expense #{expense_idx+1}**")
            expense = {
                "description": st.text_input(f"Description {expense_idx+1}", value=f"Expense {expense_idx+1}",
                                           help="Description of the expense (e.g., Vacation Home)", key=f"expense_desc_{expense_idx}"),
                "amount": st.number_input(f"Expense Amount {expense_idx+1}", 0, 5_000_000, 0, step=1_000,
                                        help="Amount needed for this expense", key=f"expense_amount_{expense_idx}"),
                "purchase_year": st.number_input(f"Purchase Year {expense_idx+1}", 1, total_years, 1,
                                               help="Year when expense will be paid", key=f"expense_purchase_year_{expense_idx}")
            }
            if expense["amount"] > 0:
                luxury_expenses.append(expense)
    
    st.sidebar.header("7️⃣ Simulation Settings")
    n_sims = st.sidebar.number_input("Number of Simulations", 100, 2000, 200, help="More simulations = more accurate results but slower")
    
    # Optional random seed for reproducibility
    seed_input = st.sidebar.text_input("Random Seed (Optional)", value="", help="Enter an integer to seed the random number generator for reproducible results.")
    if seed_input:
         random_seed = int(seed_input)
    else:
         random_seed = None
    
    st.sidebar.warning("""
    ⚠️ **Important Notes:**
    - Past performance doesn't guarantee future returns
    - The default 7% return is before inflation
    - Consider using conservative estimates
    - Early retirement years have outsized impact
    """)
    
    st.sidebar.header("8️⃣ Analysis Options")
    show_ss_analysis = st.sidebar.checkbox("Show Social Security Analysis", value=False)
    show_tax_analysis = st.sidebar.checkbox("Show Tax Analysis", value=False)
    show_allocation_analysis = st.sidebar.checkbox("Show Asset Allocation Analysis", value=False)
    
    if show_tax_analysis:
        st.sidebar.subheader("Additional Tax Analysis Inputs")
        pension_income = st.sidebar.number_input("Expected Annual Pension", 0, 200_000, 0, help="Expected annual pension income in retirement")
        rental_income = st.sidebar.number_input("Expected Annual Rental Income", 0, 200_000, 0, help="Expected annual rental income in retirement")
    
    if 'report_html' not in st.session_state:
        st.session_state.report_html = None
    
    col1, col2 = st.columns([1, 1])
    with col1:
        run_simulation_button = st.button("Run Simulation")
    
    if run_simulation_button:
        # Validate expenses before running simulation
        is_valid, error_message = validate_future_expenses(
            stock_initial, 
            stock_annual_contribution,
            luxury_expenses, 
            rentals_data,
            current_year,
            years_to_retirement,
            total_years
        )
        if not is_valid:
            st.error(error_message)
            st.stop()
            
        monthly_budget = sum(monthly_expenses.values())
        annual_budget = monthly_budget * 12

        with st.spinner("Simulating..."):
            results = run_simulation(
                birth_year,
                retirement_age,
                life_expectancy,
                annual_income,
                annual_budget,
                tax_deferred_balance,
                roth_balance,
                n_sims,
                stock_initial,
                stock_annual_contribution,
                stock_expected_return,
                stock_volatility,
                cap_gains_tax_rate,
                dividend_yield,
                dividend_tax_rate,
                annual_withdrawal_stocks,
                rentals_data,
                primary_residence_value,
                primary_mortgage_balance,
                primary_mortgage_interest_rate,
                primary_mortgage_years_left,
                primary_appreciation_mean,
                primary_appreciation_std,
                luxury_expenses,
                inflation_rate,
                healthcare_inflation=0.05,
                random_seed=random_seed
            )
        st.success("Simulation Complete!")
        failure_rate = results["failure_rate"]
        st.error(f"Portfolio Failure Rate: {failure_rate:.1f}%")
        st.markdown("""
        **Portfolio Failure** means the simulation couldn't maintain the desired 
        withdrawal rate adjusted for inflation. A failure rate above 5% suggests 
        the plan may be too risky.
        """)
        
        st.subheader("Total Net Worth Components Over Time")
        summary_df = results["summary"]
        fig_components = px.area(summary_df, x="Year", 
            y=["Taxable Stock Value Median", "Tax-Deferred Balance Median", "Roth Balance Median",
               "Primary Residence Equity Median"] + 
              [f"Rental{i+1} Equity Median" for i in range(num_rentals)],
            title="Net Worth Components Over Time",
            labels={"value": "Value ($)", "variable": "Component"}
        )
        fig_components.add_vline(
            x=years_to_retirement, 
            line_dash="dash",
            line_color="red",
            annotation_text="Retirement",
            annotation_position="top"
        )
        fig_components.add_annotation(
            x=years_to_retirement/2,
            y=fig_components.data[0].y.max(),
            text="Accumulation Phase",
            showarrow=False,
            yshift=10
        )
        fig_components.add_annotation(
            x=years_to_retirement + years_in_retirement/2,
            y=fig_components.data[0].y.max(),
            text="Retirement Phase",
            showarrow=False,
            yshift=10
        )
        st.plotly_chart(fig_components, use_container_width=True)
        
        st.subheader("Accumulation Phase Summary")
        accum_end = summary_df[summary_df['Year'] == years_to_retirement].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Contributions", f"${accum_end['Stock Contributions Median']:,.0f}",
                      help="Total amount contributed during working years")
        with col2:
            st.metric("Investment Growth", f"${accum_end['Stock Appreciation Median']:,.0f}",
                      help="Total investment returns during accumulation")
        with col3:
            st.metric("Net Worth at Retirement", f"${accum_end['Total Net Worth Median']:,.0f}",
                      help="Total net worth when entering retirement")
        
        st.subheader("Stock Portfolio Value Over Time")
        fig_stock = px.line(
            summary_df, 
            x="Year", 
            y="Taxable Stock Value Median", 
            title="Taxable Stock Value (Median, 10th-90th)",
            color="Phase"
        )
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Taxable Stock Value 10th").data)
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Taxable Stock Value 90th").data)
        fig_stock.data[1].update(fill=None)
        fig_stock.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        fig_stock.add_vline(
            x=years_to_retirement, 
            line_dash="dash",
            line_color="red",
            annotation_text="Retirement",
            annotation_position="top"
        )
        st.plotly_chart(fig_stock, use_container_width=True)
        
        st.subheader("Cash Flow Pattern")
        cash_flow_data = pd.DataFrame({
            'Year': summary_df['Year'],
            'Phase': summary_df['Phase'],
            'Amount': [
                stock_annual_contribution if year < years_to_retirement 
                else -(annual_budget * ((1 + inflation_rate) ** (year - years_to_retirement)))
                for year in range(total_years)
            ]
        })
        fig_cash_flow = px.bar(
            cash_flow_data,
            x='Year',
            y='Amount',
            color='Phase',
            title='Annual Contributions vs Withdrawals (Inflation Adjusted)',
            labels={'Amount': 'Cash Flow ($)'}
        )
        fig_cash_flow.add_hline(y=0, line_color='black', line_width=1)
        if years_in_retirement > 0:
            initial_withdrawal = annual_budget
            final_withdrawal = annual_budget * ((1 + inflation_rate) ** years_in_retirement)
            st.info(f"""
            **Withdrawal Amounts (Today's vs Final Year):**
            - Initial Annual Withdrawal: ${initial_withdrawal:,.2f}
            - Final Year Withdrawal: ${final_withdrawal:,.2f}
            - Total Increase: {((final_withdrawal/initial_withdrawal - 1) * 100):.1f}%
            """)
        st.plotly_chart(fig_cash_flow, use_container_width=True)
        
        num_r = len(rentals_data)
        for i in range(num_r):
            r_label_eq = f"Rental{i+1} Equity"
            r_label_mort = f"Rental{i+1} Mortgage"
            if f"{r_label_eq} Median" in summary_df.columns:
                st.subheader(f"**{r_label_eq}** Over Time")
                fig_eq = px.line(summary_df, x="Year", y=f"{r_label_eq} Median", 
                                 title=f"{r_label_eq} (Median, 10-90th)")
                fig_eq.add_traces(px.line(summary_df, x="Year", y=f"{r_label_eq} 10th").data)
                fig_eq.add_traces(px.line(summary_df, x="Year", y=f"{r_label_eq} 90th").data)
                fig_eq.data[1].update(fill=None)
                fig_eq.data[2].update(fill='tonexty', fillcolor='rgba(100,0,80,0.2)')
                st.plotly_chart(fig_eq, use_container_width=True)
            if f"{r_label_mort} Median" in summary_df.columns:
                st.subheader(f"**{r_label_mort}** Over Time")
                fig_mort = px.line(summary_df, x="Year", y=f"{r_label_mort} Median", 
                                   title=f"{r_label_mort} (Median, 10-90th)")
                fig_mort.add_traces(px.line(summary_df, x="Year", y=f"{r_label_mort} 10th").data)
                fig_mort.add_traces(px.line(summary_df, x="Year", y=f"{r_label_mort} 90th").data)
                fig_mort.data[1].update(fill=None)
                fig_mort.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
                st.plotly_chart(fig_mort, use_container_width=True)
        
        st.subheader("Detailed Simulation Results")
        st.markdown("Complete simulation results showing mean, median, and percentile values for all metrics:")
        st.dataframe(summary_df)
        
        st.markdown("""
        ### Key Assumptions and Notes
        - Stock returns include both capital appreciation and dividends
        - Property values are adjusted for mortgage paydown and appreciation
        - All withdrawals are adjusted for inflation
        - Rental income (positive cash flow) is reinvested in stocks
        - Tax-deferred and Roth accounts grow at market rates; RMDs are applied to the tax-deferred account in retirement
        - The simulation accounts for taxes on dividends, rental income, and capital gains
        """)
        
        st.subheader("Monthly Budget Breakdown")
        fig_budget = px.pie(
            values=list(monthly_expenses.values()),
            names=list(monthly_expenses.keys()),
            title=f"Monthly Budget (Total: ${total_monthly:,.2f})"
        )
        st.plotly_chart(fig_budget)
        
        ss_benefit = calculate_social_security(birth_year, retirement_age, annual_income)
        st.info(f"Estimated Annual Social Security Benefit: ${ss_benefit:,.2f}")
        
        st.subheader("Projected Healthcare Costs")
        healthcare_costs = estimate_healthcare_costs(current_age, retirement_age, life_expectancy)
        healthcare_df = pd.DataFrame({
            "Age": range(retirement_age, life_expectancy + 1),
            "Annual Cost": healthcare_costs
        })
        fig_healthcare = px.line(healthcare_df, x="Age", y="Annual Cost", title="Projected Annual Healthcare Costs")
        st.plotly_chart(fig_healthcare)
        
        if show_ss_analysis:
            st.subheader("Social Security Claiming Strategy Analysis")
            ss_analysis = analyze_social_security_strategy(birth_year, annual_income)
            st.dataframe(ss_analysis.style.format({
                'Monthly Benefit': '${:,.2f}',
                'Total by Age 80': '${:,.0f}',
                'Total by Age 85': '${:,.0f}',
                'Total by Age 90': '${:,.0f}'
            }))
        
        if show_tax_analysis:
            st.subheader("Estimated Tax Analysis in Retirement")
            total_retirement_income = (annual_withdrawal_stocks + ss_benefit + pension_income + rental_income)
            tax_breakdown, total_tax = calculate_tax_brackets(total_retirement_income)
            st.write(f"Estimated Total Retirement Income: ${total_retirement_income:,.2f}")
            st.write(f"Estimated Annual Tax: ${total_tax:,.2f}")
            st.write(f"Effective Tax Rate: {(total_tax/total_retirement_income)*100:.1f}%")
            st.dataframe(tax_breakdown.style.format({
                'Income in Bracket': '${:,.2f}',
                'Tax in Bracket': '${:,.2f}'
            }))
        
        if show_allocation_analysis:
            st.subheader("Asset Allocation Analysis")
            st.write("Compare different stock/bond allocations over your investment horizon")
            allocation_results = analyze_asset_allocations(
                initial_balance=stock_initial,
                years=years_to_retirement + years_in_retirement
            )
            st.dataframe(allocation_results.style.format({
                'Median': '${:,.0f}',
                'Worst 5%': '${:,.0f}',
                'Best 5%': '${:,.0f}'
            }))
        
        report_figures = [
            {'figure': prepare_figure_for_report(fig_components, "Net Worth Components"), 'title': "Net Worth Components"},
            {'figure': prepare_figure_for_report(fig_stock, "Taxable Stock Value"), 'title': "Taxable Stock Value"},
            {'figure': prepare_figure_for_report(fig_cash_flow, "Cash Flow Pattern"), 'title': "Cash Flow Pattern"},
            {'figure': prepare_figure_for_report(fig_budget, "Monthly Budget Breakdown"), 'title': "Monthly Budget Breakdown"},
            {'figure': prepare_figure_for_report(fig_healthcare, "Projected Healthcare Costs"), 'title': "Projected Healthcare Costs"}
        ]
        
        report_params = {
            'generation_date': date.today().strftime("%B %d, %Y"),
            'failure_rate': f"{failure_rate:.1f}",
            'initial_portfolio': stock_initial,
            'annual_contribution': stock_annual_contribution,
            'initial_withdrawal': annual_budget,
            'final_withdrawal': annual_budget * ((1 + inflation_rate) ** years_in_retirement),
            'current_age': current_age,
            'retirement_age': retirement_age,
            'years_to_retirement': years_to_retirement,
            'years_in_retirement': years_in_retirement,
            'monthly_expenses': monthly_expenses,
            'total_monthly': total_monthly,
            'ss_benefit': ss_benefit,
            'assumptions': {
                'Expected Return': f"{stock_expected_return:.1%}",
                'Volatility': f"{stock_volatility:.1%}",
                'Inflation Rate': f"{inflation_rate:.1%}",
                'Dividend Yield': f"{dividend_yield:.1%}",
                'Capital Gains Tax': f"{cap_gains_tax_rate:.1%}",
                'Healthcare Inflation': "5.0%",
            }
        }
        
        ss_analysis_data = ss_analysis.to_dict('records') if show_ss_analysis else None
        tax_analysis_data = None
        if show_tax_analysis:
            tax_analysis_data = {
                'income': total_retirement_income,
                'tax': total_tax,
                'effective_rate': (total_tax/total_retirement_income)*100,
                'breakdown': tax_breakdown.to_dict('records')
            }
        allocation_analysis_data = allocation_results.to_dict('records') if show_allocation_analysis else None
        
        report_html = generate_html_report(
            results, 
            report_params, 
            report_figures,
            monthly_expenses,
            ss_analysis_data,
            tax_analysis_data,
            allocation_analysis_data
        )
        
        with col2:
            st.download_button("Download Report", data=report_html, file_name="retirement_report.html",
                               mime="text/html", key="download_report")

        if luxury_expenses:
            st.subheader("Planned One-time Expenses")
            expense_df = pd.DataFrame(luxury_expenses)
            expense_df.columns = ['Amount', 'Purchase Year', 'Description']
            st.dataframe(expense_df.style.format({
                'Amount': '${:,.2f}'
            }))

        if num_rentals > 0:
            st.subheader("Rental Properties Summary (Detailed Cash Flow)")
            rental_summary = []
            for i, rental in enumerate(rentals_data):
                # For currently owned rentals (purchase_year == 0), calculate the cash flow breakdown.
                if rental['purchase_year'] == 0:
                    breakdown = calculate_rental_cashflow_breakdown(rental, year_index=0)
                else:
                    # For future purchases, the detailed cash flow is not available yet.
                    breakdown = {
                        "Gross Annual Rent": None, "Property Tax": None, "Maintenance": None,
                        "Total Mortgage Interest": None, "Net Operating Income": None,
                        "Depreciation": None, "Taxable Income": None, "Rental Income Tax": None,
                        "Net Rental Cash Flow": None
                    }
                rental_summary.append({
                    'Description': rental['description'],
                    'Purchase Year': rental['purchase_year'],
                    'Current Value': rental['property_value'],
                    'Mortgage Balance': rental['mortgage_balance'],
                    'Monthly Rent': rental['monthly_rent_base'],
                    'Gross Annual Rent': breakdown["Gross Annual Rent"],
                    'Property Tax': breakdown["Property Tax"],
                    'Maintenance': breakdown["Maintenance"],
                    'Total Mortgage Interest': breakdown["Total Mortgage Interest"],
                    'NOI': breakdown["Net Operating Income"],
                    'Depreciation': breakdown["Depreciation"],
                    'Taxable Income': breakdown["Taxable Income"],
                    'Rental Income Tax': breakdown["Rental Income Tax"],
                    'Net Rental Cash Flow': breakdown["Net Rental Cash Flow"],
                    'Planned Sale': f"Year {rental['sale_year']}" if rental['sale_year'] > 0 else "No plan"
                })
            rental_df = pd.DataFrame(rental_summary)
            st.dataframe(rental_df.style.format({
                'Current Value': safe_formatter('${:,.2f}'),
                'Mortgage Balance': safe_formatter('${:,.2f}'),
                'Monthly Rent': safe_formatter('${:,.2f}'),
                'Gross Annual Rent': safe_formatter('${:,.2f}'),
                'Property Tax': safe_formatter('${:,.2f}'),
                'Maintenance': safe_formatter('${:,.2f}'),
                'Total Mortgage Interest': safe_formatter('${:,.2f}'),
                'NOI': safe_formatter('${:,.2f}'),
                'Depreciation': safe_formatter('${:,.2f}'),
                'Taxable Income': safe_formatter('${:,.2f}'),
                'Rental Income Tax': safe_formatter('${:,.2f}'),
                'Net Rental Cash Flow': safe_formatter('${:,.2f}')
            }))
    
if __name__ == "__main__":
    main()
