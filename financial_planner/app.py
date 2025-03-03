import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date

# -----------------------------------------------
# Mortgage Payment Formula
# -----------------------------------------------
def standard_mortgage_payment(principal, annual_interest_rate, mortgage_years):
    """
    Computes the monthly payment for a standard amortizing mortgage
    with a fixed rate and remaining term using the standard formula.
    """
    if principal <= 0 or mortgage_years <= 0 or annual_interest_rate < 0:
        return 0.0

    monthly_rate = annual_interest_rate / 12.0
    n_months = mortgage_years * 12

    if monthly_rate == 0 or n_months == 0:
        return 0.0  # e.g., interest-free or no term left

    # Payment = P * [r(1+r)^n] / [(1+r)^n - 1]
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
        - If positive, we add it to stocks
        - If negative, we assume it's covered externally (no stock deduction).
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
    net_operating_income = (gross_annual_rent
                            - annual_property_tax
                            - annual_maintenance
                            - total_interest_paid)

    # 6) Depreciation reduces taxable income
    taxable_income = net_operating_income - annual_depreciation
    if taxable_income < 0:
        taxable_income = 0.0  # ignoring negative carry-forward

    # 7) Pay rental income tax
    rental_income_tax = taxable_income * rental_income_tax_rate

    # 8) Net rental cash flow AFTER tax
    net_rental_cash_flow = net_operating_income - rental_income_tax
    # If negative, we do NOT reduce stock in this model.

    # 9) Update cost basis
    new_cost_basis = cost_basis - annual_depreciation
    if new_cost_basis < 0:
        new_cost_basis = 0.0

    # Store updates
    updated_rental = rental.copy()
    updated_rental["property_value"] = property_value
    updated_rental["mortgage_balance"] = new_mortgage_balance
    updated_rental["cost_basis"] = new_cost_basis

    return updated_rental, net_rental_cash_flow

# -----------------------------------------------
# Stock Update (1 year)
# -----------------------------------------------
def simulate_stock(
    current_stock_value,
    stock_annual_contribution,
    expected_return,
    volatility,
    cap_gains_tax_rate,
    dividend_yield,
    dividend_tax_rate,
    withdrawal=0.0,
    inflation_adjusted_withdrawal=True,
    current_year=0,
    inflation_rate=0.03
):
    """
    One-year update for stock portfolio with inflation-adjusted withdrawals.
    """
    # Adjust withdrawal for inflation if needed
    if inflation_adjusted_withdrawal and current_year > 0:
        withdrawal *= (1 + inflation_rate) ** current_year

    # Withdraw (with check for sufficient funds)
    if withdrawal > current_stock_value:
        withdrawal = current_stock_value
        
    current_stock_value -= withdrawal

    # Add contribution
    current_stock_value += stock_annual_contribution

    # Dividends (after tax)
    dividends = current_stock_value * dividend_yield
    dividends_after_tax = dividends * (1 - dividend_tax_rate)
    current_stock_value += dividends_after_tax

    # Random return (using normal distribution)
    annual_return = np.random.normal(loc=expected_return, scale=volatility)
    current_stock_value *= (1 + annual_return)

    if current_stock_value < 0:
        current_stock_value = 0
    
    return current_stock_value, withdrawal

# Add new helper functions
def calculate_social_security(birth_year, retirement_age, annual_income):
    """
    Rough estimation of social security benefits based on retirement age and income.
    """
    # Simplified calculation - would need more complex rules for accuracy
    full_retirement_age = 67 if birth_year >= 1960 else 66
    monthly_benefit_at_full = min(annual_income * 0.4, 3895)  # Cap at max benefit
    
    if retirement_age < full_retirement_age:
        reduction = 0.0625 * (full_retirement_age - retirement_age)  # 6.25% per year early
        monthly_benefit = monthly_benefit_at_full * (1 - reduction)
    else:
        increase = 0.08 * (retirement_age - full_retirement_age)  # 8% per year delayed
        monthly_benefit = monthly_benefit_at_full * (1 + increase)
    
    return monthly_benefit * 12

def estimate_healthcare_costs(current_age, retirement_age, life_expectancy):
    """
    Estimates healthcare costs in retirement including Medicare premiums and out-of-pocket.
    """
    base_annual_cost = 12000  # Average healthcare cost for 65-year-old
    inflation_rate = 0.05  # Healthcare inflation typically higher than CPI
    
    costs = []
    for age in range(retirement_age, life_expectancy + 1):
        year = age - retirement_age
        if age < 65:
            # Pre-Medicare costs are higher
            annual_cost = base_annual_cost * 1.5 * (1 + inflation_rate) ** year
        else:
            annual_cost = base_annual_cost * (1 + inflation_rate) ** year
        costs.append(annual_cost)
    
    return costs

def calculate_rmd(age, account_balance):
    """
    Calculate Required Minimum Distribution for tax-deferred accounts.
    """
    # Simplified RMD table
    rmd_factors = {
        72: 25.6, 75: 22.9, 80: 18.7, 85: 14.8, 90: 11.4, 95: 8.6
    }
    
    # Find closest age factor
    closest_age = min(rmd_factors.keys(), key=lambda x: abs(x - age))
    return account_balance / rmd_factors[closest_age]

def analyze_social_security_strategy(birth_year, annual_income):
    """
    Analyzes different Social Security claiming strategies.
    Returns a DataFrame comparing benefits at different ages.
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

def calculate_tax_brackets(retirement_income):
    """
    Calculates estimated tax brackets in retirement.
    """
    # 2024 tax brackets (married filing jointly)
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

def analyze_asset_allocations(initial_balance, years, n_sims=1000):
    """
    Analyzes different stock/bond allocations using Monte Carlo simulation.
    """
    allocations = [
        (100, 0),  # 100% stocks
        (80, 20),  # 80/20
        (60, 40),  # 60/40
        (40, 60),  # 40/60
    ]
    
    results = []
    
    for stocks, bonds in allocations:
        stock_pct = stocks / 100
        bond_pct = bonds / 100
        
        # Simplified return assumptions
        stock_return = 0.07
        stock_vol = 0.15
        bond_return = 0.03
        bond_vol = 0.05
        
        portfolio_return = (stock_return * stock_pct) + (bond_return * bond_pct)
        portfolio_vol = np.sqrt((stock_vol**2 * stock_pct**2) + 
                              (bond_vol**2 * bond_pct**2))
        
        final_values = []
        for _ in range(n_sims):
            value = initial_balance
            for _ in range(years):
                return_this_year = np.random.normal(portfolio_return, portfolio_vol)
                value *= (1 + return_this_year)
            final_values.append(value)
            
        results.append({
            'Allocation': f"{stocks}/{bonds}",
            'Median': np.median(final_values),
            'Worst 5%': np.percentile(final_values, 5),
            'Best 5%': np.percentile(final_values, 95)
        })
    
    return pd.DataFrame(results)

# -----------------------------------------------
# Main Simulation
# -----------------------------------------------
def run_simulation(
    # Required parameters first
    birth_year,
    retirement_age,
    life_expectancy,
    annual_income,
    annual_budget,  # Renamed from monthly_budget to annual_budget
    tax_deferred_balance,
    roth_balance,
    n_sims,
    # Stock parameters
    stock_initial,
    stock_annual_contribution,
    stock_expected_return,
    stock_volatility,
    cap_gains_tax_rate,
    dividend_yield,
    dividend_tax_rate,
    withdrawal,
    # Rentals
    rentals_data,
    # Primary Residence
    primary_residence_value,
    primary_mortgage_balance,
    primary_mortgage_interest_rate,
    primary_mortgage_years_left,
    primary_appreciation_mean,
    primary_appreciation_std,
    # Luxury Expense
    luxury_expense_amount,
    luxury_expense_year,
    # Optional parameters last
    inflation_rate=0.03,
    healthcare_inflation=0.05
):
    """
    Enhanced simulation including social security, healthcare, and tax-advantaged accounts.
    """
    # Calculate simulation years based on age parameters
    current_year = date.today().year
    current_age = current_year - birth_year
    years_to_retirement = retirement_age - current_age
    years_in_retirement = life_expectancy - retirement_age
    total_years = years_to_retirement + years_in_retirement
    
    all_sims = []
    failure_count = 0  # Track simulations that run out of money
    
    num_rentals = len(rentals_data)
    
    # Calculate social security benefit
    annual_ss_benefit = calculate_social_security(birth_year, retirement_age, annual_income)
    
    # Get healthcare cost projections
    healthcare_costs = estimate_healthcare_costs(date.today().year - birth_year, 
                                              retirement_age, life_expectancy)
    
    # Add phase labels for visualization
    phase_labels = ['Accumulation' if year < years_to_retirement else 'Retirement' 
                   for year in range(total_years)]
    
    for sim_i in range(n_sims):
        # Initialize stock
        current_stock_value = stock_initial
        total_contributions = 0.0  # Track total contributions to stocks
        total_appreciation = 0.0  # Track total appreciation of stocks

        # Initialize primary residence
        current_primary_residence_value = primary_residence_value
        current_primary_mortgage_balance = primary_mortgage_balance

        # Calculate monthly payment for primary residence
        primary_monthly_payment = standard_mortgage_payment(
            principal=primary_mortgage_balance,
            annual_interest_rate=primary_mortgage_interest_rate,
            mortgage_years=primary_mortgage_years_left
        )

        # Copy rentals so as not to overwrite original
        my_rentals = [rd.copy() for rd in rentals_data]

        # Lists to store annual results
        year_list = []
        stock_vals = []
        total_net_worth_list = []
        primary_residence_equity_list = []
        stock_contributions_list = []
        stock_appreciation_list = []
        
        rentals_equity_lists = [ [] for _ in range(num_rentals) ]
        rentals_mortgage_lists = [ [] for _ in range(num_rentals) ]
        rentals_cashflow_lists = [ [] for _ in range(num_rentals) ]

        portfolio_failed = False
        actual_withdrawals = []  # Track actual withdrawal amounts

        for year in range(total_years):
            current_age = (date.today().year - birth_year) + year
            year_label = year + 1
            is_retirement = (year >= years_to_retirement)

            # 1) Update each rental
            for r_i, rental in enumerate(my_rentals):
                if rental["property_value"] <= 0:
                    # Already sold/zero
                    rentals_equity_lists[r_i].append(0)
                    rentals_mortgage_lists[r_i].append(0)
                    rentals_cashflow_lists[r_i].append(0)
                    continue
                
                updated_r, net_cf = update_rental_property_monthly(rental, year_index=year)
                my_rentals[r_i] = updated_r

                # If net CF > 0, add it to stocks
                if net_cf > 0:
                    current_stock_value += net_cf

                # Check if we sell this year
                sale_year = updated_r["sale_year"]
                if sale_year == year_label:
                    # Gains
                    gains = updated_r["property_value"] - updated_r["cost_basis"]
                    if gains < 0:
                        gains = 0
                    cap_gains_tax = gains * updated_r["rental_cap_gains_tax"]
                    net_sale_proceeds = (updated_r["property_value"] - updated_r["mortgage_balance"]) - cap_gains_tax
                    if net_sale_proceeds < 0:
                        net_sale_proceeds = 0
                    current_stock_value += net_sale_proceeds
                    
                    # Zero out
                    updated_r["property_value"] = 0
                    updated_r["mortgage_balance"] = 0
                    updated_r["cost_basis"] = 0
                    my_rentals[r_i] = updated_r
                
                # Record equity & mortgage
                eq = max(updated_r["property_value"] - updated_r["mortgage_balance"], 0)
                rentals_equity_lists[r_i].append(eq)
                rentals_mortgage_lists[r_i].append(updated_r["mortgage_balance"])
                rentals_cashflow_lists[r_i].append(net_cf)

            # 2) Luxury Expense in this year?
            if (luxury_expense_amount > 0) and (luxury_expense_year == year_label):
                # clamp to stock
                expense = min(luxury_expense_amount, current_stock_value)
                current_stock_value -= expense

            # 3) Retirement Income and Expenses
            if is_retirement:
                # Add social security income
                ss_income = annual_ss_benefit * (1 + inflation_rate) ** (year - years_to_retirement)
                current_stock_value += ss_income
                
                # Subtract healthcare costs
                if year - years_to_retirement < len(healthcare_costs):
                    healthcare_cost = healthcare_costs[year - years_to_retirement]
                    current_stock_value -= healthcare_cost
                
                # Handle RMDs from tax-deferred accounts
                if current_age >= 72 and tax_deferred_balance > 0:
                    rmd = calculate_rmd(current_age, tax_deferred_balance)
                    tax_deferred_balance -= rmd
                    current_stock_value += rmd * 0.8  # Assuming 20% tax rate on RMDs

            # 4) Stocks (accum or retire)
            stock_contribution = stock_annual_contribution if not is_retirement else 0
            stock_withdrawal = withdrawal if is_retirement else 0
            
            current_stock_value, actual_withdrawal = simulate_stock(
                current_stock_value,
                stock_contribution,
                stock_expected_return,
                stock_volatility,
                cap_gains_tax_rate,
                dividend_yield,
                dividend_tax_rate,
                withdrawal=stock_withdrawal,
                inflation_adjusted_withdrawal=True,
                current_year=year,
                inflation_rate=inflation_rate
            )
            
            # Track if portfolio fails (can't meet withdrawal needs)
            if is_retirement and actual_withdrawal < stock_withdrawal:
                portfolio_failed = True

            actual_withdrawals.append(actual_withdrawal)
            
            # Track contributions and appreciation
            total_contributions += stock_contribution
            total_appreciation = current_stock_value - total_contributions

            # Update primary residence
            annual_appreciation = np.random.normal(loc=primary_appreciation_mean, scale=primary_appreciation_std)
            current_primary_residence_value *= (1 + annual_appreciation)

            # Monthly mortgage loop for primary residence
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

            # Calculate total net worth
            total_net_worth = (
                current_stock_value +
                sum(max(r["property_value"] - r["mortgage_balance"], 0) for r in my_rentals) +
                max(current_primary_residence_value - current_primary_mortgage_balance, 0)
            )

            # Record values for this year
            year_list.append(year_label)
            stock_vals.append(current_stock_value)
            total_net_worth_list.append(total_net_worth)
            primary_residence_equity_list.append(max(current_primary_residence_value - current_primary_mortgage_balance, 0))
            stock_contributions_list.append(total_contributions)
            stock_appreciation_list.append(total_appreciation)

        # Build a DataFrame for this simulation
        sim_data = {
            "Year": year_list,
            "Stock Value": stock_vals,
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

    # Combine all simulations => summary
    combined_df = None
    for i, df_i in enumerate(all_sims):
        rename_map = {}
        for col in df_i.columns:
            if col != "Year":
                rename_map[col] = f"{col} {i}"
        temp = df_i.rename(columns=rename_map)
        if combined_df is None:
            combined_df = temp
        else:
            combined_df = pd.merge(combined_df, temp, on="Year", how="inner")
    
    summary_df = pd.DataFrame({"Year": combined_df["Year"]})
    
    # Gather columns by original name
    col_groups = {}
    for col in all_sims[0].columns:
        if col == "Year":
            continue
        matching = [c for c in combined_df.columns if c.startswith(col)]
        col_groups[col] = matching
    
    # For each group, compute mean/median/10th/90th
    for group_name, cols in col_groups.items():
        summary_df[f"{group_name} Mean"]   = combined_df[cols].mean(axis=1)
        summary_df[f"{group_name} Median"] = combined_df[cols].median(axis=1)
        summary_df[f"{group_name} 10th"]   = combined_df[cols].quantile(0.1, axis=1)
        summary_df[f"{group_name} 90th"]   = combined_df[cols].quantile(0.9, axis=1)
    
    # Add phase labels to summary DataFrame
    summary_df['Phase'] = phase_labels

    return {
        "all_sims": all_sims,
        "combined": combined_df,
        "summary": summary_df,
        "failure_rate": failure_rate
    }

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
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
    """)

    # ---------- SIDEBAR ----------
    st.sidebar.title("Your Information")
    
    # 1. Personal Information (moved to top)
    st.sidebar.header("1️⃣ Personal Details")
    birth_year = st.sidebar.number_input(
        "Birth Year",
        1940, 2000, 1970,
        help="Used to calculate Social Security benefits and RMDs"
    )
    retirement_age = st.sidebar.number_input(
        "Planned Retirement Age",
        55, 75, 65,
        help="Age at which you plan to retire"
    )
    life_expectancy = st.sidebar.number_input(
        "Life Expectancy",
        retirement_age, 100, 90,
        help="Plan through this age for safety"
    )
    annual_income = st.sidebar.number_input(
        "Current Annual Income",
        0, 500_000, 80_000,
        help="Used for Social Security calculation"
    )
    
    # Calculate and display timeline
    current_year = date.today().year
    current_age = current_year - birth_year
    years_to_retirement = retirement_age - current_age
    years_in_retirement = life_expectancy - retirement_age
    
    st.sidebar.info(f"""
    **Your Timeline:**
    🎂 Current Age: {current_age}
    ⏳ Years until retirement: {years_to_retirement}
    🌅 Years in retirement: {years_in_retirement}
    """)
    
    # 2. Monthly Budget
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
    
    # 3. Current Assets
    st.sidebar.header("3️⃣ Current Assets")
    
    # Retirement Accounts
    with st.sidebar.expander("Retirement Accounts", expanded=True):
        tax_deferred_balance = st.number_input(
            "Tax-Deferred Balance (401k, IRA)",
            0, 10_000_000, 100_000,
            help="Current balance in traditional retirement accounts"
        )
        roth_balance = st.number_input(
            "Roth Balance",
            0, 10_000_000, 50_000,
            help="Current balance in Roth accounts"
        )
        stock_initial = st.number_input(
            "Taxable Investment Balance", 
            0, 10_000_000, 50_000, step=1_000,
            help="Current balance in taxable investment accounts"
        )
    
    # Primary Residence
    with st.sidebar.expander("Primary Residence", expanded=True):
        primary_residence_value = st.number_input(
            "Home Value",
            0, 10_000_000, 300_000,
            help="Current market value of your primary residence"
        )
        primary_mortgage_balance = st.number_input(
            "Mortgage Balance",
            0, 10_000_000, 150_000,
            help="Remaining balance on your mortgage"
        )
        primary_mortgage_interest_rate = st.slider(
            "Mortgage Rate",
            0.0, 0.2, 0.04, 0.005,
            help="Annual interest rate on your mortgage"
        )
        primary_mortgage_years_left = st.number_input(
            "Years Left on Mortgage",
            0, 40, 25,
            help="Remaining years on your mortgage term"
        )
        primary_appreciation_mean = st.slider(
            "Home Appreciation Rate",
            0.0, 0.2, 0.03, 0.01,
            help="Expected annual appreciation rate for your home"
        )
        primary_appreciation_std = st.slider(
            "Home Appreciation Volatility",
            0.0, 0.5, 0.1, 0.01,
            help="Standard deviation of annual home appreciation"
        )
    
    # Rental Properties
    with st.sidebar.expander("Rental Properties", expanded=True):
        num_rentals = st.number_input(
            "Number of Rental Properties",
            0, 5, 0,
            help="How many rental properties do you own?"
        )
        
        rentals_data = []
        for i in range(num_rentals):
            st.markdown(f"**Rental Property #{i+1}**")
            
            rental_data = {
                "property_value": st.number_input(
                    f"Property Value (Rental {i+1})", 
                    0, 10_000_000, 200_000, step=1_000,
                    help="Current market value of the rental property"
                ),
                "mortgage_balance": st.number_input(
                    f"Mortgage Balance (Rental {i+1})", 
                    0, 10_000_000, 120_000, step=1_000,
                    help="Remaining balance on the mortgage"
                ),
                "mortgage_interest_rate": st.slider(
                    f"Mortgage Rate (Rental {i+1})", 
                    0.0, 0.2, 0.04, 0.005,
                    help="Annual interest rate on the mortgage"
                ),
                "monthly_rent_base": st.number_input(
                    f"Monthly Rent (Rental {i+1})", 
                    0, 50_000, 1500, step=100,
                    help="Current monthly rent charged"
                ),
                "cost_basis": st.number_input(
                    f"Cost Basis (Rental {i+1})", 
                    0, 2_000_000, 180_000, step=1_000,
                    help="Original purchase price plus improvements"
                ),
                "sale_year": st.number_input(
                    f"Planned Sale Year (Rental {i+1})", 
                    0, 100, 0, step=1,
                    help="Year you plan to sell (0 = no plan)"
                ),
                "rental_cap_gains_tax": st.slider(
                    f"Capital Gains Tax Rate (Rental {i+1})", 
                    0.0, 0.5, 0.15, 0.01,
                    help="Expected tax rate on sale profits"
                ),
                "property_tax_rate": st.slider(
                    f"Property Tax Rate (Rental {i+1})", 
                    0.0, 0.05, 0.01, 0.001,
                    help="Annual property tax as % of value"
                ),
                "maintenance_rate": st.slider(
                    f"Maintenance Rate (Rental {i+1})", 
                    0.0, 0.05, 0.01, 0.001,
                    help="Annual maintenance as % of value"
                ),
                "vacancy_rate": st.slider(
                    f"Vacancy Rate (Rental {i+1})", 
                    0.0, 1.0, 0.05, 0.01,
                    help="Expected vacancy rate"
                ),
                "rental_income_tax_rate": st.slider(
                    f"Rental Income Tax Rate (Rental {i+1})", 
                    0.0, 0.5, 0.2, 0.01,
                    help="Tax rate on rental income"
                ),
                "rent_inflation": st.slider(
                    f"Rent Inflation (Rental {i+1})", 
                    0.0, 0.1, 0.02, 0.01,
                    help="Expected annual increase in rent"
                ),
                "app_mean": st.slider(
                    f"Appreciation Rate (Rental {i+1})", 
                    0.0, 0.2, 0.03, 0.01,
                    help="Expected annual property appreciation"
                ),
                "app_std": st.slider(
                    f"Appreciation Volatility (Rental {i+1})", 
                    0.0, 0.5, 0.1, 0.01,
                    help="Standard deviation of appreciation"
                )
            }
            
            # Calculate monthly payment and depreciation
            monthly_payment = standard_mortgage_payment(
                principal=rental_data["mortgage_balance"],
                annual_interest_rate=rental_data["mortgage_interest_rate"],
                mortgage_years=30  # Assuming 30-year mortgages for rentals
            )
            rental_data["monthly_payment"] = monthly_payment
            
            # Standard 27.5 year depreciation for residential rentals
            rental_data["annual_depreciation"] = rental_data["cost_basis"] / 27.5
            
            rentals_data.append(rental_data)

    # 4. Investment Settings
    st.sidebar.header("4️⃣ Investment Strategy")
    
    # Contributions
    with st.sidebar.expander("Contributions & Withdrawals", expanded=True):
        stock_annual_contribution = st.number_input(
            "Annual Investment Contribution", 
            0, 1_000_000, 10_000, step=1_000,
            help="Amount you plan to invest annually during working years"
        )
        # Remove the manual withdrawal input and replace with calculated value
        withdrawal_adjustment = st.slider(
            "Retirement Budget Adjustment",
            0.5, 1.5, 1.0, 0.05,
            help="Adjust retirement spending relative to current budget (1.0 = same as current)"
        )
        
        # Calculate annual withdrawal based on monthly expenses
        monthly_budget = sum(monthly_expenses.values())
        annual_withdrawal_stocks = monthly_budget * 12 * withdrawal_adjustment
        
        # Display the calculated withdrawal amount
        st.info(f"""
        **Planned Annual Withdrawal: ${annual_withdrawal_stocks:,.2f}**
        - Based on current monthly expenses: ${monthly_budget:,.2f}
        - Adjusted by factor: {withdrawal_adjustment:.2f}x
        - Withdrawal Rate: {(annual_withdrawal_stocks / stock_initial * 100):.1f}% of current portfolio
        """)

    # Market Assumptions
    with st.sidebar.expander("Market Assumptions", expanded=False):
        stock_expected_return = st.slider(
            "Expected Return (mean)", 
            0.0, 0.2, 0.07, 0.01,
            help="Average annual investment return before inflation"
        )
        stock_volatility = st.slider(
            "Volatility (std dev)", 
            0.0, 0.5, 0.15, 0.01,
            help="Standard deviation of annual returns"
        )
        dividend_yield = st.slider(
            "Dividend Yield", 
            0.0, 0.1, 0.02, 0.01,
            help="Expected annual dividend yield"
        )
    
    # 5. Tax & Economic Assumptions
    st.sidebar.header("5️⃣ Tax & Economic Factors")
    with st.sidebar.expander("Tax & Inflation Settings", expanded=False):
        cap_gains_tax_rate = st.slider(
            "Capital Gains Tax Rate", 
            0.0, 0.5, 0.15, 0.01,
            help="Long-term capital gains tax rate"
        )
        dividend_tax_rate = st.slider(
            "Dividend Tax Rate", 
            0.0, 0.5, 0.15, 0.01,
            help="Tax rate on dividend income"
        )
        inflation_rate = st.slider(
            "Inflation Rate", 
            0.01, 0.10, 0.03, 0.001,
            help="Expected annual inflation rate"
        )
    
    # 6. Optional Planning
    st.sidebar.header("6️⃣ Optional Planning")
    with st.sidebar.expander("Future Expenses", expanded=False):
        luxury_expense_amount = st.number_input(
            "One-time Expense Amount", 
            0, 5_000_000, 0, step=1_000,
            help="Optional future large expense (e.g., vacation home)"
        )
        if luxury_expense_amount > 0:
            luxury_expense_year = st.number_input(
                "Year for Expense", 
                1, years_to_retirement + years_in_retirement, 1,
                help="Which year to plan this expense"
            )
        else:
            luxury_expense_year = 0
    
    # 7. Simulation Settings
    st.sidebar.header("7️⃣ Simulation Settings")
    n_sims = st.sidebar.number_input(
        "Number of Simulations", 
        100, 2000, 200, 
        help="More simulations = more accurate results but slower"
    )
    
    # Warning about assumptions
    st.sidebar.warning("""
    ⚠️ **Important Notes:**
    - Past performance doesn't guarantee future returns
    - The default 7% return is before inflation
    - Consider using conservative estimates
    - Early retirement years have outsized impact
    """)

    # Add before the simulation button
    # 8. Analysis Options
    st.sidebar.header("8️⃣ Analysis Options")
    show_ss_analysis = st.sidebar.checkbox("Show Social Security Analysis", value=False)
    show_tax_analysis = st.sidebar.checkbox("Show Tax Analysis", value=False)
    show_allocation_analysis = st.sidebar.checkbox("Show Asset Allocation Analysis", value=False)
    
    if show_tax_analysis:
        st.sidebar.subheader("Additional Tax Analysis Inputs")
        pension_income = st.sidebar.number_input(
            "Expected Annual Pension", 
            0, 200_000, 0,
            help="Expected annual pension income in retirement"
        )
        rental_income = st.sidebar.number_input(
            "Expected Annual Rental Income", 
            0, 200_000, 0,
            help="Expected annual rental income in retirement"
        )

    # ---------- MAIN CONTENT ----------
    if st.button("Run Simulation"):
        # Calculate monthly and annual budget from expenses
        monthly_budget = sum(monthly_expenses.values())
        annual_budget = monthly_budget * 12

        with st.spinner("Simulating..."):
            results = run_simulation(
                birth_year,
                retirement_age,
                life_expectancy,
                annual_income,
                annual_budget,  # Pass annual budget instead of monthly
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
                luxury_expense_amount,
                luxury_expense_year,
                inflation_rate,
                healthcare_inflation=0.05
            )
        
        st.success("Simulation Complete!")

        # Display failure rate at the top
        failure_rate = results["failure_rate"]
        st.error(f"Portfolio Failure Rate: {failure_rate:.1f}%")
        st.markdown("""
        **Portfolio Failure** means the simulation couldn't maintain the desired 
        withdrawal rate adjusted for inflation. A failure rate above 5% suggests 
        the plan may be too risky.
        """)

        # Net Worth Components with Phase Highlighting
        st.subheader("Total Net Worth Components Over Time")
        summary_df = results["summary"]
        
        # Create the stacked area chart
        fig_components = px.area(summary_df, x="Year", 
            y=["Stock Value Median", "Primary Residence Equity Median"] + 
              [f"Rental{i+1} Equity Median" for i in range(num_rentals)],
            title="Net Worth Components Over Time",
            labels={"value": "Value ($)", "variable": "Component"}
        )
        
        # Add vertical line at retirement
        fig_components.add_vline(
            x=years_to_retirement, 
            line_dash="dash",
            line_color="red",
            annotation_text="Retirement",
            annotation_position="top"
        )
        
        # Add phase labels
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

        # Add accumulation phase summary
        st.subheader("Accumulation Phase Summary")
        accum_end = summary_df[summary_df['Year'] == years_to_retirement].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Contributions",
                f"${accum_end['Stock Contributions Median']:,.0f}",
                help="Total amount contributed during working years"
            )
        with col2:
            st.metric(
                "Investment Growth",
                f"${accum_end['Stock Appreciation Median']:,.0f}",
                help="Total investment returns during accumulation"
            )
        with col3:
            st.metric(
                "Net Worth at Retirement",
                f"${accum_end['Total Net Worth Median']:,.0f}",
                help="Total net worth when entering retirement"
            )

        # Update Stock Value chart to show phases
        st.subheader("Stock Portfolio Value Over Time")
        fig_stock = px.line(
            summary_df, 
            x="Year", 
            y="Stock Value Median", 
            title="Stock Value (Median, 10th-90th)",
            color="Phase"  # Color lines by phase
        )
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Stock Value 10th").data)
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Stock Value 90th").data)
        fig_stock.data[1].update(fill=None)
        fig_stock.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        
        # Add vertical retirement line
        fig_stock.add_vline(
            x=years_to_retirement, 
            line_dash="dash",
            line_color="red",
            annotation_text="Retirement",
            annotation_position="top"
        )
        
        st.plotly_chart(fig_stock, use_container_width=True)

        # Add contribution vs withdrawal visualization
        st.subheader("Cash Flow Pattern")
        total_years = years_to_retirement + years_in_retirement
        cash_flow_data = pd.DataFrame({
            'Year': summary_df['Year'],
            'Phase': summary_df['Phase'],
            'Amount': [stock_annual_contribution if year < years_to_retirement else -annual_withdrawal_stocks 
                      for year in range(total_years)]
        })
        
        fig_cash_flow = px.bar(
            cash_flow_data,
            x='Year',
            y='Amount',
            color='Phase',
            title='Annual Contributions vs Withdrawals',
            labels={'Amount': 'Cash Flow ($)'}
        )
        fig_cash_flow.add_hline(y=0, line_color='black', line_width=1)
        st.plotly_chart(fig_cash_flow, use_container_width=True)

        # Detailed Rental Plots
        num_r = len(rentals_data)
        for i in range(num_r):
            r_label_eq = f"Rental{i+1} Equity"
            r_label_mort = f"Rental{i+1} Mortgage"
            
            # Equity
            if f"{r_label_eq} Median" in summary_df.columns:
                st.subheader(f"**{r_label_eq}** Over Time")
                fig_eq = px.line(summary_df, x="Year", y=f"{r_label_eq} Median", 
                                 title=f"{r_label_eq} (Median, 10-90th)")
                fig_eq.add_traces(px.line(summary_df, x="Year", y=f"{r_label_eq} 10th").data)
                fig_eq.add_traces(px.line(summary_df, x="Year", y=f"{r_label_eq} 90th").data)
                fig_eq.data[1].update(fill=None)
                fig_eq.data[2].update(fill='tonexty', fillcolor='rgba(100,0,80,0.2)')
                fig_eq.update_traces(name='10th percentile', selector=dict(name=f"{r_label_eq} 10th"))
                fig_eq.update_traces(name='90th percentile', selector=dict(name=f"{r_label_eq} 90th"))
                fig_eq.update_traces(name='Median', selector=dict(name=f"{r_label_eq} Median"))
                st.plotly_chart(fig_eq, use_container_width=True)

            # Mortgage
            if f"{r_label_mort} Median" in summary_df.columns:
                st.subheader(f"**{r_label_mort}** Over Time")
                fig_mort = px.line(summary_df, x="Year", y=f"{r_label_mort} Median", 
                                   title=f"{r_label_mort} (Median, 10-90th)")
                fig_mort.add_traces(px.line(summary_df, x="Year", y=f"{r_label_mort} 10th").data)
                fig_mort.add_traces(px.line(summary_df, x="Year", y=f"{r_label_mort} 90th").data)
                fig_mort.data[1].update(fill=None)
                fig_mort.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
                fig_mort.update_traces(name='10th percentile', selector=dict(name=f"{r_label_mort} 10th"))
                fig_mort.update_traces(name='90th percentile', selector=dict(name=f"{r_label_mort} 90th"))
                fig_mort.update_traces(name='Median', selector=dict(name=f"{r_label_mort} Median"))
                st.plotly_chart(fig_mort, use_container_width=True)

        # Move summary table to bottom
        st.subheader("Detailed Simulation Results")
        st.markdown("Complete simulation results showing mean, median, and percentile values for all metrics:")
        st.dataframe(summary_df)

        # Final recap
        st.markdown("""
        ### Key Assumptions and Notes
        - Stock returns include both capital appreciation and dividends
        - Property values are adjusted for mortgage paydown and appreciation
        - All withdrawals are adjusted for inflation
        - Rental income (positive cash flow) is reinvested in stocks
        - The simulation accounts for taxes on dividends, rental income, and capital gains
        """)

        # Add new visualizations after simulation
        st.subheader("Monthly Budget Breakdown")
        fig_budget = px.pie(
            values=list(monthly_expenses.values()),
            names=list(monthly_expenses.keys()),
            title=f"Monthly Budget (Total: ${total_monthly:,.2f})"
        )
        st.plotly_chart(fig_budget)
        
        # Show estimated Social Security benefit
        ss_benefit = calculate_social_security(birth_year, retirement_age, annual_income)
        st.info(f"Estimated Annual Social Security Benefit: ${ss_benefit:,.2f}")
        
        # Show healthcare cost projection
        st.subheader("Projected Healthcare Costs")
        healthcare_costs = estimate_healthcare_costs(
            date.today().year - birth_year,
            retirement_age,
            life_expectancy
        )
        healthcare_df = pd.DataFrame({
            "Age": range(retirement_age, life_expectancy + 1),
            "Annual Cost": healthcare_costs
        })
        fig_healthcare = px.line(
            healthcare_df,
            x="Age",
            y="Annual Cost",
            title="Projected Annual Healthcare Costs"
        )
        st.plotly_chart(fig_healthcare)

        # Add Social Security strategy analysis
        if show_ss_analysis:
            st.subheader("Social Security Claiming Strategy Analysis")
            ss_analysis = analyze_social_security_strategy(birth_year, annual_income)
            st.dataframe(ss_analysis.style.format({
                'Monthly Benefit': '${:,.2f}',
                'Total by Age 80': '${:,.0f}',
                'Total by Age 85': '${:,.0f}',
                'Total by Age 90': '${:,.0f}'
            }))

        # Add tax analysis
        if show_tax_analysis:
            st.subheader("Estimated Tax Analysis in Retirement")
            total_retirement_income = (
                annual_withdrawal_stocks +
                ss_benefit +
                pension_income +
                rental_income
            )
            
            tax_breakdown, total_tax = calculate_tax_brackets(total_retirement_income)
            
            st.write(f"Estimated Total Retirement Income: ${total_retirement_income:,.2f}")
            st.write(f"Estimated Annual Tax: ${total_tax:,.2f}")
            st.write(f"Effective Tax Rate: {(total_tax/total_retirement_income)*100:.1f}%")
            st.dataframe(tax_breakdown.style.format({
                'Income in Bracket': '${:,.2f}',
                'Tax in Bracket': '${:,.2f}'
            }))

        # Add asset allocation analysis
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


if __name__ == "__main__":
    main()
