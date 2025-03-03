import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

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


# -----------------------------------------------
# Main Simulation
# -----------------------------------------------
def run_simulation(
    n_sims,
    years_accum,
    years_retire,
    # Stock
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
    inflation_rate=0.03
):
    """
    Simulates multiple rentals + stock + luxury expense with inflation adjustment
    and failure rate tracking.
    """
    total_years = years_accum + years_retire
    all_sims = []
    failure_count = 0  # Track simulations that run out of money
    
    num_rentals = len(rentals_data)
    
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
            year_label = year + 1
            is_retirement = (year >= years_accum)

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

            # 3) Stocks (accum or retire)
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

            # Record total net worth and primary residence equity
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
    st.sidebar.header("Simulation Settings")
    n_sims = st.sidebar.number_input(
        "Number of Simulations", 
        1, 2000, 200, 
        help="The number of simulation runs to perform. More simulations provide better statistical accuracy."
    )
    years_accum = st.sidebar.number_input(
        "Years of Accumulation", 
        1, 50, 20, 
        help="The number of years you plan to accumulate wealth before retirement."
    )
    years_retire = st.sidebar.number_input(
        "Years of Retirement", 
        1, 50, 30, 
        help="The number of years you expect to be in retirement."
    )
    
    # Stock
    st.sidebar.header("Stock Parameters")
    stock_initial = st.sidebar.number_input(
        "Initial Stock Value", 
        0, 10_000_000, 50_000, step=1_000,
        help="The starting value of your stock portfolio."
    )
    stock_annual_contribution = st.sidebar.number_input(
        "Annual Stock Contribution (Accum)", 
        0, 1_000_000, 10_000, step=1_000,
        help="The amount you plan to contribute to your stock portfolio annually during the accumulation phase."
    )
    stock_expected_return = st.sidebar.slider(
        "Stock Return (mean)", 
        0.0, 0.2, 0.07, 0.01,
        help="The average annual return you expect from your stock investments."
    )
    stock_volatility = st.sidebar.slider(
        "Stock Volatility (std dev)", 
        0.0, 0.5, 0.10, 0.01,
        help="The standard deviation of the stock's annual returns, representing the investment's risk."
    )
    stock_cap_gains_tax = st.sidebar.slider(
        "Stock Cap Gains Tax Rate", 
        0.0, 0.5, 0.15, 0.01,
        help="The tax rate applied to capital gains from stock sales."
    )
    stock_dividend_yield = st.sidebar.slider(
        "Dividend Yield (Stocks)", 
        0.0, 0.1, 0.02, 0.01,
        help="The annual dividend income as a percentage of the stock's value."
    )
    stock_dividend_tax_rate = st.sidebar.slider(
        "Dividend Tax Rate (Stocks)", 
        0.0, 0.5, 0.15, 0.01,
        help="The tax rate applied to dividend income from stocks."
    )
    annual_withdrawal_stocks = st.sidebar.number_input(
        "Annual Stock Withdrawal (Retirement)", 
        0, 500_000, 20_000, step=1_000,
        help="The amount you plan to withdraw from your stock portfolio annually during retirement."
    )

    # Rentals
    st.sidebar.header("Multiple Rentals")
    num_rentals = st.sidebar.number_input(
        "Number of Rentals", 
        0, 5, 1, step=1,
        help="The number of rental properties you own."
    )

    rentals_data = []
    for i in range(num_rentals):
        st.sidebar.subheader(f"Rental #{i+1}")
        property_value = st.sidebar.number_input(
            f"Property Value (Rental {i+1})", 
            0, 10_000_000, 200_000, step=1_000,
            help="The current market value of the rental property."
        )
        mortgage_balance = st.sidebar.number_input(
            f"Mortgage Balance (Rental {i+1})", 
            0, 10_000_000, 120_000, step=1_000,
            help="The remaining balance on the mortgage for the rental property."
        )
        mortgage_interest_rate = st.sidebar.slider(
            f"Mortgage Rate (Rental {i+1})", 
            0.0, 0.2, 0.04, 0.005,
            help="The annual interest rate on the mortgage for the rental property."
        )
        mortgage_years_left = st.sidebar.number_input(
            f"Years Left (Rental {i+1})", 
            0, 40, 25, step=1,
            help="The number of years remaining on the mortgage for the rental property."
        )
        monthly_rent_base = st.sidebar.number_input(
            f"Base Monthly Rent (Rental {i+1})", 
            0, 50_000, 1500, step=100,
            help="The base monthly rent you charge for the rental property."
        )
        cost_basis = st.sidebar.number_input(
            f"Cost Basis (Rental {i+1})", 
            0, 2_000_000, 180_000, step=1_000,
            help="The original cost basis of the rental property for tax purposes."
        )
        rental_sale_year = st.sidebar.number_input(
            f"Sale Year (Rental {i+1})", 
            0, 100, 0, step=1,
            help="The year in which you plan to sell the rental property."
        )
        rental_cap_gains_tax = st.sidebar.slider(
            f"Cap Gains Tax (Rental {i+1})", 
            0.0, 0.5, 0.15, 0.01,
            help="The tax rate applied to capital gains from the sale of the rental property."
        )
        
        property_tax_rate = st.sidebar.slider(
            f"Prop Tax Rate (R{i+1})", 
            0.0, 0.05, 0.01, 0.001,
            help="The annual property tax rate for the rental property."
        )
        maintenance_rate = st.sidebar.slider(
            f"Maintenance Rate (R{i+1})", 
            0.0, 0.05, 0.01, 0.001,
            help="The annual maintenance cost as a percentage of the property's value."
        )
        vacancy_rate = st.sidebar.slider(
            f"Vacancy Rate (R{i+1})", 
            0.0, 1.0, 0.05, 0.01,
            help="The expected vacancy rate for the rental property."
        )
        rental_income_tax_rate = st.sidebar.slider(
            f"Rental Income Tax (R{i+1})", 
            0.0, 0.5, 0.2, 0.01,
            help="The tax rate applied to rental income."
        )
        rent_inflation = st.sidebar.slider(
            f"Rent Inflation (R{i+1})", 
            0.0, 0.1, 0.02, 0.01,
            help="The expected annual increase in rent prices."
        )
        app_mean = st.sidebar.slider(
            f"Appreciation Mean (R{i+1})", 
            0.0, 0.2, 0.03, 0.01,
            help="The average annual appreciation rate of the property's value."
        )
        app_std = st.sidebar.slider(
            f"Appreciation StdDev (R{i+1})", 
            0.0, 0.5, 0.1, 0.01,
            help="The standard deviation of the property's annual appreciation rate."
        )
        depr_years = st.sidebar.number_input(
            f"Depreciation Yrs (R{i+1})", 
            0, 50, 27, step=1,
            help="The number of years over which the property is depreciated for tax purposes."
        )

        monthly_payment = standard_mortgage_payment(
            principal=mortgage_balance,
            annual_interest_rate=mortgage_interest_rate,
            mortgage_years=mortgage_years_left
        )
        if depr_years > 0:
            annual_depreciation = cost_basis / depr_years
        else:
            annual_depreciation = 0.0

        rentals_data.append({
            "property_value": property_value,
            "mortgage_balance": mortgage_balance,
            "mortgage_interest_rate": mortgage_interest_rate,
            "monthly_payment": monthly_payment,
            "monthly_rent_base": monthly_rent_base,
            "vacancy_rate": vacancy_rate,
            "property_tax_rate": property_tax_rate,
            "maintenance_rate": maintenance_rate,
            "rental_income_tax_rate": rental_income_tax_rate,
            "annual_depreciation": annual_depreciation,
            "app_mean": app_mean,
            "app_std": app_std,
            "rent_inflation": rent_inflation,
            "cost_basis": cost_basis,
            "rental_cap_gains_tax": rental_cap_gains_tax,
            "sale_year": rental_sale_year
        })

    # Primary Residence
    st.sidebar.header("Primary Residence")
    primary_residence_value = st.sidebar.number_input(
        "Primary Residence Value", 
        0, 10_000_000, 300_000, step=1_000,
        help="The current market value of your primary residence."
    )
    primary_mortgage_balance = st.sidebar.number_input(
        "Primary Mortgage Balance", 
        0, 10_000_000, 150_000, step=1_000,
        help="The remaining balance on the mortgage for your primary residence."
    )
    primary_mortgage_interest_rate = st.sidebar.slider(
        "Primary Mortgage Rate", 
        0.0, 0.2, 0.04, 0.005,
        help="The annual interest rate on the mortgage for your primary residence."
    )
    primary_mortgage_years_left = st.sidebar.number_input(
        "Years Left on Primary Mortgage", 
        0, 40, 25, step=1,
        help="The number of years remaining on the mortgage for your primary residence."
    )
    primary_appreciation_mean = st.sidebar.slider(
        "Primary Residence Appreciation Mean", 
        0.0, 0.2, 0.03, 0.01,
        help="The average annual appreciation rate of your primary residence's value."
    )
    primary_appreciation_std = st.sidebar.slider(
        "Primary Residence Appreciation StdDev", 
        0.0, 0.5, 0.1, 0.01,
        help="The standard deviation of your primary residence's annual appreciation rate."
    )

    # Luxury Expense
    st.sidebar.header("Luxury Expense (Replaces Land)")
    luxury_expense_amount = st.sidebar.number_input(
        "Luxury Expense Amount", 
        0, 5_000_000, 0, step=1_000,
        help="The amount of money you plan to spend on a one-time luxury expense."
    )
    luxury_expense_year = st.sidebar.number_input(
        "Year for Luxury Expense", 
        0, 100, 0, step=1,
        help="The year in which you plan to incur the luxury expense."
    )

    # Add inflation rate input
    st.sidebar.header("Economic Parameters")
    inflation_rate = st.sidebar.slider(
        "Inflation Rate", 
        0.01, 0.10, 0.03, 0.001,
        help="Annual inflation rate used to adjust retirement withdrawals."
    )

    # Add warnings about market assumptions
    st.sidebar.markdown("""
    ⚠️ **Important Notes:**
    - Past performance does not guarantee future returns
    - The default 7% return assumption is before inflation
    - Consider using more conservative estimates
    - Higher volatility early in retirement can significantly impact outcomes
    """)

    if st.button("Run Simulation"):
        with st.spinner("Simulating..."):
            results = run_simulation(
                n_sims,
                years_accum,
                years_retire,
                # Stock
                stock_initial,
                stock_annual_contribution,
                stock_expected_return,
                stock_volatility,
                stock_cap_gains_tax,
                stock_dividend_yield,
                stock_dividend_tax_rate,
                annual_withdrawal_stocks,
                # Rentals
                rentals_data,
                # Primary Residence
                primary_residence_value,
                primary_mortgage_balance,
                primary_mortgage_interest_rate,
                primary_mortgage_years_left,
                primary_appreciation_mean,
                primary_appreciation_std,
                # Luxury
                luxury_expense_amount,
                luxury_expense_year,
                inflation_rate=inflation_rate
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

        # Net Worth Components (Stacked) - moved to top
        st.subheader("Total Net Worth Components Over Time")
        summary_df = results["summary"]
        fig_components = px.area(summary_df, x="Year", 
            y=["Stock Value Median", "Primary Residence Equity Median"] + 
              [f"Rental{i+1} Equity Median" for i in range(num_rentals)],
            title="Net Worth Components (Median)",
            labels={"value": "Value ($)", "variable": "Component"}
        )
        st.plotly_chart(fig_components, use_container_width=True)

        # Plot: Stock Value
        st.subheader("Stock Portfolio Value Over Time")
        fig_stock = px.line(summary_df, x="Year", y="Stock Value Median", title="Stock Value (Median, 10th-90th)")
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Stock Value 10th").data)
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Stock Value 90th").data)
        fig_stock.data[1].update(fill=None)
        fig_stock.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        fig_stock.update_traces(name='10th percentile', selector=dict(name='Stock Value 10th'))
        fig_stock.update_traces(name='90th percentile', selector=dict(name='Stock Value 90th'))
        fig_stock.update_traces(name='Median', selector=dict(name='Stock Value Median'))
        st.plotly_chart(fig_stock, use_container_width=True)

        # Plot: Primary Residence Equity
        st.subheader("Primary Residence Equity Over Time")
        fig_primary_residence = px.line(summary_df, x="Year", y="Primary Residence Equity Median", title="Primary Residence Equity (Median, 10th-90th)")
        fig_primary_residence.add_traces(px.line(summary_df, x="Year", y="Primary Residence Equity 10th").data)
        fig_primary_residence.add_traces(px.line(summary_df, x="Year", y="Primary Residence Equity 90th").data)
        fig_primary_residence.data[1].update(fill=None)
        fig_primary_residence.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        fig_primary_residence.update_traces(name='10th percentile', selector=dict(name='Primary Residence Equity 10th'))
        fig_primary_residence.update_traces(name='90th percentile', selector=dict(name='Primary Residence Equity 90th'))
        fig_primary_residence.update_traces(name='Median', selector=dict(name='Primary Residence Equity Median'))
        st.plotly_chart(fig_primary_residence, use_container_width=True)

        # Plot: Stock Contributions vs Appreciation
        st.subheader("Stock Contributions vs Appreciation Over Time")
        fig_stock_breakdown = px.line(summary_df, x="Year", y=["Stock Contributions Median", "Stock Appreciation Median"], title="Stock Contributions vs Appreciation")
        st.plotly_chart(fig_stock_breakdown, use_container_width=True)

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


if __name__ == "__main__":
    main()
