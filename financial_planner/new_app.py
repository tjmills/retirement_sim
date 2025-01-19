import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def standard_mortgage_payment(principal, annual_interest_rate, mortgage_years):
    """
    Computes the monthly payment for a standard amortizing mortgage
    with a fixed rate and remaining term using the standard formula.
    """
    monthly_rate = annual_interest_rate / 12.0
    n_months = mortgage_years * 12

    if n_months == 0 or monthly_rate == 0:
        return 0.0

    # Payment = P * [r(1+r)^n] / [(1+r)^n - 1]
    payment = principal * (monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1)
    return payment


def update_rental_property_monthly(
    property_value,
    mortgage_balance,
    cost_basis,
    annual_interest_rate,
    monthly_payment,
    monthly_rent,
    vacancy_rate,
    property_tax_rate,
    maintenance_rate,
    rental_income_tax_rate,
    annual_depreciation,
    appreciation_mean,
    appreciation_std,
    rent_inflation,
    year_index=0
):
    """
    1-year update to the rental property with monthly mortgage amortization.
    This version cleanly separates the property value from the mortgage.
    """
    # 1) Property Appreciation
    annual_appreciation = np.random.normal(loc=appreciation_mean, scale=appreciation_std)
    new_property_value = property_value * (1 + annual_appreciation)
    
    # 2) Rent (inflation applied each year)
    current_monthly_rent = monthly_rent * ((1 + rent_inflation) ** year_index)
    gross_annual_rent = current_monthly_rent * 12 * (1 - vacancy_rate)
    
    # 3) Expenses: property tax + maintenance
    annual_property_tax = new_property_value * property_tax_rate
    annual_maintenance = new_property_value * maintenance_rate
    
    # 4) Monthly mortgage loop
    monthly_interest_rate = annual_interest_rate / 12.0
    new_mortgage_balance = mortgage_balance
    total_interest_paid = 0.0
    total_principal_paid = 0.0
    
    for _ in range(12):
        interest_for_month = new_mortgage_balance * monthly_interest_rate
        principal_for_month = monthly_payment - interest_for_month
        
        if principal_for_month < 0:
            principal_for_month = 0.0
            interest_for_month = monthly_payment
        
        if principal_for_month > new_mortgage_balance:
            principal_for_month = new_mortgage_balance
        
        new_mortgage_balance -= principal_for_month
        total_interest_paid += interest_for_month
        total_principal_paid += principal_for_month
    
    # 5) Net operating income before principal
    net_operating_income = gross_annual_rent - annual_property_tax - annual_maintenance - total_interest_paid
    
    # 6) Depreciation -> reduces taxable income
    taxable_income = net_operating_income - annual_depreciation
    if taxable_income < 0:
        taxable_income = 0.0  # ignoring negative/carry-forward
    
    # 7) Rental income tax
    rental_income_tax = taxable_income * rental_income_tax_rate
    
    # 8) Net rental cash flow after tax
    net_rental_cash_flow = net_operating_income - rental_income_tax
    
    # 9) Update cost basis (for capital gains)
    new_cost_basis = cost_basis - annual_depreciation
    if new_cost_basis < 0:
        new_cost_basis = 0.0
    
    return (new_property_value,
            new_mortgage_balance,
            new_cost_basis,
            net_rental_cash_flow,
            total_interest_paid,
            total_principal_paid)


def simulate_stock(
    current_stock_value,
    stock_annual_contribution,
    expected_return,
    volatility,
    capital_gains_tax_rate,
    dividend_yield,
    dividend_tax_rate,
    withdrawal=0.0
):
    """
    One-year update for a stock portfolio:
    - If withdrawal > current_stock_value, we clamp it, so the portfolio goes to zero.
    """
    # 1) Withdraw first (clamped)
    if withdrawal > current_stock_value:
        withdrawal = current_stock_value
    current_stock_value -= withdrawal
    
    # 2) Add annual contribution (accumulation only)
    current_stock_value += stock_annual_contribution
    
    # 3) Dividends
    dividends = current_stock_value * dividend_yield
    dividends_after_tax = dividends * (1 - dividend_tax_rate)
    current_stock_value += dividends_after_tax
    
    # 4) Random return
    annual_return = np.random.normal(loc=expected_return, scale=volatility)
    current_stock_value *= (1 + annual_return)
    
    # Ensure no negative
    if current_stock_value < 0:
        current_stock_value = 0
    
    return current_stock_value


def simulate_land(
    current_land_value,
    land_annual_contribution,
    expected_return,
    volatility,
    land_property_tax_rate,
    land_capital_gains_tax_rate,
    withdrawal=0.0
):
    """
    One-year update for land. 
    - If withdrawal > current_land_value, clamp it to bring land to zero.
    """
    # 1) Withdraw
    if withdrawal > current_land_value:
        withdrawal = current_land_value
    current_land_value -= withdrawal
    
    # 2) Add annual contribution
    current_land_value += land_annual_contribution
    
    # 3) Random return
    annual_return = np.random.normal(loc=expected_return, scale=volatility)
    current_land_value *= (1 + annual_return)
    
    # No forced negative check, but do it anyway for safety
    if current_land_value < 0:
        current_land_value = 0
    
    return current_land_value


def rebalance_stock_land(
    stock_val,
    land_val,
    target_allocations,
    stock_capital_gains_tax_rate,
    land_capital_gains_tax_rate
):
    """
    Rebalance only stock & land. 
    If rebalancing is on, we adjust to the target ratio.
    """
    total_val = stock_val + land_val
    if total_val <= 0:
        return stock_val, land_val
    
    desired_stock = total_val * target_allocations['stock']
    desired_land = total_val * target_allocations['land']
    
    # If stock > desired => sell some stock
    if stock_val > desired_stock:
        amount_sold = stock_val - desired_stock
        after_tax = amount_sold * (1 - stock_capital_gains_tax_rate)
        stock_val = desired_stock
        # Buy land if needed
        if land_val < desired_land:
            needed = desired_land - land_val
            purchase = min(needed, after_tax)
            land_val += purchase
            after_tax -= purchase
    
    # If stock < desired => buy more stock from land
    elif stock_val < desired_stock:
        needed = desired_stock - stock_val
        if land_val > desired_land:
            land_sold = land_val - desired_land
            after_tax_land = land_sold * (1 - land_capital_gains_tax_rate)
            to_sell = min(land_sold, needed)
            ratio = to_sell / land_sold if land_sold != 0 else 0
            proceeds = after_tax_land * ratio
            land_val -= to_sell
            stock_val += proceeds
    
    return stock_val, land_val


def run_simulation(
    n_sims,
    years_accum,
    years_retire,
    # Stocks
    stock_initial,
    stock_annual_contribution,
    stock_expected_return,
    stock_volatility,
    stock_cap_gains_tax,
    stock_dividend_yield,
    stock_dividend_tax_rate,
    annual_withdrawal_stocks,
    # Land
    land_initial,
    land_annual_contribution,
    land_expected_return,
    land_volatility,
    land_prop_tax,
    land_cap_gains_tax,
    annual_withdrawal_land,
    # Rental
    property_current_value,
    mortgage_current_balance,
    mortgage_interest_rate,
    mortgage_term_years,
    rental_cost_basis,
    property_tax_rate,
    maintenance_rate,
    vacancy_rate,
    rental_income_tax_rate,
    rent_inflation,
    app_mean,
    app_std,
    rental_depr_years,
    rental_sale_year,
    rental_cap_gains_tax_rate,
    monthly_rent_base,
    # Rebalancing
    do_rebalance,
    target_allocations
):
    """
    Runs multiple Monte Carlo simulations for Stocks, Land, and a single Rental
    property. The rental property can be sold in a discrete year, with net
    proceeds moving to stocks. Mortgage is handled monthly.
    
    The 'monthly_rent_base' is the starting monthly rent. We'll inflate it each year.
    """
    total_years = years_accum + years_retire
    
    # 1) Depreciation
    if rental_depr_years > 0:
        annual_depreciation = rental_cost_basis / rental_depr_years
    else:
        annual_depreciation = 0.0
    
    # 2) Mortgage Payment (standard formula)
    monthly_payment = standard_mortgage_payment(
        principal=mortgage_current_balance,
        annual_interest_rate=mortgage_interest_rate,
        mortgage_years=mortgage_term_years
    )
    
    all_sims = []
    
    for sim in range(n_sims):
        # Initialize
        current_stock_value = stock_initial
        current_land_value = land_initial
        
        property_value = property_current_value
        mortgage_balance = mortgage_current_balance
        current_cost_basis = rental_cost_basis
        
        years_list = []
        stock_vals = []
        land_vals = []
        rental_equities = []
        mortgage_balances = []
        rental_cash_flows = []
        
        for year in range(total_years):
            year_label = year + 1
            is_retirement = (year >= years_accum)
            
            # 1) Update Rental if not sold
            if property_value > 0:
                (new_prop_val,
                 new_mort_balance,
                 new_cost_basis,
                 net_rental_cf,
                 _interest_paid,
                 _principal_paid) = update_rental_property_monthly(
                    property_value=property_value,
                    mortgage_balance=mortgage_balance,
                    cost_basis=current_cost_basis,
                    annual_interest_rate=mortgage_interest_rate,
                    monthly_payment=monthly_payment,
                    monthly_rent=monthly_rent_base,  # We'll inflate inside the function
                    vacancy_rate=vacancy_rate,
                    property_tax_rate=property_tax_rate,
                    maintenance_rate=maintenance_rate,
                    rental_income_tax_rate=rental_income_tax_rate,
                    annual_depreciation=annual_depreciation,
                    appreciation_mean=app_mean,
                    appreciation_std=app_std,
                    rent_inflation=rent_inflation,
                    year_index=year
                )
                property_value = new_prop_val
                mortgage_balance = new_mort_balance
                current_cost_basis = new_cost_basis
                rental_cash_flows.append(net_rental_cf)
            else:
                # Already sold
                net_rental_cf = 0
                rental_cash_flows.append(0)
            
            # 2) Discrete Sale
            if (year_label == rental_sale_year) and (property_value > 0):
                # Gains = property_value - cost_basis
                gains = property_value - current_cost_basis
                if gains < 0:
                    gains = 0
                cap_gains_tax = gains * rental_cap_gains_tax_rate
                net_sale_proceeds = (property_value - mortgage_balance) - cap_gains_tax
                if net_sale_proceeds < 0:
                    net_sale_proceeds = 0
                
                # Add to stocks
                current_stock_value += net_sale_proceeds
                
                # Zero out rental
                property_value = 0
                mortgage_balance = 0
                current_cost_basis = 0
            
            # 3) Stocks
            stock_contribution = stock_annual_contribution if not is_retirement else 0
            stock_withdrawal = annual_withdrawal_stocks if is_retirement else 0
            current_stock_value = simulate_stock(
                current_stock_value,
                stock_contribution,
                stock_expected_return,
                stock_volatility,
                stock_cap_gains_tax,
                stock_dividend_yield,
                stock_dividend_tax_rate,
                withdrawal=stock_withdrawal
            )
            
            # 4) Land
            land_contribution = land_annual_contribution if not is_retirement else 0
            land_withdrawal = annual_withdrawal_land if is_retirement else 0
            current_land_value = simulate_land(
                current_land_value,
                land_contribution,
                land_expected_return,
                land_volatility,
                land_prop_tax,
                land_cap_gains_tax,
                withdrawal=land_withdrawal
            )
            
            # 5) Rebalance stock & land
            if do_rebalance:
                current_stock_value, current_land_value = rebalance_stock_land(
                    stock_val=current_stock_value,
                    land_val=current_land_value,
                    target_allocations=target_allocations,
                    stock_capital_gains_tax_rate=stock_cap_gains_tax,
                    land_capital_gains_tax_rate=land_cap_gains_tax
                )
            
            years_list.append(year_label)
            stock_vals.append(current_stock_value)
            land_vals.append(current_land_value)
            net_equity = max(property_value - mortgage_balance, 0)
            rental_equities.append(net_equity)
            mortgage_balances.append(mortgage_balance)
        
        # Compile DF for this simulation
        sim_df = pd.DataFrame({
            'Year': years_list,
            'Stock Value': stock_vals,
            'Land Value': land_vals,
            'Rental Equity': rental_equities,
            'Mortgage Balance': mortgage_balances,
            'Rental Cash Flow': rental_cash_flows
        })
        all_sims.append(sim_df)
    
    # Combine results for summary stats
    combined_df = None
    for i, df_i in enumerate(all_sims):
        rename_map = {
            'Stock Value': f'Stock Value {i}',
            'Land Value': f'Land Value {i}',
            'Rental Equity': f'Rental Equity {i}',
            'Mortgage Balance': f'Mortgage Balance {i}',
            'Rental Cash Flow': f'Rental Cash Flow {i}'
        }
        temp = df_i.rename(columns=rename_map)
        if combined_df is None:
            combined_df = temp
        else:
            combined_df = pd.merge(combined_df, temp, on='Year', how='inner')
    
    summary_df = pd.DataFrame({'Year': combined_df['Year']})
    for asset in ['Stock Value', 'Land Value', 'Rental Equity', 'Mortgage Balance', 'Rental Cash Flow']:
        asset_cols = [c for c in combined_df.columns if c.startswith(asset)]
        summary_df[f'{asset} Mean'] = combined_df[asset_cols].mean(axis=1)
        summary_df[f'{asset} Median'] = combined_df[asset_cols].median(axis=1)
        summary_df[f'{asset} 10th'] = combined_df[asset_cols].quantile(0.1, axis=1)
        summary_df[f'{asset} 90th'] = combined_df[asset_cols].quantile(0.9, axis=1)
    
    return {
        'all_sims': all_sims,
        'combined': combined_df,
        'summary': summary_df
    }


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

def main():
    st.set_page_config(page_title="Full Portfolio with Proper Drawdown", layout="wide")
    st.title("Portfolio Simulation with Correct Stock Drawdown and Separate Plots")
    
    st.markdown("""
    This app demonstrates:
    1. **Stocks** (properly hitting zero if withdrawals exceed current value).
    2. **Land** (similar logic).
    3. **Rental Property** with monthly mortgage amortization, 
       a discrete sale, and proceeds moving to stocks.
    4. **Separate plots** for each asset (mean & percentile bands).
    
    **Try** different scenarios:
    - Large withdrawals on stocks,
    - Selling rental at a certain year,
    - Buying land or using it as a “luxury item” with no income,
    - Rebalancing or not rebalancing.
    """)

    st.sidebar.header("Global Settings")
    n_sims = st.sidebar.number_input("Number of Simulations", 1, 5000, 200)
    years_accum = st.sidebar.number_input("Years of Accumulation", 1, 50, 20)
    years_retire = st.sidebar.number_input("Years of Retirement", 1, 50, 30)
    
    # Stocks
    st.sidebar.header("Stock Parameters")
    stock_initial = st.sidebar.number_input("Initial Stock Value", 0, 10_000_000, 50_000, step=1_000)
    stock_annual_contribution = st.sidebar.number_input("Annual Stock Contribution (Accum)", 0, 1_000_000, 10_000, step=1_000)
    stock_expected_return = st.sidebar.slider("Stock Return (mean)", 0.0, 0.2, 0.07, 0.01)
    stock_volatility = st.sidebar.slider("Stock Volatility (std dev)", 0.0, 0.5, 0.15, 0.01)
    stock_cap_gains_tax = st.sidebar.slider("Stock Cap Gains Tax Rate", 0.0, 0.5, 0.15, 0.01)
    stock_dividend_yield = st.sidebar.slider("Dividend Yield (Stocks)", 0.0, 0.1, 0.02, 0.01)
    stock_dividend_tax_rate = st.sidebar.slider("Dividend Tax Rate", 0.0, 0.5, 0.15, 0.01)
    annual_withdrawal_stocks = st.sidebar.number_input("Stock Withdrawal (Retirement)", 0, 500_000, 20_000, step=1_000)
    
    # Land
    st.sidebar.header("Land Parameters")
    land_initial = st.sidebar.number_input("Initial Land Value", 0, 10_000_000, 30_000, step=1_000)
    land_annual_contribution = st.sidebar.number_input("Annual Land Contribution (Accum)", 0, 1_000_000, 5_000, step=1_000)
    land_expected_return = st.sidebar.slider("Land Return (mean)", 0.0, 0.2, 0.03, 0.01)
    land_volatility = st.sidebar.slider("Land Volatility (std dev)", 0.0, 0.5, 0.1, 0.01)
    land_prop_tax = st.sidebar.slider("Land Prop Tax Rate", 0.0, 0.05, 0.005, 0.001)
    land_cap_gains_tax = st.sidebar.slider("Land Cap Gains Tax", 0.0, 0.5, 0.15, 0.01)
    annual_withdrawal_land = st.sidebar.number_input("Land Withdrawal (Retirement)", 0, 500_000, 5_000, step=1_000)
    
    # Rental
    st.sidebar.header("Rental Parameters")
    property_current_value = st.sidebar.number_input("Current Property Value", 0, 10_000_000, 200_000, step=1_000)
    mortgage_current_balance = st.sidebar.number_input("Current Mortgage Balance", 0, 10_000_000, 120_000, step=1_000)
    mortgage_interest_rate = st.sidebar.slider("Mortgage Interest Rate", 0.0, 0.2, 0.04, 0.01)
    mortgage_term_years = st.sidebar.number_input("Mortgage Years Left", 0, 40, 25, step=1)
    
    rental_cost_basis = st.sidebar.number_input("Rental Cost Basis", 0, 10_000_000, 180_000, step=1_000)
    property_tax_rate = st.sidebar.slider("Rental Prop Tax Rate", 0.0, 0.05, 0.01, 0.001)
    maintenance_rate = st.sidebar.slider("Maintenance Rate (Rental)", 0.0, 0.05, 0.01, 0.001)
    vacancy_rate = st.sidebar.slider("Vacancy Rate", 0.0, 1.0, 0.05, 0.01)
    rental_income_tax_rate = st.sidebar.slider("Rental Income Tax Rate", 0.0, 0.5, 0.2, 0.01)
    rent_inflation = st.sidebar.slider("Annual Rent Inflation", 0.0, 0.1, 0.02, 0.01)
    app_mean = st.sidebar.slider("Rental Appreciation (mean)", 0.0, 0.2, 0.03, 0.01)
    app_std = st.sidebar.slider("Rental Appreciation (std dev)", 0.0, 0.5, 0.1, 0.01)
    rental_depr_years = st.sidebar.number_input("Depreciation Years", 0, 50, 27, step=1)
    rental_sale_year = st.sidebar.number_input("Discrete Sale Year (Rental)", 0, 100, 0, step=1)
    rental_cap_gains_tax = st.sidebar.slider("Rental Cap Gains Tax Rate", 0.0, 0.5, 0.15, 0.01)
    
    monthly_rent_base = st.sidebar.number_input("Base Monthly Rent Today", 0, 50_000, 1500, step=100)
    
    # Rebalancing
    st.sidebar.header("Rebalancing")
    do_rebalance = st.sidebar.checkbox("Enable Rebalancing?", value=False)
    st.sidebar.caption("Target % for Stock & Land must sum > 0; we will normalize.")
    target_stock = st.sidebar.slider("Target % Stock", 0.0, 1.0, 0.5, 0.05)
    target_land = st.sidebar.slider("Target % Land", 0.0, 1.0, 0.5, 0.05)
    total_alloc = target_stock + target_land
    if total_alloc <= 0:
        target_allocations = {'stock': 0.5, 'land': 0.5}
    else:
        target_allocations = {
            'stock': target_stock / total_alloc,
            'land': target_land / total_alloc
        }
    
    if st.button("Run Simulation"):
        with st.spinner("Running Monte Carlo..."):
            results = run_simulation(
                n_sims,
                years_accum,
                years_retire,
                # Stocks
                stock_initial,
                stock_annual_contribution,
                stock_expected_return,
                stock_volatility,
                stock_cap_gains_tax,
                stock_dividend_yield,
                stock_dividend_tax_rate,
                annual_withdrawal_stocks,
                # Land
                land_initial,
                land_annual_contribution,
                land_expected_return,
                land_volatility,
                land_prop_tax,
                land_cap_gains_tax,
                annual_withdrawal_land,
                # Rental
                property_current_value,
                mortgage_current_balance,
                mortgage_interest_rate,
                mortgage_term_years,
                rental_cost_basis,
                property_tax_rate,
                maintenance_rate,
                vacancy_rate,
                rental_income_tax_rate,
                rent_inflation,
                app_mean,
                app_std,
                rental_depr_years,
                rental_sale_year,
                rental_cap_gains_tax,
                monthly_rent_base,
                # Rebalancing
                do_rebalance,
                target_allocations
            )
        
        st.success("Simulation complete!")
        summary_df = results['summary']
        
        st.subheader("Summary of Simulations (Mean, Median, 10th, 90th)")
        st.dataframe(summary_df)

        # --- Plots ---
        # 1) Stock Value plot with percentile shading
        st.subheader("Stock Value Over Time")
        fig_stock = px.line(summary_df, x="Year", y="Stock Value Median", title="Stock Value (Median with 10-90th)")
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Stock Value 10th").data)
        fig_stock.add_traces(px.line(summary_df, x="Year", y="Stock Value 90th").data)
        fig_stock.data[1].update(fill=None)
        fig_stock.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        fig_stock.update_layout(showlegend=True)
        fig_stock.update_traces(name='10th percentile', selector=dict(name='Stock Value 10th'))
        fig_stock.update_traces(name='90th percentile', selector=dict(name='Stock Value 90th'))
        fig_stock.update_traces(name='Median', selector=dict(name='Stock Value Median'))
        st.plotly_chart(fig_stock, use_container_width=True)
        
        # 2) Rental Equity with percentile shading
        st.subheader("Rental Equity Over Time")
        fig_rent = px.line(summary_df, x="Year", y="Rental Equity Median", title="Rental Equity (Median with 10-90th)")
        fig_rent.add_traces(px.line(summary_df, x="Year", y="Rental Equity 10th").data)
        fig_rent.add_traces(px.line(summary_df, x="Year", y="Rental Equity 90th").data)
        fig_rent.data[1].update(fill=None)
        fig_rent.data[2].update(fill='tonexty', fillcolor='rgba(100,0,80,0.2)')
        fig_rent.update_layout(showlegend=True)
        fig_rent.update_traces(name='10th percentile', selector=dict(name='Rental Equity 10th'))
        fig_rent.update_traces(name='90th percentile', selector=dict(name='Rental Equity 90th'))
        fig_rent.update_traces(name='Median', selector=dict(name='Rental Equity Median'))
        st.plotly_chart(fig_rent, use_container_width=True)
        
        # 3) Land Value with percentile shading
        st.subheader("Land Value Over Time")
        fig_land = px.line(summary_df, x="Year", y="Land Value Median", title="Land Value (Median with 10-90th)")
        fig_land.add_traces(px.line(summary_df, x="Year", y="Land Value 10th").data)
        fig_land.add_traces(px.line(summary_df, x="Year", y="Land Value 90th").data)
        fig_land.data[1].update(fill=None)
        fig_land.data[2].update(fill='tonexty', fillcolor='rgba(100,100,0,0.2)')
        fig_land.update_layout(showlegend=True)
        fig_land.update_traces(name='10th percentile', selector=dict(name='Land Value 10th'))
        fig_land.update_traces(name='90th percentile', selector=dict(name='Land Value 90th'))
        fig_land.update_traces(name='Median', selector=dict(name='Land Value Median'))
        st.plotly_chart(fig_land, use_container_width=True)
        
        # 4) Mortgage Balance
        st.subheader("Mortgage Balance Over Time (Mean)")
        fig_mort = px.line(summary_df, x="Year", y="Mortgage Balance Mean", title="Mean Mortgage Balance")
        fig_mort.update_layout(yaxis_title="Mortgage Balance ($)")
        st.plotly_chart(fig_mort, use_container_width=True)
        
        st.markdown("""
        ### How It Works
        - **Stock withdrawals** are explicitly clamped so if the withdrawal (e.g., \$100k) 
          is bigger than the current stock value (e.g., \$40k), the portfolio hits 0.
        - The same logic applies to **land** drawdowns.
        - The **rental** is updated monthly for its mortgage. If you sell it at `rental_sale_year`, 
          net proceeds go into **stocks**.
        - **Rebalancing** (if enabled) can move money between stocks & land 
          after withdrawals and growth each year.

        Adjust the sidebars to see how different assumptions 
        (tax rates, returns, withdrawals, etc.) affect your portfolio!
        """)


if __name__ == "__main__":
    main()
