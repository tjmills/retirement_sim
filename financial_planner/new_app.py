import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def calculate_rental_cash_flow(
    property_value,
    monthly_rent,
    vacancy_rate,
    property_tax_rate,
    maintenance_rate,
    mortgage_payment,
    rental_income_tax_rate,
    rent_inflation,
    year_index=0
):
    """
    Calculate the net rental cash flow for a given year.
    - property_value: current property value
    - monthly_rent: monthly rent in the current year
    - vacancy_rate: fraction of the year property is vacant
    - property_tax_rate: % of property value paid in property tax
    - maintenance_rate: % of property value used for maintenance
    - mortgage_payment: annual mortgage payment (interest + principal)
    - rental_income_tax_rate: fraction of net rental income taxed
    - rent_inflation: inflation rate applied to rent each year
    - year_index: optional, for applying incremental rent inflation

    Returns:
        net_cash_flow (float): net cash flow after all expenses and taxes
        updated_monthly_rent (float): updated rent for next year
    """
    # Adjust rent for inflation
    effective_monthly_rent = monthly_rent * ((1 + rent_inflation) ** year_index)
    
    # Annual rent collected minus vacancy
    gross_annual_rent = effective_monthly_rent * 12 * (1 - vacancy_rate)
    
    # Expenses
    annual_property_tax = property_value * property_tax_rate
    annual_maintenance = property_value * maintenance_rate
    
    # Net operating income before mortgage
    net_operating_income = gross_annual_rent - annual_property_tax - annual_maintenance
    
    # Subtract mortgage payment
    # (For simplicity, assume a fixed annual mortgage payment, ignoring changing principal)
    net_income_before_tax = net_operating_income - mortgage_payment
    
    # Rental income tax
    # (If net_income_before_tax is negative, assume no tax credit or offset to keep it simple)
    if net_income_before_tax > 0:
        net_income_after_tax = net_income_before_tax * (1 - rental_income_tax_rate)
    else:
        net_income_after_tax = net_income_before_tax
    
    return net_income_after_tax, effective_monthly_rent


def simulate_stock(
    current_stock_value,
    stock_annual_contribution,
    expected_return,
    volatility,
    capital_gains_tax_rate,
    dividend_yield,
    dividend_tax_rate,
    do_rebalance=False,
    rebalancing_target=0.0
):
    """
    Simulate one year of stock returns using a random draw (Monte Carlo) 
    from a normal distribution (expected_return +/- volatility).
    Also accounts for dividends, taxes, and annual contributions.

    Returns: new_stock_value
    """
    # Draw a random return from normal distribution
    annual_return = np.random.normal(loc=expected_return, scale=volatility)
    
    # Add annual contribution first (pre-growth)
    # (In a real scenario, you might add monthly or quarterly, but we'll keep it simple.)
    current_stock_value += stock_annual_contribution
    
    # Calculate dividends (as a fraction of current value), then pay dividend tax
    dividends = current_stock_value * dividend_yield
    dividends_after_tax = dividends * (1 - dividend_tax_rate)
    
    # Add dividends to the portfolio (assuming reinvestment)
    current_stock_value += dividends_after_tax
    
    # Grow the portfolio by (annual_return). 
    # For simplicity, treat total return as price appreciation above the dividends.
    current_stock_value *= (1 + annual_return)
    
    # If do_rebalance is True, we won't apply capital gains tax here on partial sales 
    # because rebalancing is handled at the portfolio level. 
    # So we skip capital gains tax *during* the year. 
    # If you wanted to model annual tax realization, you could do so here.
    
    return current_stock_value


def simulate_land(
    current_land_value,
    land_annual_contribution,
    expected_return,
    volatility,
    property_tax_rate_land,
    land_inflation,
    capital_gains_tax_rate,
    do_rebalance=False,
    rebalancing_target=0.0
):
    """
    Simulate one year of land value changes using a random draw
    from a normal distribution. Factor in annual contributions 
    (e.g., additional purchases of land), property tax, etc.
    """
    # Draw a random return from normal distribution
    annual_return = np.random.normal(loc=expected_return, scale=volatility)
    
    # Add annual contribution to land (buy more land or invest in improvements).
    current_land_value += land_annual_contribution
    
    # For simplicity, we model property tax as a % of the land value each year
    # and assume it's paid in cash from outside the land value.
    # Alternatively, you could subtract it from 'current_land_value' if the property
    # has to pay from the asset itself.
    property_tax = current_land_value * property_tax_rate_land
    
    # We'll assume the user pays property_tax from an external source, 
    # so we won't reduce current_land_value. 
    # If you want the land value to reduce from paying property taxes, uncomment:
    # current_land_value -= property_tax
    
    # Grow land by random return
    current_land_value *= (1 + annual_return)
    
    # No direct annual income from land (unless you model farmland rent or timber sales),
    # so no immediate income tax. 
    # If you rebalance land (which is unusual in practice), 
    # you'd consider capital gains upon sale.
    
    return current_land_value


def rebalance_portfolio(
    stock_val,
    rental_val,
    land_val,
    target_allocations,
    stock_capital_gains_tax_rate,
    land_capital_gains_tax_rate
):
    """
    Rebalance the portfolio to the target allocations among stock, rental, and land.
    For demonstration, we assume:
      - We can freely rebalance the real estate (selling partial property) 
        and realize capital gains taxes on any 'gains'.
      - We do the same with land.
    This is highly simplified and not realistic for illiquid assets.
    """
    total_val = stock_val + rental_val + land_val
    if total_val <= 0:
        return stock_val, rental_val, land_val  # Nothing to rebalance if everything is zero or negative
    
    # Current allocations
    # For simplicity, we won't track original cost basis. 
    # We'll assume that any sale is 100% capital gain (which is unrealistic).
    # Then we apply capital gains tax if we reduce an asset.
    
    desired_stock = total_val * target_allocations['stock']
    desired_rental = total_val * target_allocations['rental']
    desired_land = total_val * target_allocations['land']
    
    # Rebalance stock
    if stock_val > desired_stock:
        # Selling some stock, assume all gains are taxed
        # (This is a simplification ignoring cost basis.)
        amount_sold = stock_val - desired_stock
        amount_after_tax = amount_sold * (1 - stock_capital_gains_tax_rate)
        stock_val = desired_stock
        # We'll redistribute 'amount_after_tax' among rental and land if needed
    else:
        amount_sold = 0
        amount_after_tax = 0
    
    # Rebalance land
    if land_val > desired_land:
        # Selling some land
        land_sold = land_val - desired_land
        land_after_tax = land_sold * (1 - land_capital_gains_tax_rate)
        land_val = desired_land
        # We'll redistribute 'land_after_tax' among stock/rental if needed
    else:
        land_sold = 0
        land_after_tax = 0
    
    # For demonstration, we do not "sell" rental property to rebalance. 
    # Because partial property sales are tricky. 
    # We'll keep rental as-is. 
    # Alternatively, we could do the same approach for rental, but it's not typical.
    # If you want to do it, you'd replicate the logic above with capital gains for the property.
    
    # If stock_val < desired_stock, we can buy more stock from extra cash
    # from the sold stock or sold land. This is a bit contradictory (why sell stock if it's below target?)
    # so let's keep it super simple.
    
    # In a real scenario, you combine all "excess" from assets above target 
    # and redistribute to assets below target proportionally.
    
    # Let's combine the after_tax proceeds
    total_after_tax_proceeds = amount_after_tax + land_after_tax
    
    # If we want to push the rental to its target, we'd need to model selling or buying property 
    # with capital gains. We'll skip. We'll only rebalance into stock or land.
    
    # If stock_val < desired_stock, buy up to desired using the proceeds
    if stock_val < desired_stock and total_after_tax_proceeds > 0:
        needed = desired_stock - stock_val
        purchase = min(needed, total_after_tax_proceeds)
        stock_val += purchase
        total_after_tax_proceeds -= purchase
    
    # If land_val < desired_land, buy up to desired with leftover proceeds
    if land_val < desired_land and total_after_tax_proceeds > 0:
        needed = desired_land - land_val
        purchase = min(needed, total_after_tax_proceeds)
        land_val += purchase
        total_after_tax_proceeds -= purchase
    
    # If there's still leftover, we won't do anything with it for simplicity.
    
    return stock_val, rental_val, land_val


def run_simulation(
    n_simulations,
    years_accumulation,
    years_retirement,
    # Stock
    stock_initial,
    stock_annual_contribution,
    stock_expected_return,
    stock_volatility,
    stock_capital_gains_tax_rate,
    stock_dividend_yield,
    stock_dividend_tax_rate,
    # Rental
    rental_initial_equity,
    property_initial_value,
    monthly_rent,
    vacancy_rate,
    property_tax_rate,
    maintenance_rate,
    mortgage_payment,
    rental_income_tax_rate,
    rent_inflation,
    property_appreciation_mean,
    property_appreciation_std,
    # Land
    land_initial,
    land_annual_contribution,
    land_expected_return,
    land_volatility,
    land_property_tax_rate,
    land_capital_gains_tax_rate,
    # Rebalancing
    do_rebalance,
    target_allocations,
    # Withdrawal strategy
    annual_withdrawal_stocks,
    annual_withdrawal_rental,
    annual_withdrawal_land,
    # Global inflation or separate inflation for each? (Here we do separate for rent, land, etc.)
):
    """
    Runs a Monte Carlo simulation for the specified number of simulations.
    Returns a dictionary of result DataFrames for average, 10th percentile, 90th percentile, etc.
    """
    
    # We will store each simulation's yearly portfolio values in a list of DataFrames or arrays.
    all_simulations = []
    
    for sim in range(n_simulations):
        # Initialize asset values
        current_stock_value = stock_initial
        current_rental_equity = rental_initial_equity  # Not used heavily, but could track mortgage payoff
        current_property_value = property_initial_value
        current_land_value = land_initial
        
        # Lists to track annual values
        years_list = []
        stock_values = []
        rental_values = []  # We'll treat "rental value" as the property value plus or minus equity
        land_values = []
        
        effective_monthly_rent = monthly_rent
        
        # 1) Accumulation phase
        for year in range(years_accumulation):
            years_list.append(year + 1)
            
            # Simulate rental property appreciation
            # (We do this before cash flow so property_value is updated each year)
            property_return = np.random.normal(loc=property_appreciation_mean, scale=property_appreciation_std)
            # Grow property value
            current_property_value *= (1 + property_return)
            
            # Calculate net rental cash flow
            net_cash_flow_rent, effective_rent_this_year = calculate_rental_cash_flow(
                property_value=current_property_value,
                monthly_rent=monthly_rent,
                vacancy_rate=vacancy_rate,
                property_tax_rate=property_tax_rate,
                maintenance_rate=maintenance_rate,
                mortgage_payment=mortgage_payment,
                rental_income_tax_rate=rental_income_tax_rate,
                rent_inflation=rent_inflation,
                year_index=year
            )
            
            # We assume net_cash_flow_rent is "invested" in the rental equity (paying down mortgage or 
            # kept as separate cash?). For simplicity, we add it to our "rental value" or "equity".
            # If positive, it grows equity; if negative, it reduces equity (not fully realistic).
            # We'll treat "rental_value" as the property value for clarity, 
            # but you can track equity separately.
            current_rental_equity += net_cash_flow_rent
            
            # Stock simulation
            current_stock_value = simulate_stock(
                current_stock_value,
                stock_annual_contribution,
                stock_expected_return,
                stock_volatility,
                stock_capital_gains_tax_rate,
                stock_dividend_yield,
                stock_dividend_tax_rate,
                do_rebalance=do_rebalance
            )
            
            # Land simulation
            current_land_value = simulate_land(
                current_land_value,
                land_annual_contribution,
                land_expected_return,
                land_volatility,
                land_property_tax_rate,
                land_inflation=0.0,  # If you want land-specific inflation, incorporate it here or in return
                capital_gains_tax_rate=land_capital_gains_tax_rate,
                do_rebalance=do_rebalance
            )
            
            # Rebalancing
            if do_rebalance:
                (current_stock_value, 
                 current_rental_equity, 
                 current_land_value) = rebalance_portfolio(
                     stock_val=current_stock_value,
                     rental_val=current_rental_equity,  # we treat equity as "value" for rebalancing
                     land_val=current_land_value,
                     target_allocations=target_allocations,
                     stock_capital_gains_tax_rate=stock_capital_gains_tax_rate,
                     land_capital_gains_tax_rate=land_capital_gains_tax_rate
                 )
            
            # For reporting: treat "rental value" as property value + rental equity
            # or simply as property_value if we consider that the main asset.
            # We'll do property_value + rental_equity for demonstration.
            total_rental_value = current_property_value + current_rental_equity
            
            stock_values.append(current_stock_value)
            rental_values.append(total_rental_value)
            land_values.append(current_land_value)
        
        # 2) Retirement (drawdown) phase
        for year in range(years_accumulation, years_accumulation + years_retirement):
            years_list.append(year + 1)
            
            # The property continues to appreciate randomly
            property_return = np.random.normal(loc=property_appreciation_mean, scale=property_appreciation_std)
            current_property_value *= (1 + property_return)
            
            # Rental cash flow (still receiving rent)
            net_cash_flow_rent, _ = calculate_rental_cash_flow(
                property_value=current_property_value,
                monthly_rent=monthly_rent,
                vacancy_rate=vacancy_rate,
                property_tax_rate=property_tax_rate,
                maintenance_rate=maintenance_rate,
                mortgage_payment=mortgage_payment,
                rental_income_tax_rate=rental_income_tax_rate,
                rent_inflation=rent_inflation,
                year_index=year  # continuing the inflation
            )
            
            current_rental_equity += net_cash_flow_rent
            
            # Withdraw from each portfolio portion
            # (For a real plan, you might specify a total withdrawal across the entire portfolio 
            #   or apply a "bucket strategy." We'll do separate withdrawals for simplicity.)
            
            # STOCK withdrawal
            withdrawal_stock = annual_withdrawal_stocks
            # If there's not enough stock to withdraw, clamp it to the stock_value
            if withdrawal_stock > current_stock_value:
                withdrawal_stock = current_stock_value
            current_stock_value -= withdrawal_stock
            
            # Land withdrawal
            withdrawal_land = annual_withdrawal_land
            if withdrawal_land > current_land_value:
                withdrawal_land = current_land_value
            current_land_value -= withdrawal_land
            
            # For rental, we might assume we withdraw from the net rental cash flow each year
            # rather than the property value. Or we can force property sales. 
            # We'll do a simple approach: a "rental withdrawal" from the rental_equity if positive.
            withdrawal_rental = annual_withdrawal_rental
            if withdrawal_rental > current_rental_equity:
                withdrawal_rental = current_rental_equity
            current_rental_equity -= withdrawal_rental
            
            # Now apply returns/growth for stocks in retirement (still subject to random returns)
            current_stock_value = simulate_stock(
                current_stock_value,
                0,  # no more contributions
                stock_expected_return, 
                stock_volatility,
                stock_capital_gains_tax_rate,
                stock_dividend_yield,
                stock_dividend_tax_rate
            )
            
            # Land continues to appreciate
            current_land_value = simulate_land(
                current_land_value,
                0,  # no more contributions
                land_expected_return,
                land_volatility,
                land_property_tax_rate,
                0.0,  # land_inflation not used in this example, incorporate if desired
                land_capital_gains_tax_rate
            )
            
            # Rebalancing in retirement?
            if do_rebalance:
                (current_stock_value, 
                 current_rental_equity, 
                 current_land_value) = rebalance_portfolio(
                     stock_val=current_stock_value,
                     rental_val=current_rental_equity,
                     land_val=current_land_value,
                     target_allocations=target_allocations,
                     stock_capital_gains_tax_rate=stock_capital_gains_tax_rate,
                     land_capital_gains_tax_rate=land_capital_gains_tax_rate
                 )
            
            total_rental_value = current_property_value + current_rental_equity
            stock_values.append(current_stock_value)
            rental_values.append(total_rental_value)
            land_values.append(current_land_value)
        
        # Compile into a DataFrame for this simulation
        sim_df = pd.DataFrame({
            'Year': years_list,
            'Stock Value': stock_values,
            'Rental Value': rental_values,
            'Land Value': land_values
        })
        all_simulations.append(sim_df)
    
    # After all simulations, we want to combine the results to show 
    # average (mean), median, percentile, etc.
    # We'll assume each sim_df has the same length = years_accumulation + years_retirement
    combined_df = None
    for i, sim_df in enumerate(all_simulations):
        if combined_df is None:
            combined_df = sim_df.copy()
            combined_df.rename(columns={
                'Stock Value': f'Stock Value {i}',
                'Rental Value': f'Rental Value {i}',
                'Land Value': f'Land Value {i}'
            }, inplace=True)
        else:
            temp = sim_df.copy()
            temp.rename(columns={
                'Stock Value': f'Stock Value {i}',
                'Rental Value': f'Rental Value {i}',
                'Land Value': f'Land Value {i}'
            }, inplace=True)
            combined_df = pd.merge(combined_df, temp, on='Year', how='inner')
    
    # Calculate summary stats across simulations
    # We'll create columns for mean, median, 10th percentile, 90th percentile for each asset
    summary_df = pd.DataFrame({'Year': combined_df['Year']})
    
    for asset in ['Stock Value', 'Rental Value', 'Land Value']:
        asset_columns = [col for col in combined_df.columns if col.startswith(asset)]
        # axis=1 means row-wise
        summary_df[f'{asset} Mean'] = combined_df[asset_columns].mean(axis=1)
        summary_df[f'{asset} Median'] = combined_df[asset_columns].median(axis=1)
        summary_df[f'{asset} 10th'] = combined_df[asset_columns].quantile(0.1, axis=1)
        summary_df[f'{asset} 90th'] = combined_df[asset_columns].quantile(0.9, axis=1)
    
    return {
        'all_simulations': all_simulations,
        'combined': combined_df,
        'summary': summary_df
    }


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

def main():
    st.set_page_config(page_title="Enhanced Investment Comparison", layout="wide")
    st.title("Enhanced Investment Comparison and Retirement Planning")
    
    st.markdown("""
    This **demo** illustrates a more advanced financial model that includes:
    - Taxes (capital gains tax, rental income tax, property tax).
    - Detailed rental property cash flow (vacancy, maintenance, mortgage).
    - Monte Carlo simulations to capture volatility in returns.
    - Periodic rebalancing or changing withdrawal strategies.
    - Different inflation assumptions for rent vs. land vs. stock dividends.
    
    **Disclaimer**: This example is **highly simplified** and makes
    many assumptions that might not hold in a real-world scenario 
    (e.g., partial sales of property for rebalancing, simplistic tax handling).
    Use this as a **starting point** and customize to your real needs.
    """)

    st.sidebar.header("Simulation Settings")
    n_simulations = st.sidebar.number_input("Number of Simulations (Monte Carlo)", min_value=1, max_value=2000, value=200)
    years_accumulation = st.sidebar.number_input("Years of Accumulation", min_value=1, max_value=50, value=20)
    years_retirement = st.sidebar.number_input("Years in Retirement", min_value=1, max_value=50, value=30)
    
    st.sidebar.header("Stocks - Parameters")
    stock_initial = st.sidebar.number_input("Initial Stock Investment", min_value=0, value=50_000, step=1000)
    stock_annual_contribution = st.sidebar.number_input("Annual Contribution (Stocks)", min_value=0, value=10_000, step=1000)
    stock_expected_return = st.sidebar.slider("Expected Annual Return (Stocks)", min_value=0.0, max_value=0.2, value=0.07, step=0.01)
    stock_volatility = st.sidebar.slider("Volatility (Std Dev) (Stocks)", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    stock_capital_gains_tax_rate = st.sidebar.slider("Capital Gains Tax Rate (Stocks)", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    stock_dividend_yield = st.sidebar.slider("Dividend Yield (Stocks)", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
    stock_dividend_tax_rate = st.sidebar.slider("Dividend Tax Rate (Stocks)", min_value=0.0, max_value=0.5, value=0.15, step=0.01)

    st.sidebar.header("Rental - Parameters")
    rental_initial_equity = st.sidebar.number_input("Rental Initial Equity", min_value=0, value=50_000, step=1000)
    property_initial_value = st.sidebar.number_input("Property Value", min_value=0, value=200_000, step=1000)
    monthly_rent = st.sidebar.number_input("Monthly Rent (initial)", min_value=0, value=1500, step=100)
    vacancy_rate = st.sidebar.slider("Vacancy Rate", 0.0, 1.0, 0.05, 0.01)
    property_tax_rate = st.sidebar.slider("Property Tax Rate (Rental)", min_value=0.0, max_value=0.05, value=0.01, step=0.001)
    maintenance_rate = st.sidebar.slider("Maintenance Rate (% of property value)", 0.0, 0.05, 0.01, 0.001)
    mortgage_payment = st.sidebar.number_input("Annual Mortgage Payment", min_value=0, value=10_000, step=1000)
    rental_income_tax_rate = st.sidebar.slider("Rental Income Tax Rate", 0.0, 0.5, 0.2, 0.01)
    rent_inflation = st.sidebar.slider("Annual Rent Inflation", 0.0, 0.1, 0.02, 0.01)
    property_appreciation_mean = st.sidebar.slider("Property Appreciation (Mean)", 0.0, 0.2, 0.03, 0.01)
    property_appreciation_std = st.sidebar.slider("Property Appreciation (Std Dev)", 0.0, 0.2, 0.1, 0.01)

    st.sidebar.header("Land - Parameters")
    land_initial = st.sidebar.number_input("Initial Land Investment", min_value=0, value=30_000, step=1000)
    land_annual_contribution = st.sidebar.number_input("Annual Land Contribution", min_value=0, value=5_000, step=1000)
    land_expected_return = st.sidebar.slider("Expected Annual Return (Land)", min_value=0.0, max_value=0.2, value=0.03, step=0.01)
    land_volatility = st.sidebar.slider("Volatility (Std Dev) (Land)", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    land_property_tax_rate = st.sidebar.slider("Annual Property Tax Rate (Land)", 0.0, 0.05, 0.005, 0.001)
    land_capital_gains_tax_rate = st.sidebar.slider("Capital Gains Tax Rate (Land)", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    
    st.sidebar.header("Rebalancing")
    do_rebalance = st.sidebar.checkbox("Enable Annual Rebalancing?", value=False)
    st.sidebar.caption("Target Allocations (Only relevant if rebalancing is enabled)")
    target_alloc_stock = st.sidebar.slider("Target % Stock", 0.0, 1.0, 0.4, 0.05)
    target_alloc_rental = st.sidebar.slider("Target % Rental", 0.0, 1.0, 0.4, 0.05)
    target_alloc_land = st.sidebar.slider("Target % Land", 0.0, 1.0, 0.2, 0.05)
    if (target_alloc_stock + target_alloc_rental + target_alloc_land) != 1.0:
        st.sidebar.warning("Sum of target allocations != 1.0. Will be normalized automatically.")
    
    # Normalize allocations
    total_alloc = target_alloc_stock + target_alloc_rental + target_alloc_land
    if total_alloc == 0:
        target_allocations = {'stock': 0.0, 'rental': 0.0, 'land': 0.0}
    else:
        target_allocations = {
            'stock': target_alloc_stock / total_alloc,
            'rental': target_alloc_rental / total_alloc,
            'land': target_alloc_land / total_alloc
        }
    
    st.sidebar.header("Retirement Withdrawals")
    annual_withdrawal_stocks = st.sidebar.number_input("Annual Withdrawal from Stocks", min_value=0, value=20_000, step=1000)
    annual_withdrawal_rental = st.sidebar.number_input("Annual Withdrawal from Rental", min_value=0, value=10_000, step=1000)
    annual_withdrawal_land = st.sidebar.number_input("Annual Withdrawal from Land", min_value=0, value=5_000, step=1000)
    
    # Run Simulation
    if st.button("Run Simulation"):
        with st.spinner("Running Monte Carlo simulation..."):
            results = run_simulation(
                n_simulations,
                years_accumulation,
                years_retirement,
                # Stock
                stock_initial,
                stock_annual_contribution,
                stock_expected_return,
                stock_volatility,
                stock_capital_gains_tax_rate,
                stock_dividend_yield,
                stock_dividend_tax_rate,
                # Rental
                rental_initial_equity,
                property_initial_value,
                monthly_rent,
                vacancy_rate,
                property_tax_rate,
                maintenance_rate,
                mortgage_payment,
                rental_income_tax_rate,
                rent_inflation,
                property_appreciation_mean,
                property_appreciation_std,
                # Land
                land_initial,
                land_annual_contribution,
                land_expected_return,
                land_volatility,
                land_property_tax_rate,
                land_capital_gains_tax_rate,
                # Rebalancing
                do_rebalance,
                target_allocations,
                # Withdrawals
                annual_withdrawal_stocks,
                annual_withdrawal_rental,
                annual_withdrawal_land
            )
        
        st.success("Simulation Complete!")
        
        # Display summary DataFrame
        summary_df = results['summary']
        st.subheader("Summary of Simulations")
        st.dataframe(summary_df)
        
        # Plot the results (Mean)
        fig = px.line(
            summary_df, 
            x="Year", 
            y=[
                'Stock Value Mean', 
                'Rental Value Mean', 
                'Land Value Mean'
            ],
            title="Mean Portfolio Values Over Time (Monte Carlo)"
        )
        fig.update_layout(yaxis_title="Mean Value ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot Confidence Intervals (10th & 90th percentiles)
        st.subheader("Confidence Interval Plots (10th - 90th)")
        
        # We'll make a line chart with shaded area
        # For example, for Stock:
        stock_fig = px.line(summary_df, x="Year", y="Stock Value Median", title="Stock Value: Median (with 10-90th Range)")
        stock_fig.add_traces(px.line(summary_df, x="Year", y="Stock Value 10th").data)
        stock_fig.add_traces(px.line(summary_df, x="Year", y="Stock Value 90th").data)
        stock_fig.update_layout(showlegend=True)
        # Add fill between 10th and 90th
        stock_fig.data[1].update(fill=None)
        stock_fig.data[2].update(fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        stock_fig.update_traces(name='10th percentile', selector=dict(name='Stock Value 10th'))
        stock_fig.update_traces(name='90th percentile', selector=dict(name='Stock Value 90th'))
        stock_fig.update_traces(name='Median', selector=dict(name='Stock Value Median'))
        st.plotly_chart(stock_fig, use_container_width=True)
        
        # Similarly for rental
        rental_fig = px.line(summary_df, x="Year", y="Rental Value Median", title="Rental Value: Median (with 10-90th Range)")
        rental_fig.add_traces(px.line(summary_df, x="Year", y="Rental Value 10th").data)
        rental_fig.add_traces(px.line(summary_df, x="Year", y="Rental Value 90th").data)
        rental_fig.update_layout(showlegend=True)
        rental_fig.data[1].update(fill=None)
        rental_fig.data[2].update(fill='tonexty', fillcolor='rgba(100,0,80,0.2)')
        rental_fig.update_traces(name='10th percentile', selector=dict(name='Rental Value 10th'))
        rental_fig.update_traces(name='90th percentile', selector=dict(name='Rental Value 90th'))
        rental_fig.update_traces(name='Median', selector=dict(name='Rental Value Median'))
        st.plotly_chart(rental_fig, use_container_width=True)
        
        # Similarly for land
        land_fig = px.line(summary_df, x="Year", y="Land Value Median", title="Land Value: Median (with 10-90th Range)")
        land_fig.add_traces(px.line(summary_df, x="Year", y="Land Value 10th").data)
        land_fig.add_traces(px.line(summary_df, x="Year", y="Land Value 90th").data)
        land_fig.update_layout(showlegend=True)
        land_fig.data[1].update(fill=None)
        land_fig.data[2].update(fill='tonexty', fillcolor='rgba(100,100,0,0.2)')
        land_fig.update_traces(name='10th percentile', selector=dict(name='Land Value 10th'))
        land_fig.update_traces(name='90th percentile', selector=dict(name='Land Value 90th'))
        land_fig.update_traces(name='Median', selector=dict(name='Land Value Median'))
        st.plotly_chart(land_fig, use_container_width=True)
        
        st.markdown("""
        ### Interpreting These Results
        - **Median** line: Where half the simulations ended up above, half below.
        - **10th percentile**: An optimistic measure that 90% of simulations ended at or above this line.
        - **90th percentile**: A more bullish outcome that 10% of simulations matched or exceeded.
        
        The **mean** might be skewed by a few extremely high outcomes in Monte Carlo simulations.
        
        ### Potential Next Steps
        1. **Refine the tax model**: Track cost basis, depreciation recapture for rental property, 
           progressive tax brackets, etc.
        2. **Mortgage dynamics**: Model changing mortgage principal, interest amortization, and possible refinancing.
        3. **Correlations**: Real estate and stock market might have less than 1 correlation, 
           but not truly independent. Incorporate correlation in the random draws.
        4. **Partial retirement**: Vary withdrawal rates or include part-time work in early retirement.
        5. **Alternative rebalancing strategies**: e.g., only rebalance if allocations drift beyond a band.
        6. **Scenario testing**: Stress test with high inflation or severe market downturns.
        """)


if __name__ == "__main__":
    main()
