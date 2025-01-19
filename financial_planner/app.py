import streamlit as st
import matplotlib.pyplot as plt
from scenario import ScenarioManager
from models import StockPortfolio, RentalPortfolio, LandInvestment, Cost

# Initialize the scenario manager
scenario = ScenarioManager()

st.title("Financial Planner App")
st.sidebar.header("Add Investments and Costs")

# Sidebar inputs for investments
with st.sidebar.expander("Add Stock Portfolio"):
    stock_amount = st.number_input("Initial Amount ($)", min_value=0.0, value=10000.0, step=1000.0, key="stock_amount")
    stock_rate = st.number_input("Annual Growth Rate (%)", min_value=0.0, max_value=100.0, value=7.0, step=0.1, key="stock_rate")
    if st.button("Add Stock Portfolio"):
        scenario.add_investment(StockPortfolio(stock_amount, stock_rate / 100))
        st.success(f"Stock Portfolio added: ${stock_amount:.2f} at {stock_rate:.2f}%")

with st.sidebar.expander("Add Rental Portfolio"):
    rental_equity = st.number_input("Initial Equity ($)", min_value=0.0, value=50000.0, step=1000.0, key="rental_equity")
    rental_rate = st.number_input("Annual Return Rate (%)", min_value=0.0, max_value=100.0, value=6.0, step=0.1, key="rental_rate")
    if st.button("Add Rental Portfolio"):
        scenario.add_investment(RentalPortfolio(rental_equity, rental_rate / 100))
        st.success(f"Rental Portfolio added: ${rental_equity:.2f} at {rental_rate:.2f}%")

with st.sidebar.expander("Add Land Investment"):
    land_amount = st.number_input("Initial Amount ($)", min_value=0.0, value=30000.0, step=1000.0, key="land_amount")
    land_rate = st.number_input("Annual Appreciation Rate (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1, key="land_rate")
    if st.button("Add Land Investment"):
        scenario.add_investment(LandInvestment(land_amount, land_rate / 100))
        st.success(f"Land Investment added: ${land_amount:.2f} at {land_rate:.2f}%")

# Sidebar inputs for costs
with st.sidebar.expander("Add One-Time Costs"):
    cost_amount = st.number_input("Cost Amount ($)", min_value=0.0, value=20000.0, step=1000.0, key="cost_amount")
    cost_year = st.number_input("Year of Cost", min_value=1, value=5, step=1, key="cost_year")
    if st.button("Add Cost"):
        scenario.add_cost(Cost(cost_amount, cost_year))
        st.success(f"One-Time Cost added: ${cost_amount:.2f} in year {cost_year}")

# Main app inputs for projections
st.header("Projection and Retirement")
years_before_retirement = st.number_input("Years Before Retirement", min_value=1, value=20, step=1)
annual_withdrawal = st.number_input("Annual Withdrawal in Retirement ($)", min_value=0.0, value=40000.0, step=1000.0)
retirement_growth = st.number_input("Growth Rate During Retirement (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

if st.button("Calculate Portfolio"):
    total_value = scenario.total_value_after_years(years_before_retirement)
    deplete_years = scenario.retirement_duration(years_before_retirement, annual_withdrawal, retirement_growth / 100)

    st.subheader("Results")
    st.write(f"Total Portfolio Value at Retirement: **${total_value:,.2f}**")
    st.write(f"Portfolio lasts approximately **{deplete_years} years** in retirement with annual withdrawal of ${annual_withdrawal:,.2f}.")

# Plot growth over time
if st.button("Plot Portfolio Growth"):
    st.subheader("Portfolio Growth Over Time")
    portfolio_values = scenario.simulate_growth(years_before_retirement)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(0, years_before_retirement + 1), portfolio_values, marker='o')
    ax.set_title("Portfolio Growth Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Value ($)")
    ax.grid(True)

    st.pyplot(fig)
