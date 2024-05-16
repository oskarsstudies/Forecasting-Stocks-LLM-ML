import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def app():
    # Set up the title of the dashboard
    st.title('Fundatemental Stock Analysis')

    # Dropdown to select a company
    company = st.selectbox('Select a Company Ticker', ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOG', 'META', 'LLY', 'TSM', 'AVGO', 'V', 'NVO', 'TSLA', 'WMT', 'XOM', 'MA', 'UNH', 'ASML', 'JNJ', 'PG'])

    # Function to fetch and display fundamental data
    def display_financials_data(ticker):
        stock = yf.Ticker(ticker)

        # Fetch annual and quarterly financials
        financials_annual = stock.financials
        financials_quarterly = stock.quarterly_financials

        # Convert datetime index to date-only format if it is datetime
        if isinstance(financials_annual.columns, pd.DatetimeIndex):
            financials_annual.columns = financials_annual.columns.date
        if isinstance(financials_quarterly.columns, pd.DatetimeIndex):
            financials_quarterly.columns = financials_quarterly.columns.date

        # Display annual financials
        if not financials_annual.empty:
            st.write(f"**Annual Financials for {ticker}**")
            st.dataframe(financials_annual)

        # Display quarterly financials
        if not financials_quarterly.empty:
            st.write(f"**Quarterly Financials for {ticker}**")
            st.dataframe(financials_quarterly)

        # Graphs for key metrics (e.g., Total Revenue and Net Income)
        if 'Total Revenue' in financials_annual.index and 'Net Income' in financials_annual.index:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=financials_annual.columns, y=financials_annual.loc['Total Revenue'], name='Total Revenue'))
            fig.add_trace(go.Bar(x=financials_annual.columns, y=financials_annual.loc['Net Income'], name='Net Income'))
            fig.update_layout(title='Annual Revenue and Net Income', barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    # Button to load financial data
    if st.button('Show Financial Data'):
        display_financials_data(company)
app()