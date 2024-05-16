import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pandas_ta as ta

def app():
    # Set up the title of the dashboard
    st.title('Interactive Technical Analysis Dashboard')

    # Dropdown to select a company
    company = st.selectbox('Select a Company Ticker', ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOG', 'META', 'LLY', 'TSM', 'AVGO', 'V', 'NVO', 'TSLA', 'WMT', 'XOM', 'MA', 'UNH', 'ASML', 'JNJ', 'PG'])

    # Date input for range selection
    start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('today'))

    # Dropdown for selecting the resampling frequency
    timeframe = st.selectbox('Select Timeframe for Analysis for candlesticks', ['1d', '1h', '1wk', '1mo'])

    # Load data button
    if st.button('Load Data'):
        data = yf.download(company, start=start_date, end=end_date, interval=timeframe)
        st.session_state['data'] = data  # Store data in session state
        st.write(f"Data for {company} from {start_date} to {end_date}")

    # If data is loaded
    if 'data' in st.session_state:
        data = st.session_state['data']

        # Plotting the candlestick chart
        fig_candlestick = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        )])
        fig_candlestick.update_layout(title=f'Candlestick chart for {company} - {timeframe}', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_candlestick, use_container_width=True)

        # Aroon Indicator
        aroon = ta.aroon(data['High'], data['Low'])
        fig_aroon = go.Figure()
        fig_aroon.add_trace(go.Scatter(x=data.index, y=aroon['AROONU_14'], name='Aroon Up', line=dict(color='green', width=2)))
        fig_aroon.add_trace(go.Scatter(x=data.index, y=aroon['AROOND_14'], name='Aroon Down', line=dict(color='red', width=2)))
        fig_aroon.update_layout(title='Aroon Indicator', xaxis_title='Date', yaxis_title='Aroon')
        st.plotly_chart(fig_aroon, use_container_width=True)

        # MACD
        macd = ta.macd(data['Close'])
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=macd['MACD_12_26_9'], name='MACD Line', line=dict(color='blue', width=2)))
        fig_macd.add_trace(go.Scatter(x=data.index, y=macd['MACDh_12_26_9'], name='Signal Line', line=dict(color='red', width=2)))
        fig_macd.add_trace(go.Scatter(x=data.index, y=macd['MACDs_12_26_9'], name='Histogram', line=dict(color='orange', width=2)))
        fig_macd.update_layout(title='MACD', xaxis_title='Date', yaxis_title='MACD')
        st.plotly_chart(fig_macd, use_container_width=True)

        # RSI
        rsi = ta.rsi(data['Close'])
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple', width=2)))
        fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI', yaxis_range=[0,100])
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Stochastic Oscillator
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=data.index, y=stoch['STOCHk_14_3_3'], name='Stochastic %K', line=dict(color='green', width=2)))
        fig_stoch.add_trace(go.Scatter(x=data.index, y=stoch['STOCHd_14_3_3'], name='Stochastic %D', line=dict(color='red', width=2)))
        fig_stoch.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Stochastic')
        st.plotly_chart(fig_stoch, use_container_width=True)

app()        
