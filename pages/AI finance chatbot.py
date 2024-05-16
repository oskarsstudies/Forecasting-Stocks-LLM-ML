import json
import numpy as np
import openai 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import random

# Load your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["global"]["OPENAI_API_KEY"]

# Define your functions
def fetch_stock_price(symbol):
    return str(yf.Ticker(symbol).history(period='1y').iloc[-1].Close)

def compute_sma(symbol, period=20):
    prices = yf.Ticker(symbol).history(period='1y').Close
    sma = prices.rolling(window=period).mean()
    return str(sma.iloc[-1])

def compute_ema(symbol, period=20):
    prices = yf.Ticker(symbol).history(period='1y').Close
    ema = prices.ewm(span=period, adjust=False).mean()
    return str(ema.iloc[-1])

def compute_rsi(symbol):
    prices = yf.Ticker(symbol).history(period='1y').Close
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=14-1, adjust=False).mean()
    avg_loss = loss.ewm(com=14-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return str(rsi.iloc[-1])

def compute_macd(symbol):
    prices = yf.Ticker(symbol).history(period='1y').Close
    short_ema = prices.ewm(span=12, adjust=False).mean()
    long_ema = prices.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal
    return f'{macd.iloc[-1]}, {signal.iloc[-1]}, {macd_histogram.iloc[-1]}'

def plot_price_chart(symbol):
    data = yf.Ticker(symbol).history(period='1y')
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data.Close)
    plt.title(f'{symbol} Stock Price Over Last Year')  
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def plot_sma_chart(symbol):
    prices = yf.Ticker(symbol).history(period='1y').Close
    sma = prices.rolling(window=20).mean()
    
    plt.figure(figsize=(10,5))
    plt.plot(prices.index, prices, label='Stock Price', color='blue')
    plt.plot(prices.index, sma, label='SMA (20)', color='red')
    plt.title(f'{symbol} SMA Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def plot_ema_chart(symbol):
    prices = yf.Ticker(symbol).history(period='1y').Close
    ema = prices.ewm(span=20, adjust=False).mean()
    
    plt.figure(figsize=(10,5))
    plt.plot(prices.index, prices, label='Stock Price', color='blue')
    plt.plot(prices.index, ema, label='EMA (20)', color='red')
    plt.title(f'{symbol} EMA Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def plot_rsi_chart(symbol):
    prices = yf.Ticker(symbol).history(period='1y').Close
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=14-1, adjust=False).mean()
    avg_loss = loss.ewm(com=14-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    plt.figure(figsize=(10,5))
    plt.plot(prices.index, rsi)
    plt.title(f'{symbol} RSI Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def plot_macd_chart(symbol):
    prices = yf.Ticker(symbol).history(period='1y').Close
    short_ema = prices.ewm(span=12, adjust=False).mean()
    long_ema = prices.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal
    
    plt.figure(figsize=(10,5))
    plt.plot(prices.index, macd, label='MACD', color='b')
    plt.plot(prices.index, signal, label='Signal Line', color='r')
    plt.bar(prices.index, macd_histogram, label='MACD Histogram', color='g')
    plt.title(f'{symbol} MACD Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend(loc='upper left')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

functions = [
    {
        'name':'fetch_stock_price',
        'description':'Gets the latest stock price given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol']
        }
    },
    {
        'name':'compute_sma',
        'description':'Calculate the simple moving average for a given stock ticker.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'compute_ema',
        'description':'Calculate the exponential moving average for a given stock ticker.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'compute_rsi',
        'description':'Calculate the RSI for a given stock ticker.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'compute_macd',
        'description':'Calculate the MACD for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'plot_price_chart',
        'description':'Plot the stock price for the last year given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'plot_sma_chart',
        'description':'Plot the SMA for the last year given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'plot_ema_chart',
        'description':'Plot the EMA for the last year given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'plot_rsi_chart',
        'description':'Plot the RSI for the last year given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    },
    {
        'name':'plot_macd_chart',
        'description':'Plot the MACD for the last year given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['symbol'],
        },
    }
]

available_functions = {
    'fetch_stock_price': fetch_stock_price,
    'compute_sma': compute_sma,
    'compute_ema': compute_ema,
    'compute_rsi': compute_rsi,
    'compute_macd': compute_macd,
    'plot_price_chart': plot_price_chart,
    'plot_sma_chart': plot_sma_chart,
    'plot_ema_chart': plot_ema_chart,
    'plot_rsi_chart': plot_rsi_chart,
    'plot_macd_chart': plot_macd_chart,
}

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('Stock Analysis Assistant')

# CAPTCHA generation
if 'captcha_num1' not in st.session_state:
    st.session_state['captcha_num1'] = random.randint(1, 10)
if 'captcha_num2' not in st.session_state:
    st.session_state['captcha_num2'] = random.randint(1, 10)

captcha_answer = st.session_state['captcha_num1'] + st.session_state['captcha_num2']

user_input = st.text_input('Ask a question:')
captcha_input = st.text_input(f'(Captcha) - Solve this to continue: {st.session_state["captcha_num1"]} + {st.session_state["captcha_num2"]} =')

if user_input and captcha_input:
    try:
        if int(captcha_input) == captcha_answer:
            st.session_state['messages'].append({'role': 'user', 'content': f'{user_input}'})

            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-0613',
                messages=st.session_state['messages'],
                functions=functions,
                function_call='auto'
            )

            response_message = response['choices'][0]['message']
            if response_message.get('function_call'):
                function_name = response_message['function_call']['name']
                function_args = response_message['function_call']['arguments']
                if function_args:
                    function_args = json.loads(function_args)
                    if function_name in ['fetch_stock_price', 'compute_rsi', 'compute_macd', 'plot_price_chart', 'plot_sma_chart', 'plot_ema_chart', 'plot_rsi_chart', 'plot_macd_chart']:
                        args_dict = {'symbol': function_args.get('symbol')}
                    elif function_name in ['compute_sma', 'compute_ema']:
                        args_dict = {'symbol': function_args.get('symbol')}
                    
                    function_to_call = available_functions[function_name]
                    if function_name in ['plot_price_chart', 'plot_sma_chart', 'plot_ema_chart', 'plot_rsi_chart', 'plot_macd_chart']:
                        function_to_call(**args_dict)
                    else:
                        function_response = function_to_call(**args_dict)
                        st.session_state['messages'].append(response_message)
                        st.session_state['messages'].append(
                            {
                                'role': 'function',
                                'name': function_name,
                                'content': function_response
                            }
                        )
                        second_response = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo-0613',
                            messages=st.session_state['messages']
                        )
                        st.text_area("Response:", second_response['choices'][0]['message']['content'], height=200)
                        st.session_state['messages'].append({'role': 'assistant', 'content': second_response['choices'][0]['message']['content']})
                else:
                    st.text("No arguments found for the function call.")
            else: 
                st.text_area("Response:", response_message['content'], height=200)
                st.session_state['messages'].append({'role': 'assistant', 'content': response_message['content']})
        else:
            st.error("Incorrect CAPTCHA answer.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
