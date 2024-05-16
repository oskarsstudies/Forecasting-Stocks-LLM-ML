import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ta import add_all_ta_features
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib 


def load_data(ticker):

    # Define the path to your file using the ticker variable
    file_path = f'Data/{ticker}_historical_data.csv'
    # Read the CSV file into a DataFrame
    df_prices = pd.read_csv(file_path)

    # Initialize lists to store data from all files
    all_dates, all_sentiment_scores, all_relevance_scores, all_avg_relevance_scores = [], [], [], []

    # List all JSON files in the directory corresponding to the ticker
    json_files = [file for file in os.listdir('Data') if file.startswith(f'{ticker}_news_data') and file.endswith('.json')]

    # Iterate over each JSON file
    for file in json_files:
        with open(os.path.join('Data', file), 'r', encoding='utf-8') as file:
            data = json.load(file)
            news_feed = data['feed']
        
        # Iterate over the news feed in each file to extract information
        for item in news_feed:
            ticker_info = [ts for ts in item['ticker_sentiment'] if ts['ticker'] == ticker]
            if ticker_info:  # Check if info is available in the ticker sentiment
                all_dates.append(item['time_published'])
                all_sentiment_scores.append(ticker_info[0]['ticker_sentiment_score'])
                all_relevance_scores.append(ticker_info[0]['relevance_score'])
                # Calculate the average relevance score for all tickers in the news item
                avg_relevance = sum(float(ts['relevance_score']) for ts in item['ticker_sentiment']) / len(item['ticker_sentiment'])
                all_avg_relevance_scores.append(avg_relevance)

        # Print length of lists after each file
        print(f"File: {file}, Entries: {len(all_dates)}")

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'Date': all_dates,
        'Sentiment_Score': all_sentiment_scores,
        'Relevance_Score_ticker': all_relevance_scores,
        'Avg_Relevance_Score_topics': all_avg_relevance_scores
    })

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

    # Convert relevant columns to float
    df['Sentiment_Score'] = df['Sentiment_Score'].astype(float)
    df['Relevance_Score_ticker'] = df['Relevance_Score_ticker'].astype(float)
    df['Avg_Relevance_Score_topics'] = df['Avg_Relevance_Score_topics'].astype(float)

    # Group by 'Date' and calculate mean, min, and max for 'Sentiment_Score'
    df_grouped = df.groupby('Date').agg({
        'Sentiment_Score': ['mean', 'min', 'max'],
        'Relevance_Score_ticker': 'mean',
        'Avg_Relevance_Score_topics': 'mean'
    }).reset_index()

    # Flatten multi-index columns
    df_grouped.columns = ['Date', 'Sentiment_Score_mean', 'Sentiment_Score_min', 'Sentiment_Score_max',
                        'Relevance_Score_ticker_mean', 'Avg_Relevance_Score_topics_mean']

    # Ensure df_grouped is loaded and has the necessary columns converted to datetime if not already done
    df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])
    df_grouped.set_index('Date', inplace=True)

    # Create a date range from start date to end date
    date_range = pd.date_range(start='2023-08-31', end='2024-04-07', freq='D')

    # Reindex the DataFrame to include all dates in the range
    df_grouped = df_grouped.reindex(date_range)

    # Use backward fill to fill initial missing values, then forward fill the rest
    df_grouped.bfill(inplace=True)
    df_grouped.ffill(inplace=True)

    # Resetting index to turn 'Date' back into a column
    df_grouped.reset_index(inplace=True)
    df_grouped.rename(columns={'index': 'Date'}, inplace=True)

    df_prices['Date'] = df_prices['Date'].astype(str)
    # Slice the string to extract only the date part
    df_prices['Date'] = df_prices['Date'].str[0:10]
    # Convert back to datetime
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    # Merge the data on 'Date'
    df1 = pd.merge(df_grouped, df_prices[['Date', 'Close', 'Open', 'Volume', 'Low', 'High']], on='Date', how='right')
    df = df1
    df['Next_Day_Open'] = df['Open'].shift(-1)
    df.dropna(subset='Next_Day_Open', inplace=True)
    df = df[df.Date <= '2024-04-05']
    ### Social meadia - Reddit sentiment
    df_reddit_raw = pd.read_csv("Reddit_sentiment_grouped.csv")
    df_reddit = df_reddit_raw
    df_reddit['Date'] = pd.to_datetime(df_reddit['Date'])

    # Find the global min and max dates across all tickers
    global_start_date = df_reddit['Date'].min()
    global_end_date = df_reddit['Date'].max()

    # Create an empty DataFrame to store results
    result_df = pd.DataFrame()

    # Process each ticker group separately
    for ticker_reddit, group in df_reddit.groupby('Ticker'):
        # Create date range from the global earliest to the global latest date
        date_range = pd.date_range(start=global_start_date, end=global_end_date, freq='D')
        
        # Reindex the group to include all days in the range, setting Date as the index
        group.set_index('Date', inplace=True)
        group_reindexed = group.reindex(date_range, method='ffill')  # Ensure forward fill is called here

        # Reset the index to turn the date index back into a column
        group_reindexed.reset_index(inplace=True)
        group_reindexed.rename(columns={'index': 'Date'}, inplace=True)

        # Set the Ticker for all rows in the reindexed DataFrame
        group_reindexed['Ticker'] = ticker_reddit
        
        # Concatenate this reindexed group to the result DataFrame
        result_df = pd.concat([result_df, group_reindexed], ignore_index=True)
        
    result_df = result_df[result_df.Ticker == ticker]
    df = df.merge(result_df.drop(columns={'Ticker'}), on = 'Date', how = 'left')
    full_ta = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume"
    )
    df = df.dropna(subset=['Next_Day_Open','Sentiment_Score_mean'])

    df = df.dropna(axis=1, how='any')
    X = df.drop(columns=['Next_Day_Open','Close','Date'])
    y = df['Next_Day_Open']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    return X_train, X_test, y_train, y_test, df




def simulate_trading(predictions, X_test, y_test, initial_capital):
    y_pred_shifted = np.roll(predictions, -1)[:-1]
    signals = ['Buy' if pred > op else 'Sell' for pred, op in zip(y_pred_shifted, X_test['Open'][:-1])]

    capital = initial_capital
    shares_owned = 0
    portfolio_value = []
    hold_portfolio_value = []

    # Calculate the hold strategy from start to end using the initial capital
    shares_bought = initial_capital // X_test['Open'].iloc[0]
    hold_portfolio_value = [shares_bought * price for price in X_test['Open']]

    for signal, actual_open, close in zip(signals, X_test['Open'], y_test):
        if signal == 'Buy' and capital >= actual_open:
            shares_to_buy = int(capital // actual_open)
            capital -= shares_to_buy * actual_open
            shares_owned += shares_to_buy
        elif signal == 'Sell' and shares_owned > 0:
            capital += shares_owned * close
            shares_owned = 0
        portfolio_value.append(capital + shares_owned * close)

    final_portfolio_value = capital + (shares_owned * y_test.iloc[-1] if shares_owned > 0 else 0)

    # Generate plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio_value, label='Trading Strategy Portfolio Value', color='blue')
    ax.plot(hold_portfolio_value, label='Hold Strategy Portfolio Value', color='green')
    ax.axhline(y=initial_capital, color='red', linestyle='--', label='Initial Capital')
    ax.set_title('Comparison of Trading vs. Hold Strategy Over Time')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)

    return fig, portfolio_value, hold_portfolio_value, final_portfolio_value