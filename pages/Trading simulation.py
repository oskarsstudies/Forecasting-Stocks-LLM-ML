import streamlit as st
import joblib
from Website_calculations import load_data, simulate_trading  

def app():
    st.title('Stock Market Forecasting and Trading Simulator')
    companies = [
        'AAPL', 'NVDA', 'AMZN', 'GOOG', 'META', 'LLY', 'TSM', 'AVGO', 'V', 'NVO', 
        'TSLA', 'MSFT', 'WMT', 'XOM', 'MA', 'UNH', 'ASML', 'JNJ', 'PG'
    ]
    company = st.selectbox('Choose a stock:', companies)
    initial_capital = st.number_input('Enter the amount you want to trade with:', min_value=1000, value=10000, step=500)

    model_choice = st.selectbox('Choose a model:', ['RandomForest', 'AdaBoost', 'XGBoost'])
    model_path = f'ML_models/{company}_{model_choice}.pkl'
    model = joblib.load(model_path)

    if st.button('Make Prediction'):
        X_train, X_test, y_train, y_test, df = load_data(company)  # Load data
        predictions = model.predict(X_test)
        fig, portfolio_value, hold_portfolio_value, final_portfolio_value = simulate_trading(predictions, X_test, y_test, initial_capital)

        st.pyplot(fig)  # Display the plot        
        st.write("Trading strategy results:")
        st.write(f"Initial capital: ${initial_capital}")
        st.write(f"Final trading strategy portfolio value: ${format(portfolio_value[-1], '.2f')}")
        st.write(f"Final hold strategy portfolio value: ${format(hold_portfolio_value[-1], '.2f')}")

def main():
    app()

if __name__ == '__main__':
    main()
