import pandas as pd
import pmdarima as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
import pandas_datareader.data as web
from datetime import datetime, timedelta
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error



# Load your cleaned dataset
df = pd.read_csv("full_cleaned_bank_stocks.csv", parse_dates=["Date"])
df.sort_values(by=["Stock", "Date"], inplace=True)
df = df.sort_values(by=['Stock', 'Date'])

# log return calculation
df['Log_Returns'] = df.groupby('Stock')['Close'].transform(lambda x: np.log(x / x.shift(1)))
df.dropna(subset=['Log_Returns'], inplace=True)

#Structure of the Data
print(df.head())
print(df.info())
print(df.describe())
print(df['Stock'].value_counts())

#Time Series Plot of Closing Prices
plt.figure(figsize=(12, 6))
for stock in df['Stock'].unique():
    stock_data = df[df['Stock'] == stock]
    plt.plot(stock_data['Date'], stock_data['Close'], label=stock)

plt.title("Stock Closing Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()

plt.show(block=False)

df = df.sort_values(by=['Stock', 'Date'])

# log return calculation
df['Log_Returns'] = df.groupby('Stock')['Close'].transform(lambda x: np.log(x / x.shift(1)))
df.dropna(subset=['Log_Returns'], inplace=True)

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Log_Returns', hue='Stock', kde=True, bins=100)
plt.title("Distribution of Log Returns")
plt.xlabel("Log Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Volatility Over Time
plt.figure(figsize=(12, 6))

for stock in df['Stock'].unique():
    stock_data = df[df['Stock'] == stock].copy()  # Fix: Use .copy() to avoid SettingWithCopyWarning
    stock_data['Rolling_Std'] = stock_data['Log_Returns'].rolling(window=30).std()
    plt.plot(stock_data['Date'], stock_data['Rolling_Std'], label=stock)

plt.title("Rolling 30-Day Volatility")
plt.xlabel("Date")
plt.ylabel("Standard Deviation of Returns")
plt.legend()
plt.tight_layout()
plt.show()


#Correlation Between Stock Prices

# Pivot data: Date as index, stocks as columns, values as Close prices
price_pivot = df.pivot(index='Date', columns='Stock', values='Close')

# Drop any missing values
price_pivot.dropna(inplace=True)

# Correlation matrix
corr_matrix = price_pivot.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation of Closing Prices Between Stocks")
plt.show()


#Autocorrelation & Partial Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

for stock in df['Stock'].unique():
    stock_data = df[df['Stock'] == stock]['Log_Returns'].dropna()

    plt.figure()
    plot_acf(stock_data, lags=30)
    plt.title(f"Autocorrelation for {stock}")
    plt.show()

    plt.figure()
    plot_pacf(stock_data, lags=30)
    plt.title(f"Partial Autocorrelation for {stock}")
    plt.show()


#Check for Outliers
sns.boxplot(data=df, x='Stock', y='Log_Returns')
plt.title("Boxplot of Log Returns by Stock")
plt.show()

# Load your cleaned data
df = pd.read_csv("full_cleaned_bank_stocks.csv", parse_dates=["Date"])
df.sort_values(by=["Stock", "Date"], inplace=True)

# List of selected stocks
stocks = ['AXISBANK', 'BAJFINANCE', 'BAJAJFINSV', 'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'SBIN']


# Function to perform ADF test
def check_stationarity(ts, stock_name):
    result = adfuller(ts.dropna())
    print(f"\n📊 ADF Test for {stock_name}:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("✅ The series is stationary.")
    else:
        print("❌ The series is not stationary. Differencing may be needed.")

# Run ADF test for each stock
for stock in stocks:
    stock_data = df[df["Stock"] == stock]
    stock_data.set_index("Date", inplace=True)
    ts = stock_data["Close"]
    check_stationarity(ts, stock)

def find_d(ts, max_diff=3):
    d = 0
    p_value = adfuller(ts.dropna())[1]
    while p_value > 0.05 and d < max_diff:
        ts = ts.diff().dropna()
        p_value = adfuller(ts)[1]
        d += 1
    return d

def plot_acf_pacf(ts, stock_name, lags=30):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    plot_acf(ts.dropna(), lags=lags, ax=ax[0])
    plot_pacf(ts.dropna(), lags=lags, ax=ax[1])
    ax[0].set_title(f'ACF Plot for {stock_name}')
    ax[1].set_title(f'PACF Plot for {stock_name}')
    plt.show()

for stock in stocks:
    stock_data = df[df["Stock"] == stock]
    stock_data.set_index("Date", inplace=True)
    ts = stock_data["Close"]
    
    # Find d
    d = find_d(ts)
    print(f"\nFor {stock}, optimal differencing order d = {d}")
    
    # Differenced series based on d
    ts_diff = ts.copy()
    for _ in range(d):
        ts_diff = ts_diff.diff()
    
    # Plot ACF and PACF for differenced series
    plot_acf_pacf(ts_diff, stock)

def find_d(ts, max_diff=3):
    d = 0
    p_value = adfuller(ts.dropna())[1]
    ts_diff = ts.copy()
    
    while p_value > 0.05 and d < max_diff:
        ts_diff = ts_diff.diff().dropna()
        p_value = adfuller(ts_diff)[1]
        d += 1
    
    return d, ts_diff

# Function to plot PACF and suggest p
def suggest_p(ts, stock_name, lags=20):
    plt.figure(figsize=(10, 5))
    plot_pacf(ts.dropna(), lags=lags)
    plt.title(f"PACF Plot for {stock_name}")
    plt.show()
    print(f"👆 Use the PACF plot above to estimate p for {stock_name}.\nChoose the lag where the bars drop within the confidence interval.")

# Iterate over stocks
for stock in stocks:
    stock_data = df[df["Stock"] == stock]
    stock_data.set_index("Date", inplace=True)
    ts = stock_data["Close"]

    # Differencing to get stationary series
    d, ts_diff = find_d(ts)

    print(f"\n🔎 Stock: {stock}")
    print(f"Optimal differencing order d = {d}")

    # Plot PACF to estimate p
    suggest_p(ts_diff, stock)



# ✅ Define suggest_q function BEFORE you use it
def suggest_q(ts, stock_name, lags=20):
    plt.figure(figsize=(10, 5))
    plot_acf(ts.dropna(), lags=lags)
    plt.title(f"ACF Plot for {stock_name}")
    plt.show()
    print(f"👆 Use the ACF plot above to estimate q for {stock_name}.\nChoose the lag where the bars drop within the confidence interval.")

# Now run for each stock
for stock in stocks:
    stock_data = df[df["Stock"] == stock]
    stock_data.set_index("Date", inplace=True)
    ts = stock_data["Close"]

    # Get differenced series
    d, ts_diff = find_d(ts)

    print(f"\n🔎 Stock: {stock}")
    print(f"Optimal differencing order d = {d}")

    # Plot ACF for estimating q
    suggest_q(ts_diff, stock)\

# Function to find the optimal non-seasonal ARIMA(p, 1, q) order using AIC
def find_optimal_order(ts_series, stock_name):
    """
    Uses pmdarima.auto_arima to find the best ARIMA(p, 1, q) order 
    by minimizing the AIC. We fix d=1 based on prior analysis.
    We limit the search to p,q <= 5 for parsimony.
    """
    print(f"\nSearching optimal ARIMA(p, 1, q) for {stock_name} using AIC...")
    
    # We must use the original 'Close' price series for auto_arima
    # It will handle the differencing internally since d=1 is fixed.
    
    model = pm.auto_arima(
        ts_series.dropna(), # Use the time series data
        d=1,                # Fixed difference order based on your analysis
        start_p=0, max_p=5, # Search p from 0 to 5
        start_q=0, max_q=5, # Search q from 0 to 5
        m=1,                # Non-seasonal model
        seasonal=False,
        trace=False,        # Set to True to see the search process
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True       # Efficient search method
    )
    
    optimal_order = model.order
    optimal_aic = model.aic()
    print(f"✅ Optimal Order for {stock_name}: ARIMA{optimal_order} (AIC: {optimal_aic:.2f})")
    
    return optimal_order

# 2. Define initial orders (from visual analysis for simple cases)
# We will overwrite the complex ones with AIC results.
arima_orders = {
    'AXISBANK': (0, 1, 0),
    'BAJFINANCE': (0, 1, 0),
    'HDFCBANK': (0, 1, 0),
    'ICICIBANK': (0, 1, 0),
    'SBIN': (0, 1, 0), 
    'BAJAJFINSV': (10, 1, 10),
    'INDUSINDBK': (7, 1, 7),
    'KOTAKBANK': (6, 1, 6),
}

# 3. Perform AIC optimization for the problematic stocks

stocks_to_optimize = ['BAJAJFINSV', 'INDUSINDBK', 'KOTAKBANK']

# Load data once for the optimization loop
df_close = df.pivot(index='Date', columns='Stock', values='Close').ffill().bfill()


for stock_name in stocks_to_optimize:
    # Get the 'Close' price series for the stock
    ts = df_close[stock_name].asfreq('B').ffill()
    
    # Find the optimal order and update the dictionary
    optimal_order = find_optimal_order(ts, stock_name)
    arima_orders[stock_name] = optimal_order

print("\n--- Final ARIMA Orders (Optimized) ---")
print(arima_orders)

# ARIMA MODEL
# Your specific ARIMA orders per stock 
arima_orders = {
    'AXISBANK': (0, 1, 0),
    'BAJAJFINSV': (0, 1, 0),
    'BAJFINANCE': (0, 1, 0),
    'HDFCBANK': (0, 1, 0),
    'ICICIBANK': (0, 1, 0),
    'INDUSINDBK': (1, 1, 0),
    'KOTAKBANK': (0, 1,0),
    'SBIN': (0, 1, 0)
}

default_order = (1, 1, 0)  # fallback ARIMA order

# Load and sort dataset
df = pd.read_csv("full_cleaned_bank_stocks.csv", parse_dates=["Date"])
df.sort_values(by=["Stock", "Date"], inplace=True)

stocks = list(arima_orders.keys())

for stock_name in stocks:
    print(f"\nProcessing {stock_name}...")
    stock_df = df[df["Stock"] == stock_name].copy()
    stock_df.set_index("Date", inplace=True)
    stock_df = stock_df.asfreq('B')  # business day frequency
    stock_df = stock_df.fillna(method='ffill')

    n = int(len(stock_df) * 0.8)
    train = stock_df['Close'][:n]
    test = stock_df['Close'][n:]

    # Try to fit your specific ARIMA order first
    order = arima_orders[stock_name]
    try:
        model = ARIMA(train, order=order)
        result = model.fit()
        print(f"Fitted ARIMA{order} for {stock_name}")
    except Exception as e:
        print(f"Failed to fit ARIMA{order} for {stock_name}, trying default order {default_order}. Error: {e}")
        try:
            model = ARIMA(train, order=default_order)
            result = model.fit()
            print(f"Fitted default ARIMA{default_order} for {stock_name}")
        except Exception as e2:
            print(f"Failed to fit default ARIMA for {stock_name} too. Skipping. Error: {e2}")
            continue

    # Forecast 90 steps ahead
    step = 90
    forecast_result = result.get_forecast(steps=step)
    fc = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Plot forecast vs actual
    plt.figure(figsize=(14,6))
    plt.plot(test[:step], label="Actual", color='blue')
    plt.plot(fc, label="Forecast", color='orange')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3)
    plt.title(f"{stock_name} - 90 Day Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Plot residuals
    residuals = pd.DataFrame(result.resid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(residuals, color='purple')
    ax1.set_title(f"{stock_name} Residuals Over Time")
    ax2.hist(residuals, bins=30, density=True, color='green', alpha=0.6)
    ax2.set_title(f"{stock_name} Residuals Distribution")
    plt.tight_layout()
    plt.show()

# log return calculation
df['Log_Returns'] = df.groupby('Stock')['Close'].transform(lambda x: np.log(x / x.shift(1)))
df.dropna(subset=['Log_Returns'], inplace=True)


# Define the list of stocks
stocks = ['AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'SBIN']

# --- 2. DIAGNOSTICS: LJUNG-BOX TEST ON SQUARED RETURNS (ARCH/GARCH Test) ---

print("--- 2. Ljung-Box Test on Squared Log Returns (ARCH/GARCH Check) ---")
print("Reject H0 (p < 0.05) --> Volatility Clustering Exists (Need GARCH)")
print("-" * 70)

for stock in stocks:
    # Get the log returns for the current stock
    returns = df[df['Stock'] == stock]['Log_Returns'].dropna()

    # Calculate squared returns
    squared_returns = returns ** 2

    # Run Ljung-Box test on squared returns
    lb_test = acorr_ljungbox(squared_returns, lags=[10], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]

    if p_value < 0.05:
        decision = "Reject H0 (ARCH Effects Confirmed)"
    else:
        decision = "Do Not Reject H0 (No Strong ARCH Effects)"

    print(f"{stock}: Ljung-Box p-value (lag=10) = {p_value:.4f} --> {decision}")

# --- 3. GARCH MODELING AND FORECASTING ---

# Define the forecast horizon (e.g., 90 days, matching your ARIMA plots)
forecast_horizon = 90
GARCH_p = 1
GARCH_q = 1  # Using standard GARCH(1, 1)

garch_forecasts = {}

print("\n--- 3. Rolling GARCH(1, 1) Volatility Forecasting ---")
print(f"Using GARCH({GARCH_p}, {GARCH_q}) for all stocks.")
print("-" * 70)

for stock in stocks:
    print(f"Processing {stock}...")
    returns = df[df['Stock'] == stock]['Log_Returns'].dropna()
    
    # Define the train/test split point for the forecast
    train_size = len(returns) - forecast_horizon
    test_returns = returns.iloc[train_size:]
    
    # Rolling window forecasting for the volatility (Standard Deviation)
    rolling_predictions = []
    
    # Start loop from the point where the test set begins
    for i in range(forecast_horizon):
        # Rolling window training data: takes all data up to the current forecast day
        train_data = returns.iloc[:train_size + i] 
        
        # Define the GARCH model (mean='zero' because ARIMA handled the mean)
        # Using vol='Garch' for GARCH(p, q)
        model = arch_model(train_data * 100, mean='Zero', vol='Garch', p=GARCH_p, q=GARCH_q, dist='t')
        
        # Fit the model quietly
        model_fit = model.fit(disp='off')
        
        # Predict the next day's conditional variance
        # horizon=1 for 1-step ahead forecast
        pred = model_fit.forecast(horizon=1)
        
        # Extract the conditional volatility (Standard Deviation) and append
        # We divide by 100 because we scaled the data by 100 for better fitting
        volatility = np.sqrt(pred.variance.values[-1, 0]) / 100
        rolling_predictions.append(volatility)

    # Convert to Series with the correct date index (from the test set)
    garch_forecasts[stock] = pd.Series(rolling_predictions, index=test_returns.index)

# Define the list of stocks (repeated for clarity)
stocks = ['AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'HDFCBANK', 
          'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'SBIN']
forecast_horizon = 90
GARCH_p = 1
GARCH_q = 1

# --- 4. PLOTTING THE VOLATILITY FORECAST (Split into Two Figures) ---

# --- Figure 1: First 4 Stocks ---
stocks_fig1 = stocks[:4]
print("\n--- 4. Generating Volatility Forecast Plot - Figure 1 (4 Plots) ---")

plt.figure(figsize=(15, 12)) # Larger figure size for better clarity

for i, stock in enumerate(stocks_fig1):
    plt.subplot(2, 2, i + 1) # 2 rows, 2 columns
    
    # Recalculate train_size and test_returns for plotting context
    returns = df[df['Stock'] == stock]['Log_Returns'].dropna()
    train_size = len(returns) - forecast_horizon
    returns_test = returns.iloc[train_size:]
    
    # Calculate actual realized volatility (using squared returns)
    actual_volatility = np.sqrt(returns_test**2).rolling(window=5).mean() # 5-day rolling avg
    
    plt.plot(actual_volatility.index, actual_volatility.values, 
             label='Actual Volatility (5-day avg)', color='blue')
    
    # Ensure forecast series index aligns correctly with plot
    plt.plot(garch_forecasts[stock].index, garch_forecasts[stock].values, 
             label=f'GARCH({GARCH_p},{GARCH_q}) Forecast', color='red', linestyle='--')
    
    plt.title(f'{stock} - GARCH Volatility Forecast')
    plt.xlabel('Date Index') # Note: Using index numbers from the data frame for x-axis
    plt.ylabel('Conditional Volatility (Std. Dev.)')
    plt.legend(loc='upper right')

plt.suptitle("Figure 1: GARCH(1, 1) Volatility Forecasts (Stocks 1-4)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()

# --- Figure 2: Last 4 Stocks ---
stocks_fig2 = stocks[4:]
print("\n--- 4. Generating Volatility Forecast Plot - Figure 2 (4 Plots) ---")

plt.figure(figsize=(15, 12)) # Larger figure size for better clarity

for i, stock in enumerate(stocks_fig2):
    plt.subplot(2, 2, i + 1) # 2 rows, 2 columns
    
    # Recalculate train_size and test_returns for plotting context
    returns = df[df['Stock'] == stock]['Log_Returns'].dropna()
    train_size = len(returns) - forecast_horizon
    returns_test = returns.iloc[train_size:]
    
    # Calculate actual realized volatility (using squared returns)
    actual_volatility = np.sqrt(returns_test**2).rolling(window=5).mean()
    
    plt.plot(actual_volatility.index, actual_volatility.values, 
             label='Actual Volatility (5-day avg)', color='blue')
    
    # Ensure forecast series index aligns correctly with plot
    plt.plot(garch_forecasts[stock].index, garch_forecasts[stock].values, 
             label=f'GARCH({GARCH_p},{GARCH_q}) Forecast', color='red', linestyle='--')
    
    plt.title(f'{stock} - GARCH Volatility Forecast')
    plt.xlabel('Date Index')
    plt.ylabel('Conditional Volatility (Std. Dev.)')
    plt.legend(loc='upper right')

plt.suptitle("Figure 2: GARCH(1, 1) Volatility Forecasts (Stocks 5-8)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()





bank_mape = {}
bank_rmse = {}
bank_strategy_returns = {}
bank_passive_returns = {}

print("\n\n🔁 Running post-analysis for MAPE, RMSE, and Strategy Backtest...")

for stock_name in arima_orders.keys():
    print(f"\n🔎 {stock_name}:")

    stock_df = df[df["Stock"] == stock_name].copy()
    stock_df.set_index("Date", inplace=True)
    stock_df = stock_df.asfreq('B')
    stock_df = stock_df.fillna(method='ffill')

    n = int(len(stock_df) * 0.8)
    train = stock_df['Close'][:n]
    test = stock_df['Close'][n:]

    order = arima_orders.get(stock_name, default_order)

    try:
        model = ARIMA(train, order=order)
        result = model.fit()
        step = 90

        forecast_result = result.get_forecast(steps=step)
        fc = forecast_result.predicted_mean

        if len(test) >= step:
            actual = test[:step].values
            forecast = fc.values

            # ✅ MAPE & RMSE
            mape = mean_absolute_percentage_error(actual, forecast) * 100
            rmse = np.sqrt(mean_squared_error(actual, forecast))

            bank_mape[stock_name] = mape
            bank_rmse[stock_name] = rmse

            print(f"MAPE: {mape:.2f}% | RMSE: {rmse:.2f}")

            # ✅ Backtest Trading Strategy
            actual_returns = pd.Series(np.diff(actual), index=test[:step].index[1:])
            predicted_returns = pd.Series(np.diff(forecast), index=test[:step].index[1:])

            strategy_returns = actual_returns * (predicted_returns > 0)
            cumulative_strategy_return = strategy_returns.sum()
            cumulative_passive_return = actual_returns.sum()

            bank_strategy_returns[stock_name] = cumulative_strategy_return
            bank_passive_returns[stock_name] = cumulative_passive_return

            print(f"Strategy Return: {cumulative_strategy_return:.2f} | Passive Return: {cumulative_passive_return:.2f}")

        else:
            print("⚠️ Not enough test data for 90-step forecast.")

    except Exception as e:
        print(f"❌ Could not fit ARIMA for {stock_name}. Skipping. Error: {e}")
        continue

# 🔚 Final Summary Block
print("\n\n📌 Final Summary:")
print("🔹 MAPE & RMSE:")
for stock in bank_mape:
    print(f"{stock}: MAPE = {bank_mape[stock]:.2f}%, RMSE = {bank_rmse[stock]:.2f}")
print(f"\nAverage MAPE: {np.mean(list(bank_mape.values())):.2f}%")
print(f"Average RMSE: {np.mean(list(bank_rmse.values())):.2f}")

print("\n🔹 Strategy vs Passive Return:")
for stock in bank_strategy_returns:
    s = bank_strategy_returns[stock]
    p = bank_passive_returns[stock]
    delta = s - p
    print(f"{stock}: Strategy = {s:.2f}, Passive = {p:.2f} → Δ = {delta:.2f}")
