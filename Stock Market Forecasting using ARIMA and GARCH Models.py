import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    suggest_q(ts_diff, stock)

# ARIMA MODEL
# Your specific ARIMA orders per stock (based on what you gave earlier)
arima_orders = {
    'AXISBANK': (8, 1, 0),
    'BAJAJFINSV': (10, 1, 4),
    'BAJFINANCE': (7, 1, 0),
    'HDFCBANK': (1, 1, 0),
    'ICICIBANK': (20, 1, 0),
    'INDUSINDBK': (7, 1, 3),
    'KOTAKBANK': (6, 1, 1),
    'SBIN': (3, 1, 1)
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

# Run Ljung-Box test on squared returns for all stocks
for stock in df['Stock'].unique():
    squared_returns = df[df['Stock'] == stock]['Log_Returns'] ** 2
    lb_test = acorr_ljungbox(squared_returns, lags=[10], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    
    print(f"{stock}: Ljung-Box p-value (lag=10) = {p_value:.4f} --> {'Reject H0 (ARCH effects)' if p_value < 0.05 else 'Do not reject H0'}")

# Step 3: PACF plot for squared log returns
for stock in df['Stock'].unique():
    plt.figure(figsize=(10, 4))
    stock_returns = df[df['Stock'] == stock]['Log_Returns']
    plot_pacf(stock_returns ** 2, lags=40, title=f'PACF of Squared Returns - {stock}')
    plt.tight_layout()
    plt.show()

# bajajfinsv 
returns_bajaj = df[df['Stock'] == 'BAJAJFINSV']['Log_Returns'].dropna()

# Rolling forecast using ARCH(3) = GARCH(3,0)
rolling_predictions_bajaj = []
test_size = 365

for i in range(test_size):
    train = returns_bajaj[:-(test_size - i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions_bajaj.append(np.sqrt(pred.variance.values[-1, :][0]))

# Convert to Series
rolling_predictions_bajaj = pd.Series(rolling_predictions_bajaj, index=returns_bajaj.index[-test_size:])

# Plot
plt.figure(figsize=(10, 4))
plt.plot(returns_bajaj[-test_size:], label='True Returns')
plt.plot(rolling_predictions_bajaj, label='Predicted Volatility')
plt.title('Volatility Prediction - BAJAJFINSV (ARCH(3))', fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()


# Filter INDUSINDBK returns
returns_indus = df[df['Stock'] == 'INDUSINDBK']['Log_Returns'].dropna()

# Rolling forecast using GARCH(2,2)
rolling_predictions_indus = []

for i in range(test_size):
    train = returns_indus[:-(test_size - i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions_indus.append(np.sqrt(pred.variance.values[-1, :][0]))

# Convert to Series
rolling_predictions_indus = pd.Series(rolling_predictions_indus, index=returns_indus.index[-test_size:])

# Plot
plt.figure(figsize=(10, 4))
plt.plot(returns_indus[-test_size:], label='True Returns')
plt.plot(rolling_predictions_indus, label='Predicted Volatility')
plt.title('Volatility Prediction - INDUSINDBK (GARCH(2,2))', fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

