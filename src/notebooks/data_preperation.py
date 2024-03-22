import pandas as pd
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.api import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# format the AEB_intern format to a dataframe with Abgabe, amount and date
def process_data(df, date_column, amount_column):
    df[date_column] = pd.to_datetime(df[date_column], format='%d.%m.%Y')
    df[amount_column] = df[amount_column].str.replace(',', '.').astype(float)
    date_import_declartions = df.groupby(date_column)[amount_column].sum().reset_index()
    count_declarations_per_day = df.groupby(date_column).size().reset_index(name='Amount_of_declarations_per_day')
    merged_data = pd.merge(date_import_declartions, count_declarations_per_day, on=date_column, how='left')
    
    return merged_data

# split data in train and test set
def split_data(data, test_split):
    X = data
    train_size = int(len(X) * test_split)
    train, test = X[0:train_size], X[train_size:len(X)]
    print(f'train: {len(train)} , test: {len(test)}')
    return train, test

# check if the data is stationary (Atwan 2022, S. 336)
def check_stationarity(df): 
    results = adfuller(df)[1:3]
    s = 'Non-Stationary'
    if results[0] < 0.05:
        s = 'Stationary'
    print(f"'{s}\t p-value:{results[0]} \t lags:{results[1]}")
    return (s, results[0])

# plot a forecast (Atwan 2022, S. 337)
def plot_forecast(model, start, train, test):
    forecast = pd.DataFrame(model.forecast(test. shape[0]), index=test.index)
    ax = train.loc[start:].plot(style='--')
    test.plot(ax=ax)
    forecast.plot(ax=ax, style = '-.')
    ax.legend(['orig_train', 'orig_test', 'forecast'])
    plt.show()
    
def get_validation_matrix(prediction, test):
    mae = mean_absolute_error(test, prediction)
    mse = mean_squared_error(test, prediction)
    r2 = r2_score(test, prediction) # beschreibt die "Anpassungsgüte einer Regression"
    forecast_bias = (prediction - test).mean()
    
    naive_forecast = test.shift(1).dropna()
    scaled_errors = abs(prediction - test) / abs(test - naive_forecast)
    mean_MASE = scaled_errors.mean()
    
    forecast_accuracy = (1 / mean_MASE) * 100 if mean_MASE != 0 else float('inf')

    
    return {
        'MAE': mae,
        'MSE': mse,
        'meanMASE': mean_MASE,
        'r2': r2,
        'Forecast Bias': forecast_bias,
        'Forecast Accuracy (%)': forecast_accuracy
    }
    
def save_metrics_to_csv(metrics):
    # Check if a file is generated already
    filename = 'metrics.csv'
    try:
        with open(filename, 'r') as f:
            existing_metrics = csv.DictReader(f)
            fieldnames = existing_metrics.fieldnames
            write_header = False
    except FileNotFoundError:
        write_header = True

    # Write the metrics to the CSV file
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames if not write_header else metrics.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(metrics)

    print(f"Metrics saved to {filename}")
    
    
def create_lags(data, lags_amount):
    lagged_features = pd.DataFrame(index=data.index)
    for i in range(1, lags_amount + 1):
        lagged_features[f'lag_{i}'] = data['Abgabe'].shift(i)
    return lagged_features

def create_rolling_avg(data, window_size):
    data = data.rolling(window=window_size).mean()
    data = data.dropna(axis=1)