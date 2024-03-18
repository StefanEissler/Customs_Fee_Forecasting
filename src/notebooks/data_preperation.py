import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import adfuller



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
    l = len(data)
    t_idx = round(l*(1-test_split))
    train, test = data[ : t_idx], data[t_idx : ]
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