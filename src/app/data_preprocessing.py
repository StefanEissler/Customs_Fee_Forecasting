import numpy as np
import pandas as pd

from models import ArimaModel, ETSModel, ForestModel, LTSMModel, RNNModel, XGBoostModel

from sktime.transformations.series.lag import Lag
from sktime.transformations.series.impute import Imputer

def prep_data(data):
    
    # Felder zuordnen
    extracted_data = []
    for _, row in data.iterrows():
        abgabenbescheid = row['Abgabenbescheid']['Gesamtabgabe']
        datum_erstellung = row['Abgabenbescheid']['Datum Erstellung']
        
        extracted_data.append({'Abgabe': abgabenbescheid, 'Datum': datum_erstellung})

    df = pd.DataFrame(extracted_data)
    df.Datum = pd.to_datetime(df.Datum)
    df.set_index('Datum', inplace=True)
    
    # Anzahl der Deklarationen am Tag extrahieren
    deklarationen_pro_tag = df.groupby(df.index).size().reset_index(name='deklarationen_pro_tag')
    deklarationen_pro_tag.set_index('Datum', inplace=True)
    deklarationen_pro_tag = deklarationen_pro_tag.resample('D').asfreq().fillna(0)
    
    # Abgabenschuld pro Tag
    abgabe_pro_tag = df.groupby(df.index)['Abgabe'].sum().reset_index()
    abgabe_pro_tag.set_index('Datum', inplace=True)
    abgabe_pro_tag = abgabe_pro_tag.resample('D').asfreq().fillna(0)
    
    # Abgabe gleitender Durchschnitt
    abgabe_pro_tag['Abgabe_movavg'] = abgabe_pro_tag.Abgabe.rolling(window=7).mean()
    
    # Erstellen von lag features
    t = Lag([2,4,6]) * Imputer("nearest")
    lags = t.fit_transform(abgabe_pro_tag.Abgabe_movavg.dropna())
    
    # Zusammenführung der Komponenten
    merged_df = pd.merge(abgabe_pro_tag, lags, left_index=True, right_index=True, how='right')
    combined_df = pd.merge(merged_df, deklarationen_pro_tag, left_index=True, right_index=True, how='right')
    combined_df = combined_df.dropna(subset=["Abgabe_movavg"])
    df = combined_df
    df = df.drop("Abgabe", axis=1)
    
    # Datum aufsplitten in eigene Zeilen
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    
    return df

def selectModel(modeltype):
    if modeltype == 'arima':
        return ArimaModel()
    elif modeltype == 'ets':
        return ETSModel()
    elif modeltype == 'forest':
        return ForestModel()
    elif modeltype == 'xgboost':
        return XGBoostModel()
    elif modeltype == 'rnn':
        return RNNModel()
    elif modeltype == 'lstm':
        return LTSMModel()
    else:
        return None

def create_forecasting_horizon(horizon):
    current_date = pd.to_datetime('now')
    # Erstellen des Datumsindexes für den Horizon
    horizon_dates = pd.date_range(start=current_date, periods=horizon, freq='D') + pd.Timedelta(days=1)  # Beginnen Sie am nächsten Tag
    # Erstellen des DataFrame für den Forecast-Horizont
    df = pd.DataFrame(index=horizon_dates)
    
    # Hinzufügen der erforderlichen Spalten
    df['lag_2__Abgabe_movavg'] = np.nan
    df['lag_4__Abgabe_movavg'] = np.nan
    df['lag_6__Abgabe_movavg'] = np.nan
    df['deklarationen_pro_tag'] = np.nan
    
    # Extrahieren von Jahr, Monat und Tag aus dem Datumindex
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    
    # Wiederherstellen der 'Datum'-Spalte
    df['Datum'] = df.index
    df = df.drop('Datum', axis=1)
    
    # Umsortierung der Spalten
    df = df[['lag_2__Abgabe_movavg', 'lag_4__Abgabe_movavg', 'lag_6__Abgabe_movavg', 'deklarationen_pro_tag', 'Year', 'Month', 'Day']]
    
    return df
