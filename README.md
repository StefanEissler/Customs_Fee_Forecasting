# Forecasting von Customs Duties - Bachelor Thesis Stefan Eißler 

## Installation

``pip install -r requirements.txt ``

### Mit SSL Error WSL Ubuntu

``pip install --trusted-host pypi.org --trusted-host=files.pythonhosted.org -r requirements.txt``

oder

``pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org [dependency]``

## Projektsturktur

1. Data - Ablage von Beispieldaten
a) processed - Bearbeitete Datensätze 
2. src - root der App
a) app - flask Server mit conifg
b) models - Ablage trainierter Modelle
c) notebooks - Notebooks zur Datenvorbereitung etc.
d) test - Testing Module

## Projekt Architektur

Die beste Performance auf Zeitreihendaten hatten X Modelle. Diese werden nun in der Ensembling Module trainiert. Durch das X Objekt im X Modul können neue Modelle trainiert und gespeichert werden. Über die API können diese nun trainiert und predictions eingereicht werden.

## Run Application

``python3 src/app/api.py``

## API Beschreibung

### Request

`POST /train`

    curl -i -H 'Accept: application/json' http://localhost:5000/predict

### Request body

`
{
    "customerid": "example",    
    "modeltype": "ets",
    "data":[
        {
            "Abgabenbescheid" : {
            "Gesamtabgabe" : 305.33,
            "Datum Erstellung" : "2014-11-13"
            }
        }
        ,{
            "Abgabenbescheid" : {
                "Gesamtabgabe" : 0,
                "Datum Erstellung" : "2016-01-29"
            }
        }
    ]
}
`

#### customerid
- Kundenname

#### modeltype
- arima
- ets
- forest
- xgboost
- rnn

#### data
- Data in AEB Format as JSON

### Response

    HTTP/1.1 200 OK
    Server: Werkzeug/3.0.1 Python/3.10.12
    Date: Mon, 08 Apr 2024 13:18:33 GMT
    Content-Type: application/json
    Content-Length: 333
    Connection: close

    [{
        "message": "ets for example trained and saved successfully",
        "success": true,
        "validation_matrix": {
            "Forecast Accuracy (%)": 4.0085900928985385,
            "Forecast Bias": -3145.8878176275794,
            "MAE": 5208.024764868966,
            "MSE": 63403670.98435483,
            "meanMASE": 24.946426968713038,
            "r2": -0.18495895698303766
        }
    }]

## Create a new Thing

### Request

`POST /forecast`

    curl -i -H 'Accept: application/json' http://localhost:5000/forecast

### Request body
`
{
    "customerid": "example",
    "modeltype": "forest",
    "horizon": "60"
}
`

#### customerid
- Kundenname

#### modeltype
- arima
- ets
- forest
- xgboost
- rnn

#### horizon
- gibt den Horizont des Forecasts in Tagen

### Response

    HTTP/1.1 200 OK
    Server: Werkzeug/3.0.1 Python/3.10.12
    Date: Mon, 08 Apr 2024 13:45:46 GMT
    Content-Type: application/json
    Content-Length: 847
    Connection: close

`
{
    "data": {
        "2024-04-09": 23.424447619047598,
        "2024-04-10": 22.49140952380951,
        "2024-04-11": 22.49140952380951,
        "2024-04-12": 21.558371428571416,
        "2024-04-13": 19.725590476190465,
        "2024-04-14": 21.275561904761897,
        "2024-04-15": 10.93853333333333,
        "2024-04-16": 9.763857142857145,
        "2024-04-17": 12.773714285714286,
        "2024-04-18": 12.828152380952382,
        "2024-04-19": 22.335238095238093,
        "2024-04-20": 16.00308571428571,
        "2024-04-21": 16.00308571428571,
        "2024-04-22": 19.06738095238095,
        "2024-04-23": 83.14954285714285,
        "2024-04-24": 218.10132380952388,
        "2024-04-25": 218.10132380952388,
        "2024-04-26": 205.6808285714286,
        "2024-04-27": 202.61653333333336,
        "2024-04-28": 202.61653333333336
    },
    "message": "forest for example predicted successfully",
    "success": true
}
`


## Libaries

- pandas [https://pandas.pydata.org/]
- numpy [https://numpy.org/doc/stable/index.html]
- matplotlib [https://matplotlib.org/stable/]
- scikit-learn [https://scikit-learn.org/]
- tensorflow / keras [https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM]
- statsmodels [https://www.statsmodels.org/devel/index.html]
- sktime [https://www.sktime.net/en/latest/index.html]
- flask [https://flask.palletsprojects.com/en/3.0.x/]

