# Forecasting von Customs Duties - Bachelor Thesis Stefan Eißler 

## Installation

``pip install -r requirements.txt ``

### Mit SSL Error WSL Ubuntu

``pip install --trusted-host pypi.org --trusted-host=files.pythonhosted.org -r requirements.txt``

oder

``pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org [dependency]``

## Projektsturktur

1. Data - Ablage von Beispieldaten
a. processed - Bearbeitete Datensätze 
b. raw - Datensätze nach AEB Format
2. models - Speichern der trainierten Modelle
3. notebooks - Datenuntersuchung mittels Jupyter Notebooks
4. src - root der App (Flask Server)

## Projekt Architektur

Die beste Performance auf Zeitreihendaten hatten X Modelle. Diese werden nun in der Ensembling Module trainiert. Durch das X Objekt im X Modul können neue Modelle trainiert und gespeichert werden. Über die API können diese nun trainiert und predictions eingereicht werden.

## Run Application

``python3 src/app/api.py``

# API Beschreibung

## Training des Models
Trainiert ein Model und legt es mit dem Kundenname ab.

`POST /train`

    curl -i -H 'Accept: application/json' http://localhost:5000/predict

### Request body

    {
        "customerid": "example",    
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

#### customerid
- Kundenname

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
    }]

## Forecast erstellen
Erstellt einen Forecast für einen Kunden über einen gewissen Horizont.

### Request

`POST /forecast`

    curl -i -H 'Accept: application/json' http://localhost:5000/forecast

### Request body

    {
        "customerid": "example",
        "horizon": 60
    }


#### customerid
- Kundenname

#### horizon
- gibt den Horizont des Forecasts in Tagen

### Response

    HTTP/1.1 200 OK
    Server: Werkzeug/3.0.1 Python/3.10.12
    Date: Mon, 08 Apr 2024 13:45:46 GMT
    Content-Type: application/json
    Content-Length: 847
    Connection: close

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


### Evaluationsmatrix erstellen
Trainiert ein Model mit einem train test split und gibt die Ergebnisse der Evaluation aus.

`POST /evaluate`

    curl -i -H 'Accept: application/json' http://localhost:5000/evaluate

### Request body

    {
        "customerid": "example",   
        "modeltype": "forest" 
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

#### modeltype
- ARIMA
- ETS
- forest
- xgboost
- rnn
- ltsm

#### customerid
- Kundenname

#### data
- Data in AEB Format als JSON

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
        "validation_result": {
            "Forecast Accuracy (%)": 10.538043968968688,
            "Forecast Bias": -1974.3481661932624,
            "MAE": 3880.514737872459,
            "MSE": 39054173.60055442,
            "meanMASE": 9.48942709809044,
            "r2": 0.4348866438055564
        }
    }]

## Libaries

- pandas [https://pandas.pydata.org/]
- numpy [https://numpy.org/doc/stable/index.html]
- matplotlib [https://matplotlib.org/stable/]
- scikit-learn [https://scikit-learn.org/]
- statsmodels [https://www.statsmodels.org/devel/index.html]
- sktime [https://www.sktime.net/en/latest/index.html]
- flask [https://flask.palletsprojects.com/en/3.0.x/]

