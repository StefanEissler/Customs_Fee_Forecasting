# Forecasting von Customs Duties - Bachelor Thesis Stefan Eißler 

## Installation

``pip install -r requirements.txt ``

### Mit SSL Error WSL Ubuntu

``pip install --trusted-host pypi.org --trusted-host=files.pythonhosted.org -r requirements.txt ``

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

## API Beschreibung


## Libaries

- pandas [https://pandas.pydata.org/]
- numpy
- matplotlib [https://matplotlib.org/stable/]
- scikit-learn [https://scikit-learn.org/]
- statsmodels [https://www.statsmodels.org/devel/index.html]
- flask [https://flask.palletsprojects.com/en/3.0.x/]


## Notizen:

Mögliche weitere Libaries:
- sktime [https://sktime-backup.readthedocs.io/en/stable/examples/01_forecasting.html]
- tensorflow [https://www.tensorflow.org/]
