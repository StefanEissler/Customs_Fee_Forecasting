from abc import ABC, abstractmethod

import numpy as np
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.forecasting.neuralforecast import NeuralForecastRNN
from neuralforecast.losses.pytorch import MAE

from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import mean_absolute_error, r2_score
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

class Evaluator:
    @staticmethod
    def evaluate(prediction, y_train, y_test):
        y_test = np.array(y_test)
        y_train = np.array(y_train)
        
        mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
        r2 = r2_score(y_true=y_test, y_pred=prediction)
        forecast_bias = (prediction.sum() - y_test.sum()) / y_test.sum() * 100 
        mape = mean_absolute_percentage_error(y_true=y_test, y_pred=prediction, symmetric=True)
        mase = MeanAbsoluteScaledError()
        mase = mase(y_true=y_test, y_pred=prediction, y_train=y_train)
            
        return {
            'Mean Absolute Error': mae,
            'Mean Absolute Percentage Error (%)': mape * 100,
            'Mean Absolute Scaled Error': mase,
            'r2': r2,
            'Forecast Bias (%)': forecast_bias
        }

class BaseModel(ABC):
    def __init__(self, trained_model=None):
        self.trained_model = trained_model
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def forecast(self, X_test):
        pass
    
class ArimaModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)

    def train(self, X_train, y_train):
        model_arima = AutoARIMA(
            start_p=1, 
            start_q=1,
            test='adf', # ad_fuller test
            max_p=3, 
            max_q=3,
            d=None,
            trace=True,
            stepwise=True
        )
        model_arima.fit(y_train)
        self.trained_model = model_arima
        return model_arima
    
    def forecast(self, X_test):
        fh = ForecastingHorizon(X_test.index, is_relative=False) 
        prediction = self.trained_model.predict(fh=fh)    
        return prediction
    
class ETSModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)

    def train(self, X_train, y_train):
        model_ETS = AutoETS(auto=True, error="add", trend="add", seasonal="add", random_state=120)
        model_ETS.fit(y=y_train)
        self.trained_model = model_ETS
        return model_ETS

    def forecast(self, X_test):
        fh_length = len(X_test)
        steps = range(1, fh_length + 1)
        prediction = self.trained_model.predict(fh=steps)    
        return prediction
    
class ForestModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)

    def train(self, X_train, y_train):
        model_forest = RandomForestRegressor(n_estimators=120, random_state=120)
        model_forest.fit(X_train, y_train)
        self.trained_model = model_forest
        return model_forest
    
    def forecast(self, X_test):  
        prediction = self.trained_model.predict(X_test)    
        return prediction
    
class XGBoostModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)

    def train(self, X_train, y_train):
        model_xgboost = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.5, random_state=12)
        model_xgboost.fit(X_train, y_train)
        self.trained_model = model_xgboost
        return model_xgboost
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X_test)  
        return prediction
    
class RNNModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)
    
    def train(self, X_train, y_train, X_test):
        lstm_model = NeuralForecastRNN( 
                freq="D",
                max_steps=120,
                num_workers_loader=11
            )
        lstm_model.fit(y_train, X=X_train, fh=X_test.index)
        self.trained_model = lstm_model
        return lstm_model
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X=X_test)  
        return prediction

class LSTMModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)
    
    def train(self, X_train, y_train, X_test):
        lstm_model = NeuralForecastLSTM( 
                freq="D",
                max_steps=120,
                local_scaler_type='robust',
                loss=MAE(),
                #hist_exog_list=["deklarationen_pro_tag", "lag_2__Abgabe_movavg", "lag_4__Abgabe_movavg", "lag_6__Abgabe_movavg"],
                num_workers_loader=11,
            )
        lstm_model.fit(y_train, X=X_train, fh=X_test.index)
        self.trained_model = lstm_model
        return lstm_model
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X=X_test)  
        return prediction

'''
class EnsembleModel(BaseModel):
    def __init__(self, trained_model=None):
        super().__init__(trained_model)

    def train(self, X_train, y_train, X_test):
        fh = ForecastingHorizon(values=X_test.index, is_relative=False)
        forecasters = [
            ("naive", NaiveForecaster())
        ]
        model_ensemble = AutoEnsembleForecaster(forecasters=forecasters)
        model_ensemble.fit(X_train, y_train, fh)
        self.trained_model = model_ensemble
        return model_ensemble
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X=X_test)  
        return prediction
'''
