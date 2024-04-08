import os
import pickle

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.neuralforecast import NeuralForecastRNN

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

class BaseModel:
    def __init__(self):
        self.trained_model = None
    
    def save_model(self, customer_id, model_type):
        if self.trained_model is None:
            raise ValueError("The Model is not trained.")
        
        model_filename = f"./models/{customer_id}_{model_type}_model.pkl"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        with open(model_filename, 'wb') as file:
            pickle.dump(self.trained_model, file)
        print(f"Trained model saved as {model_filename}.")
        
    def load_model(self, customer_id, model_type):
        model_filename = os.path.join("./models", f"{customer_id}_{model_type}_model.pkl")
        if not os.path.exists(model_filename):
            raise FileNotFoundError("Saved Model not found.")
        
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        self.trained_model = loaded_model
        
    def evaluate(self, prediction, X_test, y_train, y_test):
        mae = mean_absolute_error(y_true=y_test, y_pred=prediction)
        mse = mean_squared_error(y_true=y_test, y_pred=prediction)
        r2 = r2_score(y_true=y_test, y_pred=prediction)
        forecast_bias = (prediction - y_test).mean()
            
        mase = MeanAbsoluteScaledError()
        mean_MASE = mase(y_true=y_test, y_pred=prediction, y_train=y_train)
        forecast_accuracy = (1 / mean_MASE) * 100 if mean_MASE != 0 else float('inf')
            
        return {
            'MAE': mae,
            'MSE': mse,
            'meanMASE': mean_MASE,
            'r2': r2,
            'Forecast Bias': forecast_bias,
            'Forecast Accuracy (%)': forecast_accuracy
        }

    def forecast():
        pass

    def predict():
        pass
    
class ArimaModel(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, X_test):
        model_arima = AutoARIMA(
            start_p=1, 
            start_q=1,
            test='adf', # use adftest to find optimal 'd'
            max_p=3, 
            max_q=3, # maximum p and q
            d=None,# let model determine 'd'
            seasonal=False, # No Seasonality for standard ARIMA
            trace=True,
            error_action='warn', #shows errors 
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
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, X_test):
        model_ETS = AutoETS(auto=True, random_state=120)
        model_ETS.fit(y=y_train)
        self.trained_model = model_ETS
        return model_ETS

    def forecast(self, X_test):
        fh = ForecastingHorizon(X_test.index, is_relative=False) 
        prediction = self.trained_model.predict(fh=fh)    
        return prediction
    
class ForestModel(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, X_test):
        model_forest = RandomForestRegressor(n_estimators=150)
        model_forest.fit(X_train, y_train)
        
        self.trained_model = model_forest
        return model_forest
    
    def forecast(self, X_test):  
        prediction = self.trained_model.predict(X_test)    
        return prediction
    
class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, X_test):
        model_xgbosst = GradientBoostingRegressor(random_state=150)
        model_xgbosst.fit(X_train, y_train)
        
        self.trained_model = model_xgbosst
        return model_xgbosst
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X_test)  
        return prediction
    
class RNNModel(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, X_test):
        fh = ForecastingHorizon(X_test.index, is_relative=False)
        model_simplernn = NeuralForecastRNN("A-DEC", max_steps=20)
        model_simplernn.fit(y=y_train, X=X_train, fh=fh)
        
        self.trained_model = model_simplernn
        return model_simplernn
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X=X_test)  
        return prediction


class LTSMModel(BaseModel):
    def __init__(self):
        super().__init__()

    #def train(self, X_train, y_train):
        
        #model_ltsm = NeuralForecastLSTM()
        #model_ltsm.fit(X_train, y_train)
        
        #self.trained_model = model_ltsm
        #return model_ltsm
    
    def forecast(self, X_test):
        prediction = self.trained_model.predict(X=X_test)  
        return prediction