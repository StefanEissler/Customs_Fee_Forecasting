import pandas as pd

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from data_preprocessing import prep_data, create_forecasting_horizon
from models import ArimaModel, ETSModel, ForestModel, LTSMModel, RNNModel, XGBoostModel

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    #Request JSON auslesen
    data = request.json
    modeltype = data.get('modeltype')
    customer_id = data.get('customerid')
    data = pd.DataFrame(data["data"])
    data = prep_data(data)
    
    # X, y und train, test split
    X = data.drop("Abgabe_movavg", axis=1)
    y = data["Abgabe_movavg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)

    # Modelauswahl
    if modeltype == 'arima':
        model = ArimaModel()
    elif modeltype == 'ets':
        model = ETSModel()
    elif modeltype == 'forest':
        model = ForestModel()
    elif modeltype == 'xgboost':
        model = XGBoostModel()
    elif modeltype == 'rnn':
        model = RNNModel()
    elif modeltype == 'lstm':
        model = LTSMModel()
    else:
        return jsonify({'success': False, 'message': 'Model name could not be found'}), 400

    #Modeltraining
    try: 
        model.train(X_train, y_train, X_test)
        prediction = model.forecast(X_test)
        validation_matrix = model.evaluate(prediction, X_test, y_train, y_test)
        model.save_model(customer_id=customer_id, model_type=modeltype)
    except Exception as e:
        return jsonify({'success': False, 'message': 'Model Training failed', 'Exception': str(e)}), 500
    
    response_data = {
        'success': True,
        'message': f'{modeltype} for {customer_id} trained and saved successfully',
        'validation_matrix': validation_matrix 
    }    
    return jsonify(response_data), 200

@app.route('/forecast', methods=['GET'])
def predict():
    data = request.json
    modeltype = data.get('modeltype')
    customer_id = data.get('customerid')
    horizon = data.get('horizon')
    
    # Model von File lesen
    if modeltype == 'arima':
        model = ArimaModel()
    elif modeltype == 'ets':
        model = ETSModel()
    elif modeltype == 'forest':
        model = ForestModel()
    elif modeltype == 'xgboost':
        model = XGBoostModel()
    elif modeltype == 'rnn':
        model = RNNModel()
    elif modeltype == 'lstm':
        model = LTSMModel()
    else:
        return jsonify({'success': False, 'message': 'Model name could not be found'}), 400
    
    # DataFrame erstellen
    df = create_forecasting_horizon(horizon)
    
    #Forecast
    try:
        model.load_model(customer_id, modeltype)
        predictions = model.forecast(df)
        df.index = df.index.strftime('%Y-%m-%d')
        predictions_dict = {str(date): prediction for date, prediction in zip(df.index, predictions)}
    except Exception as e:
        return jsonify({'success': False, 'message': 'Prediction failed', 'Exception': str(e)}), 500
    
    responseData = {
        'data': predictions_dict,
        'message': f'{modeltype} for {customer_id} predicted successfully',
        'success': True
    }
    return jsonify(responseData), 200

if __name__ == '__main__':
    app.run(debug=True)