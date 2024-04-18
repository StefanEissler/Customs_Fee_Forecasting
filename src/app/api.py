from datetime import datetime
import pandas as pd
import os

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from data_preprocessing import prep_data, create_forecasting_horizon
from models import ArimaModel, ETSModel, ForestModel, LSTMModel, RNNModel, XGBoostModel
from modelio import ModelIO
from models import Evaluator

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    #Request JSON auslesen
    data = request.json
    customer_id = data.get('customerid')
    data = pd.DataFrame(data["data"])
    data = prep_data(data)
    
    # X, y
    X = data.drop("Abgabe_movavg", axis=1)
    y = data["Abgabe_movavg"]

    #Modeltraining
    try:
        modeltype = 'forest' 
        forestmodel = ForestModel()
        forestmodel.train(X, y)
        modelio = ModelIO()
        modelio.save_model(forestmodel, customer_id=customer_id, model_type=modeltype)
    except Exception as e:
        return jsonify({'success': False, 'message': 'Model Training failed', 'Exception': str(e)}), 500
    
    response_data = {
        'success': True,
        'message': f'Forecasting Model for {customer_id} trained and saved successfully',
    }    
    return jsonify(response_data), 200

@app.route('/forecast', methods=['GET'])
def forecast():
    data = request.json
    customer_id = data.get('customerid')
    horizon = data.get('horizon')
    
    # DataFrame erstellen
    df = create_forecasting_horizon(horizon)
    
    #Forecast
    try:
        modeltype = 'forest'
        foestmodel = ForestModel()
        foestmodel.load_model(customer_id, modeltype)
        predictions = foestmodel.forecast(df)
        df.index = df.index.strftime('%Y-%m-%d')
        predictions_dict = {str(date): prediction for date, prediction in zip(df.index, predictions)}
    except Exception as e:
        return jsonify({'success': False, 'message': 'Prediction failed', 'Exception': str(e)}), 500
    
    responseData = {
        'data': predictions_dict,
        'message': f'Forecasting for {customer_id} generated successfully',
        'success': True
    }
    return jsonify(responseData), 200

@app.route('/evaluate', methods=['POST'])
def evaluate():
    #Request JSON auslesen
    data = request.json
    modeltype = data.get('modeltype')
    customer_id = data.get('customerid')
    data = pd.DataFrame(data["data"])
    data = prep_data(data)
    
    # X, y und train, test split
    X = data.drop("Abgabe_movavg", axis=1)
    y = data["Abgabe_movavg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=90, random_state=42, shuffle=False)

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
        model = LSTMModel()
    else:
        return jsonify({'success': False, 'message': 'Model name could not be found'}), 400

    # Modeltraining
    try: 
        if(modeltype == 'rnn' or modeltype == 'lstm'):
            model.train(X_train, y_train, y_test)
        else:
            model.train(X_train, y_train)
        prediction = model.forecast(X_test)
        evaluater = Evaluator()
        validation_matrix = evaluater.evaluate(prediction, y_train, y_test)
    except Exception as e:
        return jsonify({'success': False, 'message': 'Model Training failed', 'Exception': str(e)}), 500
    
    # Prediction und Evaluation ablegen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    filename = f"./data/prediction/{customer_id}_{modeltype}_{timestamp}_prediction.csv"
    prediction_evaluation = pd.DataFrame({
            'y_test': y_test, 
            'prediction': prediction
    }, index=y_test.index)
    prediction_evaluation.to_csv(filename)

    # Evaluationsmatrix ablegen
    filename_matrix = f"./data/evaluation/{customer_id}_evaluation.csv"
    validation_df = pd.DataFrame(validation_matrix, index=[0])
    validation_df['modeltype'] = modeltype
    if os.path.exists(filename_matrix):
        existing_evaluation = pd.read_csv(filename_matrix)
        existing_evaluation = pd.concat([existing_evaluation, validation_df], axis=0)
        existing_evaluation.to_csv(filename_matrix, index=False)
    else:
        validation_df.to_csv(filename_matrix, index=False)
    
    response_data = {
        'success': True,
        'message': f'Evaluation with {modeltype} for {customer_id} generated successfully',
        'validation_matrix': validation_matrix 
    }    
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True)