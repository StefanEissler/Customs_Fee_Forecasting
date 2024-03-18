from flask import Flask, request, jsonify
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
model = make_pipeline(StandardScaler(), SGDRegressor())

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json 

    X = np.array(data['features']).reshape(-1, 1)  
    y = np.array(data['labels']) 

    model.partial_fit(X, y)

    return 'Model trained successfully!', 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json 

    X = np.array(data['features']).reshape(-1, 1)

    predictions = model.predict(X)

    return jsonify({'predictions': predictions.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
