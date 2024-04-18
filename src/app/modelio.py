import os
import pickle

from abc import ABC, abstractmethod

class ModelIO(ABC):
    
    @abstractmethod
    def save_model(self, customer_id: str, modeltype: str): 
        pass
    
    @abstractmethod
    def load_model(self, customer_id: str, modeltype: str): 
        pass
       

class LocalModelIO(ModelIO):
    
    def save_model(self, model, customer_id, modeltype):
        if model is None:
            raise ValueError("The Model is not trained.")
        
        model_filename = f"./models/{customer_id}_{modeltype}_model.pkl"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Trained model saved as {model_filename}.")
    
    def load_model(self, customer_id, modeltype):
        model_filename = os.path.join("./models", f"{customer_id}_{modeltype}_model.pkl")
        if not os.path.exists(model_filename):
            raise FileNotFoundError("Saved Model not found.")
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
        
    