import pickle
import numpy as np

# Load pre-trained models
knn_model = pickle.load(open("data/models/knn_model.pkl", "rb"))
random_forest_model = pickle.load(open("data/models/random_forest.pkl", "rb"))

def predict_cognitive_load(features, model_type="knn"):
    """
    Predict cognitive load by using either KNN or Random Forest models.
    """
    features = np.array(features).reshape(1, -1)
    if model_type == "knn":
        return knn_model.predict(features)[0]
    elif model_type == "random_forest":
        return random_forest_model.predict(features)[0]
