from sklearn.base import BaseEstimator, TransformerMixin

class PredictAndDecode(BaseEstimator, TransformerMixin):
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Predict using the model
        predicted_class_encoded = self.model.predict(X)

        # Decode the predicted class
        decoded_predicted_class = self.label_encoder.inverse_transform(predicted_class_encoded)

        return decoded_predicted_class.reshape(-1, 1)