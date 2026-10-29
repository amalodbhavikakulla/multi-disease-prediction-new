"""app/services/model_service.py - Module M5: prediction and risk interpretation."""
import os
import joblib
import pandas as pd
from app.diseases.config import DISEASES

BANDS = [(0.30, 'Low', 'No immediate indication of this condition. Keep a healthy lifestyle and routine check-ups.'),
         (0.50, 'Borderline', 'Some indicators are present, although below the decision threshold. Consider a routine check-up.'),
         (0.70, 'Elevated', 'The pattern is similar to people with this condition. Please consult a doctor.'),
         (1.01, 'High', 'The pattern is strongly similar to people with this condition. Please consult a doctor soon.')]


class ModelService:
    """Loads the saved pipelines once, at start-up, and answers prediction requests."""

    def __init__(self, model_dir, threshold=0.5):
        """Load the saved pipelines once. Input: model_dir (str), threshold (float)."""
        self.threshold = threshold
        self.pipelines = {}
        for key in DISEASES:
            path = os.path.join(model_dir, key + '_best_pipeline.pkl')
            if os.path.exists(path):
                self.pipelines[key] = joblib.load(path)

    def available(self, key):
        """Output: True if the model of this disease has been loaded."""
        return key in self.pipelines

    def model_name(self, key):
        """Output: class name of the classifier inside the saved pipeline."""
        return type(self.pipelines[key].named_steps['clf']).__name__

    def predict(self, key, values):
        """Input : key (str), values (dict feature -> value, validated)
        Output: probability of the positive class (float, 0..1)"""
        columns = [f['name'] for f in DISEASES[key]['features']]
        row = pd.DataFrame([[values[c] for c in columns]], columns=columns)   # training column order
        return float(self.pipelines[key].predict_proba(row)[0, 1])

    def interpret(self, probability):
        """Return (label, band name, message) for a probability."""
        label = 'Positive' if probability >= self.threshold else 'Negative'
        for upper, band, message in BANDS:
            if probability < upper:
                return label, band, message
