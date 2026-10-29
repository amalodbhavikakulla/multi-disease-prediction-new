"""Integration tests IT-01 ... IT-06: interfaces between modules."""
import json
import os
import tempfile
import joblib
import pandas as pd
import pytest
from app import create_app, db
from app.models import PredictionRecord
from config import TestConfig
from ml.preprocessing import load_dataset
from ml.train_models import DATA_DIR, MODEL_DIR, RESULT_FILE
from tests.conftest import VALID


def test_IT01_data_to_preprocessing_to_model_pipeline():
    """M2 -> M3 -> M4: saved pipeline accepts cleaned data from M2 and predicts every record."""
    for key in ('diabetes', 'heart', 'liver'):
        X, y = load_dataset(key, DATA_DIR)
        pipe = joblib.load(os.path.join(MODEL_DIR, key + '_best_pipeline.pkl'))
        p = pipe.predict_proba(X)[:, 1]
        assert len(p) == len(X) and ((p >= 0) & (p <= 1)).all()


def test_IT02_form_to_prediction_to_database(app, patient):
    """M9 -> M5 -> database: a valid form creates exactly one correct record."""
    patient.post('/liver', data=VALID['liver'])
    with app.app_context():
        recs = PredictionRecord.query.all()
        assert len(recs) == 1
        r = recs[0]
        assert r.disease == 'Liver Disease' and r.label in ('Positive', 'Negative') and 0 <= r.probability <= 1
        assert json.loads(r.inputs)['total_bilirubin'] == 1.2 and r.model_name == 'LogisticRegression'


def test_IT03_invalid_form_creates_no_record(app, patient):
    patient.post('/heart', data=dict(VALID['heart'], trestbps='5'))
    with app.app_context():
        assert PredictionRecord.query.count() == 0


def test_IT04_comparison_page_matches_stored_results(doctor):
    """M4 results file -> M7 page: the best model and its ROC-AUC on the page equal results.json."""
    stored = json.load(open(RESULT_FILE))
    for key in stored:
        html = doctor.get('/analytics/comparison?disease=' + key).data.decode()
        best = stored[key]['best']
        assert best in html and '%.3f' % stored[key]['models'][best]['roc_auc'] in html


def test_IT05_comparison_report_matches_results_file(doctor):
    stored = json.load(open(RESULT_FILE))
    lines = doctor.get('/reports/comparison.csv').data.decode().strip().splitlines()
    assert len(lines) == 1 + sum(len(v['models']) for v in stored.values())
    assert sum(1 for line in lines if line.endswith(',yes')) == 3


def test_IT06_dashboard_counts_follow_database(app, doctor):
    """database -> M7: prediction counts on the dashboard follow the stored records."""
    doctor.post('/diabetes', data=VALID['diabetes'])
    doctor.post('/diabetes', data=VALID['diabetes'])
    doctor.post('/heart', data=VALID['heart'])
    html = doctor.get('/analytics/dashboard').data.decode()
    assert '"Diabetes": 2' in html and '"Heart Disease": 1' in html
