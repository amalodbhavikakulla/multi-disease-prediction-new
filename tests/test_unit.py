"""Unit tests UT-01 ... UT-24 (Table 4.8)."""
import json
import numpy as np
import pytest
from app import create_app, db
from app.diseases.config import DISEASES
from app.diseases.validators import validate
from app.models import PredictionRecord, User
from ml.preprocessing import build_preprocessor, load_dataset
from ml.train_models import DATA_DIR, classifiers
from tests.conftest import VALID, login
from config import TestConfig


def test_UT01_register_valid_user_password_hashed(client, app):
    r = client.post('/register', data={'username': 'newuser', 'password': 'longpassword', 'confirm': 'longpassword'})
    assert r.status_code == 302
    with app.app_context():
        u = User.query.filter_by(username='newuser').first()
        assert u is not None and u.password_hash != 'longpassword' and u.password_hash.startswith('pbkdf2:sha256')


def test_UT02_duplicate_username_rejected(client):
    r = client.post('/register', data={'username': 'doctor', 'password': 'longpassword', 'confirm': 'longpassword'})
    assert r.status_code == 400 and b'already taken' in r.data


def test_UT03_wrong_password_no_session(client):
    r = client.post('/login', data={'username': 'doctor', 'password': 'wrong'})
    assert r.status_code == 401 and b'Invalid username or password' in r.data
    assert client.get('/diabetes').status_code == 302      # still redirected to login


def test_UT04_general_user_gets_403_on_dashboard(patient):
    assert patient.get('/analytics/dashboard').status_code == 403


def test_UT05_dataset_sizes():
    sizes = {k: len(load_dataset(k, DATA_DIR)[0]) for k in ('diabetes', 'heart', 'liver')}
    assert sizes == {'diabetes': 768, 'heart': 303, 'liver': 570}


def test_UT06_liver_columns_corrected():
    X, _ = load_dataset('liver', DATA_DIR)
    assert 'alk_phosphatase' in X.columns and X['total_proteins'].max() < 12 and X['alk_phosphatase'].max() > 1000


def test_UT07_imputation_removes_missing_values():
    X, y = load_dataset('heart', DATA_DIR)
    assert X.isnull().sum().sum() == 6
    out = build_preprocessor('heart').fit_transform(X, y)
    assert not np.isnan(out).any()


def test_UT08_scaled_columns_have_zero_mean_unit_sd():
    X, y = load_dataset('diabetes', DATA_DIR)
    out = build_preprocessor('diabetes').fit_transform(X, y)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-8) and np.allclose(out.std(axis=0), 1, atol=1e-8)


def test_UT09_unseen_category_handled():
    X, y = load_dataset('heart', DATA_DIR)
    pre = build_preprocessor('heart').fit(X, y)
    row = X.iloc[[0]].copy()
    row['thal'] = 5.0           # a code never seen in training
    assert pre.transform(row).shape[0] == 1


def test_UT10_five_models_per_disease():
    assert len(classifiers()) == 5


def test_UT11_results_reproducible(app):
    import json
    from ml.train_models import RESULT_FILE
    stored = json.load(open(RESULT_FILE))
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    X, y = load_dataset('heart', DATA_DIR)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    aucs = []
    for _ in range(2):
        p = Pipeline([('pre', build_preprocessor('heart')), ('clf', classifiers()['Logistic Regression'])]).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, p.predict_proba(Xte)[:, 1]))
    assert aucs[0] == aucs[1] == pytest.approx(stored['heart']['models']['Logistic Regression']['roc_auc'])


def test_UT12_best_model_has_highest_auc(app):
    for key in ('diabetes', 'heart', 'liver'):
        e = app.results_service.get(key)
        assert e['best'] == max(e['models'], key=lambda m: (e['models'][m]['roc_auc'], e['models'][m]['f1']))


@pytest.mark.parametrize('key', ['diabetes', 'heart', 'liver'])
def test_UT13_valid_prediction(app, key):
    values, errors, _ = validate(DISEASES[key], VALID[key])
    assert not errors
    p = app.model_service.predict(key, values)
    assert 0 <= p <= 1


def test_UT14_threshold_boundary(app):
    s = app.model_service
    assert s.interpret(0.50)[0] == 'Positive' and s.interpret(0.49)[0] == 'Negative'


def test_UT15_risk_bands(app):
    assert [app.model_service.interpret(p)[1] for p in (0.2, 0.4, 0.6, 0.8)] == ['Low', 'Borderline', 'Elevated', 'High']


def test_UT16_importance_sums_to_one_and_sorted(app):
    for key in ('diabetes', 'heart', 'liver'):
        vals = list(app.results_service.get(key)['importance'].values())
        assert abs(sum(vals) - 1) < 1e-6 and vals == sorted(vals, reverse=True)


def test_UT17_chart_data_for_each_disease(doctor):
    for key in ('diabetes', 'heart', 'liver'):
        assert doctor.get('/eda/' + key).status_code == 200
    assert doctor.get('/analytics/dashboard').status_code == 200


def test_UT18_prediction_report_only_own_records(app, patient):
    patient.post('/diabetes', data=VALID['diabetes'])
    with app.app_context():
        other = User.query.filter_by(username='doctor').first()
        db.session.add(
            PredictionRecord(
                user_id=other.user_id,
                disease='Diabetes',
                inputs='{}',
                label='Negative',
                probability=0.1,
                model_name='X'))
        db.session.commit()
    r = patient.get('/reports/prediction.csv')
    lines = r.data.decode().strip().splitlines()
    assert lines[0].startswith('prediction_id,disease,inputs,label,probability,model_name,created_at') and len(lines) == 2


def test_UT19_value_below_hard_limit_rejected(patient):
    d = dict(VALID['diabetes'], Glucose='-5')
    assert patient.post('/diabetes', data=d).status_code == 400


def test_UT20_non_numeric_rejected(patient):
    d = dict(VALID['diabetes'], BMI='abc')
    r = patient.post('/diabetes', data=d)
    assert r.status_code == 400 and b'must be a number' in r.data


def test_UT21_cross_field_rule(patient):
    d = dict(VALID['liver'], direct_bilirubin='5', total_bilirubin='2')
    r = patient.post('/liver', data=d)
    assert r.status_code == 400 and b'cannot exceed total bilirubin' in r.data


def test_UT22_missing_csrf_token_refused():
    class Csrf(TestConfig):
        WTF_CSRF_ENABLED = True
    app = create_app(Csrf)
    with app.app_context():
        db.create_all()
    assert app.test_client().post('/login', data={'username': 'a', 'password': 'b'}).status_code == 400


def test_UT23_empty_field_message(patient):
    d = dict(VALID['heart'], chol='')
    r = patient.post('/heart', data=d)
    assert r.status_code == 400 and b'is required' in r.data


def test_UT24_outside_training_range_gives_caution(patient):
    d = dict(VALID['diabetes'], Insulin='900')
    r = patient.post('/diabetes', data=d)
    assert r.status_code == 200 and b'outside the range seen in the training data' in r.data
