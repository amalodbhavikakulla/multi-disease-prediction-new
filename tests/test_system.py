"""System tests ST-01 ... ST-12 (Table 4.9), run with the Flask test client."""
import time
import pytest
from tests.conftest import VALID, login


@pytest.mark.parametrize('key,sid', [('diabetes', 'ST-01'), ('heart', 'ST-02'), ('liver', 'ST-03')])
def test_ST01_03_prediction_end_to_end(client, key, sid):
    r = client.post('/register', data={'username': 'e2e_' + key, 'password': 'longpassword', 'confirm': 'longpassword'})
    assert r.status_code == 302
    client.post('/login', data={'username': 'e2e_' + key, 'password': 'longpassword'})
    assert client.get('/' + key).status_code == 200
    r = client.post('/' + key, data=VALID[key])
    assert r.status_code == 200
    for text in (b'Probability', b'risk', b'not a medical diagnosis'):
        assert text in r.data
    assert (b'Positive' in r.data) or (b'Negative' in r.data)


def test_ST04_model_comparison(doctor):
    for key in ('diabetes', 'heart', 'liver'):
        r = doctor.get('/analytics/comparison?disease=' + key)
        assert r.status_code == 200 and r.data.count(b'<tr') >= 6 and b'Selected model' in r.data


def test_ST05_feature_importance(doctor):
    for key in ('diabetes', 'heart', 'liver'):
        assert doctor.get('/analytics/importance?disease=' + key).status_code == 200


def test_ST06_dashboard(doctor):
    r = doctor.get('/analytics/dashboard')
    assert r.status_code == 200 and r.data.count(b'<canvas') == 4
    assert b'Confusion matrix' in r.data and b'True positive' in r.data and b'ROC curves' in r.data


def test_ST07_report_download(doctor):
    doctor.post('/liver', data=VALID['liver'])
    for url, head in (('/reports/prediction.csv', b'prediction_id'), ('/reports/comparison.csv', b'disease,model'),
                      ('/reports/importance.csv', b'disease,rank')):
        r = doctor.get(url)
        assert r.status_code == 200 and r.mimetype == 'text/csv' and head in r.data


def test_ST08_general_user_restrictions(patient):
    for url in ('/analytics/dashboard', '/analytics/comparison', '/analytics/importance', '/eda', '/reports/comparison.csv'):
        assert patient.get(url).status_code == 403
    home = patient.get('/').data
    assert b'Dashboard' not in home and b'Model Comparison' not in home


def test_ST09_session_timeout(app, client):
    from datetime import timedelta
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=1)
    login(client, 'patient')
    assert client.get('/diabetes').status_code == 200
    time.sleep(2.2)
    assert client.get('/diabetes').status_code == 302


def test_ST10_unknown_url_custom_404(client):
    r = client.get('/no/such/page')
    assert r.status_code == 404 and b'was not found' in r.data and b'Traceback' not in r.data


def test_ST11_responsive_markup(client):
    assert b'name="viewport"' in client.get('/').data and b'bootstrap' in client.get('/').data


def test_ST12_response_time(patient):
    patient.post('/diabetes', data=VALID['diabetes'])          # warm-up
    start = time.perf_counter()
    for _ in range(20):
        assert patient.post('/diabetes', data=VALID['diabetes']).status_code == 200
    assert (time.perf_counter() - start) / 20 < 2.0


def test_ST14_reproducibility_retraining_gives_stored_results(tmp_path, monkeypatch):
    """Retrain from the script into a temporary folder; every stored metric must be identical."""
    import json
    import ml.train_models as tm
    stored = json.load(open(tm.RESULT_FILE))
    monkeypatch.setattr(tm, 'MODEL_DIR', str(tmp_path / 'models'))
    monkeypatch.setattr(tm, 'RESULT_FILE', str(tmp_path / 'results' / 'results.json'))
    tm.main()
    new = json.load(open(tm.RESULT_FILE))
    for key in stored:
        assert new[key]['best'] == stored[key]['best']
        for name, m in stored[key]['models'].items():
            for metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_auc_mean'):
                assert new[key]['models'][name][metric] == pytest.approx(m[metric], abs=1e-12)
    assert (tmp_path / 'models' / 'diabetes_best_pipeline.pkl').exists()
