"""Robustness and security tests RB-01 ... RB-14: unusual, hostile or boundary input."""
import pytest
from app import create_app, db
from app.diseases.config import DISEASES
from config import TestConfig
from tests.conftest import VALID


@pytest.mark.parametrize('bad', ['nan', 'NaN', 'inf', '-inf', '1e999', '9' * 400])
def test_RB01_non_finite_numbers_rejected(patient, bad):
    assert patient.post('/diabetes', data=dict(VALID['diabetes'], Glucose=bad)).status_code == 400


def test_RB02_html_in_field_is_escaped(patient):
    r = patient.post('/diabetes', data=dict(VALID['diabetes'], Glucose='<script>alert(1)</script>'))
    assert r.status_code == 400 and b'<script>alert(1)</script>' not in r.data


def test_RB03_sql_injection_in_login(client):
    assert client.post('/login', data={'username': "' OR '1'='1", 'password': "' OR '1'='1"}).status_code == 401


@pytest.mark.parametrize('key', ['diabetes', 'heart', 'liver'])
def test_RB04_hard_limit_boundaries_accepted(patient, key):
    for side in (0, 1):
        data = dict(VALID[key])
        for f in DISEASES[key]['features']:
            if f['kind'] == 'number':
                data[f['name']] = str(f['hard'][side])
        r = patient.post('/' + key, data=data)
        assert r.status_code == 200, (key, side, r.data[:200])


def test_RB05_just_outside_hard_limit_rejected(patient):
    assert patient.post('/diabetes', data=dict(VALID['diabetes'], Glucose='400.01')).status_code == 400
    assert patient.post('/diabetes', data=dict(VALID['diabetes'], Glucose='29.99')).status_code == 400


def test_RB06_invalid_code_in_select_rejected(patient):
    assert patient.post('/heart', data=dict(VALID['heart'], cp='9')).status_code == 400
    assert patient.post('/liver', data=dict(VALID['liver'], gender='Other')).status_code == 400


def test_RB07_fraction_in_whole_number_field_rejected(patient):
    assert patient.post('/diabetes', data=dict(VALID['diabetes'], Age='35.5')).status_code == 400


def test_RB08_unknown_extra_fields_ignored(patient):
    assert patient.post('/diabetes', data=dict(VALID['diabetes'], admin='1', role='Healthcare Professional')).status_code == 200


def test_RB09_missing_model_gives_503_not_crash(tmp_path):
    class NoModels(TestConfig):
        MODEL_DIR = str(tmp_path)
    app = create_app(NoModels)
    with app.app_context():
        db.create_all()
        from app.models import User
        u = User(username='tester', role='General User')
        u.set_password('password123')
        db.session.add(u)
        db.session.commit()
    c = app.test_client()
    c.post('/login', data={'username': 'tester', 'password': 'password123'})
    r = c.post('/diabetes', data=VALID['diabetes'])
    assert r.status_code == 503 and b'not available' in r.data


def test_RB10_short_password_rejected(client):
    assert client.post('/register', data={'username': 'abcdef', 'password': 'short', 'confirm': 'short'}).status_code == 400


def test_RB11_role_cannot_be_self_assigned(client, app):
    client.post(
        '/register',
        data={
            'username': 'sneaky',
            'password': 'longpassword',
            'confirm': 'longpassword',
            'role': 'Healthcare Professional'})
    r = client.post(
        '/register',
        data={
            'username': 'sneaky2',
            'password': 'longpassword',
            'confirm': 'longpassword',
            'professional_code': 'guess'})
    assert r.status_code == 400
    with app.app_context():
        from app.models import User
        assert User.query.filter_by(username='sneaky').first().role == 'General User'


def test_RB12_professional_code_gives_professional_role(client, app):
    client.post(
        '/register',
        data={
            'username': 'drsmith',
            'password': 'longpassword',
            'confirm': 'longpassword',
            'professional_code': 'TEST-CODE'})
    with app.app_context():
        from app.models import User
        assert User.query.filter_by(username='drsmith').first().role == 'Healthcare Professional'


def test_RB13_security_headers_and_cookie_flags(client):
    r = client.post('/login', data={'username': 'patient', 'password': 'password123'})
    assert r.headers['X-Content-Type-Options'] == 'nosniff' and r.headers['X-Frame-Options'] == 'DENY'
    assert 'HttpOnly' in r.headers.get('Set-Cookie', '')


@pytest.mark.parametrize('url',
                         ['/diabetes',
                          '/heart',
                          '/liver',
                          '/reports',
                          '/reports/prediction.csv',
                          '/reports/comparison.csv',
                          '/analytics/dashboard',
                          '/eda',
                          '/eda/diabetes'])
def test_RB14_anonymous_users_are_redirected_to_login(client, url):
    r = client.get(url)
    assert r.status_code == 302 and '/login' in r.headers['Location']


def test_RB15_unknown_disease_in_analytics_gives_404(doctor):
    assert doctor.get('/analytics/comparison?disease=cancer').status_code == 404
    assert doctor.get('/eda/cancer').status_code == 404


def test_RB16_database_failure_does_not_lose_result(patient, monkeypatch):
    """If saving the record fails, the user still sees the result and a notice; the session is rolled back."""
    from sqlalchemy.exc import SQLAlchemyError
    from app import db

    def broken_commit():
        raise SQLAlchemyError('simulated failure')
    monkeypatch.setattr(db.session, 'commit', broken_commit)
    r = patient.post('/diabetes', data=VALID['diabetes'])
    assert r.status_code == 200 and b'could not be saved' in r.data and b'Probability' in r.data


def test_RB17_login_redirects_to_requested_page_but_never_off_site(client):
    r = client.post('/login?next=/diabetes', data={'username': 'patient', 'password': 'password123'})
    assert r.status_code == 302 and r.headers['Location'].endswith('/diabetes')
    client.get('/logout')
    for evil in ('//evil.example.com', 'https://evil.example.com', 'javascript:alert(1)'):
        r = client.post('/login?next=' + evil, data={'username': 'patient', 'password': 'password123'})
        assert r.status_code == 302 and 'evil' not in r.headers['Location'] and 'javascript' not in r.headers['Location']
        client.get('/logout')


def test_RB18_unexpected_error_shows_friendly_500_page(app, patient, monkeypatch):
    app.config['PROPAGATE_EXCEPTIONS'] = False

    def boom(*a, **k):
        raise RuntimeError('internal detail that must not be shown')
    monkeypatch.setattr(app.model_service, 'predict', boom)
    r = patient.post('/diabetes', data=VALID['diabetes'])
    assert r.status_code == 500 and b'Something went wrong' in r.data and b'internal detail' not in r.data


def test_RB19_production_refuses_default_secret_key(monkeypatch):
    monkeypatch.setenv('APP_ENV', 'production')
    with pytest.raises(RuntimeError):
        create_app(TestConfig)

    class WithKey(TestConfig):
        SECRET_KEY = 'a-long-random-value-for-this-test'
    assert create_app(WithKey) is not None


def test_RB20_hsts_header_only_over_https_setting():
    class Https(TestConfig):
        SESSION_COOKIE_SECURE = True
    secure = create_app(Https).test_client().get('/')
    plain = create_app(TestConfig).test_client().get('/')
    assert 'max-age' in secure.headers.get('Strict-Transport-Security', '') and 'Strict-Transport-Security' not in plain.headers
