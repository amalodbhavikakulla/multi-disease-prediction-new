import pytest
from app import create_app, db
from app.models import GENERAL, PROFESSIONAL, User
from config import TestConfig

VALID = {
    'diabetes': {
        'Pregnancies': '2',
        'Glucose': '120',
        'BloodPressure': '70',
        'SkinThickness': '25',
        'Insulin': '100',
        'BMI': '28.5',
        'DiabetesPedigreeFunction': '0.5',
        'Age': '35'},
    'heart': {
        'age': '55',
        'sex': '1',
        'cp': '4',
        'trestbps': '140',
        'chol': '240',
        'fbs': '0',
        'restecg': '0',
        'thalach': '130',
        'exang': '1',
        'oldpeak': '1.5',
        'slope': '2',
        'ca': '1',
        'thal': '7'},
    'liver': {
        'age': '45',
        'gender': 'Male',
        'total_bilirubin': '1.2',
        'direct_bilirubin': '0.4',
        'alk_phosphatase': '210',
        'sgpt_alt': '40',
        'sgot_ast': '45',
        'total_proteins': '6.5',
        'albumin': '3.2',
        'ag_ratio': '0.9'},
}


@pytest.fixture()
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.drop_all()
        db.create_all()
        for name, role in (('doctor', PROFESSIONAL), ('patient', GENERAL)):
            u = User(username=name, role=role)
            u.set_password('password123')
            db.session.add(u)
        db.session.commit()
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client, name):
    return client.post('/login', data={'username': name, 'password': 'password123'}, follow_redirects=False)


@pytest.fixture()
def doctor(client):
    login(client, 'doctor')
    return client


@pytest.fixture()
def patient(client):
    login(client, 'patient')
    return client
