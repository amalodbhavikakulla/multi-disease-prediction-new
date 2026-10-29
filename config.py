"""config.py - application settings, kept apart from code (maintainability requirement)."""
import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # In production SECRET_KEY must be supplied through the environment (never committed to source control)
    """Settings used by the running application."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-change-me')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'app.db'))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PREDICTION_THRESHOLD = 0.50                      # label is Positive when probability >= threshold
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)   # idle time-out (FR-02)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', '0') == '1'   # set to 1 behind HTTPS
    # access code that allows registration as Healthcare Professional
    PROFESSIONAL_CODE = os.environ.get('PROFESSIONAL_CODE', '')
    MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')
    RESULT_FILE = os.path.join(BASE_DIR, 'ml', 'results', 'results.json')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')
    WTF_CSRF_ENABLED = True


class TestConfig(Config):
    """Settings used by the automated tests (in-memory database, no CSRF)."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite://'            # in-memory database for tests
    WTF_CSRF_ENABLED = False
    PROFESSIONAL_CODE = 'TEST-CODE'
