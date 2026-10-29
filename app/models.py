"""app/models.py - the two database tables of Section 4.4.3 (Tables 4.3 and 4.4)."""
import json
from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from app import db

PROFESSIONAL = 'Healthcare Professional'
GENERAL = 'General User'


class User(UserMixin, db.Model):
    """A registered user (table USERS)."""
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    predictions = db.relationship('PredictionRecord', backref='user', lazy=True)
    __table_args__ = (db.CheckConstraint("role IN ('Healthcare Professional', 'General User')", name='ck_user_role'),)

    def get_id(self):
        """Return the user id as text (required by Flask-Login)."""
        return str(self.user_id)

    def set_password(self, password):
        """Store only a salted PBKDF2-SHA256 hash, never the password itself."""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        """Input: password (str). Output: True if it matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def is_professional(self):
        """Output: True if the user is a Healthcare Professional."""
        return self.role == PROFESSIONAL


class PredictionRecord(db.Model):
    """One saved prediction (table PREDICTION_RECORDS)."""
    __tablename__ = 'prediction_records'
    prediction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False, index=True)
    disease = db.Column(db.String(20), nullable=False)
    inputs = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(10), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(40), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    __table_args__ = (
        db.CheckConstraint("disease IN ('Diabetes', 'Heart Disease', 'Liver Disease')", name='ck_pred_disease'),
        db.CheckConstraint("label IN ('Positive', 'Negative')", name='ck_pred_label'),
        db.CheckConstraint('probability BETWEEN 0 AND 1', name='ck_pred_probability'))

    def inputs_dict(self):
        """Output: the stored form inputs as a dictionary."""
        return json.loads(self.inputs)
