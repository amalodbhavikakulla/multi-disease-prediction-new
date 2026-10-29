"""app/auth/routes.py - Module M1: registration, login and logout."""
import re
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user, login_required
from app import db
from app.models import GENERAL, PROFESSIONAL, User

bp = Blueprint('auth', __name__)
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.]{3,50}$')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    """Create an account. Everyone registers as a General User; a Healthcare Professional account
    needs the professional access code held by the administrator (so the role cannot be self-assigned)."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        code = request.form.get('professional_code', '').strip()
        errors = []
        if not USERNAME_RE.match(username):
            errors.append('Username must be 3-50 letters, digits, dots or underscores.')
        if len(password) < 8:
            errors.append('Password must have at least 8 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('That username is already taken.')
        role = GENERAL
        if code:
            if current_app.config['PROFESSIONAL_CODE'] and code == current_app.config['PROFESSIONAL_CODE']:
                role = PROFESSIONAL
            else:
                errors.append('Invalid professional access code.')
        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html', username=username), 400
        user = User(username=username, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Account created. Please log in.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('register.html', username='')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Check the credentials; the same message is shown for an unknown user and a wrong password."""
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username', '').strip()).first()
        if user is None or not user.check_password(request.form.get('password', '')):
            flash('Invalid username or password.', 'danger')
            return render_template('login.html'), 401
        login_user(user)
        target = request.args.get('next', '')
        if target.startswith('/') and not target.startswith('//'):    # only local addresses (no open redirect)
            return redirect(target)
        return redirect(url_for('main.home'))
    return render_template('login.html')


@bp.route('/logout')
@login_required
def logout():
    """End the session and return to the home page."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))
