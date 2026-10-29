"""app/security.py - role-based access control decorator used by every restricted route."""
from functools import wraps
from flask import abort
from flask_login import current_user, login_required


def professional_required(view):
    """Allow only logged-in Healthcare Professionals; others receive the 403 page.
    Input: a Flask view function.  Output: the wrapped view."""
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        """Check the login and the role before running the view."""
        if not current_user.is_professional():
            abort(403)
        return view(*args, **kwargs)
    return wrapper
