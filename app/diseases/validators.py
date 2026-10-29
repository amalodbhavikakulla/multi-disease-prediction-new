"""app/diseases/validators.py - Module M9 (server-side input validation, Section 4.4.4).

validate(cfg, form) -> (values, errors, cautions)
    cfg      : one entry of DISEASES
    form     : dict-like of submitted strings (request.form)
    values   : {feature: float or str}   cleaned values (only meaningful when errors is empty)
    errors   : {feature: message}        reasons for rejection
    cautions : [message]                 value is allowed but lies outside the training range"""
import math


def validate(cfg, form):
    """Validate a submitted form. Input: cfg, form. Output: (values, errors, cautions)."""
    values, errors, cautions = {}, {}, []
    for f in cfg['features']:
        name, label = f['name'], f['label']
        raw = (form.get(name) or '').strip()
        if raw == '':
            errors[name] = '%s is required.' % label
            continue
        if f['kind'] == 'select':
            allowed = {str(v): v for v, _ in f['options']}
            if raw not in allowed:
                errors[name] = '%s: choose one of the listed options.' % label
            else:
                v = allowed[raw]
                values[name] = v if isinstance(v, str) else float(v)
            continue
        try:
            v = float(raw)
        except ValueError:
            errors[name] = '%s must be a number.' % label
            continue
        if not math.isfinite(v):
            errors[name] = '%s must be a finite number.' % label
            continue
        lo, hi = f['hard']
        if f['integer'] and v != int(v):
            errors[name] = '%s must be a whole number.' % label
        elif not lo <= v <= hi:
            errors[name] = '%s must be between %s and %s %s.' % (label, lo, hi, f['unit'])
        else:
            values[name] = v
            tlo, thi = f['train']
            if not tlo <= v <= thi:
                cautions.append('%s (%s) is outside the range seen in the training data (%s to %s); '
                                'the prediction may be less reliable.' % (label, raw, tlo, thi))
    for small, big, msg in cfg['cross_rules']:           # cross-field rules
        if small in values and big in values and values[small] > values[big]:
            errors[small] = msg
    return values, errors, cautions
