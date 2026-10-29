"""app/diseases/config.py - DiseaseConfig data: everything that differs between the three diseases
(feature names and order, labels, units, hard limits, training ranges, allowed codes). See Tables 4.5-4.7."""


def num(name, label, unit, hard, train, step='any', integer=False):
    """Describe a numeric input field. Input: name, label, unit, hard limits, training range. Output: dict."""
    return {'name': name, 'label': label, 'unit': unit, 'kind': 'number', 'hard': hard, 'train': train,
            'step': step, 'integer': integer}


def sel(name, label, options, train=None):
    """Describe a drop-down input field. Input: name, label, options. Output: dict."""
    return {'name': name, 'label': label, 'unit': '', 'kind': 'select', 'options': options, 'train': train}


DISEASES = {
    'diabetes': {
        'title': 'Diabetes', 'icon': '🩸', 'blurb': 'Preliminary diabetes risk from 8 clinical measurements.',
        'features': [
            num('Pregnancies', 'Pregnancies', 'count', (0, 20), (0, 17), 1, True),
            num('Glucose', 'Glucose', 'mg/dL', (30, 400), (44, 199)),
            num('BloodPressure', 'Blood Pressure (diastolic)', 'mm Hg', (30, 200), (24, 122)),
            num('SkinThickness', 'Skin Thickness', 'mm', (5, 100), (7, 99)),
            num('Insulin', 'Insulin', 'µU/mL', (5, 1000), (14, 846)),
            num('BMI', 'BMI', 'kg/m²', (10, 80), (18.2, 67.1)),
            num('DiabetesPedigreeFunction', 'Diabetes Pedigree Function', 'score', (0.05, 3.0), (0.08, 2.42)),
            num('Age', 'Age', 'years', (18, 100), (21, 81), 1, True)],
        'cross_rules': []},
    'heart': {
        'title': 'Heart Disease', 'icon': '❤️', 'blurb': 'Preliminary heart disease risk from 13 clinical findings.',
        'features': [
            num('age', 'Age', 'years', (18, 100), (29, 77), 1, True),
            sel('sex', 'Sex', [(0, 'Female'), (1, 'Male')]),
            sel('cp', 'Chest Pain Type',
                [(1, 'Typical angina'), (2, 'Atypical angina'), (3, 'Non-anginal pain'), (4, 'Asymptomatic')]),
            num('trestbps', 'Resting Blood Pressure', 'mm Hg', (60, 260), (94, 200)),
            num('chol', 'Cholesterol', 'mg/dL', (80, 700), (126, 564)),
            sel('fbs', 'Fasting Blood Sugar > 120 mg/dL', [(0, 'No'), (1, 'Yes')]),
            sel('restecg', 'Resting ECG', [(0, 'Normal'), (1, 'ST-T wave abnormality'), (2, 'Left ventricular hypertrophy')]),
            num('thalach', 'Maximum Heart Rate', 'bpm', (40, 250), (71, 202)),
            sel('exang', 'Exercise Induced Angina', [(0, 'No'), (1, 'Yes')]),
            num('oldpeak', 'Oldpeak (ST depression)', '', (0, 10), (0, 6.2)),
            sel('slope', 'Slope of ST segment', [(1, 'Upsloping'), (2, 'Flat'), (3, 'Downsloping')]),
            sel('ca', 'Major vessels coloured (CA)', [(0, '0'), (1, '1'), (2, '2'), (3, '3')]),
            sel('thal', 'Thalassemia (Thal)', [(3, 'Normal'), (6, 'Fixed defect'), (7, 'Reversible defect')])],
        'cross_rules': []},
    'liver': {
        'title': 'Liver Disease', 'icon': '🧪', 'blurb': 'Preliminary liver disease risk from 10 laboratory values.',
        'features': [
            num('age', 'Age', 'years', (1, 100), (4, 90), 1, True),
            sel('gender', 'Gender', [('Male', 'Male'), ('Female', 'Female')]),
            num('total_bilirubin', 'Total Bilirubin', 'mg/dL', (0.1, 100), (0.4, 75.0)),
            num('direct_bilirubin', 'Direct Bilirubin', 'mg/dL', (0.05, 50), (0.1, 19.7)),
            num('alk_phosphatase', 'Alkaline Phosphatase', 'IU/L', (20, 3000), (63, 2110)),
            num('sgpt_alt', 'SGPT (ALT)', 'IU/L', (5, 3000), (10, 2000)),
            num('sgot_ast', 'SGOT (AST)', 'IU/L', (5, 6000), (10, 4929)),
            num('total_proteins', 'Total Proteins', 'g/dL', (2, 12), (2.7, 9.6)),
            num('albumin', 'Albumin', 'g/dL', (0.5, 7), (0.9, 5.5)),
            num('ag_ratio', 'Albumin / Globulin Ratio', 'ratio', (0.1, 4), (0.3, 2.8))],
        'cross_rules': [('direct_bilirubin', 'total_bilirubin', 'Direct bilirubin cannot exceed total bilirubin.'),
                        ('albumin', 'total_proteins', 'Albumin cannot exceed total proteins.')]},
}
