"""
ml/preprocessing.py  -  Module M2 (data management) and M3 (preprocessing pipeline).

Functions
    load_dataset(key, data_dir) -> (X, y)   : load a CSV, apply the Chapter 3 cleaning steps
    build_preprocessor(key)     -> ColumnTransformer : imputation, one-hot encoding, scaling
Keys: 'diabetes', 'heart', 'liver'
"""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILES = {'diabetes': 'Diabetes.csv', 'heart': 'Heart_Disease.csv', 'liver': 'Liver_Dataset.csv'}
TARGETS = {'diabetes': 'Outcome', 'heart': 'target', 'liver': 'target'}

# Column groups used by the preprocessing pipeline of each disease (Table 3.7)
SPEC = {
    'diabetes': {'num': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age'], 'cat': [], 'binary': []},
    'heart': {'num': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
              'cat': ['cp', 'restecg', 'slope', 'thal'], 'binary': ['sex', 'fbs', 'exang']},
    'liver': {'num': ['age', 'total_bilirubin', 'direct_bilirubin', 'alk_phosphatase', 'sgpt_alt', 'sgot_ast',
                      'total_proteins', 'albumin', 'ag_ratio'], 'cat': ['gender'], 'binary': []},
}
# The header of the supplied liver file is mislabelled (Section 3.3.4); these are the corrected names
LIVER_COLUMNS = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alk_phosphatase', 'sgpt_alt',
                 'sgot_ast', 'total_proteins', 'albumin', 'ag_ratio', 'selector']


def load_dataset(key, data_dir='data'):
    """Load and clean one dataset.
    Input : key (str) - disease key; data_dir (str) - folder holding the CSV files
    Output: X (DataFrame of features in training order), y (Series of 0/1 target)
    Raises: FileNotFoundError if the CSV is missing; ValueError if expected columns are absent."""
    df = pd.read_csv(os.path.join(data_dir, FILES[key]))
    if key == 'liver':
        if df.shape[1] != len(LIVER_COLUMNS):
            raise ValueError('Liver file must have 11 columns')
        df.columns = LIVER_COLUMNS                      # correct the mislabelled header
        df = df.drop_duplicates().reset_index(drop=True)  # 583 -> 570 records
        df['target'] = (df['selector'] == 1).astype(int)  # 1 = patient, 2 = non-patient -> 1 / 0
        df = df.drop(columns='selector')
    spec = SPEC[key]
    expected = spec['num'] + spec['cat'] + spec['binary'] + [TARGETS[key]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError('Missing columns in %s: %s' % (FILES[key], missing))
    columns = spec['num'] if key == 'diabetes' else None
    feature_order = {'diabetes': spec['num'],
                     'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                               'oldpeak', 'slope', 'ca', 'thal'],
                     'liver': ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alk_phosphatase',
                               'sgpt_alt', 'sgot_ast', 'total_proteins', 'albumin', 'ag_ratio']}[key]
    return df[feature_order], df[TARGETS[key]]


def build_preprocessor(key):
    """Build the ColumnTransformer for one disease (fitted later, on training data only).
    Numeric: impute (median; most frequent for heart) then standardise.
    Nominal: impute most frequent then one-hot encode.  Binary: passed through unchanged."""
    spec = SPEC[key]
    steps = []
    if spec['num']:
        strategy = 'most_frequent' if key == 'heart' else 'median'
        steps.append(('num', Pipeline([('imp', SimpleImputer(strategy=strategy)), ('sc', StandardScaler())]), spec['num']))
    if spec['cat']:
        steps.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                       ('oh', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))]), spec['cat']))
    if spec['binary']:
        steps.append(('bin', 'passthrough', spec['binary']))
    return ColumnTransformer(steps)
