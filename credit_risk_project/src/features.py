from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_POSITIVE = {'Charged Off'}
TARGET_NEGATIVE = {'Fully Paid'}

NUMERIC = [
    'loan_amnt','annual_inc','int_rate','dti','installment','emp_length_years','term_months',
    'delinq_2yrs','inq_last_6mths','open_acc','revol_util','total_acc'
]
CATEGORICAL = ['home_ownership','purpose']


def parse_emp_length(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    if '< 1' in s or '<1' in s:
        return 0.5
    if '10+' in s:
        return 10.0
    try:
        return float(s.split()[0])
    except Exception:
        return np.nan


def prepare_loans(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'issue_d' not in d:
        raise KeyError('Thiếu cột issue_d')
    d['issue_date'] = pd.to_datetime(d['issue_d'], format='%b-%Y', errors='coerce')
    # Fallback for any already-standard date-like values.
    missing_date = d['issue_date'].isna()
    if missing_date.any():
        d['issue_date'] = d['issue_date'].astype('datetime64[us]'); d.loc[missing_date, 'issue_date'] = pd.to_datetime(d.loc[missing_date, 'issue_d'], errors='coerce').astype('datetime64[us]')
    # Force stable string dtype across CSV chunks. The Lending Club `id` column
    # may be inferred as integer in some chunks and string in others; normalizing
    # it here prevents Arrow/Parquet schema drift between chunks.
    if 'id' in d:
        d['id'] = d['id'].astype('string')
    d['loan_status'] = d['loan_status'].astype('string').str.strip()
    d = d[d['loan_status'].isin(TARGET_POSITIVE | TARGET_NEGATIVE)].copy()
    d['target_default'] = d['loan_status'].isin(TARGET_POSITIVE).astype('int8')

    d['term_months'] = pd.to_numeric(d['term'].astype('string').str.extract(r'(\d+)')[0], errors='coerce')
    d['emp_length_years'] = d['emp_length'].map(parse_emp_length) if 'emp_length' in d else np.nan
    for c in ['loan_amnt','annual_inc','int_rate','dti','installment','delinq_2yrs','inq_last_6mths','open_acc','revol_util','total_acc']:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors='coerce')

    d['loan_to_income'] = d['loan_amnt'] / d['annual_inc'].replace(0, np.nan)
    d['payment_to_income'] = (d['installment'] * 12) / d['annual_inc'].replace(0, np.nan)
    d['income_log'] = np.log1p(d['annual_inc'].clip(lower=0))
    d['amount_log'] = np.log1p(d['loan_amnt'].clip(lower=0))
    d['issue_year'] = d['issue_date'].dt.year

    # Use only fields that exist at origination/application time.
    keep = [
        'id','issue_date','issue_year','loan_status','target_default','loan_amnt','annual_inc','int_rate','dti',
        'installment','emp_length_years','delinq_2yrs','inq_last_6mths','open_acc','revol_util','total_acc',
        'term_months','home_ownership','purpose','loan_to_income','payment_to_income','income_log','amount_log'
    ]
    out = d[[c for c in keep if c in d.columns]].copy()

    # Stable on-disk schema for chunked Parquet writes. In particular, categorical
    # fields are explicitly strings and integer-like fields have fixed widths.
    string_cols = ['id', 'loan_status', 'home_ownership', 'purpose']
    for c in string_cols:
        if c in out:
            out[c] = out[c].astype('string')
    if 'issue_date' in out:
        out['issue_date'] = pd.to_datetime(out['issue_date'], errors='coerce').astype('datetime64[ns]')
    if 'issue_year' in out:
        out['issue_year'] = pd.to_numeric(out['issue_year'], errors='coerce').astype('Int32')
    if 'target_default' in out:
        out['target_default'] = pd.to_numeric(out['target_default'], errors='coerce').fillna(0).astype('int8')
    if 'term_months' in out:
        out['term_months'] = pd.to_numeric(out['term_months'], errors='coerce').astype('Int16')
    for c in [
        'loan_amnt','annual_inc','int_rate','dti','installment','emp_length_years',
        'delinq_2yrs','inq_last_6mths','open_acc','revol_util','total_acc',
        'loan_to_income','payment_to_income','income_log','amount_log'
    ]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')
    return out


def add_vietnam_context(loans: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    d = loans.copy()
    if macro is None or macro.empty:
        for c in ['vn_gdp_growth','vn_inflation_rate','vn_bank_npl_ratio']:
            d[f'{c}_lag1'] = np.nan
        return d
    m = macro.copy()
    m['year'] = pd.to_numeric(m['year'], errors='coerce').astype('Int64')
    # Use the previous calendar year, preventing look-ahead leakage.
    m['issue_year'] = m['year'] + 1
    rename = {
        'vn_gdp_growth':'vn_gdp_growth_lag1',
        'vn_inflation_rate':'vn_inflation_rate_lag1',
        'vn_bank_npl_ratio':'vn_bank_npl_ratio_lag1',
    }
    m = m.rename(columns=rename)
    cols = ['issue_year'] + list(rename.values())
    return d.merge(m[cols], on='issue_year', how='left')


def feature_columns(include_macro: bool) -> tuple[list[str], list[str]]:
    num = NUMERIC + ['loan_to_income','payment_to_income','income_log','amount_log']
    cat = CATEGORICAL
    if include_macro:
        num += ['vn_gdp_growth_lag1','vn_inflation_rate_lag1','vn_bank_npl_ratio_lag1']
    return num, cat
