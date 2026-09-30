from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 1200
issue = pd.date_range('2016-01-01', '2018-12-01', periods=n)
loan_status = rng.choice(['Fully Paid', 'Charged Off'], size=n, p=[0.82, 0.18])
annual_inc = rng.lognormal(10.6, 0.45, n)
loan_amnt = rng.integers(1000, 35000, n)
df = pd.DataFrame({
    'id': np.arange(n),
    'issue_d': issue,
    'loan_status': loan_status,
    'loan_amnt': loan_amnt,
    'funded_amnt': loan_amnt,
    'term': rng.choice(['36 months', '60 months'], n, p=[0.7, 0.3]),
    'int_rate': rng.uniform(6, 28, n).round(2),
    'installment': rng.uniform(50, 1500, n).round(2),
    'annual_inc': annual_inc.round(0),
    'emp_length': rng.choice(['1 year', '3 years', '5 years', '10+ years'], n),
    'home_ownership': rng.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n, p=[.45,.38,.15,.02]),
    'purpose': rng.choice(['debt_consolidation','credit_card','home_improvement','medical','small_business'], n),
    'dti': rng.uniform(2, 45, n).round(2),
    'delinq_2yrs': rng.poisson(0.25, n),
    'inq_last_6mths': rng.poisson(1.0, n),
    'open_acc': rng.integers(2, 20, n),
    'revol_util': rng.uniform(0, 100, n).round(2),
    'total_acc': rng.integers(5, 60, n),
})
out = RAW / 'smoke_lending_club.csv'
df.to_csv(out, index=False)
macro = pd.DataFrame({'year': [2015,2016,2017,2018,2019], 'vn_gdp_growth':[6.7,6.2,6.8,7.1,7.0], 'vn_inflation_rate':[0.6,2.7,3.5,3.5,2.8], 'vn_bank_npl_ratio':[2.0,2.1,2.0,1.9,1.8]})
macro.to_csv(ROOT/'data'/'external'/'vn_macro_worldbank.csv', index=False)
print(out)
