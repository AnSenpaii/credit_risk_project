from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
EXT = ROOT / 'data' / 'external'
REPORTS = ROOT / 'reports'

WB_INDICATORS = {
    'NY.GDP.MKTP.KD.ZG': 'vn_gdp_growth',
    'FP.CPI.TOTL.ZG': 'vn_inflation_rate',
    'FB.AST.NPER.ZS': 'vn_bank_npl_ratio',
}

# Only columns that are required at application time or for leakage checks.
LOAN_USECOLS = [
    'id', 'issue_d', 'loan_status', 'loan_amnt', 'annual_inc', 'term', 'int_rate', 'dti',
    'installment', 'emp_length', 'home_ownership', 'purpose', 'delinq_2yrs',
    'inq_last_6mths', 'open_acc', 'revol_util', 'total_acc',
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def fetch_worldbank_vietnam() -> pd.DataFrame:
    rows = []
    status = {
        'source': 'World Bank Indicators API',
        'country': 'VNM',
        'collected_at_utc': utc_now(),
        'indicators': {},
        'errors': {},
    }
    base = 'https://api.worldbank.org/v2/country/VNM/indicator'
    for code, name in WB_INDICATORS.items():
        url = f'{base}/{code}?format=json&per_page=100'
        try:
            r = requests.get(url, timeout=30, headers={'User-Agent': 'CreditRiskCapstone/2.0'})
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2:
                raise ValueError('Unexpected World Bank response')
            vals = []
            for item in payload[1]:
                year = pd.to_numeric(item.get('date'), errors='coerce')
                value = pd.to_numeric(item.get('value'), errors='coerce')
                if pd.notna(year) and pd.notna(value):
                    vals.append({'year': int(year), name: float(value)})
            piece = pd.DataFrame(vals)
            if not piece.empty:
                rows.append(piece)
                status['indicators'][name] = {'code': code, 'rows': len(piece), 'url': url}
        except Exception as exc:
            status['errors'][name] = {'code': code, 'url': url, 'error': str(exc)}

    if not rows:
        save_json({**status, 'status': 'failed'}, REPORTS / 'worldbank_status.json')
        return pd.DataFrame(columns=['year', *WB_INDICATORS.values()])

    out = rows[0]
    for piece in rows[1:]:
        out = out.merge(piece, on='year', how='outer')
    out = out.sort_values('year').reset_index(drop=True)
    EXT.mkdir(parents=True, exist_ok=True)
    out.to_csv(EXT / 'vn_macro_worldbank.csv', index=False)
    save_json({**status, 'status': 'success', 'rows': len(out)}, REPORTS / 'worldbank_status.json')
    return out


def _csv_members(z: zipfile.ZipFile, accepted_only: bool = True) -> list[str]:
    names = []
    for n in z.namelist():
        low = n.lower().replace('\\', '/')
        if not low.endswith('.csv'):
            continue
        base = Path(low).name
        if accepted_only and not base.startswith('accepted_'):
            continue
        names.append(n)
    return names


def extract_flat_lending_csv(bundle: Path, accepted_only: bool = True) -> list[Path]:
    """Extract CSV members from the Kaggle ZIP to flat data/raw filenames.

    This deliberately avoids ZipFile.extractall(), because the archive can contain
    a directory and a file with the same .csv basename in different paths.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(bundle) as z:
        for member in _csv_members(z, accepted_only=accepted_only):
            target = RAW / Path(member).name
            # Stream-copy so the ZIP is not held in RAM.
            with z.open(member, 'r') as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted.append(target)
    return extracted


def find_loan_csv() -> Path:
    """Find the accepted Lending Club CSV, preferring a flat file.

    Supports both the corrected flat extraction and legacy nested files.
    """
    smoke = RAW / 'smoke_lending_club.csv'
    if smoke.exists():
        return smoke

    preferred = [RAW / 'accepted_2007_to_2018q4.csv', RAW / 'accepted_2007_to_2018Q4.csv']
    for p in preferred:
        if p.is_file():
            return p

    # Search one level deeper for legacy extraction layouts.
    nested = [p for p in RAW.rglob('accepted_2007_to_2018q4.csv') if p.is_file()]
    nested += [p for p in RAW.rglob('accepted_2007_to_2018Q4.csv') if p.is_file()]
    if nested:
        return max(nested, key=lambda p: p.stat().st_size)

    # Last resort: inspect headers of CSV files, but only accepted files.
    candidates = []
    for p in RAW.glob('*.csv'):
        try:
            cols = set(pd.read_csv(p, nrows=5, usecols=lambda c: c in set(LOAN_USECOLS)).columns)
        except Exception:
            continue
        if {'loan_status', 'loan_amnt', 'annual_inc', 'issue_d'}.issubset(cols):
            candidates.append(p)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_size)

    raise FileNotFoundError('Không tìm thấy accepted Lending Club CSV. Hãy chạy scripts/download_data.py')


def write_collection_manifest(loan_path: Path, macro: pd.DataFrame) -> None:
    save_json({
        'loan_file': str(loan_path.relative_to(ROOT)),
        'loan_source': 'Kaggle — wordsforthewise/lending-club',
        'macro_source': 'World Bank Indicators API — Vietnam (VNM)',
        'macro_rows': int(len(macro)),
        'collected_at_utc': utc_now(),
        'accepted_only': True,
        'rejected_dataset_used_for_model': False,
        'note': 'Loan CSV is processed in chunks to avoid loading the full 1+ GB raw file into RAM.',
    }, REPORTS / 'data_collection_manifest.json')
