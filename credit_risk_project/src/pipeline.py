from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from data_collection import (
    find_loan_csv,
    fetch_worldbank_vietnam,
    write_collection_manifest,
    LOAN_USECOLS,
)
from features import prepare_loans, add_vietnam_context, feature_columns
from modeling import fit_and_select

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
PROC.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000
MAX_MODEL_TRAIN_ROWS = 500_000
PIPELINE_VERSION = '3.3'
EXPECTED_SOURCE_YEAR_SPAN = (2007, 2018)


def read_macro() -> pd.DataFrame:
    p = ROOT / 'data' / 'external' / 'vn_macro_worldbank.csv'
    if p.exists():
        return pd.read_csv(p)
    return fetch_worldbank_vietnam()


def temporal_split(df: pd.DataFrame):
    d = df.sort_values('issue_date').reset_index(drop=True)
    n = len(d)
    a = int(n * 0.70)
    b = int(n * 0.85)
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()


def _detect_usecols(loan_path: Path) -> list[str]:
    header = pd.read_csv(loan_path, nrows=0)
    available = set(header.columns)
    required = {'loan_status', 'loan_amnt', 'annual_inc', 'issue_d'}
    if not required.issubset(available):
        missing = sorted(required - available)
        raise ValueError(f'Lending Club CSV thiếu cột bắt buộc: {missing}')
    return [c for c in LOAN_USECOLS if c in available]


def audit_raw_issue_years(loan_path: Path, chunksize: int = CHUNK_SIZE) -> dict:
    """Audit issue-year coverage directly from the raw source without loading it into RAM.

    This catches stale/corrupt cached Parquet artifacts that accidentally contain only
    one vintage even though the source file is the 2007–2018 Lending Club export.
    """
    counts: dict[str, int] = {}
    total = 0
    for chunk in pd.read_csv(loan_path, usecols=['issue_d'], chunksize=chunksize, low_memory=False):
        total += len(chunk)
        dt = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce')
        missing = dt.isna()
        if missing.any():
            dt.loc[missing] = pd.to_datetime(chunk.loc[missing, 'issue_d'], errors='coerce')
        years = dt.dt.year.dropna().astype(int)
        for year, n in years.value_counts().items():
            key = str(int(year))
            counts[key] = counts.get(key, 0) + int(n)
    return {
        'raw_issue_years': {k: counts[k] for k in sorted(counts, key=int)},
        'raw_issue_year_count': len(counts),
        'raw_rows_audited': total,
    }


def build_cleaned_dataset(loan_path: Path, out_base: Path, chunksize: int = CHUNK_SIZE) -> tuple[pd.DataFrame, dict]:
    """Process the multi-GB accepted CSV incrementally.

    Only required columns are read. Rows are filtered before they are written.
    With PyArrow available, each chunk is appended directly to a Parquet file so
    the whole raw CSV is never held in RAM at once.
    """
    usecols = _detect_usecols(loan_path)
    raw_rows = 0
    kept_rows = 0
    status_counts: dict[str, int] = {}
    out_base.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    wrote_any = False
    # Remove a partial/corrupt output left by a previous failed run.
    if out_base.exists():
        out_base.unlink()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        pa = pq = None

    fallback_csv = out_base.with_suffix('.csv')
    if fallback_csv.exists():
        fallback_csv.unlink()

    try:
        for i, chunk in enumerate(pd.read_csv(loan_path, usecols=usecols, chunksize=chunksize, low_memory=False)):
            raw_rows += len(chunk)
            vc = chunk['loan_status'].astype('string').str.strip().value_counts(dropna=False)
            for k, v in vc.items():
                key = str(k)
                status_counts[key] = status_counts.get(key, 0) + int(v)

            clean = prepare_loans(chunk)
            clean = clean[clean['annual_inc'] > 0].dropna(subset=['issue_date', 'target_default']).copy()
            kept_rows += len(clean)

            if clean.empty:
                print(f'Chunk {i+1}: raw={len(chunk):,} -> usable=0')
                continue

            # Reduce memory footprint of the cleaned representation.
            for c in clean.select_dtypes(include=['float64']).columns:
                clean[c] = clean[c].astype('float32')
            for c in clean.select_dtypes(include=['int64']).columns:
                clean[c] = clean[c].astype('int32')

            if pq is not None:
                table = pa.Table.from_pandas(clean, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_base, table.schema, compression='snappy')
                writer.write_table(table)
                wrote_any = True
            else:
                clean.to_csv(fallback_csv, mode='a', header=not fallback_csv.exists(), index=False)
                wrote_any = True

            print(f'Chunk {i+1}: raw={len(chunk):,} -> usable={len(clean):,}')
    finally:
        if writer is not None:
            writer.close()

    if not wrote_any:
        raise ValueError('Không có dòng hợp lệ sau cleaning.')

    # Read only the compact cleaned dataset for downstream modeling.
    if out_base.exists():
        cleaned = pd.read_parquet(out_base)
    else:
        cleaned = pd.read_csv(fallback_csv, parse_dates=['issue_date'])

    if 'id' in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=['id']).reset_index(drop=True)
        if out_base.exists():
            cleaned.to_parquet(out_base, index=False)
        else:
            cleaned.to_csv(fallback_csv, index=False)

    audit = {
        'pipeline_version': PIPELINE_VERSION,
        'raw_rows_seen': int(raw_rows),
        'usable_rows_before_dedup': int(kept_rows),
        'cleaned_rows_after_dedup': int(len(cleaned)),
        'chunksize': int(chunksize),
        'columns_read': usecols,
        'loan_status_counts_raw': status_counts,
        'cleaned_issue_years': sorted(cleaned['issue_year'].dropna().astype(int).unique().tolist()),
        'streaming_write': 'parquet' if out_base.exists() else 'csv',
    }
    (REPORTS / 'chunk_processing_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return cleaned, audit

def cap_training_set(tr: pd.DataFrame, max_rows: int = MAX_MODEL_TRAIN_ROWS) -> pd.DataFrame:
    """Keep modeling tractable on ordinary student laptops.

    The validation and test sets remain untouched. Sampling is stratified by target.
    The cap and sampling rule are reported for transparency.
    """
    if len(tr) <= max_rows:
        return tr
    # Stratified proportional sample without using future rows.
    pos = tr[tr.target_default == 1]
    neg = tr[tr.target_default == 0]
    pos_n = int(round(max_rows * len(pos) / len(tr)))
    neg_n = max_rows - pos_n
    parts = [
        pos.sample(n=min(pos_n, len(pos)), random_state=42),
        neg.sample(n=min(neg_n, len(neg)), random_state=42),
    ]
    out = pd.concat(parts).sort_values('issue_date').reset_index(drop=True)
    return out


def save_frame(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Save Parquet when available, otherwise use CSV without breaking the pipeline."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError, OSError):
        csv_path = parquet_path.with_suffix('.csv')
        frame.to_csv(csv_path, index=False)
        return csv_path


def run(smoke=False):
    loan_path = find_loan_csv()
    base_path = PROC / 'loan_cleaned_base.parquet'

    if smoke:
        # Smoke data are small enough for ordinary read_csv and do not use external APIs.
        raw = pd.read_csv(loan_path)
        loans = prepare_loans(raw).dropna(subset=['issue_date', 'target_default']).copy()
        loans = loans[loans['annual_inc'] > 0].copy()
        macro = pd.DataFrame()
        audit = {'smoke_test': True, 'raw_rows_seen': len(raw), 'cleaned_rows': len(loans)}
    elif base_path.exists():
        print(f'[Cache] Kiểm tra cleaned Parquet: {base_path}')
        try:
            audit_path = REPORTS / 'chunk_processing_audit.json'
            cached_audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
            loans = pd.read_parquet(base_path)
            # Invalidate older/stale caches. The source artifact is explicitly a 2007–2018
            # export; a one-vintage cleaned cache is suspicious and must be rebuilt.
            raw_audit = audit_raw_issue_years(loan_path, chunksize=CHUNK_SIZE)
            cached_years = sorted(pd.to_numeric(loans['issue_year'], errors='coerce').dropna().astype(int).unique().tolist()) if 'issue_year' in loans else []
            raw_years = sorted(int(y) for y in raw_audit['raw_issue_years'])
            cache_version = cached_audit.get('pipeline_version')
            stale_reason = None
            if cache_version != PIPELINE_VERSION:
                stale_reason = f'pipeline_version cache={cache_version!r}, current={PIPELINE_VERSION!r}'
            elif len(raw_years) >= 3 and len(cached_years) <= 1:
                stale_reason = f'raw có {len(raw_years)} vintage nhưng cache chỉ có {cached_years}'
            elif cached_years and min(cached_years) < EXPECTED_SOURCE_YEAR_SPAN[0] and len(raw_years) >= 3:
                stale_reason = 'cache coverage không phù hợp với source 2007–2018'
            if stale_reason:
                print(f'[Cache] Cache cũ/không hợp lệ ({stale_reason}); rebuild từ raw chunks.')
                try:
                    base_path.unlink()
                except OSError:
                    pass
                loans, audit = build_cleaned_dataset(loan_path, base_path)
                audit['raw_issue_year_audit'] = raw_audit
            else:
                audit = cached_audit or {'cached': True, 'pipeline_version': PIPELINE_VERSION}
                audit['raw_issue_year_audit'] = raw_audit
                audit['cached'] = True
            macro = read_macro()
        except Exception as exc:
            print(f'[Cache] Parquet không đọc được ({exc}); sẽ rebuild bằng chunk processing.')
            try:
                base_path.unlink()
            except OSError:
                pass
            loans, audit = build_cleaned_dataset(loan_path, base_path)
            macro = read_macro()
    else:
        print(f'[Data] Streaming cleaning từ: {loan_path}')
        loans, audit = build_cleaned_dataset(loan_path, base_path)
        audit['raw_issue_year_audit'] = audit_raw_issue_years(loan_path, chunksize=CHUNK_SIZE)
        macro = read_macro()

    if len(loans) < 1000:
        raise ValueError(f'Dataset còn quá ít dòng sau cleaning: {len(loans)}')
    if macro.empty and not smoke:
        raise RuntimeError('Không lấy được dữ liệu Việt Nam từ World Bank API. Không train Enhanced Model để tránh kết quả giả.')

    print(f'[Data] Cleaned rows ready for modeling: {len(loans):,}', flush=True)
    print(f'[Data] Issue-year coverage: {loans.issue_year.min()} -> {loans.issue_year.max()} ({loans.issue_year.nunique()} years)', flush=True)
    loans_ctx = add_vietnam_context(loans, macro)
    macro_cols = ['vn_gdp_growth_lag1','vn_inflation_rate_lag1','vn_bank_npl_ratio_lag1']
    matched_macro_rate = float(loans_ctx[macro_cols].notna().all(axis=1).mean()) if not loans_ctx.empty else 0.0
    print(f'[Data] Vietnam macro complete-match rate: {matched_macro_rate:.1%}', flush=True)
    ctx_path = PROC / 'loan_cleaned_vietnam_context.parquet'
    save_frame(loans_ctx, ctx_path)

    write_collection_manifest(loan_path, macro)
    results = []
    train_sizes = {}
    for kind, d, use_macro in [('base', loans, False), ('vietnam', loans_ctx, True)]:
        print(f'\n========== {kind.upper()} EXPERIMENT ==========', flush=True)
        tr, v, te = temporal_split(d)
        original_train_n = len(tr)
        tr_model = cap_training_set(tr) if not smoke else tr
        train_sizes[kind] = {'chronological_train_rows': int(original_train_n), 'rows_used_for_model_fit': int(len(tr_model))}

        num_cols, cat_cols = feature_columns(use_macro)
        cols = [*num_cols, *cat_cols]
        Xtr = tr_model[cols]
        ytr = tr_model.target_default
        Xv = v[cols]
        yv = v.target_default
        Xte = te[cols]
        yte = te.target_default
        print(f'[{kind}] Feature matrix: {len(cols)} raw features ({len(num_cols)} numeric, {len(cat_cols)} categorical)', flush=True)
        leaderboard, m = fit_and_select(Xtr, ytr, Xv, yv, Xte, yte, num_cols, cat_cols, kind, ROOT)
        results.append({'kind': kind, 'best_model': leaderboard.iloc[0]['model'], **m})

    print('\n[Pipeline] Both experiments completed. Building comparison report...', flush=True)
    comp = pd.DataFrame(results)
    comp.to_csv(REPORTS / 'model_comparison_base_vs_vietnam.csv', index=False)

    if set(comp.kind) >= {'base', 'vietnam'}:
        base = comp[comp.kind == 'base'].iloc[0]
        vn = comp[comp.kind == 'vietnam'].iloc[0]
        pd.DataFrame([
            {'metric': 'PR-AUC', 'base': base['PR-AUC'], 'vietnam': vn['PR-AUC'], 'delta': vn['PR-AUC'] - base['PR-AUC']},
            {'metric': 'Recall', 'base': base['Recall'], 'vietnam': vn['Recall'], 'delta': vn['Recall'] - base['Recall']},
            {'metric': 'Top20 Capture', 'base': base['Test top20_capture_rate'], 'vietnam': vn['Test top20_capture_rate'], 'delta': vn['Test top20_capture_rate'] - base['Test top20_capture_rate']},
            {'metric': 'Brier', 'base': base['Brier'], 'vietnam': vn['Brier'], 'delta': vn['Brier'] - base['Brier']},
        ]).to_csv(REPORTS / 'vietnam_context_impact.csv', index=False)

    meta = {
        'pipeline_version': PIPELINE_VERSION,
        'loan_source': str(loan_path.relative_to(ROOT)),
        'rows_after_cleaning': int(len(loans)),
        'default_rate': float(loans.target_default.mean()),
        'date_range': [str(loans.issue_date.min().date()), str(loans.issue_date.max().date())],
        'split': 'chronological 70/15/15',
        'macro_source': 'World Bank API — VNM; issue_year minus 1',
        'results_file': 'reports/model_comparison_base_vs_vietnam.csv',
        'chunk_size': CHUNK_SIZE,
        'max_model_train_rows': MAX_MODEL_TRAIN_ROWS,
        'train_sizes': train_sizes,
        'rejected_dataset_used': False,
        'issue_years_after_cleaning': sorted(loans['issue_year'].dropna().astype(int).unique().tolist()),
        'issue_year_count_after_cleaning': int(loans['issue_year'].nunique(dropna=True)),
        'matched_vietnam_macro_rate': matched_macro_rate,
    }
    (REPORTS / 'pipeline_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()
    run(smoke=args.smoke_test)
