from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
PROC = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
sys.path.insert(0, str(ROOT / 'src'))

from analysis import run_eda  # noqa: E402
from modeling import (
    metrics, top20, bootstrap_top20, plot_confusion, plot_calibration,
    plot_pr_curve, plot_gains_curve, save_threshold_analysis,
)
# Keep the script independent of pipeline internals.


def temporal_split(df: pd.DataFrame):
    d = df.sort_values('issue_date').reset_index(drop=True)
    n = len(d)
    a = int(n * 0.70)
    b = int(n * 0.85)
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()


def load_loans(kind: str) -> pd.DataFrame:
    name = 'loan_cleaned_vietnam_context.parquet' if kind == 'vietnam' else 'loan_cleaned_base.parquet'
    p = PROC / name
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    csv_path = p.with_suffix('.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=['issue_date'])
    raise FileNotFoundError(f'Missing cleaned artifact: {p} or {csv_path}')


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    plots = REPORTS / 'plots'
    plots.mkdir(parents=True, exist_ok=True)

    base = load_loans('base')
    print(f'[Analysis] Cleaned loan rows: {len(base):,}', flush=True)
    run_eda(base, REPORTS)
    print('[Analysis] EDA pack created.', flush=True)

    summaries = []
    for kind in ('base', 'vietnam'):
        bundle_path = MODELS / f'best_model_{kind}.joblib'
        if not bundle_path.exists():
            raise FileNotFoundError(f'Missing model bundle: {bundle_path}. Run pipeline once first.')
        bundle = joblib.load(bundle_path)
        loans = base if kind == 'base' else load_loans('vietnam')
        _, _, test = temporal_split(loans)
        num_cols = bundle['feature_num_cols']
        cat_cols = bundle['feature_cat_cols']
        cols = [*num_cols, *cat_cols]
        missing = [c for c in cols if c not in test.columns]
        if missing:
            raise ValueError(f'{kind}: cleaned test set missing features: {missing}')
        p = bundle['model'].predict_proba(test[cols])[:, 1]
        y = test['target_default'].to_numpy()
        t = float(bundle['threshold'])
        m = metrics(y, p, t)
        t20 = top20(y, p)
        ci = bootstrap_top20(y, p, B=120, seed=42)
        result = {'kind': kind, 'best_model': bundle['model_name'], **m}
        result.update({f'Test {k}': v for k, v in t20.items()})
        result.update({'Top20 CI low': ci['low'], 'Top20 CI high': ci['high'], 'Threshold': t})
        summaries.append(result)

        pd.DataFrame({'y_true': y, 'predicted_pd': p}).to_csv(REPORTS / f'test_predictions_{kind}.csv', index=False)
        plot_confusion(y, p, t, plots / f'confusion_{kind}.png', f'Confusion Matrix — {kind}')
        plot_calibration(y, p, plots / f'calibration_{kind}.png', f'Calibration — {kind}')
        plot_pr_curve(y, p, plots / f'precision_recall_{kind}.png', f'Precision–Recall — {kind}')
        plot_gains_curve(y, p, plots / f'gains_curve_{kind}.png', f'Cumulative Gains — {kind}')
        save_threshold_analysis(y, p, REPORTS / f'threshold_analysis_{kind}.csv')
        print(f'[Analysis] {kind}: test predictions + business plots generated.', flush=True)

    comp = pd.DataFrame(summaries)
    comp.to_csv(REPORTS / 'model_comparison_base_vs_vietnam_analysis_recomputed.csv', index=False)

    if len(comp) == 2:
        b = comp[comp.kind == 'base'].iloc[0]
        v = comp[comp.kind == 'vietnam'].iloc[0]
        impact = pd.DataFrame([
            {'metric': 'PR-AUC', 'base': b['PR-AUC'], 'vietnam': v['PR-AUC'], 'delta': v['PR-AUC'] - b['PR-AUC']},
            {'metric': 'ROC-AUC', 'base': b['ROC-AUC'], 'vietnam': v['ROC-AUC'], 'delta': v['ROC-AUC'] - b['ROC-AUC']},
            {'metric': 'Precision', 'base': b['Precision'], 'vietnam': v['Precision'], 'delta': v['Precision'] - b['Precision']},
            {'metric': 'Recall', 'base': b['Recall'], 'vietnam': v['Recall'], 'delta': v['Recall'] - b['Recall']},
            {'metric': 'F1', 'base': b['F1'], 'vietnam': v['F1'], 'delta': v['F1'] - b['F1']},
            {'metric': 'Brier', 'base': b['Brier'], 'vietnam': v['Brier'], 'delta': v['Brier'] - b['Brier']},
            {'metric': 'Top20 Capture', 'base': b['Test top20_capture_rate'], 'vietnam': v['Test top20_capture_rate'], 'delta': v['Test top20_capture_rate'] - b['Test top20_capture_rate']},
            {'metric': 'Top20 Lift', 'base': b['Test top20_lift'], 'vietnam': v['Test top20_lift'], 'delta': v['Test top20_lift'] - b['Test top20_lift']},
        ])
        impact.to_csv(REPORTS / 'vietnam_context_impact_recomputed.csv', index=False)

        report = [
            '# Analysis Summary', '',
            f"Cleaned loans: **{len(base):,}**", 
            f"Issue-year coverage: **{int(base.issue_year.min())}–{int(base.issue_year.max())}**", '',
            '| Experiment | Model | PR-AUC | ROC-AUC | Recall | F1 | Top-20% Capture | Lift |',
            '|---|---|---:|---:|---:|---:|---:|---:|',
        ]
        for _, r in comp.iterrows():
            report.append(
                f"| {r['kind']} | {r['best_model']} | {r['PR-AUC']:.4f} | {r['ROC-AUC']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['Test top20_capture_rate']:.1%} | {r['Test top20_lift']:.2f}× |"
            )
        report += [
            '', '## How to interpret the business metric',
            '',
            'Top-20% Capture answers the capacity-constrained review question: what share of defaults is found when reviewers focus on the 20% highest-risk applications?',
            '',
            'Lift compares that capture concentration with random selection. A lift above 1 means the model concentrates defaults better than random review.',
            '',
            '## Generated artifacts', '',
            '- `eda_*.png` and grouped CSV tables',
            '- `precision_recall_*.png`',
            '- `gains_curve_*.png`',
            '- `threshold_analysis_*.csv`',
            '- `test_predictions_*.csv`',
        ]
        (REPORTS / 'analysis_summary.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

    print('[Analysis] Complete. No model retraining was performed.', flush=True)


if __name__ == '__main__':
    main()
