from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SEED = 42


def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), cat_cols),
    ])


def build_models(num_cols, cat_cols, scale_pos_weight=1.0):
    models = {
        'Logistic Regression': Pipeline([
            ('prep', build_preprocessor(num_cols, cat_cols)),
            ('model', LogisticRegression(
                max_iter=1200, class_weight='balanced', random_state=SEED,
                solver='lbfgs', n_jobs=None,
            )),
        ]),
        'Random Forest': Pipeline([
            ('prep', build_preprocessor(num_cols, cat_cols)),
            ('model', RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight='balanced_subsample',
                n_jobs=4,
                random_state=SEED,
            )),
        ]),
    }
    if XGBClassifier is not None:
        models['XGBoost'] = Pipeline([
            ('prep', build_preprocessor(num_cols, cat_cols)),
            ('model', XGBClassifier(
                n_estimators=140,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                random_state=SEED,
                n_jobs=4,
                scale_pos_weight=max(scale_pos_weight, 1.0),
            )),
        ])
    return models


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'PR-AUC': float(average_precision_score(y, p)),
        'ROC-AUC': float(roc_auc_score(y, p)),
        'Precision': float(precision_score(y, pred, zero_division=0)),
        'Recall': float(recall_score(y, pred, zero_division=0)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'Brier': float(brier_score_loss(y, p)),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def top20(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n = max(1, int(np.ceil(0.20 * len(y))))
    idx = np.argsort(-p)[:n]
    total = int(y.sum())
    captured = int(y[idx].sum())
    capture = (captured / total) if total else np.nan
    base_rate = y.mean() if len(y) else np.nan
    precision = float(y[idx].mean()) if n else np.nan
    lift = (precision / base_rate) if base_rate > 0 else np.nan
    return {
        'top20_n': n,
        'top20_defaults_captured': captured,
        'top20_capture_rate': float(capture),
        'top20_precision': precision,
        'top20_lift': float(lift),
    }


def bootstrap_top20(y, p, B=120, seed=42):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    vals = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]; pp = p[idx]
        vals.append(top20(yy, pp)['top20_capture_rate'])
    return {
        'mean': float(np.nanmean(vals)),
        'low': float(np.nanpercentile(vals, 2.5)),
        'high': float(np.nanpercentile(vals, 97.5)),
    }


def choose_threshold(y, p):
    # Illustrative business cost: false negative is five times false positive.
    best = (0.5, float('inf'))
    for t in np.linspace(0.05, 0.80, 151):
        pred = (p >= t).astype(int)
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        cost = 5 * fn + fp
        if cost < best[1]:
            best = (float(t), float(cost))
    return best[0]


def plot_confusion(y, p, t, path, title):
    cm = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0, 1], ['Pred 0', 'Pred 1'])
    ax.set_yticks([0, 1], ['Actual 0', 'Actual 1'])
    ax.set_title(title); ax.set_xlabel('Prediction'); ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(ax.images[0], ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_calibration(y, p, path, title):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    ax.plot(mean, frac, 'o-', label='Model')
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed default rate')
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_pr_curve(y, p, path, title):
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    baseline = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=f'Model AP={ap:.3f}')
    ax.axhline(baseline, linestyle='--', label=f'Random baseline={baseline:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    _save_fig(fig, path)


def _save_fig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)


def plot_gains_curve(y, p, path, title):
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p)
    yy = y[order]
    cum_defaults = np.cumsum(yy)
    total_defaults = max(int(y.sum()), 1)
    population_pct = np.arange(1, len(y) + 1) / len(y)
    gains = cum_defaults / total_defaults
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(population_pct, gains, label='Model cumulative gains')
    ax.plot([0, 1], [0, 1], '--', label='Random selection')
    ax.axvline(0.20, linestyle=':')
    top20_gain = float(gains[max(1, int(np.ceil(0.20 * len(y)))) - 1])
    ax.axhline(top20_gain, linestyle=':')
    ax.set_xlabel('Share of applications reviewed'); ax.set_ylabel('Share of defaults captured')
    ax.set_title(f'{title} | Top-20% capture={top20_gain:.1%}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    _save_fig(fig, path)


def save_threshold_analysis(y, p, path):
    rows = []
    for t in np.linspace(0.05, 0.80, 76):
        m = metrics(y, p, float(t))
        rows.append({'threshold': float(t), **m})
    pd.DataFrame(rows).to_csv(path, index=False)


def save_business_summary(y, p, threshold, path):
    m = metrics(y, p, threshold)
    t20 = top20(y, p)
    out = {**m, **{f'Test {k}': v for k, v in t20.items()}, 'random_top20_expected_capture': 0.20}
    out['top20_lift_vs_random'] = out['Test top20_capture_rate'] / 0.20
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


def fit_and_select(Xtr, ytr, Xv, yv, Xte, yte, num_cols, cat_cols, kind, root):
    root = Path(root); models_dir = root / 'models'; reports = root / 'reports'; plots = reports / 'plots'
    models_dir.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    model_set = build_models(num_cols, cat_cols, spw)
    rows = []; fitted = {}
    print(f'\n[{kind}] Start model comparison: {len(ytr):,} train rows | {len(yv):,} validation rows | {len(yte):,} test rows', flush=True)
    for idx, (name, model) in enumerate(model_set.items(), 1):
        print(f'[{kind}] [{idx}/{len(model_set)}] Training {name}...', flush=True)
        model.fit(Xtr, ytr)
        print(f'[{kind}] [{idx}/{len(model_set)}] Predicting validation...', flush=True)
        pv = model.predict_proba(Xv)[:, 1]
        rows.append({
            'model': name,
            'validation_pr_auc': average_precision_score(yv, pv),
            'validation_roc_auc': roc_auc_score(yv, pv),
        })
        fitted[name] = model
        print(f'[{kind}] [{idx}/{len(model_set)}] Done {name}: PR-AUC={rows[-1]["validation_pr_auc"]:.4f}, ROC-AUC={rows[-1]["validation_roc_auc"]:.4f}', flush=True)

    leaderboard = pd.DataFrame(rows).sort_values('validation_pr_auc', ascending=False).reset_index(drop=True)
    best_name = leaderboard.iloc[0]['model']; best = fitted[best_name]
    print(f'[{kind}] Best on validation: {best_name}', flush=True)
    print(f'[{kind}] Choosing threshold on validation...', flush=True)
    pv = best.predict_proba(Xv)[:, 1]
    threshold = choose_threshold(yv, pv)
    print(f'[{kind}] Threshold={threshold:.3f}; evaluating test...', flush=True)
    pte = best.predict_proba(Xte)[:, 1]
    m = metrics(yte, pte, threshold); t = top20(yte, pte); ci = bootstrap_top20(yte, pte)
    m.update({f'Test {k}': v for k, v in t.items()})
    m['Top20 CI low'] = ci['low']; m['Top20 CI high'] = ci['high']; m['Threshold'] = threshold
    leaderboard.to_csv(reports / f'model_leaderboard_{kind}.csv', index=False)
    pd.DataFrame({'y_true': np.asarray(yte), 'predicted_pd': pte}).to_csv(reports / f'test_predictions_{kind}.csv', index=False)
    plot_confusion(yte, pte, threshold, plots / f'confusion_{kind}.png', f'Confusion Matrix — {kind}')
    plot_calibration(yte, pte, plots / f'calibration_{kind}.png', f'Calibration — {kind}')
    plot_pr_curve(yte, pte, plots / f'precision_recall_{kind}.png', f'Precision–Recall — {kind}')
    plot_gains_curve(yte, pte, plots / f'gains_curve_{kind}.png', f'Cumulative Gains — {kind}')
    save_threshold_analysis(yte, pte, reports / f'threshold_analysis_{kind}.csv')
    save_business_summary(yte, pte, threshold, reports / f'business_summary_{kind}.json')
    bundle = {
        'model': best,
        'model_name': best_name,
        'feature_num_cols': num_cols,
        'feature_cat_cols': cat_cols,
        'threshold': threshold,
        'test_metrics': m,
        'kind': kind,
    }
    joblib.dump(bundle, models_dir / f'best_model_{kind}.joblib')

    try:
        sample_n = min(10_000, len(Xte))
        Xpi = Xte.sample(n=sample_n, random_state=SEED) if sample_n < len(Xte) else Xte
        ypi = yte.loc[Xpi.index]
        print(f'[{kind}] Computing permutation importance on {sample_n:,} test rows (3 repeats)...', flush=True)
        pi = permutation_importance(
            best, Xpi, ypi, scoring='average_precision', n_repeats=3,
            random_state=SEED, n_jobs=4, max_samples=1.0,
        )
        names = list(Xpi.columns)
        imp = pd.DataFrame({
            'feature': names,
            'importance_mean': pi.importances_mean,
            'importance_std': pi.importances_std,
        }).sort_values('importance_mean', ascending=False)
        imp.to_csv(reports / f'feature_importance_{kind}.csv', index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        top = imp.head(12).sort_values('importance_mean')
        ax.barh(top['feature'].astype(str), top['importance_mean'])
        ax.set_xlabel('Mean decrease in average precision')
        ax.set_title(f'Top feature importance — {kind}')
        _save_fig(fig, plots / f'feature_importance_{kind}.png')
        print(f'[{kind}] Permutation importance done.', flush=True)
    except Exception as exc:
        (reports / f'feature_importance_{kind}_error.txt').write_text(str(exc), encoding='utf-8')
        print(f'[{kind}] Permutation importance skipped: {exc}', flush=True)
    print(f'[{kind}] Finished.', flush=True)
    return leaderboard, m
