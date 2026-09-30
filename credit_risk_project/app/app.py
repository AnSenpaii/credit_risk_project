from __future__ import annotations
import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from features import feature_columns

st.set_page_config(page_title='Credit Risk Intelligence',page_icon='📊',layout='wide')
MODELS=ROOT/'models'; REPORTS=ROOT/'reports'; PROC=ROOT/'data'/'processed'

@st.cache_resource
def load_bundle(kind):
    p=MODELS/f'best_model_{kind}.joblib'
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_loans(kind):
    p=PROC/('loan_cleaned_vietnam_context.parquet' if kind=='vietnam' else 'loan_cleaned_base.parquet')
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    csv=p.with_suffix('.csv')
    return pd.read_csv(csv) if csv.exists() else pd.DataFrame()

def band(prob,t):
    if prob>=t: return 'CAO — Ưu tiên rà soát','error'
    if prob>=0.20: return 'TRUNG BÌNH — Cần kiểm tra bổ sung','warning'
    return 'THẤP — Xử lý theo quy trình chuẩn','success'

st.title('📊 Credit Risk Intelligence')
st.caption('Data Science Capstone · Default Risk Ranking · Vietnam Context')

with st.sidebar:
    mode=st.radio('Mô hình',['Baseline — Hồ sơ vay','Enhanced — + Bối cảnh Việt Nam'])
    kind='vietnam' if mode.startswith('Enhanced') else 'base'
    st.info('Hệ thống chỉ hỗ trợ ưu tiên rà soát; không tự động phê duyệt/từ chối khoản vay.')

bundle=load_bundle(kind); loans=load_loans(kind)
if bundle is None or loans.empty:
    st.error('Chưa có model/data artifact. Chạy `python src/pipeline.py` trước.')
    st.stop()

num_cols=bundle['feature_num_cols']; cat_cols=bundle['feature_cat_cols']; cols=[*num_cols,*cat_cols]; model=bundle['model']; threshold=float(bundle['threshold'])

T1,T2,T3,T4,T5,T6,T7=st.tabs(['🏠 Tổng quan','🧾 Một hồ sơ','📥 CSV','🧪 So sánh','🔎 Explainability','📈 Business Impact','📊 EDA'])
with T1:
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Hồ sơ',f'{len(loans):,}')
    c2.metric('Tỷ lệ nợ xấu',f'{loans.target_default.mean():.1%}')
    c3.metric('Model',bundle['model_name'])
    c4.metric('Top-20% Capture',f"{bundle['test_metrics']['Test top20_capture_rate']:.1%}")
    st.subheader('Business question')
    st.write('Nếu chỉ kiểm tra kỹ 20% hồ sơ, mô hình có tìm được phần lớn khoản vay xấu hay không?')
    if (REPORTS/'pipeline_summary.json').exists():
        summary=json.loads((REPORTS/'pipeline_summary.json').read_text(encoding='utf-8'))
        st.write(f"Khoảng thời gian: **{summary['date_range'][0]} → {summary['date_range'][1]}**")

with T2:
    st.subheader('Thông tin có tại thời điểm xét hồ sơ')
    a,b=st.columns(2)
    with a:
        amount=st.number_input('Loan amount',min_value=100.0,value=10000.0)
        income=st.number_input('Annual income',min_value=1000.0,value=60000.0)
        rate=st.number_input('Interest rate (%)',min_value=0.0,max_value=40.0,value=12.0)
        dti=st.number_input('DTI (%)',min_value=0.0,max_value=100.0,value=20.0)
        installment=st.number_input('Installment / month',min_value=0.0,value=350.0)
    with b:
        term=st.selectbox('Term',[36,60])
        emp=st.number_input('Employment length (years)',min_value=0.0,max_value=10.0,value=5.0)
        home=st.selectbox('Home ownership',['RENT','MORTGAGE','OWN','OTHER'])
        purpose=st.selectbox('Purpose',['debt_consolidation','credit_card','home_improvement','medical','small_business'])
        delinq=st.number_input('Delinquencies (2y)',0,20,0)
        inquiries=st.number_input('Recent inquiries',0,20,1)
    latest_macro={'vn_gdp_growth_lag1':6.0,'vn_inflation_rate_lag1':3.0,'vn_bank_npl_ratio_lag1':2.0}
    macro_path=ROOT/'data'/'external'/'vn_macro_worldbank.csv'
    if macro_path.exists():
        try:
            mdf=pd.read_csv(macro_path).sort_values('year')
            if not mdf.empty:
                last=mdf.iloc[-1]
                latest_macro={
                    'vn_gdp_growth_lag1':last.get('vn_gdp_growth',np.nan),
                    'vn_inflation_rate_lag1':last.get('vn_inflation_rate',np.nan),
                    'vn_bank_npl_ratio_lag1':last.get('vn_bank_npl_ratio',np.nan),
                }
        except Exception:
            pass
    row=pd.DataFrame([{
        'loan_amnt':amount,'annual_inc':income,'int_rate':rate,'dti':dti,'installment':installment,'emp_length_years':emp,
        'delinq_2yrs':delinq,'inq_last_6mths':inquiries,'open_acc':8,'revol_util':50.0,'total_acc':20,'term_months':term,
        'home_ownership':home,'purpose':purpose,'loan_to_income':amount/income,'payment_to_income':installment*12/income,
        'income_log':np.log1p(income),'amount_log':np.log1p(amount),
        **latest_macro
    }])
    for c in cols:
        if c not in row: row[c]=0.0
    p=float(model.predict_proba(row[cols])[:,1][0])
    label,level=band(p,threshold)
    st.metric('Model probability of default',f'{p:.1%}')
    getattr(st,level)(label)
    st.progress(p)
    st.caption('Đây là xác suất dự đoán của model, không phải xác suất thực tế tuyệt đối và không phải quyết định tín dụng.')

with T3:
    up=st.file_uploader('Upload CSV',type=['csv'])
    if up:
        d=pd.read_csv(up)
        for c in cols:
            if c not in d: d[c]=np.nan
        d['predicted_pd']=model.predict_proba(d[cols])[:,1]
        d['risk_band']=pd.cut(d.predicted_pd,bins=[-np.inf,.20,threshold,np.inf],labels=['THẤP','TRUNG BÌNH','CAO'],include_lowest=True,right=False)
        st.dataframe(d.sort_values('predicted_pd',ascending=False).head(100),use_container_width=True)
        st.download_button('Tải kết quả CSV',d.to_csv(index=False).encode('utf-8-sig'),'credit_risk_scored.csv','text/csv')

with T4:
    p=REPORTS/'model_comparison_base_vs_vietnam.csv'
    q=REPORTS/'vietnam_context_impact.csv'
    if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True)
    if q.exists(): st.dataframe(pd.read_csv(q),use_container_width=True)

with T6:
    st.subheader('Business Impact')
    bp = REPORTS/f'business_summary_{kind}.json'
    gp = REPORTS/'plots'/f'gains_curve_{kind}.png'
    pp = REPORTS/'plots'/f'precision_recall_{kind}.png'
    if bp.exists():
        bsum=json.loads(bp.read_text(encoding='utf-8'))
        c1,c2,c3=st.columns(3)
        c1.metric('Top-20% Capture',f"{bsum['Test top20_capture_rate']:.1%}")
        c2.metric('Top-20% Lift',f"{bsum['Test top20_lift']:.2f}×")
        c3.metric('Recall',f"{bsum['Recall']:.1%}")
    if gp.exists(): st.image(str(gp),caption='Cumulative gains — model vs random')
    if pp.exists(): st.image(str(pp),caption='Precision–Recall curve')
    tp=REPORTS/f'threshold_analysis_{kind}.csv'
    if tp.exists():
        st.markdown('### Threshold analysis')
        st.dataframe(pd.read_csv(tp),use_container_width=True,height=320)

with T5:
    p=REPORTS/f'feature_importance_{kind}.csv'
    if p.exists(): st.dataframe(pd.read_csv(p).head(20),use_container_width=True)
    for name in [f'confusion_{kind}.png',f'calibration_{kind}.png']:
        img=REPORTS/'plots'/name
        if img.exists(): st.image(str(img),caption=name)

with T7:
    st.subheader('Exploratory Data Analysis')
    eda_imgs=[
        'eda_target_distribution.png',
        'eda_default_rate_by_year.png',
        'eda_default_rate_by_loan_amount.png',
        'eda_default_rate_by_dti.png',
        'eda_default_rate_by_purpose.png',
        'eda_default_rate_by_term.png',
    ]
    for name in eda_imgs:
        img=REPORTS/'plots'/name
        if img.exists(): st.image(str(img),caption=name)
    ep=REPORTS/'eda_insights.md'
    if ep.exists():
        st.markdown(ep.read_text(encoding='utf-8'))
