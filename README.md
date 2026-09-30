# 📊 Credit Risk Analysis - Vietnam Context

## Đề tài: Phân tích và Dự đoán Rủi ro Tín dụng

Dự án phân tích rủi ro tín dụng sử dụng dữ liệu Lending Club, kết hợp bối cảnh kinh tế vĩ mô Việt Nam từ World Bank.

### 🏗️ Cấu trúc Project

```
credit_risk_project/
├── app/                  # Streamlit web application
│   └── app.py
├── config/               # Configuration files
│   └── data_sources.json
├── data/
│   ├── raw/              # Raw Lending Club data
│   ├── processed/        # Cleaned & processed data
│   └── external/         # Vietnam macro data (World Bank)
├── models/               # Trained model files (.joblib)
├── notebooks/            # Jupyter notebooks
│   └── run_pipeline.ipynb
├── reports/              # Analysis reports & visualizations
│   └── plots/            # Generated charts
├── scripts/              # Utility scripts
│   ├── download_data.py
│   ├── generate_analysis.py
│   └── make_smoke_data.py
└── src/                  # Source code modules
    ├── analysis.py
    ├── data_collection.py
    ├── features.py
    ├── modeling.py
    └── pipeline.py
```

### 🚀 Tính năng chính

- **Thu thập dữ liệu**: Download & xử lý dữ liệu Lending Club + World Bank
- **EDA (Exploratory Data Analysis)**: Phân tích tỷ lệ default theo nhiều chiều
- **Feature Engineering**: Tạo đặc trưng kết hợp bối cảnh Việt Nam
- **Modeling**: So sánh nhiều mô hình ML (base vs Vietnam context)
- **Web App**: Ứng dụng Streamlit để demo dự đoán rủi ro

### 📦 Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 📈 Chạy Pipeline

```bash
# Chạy qua notebook
jupyter notebook notebooks/run_pipeline.ipynb

# Hoặc chạy Streamlit app
streamlit run app/app.py
```

### 📊 Kết quả

Xem chi tiết trong thư mục `reports/`:
- `analysis_summary.md` - Tổng quan phân tích
- `model_comparison_base_vs_vietnam.csv` - So sánh mô hình
- `eda_insights.md` - Insights từ EDA
- `plots/` - Biểu đồ trực quan

### 🛠️ Công nghệ sử dụng

- Python 3.13
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook
