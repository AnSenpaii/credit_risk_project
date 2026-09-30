# Credit Risk Intelligence — Vietnam Context (FDSC348464)

Đây là phiên bản nâng cấp của project cuối kỳ Data Science, bám outline đã được duyệt:

> Ứng dụng Học máy trong Phân tích và Dự báo Nguy cơ Nợ xấu nhằm Tối ưu hóa Quyết định Giải ngân Tín dụng.

## Ý tưởng cốt lõi

Bài toán không phải “AI quyết định cho vay”. Mục tiêu là xếp hạng hồ sơ theo **Probability of Default (PD)** để nhân viên ưu tiên rà soát. Câu hỏi kinh doanh:

> Nếu chỉ có thể kiểm tra kỹ 20% hồ sơ, mô hình có tìm được phần lớn khoản vay xấu hay không?

## Thiết kế dữ liệu mới

### 1) Loan-level data

Nguồn khoản vay là bộ **Lending Club** công khai trên Kaggle. Project **không nhúng dữ liệu khổng lồ vào ZIP**; thay vào đó có script tải bằng Kaggle API. Dataset có các trường phù hợp với bài toán như `loan_amnt`, `annual_inc`, `term`, `int_rate`, `dti`, `purpose`, `issue_d` và trạng thái khoản vay. Chỉ dùng các khoản có kết quả cuối cùng **Fully Paid / Charged Off**.

Nguồn tham khảo: https://www.kaggle.com/datasets/wordsforthewise/lending-club

### 2) Vietnam context data

FRED được **loại khỏi pipeline chính**. Thay vào đó, project lấy chỉ số Việt Nam từ **World Bank Indicators API**:

- GDP growth: `NY.GDP.MKTP.KD.ZG`
- CPI inflation: `FP.CPI.TOTL.ZG`
- Bank NPL ratio: `FB.AST.NPER.ZS`

Dữ liệu kinh tế được ghép theo **năm phát hành khoản vay - 1** để tránh look-ahead leakage.

Nguồn: https://api.worldbank.org/ và https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

## Hai thí nghiệm chính

### Model A — Loan-only baseline

Chỉ dùng thông tin có tại thời điểm xét hồ sơ:
- loan amount
- income
- term
- interest rate
- DTI
- employment length
- home ownership
- loan purpose
- loan grade/subgrade (chỉ khi tồn tại sẵn tại thời điểm xét hồ sơ)
- credit history / inquiries / delinquencies và các biến hồ sơ tín dụng phù hợp

### Model B — Loan + Vietnam context

Model A + 3 biến kinh tế Việt Nam lag 1 năm.

Câu hỏi nghiên cứu: **bổ sung bối cảnh kinh tế Việt Nam có cải thiện PR-AUC / Recall / Top-20% Capture hay không?**

## Modeling

- Logistic Regression — baseline dễ giải thích
- Random Forest — nonlinear challenger
- XGBoost — gradient boosting challenger nếu package khả dụng

Chia dữ liệu theo thời gian **70% / 15% / 15%** dựa trên `issue_date`.

Model được chọn chỉ trên validation; test set được khóa để đánh giá cuối cùng.

## Evaluation

Metric chính:

- PR-AUC
- Recall
- Precision
- F1
- ROC-AUC
- Brier Score
- Calibration
- Confusion Matrix
- **Top-20% Capture Rate**
- **Lift vs random**
- Bootstrap 95% CI cho Top-20% Capture

Không sử dụng Accuracy làm metric quyết định chính vì nợ xấu là lớp thiểu số.

## Explainability

Project có:
- Permutation Feature Importance
- bảng hệ số Logistic Regression (khi phù hợp)
- calibration curve
- model card

## Deployment

Streamlit gồm:

1. Tổng quan
2. Chấm điểm một hồ sơ
3. Chấm điểm CSV hàng loạt
4. So sánh Model A / Model B
5. Explainability
6. Data provenance

Hệ thống chỉ đưa ra **mức ưu tiên rà soát**: thấp / trung bình / cao. Không có nút “phê duyệt” hoặc “từ chối” tự động.

## Cài đặt

### Windows / VS Code

```powershell
cd E:\credit_risk_project
conda create -n credit-risk python=3.11 -y
conda activate credit-risk
pip install -r requirements.txt
```

### Tải dữ liệu

```powershell
python scripts\download_data.py
```

Nếu dùng Kaggle API, đặt `KAGGLE_USERNAME` và `KAGGLE_KEY` trong Environment Variables hoặc cấu hình `~/.kaggle/kaggle.json`.

Sau khi tải Lending Club, script sẽ để file raw trong `data/raw/`.

### Thu thập + train

```powershell
python src\pipeline.py
```

### Chạy app

```powershell
python -m streamlit run app\app.py
```

## Smoke test không cần Internet

Nếu chỉ muốn kiểm tra code:

```powershell
python scripts\make_smoke_data.py
python src\pipeline.py --smoke-test
```

Smoke data **chỉ để test kỹ thuật**, không được dùng làm kết quả báo cáo.

## Cấu trúc

```text
credit_risk_project/
├── app/
│   └── app.py
├── config/
│   └── data_sources.json
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── plots/
├── scripts/
│   ├── download_data.py
│   └── make_smoke_data.py
├── src/
│   ├── data_collection.py
│   ├── features.py
│   ├── modeling.py
│   └── pipeline.py
└── requirements.txt
```

## Lưu ý học thuật

1. Dữ liệu Lending Club là dữ liệu Mỹ; kết quả **không đại diện hoàn toàn cho tín dụng Việt Nam**.
2. Điểm mới của project là thay FRED bằng **Vietnam context** lấy qua API chính thức và kiểm định xem context này có cải thiện ranking rủi ro hay không.
3. Không gọi dữ liệu suy diễn là dữ liệu gốc.
4. Không tạo fallback giả khi API lỗi.
5. Không đưa biến xuất hiện sau khi giải ngân vào mô hình.

## Ghi chú chạy dữ liệu lớn

Dataset Lending Club có kích thước rất lớn. Pipeline mới không đọc toàn bộ CSV vào RAM: nó chọn đúng các cột cần thiết, xử lý theo **100.000 dòng/chunk**, lọc Fully Paid/Charged Off ngay khi đọc và ghi bộ dữ liệu sạch sang Parquet. Bộ rejected không được dùng cho mô hình.

Để phù hợp máy cá nhân, pipeline giữ nguyên validation/test theo thời gian và có thể giới hạn tập train tối đa 500.000 dòng bằng lấy mẫu phân tầng; giới hạn này được ghi trong `reports/pipeline_summary.json`.
