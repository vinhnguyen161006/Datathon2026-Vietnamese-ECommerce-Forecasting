# BÁO CÁO DỰ ÁN: DỰ BÁO DOANH THU THƯƠNG MẠI ĐIỆN TỬ

**Bài toán:** Dự báo Doanh thu & COGS theo ngày (Sales Forecasting) — Datathon 2026 · The Gridbreaker

**Nộp bài:**
- **GitHub (repo này):** Code, notebooks, báo cáo LaTeX, submission.csv
- **Kaggle:** [Datathon 2026 Round 1](https://www.kaggle.com/competitions/datathon-2026-round-1) 

---

## 1. TỔNG QUAN

Dự án phân tích dữ liệu e-commerce thời trang Việt Nam (2012–2022) và dự báo doanh thu hàng ngày cho giai đoạn 01/01/2023–01/07/2024. Tiêu chí đánh giá: **MAE, RMSE, R²** — MAE càng thấp càng tốt.

Cuộc thi gồm 3 phần:
- **Part 1 (20đ):** 10 câu hỏi trắc nghiệm tính trực tiếp từ CSV
- **Part 2 (60đ):** EDA & Visualization — 6 chủ đề phân tích theo 4 cấp độ (Descriptive → Prescriptive)
- **Part 3 (20đ):** Mô hình dự báo doanh thu + báo cáo kỹ thuật

---

## 2. CẤU TRÚC DỰ ÁN

```
vinuiDatathon/
├── data/
│   ├── raw/               # 14 file CSV gốc từ Kaggle (không commit — xem mục 4)
│   └── processed/         # Parquet cache tự động sinh ra (không commit)
├── notebooks/
│   ├── 00_executive_summary.ipynb   # KPI dashboard + chiến lược tổng thể
│   ├── 01_data_quality_check.ipynb  # Kiểm tra schema, null, FK
│   ├── 02_mcq_answers.ipynb         # Part 1 — 10 câu trắc nghiệm
│   ├── 03_eda_customer.ipynb        # Vòng đời khách hàng & LTV theo kênh
│   ├── 04_eda_promotion.ipynb       # ROI khuyến mãi & tối ưu dải chiết khấu
│   ├── 05_eda_returns.ipynb         # Phân tích hoàn trả & chi phí refund
│   ├── 06_eda_inventory.ipynb       # Sức khỏe tồn kho & doanh thu mất do hết hàng
│   ├── 07_eda_traffic.ipynb         # Phễu web traffic & tối ưu chuyển đổi
│   └── 08_eda_reviews.ipynb         # Đánh giá sản phẩm & rủi ro uy tín
├── src/
│   ├── data_loader.py     # load_all_tables(), parquet cache, feature helpers
│   ├── features.py        # Feature engineering (lag, rolling, EWM, calendar)
│   ├── models.py          # train_lgbm(), train_prophet(), predict()
│   └── metrics.py         # evaluate() → MAE, RMSE, R²
├── scripts/
│   ├── train_final.py     # Pipeline huấn luyện end-to-end
│   ├── predict.py         # Sinh outputs/submission.csv
│   └── generate_shap.py   # Sinh SHAP plots → reports/
├── outputs/
│   └── submission.csv     # File nộp Kaggle (Date, Revenue, COGS)
├── reports/
│   ├── main.tex           # Báo cáo kỹ thuật (NeurIPS 2025 format, 4 trang)
│   ├── shap_bar.png       # SHAP feature importance plot
│   └── shap_summary.png   # SHAP beeswarm plot
└── requirements.txt
```

---

## 3. PIPELINE MÔ HÌNH

### Step 1: Feature Engineering (`src/features.py`)

- **Tổng số features:** 62, chia thành 11 nhóm
- **Chiến lược:** Tất cả lag/rolling đều shift(1) để tránh data leakage
- **Điểm đặc biệt:** Kết hợp dữ liệu từ 4 bảng — sales, promotions, inventory, web_traffic

| Nhóm | Số lượng | Ví dụ |
|---|---|---|
| Calendar | 12 | `month`, `day_of_week`, `is_weekend`, `sin/cos_doy` |
| Business flags | 5 | `is_tet`, `is_sale_season`, `is_mid_month` |
| Lag Revenue | 10 | Lag 1, 2, 3, 7, 14, 30, 90, 180, 365, 730 ngày |
| Rolling stats | 12 | Mean/std/max qua 7, 14, 30, 90 ngày |
| EWM | 2 | Span-7 và Span-30 |
| YoY ratio | 2 | 7-ngày và 30-ngày so cùng kỳ năm trước |
| Promo | 3 | `n_active_promos`, `promo_discount_avg` |
| Inventory | 3 | `avg_fill_rate`, `avg_stockout_flag` |
| Web traffic | 3 | `session_lag_1`, `bounce_rate_7d_avg` |
| Log transforms | 6 | `log(Revenue_lag_{1,7,30,365})` |
| Trend | 2 | `days_since_2020`, `revenue_hist_monthly_mean` |

### Step 2: Huấn luyện (`scripts/train_final.py`)

- **Kiến trúc:** LightGBM (`n_estimators=5000`, `learning_rate=0.01`, `num_leaves=63`, `max_depth=8`)
- **Chiến lược:** Forward-walk `TimeSeriesSplit(n_splits=5)` — tuyệt đối không dùng KFold shuffle để tránh leakage thời gian
- **Điểm đặc biệt:** Train 2 model song song — Revenue model và COGS model; Prophet fit riêng với Vietnamese holidays

### Step 3: Dự báo autoregressive (`scripts/predict.py`)

- **Kiến trúc:** Ensemble LightGBM 80% + Prophet 20%
- **Chiến lược:** Dự báo tuần tự ngày-qua-ngày, prediction hôm nay trở thành lag feature cho ngày mai
- **Điểm đặc biệt:** Post-COVID recovery scaling (2023×1.352, 2024×1.438) để bù phân phối lệch do COVID

```
ŷ_t     = 0.80 × LGBM(t) + 0.20 × Prophet(t)
ŷ_final = ŷ_t × scale_year
COGS_t  = ŷ_final × margin_month
```

---

## 4. HƯỚNG DẪN CHẠY LẠI

### 4.1. Tải dữ liệu từ Kaggle

Tải 14 file CSV từ [Kaggle competition page](https://www.kaggle.com/competitions/datathon-2026-round-1/data) và đặt vào thư mục `data/raw/`:

```
data/raw/products.csv, customers.csv, orders.csv, order_items.csv,
         payments.csv, shipments.csv, returns.csv, reviews.csv,
         promotions.csv, geography.csv, inventory.csv, web_traffic.csv,
         sales.csv, sample_submission.csv
```

### 4.2. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 4.3. Chạy pipeline

```bash
# Huấn luyện mô hình (lưu vào outputs/*.pkl)
python scripts/train_final.py

# Sinh submission (lưu vào outputs/submission.csv)
python scripts/predict.py

# Sinh SHAP plots (lưu vào reports/)
python scripts/generate_shap.py
```

### 4.4. Lưu ý

- Chạy từ **thư mục gốc project** (`d:/vinuiDatathon/`)
- `SEED = 42` được set cho `numpy`, `random`, `lightgbm` — kết quả hoàn toàn reproducible
- `data/processed/` sinh tự động lần đầu, các lần sau load từ parquet cache (nhanh hơn)
- `train_final.py` mất ~5–10 phút (LightGBM 5000 estimators + Prophet)

---

## 5. BÁO CÁO KỸ THUẬT

Xem `reports/main.tex` (hoặc compile ra PDF bằng Overleaf) — NeurIPS 2025 format, 4 trang, bao gồm:
- Bảng feature engineering đầy đủ
- Giải thích chiến lược CV và autoregressive prediction
- SHAP feature importance với giá trị thực tế
- Bảng ablation study 7 cấu hình

