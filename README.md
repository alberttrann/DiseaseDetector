# Hệ Thống Chẩn Đoán Lúa & Gợi Ý Cây Trồng

> Bộ công cụ AI tích hợp gồm hai mô hình học máy: **phát hiện bệnh lúa từ ảnh** và **gợi ý cây trồng tối ưu dựa trên thông số đất và khí hậu**.

---

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Cấu Trúc Thư Mục](#2-cấu-trúc-thư-mục)
3. [Mô Hình 1 – Phát Hiện Bệnh Lúa (EfficientNet)](#3-mô-hình-1--phát-hiện-bệnh-lúa-efficientnet)
4. [Mô Hình 2 – Gợi Ý Cây Trồng (NPK)](#4-mô-hình-2--gợi-ý-cây-trồng-npk)
5. [Phân Tích Dữ Liệu Khám Phá (EDA)](#5-phân-tích-dữ-liệu-khám-phá-eda)
6. [Kết Quả Đánh Giá](#6-kết-quả-đánh-giá)
7. [Cài Đặt & Yêu Cầu Hệ Thống](#7-cài-đặt--yêu-cầu-hệ-thống)
8. [Hướng Dẫn Chạy](#8-hướng-dẫn-chạy)
9. [Chi Tiết Kỹ Thuật](#9-chi-tiết-kỹ-thuật)
10. [Hạn Chế & Hướng Phát Triển](#10-hạn-chế--hướng-phát-triển)

---

## 1. Tổng Quan Dự Án

Dự án này xây dựng một **bộ công cụ chẩn đoán nông nghiệp thông minh** phục vụ người nông dân trồng lúa và canh tác hoa màu tại Việt Nam, gồm hai nhánh độc lập:

| Nhánh | Mô tả | Loại dữ liệu | Phương pháp |
|---|---|---|---|
| **Phát hiện bệnh lúa** | Phân loại ảnh lá lúa thành 21 nhãn (bệnh, sâu hại, thiếu dinh dưỡng, khỏe mạnh) | Ảnh JPG (22.841 ảnh) | Deep Learning – EfficientNet-B0 |
| **Gợi ý cây trồng** | Dự đoán loại cây phù hợp nhất dựa trên N, P, K, nhiệt độ, độ ẩm, pH, lượng mưa | CSV (2.200 mẫu, 22 loại cây) | Ensemble ML – CatBoost / LightGBM |

Cả hai mô hình đều đạt **độ chính xác ≥ 99%** trên tập kiểm tra.

---

## 2. Cấu Trúc Thư Mục

```
DATASET/
│
├── 📁 Cây lúa khỏe mạnh/              # 1.882 ảnh lúa khỏe
├── 📁 I Côn trùng trên lúa/           # Sâu hại (8 loại)
│   ├── Tungro virus/Ảnh/
│   ├── Sâu gai ( Hispa )/Ảnh/
│   └── ...
├── 📁 II Bệnh gây hại trên lúa/       # Bệnh lúa (7 loại)
│   ├── Bệnh Cháy lá ( Leaf scald )/Ảnh/
│   ├── Bệnh Đốm Vằn .../Ảnh/
│   └── ...
├── 📁 III Thiếu dinh dưỡng/           # Thiếu N, P, K
│   ├── N/Nitrogen(N)/
│   ├── P/Phosphorus(P)/
│   └── K/Potassium(K)/
│
├── 📁 Các thông số tối ưu.../
│   └── NPK.csv                        # Dữ liệu cây trồng
│
├── 📁 processed_dataset/              # Dữ liệu đã chia (train/val/test)
├── 📁 checkpoints/                    # Model checkpoint EfficientNet
├── 📁 models/                         # Model NPK đã huấn luyện
├── 📁 evaluation_results/             # Kết quả đánh giá mô hình ảnh
├── 📁 eda_outputs/                    # Biểu đồ và báo cáo EDA
│
├── data_preprocessing.py             # Tiền xử lý & chia tập dữ liệu ảnh
├── data_loaders.py                   # PyTorch DataLoader + augmentation
├── models.py                         # Kiến trúc EfficientNet-B0
├── train.py                          # Huấn luyện mô hình phân loại ảnh
├── evaluate.py                       # Đánh giá mô hình ảnh
├── npk_preprocessing.py              # Tiền xử lý dữ liệu NPK
├── npk_train_advanced.py             # Tuning siêu tham số NPK
├── npk_train_final_model.py          # Huấn luyện mô hình NPK cuối
├── npk_evaluate.py                   # Đánh giá mô hình NPK
├── predict_crop.py                   # Dự đoán cây trồng từ đầu vào
├── eda_image_analysis.py             # EDA tập ảnh lúa
├── eda_npk_analysis.py               # EDA tập dữ liệu NPK
├── explore_structure.py              # Khám phá cấu trúc thư mục
└── requirements.txt
```

`processed_dataset` có thể được xem tại [đây](https://drive.google.com/drive/u/2/folders/1OY9h1aIC5UnoeNyvmej_tRaSvwngSZCG)

`checkpoints`và 4 folder data gốc, chưa chia train/test/val, có thể được xem tại [đây](https://drive.google.com/drive/u/5/folders/14xefJ_dVonEZHFo_yvmBAtjGlPbqUVwX)

---

## 3. Mô Hình 1 – Phát Hiện Bệnh Lúa (EfficientNet)

### 3.1 Bộ Dữ Liệu Ảnh

Tổng cộng **37.978 ảnh** (sau khi gộp thêm từ `preprocessedDataset`) được chia thành 21 nhãn thuộc 4 nhóm chính:

| Nhóm | Số lớp | Tổng ảnh | Ví dụ |
|---|---|---|---|
| Khỏe mạnh | 1 | 3.764 | Cây lúa khỏe |
| Bệnh | 7 | 17.618 | Cháy lá, Đốm nâu, Bạc lá, ... |
| Sâu hại | 10 | 14.284 | Tungro virus, Rầy nâu, Sâu gai, ... |
| Thiếu dinh dưỡng | 3 | 2.312 | Thiếu N, P, K |

> **Mất cân bằng dữ liệu**: Tỷ lệ lớp lớn nhất / nhỏ nhất = 37,64x (3.764 ảnh vs 100 ảnh). Cần xem xét class weights hoặc focal loss khi huấn luyện.

**Phân phối chi tiết các lớp:**

```
Khỏe mạnh                                      3.764
Tungro virus (Sâu)                             3.480
Bệnh Cháy lá - Leaf scald                      3.340
Bệnh Đốm Vằn - Sheath Blight                   3.156
Bệnh Đốm Nâu - Brown spot                      3.140
Bệnh Bạc lá - Bacterial leaf blight            2.950
Sâu gai - Hispa                                2.922
Bệnh Gạch Nâu - Narrow Brown Spot             2.832
Bệnh Đạo ôn - Blast                           2.000
Sâu năn - Rice Gall Midge                      1.582
Sâu đục thân (Chilo suppressalis)              1.490
Sâu cuốn lá nhỏ - Leaf folder                 1.210
Bọ trĩ - Thrips                               1.160
Sâu cuốn lá lớn - Rice skipper                  950
Sâu đục thân (Yellow stem borer)                 910
Thiếu Đạm (N)                                   880
Thiếu Kali (K)                                  766
Thiếu Lân (P)                                   666
Rầy Nâu - Brown Plant Hopper                    580
Bệnh lúa von - Bakanae                          100
Bệnh Than Vàng - False Smut                     100
```

### 3.2 Tiền Xử Lý & Chia Tập Dữ Liệu

Script `data_preprocessing.py` thực hiện **phân chia tầng** (stratified split):

```
Tổng: 37.978 ảnh  →  Train: 26.584 (70%)  |  Val: 5.697 (15%)  |  Test: 5.697 (15%)
```

Mỗi ảnh được giữ nguyên đường dẫn gốc (hỗ trợ Unicode tiếng Việt) và lưu thông tin vào các file CSV:
- `processed_dataset/train_split.csv`
- `processed_dataset/val_split.csv`
- `processed_dataset/test_split.csv`

### 3.3 Data Augmentation

`data_loaders.py` áp dụng các kỹ thuật tăng cường dữ liệu cho tập train:

```python
# Tập Train (augmentation đầy đủ)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.3)
transforms.RandomRotation(30)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats

# Tập Val/Test (chỉ resize + normalize)
transforms.Resize((224, 224))
transforms.Normalize(...)
```

> Ảnh được đọc bằng **PIL** thay vì OpenCV để tránh lỗi với đường dẫn Unicode (tên thư mục tiếng Việt).

### 3.4 Kiến Trúc Mô Hình

**Backbone**: EfficientNet-B0 (pretrained ImageNet, từ thư viện `timm`)  
**Classification Head** tùy chỉnh:

```
EfficientNet-B0 backbone (1280 features)
    ↓
Dropout(0.3)
    ↓
Linear(1280 → 512) + ReLU
    ↓
Dropout(0.2)
    ↓
Linear(512 → 21)  ← 21 lớp phân loại
```

Mô hình hỗ trợ hoán đổi backbone linh hoạt (ResNet, ViT, ConvNeXt, ...) qua hàm `create_model()`.

### 3.5 Huấn Luyện

| Thông số | Giá trị |
|---|---|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Số epoch | 50 |
| Batch size | 32 |
| GPU | NVIDIA GeForce RTX 3090 |
| Framework | PyTorch 2.5.1 + CUDA 12.1 |

```bash
python train.py
```

---

## 4. Mô Hình 2 – Gợi Ý Cây Trồng (NPK)

### 4.1 Bộ Dữ Liệu NPK

File `NPK.csv` chứa **2.200 mẫu** với phân phối cân bằng hoàn hảo:

```
22 loại cây trồng × 100 mẫu/loại = 2.200 mẫu
```

**7 đặc trưng đầu vào:**

| Đặc trưng | Mô tả | Đơn vị |
|---|---|---|
| N | Hàm lượng Đạm trong đất | kg/ha |
| P | Hàm lượng Lân trong đất | kg/ha |
| K | Hàm lượng Kali trong đất | kg/ha |
| temperature | Nhiệt độ trung bình | °C |
| humidity | Độ ẩm tương đối | % |
| ph | Độ pH đất | 0 – 14 |
| rainfall | Lượng mưa trung bình | mm |

**22 loại cây trồng được hỗ trợ:** lúa, ngô, đậu xanh, đậu đen, đậu gà, đậu thận, đậu lăng, đậu bồ câu, đậu trắng, lựu, chuối, xoài, nho, dưa hấu, dưa lưới, táo, cam, đu đủ, dừa, bông, đay, cà phê.

### 4.2 Feature Engineering

`npk_preprocessing.py` tự động tạo **22 đặc trưng bổ sung** từ 7 đặc trưng gốc, nâng tổng số đặc trưng lên **29**:

```python
# Tỉ lệ dinh dưỡng
N_P_ratio, N_K_ratio, P_K_ratio

# Tổng hợp dinh dưỡng
total_NPK = N + P + K

# Chỉ số khí hậu
temp_humidity_product = temperature × humidity
temp_humidity_diff    = temperature - humidity
moisture_index        = rainfall × humidity / 100
ph_neutral_dev        = |ph - 7.0|

# Đặc trưng bậc hai
N², P², K²

# Tương tác
N_temp, P_rainfall, K_humidity

# One-hot encoding (phân vùng khí hậu)
temp_zone     : cold / moderate / warm / hot
humidity_zone : dry / moderate / humid
rainfall_zone : low / medium / high / very_high
```

**Chuẩn hóa**: `RobustScaler` (tốt hơn `StandardScaler` khi có outlier, theo kết luận EDA).

**Chia tập**: Train 70% / Val 10% / Test 20% (stratified).

### 4.3 Tuning Siêu Tham Số

`npk_train_advanced.py` thực hiện **grid search thủ công thông minh** trên 4 mô hình:

| Mô hình | Tổ hợp thử | Thời gian | Val Acc tốt nhất |
|---|---|---|---|
| Random Forest | 27 | ~5s | 99.09% |
| XGBoost | 27 | ~30s | 99.09% |
| LightGBM | 27 | ~15s | 99.09% |
| **CatBoost** | **27** | **~15 phút** | **99.55%** |

**Siêu tham số tốt nhất (CatBoost):**

```python
{
    'iterations': 200,
    'depth': 8,
    'learning_rate': 0.01
}
```

**Siêu tham số tốt nhất (LightGBM – nhanh hơn, test acc tương đương):**

```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'num_leaves': 15,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### 4.4 Kết Quả Huấn Luyện Mô Hình Cuối

| Mô hình | Train Acc | Val Acc | Test Acc |
|---|---|---|---|
| Random Forest | 100.00% | 99.09% | 99.55% |
| **LightGBM** | **100.00%** | **99.09%** | **99.55%** |
| XGBoost | 99.94% | 99.09% | 99.32% |
| CatBoost | 98.90% | 99.55% | 97.27% |

> LightGBM được chọn làm mô hình triển khai cuối vì đạt test accuracy cao nhất (99.55%) với tốc độ suy luận nhanh hơn CatBoost nhiều lần.

### 4.5 Sử Dụng Mô Hình Gợi Ý

```python
from predict_crop import predict_crop

result = predict_crop(
    N=90, P=42, K=43,
    temperature=21, humidity=82,
    ph=6.5, rainfall=203
)

# Output:
# Recommended Crop: RICE
# Confidence: 100.00%
# Top 3: rice (100%), jute (0%), coffee (0%)
```

---

## 5. Phân Tích Dữ Liệu Khám Phá (EDA)

### 5.1 EDA Tập Ảnh Lúa (`eda_image_analysis.py`)

Phân tích thực hiện trên **1.050 ảnh mẫu** (50 ảnh/lớp), cho các kết quả:

| Thống kê | Giá trị |
|---|---|
| Chiều cao trung bình | 1.223 px |
| Chiều rộng trung bình | 1.053 px |
| Kích thước file trung bình | 282 KB |
| Độ sáng trung bình | 149.5 / 255 |
| Tỷ lệ khung hình phổ biến | 1:1 (vuông) |
| Ảnh nhỏ nhất | 201 × 217 px |
| Ảnh lớn nhất | 4.364 × 4.301 px |

**Kết quả đầu ra:**
- `eda_outputs/image_analysis/class_distribution.png` – Phân phối các lớp
- `eda_outputs/image_analysis/image_properties.png` – Thống kê thuộc tính ảnh
- `eda_outputs/image_analysis/sample_images.png` – Lưới ảnh mẫu
- `eda_outputs/image_analysis/image_eda_summary.txt` – Báo cáo văn bản

### 5.2 EDA Tập NPK (`eda_npk_analysis.py`)

**Kết luận chính từ EDA:**
- Dữ liệu **cân bằng hoàn hảo** (100 mẫu/lớp) – không cần oversampling
- **Không có giá trị thiếu** – dữ liệu sạch
- Nhiệt độ và độ ẩm có **tương quan mạnh** → được khai thác qua `temp_humidity_product`
- Mỗi loại cây có **profil NPK đặc trưng** riêng biệt (e.g., táo cần P=134, K=200 rất cao)
- Phân tán dữ liệu cao ở một số đặc trưng → `RobustScaler` phù hợp hơn `StandardScaler`

**Kết quả đầu ra:**
- `eda_outputs/npk_analysis/crop_distribution.png`
- `eda_outputs/npk_analysis/feature_distributions.png`
- `eda_outputs/npk_analysis/correlation_heatmap.png`
- `eda_outputs/npk_analysis/crop_requirements.png`
- `eda_outputs/npk_analysis/feature_pairplot.png`
- `eda_outputs/npk_analysis/eda_summary_report.txt`

---

## 6. Kết Quả Đánh Giá

### 6.1 Mô Hình Phát Hiện Bệnh Lúa

Đánh giá trên **5.697 ảnh** tập test:

```
Overall Accuracy : 99.91%
Macro Precision  : 99.84%
Macro Recall     : 99.85%
Macro F1         : 99.85%
Weighted F1      : 99.91%
```

**Hiệu suất theo từng lớp (tóm tắt):**

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bệnh Cháy lá | 1.000 | 1.000 | 1.000 | 501 |
| Healthy | 1.000 | 1.000 | 1.000 | 565 |
| Tungro virus | 1.000 | 1.000 | 1.000 | 522 |
| Bệnh Đốm Nâu | 1.000 | 1.000 | 1.000 | 471 |
| Sâu đục thân (Yellow SB) | 0.986 | 1.000 | 0.993 | 136 |
| P (Thiếu Lân) | 1.000 | 0.980 | 0.990 | 100 |
| Bệnh Than Vàng | 1.000 | 1.000 | 1.000 | 15 |
| ... | ... | ... | ... | ... |

**Phân tích nhầm lẫn (5 ảnh bị sai trên 5.697):**

| Nhầm từ → sang | Số lần |
|---|---|
| Sâu đục thân Chilo → Sâu đục thân Yellow SB | 2 |
| Thiếu P → Thiếu K | 1 |
| Thiếu P → Thiếu N | 1 |
| Bệnh Bạc lá → Bệnh Đốm Vằn | 1 |

> Tỷ lệ lỗi chỉ **0.09%**. Các nhầm lẫn đều xảy ra giữa các lớp **rất giống nhau về mặt thị giác** (hai loại sâu đục thân, ba loại thiếu dinh dưỡng), phản ánh giới hạn tự nhiên ngay cả với chuyên gia.

### 6.2 Mô Hình Gợi Ý Cây Trồng

Đánh giá trên **440 mẫu** tập test (LightGBM):

```
Test Accuracy: 99.55%
```

**Hiệu suất theo từng lớp (đầy đủ):**

| Cây | Precision | Recall | F1 |
|---|---|---|---|
| apple | 1.000 | 1.000 | 1.000 |
| banana | 1.000 | 1.000 | 1.000 |
| blackgram | 1.000 | 1.000 | 1.000 |
| chickpea | 1.000 | 1.000 | 1.000 |
| coconut | 1.000 | 1.000 | 1.000 |
| coffee | 1.000 | 1.000 | 1.000 |
| cotton | 1.000 | 1.000 | 1.000 |
| grapes | 1.000 | 1.000 | 1.000 |
| jute | 1.000 | 1.000 | 1.000 |
| kidneybeans | 1.000 | 1.000 | 1.000 |
| **lentil** | 1.000 | **0.950** | 0.974 |
| maize | 1.000 | 1.000 | 1.000 |
| mango | 1.000 | 1.000 | 1.000 |
| **mothbeans** | **0.909** | 1.000 | 0.952 |
| mungbean | 1.000 | 1.000 | 1.000 |
| muskmelon | 1.000 | 1.000 | 1.000 |
| orange | 1.000 | 1.000 | 1.000 |
| papaya | 1.000 | 1.000 | 1.000 |
| **pigeonpeas** | 1.000 | **0.950** | 0.974 |
| pomegranate | 1.000 | 1.000 | 1.000 |
| rice | 1.000 | 1.000 | 1.000 |
| watermelon | 1.000 | 1.000 | 1.000 |

> Ba lớp có F1 < 1.0 đều là họ đậu có đặc trưng N/P/K rất tương đồng (lentil, mothbeans, pigeonpeas). Đây là thách thức tự nhiên của bài toán phân loại đa lớp với dữ liệu chỉ 100 mẫu/lớp.

---

## 7. Cài Đặt & Yêu Cầu Hệ Thống

### Phần Cứng Khuyến Nghị

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| GPU | 8 GB VRAM | RTX 3090 (24 GB) |
| RAM | 16 GB | 32 GB |
| Dung lượng | 30 GB | 50 GB |
| CUDA | 11.8+ | 12.1 |

### Cài Đặt Môi Trường

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

# Cài đặt thư viện cơ bản
pip install -r requirements.txt

# Cài đặt thư viện bổ sung cho NPK
pip install catboost lightgbm xgboost imbalanced-learn joblib
```

### Nội Dung `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Kiểm Tra GPU

```bash
python u.py
# Expected output:
# 2.5.1+cu121
# 12.1
# True
# NVIDIA GeForce RTX 3090
```

---

## 8. Hướng Dẫn Chạy

### Pipeline 1 – Phát Hiện Bệnh Lúa

```bash
# Bước 1: Khám phá cấu trúc dữ liệu
python explore_structure.py

# Bước 2: EDA tập ảnh
python eda_image_analysis.py

# Bước 3: Tiền xử lý & chia tập dữ liệu
python data_preprocessing.py

# Bước 4: Kiểm tra DataLoader
python data_loaders.py

# Bước 5: Kiểm tra kiến trúc mô hình
python models.py

# Bước 6: Huấn luyện (cần GPU)
python train.py

# Bước 7: Đánh giá trên tập test
python evaluate.py
```

### Pipeline 2 – Gợi Ý Cây Trồng

```bash
# Bước 1: EDA dữ liệu NPK
python eda_npk_analysis.py

# Bước 2: Tiền xử lý
python npk_preprocessing.py

# Bước 3: Tuning siêu tham số (cần ~15 phút)
python npk_train_advanced.py

# Bước 4: Huấn luyện mô hình cuối
python npk_train_final_model.py

# Bước 5: Đánh giá
python npk_evaluate.py

# Bước 6: Dự đoán thử
python predict_crop.py
```

### Chạy Toàn Bộ EDA

```bash
python run_complete_eda.py
```

---

## 9. Chi Tiết Kỹ Thuật

### Xử Lý Đường Dẫn Unicode

Một thách thức đặc thù của dự án là toàn bộ tên thư mục và file đều dùng **tiếng Việt có dấu** (ví dụ: `Bệnh Đốm Vằn ( Khô vằn ) ( Sheath Blight )/Ảnh/`). Dự án xử lý vấn đề này bằng cách:

- Dùng `PIL.Image.open()` thay vì `cv2.imread()` — PIL hỗ trợ Unicode đầy đủ
- Dùng `pathlib.Path` thay vì `os.path.join()` cho tất cả thao tác đường dẫn
- Mở file text với `encoding='utf-8'`

### Pipeline Tiền Xử Lý NPK – Tránh Data Leakage

```
df_raw → feature_engineering() → train_test_split()
                                         ↓
                               RobustScaler.fit(X_train only)
                                         ↓
                  X_train_scaled   X_val_scaled   X_test_scaled
                  (fit_transform)    (transform)    (transform)
```

> Scaler chỉ được `fit` trên tập train, tránh hoàn toàn data leakage sang tập val/test.

### Lưu Trữ Artifact

| Artifact | Đường dẫn | Mô tả |
|---|---|---|
| EfficientNet weights | `checkpoints/best_model.pth` | Checkpoint tốt nhất (val acc) |
| LightGBM model | `models/best_model_lightgbm.pkl` | Model NPK triển khai |
| RobustScaler | `models/preprocessor/scaler.pkl` | Scaler đã fit |
| LabelEncoder | `models/preprocessor/label_encoder.pkl` | Bộ mã hóa nhãn |
| Feature names | `models/preprocessor/feature_names.pkl` | Danh sách 29 đặc trưng |

---

## 10. Hạn Chế & Hướng Phát Triển

### Hạn Chế Hiện Tại

1. **Mất cân bằng dữ liệu ảnh**: Tỷ lệ 37:1 giữa lớp lớn nhất và nhỏ nhất. Các lớp nhỏ như `Bệnh Than Vàng` và `Bệnh lúa von` chỉ có 100 ảnh.

2. **Mô hình NPK nhỏ**: Chỉ 2.200 mẫu tổng cộng (100/lớp). Với dữ liệu thực tế đa dạng hơn, cần bộ dữ liệu lớn hơn nhiều.

3. **Chưa có inference thời gian thực**: Mô hình ảnh chưa được đóng gói thành API hoặc ứng dụng di động.

4. **Phạm vi địa lý**: Dữ liệu ảnh lúa được thu thập tại Việt Nam; khả năng tổng quát hóa sang điều kiện địa lý khác chưa được kiểm chứng.

### Hướng Phát Triển

- [ ] **Tăng cường dữ liệu**: Áp dụng Mixup, CutMix, hoặc sinh ảnh tổng hợp bằng GAN cho các lớp thiểu số
- [ ] **Thử nghiệm kiến trúc mạnh hơn**: EfficientNet-B3/B4, ViT-Base, ConvNeXt-Base
- [ ] **Triển khai API**: Đóng gói bằng FastAPI + Docker, expose REST endpoint
- [ ] **Ứng dụng di động**: Tích hợp TensorFlow Lite hoặc ONNX để chạy on-device
- [ ] **Phân loại phân cấp**: Nhận diện nhóm trước (bệnh / sâu / thiếu dd) rồi mới phân loại chi tiết
- [ ] **Kết hợp hai mô hình**: Dashboard nông nghiệp tích hợp chẩn đoán lúa + gợi ý cây trồng trong một giao diện
- [ ] **Mở rộng dữ liệu NPK**: Thu thập thêm dữ liệu thực địa tại Việt Nam để cải thiện độ chính xác cho cây trồng nhiệt đới
