# 🎯 Obesity Level Prediction – BTL Machine Learning

Dự án này xây dựng và triển khai mô hình **Machine Learning** nhằm **dự đoán mức độ béo phì** của một cá nhân dựa trên các đặc điểm cá nhân, thói quen ăn uống và sinh hoạt.

Dự án được thực hiện trong khuôn khổ **Bài tập lớn môn Học máy / Khai phá dữ liệu**.

---

## 1. Mục tiêu bài toán

- Phân loại mức độ béo phì của một cá nhân thành **7 mức độ**:
  - Insufficient_Weight
  - Normal_Weight
  - Overweight_Level_I
  - Overweight_Level_II
  - Obesity_Type_I
  - Obesity_Type_II
  - Obesity_Type_III
- So sánh hiệu quả của nhiều mô hình học máy
- Đánh giá tác động của đặc trưng **BMI**
- Triển khai mô hình dưới dạng **CLI** và **Web Demo (API + HTML)**

---

## 2. Dataset

### 2.1. Nguồn dữ liệu

Dataset được sử dụng trong đề tài là **Obesity Levels Dataset**, được công bố công khai trên Kaggle:

🔗 https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels

- Tên file gốc: `ObesityDataSet_raw_and_data_sinthetic.csv`
- Dataset được xây dựng từ dữ liệu khảo sát kết hợp dữ liệu tổng hợp (synthetic data)
- Dữ liệu không chứa giá trị thiếu (NaN)

---

### 2.2. Mô tả tổng quan

Dataset bao gồm thông tin của các cá nhân với nhiều nhóm thuộc tính khác nhau, phản ánh các yếu tố ảnh hưởng đến tình trạng béo phì, bao gồm:

- **Đặc điểm nhân khẩu học**: tuổi, giới tính
- **Chỉ số cơ thể**: chiều cao, cân nặng
- **Thói quen ăn uống**: tần suất ăn rau, số bữa ăn chính, mức tiêu thụ thực phẩm nhiều calo
- **Hoạt động thể chất**: mức độ vận động, thời gian sử dụng thiết bị điện tử
- **Lối sống**: hút thuốc, uống nước, phương tiện di chuyển

Mục tiêu của bài toán là dự đoán **mức độ béo phì (`NObeyesdad`)** của mỗi cá nhân.

---

### 2.3. Các nhóm thuộc tính

#### Thuộc tính số (Numerical features)
- Age: Tuổi
- Height: Chiều cao (m)
- Weight: Cân nặng (kg)
- FCVC: Tần suất ăn rau
- NCP: Số bữa ăn chính mỗi ngày
- CH2O: Lượng nước uống mỗi ngày
- FAF: Mức độ hoạt động thể chất
- TUE: Thời gian sử dụng thiết bị điện tử

📌 Đặc trưng **BMI (Body Mass Index)** được **tạo thêm trong bước tiền xử lý** từ Height và Weight để tăng khả năng phân biệt các mức độ béo phì.

#### Thuộc tính phân loại (Categorical features)
- Gender
- family_history_with_overweight
- FAVC
- CAEC
- SMOKE
- SCC
- CALC
- MTRANS

#### Biến mục tiêu (Target)
- Tên biến: `NObeyesdad`
- Số lớp: 7 mức độ béo phì

---

### 2.4. Dữ liệu mẫu (Sample data)

Repository này **không upload toàn bộ dataset**.  
Thay vào đó, cung cấp file:

- `data/sample_data.csv` (50 dòng dữ liệu mẫu)

File dữ liệu mẫu được trích từ dataset gốc bằng phương pháp **lấy mẫu ngẫu nhiên có phân tầng (stratified sampling)** theo biến mục tiêu `NObeyesdad`, nhằm:
- Minh họa cấu trúc dữ liệu
- Kiểm tra nhanh pipeline và demo
- Giữ repository gọn nhẹ khi upload GitHub

---

## 3. Pipeline xử lý dữ liệu

Pipeline được xây dựng bằng `scikit-learn` gồm các bước:

1. **Feature Engineering**
   - Tạo đặc trưng BMI từ Height và Weight

2. **Tiền xử lý dữ liệu**
   - Xử lý dữ liệu số bằng `StandardScaler`
   - Xử lý dữ liệu phân loại bằng `OneHotEncoder`
   - Xử lý giá trị thiếu bằng `SimpleImputer`

3. **Huấn luyện mô hình**
   - Áp dụng pipeline thống nhất cho cả huấn luyện và dự đoán
   - Tránh hiện tượng rò rỉ dữ liệu (data leakage)

---

## 4. Mô hình sử dụng

Các mô hình được huấn luyện và so sánh bao gồm:

- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Random Forest (không sử dụng BMI)
- Random Forest (có sử dụng BMI)

📌 **Random Forest có sử dụng BMI** cho kết quả tốt nhất và được lựa chọn làm mô hình triển khai demo.

---

## 5. Đánh giá mô hình

- Dữ liệu được chia theo tỉ lệ **70% – 15% – 15%**:
  - Train
  - Validation
  - Test
- Các chỉ số đánh giá:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 6. Cấu trúc thư mục dự án

├── app/ # Source code chính (ML + API)
│ ├── train.py # Huấn luyện mô hình
│ ├── preprocess.py # Pipeline xử lý dữ liệu
│ ├── predict.py # Dự đoán bằng CLI
│ ├── main.py # Backend API
│ └── utils.py # Hàm load/save model
│
├── data/ # Dữ liệu
│ ├── sample_data.csv
│ └── README.md
│
├── demo/ # Giao diện demo
│ └── index.html
│
├── models/ # Model đã huấn luyện (.pkl)
│
├── reports/ # Báo cáo
│
├── slides/ # Slide thuyết trình
│
├── requirements.txt
└── README.

---

## 7. Hướng dẫn cài đặt

### 7.1. Cài môi trường

```bash
pip install -r requirements.txt
---

8. Huấn luyện mô hình
python app/train.py
Sau khi huấn luyện, model sẽ được lưu trong thư mục models/.

---
9. Dự đoán bằng CLI
python app/predict.py

---
10. Chạy Web Demo
10.1. Chạy backend API
python app/main.py


Backend mặc định chạy tại:

http://127.0.0.1:5000

10.2. Mở giao diện demo

Mở file demo/index.html bằng trình duyệt

Nhập thông tin cá nhân và nhấn Dự đoán

---
11. Tác giả

Họ và tên: ....Phạm Văn Kiên...........

Mã sinh viên: .........12423047........

Lớp: ..........124231..................

Môn học: Machine Learning
