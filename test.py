import os
import pandas as pd
from dotenv import load_dotenv

# === 1. Load biến môi trường từ file .env ===
load_dotenv()

# === 2. Đọc FEATURE_LIST từ file .env ===
feature_list = [f.strip().strip("'").strip('"') for f in os.getenv("FEATURE_LIST", "").strip("[]").split(",") if f.strip()]

# === 3. Đọc dữ liệu gốc ===
file_path = "TrafficLabelling/data_binary.csv"
df = pd.read_csv(file_path)

# === 4. Lọc ra các cột có trong FEATURE_LIST ===
# (chỉ giữ các cột có tồn tại trong file CSV để tránh lỗi)
available_features = [f for f in feature_list if f in df.columns]
df_filtered = df[available_features]

# === 5. Lấy ngẫu nhiên 1000 dòng ===
df_sample = df_filtered.sample(n=10, random_state=42)

# === 6. Xuất kết quả ra file mới (tùy chọn) ===
output_path = "TrafficLabelling/data_binary_sample.csv"
df_sample.to_csv(output_path, index=False)

print(f"✅ Đã lấy ngẫu nhiên 1000 dòng và lưu vào {output_path}")
print(f"Số cột được chọn: {len(available_features)} / {len(feature_list)}")
print(f"Các cột có trong dữ liệu: {available_features}")
