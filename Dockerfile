# Sử dụng Python 3.13
FROM python:3.13-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements vào container
COPY requirements.txt .

# Cài đặt các dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Mở cổng FastAPI
EXPOSE 8000

# Chạy uvicorn khi container start
ENTRYPOINT ["./entrypoint.sh"]
