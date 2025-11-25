from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import pandas as pd
from AIAgent_pipeline import AsyncNetworkPredictor
import json

# Khởi tạo 2 predictor lúc app start
predictor_labels = AsyncNetworkPredictor(mode="predict")
predictor_proba = AsyncNetworkPredictor(mode="proba")

# FastAPI app
app = FastAPI(title="Intrusion Detection API")

# Định nghĩa model dữ liệu input
class InputData(BaseModel):
    data: list  # list các dict, mỗi dict là một hàng dữ liệu

# Hàm xử lý chung dữ liệu CSV / JSON
async def _prepare_dataframe(file, input_data):
    if file:
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            return None, f"Lỗi đọc file CSV: {e}"
    elif input_data:
        data_dict = json.loads(input_data)
        df = pd.DataFrame(data_dict["data"])
    else:
        return None, "Không có dữ liệu đầu vào"
    return df, None

@app.post("/predict_labels")
async def predict_labels_endpoint(file: UploadFile = File(None), input_data: str = Form(None)):
    """
    ### Mục đích:
    Dự đoán nhãn mạng (Network Intrusion Detection) cho từng hàng dữ liệu đầu vào, chỉ trả nhãn (label).

    ### Tham số:
    - `file`: File CSV chứa dữ liệu (ưu tiên nếu gửi cùng với JSON)
    - `input_data`: Dữ liệu JSON dạng chuỗi, ví dụ:
    ```json
    {
        "data": [
            {"feature1": 0.1, "feature2": 1.5, ...},
            {"feature1": 0.3, "feature2": 2.1, ...}
        ]
    }
    ```

    ### Trả về:
    ```json
    {
        "labels": ["BENIGN", "DOS_DDOS", ...]
    }
    ```
    """
    df, error = await _prepare_dataframe(file, input_data)
    if error:
        return {"error": error}

    labels, _ = await predictor_labels.predict(df)
    return {"labels": labels}

@app.post("/predict_proba")
async def predict_proba_endpoint(file: UploadFile = File(None), input_data: str = Form(None)):
    """
    ### Mục đích:
    Dự đoán nhãn mạng và xác suất nhãn cao nhất (max probability) cho từng hàng dữ liệu đầu vào.

    ### Tham số:
    - `file`: File CSV chứa dữ liệu (ưu tiên nếu gửi cùng với JSON)
    - `input_data`: Dữ liệu JSON dạng chuỗi, ví dụ:
    ```json
    {
        "data": [
            {"feature1": 0.1, "feature2": 1.5, ...},
            {"feature1": 0.3, "feature2": 2.1, ...}
        ]
    }
    ```

    ### Quy trình:
    1. Chuẩn hóa dữ liệu từ CSV hoặc JSON sang DataFrame.
    2. Gọi predictor ở chế độ `"proba"` để lấy nhãn và xác suất.
    3. Chỉ lấy xác suất của nhãn dự đoán cao nhất cho từng hàng.

    ### Trả về:
    ```json
    {
        "labels": ["BENIGN", "DOS_DDOS", ...],
        "probabilities": [0.95, 0.87, ...]
    }
    ```

    ### Lưu ý:
    - Nếu xác suất không có (`None`), giá trị trong `probabilities` sẽ là `null`.
    - Predictor được khởi tạo một lần, **không tạo lại workflow mỗi lần gọi**, giúp tối ưu hiệu năng.
    """
    df, error = await _prepare_dataframe(file, input_data)
    if error:
        return {"error": error}

    labels, probs = await predictor_proba.predict(df)
    # Chỉ lấy xác suất max cho mỗi hàng
    probs_list = [float(p[0]) if p and len(p) > 0 else None for p in probs]

    return {"labels": labels, "probabilities": probs_list}
