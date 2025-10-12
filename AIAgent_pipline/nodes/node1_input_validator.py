import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # Load biến môi trường từ file .env

class InputValidatorNode:
    def __init__(self):
        load_dotenv(self, env_path=".env")
        self.feature_list = [
            f.strip() for f in os.getenv("FEATURE_LIST", "").split(",") if f.strip()
        ]
        if not self.feature_list:
            raise ValueError("Không tìm thấy FEATURE_LIST")

    def run(self, input_data):
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, str) and input_data.endswith(".csv"):
            df = pd.read_csv(input_data)
        elif isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            raise TypeError(f"Dữ liệu không hợp lệ: {type(input_data)}")
        for col in self.drop_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        missing = [c for c in self.feature_list if c not in df.columns]
        if missing:
            print(f"Thiếu cột: {missing} → thêm với giá trị 0")
            for col in missing:
                df[col] = 0

        extra = [c for c in df.columns if c not in self.feature_list]
        if extra:
            print(f"Loại bỏ cột thừa: {extra}")
            df = df[self.feature_list]

        print(f"CICFlowMeter data hợp lệ. {len(df)} dòng, {len(df.columns)} cột.")
        return df
