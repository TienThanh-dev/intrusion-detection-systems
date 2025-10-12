import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
from AIAgent_pipeline.base_node import Node
load_dotenv()  # Load biến môi trường từ file .env
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class InputValidatorNode(Node):
    def __init__(self, env_path=".env",feature_list=None):
        load_dotenv(env_path)
        self.feature_list = feature_list if feature_list is not None else [f.strip() for f in os.getenv("FEATURE_LIST", "").split(",") if f.strip()]
        if not self.feature_list:
            raise ValueError("[NODE1]Không tìm thấy FEATURE_LIST")

    def process(self, input_data):
        """
        Xử lý và chuẩn hóa dữ liệu đầu vào để đảm bảo tương thích với mô hình huấn luyện.

        Chức năng:
        - Tự động nhận dạng loại đầu vào (DataFrame, file CSV, hoặc dictionary).
        - Lọc các cột chỉ nằm trong `self.feature_list`.
        - Ép kiểu dữ liệu về dạng số, thay thế các giá trị NaN, ±inf bằng 0.
        - Đưa toàn bộ giá trị âm về 0.
        - Thêm các cột bị thiếu (nếu có) với giá trị mặc định là 0.
        - Kiểm tra xem dữ liệu sau xử lý có trống không.

        Args:
            input_data (pd.DataFrame | str | dict): 
                - `pd.DataFrame`: dữ liệu đã có sẵn trong bộ nhớ.  
                - `str`: đường dẫn đến file `.csv` cần đọc.  
                - `dict`: một hàng dữ liệu duy nhất ở dạng từ điển.

        Raises:
            TypeError: Nếu kiểu dữ liệu đầu vào không hợp lệ.  
            ValueError: Nếu dữ liệu sau khi lọc trống (không có dòng hoặc không còn cột hợp lệ).

        Returns:
            pd.DataFrame: 
                DataFrame đã được làm sạch và sắp xếp theo đúng thứ tự `feature_list`.  
                Tất cả giá trị NaN, inf, và âm đều đã được thay bằng 0.
        """
        try:
            if isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            elif isinstance(input_data, str) and input_data.endswith(".csv"):
                df = pd.read_csv(input_data)
            elif isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                raise TypeError(f"[NODE1]Dữ liệu không hợp lệ: {type(input_data)}")
            df = df[[c for c in self.feature_list if c in df.columns]]
            df = df.apply(pd.to_numeric, errors="coerce")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            df[df < 0] = 0
            missing = [c for c in self.feature_list if c not in df.columns]
            if missing:
                logging.info(f"[INFO][NODE1]Thiếu cột: {missing} -> thêm với giá trị 0")
                for col in missing:
                    df[col] = 0
            df = df[self.feature_list]
            if df.empty:
                raise ValueError("[NODE1]Dữ liệu sau khi lọc bị trống!")
            logging.info(f"[INFO][NODE1]CICFlowMeter data hợp lệ. {len(df)} dòng, {len(df.columns)} cột.")
            return df
        except Exception as e:
            logging.error(f"[ERROR][NODE1]Lỗi trong InputValidatorNode: {e}",exc_info=True)
            raise 
    def validate_features(self, df):
        """
        Kiểm tra xem DataFrame có chứa đầy đủ các đặc trưng (`features`) cần thiết không.

        Args:
            df (pd.DataFrame): Dữ liệu cần kiểm tra.

        Returns:
            bool: 
                - `True` nếu tất cả các cột trong `self.feature_list` đều có mặt trong `df`.  
                - `False` nếu còn thiếu cột (và in cảnh báo các cột bị thiếu).
        """
        missing = [c for c in self.feature_list if c not in df.columns]
        if missing:
            logging.warning(f"[INFO][NODE1]Cảnh báo: Thiếu các cột sau: {missing}")
            return False
        return True
