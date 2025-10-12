import joblib
import logging
import numpy as np
from sklearn.tree import export_text, export_graphviz
from AIAgent_pipeline.base_node import Node
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MultiClassifierNode(Node):
    def __init__(self, model_path,mode="predict"):
        self.model_path = model_path
        self.model = self.load_model()
        self.mode = mode

    def load_model(self):
        """
        Tải mô hình từ đường dẫn đã chỉ định.
        Returns:
            object: Mô hình đã được load.
        Raises:
            Exception: Nếu có lỗi khi tải model.
        """
        try:
            model = joblib.load(self.model_path)
            logging.info(f"[INFO][NODE3] Tải thành công model từ {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"[ERROR][NODE3] Lỗi tải model: {e}", exc_info=True)
            raise

    def predict(self, df):
        """
        Dự đoán nhãn lớp cho dữ liệu đầu vào.
        Args:
            df (pandas.DataFrame hoặc numpy.ndarray): Dữ liệu đầu vào.
        Returns:
            numpy.ndarray: Mảng nhãn dự đoán (đa lớp).
        """
        try:
            if not hasattr(df, "shape"):
                raise TypeError("[NODE3] Đầu vào predict phải là DataFrame hoặc mảng numpy hợp lệ.")
            predictions = self.model.predict(df)
            logging.info(f"[INFO][NODE3] Dự đoán thành công {len(predictions)} mẫu.")
            return predictions
        except Exception as e:
            logging.error(f"[ERROR][NODE3] Lỗi trong predict: {e}", exc_info=True)
            raise

    def predict_proba(self, df):
        """
        Dự đoán xác suất cho mỗi lớp (nếu mô hình hỗ trợ).
        Args:
            df (pandas.DataFrame hoặc numpy.ndarray): Dữ liệu đầu vào.
        Returns:
            numpy.ndarray: Mảng xác suất thuộc về từng lớp (shape: [n_samples, n_classes]).
        """
        try:
            if not hasattr(df, "shape"):
                raise TypeError("[NODE3] Đầu vào predict_proba phải là DataFrame hoặc mảng numpy hợp lệ.")
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(df)
                logging.info(f"[INFO][NODE3] Dự đoán xác suất thành công.")
                return probas
            else:
                raise AttributeError("[NODE3] Model không hỗ trợ phương thức predict_proba.")
        except Exception as e:
            logging.error(f"[ERROR][NODE3] Lỗi trong predict_proba: {e}", exc_info=True)
            raise
    def process(self, data, mode=None):
        """
        Xử lý dữ liệu đầu vào bằng mô hình đa lớp.
        Args:
            data (pd.DataFrame hoặc np.ndarray): Dữ liệu đầu vào.
            mode (str): Chế độ hoạt động:
                        - "predict": chỉ trả về nhãn
                        - "proba": trả cả xác suất
        Returns:
            dict: Kết quả gồm dữ liệu đầu vào + nhãn dự đoán + (nếu có) xác suất.
        """
        try:
            logging.info(f"[INFO][NODE3] Bắt đầu xử lý dữ liệu ({mode})...")
            mode = mode or self.mode
            data = data["data"] if isinstance(data, dict) and "data" in data else data
            if mode not in ["predict", "proba"]:
                raise ValueError("[NODE3] mode không hợp lệ, phải là 'predict' hoặc 'proba'.")

            # Dự đoán nhãn
            if mode == "predict":
                predictions = self.predict(data)
                result = {"data": data, "label": predictions}

            # Nếu yêu cầu xác suất thì thêm vào
            elif mode == "proba":
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(data)  # shape (n_samples, n_classes)
                    pred_indices = np.argmax(probabilities, axis=1)  # index nhãn cao nhất
                    labels = [self.model.classes_[i] for i in pred_indices]
                    probs_max = [float(probabilities[i, idx]) for i, idx in enumerate(pred_indices)]  # xác suất max
                    result = {
                        "data": data,
                        "label": labels,           # nhãn max
                        "probabilities": probs_max # xác suất max
                    }
                else:
                    logging.warning("[WARN][NODE3] Model không hỗ trợ predict_proba.")

            logging.info("[INFO][NODE3] Hoàn thành xử lý dữ liệu.")
            return result

        except Exception as e:
            logging.error(f"[ERROR][NODE3] Lỗi trong quá trình xử lý dữ liệu: {e}", exc_info=True)
            raise
    def get_feature_importances(self, feature_names=None):
        """
        Lấy độ quan trọng của các đặc trưng trong mô hình (nếu có).
        Args:
            feature_names (list, optional): Tên các đặc trưng (nếu có). 
        Returns:
            dict hoặc np.ndarray: {feature_name: importance} hoặc mảng numpy.
        """
        try:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                if feature_names is not None:
                    importance_dict = dict(zip(feature_names, importances))
                    logging.info("[DEBUG][NODE3] Trích xuất thành công feature importances.")
                    return importance_dict
                return importances
            else:
                logging.warning("[WARN][NODE3] Model không có thuộc tính feature_importances_.")
                return None
        except Exception as e:
            logging.error(f"[ERROR][NODE3] Lỗi khi lấy feature importances: {e}", exc_info=True)
            raise

    def print_tree(self, feature_names=None, tree_index=0, max_depth=3, save_path=None):
        """
        In hoặc lưu cấu trúc cây ra file từ mô hình DecisionTree hoặc RandomForest.
        Args:
            feature_names (list, optional): Danh sách tên các đặc trưng.
            tree_index (int, optional): Nếu là RandomForest, chỉ định cây thứ mấy (mặc định: 0).
            max_depth (int, optional): Giới hạn độ sâu khi in.
            save_path (str, optional): Đường dẫn để lưu file .dot.
        Returns:
            str: Cấu trúc cây ở dạng văn bản.
        """
        try:
            if hasattr(self.model, "estimators_"):  # RandomForest hoặc ExtraTrees
                tree = self.model.estimators_[tree_index]
                logging.info(f"[INFO][NODE3] Xuất cây thứ {tree_index} trong rừng.")
            else:
                tree = self.model

            # Xuất dạng văn bản
            tree_text = export_text(tree, feature_names=feature_names, max_depth=max_depth)
            logging.info(f"[DEBUG][NODE3] In cấu trúc cây thành công.")

            # Nếu cần lưu file .dot
            if save_path:
                export_graphviz(
                    tree,
                    out_file=save_path,
                    feature_names=feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                logging.info(f"[INFO][NODE3] Tree exported to {save_path}")
            return tree_text
        except Exception as e:
            logging.error(f"[ERROR][NODE3] Lỗi khi in cây: {e}", exc_info=True)
            raise
