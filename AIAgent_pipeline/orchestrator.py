import asyncio
from dotenv import load_dotenv
from AIAgent_pipeline import WorkflowManager, InputValidatorNode, BinaryClassifierNode, MultiClassifierNode, MODEL_MAP
import pandas as pd
import numpy as np
import os
import logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FEATURE_LIST = [f.strip() for f in os.getenv("FEATURE_LIST", "").split(",") if f.strip()]

class AsyncNetworkPredictor:
    def __init__(self, mode="predict", model_path_binary=MODEL_MAP["binary_rf"], model_path_multi=MODEL_MAP["multi_rf"]):
        self.mode = mode
        self.model_path_binary = model_path_binary
        self.model_path_multi = model_path_multi
        self.wf = self._create_workflow()
    def _create_workflow(self):
        wf = WorkflowManager()
        wf.add_node("InputValidator", InputValidatorNode(feature_list=FEATURE_LIST))
        wf.add_node("BinaryClassifier", BinaryClassifierNode(model_path=self.model_path_binary, mode=self.mode))
        wf.add_node("MultiClassifier", MultiClassifierNode(model_path=self.model_path_multi, mode=self.mode))

        # Nối các node
        wf.connect_nodes("START", "InputValidator")
        wf.connect_nodes("InputValidator", "BinaryClassifier")
        wf.connect_nodes("BinaryClassifier", "MultiClassifier", condition=lambda data: data["label"][0] == "ATTACK")
        wf.connect_nodes("BinaryClassifier", "END", condition=lambda data: data["label"][0] == "BENIGN")
        wf.connect_nodes("MultiClassifier", "END")
        return wf

    async def predict(self, df: pd.DataFrame):
        tasks = [self._process_row(row) for _, row in df.iterrows()]
        results_list = await asyncio.gather(*tasks)
        
        # Gom nhãn và probabilities
        labels = [r["label"][0] for r in results_list]
        probs = [r.get("probabilities", None) for r in results_list]
        return labels, probs

    async def _process_row(self, row):
        # Chỉ chạy workflow với data row hiện tại
        result = await asyncio.to_thread(self.wf.run, pd.DataFrame([row]))
        return result