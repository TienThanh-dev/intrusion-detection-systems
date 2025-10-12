import logging
import networkx as nx
import matplotlib.pyplot as plt

class WorkflowManager:
    def __init__(self):
        self.nodes = {}  # {node_name: node_object}
        self.connections = []  # [(from, to, condition)]
        self.start_node = "START"
        self.end_node = "END"

    # ===== 1️⃣ Thêm node thật =====
    def add_node(self, name, node_object):
        """
        Thêm 1 node xử lý thực tế vào workflow.
        """
        if name in self.nodes:
            logging.warning(f"[WARN] Node '{name}' đã tồn tại — ghi đè.")
        self.nodes[name] = node_object
        logging.info(f"Đã thêm node: {name}")

    # ===== 2️⃣ Nối các node (có điều kiện tùy chọn) =====
    def connect_nodes(self, from_node, to_node, condition=None):
        """
        Kết nối 2 node (có thể từ START hoặc tới END).
        """
        if from_node != self.start_node and from_node not in self.nodes:
            logging.error(f"Node nguồn '{from_node}' chưa tồn tại.")
            return
        if to_node != self.end_node and to_node not in self.nodes:
            logging.error(f"Node đích '{to_node}' chưa tồn tại.")
            return
        self.connections.append((from_node, to_node, condition))
        logging.info(f"Kết nối: {from_node} → {to_node} (cond={condition is not None})")

    # ===== 3️⃣ Chạy toàn bộ workflow =====
    def run(self, data):
        """
        Bắt đầu chạy workflow từ START.
        """
        logging.info("Bắt đầu workflow từ START.")
        current_node = self.start_node
        current_data = data
        visited = set()

        while current_node != self.end_node:
            visited.add(current_node)

            # Tìm các đường đi từ node hiện tại
            next_nodes = [(src, dst, cond) for src, dst, cond in self.connections if src == current_node]
            if not next_nodes:
                logging.info(f"Không còn node nối tiếp sau '{current_node}', kết thúc workflow.")
                break

            # Nếu là START, chỉ cần đi tiếp node đầu tiên
            if current_node == self.start_node:
                _, next_node, _ = next_nodes[0]
                current_node = next_node
                continue

            # Xử lý node hiện tại
            node_obj = self.nodes[current_node]
            logging.info(f"Đang chạy node: {current_node}")
            current_data = node_obj.process(current_data)

            # Tìm node tiếp theo phù hợp điều kiện
            next_node = None
            for src, dst, cond in self.connections:
                if src == current_node:
                    if cond is None or cond(current_data):
                        next_node = dst
                        break

            if not next_node:
                logging.info(f"Dừng tại '{current_node}' (không có kết nối hợp lệ).")
                break

            current_node = next_node

        logging.info("Workflow hoàn tất.")
        return current_data

    # ===== 4️⃣ Vẽ sơ đồ workflow =====
    def draw_workflow(self):
        """Vẽ sơ đồ các node đã kết nối, hiển thị điều kiện."""
        G = nx.DiGraph()
        all_nodes = [self.start_node, self.end_node] + list(self.nodes.keys())

        for node in all_nodes:
            G.add_node(node)

        for src, dst, cond in self.connections:
            label = "cond" if cond else ""
            G.add_edge(src, dst, label=label)

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2800, node_color="lightgreen", arrows=True)
        labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Workflow Diagram")
        plt.show()
        return G

