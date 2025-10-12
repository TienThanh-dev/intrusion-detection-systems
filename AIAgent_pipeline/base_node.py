from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def process(self, data):
        """
        Hàm xử lý chính của mỗi node.
        Mỗi node con phải override hàm này.
        """
        pass
