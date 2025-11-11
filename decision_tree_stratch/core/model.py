from abc import ABC , abstractmethod

class FitPrediction(ABC):
    
    @abstractmethod
    def fit(self):...
    
    @abstractmethod
    def predict(self):...
