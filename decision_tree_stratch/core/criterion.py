from abc import ABC, abstractmethod
import numpy as np

class Criterion(ABC):
    
    @abstractmethod
    def score(self, y): ...
    
   
                
class Entropy(Criterion):
    
    def score(self, y): 
        result = 0
        for label in np.unique(y):
            sample_label = y[y == label]
            p1 = len(sample_label) / len(y)
            result += -p1 * np.log2(p1)
        return result
    
    
class Gini(Criterion):
    
    def score(self, y): 
        result = 0
        for label in np.unique(y):
            sample_label = y[y == label]
            p1 = len(sample_label) / len(y)
            result += p1 * (1 - p1)
        return result
    
    
      
 
