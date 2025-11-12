from abc import ABC, abstractmethod
import numpy as np

class Criterion(ABC):
    
    @abstractmethod
    def score(self, y): ...
    
                   
class Entropy(Criterion):
    """
    Implementa o cálculo da entropia, uma medida clássica de desordem vinda da teoria da informação.
    """
    def score(self, y): 
        """
        Cálcula a desordem de um determinado conjunto de dados
        Parâmetros:
            - y -> Target

        Retorna:
            - entropia do conjunto
        """
        result = 0
        for label in np.unique(y):
            sample_label = y[y == label]
            p1 = len(sample_label) / len(y)
            result += -p1 * np.log2(p1)
        return result
    
    
class Gini(Criterion):
    """
    O índice de Gini mede a probabilidade de classificação incorreta de uma 
    amostra se ela fosse rotulada aleatoriamente com base na distribuição das classes.
    """
    def score(self, y): 
        """
        Calcula a desordem do conjunto de dados
        
        Parâmetros:
            - y -> Target

        Retorna:
            - Impureza do conjunto
        """
        result = 0
        for label in np.unique(y):
            sample_label = y[y == label]
            p1 = len(sample_label) / len(y)
            result += p1 * (1 - p1)
        return result
    
    
      
 
