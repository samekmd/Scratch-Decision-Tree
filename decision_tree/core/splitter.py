import numpy as np 
from abc import ABC, abstractmethod
from criterion import Criterion

class Splitter(ABC):
    """
    Classe abstrata que define a estrutura base para classes responsáveis por dividir 
    conjuntos de dados em partes menores durante a construção da árvore de decisão.
    """
    
    @abstractmethod
    def _split(self):...
    
    
class BestSplitter(Splitter):
    """
    Implementa o processo de divisão dos dados durante a construção da árvore, 
    identificando a melhor feature e o limiar (threshold) que maximizam o ganho de informação.
    
    Atributos:
        - criterion -> Critério que irá ser utilizado para a equação do ganho de informação
    """
    def __init__(self, criterion:Criterion):
        self.criterion = criterion()
        
    def _split(self, feature_column, threshold):
        """
        Divide os índices das amostras em dois grupos com base em um limiar (threshold)
        
        Parâmetros:
            - feature_column -> Coluna do conjunto de dados X que será dividida
            - threshold -> "Limiar" imposto a coluna de X, ex: altura >= 1.60
            
        Retorna:
            - Lado esquerdo: amostras com valor ≤ threshold
            - Lado direito: amostras com valor > threshold
        """
        left = np.argwhere(feature_column <= threshold).flatten()
        right = np.argwhere(feature_column > threshold).flatten()
        return left, right
    
    def _best_split(self, X, y, feats_idxs):
        """
        Determina qual feature e qual limiar de corte geram a melhor separação dos dados.
        
        Parâmetros:
            - X -> features (atributos) do conjunto de dados
            - y -> classes (labels) do conjunto de dados
            - feats_idxs -> Indíces escolhidos de X
            
        Retorna:
            - best_feature -> feature com maior ganho de informação
            - best_threshold -> limiar com maior ganho de informação
        """
        best_gain = -1 
        best_feature, best_threshold = None, None
        
        """
        Itera sobre as features selecionadas, 
        testando diferentes thresholds e calculando o ganho de informação de cada 
        combinação para identificar a divisão mais informativa.
        """
        for feat in feats_idxs: 
            feature_column = X[:, feat] # Todos os dados da coluna 
            thresholds = np.unique(feature_column) # Todos os valores únicos da coluna (limiares)
            for thr in thresholds: # Looping para calcular o ganho de informação dos limiares encontrados
                gain_information = self._information_gain(feature_column,y, thr) # Cálculo do ganho de informação
                if gain_information > best_gain: 
                    # Escolhendo o melhor ganho de informação, feature e threshold
                    best_gain = gain_information
                    best_feature = feat
                    best_threshold = thr
        return best_feature, best_threshold
    
    
    def _information_gain(self, X_column, parent, threshold):
        """
        Calcula a redução da impureza (entropia ou índice de Gini) 
        resultante da divisão dos dados com base em uma determinada feature e threshold.
        
        Parâmetros:
            - parent -> rótulos (classes) do nó pai
            - X_column -> Coluna de X e seus respectivos valores (com base no feat_idxs)
            - threshold -> Valor que será utilizado para o split (ex: x > 5)

        Retorna:
            - Ganho de informação de um determinado split
        """
        
        parent_impurity = self.criterion.score(parent) # calcula impureza da variável alvo
        left, right = self._split(X_column, threshold) # divisão dos dados 
        
        parent_size = len(parent) 
        size_left, size_right = len(left), len(right)
        
        if size_left == 0 or size_right == 0:
            return 0
        
        # Calculo da impureza da variável alvo no conjunto de dados da esquerda e da direita
        left_impurity, right_impurity = self.criterion.score(parent[left]) , self.criterion.score(parent[right]) 
        
        # Cálculo da média ponderada das impurezas dos nós filho (esquerda e direita).
        child_impurity = (size_left/parent_size) * left_impurity + (size_right/parent_size) * right_impurity 
         
        # Cálculo do ganho de informação 
        information_gain = parent_impurity - child_impurity
        
        return information_gain
    
    
