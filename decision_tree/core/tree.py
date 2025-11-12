from typing import Literal
from criterion import Entropy, Gini
from node import Node
import numpy as np
import pandas as pd
from model import FitPrediction
from splitter import BestSplitter

TREE_CLF_CRITERION = {
        'entropy': Entropy,
        'gini': Gini
}


class DecisionTree(FitPrediction):
    """
    Classe que cria uma árvore de decisão de classificação
    
    Parâmetros: 
        - criterion -> Critério que será utilizado para achar a melhor divisão dos dados
        - max_depth -> Profundidade máxima da árvore (máximo de passos para chegar a uma classificação)
        - min_samples_split -> Número minímo de amostras para divisão
        - n_features -> Quantidade de features que serão utilizadas para construção da árvore     
        
    Atributos: 
        - _splitter -> Instância da classe BestSplitter, que será utilizado para achar a melhor divisão dos dados 
        - root -> Nó raiz da árvore
    """
    
    def __init__(self, 
                 criterion: Literal['entropy', 'gini'] = 'entropy',
                 max_depth = 10, 
                 min_samples_split = 100, 
                 random_state = 0,
                 n_features = 10
                 ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.criterion = criterion 
        self.n_features = n_features
        self._splitter = BestSplitter(criterion=TREE_CLF_CRITERION[criterion])
        self.root = None
    
    
    def fit(self, X, y):
        """
        Função que treina o modelo
        
        Parâmetros:
            - X -> Amostras
            - y -> Variável alvo
        """
        self.root = self._grow_tree(X,y)
        
    def _grow_tree(self, X,y, depth=0):
        """
        Função para montar a árvore
        
        Parâmetros:
            - X -> Amostras
            - y -> Varíavel alvo
            - depth -> Contador para medir a profundidade
        """
        
        np.random.seed(self.random_state) # Definindo semente de aleatorieadade
        
        n_samples, n_feats = X.shape # Obtendo número de amostras e features dos dados
        n_labels = len(np.unique(y))  # Obtendo número de classes           
        
        # Selecionando randomicamente features para construção da árvore (evita overfitting )
        selected_features = np.random.choice(n_feats, self.n_features, replace=False) 
        
        # Condição de parada
        stop_param = (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split)
        
        if stop_param:
            # Atribuindo a classe que mais se repete a um nó folha
            leaf_value = self._most_common_target(y)
            return Node(value=leaf_value)
        
        # Obtendo melhor feature e threshold
        best_feature, best_threshold = self._splitter._best_split(X,y, selected_features)
        
        # Divisão das amostras conforme features e thresholds selecionadas
        left_idxs, right_idxs = self._splitter._split(X[:, best_feature], best_threshold)
        
        # Criação de nós filhos
        left = self._grow_tree(X[left_idxs, : ], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, : ], y[right_idxs], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            right=right,
            left=left
        ) 
        
        
    def _most_common_target(self, y):
        """
        Função que encontra classe que mais se repete dentro de um conjunto
        
        Parâmetros:
            - y -> Conjunto de classes
            
        Retorna:
            - Classe que mais se repete
        """
        y_series = pd.Series(y)
        return y_series.mode()[0]
    
    def predict(self, X):
        """
        Função para fazer a predição de um conjunto de dados, com base na árvore criada
        
        Parâmetros:
            - X -> Conjunto de amostras
            
        Retorna:
            - Array com os valores previstos
        
        Funcionamento: 
            - Para cada amostra em X, percorre a árvore com base em seus atributos até chegar a um nó folha, 
              retornando a classe correspondente. 
        """
        return np.array([self._walk_tree(sample,self.root) for sample in X])
    
    def _walk_tree(self, x, node:Node):
        """
        Percorre recursivamente os nós da árvore até alcançar um nó folha, retornando a classe prevista para a amostra.
        
        Parâmetros:
            - x -> Amostra
            - node -> Nó da árvore
        """
       
        # Verificando se o nó é um nó folha
        if node.is_leaf_node():
            # Retorna o valor do nó folha
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._walk_tree(x, node.left)
        return self._walk_tree(x, node.right)
    
    