from typing import Literal
from criterion import Entropy, Gini
from node import Node
import numpy as np
import pandas as pd
from model import FitPrediction
from splitter import BestSplitter

TREE_CRITERION = {
        'entropy': Entropy,
        'gini': Gini
}


class DecisionTree(FitPrediction):
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
        self._splitter = BestSplitter(criterion=TREE_CRITERION[criterion])
        self.root = None
    
    
    def fit(self, X, y):
        self.root = self._grow_tree(X,y)
        
    
    def _grow_tree(self, X,y, depth=0):
        np.random.seed(self.random_state)
        
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))             
        
        selected_features = np.random.choice(n_feats, self.n_features, replace=False)
        
        stop_param = (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split)
        
        if stop_param:
            leaf_value = self._most_comman_target(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._splitter._best_split(X,y, selected_features)
        
        left_idxs, right_idxs = self._splitter._split(X[:, best_feature], best_threshold)
        
        left = self._grow_tree(X[left_idxs, : ], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, : ], y[right_idxs], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            right=right,
            left=left
        ) 
        
        
    def _most_comman_target(self, y):
        y_series = pd.Series(y)
        return y_series.mode()[0]
    
    def predict(self, X):
        return np.array([self._walk_tree(sample,self.root) for sample in X])
    
    def _walk_tree(self, x, node:Node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._walk_tree(x, node.left)
        return self._walk_tree(x, node.right)
    
    