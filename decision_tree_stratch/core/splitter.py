import numpy as np 
from abc import ABC, abstractmethod
from criterion import Criterion


class Splitter(ABC):
    
    
    @abstractmethod
    def _split(self):...
    
    
    
class BestSplitter(Splitter):
    def __init__(self, criterion:Criterion):
        self.criterion = criterion()
        
    def _split(self, feature_column, threshold):
        left = np.argwhere(feature_column <= threshold).flatten()
        right = np.argwhere(feature_column > threshold).flatten()
        return left, right
    
    def _best_split(self, X, y, feats_idxs):
        best_gain = -1 
        best_feature, best_threshold = None, None
        for feat in feats_idxs:
            feature_colum = X[:, feat]
            thresholds = np.unique(feature_colum)
            for thr in thresholds:
                gain_information = self._information_gain(feature_colum,y, thr)
                if gain_information > best_gain:
                    best_gain = gain_information
                    best_feature = feat
                    best_threshold = thr
        return best_feature, best_threshold
    
    
    def _information_gain(self, X_colum, parent, threshold):
        parent_impurity = self.criterion.score(parent)
        left, right = self._split(X_colum, threshold)
        
        parent_size = len(parent)
        size_left, size_right = len(left), len(right)
        
        if size_left == 0 or size_right == 0:
            return 0
        
        left_impurity, right_impurity = self.criterion.score(parent[left]) , self.criterion.score(parent[right])
        
        child_impurity = (size_left/parent_size) * left_impurity + (size_right/parent_size) * right_impurity 
         
        information_gain = parent_impurity - child_impurity
        
        return information_gain
    
    
