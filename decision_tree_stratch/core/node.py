class Node:
    def __init__(self, feature=None, threshold=None, right=None, left=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value 
         
    def is_leaf_node(self):
        return self.value is not None
    