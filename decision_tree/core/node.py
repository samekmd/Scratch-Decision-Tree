class Node:
    """
    Classe que irá representar cada nó de uma árvore, representando a sua estrutura
    Atributos:
        - feature -> Representa o índice da coluna (feature) usada para dividir os dados nesse nó.
        - threshold -> É o valor limite (limiar) utilizado para a decisão.
        - right, left -> São ponteiros (referências) para os nós filhos.
        - value -> É o valor final da classe (ou da média, no caso de regressão) armazenado apenas nos nós folha.
    """
    def __init__(self, feature=None, threshold=None, right=None, left=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value 
         
    def is_leaf_node(self):
        """
        Esse método serve para verificar se o nó atual é um nó folha.
        """
        return self.value is not None
    