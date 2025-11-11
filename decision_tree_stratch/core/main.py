from sklearn import datasets
from sklearn.model_selection import train_test_split
from tree import DecisionTree
from sklearn.metrics import accuracy_score

data = datasets.load_breast_cancer()
X,y = data.data, data.target

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)


mdl = DecisionTree(criterion='entropy', max_depth=10, min_samples_split=100, random_state=42)


mdl.fit(X_train, y_train)

y_pred = mdl.predict(X_val)


print(accuracy_score(y_val, y_pred))