import sys
import os
import numpy as np
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.tree import DecisionTree
from core.exceptions import NumberOfFeaturesOutOfRange


def make_simple_dataset():
	# Simple binary classification separable dataset
	X = np.array([
		[0.0, 0.0],
		[0.1, 0.2],
		[0.2, 0.1],
		[1.0, 1.0],
		[1.1, 0.9],
		[0.9, 1.2]
	])
	y = np.array([0, 0, 0, 1, 1, 1])
	return X, y


def test_fit_predict_basic():
	X, y = make_simple_dataset()
	clf = DecisionTree(criterion='gini', max_depth=3, min_samples_split=1, random_state=42, n_features=2)
	clf.fit(X, y)

	preds = clf.predict(X)
	assert isinstance(preds, np.ndarray)
	assert preds.shape == y.shape
	# Should perfectly separate this simple dataset
	assert np.array_equal(preds, y)


def test_number_of_features_out_of_range():
	X, y = make_simple_dataset()
	# Request more features than available should raise
	clf = DecisionTree(n_features=10)
	with pytest.raises(NumberOfFeaturesOutOfRange):
		clf.fit(X, y)


def test_predict_after_partial_fit_returns_array_shape():
	X, y = make_simple_dataset()
	clf = DecisionTree(max_depth=0, min_samples_split=100, n_features=2)
	# With max_depth=0 the tree should immediately create a leaf with majority class
	clf.fit(X, y)
	preds = clf.predict(X)
	assert preds.shape == y.shape
	# All predictions should be the majority class (0 or 1); for this dataset majority is 0 and 1 tied
	# Ensure returned values are scalar integers
	assert preds.dtype == y.dtype or np.issubdtype(preds.dtype, np.integer)

