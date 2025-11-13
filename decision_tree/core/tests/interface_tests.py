import sys
import os
import numpy as np
import pytest
from abc import ABC

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.criterion import Criterion, Entropy, Gini
from core.model import FitPrediction


# Test data fixtures
@pytest.fixture
def pure_dataset():
    """Dataset with all samples of the same class (no impurity)"""
    return np.array([1, 1, 1, 1, 1])


@pytest.fixture
def balanced_binary_dataset():
    """Perfectly balanced binary dataset (maximum impurity)"""
    return np.array([0, 0, 1, 1])


@pytest.fixture
def imbalanced_dataset():
    """Imbalanced dataset with multiple classes"""
    return np.array([0, 0, 0, 1, 2])


@pytest.fixture
def single_sample():
    """Single sample dataset"""
    return np.array([1])


class TestCriterionInterface:
    """Tests for the abstract Criterion interface"""
    
    def test_criterion_is_abstract(self):
        """Test that Criterion cannot be instantiated directly"""
        with pytest.raises(TypeError):
            Criterion()
    
    def test_criterion_subclasses_implement_score(self):
        """Test that concrete implementations have score method"""
        entropy = Entropy()
        gini = Gini()
        
        assert hasattr(entropy, 'score')
        assert hasattr(gini, 'score')
        assert callable(entropy.score)
        assert callable(gini.score)


class TestEntropy:
    """Tests for Entropy criterion implementation"""
    
    def test_entropy_pure_dataset(self, pure_dataset):
        """Test entropy of pure dataset should be 0"""
        entropy = Entropy()
        result = entropy.score(pure_dataset)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_entropy_balanced_binary(self, balanced_binary_dataset):
        """Test entropy of balanced binary dataset should be 1.0"""
        entropy = Entropy()
        result = entropy.score(balanced_binary_dataset)
        # For balanced binary: -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        assert result == pytest.approx(1.0, abs=1e-10)
    
    def test_entropy_single_sample(self, single_sample):
        """Test entropy of single sample should be 0"""
        entropy = Entropy()
        result = entropy.score(single_sample)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_entropy_imbalanced_dataset(self, imbalanced_dataset):
        """Test entropy calculation for imbalanced multiclass dataset"""
        entropy = Entropy()
        result = entropy.score(imbalanced_dataset)
        
        # Manual calculation for [0,0,0,1,2]:
        # p(0) = 3/5, p(1) = 1/5, p(2) = 1/5
        # entropy = -(3/5)*log2(3/5) - (1/5)*log2(1/5) - (1/5)*log2(1/5)
        expected = -(3/5)*np.log2(3/5) - 2*(1/5)*np.log2(1/5)
        assert result == pytest.approx(expected, abs=1e-10)
    
    def test_entropy_returns_float(self, balanced_binary_dataset):
        """Test that entropy returns a numeric value"""
        entropy = Entropy()
        result = entropy.score(balanced_binary_dataset)
        assert isinstance(result, (int, float, np.number))
        assert result >= 0  # Entropy is always non-negative
    
    def test_entropy_empty_array(self):
        """Test entropy behavior with empty array"""
        entropy = Entropy()
        empty_array = np.array([])
        # This might raise an error or return 0, depending on implementation
        # Let's see what happens
        try:
            result = entropy.score(empty_array)
            # If it doesn't raise an error, result should be 0
            assert result == pytest.approx(0.0, abs=1e-10)
        except (ValueError, ZeroDivisionError):
            # It's acceptable for empty arrays to raise an error
            pass


class TestGini:
    """Tests for Gini criterion implementation"""
    
    def test_gini_pure_dataset(self, pure_dataset):
        """Test Gini impurity of pure dataset should be 0"""
        gini = Gini()
        result = gini.score(pure_dataset)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_gini_balanced_binary(self, balanced_binary_dataset):
        """Test Gini impurity of balanced binary dataset should be 0.5"""
        gini = Gini()
        result = gini.score(balanced_binary_dataset)
        # For balanced binary: 2 * 0.5 * (1 - 0.5) = 0.5
        assert result == pytest.approx(0.5, abs=1e-10)
    
    def test_gini_single_sample(self, single_sample):
        """Test Gini impurity of single sample should be 0"""
        gini = Gini()
        result = gini.score(single_sample)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_gini_imbalanced_dataset(self, imbalanced_dataset):
        """Test Gini calculation for imbalanced multiclass dataset"""
        gini = Gini()
        result = gini.score(imbalanced_dataset)
        
        # Manual calculation for [0,0,0,1,2]:
        # p(0) = 3/5, p(1) = 1/5, p(2) = 1/5
        # gini = (3/5)*(1-3/5) + (1/5)*(1-1/5) + (1/5)*(1-1/5)
        # gini = (3/5)*(2/5) + 2*(1/5)*(4/5) = 6/25 + 8/25 = 14/25 = 0.56
        expected = (3/5)*(2/5) + 2*(1/5)*(4/5)
        assert result == pytest.approx(expected, abs=1e-10)
    
    def test_gini_returns_float(self, balanced_binary_dataset):
        """Test that Gini returns a numeric value"""
        gini = Gini()
        result = gini.score(balanced_binary_dataset)
        assert isinstance(result, (int, float, np.number))
        assert result >= 0  # Gini impurity is always non-negative
        assert result <= 1  # Gini impurity is bounded by 1
    
    def test_gini_empty_array(self):
        """Test Gini behavior with empty array"""
        gini = Gini()
        empty_array = np.array([])
        try:
            result = gini.score(empty_array)
            assert result == pytest.approx(0.0, abs=1e-10)
        except (ValueError, ZeroDivisionError):
            # It's acceptable for empty arrays to raise an error
            pass


class TestCriterionComparison:
    """Tests comparing different criterion implementations"""
    
    def test_pure_datasets_both_zero(self, pure_dataset):
        """Both criteria should return 0 for pure datasets"""
        entropy = Entropy()
        gini = Gini()
        
        entropy_result = entropy.score(pure_dataset)
        gini_result = gini.score(pure_dataset)
        
        assert entropy_result == pytest.approx(0.0, abs=1e-10)
        assert gini_result == pytest.approx(0.0, abs=1e-10)
    
    def test_maximum_impurity_comparison(self):
        """Compare entropy and gini at maximum impurity"""
        # For binary classification, maximum impurity occurs at 50/50 split
        balanced_data = np.array([0, 1])
        
        entropy = Entropy()
        gini = Gini()
        
        entropy_result = entropy.score(balanced_data)
        gini_result = gini.score(balanced_data)
        
        # Entropy should be 1.0, Gini should be 0.5
        assert entropy_result == pytest.approx(1.0, abs=1e-10)
        assert gini_result == pytest.approx(0.5, abs=1e-10)


class MockFitPrediction(FitPrediction):
    """Mock implementation of FitPrediction for testing"""
    
    def __init__(self):
        self.is_fitted = False
        self.training_data = None
    
    def fit(self, X, y):
        self.training_data = (X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Simple mock prediction: return first class for all samples
        return np.zeros(X.shape[0], dtype=int)


class TestFitPredictionInterface:
    """Tests for the abstract FitPrediction interface"""
    
    def test_fitprediction_is_abstract(self):
        """Test that FitPrediction cannot be instantiated directly"""
        with pytest.raises(TypeError):
            FitPrediction()
    
    def test_fitprediction_subclass_implements_methods(self):
        """Test that concrete implementations have required methods"""
        mock_model = MockFitPrediction()
        
        assert hasattr(mock_model, 'fit')
        assert hasattr(mock_model, 'predict')
        assert callable(mock_model.fit)
        assert callable(mock_model.predict)
    
    def test_fit_method_functionality(self):
        """Test basic fit functionality"""
        mock_model = MockFitPrediction()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        # Should return self for method chaining
        result = mock_model.fit(X, y)
        assert result is mock_model
        assert mock_model.is_fitted
        assert mock_model.training_data is not None
    
    def test_predict_method_functionality(self):
        """Test basic predict functionality"""
        mock_model = MockFitPrediction()
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        X_test = np.array([[2, 3], [4, 5]])
        
        # Fit first
        mock_model.fit(X_train, y_train)
        
        # Then predict
        predictions = mock_model.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises appropriate error"""
        mock_model = MockFitPrediction()
        X_test = np.array([[1, 2]])
        
        with pytest.raises(ValueError):
            mock_model.predict(X_test)
    
    def test_fit_predict_workflow(self):
        """Test complete fit-predict workflow"""
        mock_model = MockFitPrediction()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        X_test = np.array([[2, 3]])
        
        # Full workflow
        mock_model.fit(X_train, y_train)
        predictions = mock_model.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == 1
