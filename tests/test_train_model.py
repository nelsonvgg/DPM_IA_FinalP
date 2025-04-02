import os
import sys
import pytest
import pickle
import tempfile
from unittest.mock import patch, mock_open, MagicMock
from surprise import KNNBasic, Reader, Dataset
from surprise.trainset import Trainset

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from train_model import train_knn_model, train_svd_model

@pytest.fixture
def mock_trainset():
    """Create a mock Trainset object."""
    # Create a minimal dataset
    reader = Reader(rating_scale=(1, 5))
    data = [
        ('user1', 'item1', 4.0),
        ('user1', 'item2', 3.5),
        ('user2', 'item1', 5.0),
        ('user2', 'item3', 2.0),
    ]
    dataset = Dataset.load_from_folds([(data, data)], reader)
    return dataset.build_full_trainset()

def test_train_knn_model_returns_model(mock_trainset):
    """Test that train_knn_model returns a KNNBasic model."""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_dump:
            # Call the function
            model = train_knn_model(mock_trainset, "dummy_path.pkl")
            
            # Check that the returned object is a KNNBasic model
            assert isinstance(model, KNNBasic)
            
            # Check that the model was fit with the trainset
            assert model.trainset is mock_trainset

def test_train_knn_model_uses_correct_parameters(mock_trainset):
    """Test that train_knn_model initializes the model with correct parameters."""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_dump:
            # Call the function
            model = train_knn_model(mock_trainset, "dummy_path.pkl")
            
            # Check that the model has the expected similarity options
            assert model.sim_options['name'] == 'cosine'
            assert model.sim_options['user_based'] is True

def test_train_knn_model_saves_model(mock_trainset):
    """Test that train_knn_model saves the model to disk."""
    # Use a real temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        model_path = temp_file.name
    
    try:
        # Train and save the model
        model = train_knn_model(mock_trainset, model_path)
        
        # Check that the file exists
        assert os.path.exists(model_path)
        
        # Check that we can load the model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Verify it's a KNNBasic model with the same parameters
        assert isinstance(loaded_model, KNNBasic)
        assert loaded_model.sim_options['name'] == 'cosine'
        assert loaded_model.sim_options['user_based'] is True
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)

def test_train_knn_model_with_custom_path(mock_trainset):
    """Test that train_knn_model uses the custom path provided."""
    custom_path = "custom_model_path.pkl"
    
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_dump:
            # Call the function with a custom path
            model = train_knn_model(mock_trainset, custom_path)
            
            # Check that open was called with the custom path
            mock_file.assert_called_once_with(custom_path, "wb")

def test_train_knn_model_with_empty_trainset():
    """Test that train_knn_model handles an empty trainset gracefully."""
    # Create an empty trainset
    reader = Reader(rating_scale=(1, 5))
    data = []  # No ratings
    dataset = Dataset.load_from_folds([(data, data)], reader)
    empty_trainset = dataset.build_full_trainset()
    
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_dump:
            # Call the function with the empty trainset
            model = train_knn_model(empty_trainset, "dummy_path.pkl")
            
            # Check that a model was still returned
            assert isinstance(model, KNNBasic)

def test_train_knn_model_with_different_sim_options():
    """Test that train_knn_model can be modified to use different similarity options."""
    # Create a mock for the KNNBasic class to capture initialization parameters
    with patch('surprise.KNNBasic') as mock_knn:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_knn.return_value = mock_instance
        
        # Create a mock trainset
        mock_ts = MagicMock(spec=Trainset)
        
        # Modify the train_knn_model function to use different sim_options
        def modified_train_knn_model(trainset, model_save_path):
            sim_options = {
                "name": "pearson",
                "user_based": False,  # item-based
            }
            model = KNNBasic(sim_options=sim_options)
            model.fit(trainset)
            with open(model_save_path, "wb") as f:
                pickle.dump(model, f)
            return model
        
        # Call the modified function
        with patch("builtins.open", mock_open()):
            with patch("pickle.dump"):
                modified_train_knn_model(mock_ts, "dummy_path.pkl")
        
        # Check that KNNBasic was called with the expected sim_options
        # Note: We need to look at the actual calls since we're using a patched version
        args, kwargs = mock_knn.call_args
        assert kwargs['sim_options']['name'] == 'pearson'
        assert kwargs['sim_options']['user_based'] is False