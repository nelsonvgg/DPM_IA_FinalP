import os
import sys
import pytest
import pickle
from unittest.mock import patch, MagicMock, mock_open
from surprise import KNNBasic, SVD

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from train_model import train_svd_model, train_knn_model

# --- Fixtures ---
@pytest.fixture
def mock_trainset():
    """Creates a mock Surprise trainset object."""
    return MagicMock(name="MockTrainset")

# --- Test Cases ---
# Use @patch decorator to mock dependencies within the 'train_model' module context
@patch('train_model.pickle.dump')
@patch('builtins.open', new_callable=mock_open)
@patch('train_model.SVD')
def test_train_svd_model(mock_svd_class, mock_open_func, mock_pickle_dump, mock_trainset, tmp_path, capsys):
    """
    Tests the train_svd_model function.
    Checks model instantiation, fitting, saving, return value, and prints.
    """
    # --- Arrange ---
    mock_svd_instance = MagicMock(name="MockSVDInstance")
    mock_svd_class.return_value = mock_svd_instance
    test_save_path = tmp_path / "test_svd_model.pkl"
    expected_default_save_path = "./models/SVD_model.pkl"

    # --- Act ---
    returned_model = train_svd_model(mock_trainset, model_save_path=str(test_save_path))

    # --- Assert ---
    mock_svd_class.assert_called_once_with(random_state=42)
    mock_svd_instance.fit.assert_called_once_with(mock_trainset)
    mock_open_func.assert_called_once_with(str(test_save_path), "wb")
    mock_pickle_dump.assert_called_once_with(mock_svd_instance, mock_open_func())
    assert returned_model is mock_svd_instance
    captured = capsys.readouterr()
    assert "SVD model trained successfully." in captured.out
    assert "SVD model saved successfully at" in captured.out

    # --- Act & Assert (Default Path) ---
    mock_svd_class.reset_mock()
    mock_svd_instance.fit.reset_mock()
    mock_open_func.reset_mock()
    mock_pickle_dump.reset_mock()
    _ = train_svd_model(mock_trainset)
    mock_open_func.assert_called_once_with(expected_default_save_path, "wb")
    mock_pickle_dump.assert_called_once_with(mock_svd_instance, mock_open_func())

@patch('train_model.pickle.dump')
@patch('builtins.open', new_callable=mock_open)
@patch('train_model.KNNBasic')
def test_train_knn_model(mock_knn_class, mock_open_func, mock_pickle_dump, mock_trainset, tmp_path, capsys):
    """
    Tests the train_knn_model function.
    Checks model instantiation, fitting, saving, return value, and prints.
    """
    # --- Arrange ---
    mock_knn_instance = MagicMock(name="MockKNNInstance")
    mock_knn_class.return_value = mock_knn_instance
    test_save_path = tmp_path / "test_knn_model.pkl"
    expected_default_save_path = "./models/KNN_model.pkl"
    
    expected_sim_options = {
        "name": "cosine",
        "user_based": True,
    }

    # --- Act ---
    returned_model = train_knn_model(mock_trainset, model_save_path=str(test_save_path))

    # --- Assert ---
    mock_knn_class.assert_called_once_with(sim_options=expected_sim_options)
    mock_knn_instance.fit.assert_called_once_with(mock_trainset)
    mock_open_func.assert_called_once_with(str(test_save_path), "wb")
    mock_pickle_dump.assert_called_once_with(mock_knn_instance, mock_open_func())
    assert returned_model is mock_knn_instance
    captured = capsys.readouterr()
    assert "KNN model trained successfully." in captured.out
    assert "KNN model saved successfully at" in captured.out

    # --- Act & Assert (Default Path) ---
    mock_knn_class.reset_mock()
    mock_knn_instance.fit.reset_mock()
    mock_open_func.reset_mock()
    mock_pickle_dump.reset_mock()
    _ = train_knn_model(mock_trainset)
    mock_open_func.assert_called_once_with(expected_default_save_path, "wb")
    mock_pickle_dump.assert_called_once_with(mock_knn_instance, mock_open_func())
