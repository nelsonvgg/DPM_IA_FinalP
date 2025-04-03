import os
import sys
import pytest
import pickle
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from surprise import KNNBasic, SVD, Reader, Dataset


# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
# We need to mock dependencies BEFORE they are imported by the module under test usually,
# but here we mock them during the test execution using @patch.
from train_model import train_svd_model, train_knn_model

# --- Fixtures ---

@pytest.fixture
def mock_trainset():
    """Creates a mock Surprise trainset object."""
    # A simple MagicMock is often sufficient if we only care that it's passed to fit
    return MagicMock(name="MockTrainset")

# --- Test Cases ---

# Use @patch decorator to mock dependencies within the 'train_model' module context
# Decorators are applied bottom-up
@patch('train_model.pickle.dump') # Mock pickle.dump first
@patch('builtins.open', new_callable=mock_open) # Mock the built-in open function
@patch('train_model.SVD') # Mock the SVD class from surprise where it's used in train_model
def test_train_svd_model(mock_svd_class, mock_open_func, mock_pickle_dump, mock_trainset, tmp_path, capsys):
    """
    Tests the train_svd_model function.
    Checks model instantiation, fitting, saving, return value, and prints.
    """
    # --- Arrange ---
    # Configure the mock SVD class
    mock_svd_instance = MagicMock(name="MockSVDInstance")
    mock_svd_class.return_value = mock_svd_instance # When SVD() is called, return our mock instance

    # Define a specific save path within the temporary directory
    test_save_path = tmp_path / "test_svd_model.pkl"
    expected_default_save_path = "D:/GitHub/DPM_IA_Mid-Term/models/SVD_model.pkl" # Default path from function signature

    # --- Act ---
    # Call the function with a specific path
    returned_model = train_svd_model(mock_trainset, model_save_path=str(test_save_path))

    # --- Assert ---
    # 1. Check SVD instantiation
    mock_svd_class.assert_called_once_with(random_state=42)

    # 2. Check model fitting
    mock_svd_instance.fit.assert_called_once_with(mock_trainset)

    # 3. Check file opening
    mock_open_func.assert_called_once_with(str(test_save_path), "wb")

    # 4. Check pickling (saving)
    # mock_open returns a mock file handle. We need to check dump was called with the model and this handle.
    mock_pickle_dump.assert_called_once_with(mock_svd_instance, mock_open_func()) # mock_open() gets the handle

    # 5. Check return value
    assert returned_model is mock_svd_instance

    # 6. Check prints (optional but good)
    captured = capsys.readouterr()
    assert "SVD model trained successfully." in captured.out
    assert "SVD model saved successfully." in captured.out

    # --- Act & Assert (Default Path) ---
    # Reset mocks for the next call
    mock_svd_class.reset_mock()
    mock_svd_instance.fit.reset_mock()
    mock_open_func.reset_mock()
    mock_pickle_dump.reset_mock()

    # Call the function with the default path
    _ = train_svd_model(mock_trainset) # Path not provided

    # Assert file opening uses the default path
    mock_open_func.assert_called_once_with(expected_default_save_path, "wb")
    mock_pickle_dump.assert_called_once_with(mock_svd_instance, mock_open_func())

# Use separate patches for KNN test
@patch('train_model.pickle.dump')
@patch('builtins.open', new_callable=mock_open)
@patch('train_model.KNNBasic') # Mock the KNNBasic class
def test_train_knn_model(mock_knn_class, mock_open_func, mock_pickle_dump, mock_trainset, tmp_path, capsys):
    """
    Tests the train_knn_model function.
    Checks model instantiation, fitting, saving, return value, and prints.
    """
    # --- Arrange ---
    mock_knn_instance = MagicMock(name="MockKNNInstance")
    mock_knn_class.return_value = mock_knn_instance

    test_save_path = tmp_path / "test_knn_model.pkl"
    expected_default_save_path = "D:/GitHub/DPM_IA_Mid-Term/models/KNN_model.pkl"

    expected_sim_options = {
        "name": "cosine",
        "user_based": True,
    }

    # --- Act ---
    returned_model = train_knn_model(mock_trainset, model_save_path=str(test_save_path))

    # --- Assert ---
    # 1. Check KNN instantiation
    mock_knn_class.assert_called_once_with(sim_options=expected_sim_options)

    # 2. Check model fitting
    mock_knn_instance.fit.assert_called_once_with(mock_trainset)

    # 3. Check file opening
    mock_open_func.assert_called_once_with(str(test_save_path), "wb")

    # 4. Check pickling
    mock_pickle_dump.assert_called_once_with(mock_knn_instance, mock_open_func())

    # 5. Check return value
    assert returned_model is mock_knn_instance

    # 6. Check prints
    captured = capsys.readouterr()
    assert "KNN model trained successfully." in captured.out
    assert "KNN model saved successfully." in captured.out

    # --- Act & Assert (Default Path) ---
    mock_knn_class.reset_mock()
    mock_knn_instance.fit.reset_mock()
    mock_open_func.reset_mock()
    mock_pickle_dump.reset_mock()

    _ = train_knn_model(mock_trainset) # Use default path

    mock_open_func.assert_called_once_with(expected_default_save_path, "wb")
    mock_pickle_dump.assert_called_once_with(mock_knn_instance, mock_open_func())


# # Potential test for file saving error (if the function handled it)
# @patch('train_model.pickle.dump', side_effect=IOError("Disk full"))
# @patch('builtins.open', new_callable=mock_open)
# @patch('train_model.SVD')
# def test_train_svd_model_save_error(mock_svd_class, mock_open_func, mock_pickle_dump, mock_trainset, tmp_path):
#     """ Tests how the function behaves if saving fails. """
#     mock_svd_instance = MagicMock()
#     mock_svd_class.return_value = mock_svd_instance
#     test_save_path = tmp_path / "test_svd_model.pkl"

#     # Depending on how error handling *should* work (e.g., raise error, log warning)
#     # Currently, the original code doesn't handle this, so the IOError would just propagate
#     with pytest.raises(IOError, match="Disk full"):
#          train_svd_model(mock_trainset, model_save_path=str(test_save_path))

#     # Assert that fit was still called before the error
#     mock_svd_instance.fit.assert_called_once_with(mock_trainset)
#     mock_open_func.assert_called_once_with(str(test_save_path), "wb")
#     mock_pickle_dump.assert_called_once_with(mock_svd_instance, mock_open_func())
