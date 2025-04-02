# Import the function to test using absolute import
import os
import sys
import tempfile
import pytest

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from load_data import load_and_split_data, load_data

@pytest.fixture
def mock_movie_data():
    """Create a temporary CSV file with mock movie data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("movieId,title,genres\n")
        f.write("1,Movie 1,Action\n")
        f.write("2,Movie 2,Comedy\n")
        f.write("3,Movie 3,Drama\n")
        f.write("4,Movie 4,Action|Comedy\n")
        f.write("5,Movie 5,Comedy|Drama\n")
        temp_file = f.name
    
    yield temp_file
    os.unlink(temp_file)  # Delete the temporary file

@pytest.fixture
def mock_rating_data():
    """Create a temporary CSV file with mock rating data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("userId,movieId,rating,timestamp\n")
        f.write("1,1,4.5,1234567890\n")
        f.write("1,2,3.0,1234567891\n")
        f.write("1,3,5.0,1234567892\n")
        f.write("2,1,2.5,1234567893\n")
        f.write("2,4,3.5,1234567894\n")
        f.write("3,2,4.0,1234567895\n")
        f.write("3,5,3.0,1234567896\n")
        temp_file = f.name
    
    yield temp_file
    os.unlink(temp_file)  # Delete the temporary file

def test_load_and_split_data_basic(mock_movie_data, mock_rating_data):
    """Test that load_and_split_data returns objects of the correct types."""
    trainset, testset = load_and_split_data(mock_movie_data, mock_rating_data)
    
    # Check that the trainset is a Surprise Trainset
    assert hasattr(trainset, 'build_anti_testset')
    assert hasattr(trainset, 'all_ratings')
    
    # Check that the testset is a list of tuples (user, item, rating)
    assert isinstance(testset, list)
    assert all(isinstance(x, tuple) and len(x) == 3 for x in testset)
    
    # Check that we have ratings in both sets
    assert trainset.n_ratings > 0
    assert len(testset) > 0

def test_load_and_split_data_test_size(mock_movie_data, mock_rating_data):
    """Test that the test_size parameter works correctly."""
    test_size = 0.3
    trainset, testset = load_and_split_data(mock_movie_data, mock_rating_data, test_size=test_size)
    
    total_ratings = len(testset) + trainset.n_ratings
    
    # Allow for some rounding errors in the split
    assert abs(len(testset) / total_ratings - test_size) < 0.05

def test_load_and_split_data_custom_rating_scale(mock_movie_data, mock_rating_data):
    """Test that the rating_scale parameter is respected."""
    custom_scale = (1, 10)
    
    # Create a new rating file with values in the custom range
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("userId,movieId,rating,timestamp\n")
        f.write("1,1,8.5,1234567890\n")
        f.write("1,2,7.0,1234567891\n")
        f.write("2,1,9.5,1234567893\n")
        custom_rating_file = f.name
    
    try:
        trainset, testset = load_and_split_data(mock_movie_data, custom_rating_file, rating_scale=custom_scale)
        
        # Verify the rating scale was applied correctly
        assert trainset.rating_scale == custom_scale
    finally:
        os.unlink(custom_rating_file)

def test_load_and_split_data_with_missing_values(mock_movie_data):
    """Test that the function handles missing values correctly."""
    # Create data with missing values
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("userId,movieId,rating,timestamp\n")
        f.write("1,1,4.5,1234567890\n")
        f.write("1,2,,1234567891\n")  # Missing rating
        f.write("2,1,3.5,1234567893\n")
        rating_with_missing = f.name
    
    try:
        trainset, testset = load_and_split_data(mock_movie_data, rating_with_missing)
        
        # Only 2 valid ratings in the file
        total_ratings = trainset.n_ratings + len(testset)
        assert total_ratings == 2
    finally:
        os.unlink(rating_with_missing)

def test_load_and_split_data_reproducibility(mock_movie_data, mock_rating_data):
    """Test that the random_state parameter ensures reproducible splits."""
    trainset1, testset1 = load_and_split_data(mock_movie_data, mock_rating_data)
    trainset2, testset2 = load_and_split_data(mock_movie_data, mock_rating_data)
    
    # Since random_state is fixed at 42 in the function, both splits should be identical
    assert len(testset1) == len(testset2)
    
    # Convert tuples to strings for comparison
    testset1_str = sorted([f"{u}-{i}-{r}" for u, i, r in testset1])
    testset2_str = sorted([f"{u}-{i}-{r}" for u, i, r in testset2])
    
    assert testset1_str == testset2_str

def test_load_and_split_data_empty_file():
    """Test that the function handles empty files appropriately."""
    # Create empty data files