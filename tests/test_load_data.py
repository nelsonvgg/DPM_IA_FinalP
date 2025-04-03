# Import the function to test using absolute import
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split


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
        f.write("6,Movie 6,Action|Drama\n")  
        f.write("7,Movie 7,Action|Comedy|Drama\n")      
        f.write("8,Movie 8,Action|Comedy|Drama|Romance\n")
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
        f.write("2,4,2.5,1234567893\n")
        f.write("2,5,3.5,1234567894\n")
        f.write("3,1,4.0,1234567895\n")
        f.write("3,2,4.5,1234567896\n")
        f.write("3,3,2.0,1234567897\n")
        temp_file = f.name
    
    yield temp_file
    os.unlink(temp_file)  # Delete the temporary file

# Test the load_and_split_data function directly
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
    test_size = 0.25
    trainset, testset = load_and_split_data(mock_movie_data, mock_rating_data, test_size=test_size)
    #print(len(testset))
    #print(trainset.n_ratings)

    total_ratings = len(testset) + trainset.n_ratings
    #print(total_ratings)
    
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
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as movie_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as rating_file:
        empty_movie_file = movie_file.name
        empty_rating_file = rating_file.name

    try:
        # Attempt to load and split data from empty files
        with pytest.raises(ValueError, match="No data found"):
            load_and_split_data(empty_movie_file, empty_rating_file)
    finally:
        os.unlink(empty_movie_file)
        os.unlink(empty_rating_file)
        
# Test the load_data function directly
def test_load_data_basic(mock_movie_data, mock_rating_data):
    """Test that load_data returns DataFrames with the correct structure."""
    movies, ratings = load_data(mock_movie_data, mock_rating_data)
    
    # Check that the returned objects are pandas DataFrames
    assert isinstance(movies, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)
    
    # Check that the DataFrames contain the expected columns
    assert all(col in movies.columns for col in ['movieId', 'title', 'genres'])
    assert all(col in ratings.columns for col in ['userId', 'movieId', 'rating', 'timestamp'])
    
    # Check that the DataFrames contain the expected number of rows
    assert len(movies) == 8  # Based on the mock_movie_data fixture
    assert len(ratings) == 8  # Based on the mock_rating_data fixture

def test_load_data_handles_missing_values(mock_movie_data, mock_rating_data):
    """Test that load_data properly handles missing values."""
    # Create data with missing values
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("movieId,title,genres\n")
        f.write("1,Movie 1,Action\n")
        f.write("2,,Comedy\n")  # Missing title
        f.write("3,Movie 3,Drama\n")
        movie_with_missing = f.name
        
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("userId,movieId,rating,timestamp\n")
        f.write("1,1,4.5,1234567890\n")
        f.write("1,2,,1234567891\n")  # Missing rating
        f.write("2,1,3.5,1234567893\n")
        rating_with_missing = f.name
    
    try:
        movies, ratings = load_data(movie_with_missing, rating_with_missing)
        
        # Check that rows with missing values were dropped
        assert len(movies) == 2  # One row was dropped
        assert len(ratings) == 2  # One row was dropped
        
        # Verify that the missing data is not in the DataFrames
        assert not movies['title'].isna().any()
        assert not ratings['rating'].isna().any()
    finally:
        os.unlink(movie_with_missing)
        os.unlink(rating_with_missing)

def test_load_data_empty_file():
    """Test that load_data handles empty files appropriately."""
    # Create empty data files with only headers
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("movieId,title,genres\n")
        empty_movie_file = f.name
        
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("userId,movieId,rating,timestamp\n")
        empty_rating_file = f.name
    
    try:
        movies, ratings = load_data(empty_movie_file, empty_rating_file)
        
        # Check that we get empty DataFrames
        assert len(movies) == 0
        assert len(ratings) == 0
        
        # But they should still have the correct columns
        assert all(col in movies.columns for col in ['movieId', 'title', 'genres'])
        assert all(col in ratings.columns for col in ['userId', 'movieId', 'rating', 'timestamp'])
    finally:
        os.unlink(empty_movie_file)
        os.unlink(empty_rating_file)

def test_load_data_nonexistent_file(mock_movie_data, mock_rating_data):
    """Test that load_data raises an appropriate error for nonexistent files."""
    nonexistent_file = "/path/to/nonexistent/file.csv"
    
    # Should raise FileNotFoundError when trying to read a file that doesn't exist
    with pytest.raises(FileNotFoundError):
        load_data(nonexistent_file, mock_rating_data)
    
    with pytest.raises(FileNotFoundError):
        load_data(mock_movie_data, nonexistent_file)

def test_load_data_file_with_incorrect_format(mock_rating_data):
    """Test that load_data handles files with incorrect format."""
    # Create a file with incorrect CSV format
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        f.write("This FILE is not a valid CSV format\n")
        f.write("No commas in THIS FILE\n")        
        invalid_file = f.name    
    try:
        # Should raise an error when parsing an incorrectly formatted file
        with pytest.raises(ValueError):
            load_data(invalid_file, mock_rating_data)
    finally:
        os.unlink(invalid_file)

def test_load_data_integration_with_surprise():
    """Test that data loaded with load_data can be used with Surprise functions."""
    # Create temporary files with sample data
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as movie_file:
        movie_file.write("movieId,title,genres\n")
        movie_file.write("1,Test Movie 1,Action\n")
        movie_file.write("2,Test Movie 2,Comedy\n")
        test_movie_file = movie_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as rating_file:
        rating_file.write("userId,movieId,rating,timestamp\n")
        rating_file.write("1,1,4.5,1234567890\n")
        rating_file.write("2,1,3.0,1234567891\n")
        rating_file.write("1,2,5.0,1234567892\n")
        test_rating_file = rating_file.name
    
    try:
        # Load the data using load_data
        _, ratings = load_data(test_movie_file, test_rating_file)
        
        # Try to convert to Surprise format - this verifies that the data loaded
        # by load_data can be used with Surprise functionality
        reader = Reader(rating_scale=(0.5, 5.0))
        surprise_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset = surprise_data.build_full_trainset()
        
        # Verify that the data was correctly loaded into Surprise
        assert trainset.n_users == 2
        assert trainset.n_items == 2
        assert trainset.n_ratings == 3
    finally:
        os.unlink(test_movie_file)
        os.unlink(test_rating_file)

def test_load_data_preserves_types(mock_movie_data, mock_rating_data):
    """Test that load_data preserves appropriate data types."""
    movies, ratings = load_data(mock_movie_data, mock_rating_data)
    
    # Check data types
    assert pd.api.types.is_integer_dtype(movies['movieId'])
    assert pd.api.types.is_string_dtype(movies['title'])
    assert pd.api.types.is_string_dtype(movies['genres'])
    
    assert pd.api.types.is_integer_dtype(ratings['userId'])
    assert pd.api.types.is_integer_dtype(ratings['movieId'])
    assert pd.api.types.is_numeric_dtype(ratings['rating'])
    assert pd.api.types.is_integer_dtype(ratings['timestamp'])
