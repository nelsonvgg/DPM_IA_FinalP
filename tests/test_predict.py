# Import the function to test using absolute import
import os
import sys
import tempfile
import pytest

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from predict import recommend_movies

class MockPrediction:
    def __init__(self, user_id, item_id, rating):
        self.uid = user_id
        self.iid = item_id
        self.est = rating

@pytest.fixture
def mock_model():
    """Create a mock recommendation model."""
    model = MagicMock()
    
    # Configure the model's predict method to return different ratings
    # based on movie IDs (higher IDs get higher ratings for predictability)
    def mock_predict(user_id, movie_id):
        # Create predictable but varied ratings between 1 and 5
        rating = 1.0 + (movie_id % 5)
        return MockPrediction(user_id, movie_id, rating)
    
    model.predict.side_effect = mock_predict
    return model

@pytest.fixture
def sample_movies():
    """Create a sample movies DataFrame."""
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [f'Movie {i}' for i in range(1, 11)],
        'genres': ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 
                   'Adventure', 'Romance', 'Thriller', 'Fantasy', 'Animation']
    })

@pytest.fixture
def sample_ratings():
    """Create a sample ratings DataFrame."""
    return pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 3, 3, 3],
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8],
        'rating': [4.0, 3.5, 5.0, 2.0, 3.0, 4.5, 3.5, 4.0],