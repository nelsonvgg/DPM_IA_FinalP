import pytest
import pandas as pd
import pickle
from unittest.mock import MagicMock, patch # Using unittest.mock, standard library
import os
import sys

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from predict import load_model, recommend_movies 

# --- Fixtures ---

@pytest.fixture
def dummy_model_path(tmp_path):
    """Creates a dummy pickle file in a temporary directory."""
    model_content = {"type": "dummy", "parameters": [1, 2, 3]}
    file_path = tmp_path / "dummy_model.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(model_content, f)
    return str(file_path) # Return path as string

@pytest.fixture
def mock_movies_df():
    """Returns a sample movies DataFrame."""
    data = {
        'movieId': [1, 2, 3, 4, 5, 6],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E', 'Movie F'],
        'genres': ['Action', 'Comedy', 'Drama', 'Action|Comedy', 'Drama|Thriller', 'Comedy|Romance']
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_ratings_df():
    """Returns a sample ratings DataFrame."""
    data = {
        'userId': [1, 1, 2, 2, 1, 3],
        'movieId': [1, 3, 2, 4, 5, 1],
        'rating': [5.0, 4.0, 3.0, 4.5, 2.0, 5.0],
        'timestamp': [964982703, 964982703, 964982703, 964982703, 964982703, 964982703]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_surprise_model():
    """Creates a mock model object mimicking Surprise's SVD/KNN."""
    model = MagicMock()
    # Define the behavior of the predict method
    # It should return an object with an 'est' attribute (like Surprise Prediction)
    def mock_predict(user_id, item_id):
        # Simple predictable estimation logic for testing
        prediction = MagicMock()
        # Make prediction dependent on item_id for sorting tests
        prediction.est = 4.0 - (item_id / 10.0) if user_id == 1 else 3.0 # Example logic
        return prediction

    model.predict.side_effect = mock_predict
    return model

# --- Test Cases ---

# Tests for load_model
def test_load_model_success(dummy_model_path, capsys):
    """Tests successful loading of a model."""
    loaded_model = load_model(dummy_model_path)
    assert loaded_model == {"type": "dummy", "parameters": [1, 2, 3]}
    captured = capsys.readouterr()
    assert f"Model loaded successfully from {dummy_model_path}" in captured.out

def test_load_model_file_not_found():
    """Tests behavior when the model file does not exist."""
    non_existent_path = "D:/non_existent_path/model.pkl"
    with pytest.raises(FileNotFoundError):
        load_model(non_existent_path)

# Tests for recommend_movies
def test_recommend_movies_success(mock_surprise_model, mock_movies_df, mock_ratings_df):
    """Tests successful movie recommendation."""
    user_id = 1
    num_recommendations = 2

    # Movies rated by user 1: 1, 3, 5
    # Movies not rated by user 1: 2, 4, 6
    # Expected predictions for user 1 (using mock_predict logic):
    # Movie 2: 4.0 - (2/10) = 3.8
    # Movie 4: 4.0 - (4/10) = 3.6
    # Movie 6: 4.0 - (6/10) = 3.4
    # Expected top 2: Movie 2, Movie 4

    recommendations = recommend_movies(user_id, mock_surprise_model, mock_movies_df, mock_ratings_df, num_recommendations)

    assert isinstance(recommendations, pd.DataFrame)
    assert len(recommendations) == num_recommendations
    assert list(recommendations.columns) == ['movieId', 'title', 'predicted_rating']
    # Check if the correct movies are recommended (based on mock prediction logic)
    assert recommendations.iloc[0]['movieId'] == 2 # Highest predicted rating (3.8)
    assert recommendations.iloc[1]['movieId'] == 4 # Second highest (3.6)
    # Check predicted ratings (allow for float precision issues)
    assert recommendations.iloc[0]['predicted_rating'] == pytest.approx(3.8)
    assert recommendations.iloc[1]['predicted_rating'] == pytest.approx(3.6)
    # Ensure already rated movies are not recommended
    rated_movie_ids = mock_ratings_df[mock_ratings_df['userId'] == user_id]['movieId'].tolist()
    assert not any(rec_id in rated_movie_ids for rec_id in recommendations['movieId'])

def test_recommend_movies_user_not_found(mock_surprise_model, mock_movies_df, mock_ratings_df):
    """Tests behavior when the user ID does not exist in ratings."""
    user_id = 99 # This user ID is not in mock_ratings_df
    with pytest.raises(ValueError, match=f"User ID {user_id} does not exist"):
        recommend_movies(user_id, mock_surprise_model, mock_movies_df, mock_ratings_df)

def test_recommend_movies_fewer_available_than_requested(mock_surprise_model, mock_movies_df, mock_ratings_df):
    """Tests when fewer unrated movies are available than requested."""
    user_id = 1
    num_recommendations = 5 # Request more than available unrated (3)

    recommendations = recommend_movies(user_id, mock_surprise_model, mock_movies_df, mock_ratings_df, num_recommendations)

    # Should return all unrated movies (movie IDs 2, 4, 6)
    assert len(recommendations) == 3
    assert sorted(list(recommendations['movieId'])) == [2, 4, 6]
    # Check sorting still works
    assert recommendations.iloc[0]['movieId'] == 2 # Highest predicted rating (3.8)
    assert recommendations.iloc[1]['movieId'] == 4 # Second highest (3.6)
    assert recommendations.iloc[2]['movieId'] == 6 # Third highest (3.4)

def test_recommend_movies_all_movies_rated(mock_surprise_model, mock_movies_df, mock_ratings_df):
    """Tests when the user has rated all available movies."""
    user_id = 1
    # Modify ratings so user 1 rated all movies
    all_movie_ids = mock_movies_df['movieId'].tolist()
    new_ratings = []
    for mid in all_movie_ids:
         new_ratings.append({'userId': 1, 'movieId': mid, 'rating': 4.0, 'timestamp': 964982703})
    # Add ratings for other users too to keep the structure
    new_ratings.extend([
        {'userId': 2, 'movieId': 2, 'rating': 3.0, 'timestamp': 964982703},
        {'userId': 3, 'movieId': 1, 'rating': 5.0, 'timestamp': 964982703}
    ])
    modified_ratings_df = pd.DataFrame(new_ratings)

    recommendations = recommend_movies(user_id, mock_surprise_model, mock_movies_df, modified_ratings_df)

    assert isinstance(recommendations, pd.DataFrame)
    assert recommendations.empty
    assert list(recommendations.columns) == ['movieId', 'title', 'predicted_rating']


def test_recommend_movies_num_recommendations_zero(mock_surprise_model, mock_movies_df, mock_ratings_df):
    """Tests when num_recommendations is 0."""
    user_id = 1
    num_recommendations = 0

    recommendations = recommend_movies(user_id, mock_surprise_model, mock_movies_df, mock_ratings_df, num_recommendations)

    assert isinstance(recommendations, pd.DataFrame)
    assert recommendations.empty
    # Even if empty, the columns should ideally be preserved if pandas handles it that way
    # Depending on implementation, this might vary, but checking expected columns is good
    # Adjust if necessary based on actual empty DataFrame behavior in your pandas version
    # If .head(0) preserves columns:
    assert list(recommendations.columns) == ['movieId', 'title', 'predicted_rating']
    # If .head(0) on an empty frame results in different columns: assert recommendations.empty is sufficient