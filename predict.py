import os
import pickle
from load_data import load_data

def load_model(model_path="./models/SVD_model.pkl"):
    """
    Load a trained model from the specified path.
    Parameters:
        model_path (str): Path to the saved model file.
    Returns:
        model: The loaded model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {model_path}.")
    return model


def recommend_movies(user_id, model, movies, ratings, num_recommendations=10):
    # Check if the user_id exists in the ratings DataFrame
    if user_id not in ratings['userId'].unique():
        raise ValueError(f"User ID {user_id} does not exist in the ratings dataset.")
    # Filter out movies already rated by the user
    rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    unique_movies = movies[~movies['movieId'].isin(rated_movies)].copy()
    # Predict ratings for the remaining movies
    unique_movies['predicted_rating'] = unique_movies['movieId'].apply(lambda x: model.predict(user_id, x).est)
    # Sort movies by predicted rating and select top N recommendations
    recommendations = unique_movies.sort_values(by='predicted_rating', ascending=False).head(num_recommendations)
    #print(f"Top {num_recommendations} recommendations for user {user_id}:")
    #print(recommendations[['movieId', 'title', 'predicted_rating']])

    return recommendations[['movieId', 'title', 'predicted_rating']]

# MAIN FUNCTION
def main():
    # Define file paths for movies and ratings datasets
    current_directory = os.getcwd()    
    filepath_movies = os.path.join(current_directory, 'data/movies.csv')
    filepath_ratings = os.path.join(current_directory, 'data/ratings.csv')
    # Load the movies and ratings dataset
    current_directory = os.getcwd() 
    movies, ratings = load_data(filepath_movies, filepath_ratings)
    # Reccommendation with SVD model
    model_path = "./models/SVD_model.pkl"
    model = load_model(model_path)
    user_id = 1  # Example user ID
    recommendations = recommend_movies(user_id, model, movies, ratings, num_recommendations=5)
    print(f"Top 5 recommendations for user {user_id} using SVD_model:\n")
    print(f"{recommendations[['movieId', 'title', 'predicted_rating']]}\n")
    # Reccommendation with KNN model
    model_path = "./models/KNN_model.pkl"
    model = load_model(model_path)
    user_id = 1  # Example user ID
    recommendations = recommend_movies(user_id, model, movies, ratings, num_recommendations=5)
    print(f"Top 5 recommendations for user {user_id} using KNN_model:\n")
    print(f"{recommendations[['movieId', 'title', 'predicted_rating']]}\n")

if __name__ == "__main__":
    main()

    