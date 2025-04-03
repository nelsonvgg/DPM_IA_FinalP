import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

def load_and_split_data(filepath_movies,filepath_ratings, rating_scale=(0.5, 5.0), test_size=0.2):
    """
    Load data from a CSV file, encode it using Surprise library, and split into train and test sets.

    Parameters:
        file_path (str): Path to the CSV file containing the data.
        rating_scale (tuple): The minimum and maximum rating values in the dataset.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        trainset: The training set in Surprise format.
        testset: The test set in Surprise format.
    """
    
    #filepath_movies = './data/movies.csv'
    #filepath_ratings = './data/ratings.csv'
    
    # Load the data using pandas
    try:
        movies = pd.read_csv(filepath_movies, sep=',') 
        ratings = pd.read_csv(filepath_ratings, sep=',')
    except pd.errors.EmptyDataError:
        raise ValueError("No data found")    
    
    # Validate the structure of the movies DataFrame
    expected_movie_columns = ['movieId', 'title', 'genres']
    if not all(column in movies.columns for column in expected_movie_columns):
        raise ValueError(f"Movies file does not have the required columns: {expected_movie_columns}")
    
    # Validate the structure of the ratings DataFrame
    expected_rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
    if not all(column in ratings.columns for column in expected_rating_columns):
        raise ValueError(f"Ratings file does not have the required columns: {expected_rating_columns}")
    
    # Remove row with missing values
    movies.dropna(inplace=True)
    ratings.dropna(inplace=True)
    
    # Define the Reader with the rating scale
    reader = Reader(rating_scale=rating_scale)

    # Load the data into Surprise's Dataset format
    surprise_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Split the data into train and test sets
    trainset, testset = train_test_split(surprise_data, test_size=test_size, random_state=42)
    print("Trainset and testset loaded successfully.")

    return trainset, testset

def load_data(filepath_movies,filepath_ratings):
    """
    Load data from a CSV file

    Parameters:
        file_path (str): Path to the CSV file containing the data.
    
    Returns:
        movie: The movies dataset
        ratings: The ratings dataset
    """
    
    #filepath_movies = './data/movies.csv'
    #filepath_ratings = './data/ratings.csv'
    
     # Load the data using pandas
    try:
        movies = pd.read_csv(filepath_movies, sep=',')
        ratings = pd.read_csv(filepath_ratings, sep=',')
    except pd.errors.EmptyDataError:
        raise ValueError("No data found")
    
    # Validate the structure of the movies DataFrame
    expected_movie_columns = ['movieId', 'title', 'genres']
    if not all(column in movies.columns for column in expected_movie_columns):
        raise ValueError(f"Movies file does not have the required columns: {expected_movie_columns}")
    
    # Validate the structure of the ratings DataFrame
    expected_rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
    if not all(column in ratings.columns for column in expected_rating_columns):
        raise ValueError(f"Ratings file does not have the required columns: {expected_rating_columns}")
    
    # Remove row with missing values
    movies.dropna(inplace=True)
    ratings.dropna(inplace=True)    
    print("Movies and ratings datasets loaded successfully.")

    return movies, ratings