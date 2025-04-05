import os
import sys
import pickle
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from surprise import Dataset, Reader
from collections import defaultdict

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get snapshot from the Kafka topic
def create_random_ratings_df(movies, ratings, num_rows):
    """Create a random ratings DataFrame for the Kafka topic"""
    
    # Get the minimum and maximum user IDs
    min_user_id = ratings['userId'].min()
    max_user_id = ratings['userId'].max()
    
    # Create a dictionary mapping movieId to title for faster lookup
    movie_id_to_title = dict(zip(movies['movieId'], movies['title']))
    
    # Create a dictionary mapping (userId, movieId) to rating for faster lookup
    user_movie_ratings = ratings.set_index(['userId', 'movieId'])['rating'].to_dict()
    
    # Generate random date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years ago
    
    # Lists to store the generated data
    timestamps = []
    user_ids = []
    movie_ids = []
    titles = []
    rating_values = []
    
    rows_created = 0
    while rows_created < num_rows:
        # Generate random userId between min and max
        user_id = random.randint(min_user_id, max_user_id)
        
        # Get all movies rated by this user
        user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
        
        # If the user hasn't rated any movies, try another user
        if len(user_rated_movies) == 0:
            continue
        
        # Select a random movie that the user has rated
        movie_id = np.random.choice(user_rated_movies)
        
        # Get the movie title
        title = movie_id_to_title.get(movie_id, "Unknown")
        
        # Get the rating (which must exist since we filtered for rated movies)
        rating = user_movie_ratings.get((user_id, movie_id))
        
        # Generate a random timestamp
        random_timestamp = start_date + (end_date - start_date) * random.random()
        timestamp_str = random_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to the lists
        timestamps.append(timestamp_str)
        user_ids.append(user_id)
        movie_ids.append(movie_id)
        titles.append(title)
        rating_values.append(rating)
        
        rows_created += 1
    
    # Create the DataFrame
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'userId': user_ids,
        'movieId': movie_ids,
        'title': titles,
        'rating': rating_values
    })
    
    return result_df


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )
        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        av_precisions=sum(prec for prec in precisions.values()) / len(precisions)
        av_recalls=sum(rec for rec in recalls.values()) / len(recalls)
                   
    return av_precisions, av_recalls


def ndcg_at_k(predictions, k=10):
    """Return the NDCG score at k for each user"""    
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    # Calculate NDCG score for each user
    ndcg_scores = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)        
        # Take only top k ratings
        user_ratings_k = user_ratings[:k]        
        # Extract true ratings as relevance scores
        relevance_scores = np.array([true_r for (_, true_r) in user_ratings_k])        
        # Create ideal ranking based on true ratings (sorted by true ratings)
        ideal_ranking = sorted(relevance_scores, reverse=True)        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))        
        # Calculate IDCG (Ideal DCG)
        idcg = np.sum(ideal_ranking / np.log2(np.arange(2, len(ideal_ranking) + 2)))        
        # Calculate NDCG
        ndcg_scores[uid] = dcg / idcg if idcg > 0 else 0
        av_ndcg = np.mean(list(ndcg_scores.values()))
         
    return av_ndcg


# MAIN FUNCTION
def main():
    # Load the dataset
    movies = pd.read_csv('./data/movies.csv')
    ratings = pd.read_csv('./data/ratings.csv')
    
    # Create a snapshot of Kafka DataFrame with 50 rows
    kafka_ratings_df = create_random_ratings_df(movies, ratings, 100)
    
    # Save the DataFrame to a CSV file
    kafka_ratings_df.to_csv('./evaluation/kafka_ratings.csv', index=False)
    
    # Calculate predictions using the trained model 
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(kafka_ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    # Load the trained model
    model_path = "./models/SVD_model.pkl"
    with open(model_path, 'rb') as file:
        algo = pickle.load(file)
    predictions = algo.test(testset)
    print(pd.DataFrame(predictions).head(10))
    k = 10  #  k highest estimated ratings
    threshold = 3.5  # Threshold for relevant items
    # Calculate precision and recall at k   
    av_precisions, av_recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
    # Precision and recall can then be averaged over all users
    print(f"Average Precision@k = {av_precisions}")
    print(f"Average Recall@k = {av_recalls}")
    
    # Calculate NDCG at k
    # NDCG is calculated for each user and then averaged
    av_ndcg = ndcg_at_k(predictions, k=k)
    print(f"Average NDCG@k = {av_ndcg}")

    # Create a DataFrame with the evaluation metrics
    evaluation_metrics_df = pd.DataFrame({
        'Metric': ['Average Precision@k', 'Average Recall@k', 'Average NDCG@k'],
        'Value': [av_precisions, av_recalls, av_ndcg]
    })
    
    # Save the DataFrame to a CSV file
    evaluation_metrics_df.to_csv('./evaluation/online_evaluation.csv', index=False)
    print("Evaluation metrics saved successfully.")

if __name__ == "__main__":
    main()