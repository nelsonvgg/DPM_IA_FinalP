from load_data import load_and_split_data
from surprise import SVD, KNNBasic, KNNWithMeans
from surprise import Dataset
from surprise.model_selection import cross_validate
import pickle

filepath_movies = './data/movies.csv'
filepath_ratings = './data/ratings.csv'
# Load the data using the load_and_split_data function
trainset, testset = load_and_split_data(filepath_movies, filepath_ratings, test_size=0.2, rating_scale=(0.5, 5.0))
# Validate the structure of the trainset and testset
print("Trainset and testset loaded successfully.")


def train_svd_model(trainset, model_save_path="D:/GitHub/DPM_IA_Mid-Term/models/SVD_model.pkl"):
    model = SVD(random_state=42) # Create an SVD model instance
    model.fit(trainset) # Fit the model to the training set
    print("SVD model trained successfully.")
    # Save the model to disk    
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print("SVD model saved successfully.")
    return model
svd_model = train_svd_model(trainset)

def train_knn_model(trainset, model_save_path="D:/GitHub/DPM_IA_Mid-Term/models/KNN_model.pkl"):
    sim_options = { # compute similarities between items
    "name": "cosine", # cosine similarity
    "user_based": True,  # compute similarities between users
    }   
    model = KNNBasic(sim_options=sim_options)  # Create an KNN model instance
    model.fit(trainset) # Fit the model to the training set
    print("KNN model trained successfully.")
    # Save the model to disk
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print("KNN model saved successfully.")
    return model

knn_model = train_knn_model(trainset)
