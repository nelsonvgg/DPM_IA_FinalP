import os
from surprise import SVD, KNNBasic
import pickle
from load_data import load_and_split_data


def train_svd_model(trainset, model_save_path="./models/SVD_model.pkl"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model = SVD(random_state=42) # Create an SVD model instance
    model.fit(trainset) # Fit the model to the training set
    print("SVD model trained successfully.")
    # Save the model to disk    
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"SVD model saved successfully at {os.path.abspath(model_save_path)}.")
    return model


def train_knn_model(trainset, model_save_path="./models/KNN_model.pkl"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

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
    print(f"KNN model saved successfully at at {os.path.abspath(model_save_path)}.")
    return model
    

# MAIN FUNCTION
def main():    
    # Define file paths for movies and ratings datasets
    current_directory = os.getcwd()    
    filepath_movies = os.path.join(current_directory, 'data/movies.csv')
    filepath_ratings = os.path.join(current_directory, 'data/ratings.csv')    
    # Load the data using the load_and_split_data function
    trainset, _, _, _ = load_and_split_data(filepath_movies, filepath_ratings, test_size=0.2, rating_scale=(0.5, 5.0))    
    # Train the models using the training set
    SVD_model=train_svd_model(trainset,"./models/SVD_model.pkl")
    KNN_model=train_knn_model(trainset,"./models/KNN_model.pkl")
    print("Models has been trained and saved in the 'models' directory.")

if __name__ == "__main__":
    main()
