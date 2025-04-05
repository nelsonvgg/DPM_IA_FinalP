# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, accuracy
from surprise import SVD, KNNBasic
from surprise.model_selection import train_test_split
import time
import psutil
import os
import sys

# Add the parent directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_data import load_data, load_and_split_data
from predict import load_model, recommend_movies

# 1. Accuracy Prediction --> RMSE
def calculate_rmse(models, testset):
    #models (dict): A dictionary where keys are model names and values are trained model objects.
    #testset (list): A list of tuples (userId, movieId, actual_rating) from the Surprise library.    
    results = {}
    for model_name, model in models.items():
        # Generate predictions using the model
        predictions = model.test(testset)
        
        # Calculate RMSE using Surprise's accuracy module
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        # Store RMSE and MAE in the results dictionary
        results[model_name] = {"RMSE": rmse, "MAE": mae}
    
    return results

# 2. Training Cost --> Training time, CPU usage and memory usage
def calculate_training_cost(models, trainset):
    #models (dict): A dictionary where keys are model names and values are trained model objects.
    #trainset (Surprise trainset): The training set used to fit the models.
    training_costs = {}
    
    for model_name, model in models.items():
        # Record start time and initial CPU/memory usage
        start_time = time.time()
        initial_cpu = psutil.cpu_percent(interval=None)
        #initial_memory = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
        initial_memory = psutil.virtual_memory().percent  # Convert to MB
        
        # Train the model
        model.fit(trainset)
        
        # Record end time and final CPU/memory usage
        end_time = time.time()
        final_cpu = psutil.cpu_percent(interval=None)
        #final_memory = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
        final_memory = psutil.virtual_memory().percent  # Convert to MB

        # Calculate metrics
        time_taken = end_time - start_time
        cpu_usage = final_cpu - initial_cpu
        memory_usage = final_memory - initial_memory
        
        # Store results
        training_costs[model_name] = {
            "time_taken": time_taken,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }
    
    return training_costs

# 3. Inference Cost --> Inference time
def calculate_inference_cost(models, user_ids, movies, ratings, num_recommendations=10):
    #models (dict): A dictionary where keys are model names and values are trained model objects.
    #user_ids (list): A list of user IDs for whom recommendations are to be generated.
    #movies (DataFrame): A DataFrame containing movie information.
    #ratings (DataFrame): A DataFrame containing user ratings for movies.
    #num_recommendations (int): The number of recommendations to generate for each user.
    
    inference_costs = {}
    
    for model_name, model in models.items():
        # Record start time
        start_time = time.time()
        
        # Generate recommendations for the user
        #recommend_movies(user_id, model, movies, ratings, num_recommendations)
        for user_id in user_ids:
            recommend_movies(user_id, model, movies, ratings, num_recommendations)
                
        # Record end time
        end_time = time.time()
        
        # Calculate time taken
        time_taken = end_time - start_time
        
        # Store results
        inference_costs[model_name] = time_taken
    
    return inference_costs

# 3. Inference Cost --> Troughput
def calculate_throughput(models, user_ids, movies, ratings, num_recommendations=10):
    #models (dict): A dictionary where keys are model names and values are trained model objects.
    #user_ids (list): A list of user IDs for whom recommendations are to be generated.
    #movies (DataFrame): A DataFrame containing movie information.
    #ratings (DataFrame): A DataFrame containing user ratings for movies.
    #num_recommendations (int): The number of recommendations to generate for each user.

    throughput_results = {}
    
    for model_name, model in models.items():
        # Record start time
        start_time = time.time()
        
        # Generate recommendations for the user
        #recommend_movies(user_id, model, movies, ratings, num_recommendations)
        for user_id in user_ids:
            recommend_movies(user_id, model, movies, ratings, num_recommendations)
        
        # Record end time
        end_time = time.time()
        
        # Calculate time taken
        time_taken = end_time - start_time
        
        # Calculate throughput (recommendations per second)
        if time_taken > 0:
            throughput = num_recommendations / time_taken
        else:
            throughput = float('inf')  # Handle edge case where time_taken is 0
        
        # Store results
        throughput_results[model_name] = throughput
    
    return throughput_results

# 4. Model Size and Memory Usage --> Disk size and memory usage
def calculate_model_size_and_memory(models, user_ids, movies, ratings, num_recommendations=10, model_save_path="models/"):
    #models (dict): A dictionary where keys are model names and values are trained model objects.
    #user_ids (list): A list of user IDs for whom recommendations are to be generated.
    #movies (DataFrame): A DataFrame containing movie information.
    #ratings (DataFrame): A DataFrame containing user ratings for movies.
    #num_recommendations (int): The number of recommendations to generate for each user.
    #model_save_path (str): The directory where the models will be saved temporarily for size calculation.

    results = {}

    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)

    for model_name, model in models.items():
        # Save the model to disk temporarily to calculate its size
        model_file_path = os.path.join(model_save_path, f"{model_name}.pkl")
        with open(model_file_path, "wb") as f:
            import pickle
            pickle.dump(model, f)
        
        # Calculate disk size in MB
        disk_size = os.path.getsize(model_file_path) / (1024 ** 2)  # Convert bytes to MB

        # Measure memory usage during inference
        process = psutil.Process(os.getpid())  # Get the current process
        initial_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB

        # Perform inference for all users
        for user_id in user_ids:
            recommend_movies(user_id, model, movies, ratings, num_recommendations)

        final_memory = process.memory_info().rss / (1024 ** 2)  # Memory in MB
        memory_usage = final_memory - initial_memory

        # Store results
        results[model_name] = {
            "disk_size": disk_size,
            "memory_usage": memory_usage
        }

        # Clean up the saved model file
        os.remove(model_file_path)

    return results

# 5. Generate Summary Table
def generate_summary_table(models, trainset, testset, user_ids, movies, ratings, num_recommendations=10, model_save_path="models/"):
   
    # Initialize metrics storage
    summary_data = []

    # Calculate RMSE
    rmse_results = calculate_rmse(models, testset)

    # Calculate training cost
    training_costs = calculate_training_cost(models, trainset)

    # Calculate inference cost
    inference_costs = calculate_inference_cost(models, user_ids, movies, ratings, num_recommendations)

    # Calculate throughput
    throughput_results = calculate_throughput(models, user_ids, movies, ratings, num_recommendations)

    # Calculate disk size and memory usage
    model_metrics = calculate_model_size_and_memory(models, user_ids, movies, ratings, num_recommendations, model_save_path)

    # Combine all metrics into a summary table
    for model_name in models.keys():
        summary_data.append({
            "Model": model_name,
            "RMSE": rmse_results.get(model_name, {}).get("RMSE", None),
            "MAE": rmse_results.get(model_name, {}).get("MAE", None),
            "Training Time (s)": training_costs[model_name]["time_taken"],
            "Training CPU (%)": training_costs[model_name]["cpu_usage"],
            "Training Memory (MB)": training_costs[model_name]["memory_usage"],
            "Inference Time (s)": inference_costs.get(model_name, None),
            "Throughput (rec/sec)": throughput_results.get(model_name, None),
            "Disk Size (MB)": model_metrics[model_name]["disk_size"],
            "Memory Usage (MB)": model_metrics[model_name]["memory_usage"]
        })

    # Convert to pandas DataFrame
    summary_table = pd.DataFrame(summary_data).set_index("Model").T

    return summary_table

# 6. Plotting
def plot_summary_table(summary_table):
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Performance Metrics", fontsize=16)

    # 1. Prediction Accuracy (RMSE and MAE)
    ax1 = axes[0, 0]
    summary_table.loc[["RMSE", "MAE"]].T.plot(kind="bar", ax=ax1)
    ax1.set_title("Prediction Accuracy")
    ax1.set_ylabel("Error")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_xlabel("")  # Set x-axis label to empty
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.2f")

    # 2. Training Cost (Training Time, CPU Usage, Memory Usage)
    ax2 = axes[0, 1]
    summary_table.loc[["Training Time (s)", "Training CPU (%)", "Training Memory (MB)"]].T.plot(kind="bar", ax=ax2)
    ax2.set_title("Training Cost")
    ax2.set_ylabel("Cost")
    ax2.set_xlabel("")  # Set x-axis label to empty
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")    
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.2f")

    # 3. Inference Cost (Inference Time, Throughput)
    ax3 = axes[1, 0]
    summary_table.loc[["Inference Time (s)", "Throughput (rec/sec)"]].T.plot(kind="bar", ax=ax3)
    ax3.set_title("Inference Cost")
    ax3.set_ylabel("Cost")
    ax3.set_xlabel("")  # Set x-axis label to empty
    ax3.legend(loc="upper right")
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f")

    # 4. Storage Cost (Disk Size, Memory Usage)
    ax4 = axes[1, 1]
    summary_table.loc[["Disk Size (MB)", "Memory Usage (MB)"]].T.plot(kind="bar", ax=ax4)
    ax4.set_title("Storage Cost")
    ax4.set_ylabel("Cost (MB)")
    ax4.set_xlabel("")  # Set x-axis label to empty
    ax4.legend(loc="upper right")
    ax4.grid(axis="y", linestyle="--", alpha=0.7)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
    for container in ax4.containers:
        ax4.bar_label(container, fmt="%.2f")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# MAIN FUNCTION
def main():
    # Load the movies and ratings datasets
    movies = "movies.csv"  
    ratings = "ratings.csv"
    data_path = "D:/GitHub/DPM_IA_FinalP/data/" 
    movies_path = os.path.join(data_path, "movies.csv")
    ratings_path = os.path.join(data_path, "ratings.csv")
    trainset, testset, movies, ratings = load_and_split_data(movies_path, ratings_path)

    # Load the SVD trained model from disk
    model_name = "SVD_model.pkl"  # The model filename
    models_path = "D:/GitHub/DPM_IA_FinalP/models/" 
    model_path = os.path.join(models_path, model_name)
    SVD_model = load_model(model_path)
    # Load the KNN trained model from disk
    model_name = "KNN_model.pkl"  # The model filename
    models_path = "D:/GitHub/DPM_IA_FinalP/models/" 
    model_path = os.path.join(models_path, model_name)
    KNN_model = load_model(model_path)
    

    # Define models to evaluate
    models = {
        "SVD model_eval": SVD_model,
        "KNN model_eval": KNN_model
    }

    # Generate summary table
    all_user_ids = ratings['userId'].unique().tolist()
    user_ids = np.random.choice(all_user_ids, size=5, replace=False).tolist()
       
    summary_table = generate_summary_table(models, trainset, testset, user_ids, movies, ratings)
    print(summary_table)
    # Save the summary table to a CSV file
    summary_table.to_csv("./evaluation/offline_evaluation.csv", index=True)

    # Plot the summary table
    plot_summary_table(summary_table)
    # Save the plot to a file
    plt.savefig("./evaluation/offline_evaluation.png")


if __name__ == "__main__":
    main()