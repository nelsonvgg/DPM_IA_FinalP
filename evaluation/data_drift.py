# Import necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from online_evaluation import create_random_ratings_df


'''
DEFINITION: Data drift is a phenomenon where the statistical properties 
of the model's input data change over time, causing degradation
in model performance. It's a critical challenge in maintaining 
ML systems in production environments. 
'''
def data_drift_kst(snapshot1_path, snapshot2_path, output_file='./evaluation/data_drift_evaluation_kst.txt'):
    # Load data from both files
    df1 = pd.read_csv(snapshot1_path)
    df2 = pd.read_csv(snapshot2_path)
    
    # Cleaning data and extract the rating column
    ratings1 = df1['rating'].dropna()
    ratings2 = df2['rating'].dropna()
        
    # Kolmogorov-Smirnov Test for data drift
    ks_stat, ks_p_value = ks_2samp(ratings1, ratings2)
    
    # Prepare output content
    results = [
        f"Kolmogorov-Smirnov test statistic: {ks_stat:.4f}",
        f"p-value: {ks_p_value:.4f}",
        "Interpretation: "
    ]
    
    if ks_p_value < 0.05:
        results.append("The distributions of ratings between the two files are significantly different (data drift detected).")
    elif 0.05 <= ks_p_value < 0.1:
        results.append("The distributions of ratings between the two files are moderately different (possible data drift).")
    else:
        results.append("No significant data drift detected between the two files' rating distributions.")

    # Print the results to the console
    for line in results:
        print(line)

    # Write the results to the output file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')


def data_drift_rfc(reference_data_path, current_data_path, cv_folds=5, output_file='./evaluation/data_drift_evaluation_rfc.txt'):
    # Load data from both files
    df1 = pd.read_csv(reference_data_path)
    df2 = pd.read_csv(current_data_path)
    
    # Cleaning data and extract the rating column
    reference_data = df1['rating'].dropna()
    current_data = df2['rating'].dropna()

    # Label reference data as 0, current data as 1
    ref_labeled = np.column_stack([reference_data, np.zeros(len(reference_data))])
    current_labeled = np.column_stack([current_data, np.ones(len(current_data))])
    
    # Combine datasets
    combined = np.vstack([ref_labeled, current_labeled])
    X, y = combined[:, :-1], combined[:, -1]
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv_folds)
    
    mean_score = cv_scores.mean()
    
    # Prepare output content
    results = [
        f"Cross Validation Score: {mean_score:.4f}",
        "Interpretation: "
    ]

    if mean_score > 0.8:
        results.append("The distributions of ratings between the files are significantly different (Severe data drift).")
    elif 0.6 < mean_score <= 0.8:
        results.append("The distributions of ratings between the files are moderately different (Moderate data drift).")
    else:
        results.append("The distributions of ratings between the files are similar (No data drift).")
    
    # Print the results to the console
    for line in results:
        print(line)

    # Write the results to the output file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

# Plot the distributions of ratings from both files
def plot_distributions(snapshot1_path, snapshot2_path):
    # Load data from both files
    df1 = pd.read_csv(snapshot1_path)
    df2 = pd.read_csv(snapshot2_path)
    
    # Cleaning data and extract the rating column
    ratings1 = df1['rating'].dropna()
    ratings2 = df2['rating'].dropna()

    plt.figure(figsize=(10, 6))
    sns.histplot(ratings1, kde=True, label='Snapshot1', bins=20, element='step')
    sns.histplot(ratings2, kde=True, label='Snapshot2', bins=20, element='step')
    plt.title("Distribution of Ratings Over Time")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()  # Show the plot

    # Save the plot to a file
    plt.savefig('./evaluation/data_drif_evaluation.png')  # Save plot as PNG
    plt.close()  # Close the plot to free memory


# MAIN FUNCTION 
def main():
    # Define file paths for movies and ratings datasets
    current_directory = os.getcwd()    
    filepath_movies = os.path.join(current_directory, 'data/movies.csv')
    filepath_ratings = os.path.join(current_directory, 'data/ratings.csv')
    # Load the dataset
    movies = pd.read_csv(filepath_movies)
    ratings = pd.read_csv(filepath_ratings)
    
    # Create a first snapshot of Kafka DataFrame with 100 rows
    kafka_ratings_df1 = create_random_ratings_df(movies, ratings, 100)
    # Save the DataFrame to a CSV file
    kafka_ratings_df1.to_csv('./evaluation/kafka_ratings_df1.csv', index=False)
    
    # Create a second snapshot of Kafka DataFrame with 100 rows
    kafka_ratings_df2 = create_random_ratings_df(movies, ratings, 100)
    # Save the DataFrame to a CSV file
    kafka_ratings_df2.to_csv('./evaluation/kafka_ratings_df2.csv', index=False)   

    # Plot the distributions of ratings from both files
    plot_distributions('./evaluation/kafka_ratings_df1.csv', './evaluation/kafka_ratings_df2.csv')

    # Perform data drift evaluation with Kolmogorov-Smirnov test
    data_drift_kst('./evaluation/kafka_ratings_df1.csv', './evaluation/kafka_ratings_df2.csv')
    # Perform data drift evaluation with Random Forest Classifier
    data_drift_rfc('./evaluation/kafka_ratings_df1.csv', './evaluation/kafka_ratings_df2.csv')

if __name__ == "__main__":
    main()