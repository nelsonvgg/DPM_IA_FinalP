from flask import Flask, render_template, request, jsonify 
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
import pickle
import pandas as pd
from surprise import Dataset, Reader
import logging
import time
import os
from load_data import load_data
from predict import load_model, recommend_movies

###### FLASK API ######
app = Flask(__name__)

# Kafka configuration
KAFKA_TOPIC = "movielogN"  # Set the topic name
KAFKA_BROKER = "127.0.0.1:9092"  # Change if Kafka is on another host
#KAFKA_BROKER = "localhost:9092"  # Change if Kafka is on another host
producer = Producer({'bootstrap.servers': KAFKA_BROKER}) ## Create a Kafka producer

# Load the movies and ratings dataset
movies = "movies.csv"  
ratings = "ratings.csv"
data_path = "D:/GitHub/DPM_IA_Mid-Term/data/" 
movies_path = os.path.join(data_path, "movies.csv")
ratings_path = os.path.join(data_path, "ratings.csv")
movies, ratings = load_data(movies_path, ratings_path)

# Load the trained model from disk
model_name = "SVD_model.pkl"  # The model filename
models_path = "D:/GitHub/DPM_IA_Mid-Term/models/" 
model_path = os.path.join(models_path, model_name)
model = load_model(model_path)


@app.route("/")
def home():
    return render_template('index.html')

# API at /recommend/<int:user_id>
@app.route("/recommend/<int:user_id>", methods=['GET'])
def recomendation(user_id):
    start_time = time.time() # Start time for latency measurement
    try:
        # Check if the user_id exists in the ratings DataFrame
        if user_id not in ratings['userId'].unique():
            return jsonify({'error': f"User ID {user_id} does not exist in the ratings dataset."}), 400
        recommendations = recommend_movies(user_id, model, movies, ratings, num_recommendations=10)
        #print(recommendations)        
        final_time = round((time.time() - start_time) * 1000, 2)  # Calculate latency ms
        # Create a log entry
        log_entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "userid": user_id,
            "recommendation request": request.host,
            "status": 200,
            "result": recommendations.to_dict(orient='records'),
            "responsetime": final_time,
        }
        #print(log_entry)
        # Produce the log entry to Kafka
        producer.produce(KAFKA_TOPIC, key=str(user_id), value=str(log_entry))
        producer.flush()  # Ensure the message is sent
        #return jsonify(recommendations.to_dict(orient='records')), 200

        # Return HTML instead of JSON
        return render_template(
            'recommendations.html', 
            recommendations=recommendations.to_dict(orient='records'),
            user_id=user_id,
            response_time=final_time
        )
           
    except Exception as e:
        # Log the error entry to Kafka  
        #return jsonify({'error': 'Error in processing'}), 500
        return render_template('error.html', error='Error in processing recommendation request'), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8082, debug=True)