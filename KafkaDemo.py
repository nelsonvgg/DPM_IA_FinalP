import os
from datetime import datetime
from json import dumps, loads
from time import sleep
from random import randint
import random
from kafka import KafkaProducer

def generate_log_entry():
    """Generate a log entry with expected consumer format."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_id = f"user_{randint(1, 100)}"    
    request_type = random.choice(["action", "drama", "comedy", "horror", "romance"])
    status_code = random.choice([200, 400, 404, 500])
    recommendations = random.choice(["p1", "p2", "p3", "p4", "p5"])
    latency = randint(50, 500)  # Latency in milliseconds
    # log format: user_id,timestamp,request_type,status_code,latency
    log_entry = {
        "time": timestamp,
        "userid": user_id,        
        "recommendation request": request_type,
        "status": status_code,
        "result": recommendations,
        "responsetime": latency,
    }
    return log_entry

def write_log_to_kafka():
    """Write log entries to Kafka topic."""
    producer = KafkaProducer(bootstrap_servers=["localhost:9092"],
                            value_serializer=lambda x: dumps(x).encode('utf-8'))
    topic = "movielogN"
    while True:
        log_entry = generate_log_entry()
        producer.send(topic=topic, value=log_entry)
        print(f"Produced log entry: {log_entry}")
        sleep(1)  # Adjust the sleep time as needed

# def read_log_from_kafka():
#     """Read log entries from Kafka topic."""
#     consumer = KafkaConsumer(
#         "movielogN",
#         bootstrap_servers=["localhost:9092"],
#         auto_offset_reset="earliest",
#         group_id="movielogN_group",
#         enable_auto_commit=True,
#         value_deserializer=lambda x: loads(x.decode('utf-8'))
    # )
    # for message in consumer:
    #     log_entry = message.value
    #     print(f"Consumed log entry: {log_entry}")
    #     # Process the log entry as needed
    #     # For example, incrementing metrics or storing in a database
    #     # Here we just print it for demonstration   

def main():
    """Main function to start the Kafka producer and consumer."""
    # Start writing logs to Kafka
    write_log_to_kafka()
    # Uncomment the following line to start reading logs from Kafka
    # read_log_from_kafka()

if __name__ == "__main__":
    # Start writing logs to Kafka
    write_log_to_kafka()