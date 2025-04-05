import csv
from datetime import datetime
import os

# --- Input Data Configuration ---
EXPECTED_COLUMNS = 5
EXPECTED_HEADER = ['timestamp', 'userId', 'movieId', 'title', 'rating']
MIN_RATING = 0.5  # Minimum rating value from EDA
MAX_RATING = 5.0  # Maximum rating value from EDA
RATING_INCREMENT = 0.5  # Rating increment from EDA or plataform configuration
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'


def is_valid_rating_increment(value):
    # Checks if a rating is a multiple of RATING_INCREMENT.    
    try:
        inverse_increment = 1 / RATING_INCREMENT
        # Check if the value times the inverse increment has a zero fractional part
        return abs((value * inverse_increment) - round(value * inverse_increment)) < 1e-9
    except ZeroDivisionError:
        return False # Should not happen with RATING_INCREMENT > 0


def validate_ratings_csv(filepath):
    """
    Validates the schema and data quality of a movie ratings CSV file.
    Input:
        filepath (str): The path to the CSV file.
    Returns:
        tuple: A tuple containing:
            - bool: True if the file is valid according to the schema, False otherwise.
            - list: A list of validated data rows (as dictionaries) if validation passes
                    up to the point of failure, or an empty list if header fails.
                    Can be used for further processing if needed.
            - list: A list of error/warning messages found during validation.
    """
    is_valid = True
    validated_data = []
    error_messages = []
    line_num = 0

    if not os.path.exists(filepath):
        error_messages.append(f"ERROR: File not found at path: {filepath}")
        return False, [], error_messages

    try:
        with open(filepath, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)

            # 1. Validate Header
            try:
                header = next(reader)
                line_num += 1
                if header != EXPECTED_HEADER:
                    error_messages.append(
                        f"ERROR: Line {line_num}: Invalid header. "
                        f"Expected {EXPECTED_HEADER}, got {header}"
                    )
                    # Stop validation if header is wrong, as column names are critical
                    return False, [], error_messages
            except StopIteration:
                error_messages.append("ERROR: File is empty.")
                return False, [], error_messages # File is empty

            # 2. Validate Data Rows
            for row in reader:
                line_num += 1
                current_row_valid = True
                row_errors = []

                # Check 2a: Correct number of columns
                if len(row) != EXPECTED_COLUMNS:
                    row_errors.append(
                        f"Line {line_num}: Invalid number of columns. "
                        f"Expected {EXPECTED_COLUMNS}, got {len(row)}. Row data: {row}"
                    )
                    is_valid = False
                    current_row_valid = False
                    # Skip further validation for this malformed row
                    error_messages.extend(row_errors)
                    continue # Move to the next row

                # Unpack row for easier access
                timestamp_str, user_id_str, movie_id_str, title_str, rating_str = row

                # --- Column Validations ---

                # Check 2b: Timestamp validation
                try:
                    datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
                except ValueError:
                    row_errors.append(
                        f"Line {line_num}: Invalid timestamp format for '{timestamp_str}'. "
                        f"Expected '{TIMESTAMP_FORMAT}'."
                    )
                    current_row_valid = False

                # Check 2c: User ID validation
                user_id = None
                try:
                    user_id = int(user_id_str)
                    if user_id <= 0:
                        row_errors.append(
                            f"Line {line_num}: Invalid userId '{user_id_str}'. "
                            f"Must be a positive integer."
                        )
                        current_row_valid = False
                except (ValueError, TypeError):
                    row_errors.append(
                        f"Line {line_num}: Invalid userId '{user_id_str}'. "
                        f"Must be an integer."
                    )
                    current_row_valid = False

                # Check 2d: Movie ID validation
                movie_id = None
                try:
                    movie_id = int(movie_id_str)
                    if movie_id <= 0:
                        row_errors.append(
                            f"Line {line_num}: Invalid movieId '{movie_id_str}'. "
                            f"Must be a positive integer."
                        )
                        current_row_valid = False
                except (ValueError, TypeError):
                    row_errors.append(
                        f"Line {line_num}: Invalid movieId '{movie_id_str}'. "
                        f"Must be an integer."
                    )
                    current_row_valid = False

                # Check 2e: Title validation
                if not title_str or title_str.isspace():
                    row_errors.append(f"Line {line_num}: Title cannot be empty or whitespace.")
                    current_row_valid = False
                # Optional: Check if title looks reasonable (e.g., not excessively long)
                # if len(title_str) > 500: # Example limit
                #    error_messages.append(f"WARNING: Line {line_num}: Title seems very long ({len(title_str)} chars).")

                # Check 2f: Rating validation
                rating = None
                try:
                    rating = float(rating_str)
                    if not (MIN_RATING <= rating <= MAX_RATING):
                        row_errors.append(
                            f"Line {line_num}: Invalid rating '{rating_str}'. "
                            f"Must be between {MIN_RATING} and {MAX_RATING} (inclusive)."
                        )
                        current_row_valid = False
                    elif not is_valid_rating_increment(rating):
                         row_errors.append(
                            f"Line {line_num}: Invalid rating increment '{rating_str}'. "
                            f"Must be in increments of {RATING_INCREMENT}."
                        )
                         current_row_valid = False

                except (ValueError, TypeError):
                    row_errors.append(
                        f"Line {line_num}: Invalid rating '{rating_str}'. "
                        f"Must be a number."
                    )
                    current_row_valid = False

                # --- Consolidate Row Results ---
                if not current_row_valid:
                    is_valid = False
                    error_messages.extend(row_errors)
                else:
                    # If row is valid, add parsed data (optional, but useful)
                    validated_data.append({
                        'timestamp': timestamp_str, # Keep original validated string or parse to datetime
                        'userId': user_id,
                        'movieId': movie_id,
                        'title': title_str.strip(), # Store stripped title
                        'rating': rating
                    })

    except FileNotFoundError:
        # This case is handled at the beginning, but kept for robustness
        error_messages.append(f"ERROR: File not found at path: {filepath}")
        return False, [], error_messages
    except Exception as e:
        # Catch unexpected errors during file processing
        error_messages.append(f"ERROR: An unexpected error occurred processing line {line_num}: {e}")
        return False, validated_data, error_messages # Return partial data if desired

    return is_valid, validated_data, error_messages

# --- Example Usage ---
if __name__ == "__main__":
    # EXAMPLE WITH kafka_ratings.csv
    print("\n--- Validating actual file: kafka_ratings.csv ---")
    current_directory = os.getcwd()    
    actual_file_path = os.path.join(current_directory,'evaluation/kafka_ratings.csv') # Assuming it's in the same directory
    if os.path.exists(actual_file_path):
        is_valid, data, errors = validate_ratings_csv(actual_file_path)
        if is_valid:
            print("\nValidation PASSED.")
            print(f"Processed {len(data)} valid rows.")
        else:
            print("\nValidation FAILED.")
            print("Errors found:")
            for msg in errors:
                print(f"- {msg}")
    else:
         print(f"\nActual file '{actual_file_path}' not found in the current directory.\n")

    # EXAMPLE WITH test file
    print(f"--- Validating file: kafka_ratings_test.csv ---")
    current_directory = os.getcwd()        
    test_file_path = os.path.join(current_directory,'evaluation/kafka_ratings_test.csv')
    if os.path.exists(test_file_path):
        is_valid, data, errors = validate_ratings_csv(test_file_path)
        if is_valid:
            print("\nValidation PASSED.")
            print(f"Processed {len(data)} valid rows.")
        else:
            print("\nValidation FAILED.")
            print("Errors found:")
            for msg in errors:
                print(f"- {msg}")
    else:
         print(f"\nTest file '{test_file_path}' not found in the current directory.\n")
