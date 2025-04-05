import pytest
import os

if __name__ == "__main__":
    pytest.main([
        "--cov=.",  # Measure coverage for the entire project
        f"--cov-report=xml:{os.path.join('./tests', 'coverage.xml')}",  # Generate coverage report in XML format
        "--cov-report=term",  # Display coverage summary in the terminal
        "--cov-config=tests/.coveragerc",  # Use the coverage configuration file
        "tests/"  # Specify the tests directory
    ])