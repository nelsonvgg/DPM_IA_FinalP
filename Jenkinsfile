pipeline {
    agent any

    environment {
        
        PYTHONPATH = "${WORKSPACE}"
        PROJECT_NAME = "FinalProject"
        TESTING = "1"
        BUILD_NUMBER = "${env.BUILD_NUMBER}" 

    }

    stages {
        stage('Build') {
            steps {
                sh '''
                # Create a virtual environment and activate it
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                deactivate
                '''
            }
        }
        
        stage('Run Unit Tests and generate test report') {
            steps {
                echo 'Running Unit Tests'
                sh '''
                . venv/bin/activate
                cd tests
                python run_tests_with_coverage.py 
                deactivate
                '''
            }
        }

        stage('Run Offline Evaluation') {
            steps {
                echo 'Running Offline Evaluation'
                sh '''
                . venv/bin/activate
                cd evaluation
                python offline_evaluation.py 
                deactivate
                '''
            }
        }

        stage('Run Online Evaluation') {
            steps {
                echo 'Running Online Evaluation'
                sh '''
                . venv/bin/activate
                cd evaluation
                python online_evaluation.py
                deactivate
                '''
            }
        }

        stage('Run Schema Enforcement') {
            steps {
                echo 'Running Schema Enforcement'
                sh '''
                . venv/bin/activate
                cd evaluation
                python schema_enforcement.py
                deactivate
                '''
            }
        }

        stage('Run Data Drift') {
            steps {
                echo 'Running Data Drift'
                sh '''
                . venv/bin/activate
                cd evaluation
                python data_drift.py
                deactivate
                '''
            }
        }        
    }

    post {
        success {
            junit 'report.xml'
            echo 'Pipeline completed successfully!'
        }
        failure {
        echo 'Pipeline failed.'
        }
    }
}
