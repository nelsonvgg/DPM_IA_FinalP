pipeline {
    agent any

    environment {        
        PYTHONPATH = "${WORKSPACE}"
        PROJECT_NAME = "FinalProject"        
    }

    stages {
        stage('Build') {
            steps {
                echo 'Installing dependencies'
                sh '''#!/bin/bash                
                source /var/jenkins_home/miniconda3/etc/profile.d/conda.sh
                conda init
                conda activate
                python --version
                conda install --file requirements.txt -c conda-forge --yes
                conda list
                conda deactivate
                '''
            }
        }
        
        stage('Run Unit Tests and generate test report') {
            steps {
                echo 'Running Unit Tests'
                sh '''#!/bin/bash
                source /var/jenkins_home/miniconda3/etc/profile.d/conda.sh
                conda init
                conda activate
                cd tests
                python run_tests_with_coverage.py 
                conda deactivate
                '''
            }
        }

        stage('Run Offline Evaluation') {
            steps {
                echo 'Running Offline Evaluation'
                sh '''#!/bin/bash
                source /var/jenkins_home/miniconda3/etc/profile.d/conda.sh
                conda init
                conda activate
                cd evaluation
                python offline_evaluation.py 
                conda deactivate
                '''
            }
        }

        stage('Run Online Evaluation') {
            steps {
                echo 'Running Online Evaluation'
                sh '''#!/bin/bash
                source /var/jenkins_home/miniconda3/etc/profile.d/conda.sh
                conda init
                conda activate
                cd evaluation
                python online_evaluation.py
                conda deactivate
                '''
            }
        }

        stage('Run Schema Enforcement') {
            steps {
                echo 'Running Schema Enforcement'
                sh '''#!/bin/bash
                source /var/jenkins_home/miniconda3/etc/profile.d/conda.sh
                conda init
                conda activate
                cd evaluation
                python schema_enforcement.py
                conda deactivate
                '''
            }
        }

        stage('Run Data Drift') {
            steps {
                echo 'Running Data Drift'
                sh '''#!/bin/bash
                source /var/jenkins_home/miniconda3/etc/profile.d/conda.sh
                conda init
                conda activate
                cd evaluation
                python data_drift.py
                conda deactivate
                '''
            }
        }        
    }

    post {
        success {                                    
            echo 'Pipeline completed successfully!'
        }
        failure {
        echo 'Pipeline failed.'
        }
    }
}
