pipeline {
    agent any

    environment {
        PYTHON_CMD = 'python'
        MLFLOW_EXPERIMENT_NAME = 'Customer_Churn_Prediction'
    }

    stages {
        stage('1. Checkout Code') {
            steps {
                echo 'Checking out source code from Git repository...'
                checkout scm
            }
        }

        stage('2. Environment & Dependency Setup') {
            steps {
                echo 'Installing required Python packages from requirements.txt...'
                bat 'pip install --upgrade pip'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('3. Automated Testing (PyTest)') {
            steps {
                echo 'Executing Unit and Integration Tests...'
                bat 'pytest tests/ --junitxml=test-results.xml -v'
            }
        }

        stage('4. Model Training & MLflow Logging') {
            steps {
                echo 'Training ML Model and tracking experiments in MLflow...'
                bat 'python src/train.py'
            }
        }

        stage('5. Deployment / Smoke Test') {
            steps {
                echo 'Verifying production deployment readiness...'
                bat 'python -c "import os; assert os.path.exists(\'models/churn_model.pkl\'), \'Model artifact missing!\'; print(\'Deployment Validation Passed!\')"'
            }
        }
    }

    post {
        always {
            echo 'Archiving test results...'
            junit allowEmptyResults: true, testResults: 'test-results.xml'
        }
        success {
            echo '========================================'
            echo ' Jenkins CI/CD Pipeline Build SUCCESS!  '
            echo ' All tests passed and model logged.     '
            echo '========================================'
        }
        failure {
            echo '========================================'
            echo ' Jenkins CI/CD Pipeline Build FAILED!   '
            echo ' Check console logs and update Jira.    '
            echo '========================================'
        }
    }
}
