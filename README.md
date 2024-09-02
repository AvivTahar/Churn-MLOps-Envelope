# Churn MLOps Envelope

This project is designed to deploy and manage a churn prediction machine learning model in a production environment. The application provides an envelope around a pre-trained model (stored as a .pkl file) to facilitate its integration and use in production systems.

## Features

Dockerized Environment: The entire setup, including model serving, monitoring with Prometheus, and visualization with Grafana, is containerized using Docker Compose.
Model Serving: A FastAPI application serves the model and handles prediction requests.
Monitoring and Visualization: Prometheus and Grafana are integrated for monitoring the application’s performance and visualizing metrics.

## Getting Started

Prerequisites

Ensure you have Docker and Docker Compose installed on your machine.
Clone this repository.

## Usage

Follow the steps below to set up and run the application:

1. Clone the repository:
 
		git clone https://github.com/your-github-username/Churn-MLOps-Envelope.git

2. Navigate to the project directory:
		cd Churn-MLOps-Envelope

3. Start the Docker containers:
		docker compose up -d

  This will start the FastAPI app, Prometheus, and Grafana containers in detached mode.

4. Send a prediction request:

Navigate to the external_data_app directory to send a prediction request using the provided script:

		cd external_data_app
		python prediction_request.py --file_name data/{DATASET_NAME} --from_percentage (optional) --to_percentage (optional) --url http://localhost:8080/predict/

Replace {DATASET_NAME} with the name of your dataset file, and optionally specify the percentage range of data to use.

Monitoring

	Grafana: Access Grafana by navigating to http://localhost:3000 in your browser. The default credentials are:
	Username: admin
	Password: admin
	Prometheus: Access Prometheus at http://localhost:9090 to view raw metrics.


  
