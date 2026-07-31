End-to-End Energy Data Forecasting Pipeline
An automated, end-to-end pipeline designed to predict day-ahead electricity market prices. The primary focus of this project is a robust, production-ready architecture that handles automated data ingestion, dynamic feature engineering, and inference.



Pipeline Workflow
Data Ingestion: Automatically extracts historical archives and fetches the latest external data sources.

Feature Engineering: Dynamically updates model inputs using incoming streams for solar generation, wind demand, and total load, alongside self-generated auxiliary features.

Day-Ahead Prediction: Automatically processes generation and load forecasts to output next-day electricity prices.

Quick Start & Installation
To set up the project environment and run the pipeline, open your project folder in your terminal and follow these steps:

1. Create Virtual Environment
Run the following command to create an isolated virtual environment named .venv:

Bash


python -m venv .venv
(Note: On some systems, you may need to use python3 -m venv .venv or py -m venv .venv.)

2. Activate the Virtual Environment
Before installing packages, activate the environment based on your operating system:

Windows (PowerShell):

PowerShell


.\.venv\Scripts\Activate.ps1
Windows (Command Prompt):

DOS


.venv\Scripts\activate.bat
macOS / Linux:

Bash


source .venv/bin/activate
3. Install Dependencies
Install all the required packages specified in the configuration file:

Bash


pip install -r requirements.txt
(Optional) If you need to explicitly install or update scikit-learn within your active environment:

Bash


pip install scikit-learn
▶️ Usage
To run the entire end-to-end pipeline and generate day-ahead price predictions, you only need to run the application entry point:

Bash


python app.py