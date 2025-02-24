<<<<<<< HEAD
# Sentiment-Analysis
=======
# Sentiment Analysis Project

## Overview

This project implements a sentiment analysis application that classifies text into various sentiment categories. The application uses a machine learning model and TF-IDF vectorization to predict the sentiment of user-input text.

## Project Structure

├── main.py # The main script to run the Streamlit application.
├── model.h5 # Trained Keras model for sentiment classification.
├── tfidf.pkl # Serialized TF-IDF vectorizer for text feature extraction.
├── README.md # This file, containing project details.
├── requirements.txt # List of dependencies required to run the project.
└── data/ # Folder containing data files used for training/testing.


## Installation

1. **Clone the repository** or **download the project files**.

2. Navigate to the project directory:

   ```bash
   cd path/to/your/project

Install the required dependencies:
	pip install -r requirements.txt

Usage
Start the Streamlit application:
	streamlit run main.py

Files
main.py: Contains the Streamlit application code that allows users to input text and get sentiment predictions.
model.h5: The pre-trained Keras model for sentiment classification.
tfidf.pkl: The serialized TF-IDF vectorizer used for transforming text data.
requirements.txt: A list of Python packages required to run the project.

Data
The data/ directory should contain any necessary data files for training or testing the model. Ensure that data files are correctly referenced in your code if they are used.
>>>>>>> a28f3a0 (Initial commit - Sentiment Analysis project)
