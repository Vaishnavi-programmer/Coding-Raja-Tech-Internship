# Sentiment Analysis Web Application

This project is a Flask web application that leverages a sentiment analysis model to classify text into sentiment categories (Positive, Neutral, Negative). The sentiment analysis model is based on the Naive Bayes classification algorithm.

## Table of Contents

1. [Introduction](#introduction)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Web Application Details](#web-application-details)

## Introduction

This web application provides a simple interface for users to input text, and it returns the sentiment prediction using a pre-trained Naive Bayes model.

## Folder Structure

- `ML_SA_APP/`
  - `templates/`
    - `index.html`
  - `__init__.py`
  - `app.py`
  - `naive_bayes_model.joblib`
  - `vectorizer.joblib`

## Installation

To run the web application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Vaishnavilily/Coding-Raja-Technologies-Internship.git
   cd your-repository
   ```
2. Install the required dependencies:
   ```bash
   pip install Flask joblib scikit-learn nltk
   ```
3. Download NLTK data:
   ```bash
   python -m nltk.downloader stopwords wordnet punkt
   ```
4. Run the Flask application:
   ```bash
   python ML_SA_APP/app.py
   ```

## Usage
1. Start the Flask application by running python ML_SA_APP/app.py.
2. Open your web browser and go to http://localhost:5000/.
3. Enter a piece of text in the provided input box and click the "Predict" button.
4. The web application will display the predicted sentiment.

## Web Application Details

### 'app.py'
This file contains the Flask web application code. It loads the pre-trained Naive Bayes model and vectorizer, preprocesses the input text, and returns the sentiment prediction.

### 'templates/'
This folder contains the HTML templates used by the Flask application. In particular, index.html is the main page that includes the input form and displays the prediction.

### 'naive_bayes_model.joblib' and 'vectorizer.joblib'
These files contain the pre-trained Naive Bayes model and vectorizer, respectively. They are used by the Flask application for sentiment prediction.
