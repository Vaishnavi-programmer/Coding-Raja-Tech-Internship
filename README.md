# Sentiment Analysis Web Application - `index.html`

The `index.html` file in the `templates` folder is the main HTML template for the Sentiment Analysis web application. It provides the user interface for entering text and receiving sentiment predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Form Structure](#form-structure)
3. [Displaying Predictions](#displaying-predictions)

## Introduction

The `index.html` file serves as the main page of the web application. It contains a simple form where users can input text, submit it, and receive a sentiment prediction. The structure of the HTML file is designed to be user-friendly and straightforward.

## Form Structure

The HTML form in `index.html` includes the following components:

- **Heading (`<h1>)**: Displays the title of the web application, "Sentiment Analysis."

- **Form (`<form>)**: Allows users to input text and submit it for sentiment analysis.

  - **Label (`<label>)**: Describes the input field as "Enter text."

  - **Input (`<input>)**: A text input field where users can type the text to be analyzed.

  - **Button (`<button>)**: A "Predict" button to submit the form.

## Displaying Predictions

After submitting the form, the web application displays the prediction and the input text:

- **Prediction Display (`{% if prediction %})**: If a prediction is available, it displays the following:

  - **Prediction Result (`<p><strong>Prediction:</strong> {{ prediction }}</p>)**: Shows the sentiment prediction.

  - **Input Text (`<p><strong>Input Text:</strong> {{ text }}</p>)**: Displays the original input text.

This structure ensures a clear presentation of the sentiment analysis results for the user.

Feel free to customize the HTML and add styling if needed to enhance the user experience.

