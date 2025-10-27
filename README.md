# movie-sentiment-analysis
A project that predicts whether a movie review is positive or negative using machine learning and natural language processing.

Project Overview
This project implements a Sentiment Analysis System on movie reviews using Natural Language Processing (NLP). The goal is to classify movie reviews into positive or negative categories based on textual content.

Course: Natural Language Processing
Type: Individual Mini Project
Domain: Text Classification & Sentiment Analysis

Problem Statement
Sentiment analysis is widely used to understand audience opinions. This project focuses on:
Processing raw textual data
Extracting features from text
Training classification models
Evaluating performance on unseen data
The model determines whether a given movie review expresses a positive or negative sentiment.

Dataset
Source: IMDb Movie Reviews Dataset
Size: 50,000 labeled reviews
Classes: Positive and Negative
Availability: Built-in within Keras datasets (no external download required)
The dataset is balanced and widely used in academic research for binary sentiment classification.

Technologies & Libraries Used

Programming Language:
Python 3.8+
Core Libraries:
NLTK (Natural Language Toolkit)
Scikit-learn
Pandas & NumPy
Matplotlib & Seaborn
WordCloud
TensorFlow / Keras

NLP Concepts Implemented
Text Preprocessing
Lowercasing
Removal of special characters
Tokenization
Stopword removal
Lemmatization

Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency)
N-gram features (unigrams & bigrams)

Machine Learning Models
Naive Bayes
Logistic Regression

Project Workflow
Load and explore dataset
Clean and preprocess text
Convert processed text into numerical feature vectors

Train classification models
Evaluate accuracy and performance metrics
Compare models and select the best performing one

Test with custom sample reviews

Project Structure
nlp-sentiment-analysis/
│
├── NLP_Sentiment_Analysis_Project.ipynb
├── README.md
├── requirements.txt
└── screenshots


Best Model: Logistic Regression
Achieved Accuracy: Approximately 87%

Visualizations Included
Sentiment distribution graphs
Positive vs negative word clouds
Review length analysis
Model comparison chart
Confusion matrices
These visualizations help understand data patterns and model behavior.

Key Features
End-to-end NLP pipeline
Two classification algorithms implemented
Real-time prediction for custom reviews
Rich visualization support
Well-documented work pipeline

Example Usage
You can input any custom movie review, and the model will return:
Positive or Negative sentiment
Confidence score

Future Improvements
Integration of Word2Vec and GloVe embeddings
Implementation of modern transformer-based models (BERT)
Addition of neutral class
Deployment as a web application
Sarcasm handling
Multi-language support

Learning Outcomes
Through this project, I gained experience with:
Data cleaning and preprocessing for NLP
Converting text into numeric features
Selecting and comparing ML models
Visualizing textual data insights
Improving model accuracy through tuning


Contact

Name: Himanshi Baherwani 

Date: October 27, 2025
Institution: NMIMS indore
Course: Natural Language Processing
