>>Text Classification of News Article using NLP
  This project implements a text classification model for categorizing news articles into different categories using 
  Natural   Language Processing (NLP) techniques.

>>Overview
  This repository contains two main components:

  Jupyter Notebook (news classification.ipynb): Contains the code for training various machine learning models on the 
  BBC News dataset, evaluating their performance, and saving the best model.
  Tkinter GUI Application (news.py): Provides a user interface for inputting news article text and obtaining the 
  predicted category using the trained model.
  
>>Files
  news classification.ipynb: Jupyter notebook for model training, evaluation, and hyperparameter tuning.
  news.py: Tkinter-based GUI application for news article classification.
  BBC News.csv: Dataset containing labeled news articles used for training and testing.

>>Dependencies

  pandas
  matplotlib
  seaborn
  nltk
  scikit-learn
  wordcloud
  tkinter (for GUI)
  joblib (for model persistence)

>>Steps
  Open news News_classification.ipynb in Jupyter Notebook(or google colab).
  Execute the cells to preprocess the dataset, train multiple models (Logistic Regression, Random Forest, Decision Tree,   K-Nearest Neighbors, Gaussian Naive Bayes), and evaluate their performance.
  Tune hyperparameters for selected models using GridSearchCV or RandomizedSearchCV.
  Save the best performing model and TF-IDF vectorizer for production use.
  Open vs code,then open news.py file,(model and vectoriser file should be keep in same directory of as news.py),open 
  terminal and run news.py file using python news.py
  Enter a news aricle content,click on classify ,you will get category of news.


>>Dataset
  BBC News Dataset (BBC News.csv): Contains news articles categorized into business, entertainment, politics, sport, and   tech.
