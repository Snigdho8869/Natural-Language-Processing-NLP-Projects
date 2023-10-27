# IMDb Movie Reviews

This project focuses on performing sentiment analysis on IMDb movie reviews using various natural language processing (NLP) and machine learning techniques. The goal is to predict whether a movie review expresses a positive or negative sentiment.

## Data

The project uses two datasets:

- `IMDB_Movie_Review_Final_Train.csv`: This dataset contains training data, including movie reviews and their associated sentiments (positive or negative).

- `IMDB_Movie_Review_Final_Test.csv`: This dataset serves as the test data to evaluate the model's performance.

## Preprocessing

1. **Data Loading**: The training and test datasets are loaded from the CSV files.

2. **Text Cleaning**: The raw text data is cleaned, which involves removing HTML tags, special characters, and non-alphanumeric characters. It also includes lowercasing and lemmatization.

3. **Text Vectorization**: The cleaned text is vectorized using techniques like Count Vectorization and TF-IDF (Term Frequency-Inverse Document Frequency).

4. **Stopwords**: Common English stopwords are removed from the text data to improve the quality of the features.

## Sentiment Analysis Models

The project explores a variety of machine learning (ML) and deep learning (DL) models for sentiment analysis, including:

1. **Logistic Regression**: A traditional machine learning model used for binary sentiment classification.

2. **Support Vector Machine (SVC)**: A traditional machine learning model that finds an optimal hyperplane to classify sentiment.

3. **Multinomial Naive Bayes**: A probabilistic model for text classification, suitable for sentiment analysis.

4. **Random Forest**: An ensemble learning method based on decision trees, used for sentiment analysis.

5. **Gradient Boosting**: An ensemble learning technique that combines weak models into a stronger one for sentiment classification.

6. **Ensemble Model**: A model that combines the predictions of multiple models to improve the overall sentiment classification.

7. **XGBoost**: A gradient boosting algorithm that is known for its efficiency and effectiveness in sentiment analysis.

8. **LSTM-based Model**: A deep learning model that utilizes Long Short-Term Memory (LSTM) layers to understand sequential data and perform sentiment analysis.

9. **GRU-based Model**: A model using Gated Recurrent Units (GRU) for sentiment classification.

10. **CNN-based Model**: A Convolutional Neural Network (CNN) model to capture spatial features in text data for sentiment prediction.

11. **Hybrid Model**: A model that combines both CNN and LSTM layers to leverage the strengths of each architecture.

12. **BERT Model**: Utilizing a pre-trained BERT model to perform sentiment analysis with contextual understanding.

Each of these models offers a different approach to sentiment analysis, and the project evaluates their performance to determine which one works best for IMDb movie reviews.


## Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- BeautifulSoup
- tensorflow/keras
- transformers (Hugging Face)
- matplotlib
- seaborn


# Results:
The results of each model on the dataset are as follows:

|  Model | Accuracy |
|----------|----------|
| Logistic Regression | 99.96% |
| Support Vector Machine | 100.00% |
| Multinomial Naive Bayes | 99.94% |
| Randomforest Classifier| 100.00% |
| GradientBoostingClassifier | 90.36% |
| Ensemble Classifier | 100.00%% |
| XGBoost | 94.24% |
| LSTM Model | 99.84% |
| GRU Model | 99.76% |
| CNN Model | 99.92% |
| Hybrid Model | 99.80% |
| BERT | 99.64% |



