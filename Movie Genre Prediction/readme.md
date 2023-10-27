# Movie Genre Prediction

This project focuses on predicting the genre of movie scripts using various machine learning and deep learning models. The dataset contains movie scripts, and the objective is to classify them into different genres.

## Data Preprocessing

The initial step involves preprocessing the movie scripts. The following steps are performed:

1. **Text Cleaning**: The text is converted to lowercase, and any special characters are removed.

2. **HTML Tags Removal**: Any HTML tags present in the text are removed.

3. **Stopwords Removal**: Common English stopwords are removed from the text.

4. **Lemmatization**: Text is lemmatized to reduce words to their base form.

## Machine Learning Models

#### Logistic Regression
- Logistic Regression is applied to the preprocessed text to predict movie genres.
- A grid search is performed to find the best hyperparameters.

#### Support Vector Machine (SVC)
- SVC is used to classify the scripts into genres.
- A grid search is also applied to find optimal hyperparameters.

#### Multinomial Naive Bayes
- This probabilistic model is employed for text classification.

#### Gradient Boosting
- Gradient Boosting is employed to create a strong predictive model.

#### Ensemble Model
- An ensemble model combines the predictions from multiple models to enhance classification accuracy.


## Deep Learning Models

#### LSTM-based Model
- A deep learning model is built using Long Short-Term Memory (LSTM) layers to understand sequential data.

#### GRU-based Model
- A model based on Gated Recurrent Units (GRU) is used for sentiment classification.

#### CNN-based Model
- Convolutional Neural Networks (CNNs) are employed to capture spatial features in text data for genre prediction.

#### Hybrid Model
- A hybrid model combines both CNN and LSTM layers to take advantage of the strengths of each architecture.

#### BERT Model
- A pre-trained BERT model is used for genre prediction, offering contextual understanding.

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


## Results:
The results of each model on the dataset are as follows:

|  Model | Accuracy |
|----------|----------|
| Logistic Regression | 92.40% |
| Support Vector Machine | 93.00% |
| Multinomial Naive Bayes | 93.24% |
| GradientBoostingClassifier | 85.98% |
| Ensemble Classifier | 94.26%% |
| LSTM Model | 90.23% |
| GRU Model | 89.48% |
| CNN Model | 89.17% |
| Hybrid Model | 88.06% |
| BERT | 93.43% |
