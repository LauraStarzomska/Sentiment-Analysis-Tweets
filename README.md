# Sentiment Analysis Tweets

### Introduction

The aim of the project is to create a model of Sentiment Analysis based on tweets about abortion topic. :exclamation: Besides this READ.me files, every stage of the project has its own where files are explained in more details. :exclamation:

### Project Stages

1. Scraping of the tweets. 
2. Data cleaning
3. Creating of word embeddings
4. Clustering 
5. Data labeling
6. Classical ML Models
7. Neural Models
8. BERT Sentiment Classifiter


### Files

- `scraping`: files with scraper code *tweepy_scraper.ipynb* that scrapes 25000 tweets from Twitter using Tweepy library (**1 stage**)
- `embedding_clustering`: files with *data_cleaning_embedding_clustering.ipynb* Jupyter Notebook that cleans, preprocess tweets and then creates word embedding out of them. Finally it creates clusters. (**2, 3, 4 stages**) 
- `zero-shot labeling`: files with python code *sentiment_analysis_zero_shot.ipynb* that labels unlabelled tweets (**5 stage**)
- `models`: files with 3 jupyter notebooks that build classical ml models, neural network model and BERT sentiment classifier  (**6,7,8 stages**)

### Usage
To use this project, follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies mentioned in the requirements.txt file.

```pip install -r requirements.txt```


3. Run the Jupyter Notebooks in the following order:

- scraping/tweepy_scraper.ipynb
- embedding_clustering/data_cleaning_embedding_clustering.ipynb
- zero-shot labeling/sentiment_analysis_zero_shot.ipynb - :warning: This notebook was created using **Google Colab** to leverage GPU acceleration for efficient processing. :warning:
- models/classical_ml_models.ipynb
- models/neural_network_model.ipynb - :warning: This notebook was created using **Google Colab** to leverage GPU acceleration for efficient processing. :warning:
- models/BERT_sentiment_classifier.ipynb - :warning: This notebook was created using **Google Colab** to leverage GPU acceleration for efficient processing. :warning:
Ensure that you have the necessary data available and provide the appropriate file paths in the notebooks where required.

# Methodology

This section outlines the methodology employed in each stage of the Sentiment Analysis Tweets project. The following steps were undertaken to develop an effective sentiment analysis model for tweets related to the abortion topic:

1. **Scraping of the tweets**:
   - Utilized the Tweepy library and Twitter API to scrape a dataset of 25,000 tweets relevant to the abortion topic.
   - Applied appropriate filters and query parameters to retrieve a diverse range of tweets for analysis.

2. **Data cleaning**:
   - Performed comprehensive data cleaning and preprocessing techniques to prepare the raw tweet data for further analysis.
   - Removed irrelevant characters, such as URLs, hashtags, mentions, and special symbols.
   - Handled issues related to punctuation, capitalization, and excessive whitespace.
   - Addressed common challenges in text data, such as spelling corrections, stop-word removal, and tokenization.

3. **Creating of word embeddings**:
   - Employed Word2Vec, a popular word embedding technique, to convert the preprocessed tweet data into distributed word representations.
   - Trained a Word2Vec model on the cleaned tweet corpus to learn continuous word vectors capturing semantic relationships.

4. **Clustering**:
   - Applied the K-means clustering algorithm to group similar tweets based on the word embeddings obtained in the previous step.
   - Explored different cluster sizes and evaluated the cluster quality using techniques such as the silhouette score.
   - Identified clusters representing distinct sentiment patterns or themes within the dataset.

5. **Data labeling**:
   - Labeled a subset of the tweets manually or through alternative labeling techniques to create a labeled dataset for model training and evaluation.
   - Ensured a balanced distribution of positive, negative, and neutral sentiment labels to avoid bias in the sentiment analysis model.

6. **Classical ML Models**:
   - Developed classical machine learning models, such as logistic regression, decision trees, random forests, and support vector machines (SVM), to classify the sentiment of the tweets.
   - Utilized features derived from the word embeddings and other relevant textual or metadata features.
   - Evaluated and compared the performance of different models using appropriate metrics, such as accuracy, precision, recall, and F1 score.

7. **Neural Models**:
   - Utilized neural network architectures, such as recurrent neural networks (RNN), long short-term memory (LSTM), or convolutional neural networks (CNN), to capture the contextual information and dependencies in the tweet data.
   - Constructed and trained deep learning models using frameworks like TensorFlow to predict sentiment labels.
   - Optimized hyperparameters, model architectures, and regularization techniques to enhance performance.

8. **BERT Sentiment Classifier**:
   - Employed the powerful BERT (Bidirectional Encoder Representations from Transformers) model, pre-trained on a large corpus, to perform sentiment classification.


Throughout the project, attention was given to data quality, feature engineering, model selection, and performance evaluation. The methodology aimed to create a reliable and accurate sentiment analysis model to effectively analyze and classify the sentiment expressed in tweets related to the abortion topic.

### Technologies Used

- Python 3.9
- Tweepy API v2
- Jupyter Notebook
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Hugging Face Transformers
- NLTK
- Word2Vec
- K-means Clustering
- BERT (Bidirectional Encoder Representations from Transformers)

### Results
