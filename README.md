# Sentiment Analysis Tweets

### Introduction

The aim of the project is to create a model of Sentiment Analysis based on tweets about abortion topic. Besides this READ.me files, every stage of the project has its own where files are explained in more details. 

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
