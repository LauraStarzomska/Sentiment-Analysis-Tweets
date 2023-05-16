# Sentiment Analysis Tweets

### Introduction

The aim of the project is to create a model of Sentiment Analysis based on tweets about abortion topic. 

### Project Stages

1. Scraping of the tweets. 
2. Data cleaning
3. Creating of word embeddings
4. Clustering 
5. Data labeling
6. Classical ML Models
7. Neural Network Models
8. BERT Sentiment Classifiter


### Files

- `scraping`: files with scraper code *tweepy_scraper.ipynb* that scrapes 25000 tweets from Twitter using Tweepy library (**1 stage**)
- `embedding_clustering`: files with *data_cleaning_embedding_clustering.ipynb* Jupyter Notebook that cleans, preprocesses tweets and then creates word embedding out of them. Finally it creates clusters. (**2, 3, 4 stages**) 
- `zero-shot labeling`: files with python code *sentiment_analysis_zero_shot.ipynb* that labels unlabelled tweets (**5 stage**)
- `models`: files with 3 jupyter notebooks that build classical ml models, neural network model and BERT sentiment classifier  (**6,7,8 stages**)

### Usage
To use this project, follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies mentioned in the requirements.txt file.

```pip install -r requirements.txt```


3. Run the Jupyter Notebooks in the following order:

- scraping/tweepy_scraper.ipynb ❗ you have to have your own account and generate keys for twitter API ❗
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
   - Identified clusters representing distinct sentiment patterns or themes within the dataset.

5. **Data labeling**:
   - Labeled 150 of the tweets manually and then compare results with clusters
   - The results were poor so I labeled tweets with Zero-Shot Classification

6. **Classical ML Models**:
   - Developed classical machine learning models, such as logistic regression, Naive Bayes and support vector machines (SVM), to classify the sentiment of the tweets.
   - Evaluated and compared the performance of different models using appropriate metrics, such as accuracy, precision, recall, and F1 score.

7. **Neural Models**:
   - Utilized neural network architectures to capture the contextual information and dependencies in the tweet data.
   - Fine-tuning of the model by optimizing hyperparameters, model architectures, and regularization techniques to enhance performance.

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

The sentiment analysis project on tweets related to the abortion topic provided valuable insights and outcomes. However, certain challenges were encountered along the way, highlighting areas for potential improvement. In this section, we discuss the results obtained from each stage of the project and address the limitations observed.

The initial stage involved clustering the tweets based on word embeddings. Unfortunately, the clustering approach did not yield optimal results, with an accuracy of only 34% when compared to a hand-labeled dataset of 150 tweets. This indicates that the clusters generated using word embeddings did not align well with the underlying sentiment patterns in the data. Future work could focus on exploring alternative clustering algorithms or employing dimensionality reduction techniques to improve clustering performance.

To overcome the limitations of clustering, the zero-shot labeling approach was employed to label unlabelled tweets. The zero-shot labeling technique achieved an accuracy of 47%. However, it resulted in an imbalanced dataset, with 82.3% of the tweets classified as negative and only 17.7% as positive. The main challenge encountered was the class imbalance, which can adversely affect model performance and bias the predictions towards the majority class. To address this issue, several strategies can be explored:

1. Data Augmentation: Generate synthetic data points for the minority class to increase its representation in the dataset. Techniques like text augmentation, such as synonym replacement or back-translation, can be used to create additional positive sentiment instances.

2. Undersampling/Oversampling: Apply undersampling techniques to reduce the majority class instances or oversampling techniques to increase the minority class instances, thus achieving a more balanced distribution. Techniques like Random Undersampling, SMOTE (Synthetic Minority Oversampling Technique), or ADASYN (Adaptive Synthetic Sampling) can be explored.

3. Choose different topic: probably this one is too more one-sided.

4. Scrape more data: Probably the more the data the more balanced they are. It could be checked.

Moving on to model development, classical machine learning models were built to classify the sentiment of the tweets. Among the models tested, the Logistic Regression model performed the best, achieving an accuracy of 87% on the test set and 90% on the train set. However, further optimizations, such as feature selection, hyperparameter tuning, or ensemble techniques, could potentially enhance the performance of the classical ML models.

Additionally, neural network models were employed to capture the contextual information and dependencies in the tweet data. Despite fine-tuning efforts, the neural network model still faced challenges in accurately classifying sentiments, indicating that fine-tuning alone might not be sufficient. Most probably, it happened due to my ***imbalanced data structure**. 

Furthermore, the BERT (Bidirectional Encoder Representations from Transformers) model, specifically the DistilBERT variant, was utilized for sentiment classification. The DistilBERT model achieved an overall accuracy of approximately 86.45% in correctly classifying instances. Unfortunately, during testing it classified the test sentence **wrongly**. However, it is important to note that the model used in this project was not fine-tuned but rather utilized as a pre-trained classifier. Fine-tuning the DistilBERT model on a larger and more domain-specific labeled dataset could lead to further improvements in performance and better alignment with the sentiment analysis task.

In summary, the project provided valuable insights into the sentiment analysis of tweets related to the abortion topic. However, several challenges were identified, particularly in relation to the imbalance in the sentiment distribution within the dataset. This imbalance resulted in poor model performance and biased predictions towards the majority class.

To address this issue, future improvements could focus on enhancing the balance of the dataset by employing data augmentation techniques, undersampling or oversampling methods, or cost-sensitive learning approaches. These techniques aim to rectify the class imbalance and ensure a more representative distribution of positive and negative sentiment instances.
