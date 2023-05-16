# Data cleaning, creating embedding & clustering

This part of the project focuses on cleaning & preprocessing of the scraped tweets and then creating word embeddings out of it using Word2Vec library. In the end it builds clusters using K-Means method to label data by sentiments. I also prepared my own-labeled 150 tweets to compare results with K-Means clusters. Unfortunately, the accuracy was very poor - 34%. This is why I used Zero-Shot Classification in the next stage.

# Files

- `data_cleaning_embedding_clustering.ipynb.zip`- main python notebook, due to the huge size it should have got compressed
- `cleaned_dataset.csv`- cleaned dataset with bigramms
- `hands_labeled_tweets.csv` - 150 first tweets with prediction gained from K-Means clusters & my labels
- `hands_labeled_tweets.xlsx` - 150 first tweets with prediction gained from K-Means clusters & my labels
- `labeled_tweets.csv` - tweets labaled using K-Means clusters
- `sentiment_dictionary.csv` - dictionary of the whole words with their counted sentiments 
- `tweets_df_preprocessed.csv`- final file of the preprocessed, cleaned and ready for further analysis tweets
- `tweets_df_preprocessed.xlsx` - final file of the preprocessed, cleaned and ready for further analysis tweets
- `word2vec.model`- Word2Vec model

