# Models

This folder focuses on building models and consists of 3 steps:

1. Classical Machine Learning Models - Naive Bayes, SVM & Logistic Regression
2. Neural Network Model & Fine-tuning
3. Language Model - Sentiment Classifier based on BERT

On every stage of the model the results of the models were compared. In the first steps the best performer was Logistic Regression Model with train accuracy of 90% and test accuracy of 87%. Neural network model chieved an accuracy of 0.84, indicating that it correctly classified 84% of the samples. The model's precision and recall for class 0 (negative sentiment) are 0.57 and 0.48, respectively, while for class 1 (positive sentiment), the precision and recall are 0.89 and 0.92, respectively. The F1-score for class 0 is 0.52, while for class 1 it is 0.91. The main problem with every step of the model creation was that my data was **imbalanced**. 17.7% positive class to 82.3% negative class resulted in not-so-well performances of the models. 

For BERT Classifier the results also were not impressive. The evaluation accuracy, representing the proportion of correctly predicted instances, was 0.8645, indicating that approximately 86.45% of the instances were classified correctly. Unfortunately, during testing the model classified test sentence **wrongly**. 

There is a huge room for improvement in terms of data collection and labeling but also adjusting imbalance to the model architecture. I explained what could be improved in the results section in the main repository README.md file.


### Files

`BERT_model.zip` -  saved and compressed (due to huge size) language model
`bert_model.ipynb` - python notebook with code that creates BERT Sentiment Classifier
`best_model.pickle` - saved 1st version of neural network model
`best_model2.pickle` - saved 2nd version of neural network model
`classical_ml_models.ipynb`- python notebook with 3 classical machine learning models
`deep_learning.ipynb` - python notebook with neural network model creation 
`tweets_df_cleaned_labeled.csv` - cleaned and labeled tweets
`word2vec.model` - saved word embeddings
