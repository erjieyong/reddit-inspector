# Reddit Post Inspector
Inspect and tune your reddit posts before posting

# Introduction
## Background
Is science a part of philosophy or are they 2 totally different subjects? According to [nytimes](https://archive.nytimes.com/opinionator.blogs.nytimes.com/2012/04/05/philosophy-is-not-a-science/), for roughly 98 percent of the last 2,500 years of Western intellectual history, philosophy was considered the mother of all knowledge. However, at the same time, we live an age in which many seem no longer sure what philosophy is or is good for anymore. Most seem to see it as a highly abstracted discipline with little if any bearing on objective reality — something more akin to art, literature or religion. 

## Problem Statement
Without a clear distinction, many users of reddits are facing challenge deciding which subreddits (Science or Philosophy) to post to.

## Assumptions
- Both subreddits have already been human moderated and thus have the best possible split to distinguish between Science and Philosophy

## Mission
As moderators of both Science and Philosophy subreddits, both of which have substantial members (28 and 17 millions), we set out to use machine learning to assist our users with
- Making a clear distinction of which subreddit to use when they have things to share
Concurrently, to create a more useful tool for our members, we added the following functions
- Subcategory selection (eg. health, chemistry, environment) for science subreddit only. To reduce confusion for users as there are often many overlapping areas between the subcategories such as health and environment.
- Sentiment analysis to assist users to evaluate their post's sentiment and subjectivity. A more neutral and objective posts lends more credibility to their post.
- Summarise the content of a user's shared url, to encourage user to share a summary when posting. Currently, less than 0.01% of posts have any description other than the title. 

## Scope
Based on Pushshift's API, we will download around 25,000 posts each to Science and Philosophy subreddits. Through a combination of features in this dataset, we aim to 
- classify posts accordingly to Science or Philosophy (binary classification)
- classify subcategories of Science based on its current 27 subcategories (multiclass classification)
- generate a summary of user's shared url through application of hugging face's model
- perform a sentiment analysis on posts through application of spacy's model

## Success Factor
Our model success would be evaluated based on
- the accuracy of our binary classification
- the accuracy of our multiclass classification
- output of sentiment and subjectivity
- succesful web crawling of user's shared url
- generation of summary for each posts

## Dataset generated using Pushshift's API
- [`science.csv`](./datasets/science.csv): Datasets of 25,000 most recent posts' stats from [Science](https://www.reddit.com/r/science/) subreddit
- [`philosophy.csv`](./datasets/philosophy.csv): Datasets of 25,000 most recent posts' stats from [Philosophy](https://www.reddit.com/r/philosophy/) subreddit

## Notebooks
- [`project_3.1_MlFlow.ipynb`](project_3.1_MlFlow.ipynb): Activating the user interface for mlflow
- [`project_3.2_data_extraction.ipynb`](project_3.2_data_extraction.ipynb): Data extraction using Pushshift's API
- [`project_3.31_main.ipynb`](project_3.31_main.ipynb): Main notebook where EDA, Data cleaning and binary classification is performed
- [`project_3.32_multiclass.ipynb`](project_3.32_multiclass.ipynb): Notebook to focus solely on multiclass classification, building on top of datasets from the main notebook
- [`project_3.33_summary.ipynb`](project_3.33_summary.ipynb): Notebook to test run hugging's face summariser model
- [`project_3.34_sentiment.ipynb`](project_3.34_sentiment.ipynb): Notebook to test run spacy's model
- [`project_3.4_deployment.ipynb`](project_3.4_deployment.ipynb): Notebook to consolidate all the previous notebooks's output, upload to google cloud and visualise it via streamlit

## Environment
- [`dsi-p3.yml`](./environment/dsi-p3.yml): Dedicated environment for this project

# Methodology
## Data Extraction
- Through the use of Pushshift's API, we will download 25,000 of the most recent posts from Science and Philosophy subreddits

## Exploratory Data Analysis
We explore the relationship of both subreddits in the following aspects
- Title length between both subreddits
- Number of submission deleted by moderators
- Selftext occurence between both subreddits
- Top 10 common domain shared in each subreddits
- Top 10 flair in each subreddits
- Number of original content
- Average number of comments
- Average upvote ratio
- Subscriber growth vs time

## Data Processing
- Not much cleaning is needed as data is downloaded directly from reddit source
- Through the usage of natural language processing, manipulate the data into a computer understandable format. We make use of manual preprocessing, count vectorising as well as tfidf vectorising techniques from sklearn's package

## Model Logging
- All models are being logged onto MLFlow

## Binary Classification Modeling
- Baseline model was established with manual preprocessing using multinomial naive bayes
- Exploratory modeling was done using different permutations of
    - full vs moderated datasets
    - CountVectoriser vs TFIDFVectoriser
    - Multinomial Naive Bayes, Random Forest, Extra Trees
- All models have hyper parameter tuning conducted

## Multiclass Classification Modelling
- Baseline model was established using TFIDFVectoriser with pycaret
- Exploratory modeling was done using different permutations of
    - CountVectoriser vs TFIDFVectoriser
    - Pycaret
    - 5 different models of hugging face algorithms
    
## Summariser
- Usage of `sshleifer/distilbart-cnn-12-6` pipeline from hugging face

## Sentiment Analysis
- Usage of spacy and spacytextblob library

## Deployment
The best model/algorithms from binary classification, multiclass classification, summariser and sentiment analysis was deployed using the following packages/programs
- Flask
- Docker
- Google Cloud
- Streamlit

The final model is now live on streamlit: [Link](https://erjieyong-reddit-inspector-streamlit-app-tnwiyp.streamlitapp.com/)

# Summary
Through our EDA, we found that 
- Over 56% of posts from the philosopy subreddits are removed by moderators
- More than 99% of posts only have title without posts content (aka. selftext)
- Most url shared in science subreddits are from news outlet while most url shared in philosophy subreddits are videos
- The subcategories (aka. flair) are more meaningful in Science subreddits are it help to differentiate the different Science topics, whereas the different subcategories of Philosophy subreddits is more focused on type of media content
- Science subreddit have almost 4 times the engagement with 1538 comments per posts as compared to Philosophy subreddits which only has 403 comments per posts
- Philosophy subreddit is growing at a faster rate than Science subreddit with 13 Nov 2021 as the baseline until it begins to taper off around 2022 July. Science subreddit grew faster afterwards

The model result table for **binary classification**
Model | Preprocessing | CV Accuracy | 
-------- | ---------- | -------- |
MultinomialNB | Manual (all submissions) | 0.9550
MultinomialNB | Manual (remove deleted submissions) | 0.9408
MultinomialNB | CountVectorizer (all submissions) | 0.9653 *
MultinomialNB | CountVectorizer (remove deleted submissions) | 0.9441
MultinomialNB | TfidfVectorizer (all submissions) | 0.9640
MultinomialNB | TfidfVectorizer (remove deleted submissions) | 0.9406
Random Forest | CountVectorizer (all submissions) | 0.9601
Extra Forest | CountVectorizer (all submissions) | 0.9660
Random Forest | CountVectorizer (all submissions) | 0.9580

*While we download the highest performing model for CV accuracy is actually Extra Forest, we did not proceed ahead with Extra Forest because there's an issue downloading the model from MLflow as well as using joblib> Hence, we proceed with second highest model (MultinomialNB with CountVectorizer) for the purpose of streamlit visualisation.


The model result table for **multiclass classification**

Model | Accuracy | Recall |  Precision | F1 |
-------- | ---------- | -------- | -------- | -------- |
Tfidf Pycaret | 	0.5252* | 	0.3795 | 	0.5151 | 	0.5089* 
Countvec Pycaret | 	0.4858 | 	0.3476 | 	0.4841 | 	0.4770
facebook/bart-large-mnli | 	0.2400 | 	0.2400 | 	0.2650 | 	0.2444
valhalla/distilbart-mnli-12-1 | 	0.3000 | 	0.3000 | 	0.5429 | 	0.3487
valhalla/distilbart-mnli-12-3 | 	0.3400 | 	0.3400 | 	0.4547 | 	0.3585
typeform/distilbert-base-uncased-mnli | 	0.3200 | 	0.3200 | 	0.4313 | 	0.3019
Narsil/deberta-large-mnli-zero-cls | 	0.4400 | 	0.4400 | 	0.5163 | 	0.4599

*We will be using the model with TFIDFVectorizer preprocessing using pycaret recommended model (Ridge Classifier) as it gives the highest accuracy score at 52.52% and F1 score at 50.89%


# Conclusion
Based on our analysis, Science subreddit is the faster growing subreddit with more active engagements.

Our model is also able to accurately classify between Science and Philosophy subreddits. However, more work can be done to improve the accuracy of the multiclass classification. Our deployed model is also able to make use of both hugging face's and spacy's algorithms to generate summary and sentiment analysis. 

With the successful deployment of our app, our users would be able to more accurately decide which subreddit to post to and which subcategories to choose from for Science subreddit. In addition, we are also able to advise our users on the sentiment and objectivity of their shared url to help them evaluate the post's credibility in order to sustain peer review. Finally, to encourage our users to give more description to their posts, we succesfully executed a summarising feature that generates a summary based on the shared url.  

# Further Evaluation
- Fix the save error in binary classification for extra trees model. 
- Perform hyperparameter tuning on for extra trees model for binary classification.
- Improve on multiclass classification performance
- Word correction / Suggestion to user to potentially improve comments and upvotes
- Regression analysis to predict the number of comments and upvotes for the submitted post
- Auto posting after user is satisfied with post results
- Spam detection for both subreddits