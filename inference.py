from flask import Flask, request 
import pandas as pd 
import os 
import joblib
import time
import mlflow.pyfunc
# We have to import the following io, otherwise pandas will output error because we only passing in a string and not a dictionary.
# https://stackoverflow.com/questions/63553845/pandas-read-json-valueerror-protocol-not-known/63655099#63655099
from io import StringIO
# To scrap website and run summariser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
# To perform sentiment analysis
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


api = Flask('ModelEndpoint')

#########################
##    Classification   ##
#########################
model_classify = joblib.load("./models/model_classify.pkl")

#########################
## MultiClassification ##
#########################
model_multiclass = mlflow.pyfunc.load_model(model_uri="./models/model_multiclass")
transform_multiclass = joblib.load("./models/model_multiclass/tfidf.pkl")
#function to run multiclass predictions
def multiclass_predict(text):
    flair_dict = {1: 'Medicine',
                  2: 'Social Science',
                  3: 'Animal Science',
                  4: 'Anthropology',
                  5: 'Environment',
                  6: 'Psychology',
                  7: 'Health',
                  8: 'Nanoscience',
                  9: 'Engineering',
                  10: 'Biology',
                  11: 'Earth Science',
                  12: 'Astronomy',
                  13: 'Genetics',
                  14: 'Economics',
                  15: 'Paleontology',
                  16: 'Chemistry',
                  17: 'Neuroscience',
                  18: 'Cancer',
                  19: 'Mathematics',
                  20: 'Epidemiology',
                  21: 'Physics',
                  22: 'Geology',
                  23: 'Materials Science',
                  24: 'Computer Science',
                  25: 'Breaking News',
                  26: 'Retraction',
                  27: 'Best of r/science'}
    flair_no = model_multiclass.predict(transform_multiclass.transform(text))[0]
    return flair_dict[flair_no]

#########################
##     Summariser      ##
#########################
#function to run summariser
hf_summarizer = pipeline('summarization', 'sshleifer/distilbart-cnn-12-6')
def summariser(url):
    if len(url) > 1:
        # pass in header in attempt to hide that this is an automated web crawler
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'lxml')
            all_p = soup.find_all('p')
            relevant_p = ''
            for p in all_p:
                # We assume that if the sentence has less than 10 words, it will not be of value to article
                if len(p.text.split(' ')) > 10:
                    relevant_p += (p.text + ' ')
            #we cap it to 800 because there's a limit of words that the hugging face model can take
            relevant_p_trimmed = ' '.join(relevant_p.split(' ')[:700])
            return hf_summarizer(relevant_p_trimmed)[0]['summary_text'], relevant_p
        else:
            #return error message if unable to crawl website
            return "ERROR! Unable to crawl website. Please check if the link is valid or if the website allows automated web crawling", "a"
    else:
        #return error message if no the url length is only 1
        return "ERROR! Please pass in a valid url", "a"

#########################
##      Sentiment      ##
#########################
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
#function to retrieve sentiment and subjectivity
def sentiment_predict(title, selftext, page):
    text = title + selftext + page
    spacy_output = nlp(text)
    if spacy_output._.polarity > 0.33:
        sentiment = 'Positive'
    elif spacy_output._.polarity < -0.33:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    if spacy_output._.subjectivity < 0.5:
        subjectivity = 'Objective'
    else:
        subjectivity = 'Subjective'
    return sentiment, round(spacy_output._.polarity,2), subjectivity, round(spacy_output._.subjectivity,2)

#########################
##      FLASK API      ##
#########################
@api.route('/') 
def home(): 
    return {"message": "Hello!", "success": True}, 200

@api.route('/predict', methods = ['POST']) 
def make_predictions():
    user_input = request.get_json(force=True)
    df_schema = {'title':str, 'selftext': str, 'url':str}
    user_input_df = pd.read_json(StringIO(user_input), lines=True, dtype=df_schema)
    combined_user_input = pd.Series(user_input_df['title'] + user_input_df['selftext'] + user_input_df['url']) #this will output as {'0':'all the texts'}
    

    #failsafe in case someone input nothing in either of the 3 fields
    if len(user_input_df['title'].tolist()[0])==0:
        #input 'a' as a common letter that should give no meaning in nlp
        user_input_df['title'] = 'a'
    if len(user_input_df['selftext'].tolist()[0])==0:
        #input 'a' as a common letter that should give no meaning in nlp
        user_input_df['selftext'] = 'a'
    if len(user_input_df['url'].tolist()[0])==0:
        #input 'a' as a common letter that should give no meaning in nlp
        user_input_df['url'] = 'a'
    
    # return {'test':user_input_df['title'].tolist()[0]}
    # PREDICTIONS
    predict_class = model_classify.predict(combined_user_input).tolist() #we need to pass number to list so that it can be convered to json later   
    predict_flair = multiclass_predict(combined_user_input) #For text, we don't need to pass in tolist
    summary, page = summariser(user_input_df['url'].tolist()[0])
    sentiment, senti_score, subjectivity, subj_score = sentiment_predict(user_input_df['title'].tolist()[0], user_input_df['selftext'].tolist()[0], page)
    

    # RETURN OUTPUT
    if predict_class[0]==1:
        return {'subreddit': 'Science', 'flair': predict_flair, 'summary':summary, 'sentiment': sentiment, 'senti_score':senti_score, 'subjectivity':subjectivity, 'subj_score':subj_score}
    else:
        return {'subreddit': 'Philosophy', 'summary':summary, 'sentiment': sentiment, 'senti_score':senti_score, 'subjectivity':subjectivity, 'subj_score':subj_score}
    
if __name__ == '__main__': 
    api.run(host='0.0.0.0', 
            debug=True, 
            port=int(os.environ.get("PORT", 8080))
           ) 
