import streamlit as st
import requests
import json

# Title of the page
st.title("Science 🧪 vs 🧠 Philosophy Subreddit")

st.caption("""Don't know which subreddit to share your posts? Use our app!\n
For testing purposes, the cells have been populated with the *most upvoted* post on Science subreddit. Change it to your own!""")

#we have to put the inputs all inside a form to prevent the whole app from being re-run each time a input is change. 
# https://blog.streamlit.io/introducing-submit-button-and-forms/
# Get user inputs
with st.form(key='my_form'):
    title = st.text_input('Post Title', 'Physicist Stephen Hawking dies aged 76')
    selftext = st.text_area('Post Content', "We regret to hear that Stephen Hawking died tonight at the age of 76. We are creating a megathread for discussion of this topic here. The typical r/science comment rules will not apply and we will allow mature, open discussion. This post may be updated as we are able. A few relevant links: Stephen Hawking's AMA on /r/science. BBC's Obituary for Stephen Hawking. If you would like to make a donation in his memory, the Stephen Hawking Foundation has the Dignity Campaign to help buy adapted wheelchair equipment for people suffering from motor neuron diseases. You could also consider donating to the ALS Association to support research into finding a cure for ALS and to provide support to ALS patients.",
                            height = 150)
    url = st.text_input('URL', 'http://www.bbc.com/news/uk-43396008')
    submit = st.form_submit_button(label='Inspect')
    
    user_input = {'title': title,
                  'selftext': selftext,
                  'url': url}

# code to run after submit button is pressed
if submit:
    with st.spinner('🪄 ✨Gathering magic dusts...✨'):
        user_input = {'title': title,
                      'selftext': selftext,
                      'url': url}
        # Code to post the user inputs to the API and get the predictions
        # Paste the URL to your GCP Cloud Run API here!
        api_url = 'https://science-philo-reddit-class-vrfckmfjmq-as.a.run.app'
        api_route = '/predict'

        response = requests.post(f'{api_url}{api_route}', json=json.dumps(user_input)) # json.dumps() converts dict to JSON
        predictions = response.json()
        
        #SNOW!!
        st.snow()
        
        st.header("Inspection Result")

        col11, col12 = st.columns(2)
        col11.metric("Subreddit", predictions['subreddit'], help = "Most likely subreddit based on binary classification")
        if predictions['subreddit'] == 'Science':
            col12.metric("Flair", predictions['flair'], help = "Most likely subcategory based on multiclass classification")
        st.caption("""
        <p style="color: grey;font-size: 80%;">
        The above section predicts the most accurate subreddit to post the article based on machine learning of 50,000 previous posts. For Science subreddit, further subcategory (flair) is suggested for you based on 25,000 previous classification\n
        \n
        </p>
        """, unsafe_allow_html=True)

        #hide the arrow which is default by streamlit
        st.write(
            """
            <style>
            [data-testid="stMetricDelta"] svg {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        col21, col22 = st.columns(2)
        col21.metric("Sentiment", predictions['sentiment'], delta = predictions['senti_score'], delta_color = 'off', help = """
                     0.33 to 1 : Positive\n
                     -0.33 to 0.33 : Neutral\n
                     -1 to -0.33 : Negative
                     """)
        col22.metric("Subjectivity", predictions['subjectivity'], delta = predictions['subj_score'], delta_color = 'off', help = """
                     0.5 to 1 : Subjective\n
                     0 to 0.5 : Objective
                     """)
        st.caption("""
        <p style="color: grey;font-size: 80%;">
        The above section is the cumulative sentiment of post title, post content and article after crawling through the url. A more neutral sentiment with objective tone is a better showcase that your posts are not informed by bias\n
        \n
        </p>
        """, unsafe_allow_html=True)


        st.caption('<p class = "myclass">📃 Article Summary</p>', unsafe_allow_html=True)
        st.markdown(predictions['summary'])
        st.caption("""
        <p style="color: grey;font-size: 80%;">
        The above section is an auto generated summary of the article content after crawling through the url. 99.9% of subreddits does not have any post content other than title. Use our auto summary to fill up the gap!\n
        </p>
        """, unsafe_allow_html=True)

        st.markdown(
             f"""
             <style>
             .css-rvekum .myclass {{
                 color:black
             }}
             .css-dg4u6x .myclass {{
                 color:white
             }}
             </style>
             """,
             unsafe_allow_html=True
         )
        
