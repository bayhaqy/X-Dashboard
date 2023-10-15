# Data Analysis and Profiling
import pandas as pd
from ydata_profiling import ProfileReport

# Streamlit for Building the Dashboard
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

# Language Detection
from langdetect import detect

# NLP and Text Processing
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# URL Parsing
from urllib.parse import urlparse

# Data Visualization
import plotly.express as px
import matplotlib.pyplot as plt

# Word Cloud Generation
from wordcloud import WordCloud

# Other Libraries
import torch
import requests
import subprocess
import logging
import json
import re
import os

# NLTK Data Download
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

## ............................................... ##
# Function to install Node.js
@st.cache_data
def install_nodejs():
    st.sidebar.markdown('OS Information')
    result_OS = subprocess.check_output(['lsb_release', '-a']).decode("utf-8")
    st.sidebar.markdown(f'{result_OS}')

    st.sidebar.markdown('Python Information')
    result_PY = subprocess.check_output(['python', '--version']).decode("utf-8")
    st.sidebar.markdown(f'{result_PY}')

    st.sidebar.markdown('IP Information')
    result_IP = subprocess.check_output(['curl', 'ipinfo.io']).decode("utf-8")
    # Parse the JSON string into a dictionary
    result_dict = json.loads(result_IP)

    # Iterate through the dictionary and print key-value pairs
    for key, value in result_dict.items():
        st.sidebar.markdown(f'{key} : {value}')
        
    try:
        # Check if Node.js is already installed by attempting to get its version.
        node_major_version = int(subprocess.check_output(['node', '-v']).decode("utf-8").split('.')[0][1:])
    except FileNotFoundError:
        # If 'node' command is not found, it means Node.js is not installed.
        node_major_version = 0
        
    if node_major_version < 20:
        st.sidebar.markdown('Update OS')
        subprocess.check_call(['sudo', 'apt-get', 'update'])

        st.sidebar.markdown('Download Files Requirement for Nodesource')
        subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'ca-certificates', 'curl', 'gnupg'])
        subprocess.check_call(['sudo', 'mkdir', '-p', '/etc/apt/keyrings'])
        subprocess.check_call(f'curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg', shell=True)

        NODE_MAJOR = 20
        node_source_entry = f"deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_{NODE_MAJOR}.x nodistro main"
        subprocess.check_call(f'echo "{node_source_entry}" | sudo tee /etc/apt/sources.list.d/nodesource.list', shell=True)

        st.sidebar.markdown('Install Node.js')
        subprocess.check_call(['sudo', 'apt-get', 'update'])
        subprocess.check_call(['sudo', 'apt-get', 'install', 'nodejs', '-y'])

        result = subprocess.check_output(['node', '-v']).decode("utf-8")
        st.sidebar.markdown(f'Node.js version: {result}')
    else:
        st.sidebar.markdown('Node.js version already installed')
        result = subprocess.check_output(['node', '-v']).decode("utf-8")
        st.sidebar.markdown(f'Node.js version already updated to {result}')

## ............................................... ##
# Function to run tweet-harvest
@st.cache_data
def run_X_scrapping(search_keyword,from_date,to_date,limit,delay,token,filename):
    with st.expander("Scrapping Logs"):
        # Run scraping with the provided parameters
        #st.markdown('Check Tweet')
        command = f'npx --yes {X_Sources} -s "{search_keyword}" -f "{from_date}" -t "{to_date}" -l {limit} -d {delay} --token "{token}" -o "{filename}"'
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            st.markdown("Command executed successfully.")
            st.markdown(result.stdout)  # Display the standard output, give comment if you don't want to see
        except subprocess.CalledProcessError as e:
            st.markdown("Error: The command returned a non-zero exit status.")
            st.markdown("Error message:", e)
            st.markdown(f'Standard output: {e.stdout}')
            st.markdown(f'Standard error: {e.stderr}')

## ............................................... ##
# Function for get model and tokenize
@st.cache_resource
def get_models_and_tokenizers():
    model_name = X_Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    #model.eval()

    return model, tokenizer

## ............................................... ##
# Function for sentiment analysis
@st.cache_resource
def analyze_sentiment_distilbert(text, _model, _tokenizer):
    try:
        tokens_info = _tokenizer(text, truncation=True, return_tensors="pt")
        with torch.no_grad():
            raw_predictions = _model(**tokens_info).logits

        predicted_class_id = raw_predictions.argmax().item()
        predict = _model.config.id2label[predicted_class_id]

        softmaxed = int(torch.nn.functional.softmax(raw_predictions[0], dim=0)[1] * 100)
        if (softmaxed > 70):
            status = 'Not trust'
        elif (softmaxed > 40):
            status = 'Not sure'
        else:
            status = 'Trust'
        return status, predict

    except Exception as e:
        logging.error(f"Sentiment analysis error: {str(e)}")
        return 'N/A', 'N/A'

## ............................................... ##
# Function for sentiment analysis using VADER
@st.cache_resource
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

## ............................................... ##
# Function for sentiment analysis using TextBlob
@st.cache_resource
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

## ............................................... ##
# Function for translation
@st.cache_data
def translate_text(text, source='auto', target='en'):
    try:
        if source != target:
            text = GoogleTranslator(source=source, target=target).translate(text)
        return text

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

## ............................................... ##
# Function for Load and Transform Data
@st.cache_data
def selection_data(filename):
    file_path = f"tweets-data/{filename}"
    df = pd.read_csv(file_path, delimiter=";")


    # Rename columns
    column_mapping = {
        'created_at': 'Created Date',
        'user_id_str': 'User ID',
        'username': 'Username',
        'full_text': 'Tweet',
        'tweet_url': 'Tweet URL',
        'id_str': 'Tweet ID',
        'conversation_id_str': 'Conversation ID',
        'lang': 'App Language',
        'quote_count': 'Quote Count',
        'reply_count': 'Reply Count',
        'retweet_count': 'Retweet Count',
        'favorite_count': 'Favorite Count',
    }

    df = df.rename(columns=column_mapping)

    # Add a new column for detected language
    df['Detect Language'] = df['Tweet'].apply(lambda tweet: detect(tweet))

    # Mapping language codes to country names
    language_to_country = {
        'af': 'South Africa',
        'ar': 'Arabic',
        'bg': 'Bulgaria',
        'bn': 'Bangladesh',
        'ca': 'Catalan',
        'cs': 'Czech',
        'cy': 'Welsh',
        'da': 'Danish',
        'de': 'German',
        'el': 'Greek',
        'en': 'English',
        'es': 'Spanish',
        'et': 'Estonian',
        'fa': 'Persian',
        'fi': 'Finnish',
        'fr': 'French',
        'gu': 'Gujarati',
        'he': 'Hebrew',
        'hi': 'Hindi',
        'hr': 'Croatian',
        'hu': 'Hungarian',
        'id': 'Indonesian',
        'it': 'Italian',
        'ja': 'Japanese',
        'kn': 'Kannada',
        'ko': 'Korean',
        'lt': 'Lithuanian',
        'lv': 'Latvian',
        'mk': 'Macedonian',
        'ml': 'Malayalam',
        'mr': 'Marathi',
        'ne': 'Nepali',
        'nl': 'Dutch',
        'no': 'Norwegian',
        'pa': 'Punjabi',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'so': 'Somali',
        'sq': 'Albanian',
        'sv': 'Swedish',
        'sw': 'Swahili',
        'ta': 'Tamil',
        'te': 'Telugu',
        'th': 'Thai',
        'tl': 'Tagalog',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'ur': 'Urdu',
        'vi': 'Vietnamese',
        'zh-cn': 'Simplified Chinese',
        'zh-tw': 'Traditional Chinese'
    }

    # Add 'Country' column to df
    df['Language'] = df['Detect Language'].map(language_to_country)

    # Sort columns
    desired_columns = ['Created Date', 'User ID', 'Username', 'Tweet', 'Language', 'Detect Language', 'App Language', 'Tweet URL', 'Tweet ID', 'Conversation ID', 'Quote Count', 'Reply Count', 'Retweet Count', 'Favorite Count']
    df = df[desired_columns]

    # Set data types
    data_types = {
        'Created Date': 'datetime64[ns]',
        'User ID': 'int64',
        'Username': 'object',
        'Tweet': 'object',
        'Language': 'object',
        'Detect Language': 'object',
        'App Language': 'object',
        'Tweet URL': 'object',
        'Tweet ID': 'int64',
        'Conversation ID': 'int64',
        'Quote Count': 'int64',
        'Reply Count': 'int64',
        'Retweet Count': 'int64',
        'Favorite Count': 'int64',
    }

    df = df.astype(data_types)

    return df

## ............................................... ##
# Function to preprocess the data
@st.cache_data
def preprocessing_data(df):
    # Remove duplicates
    df = df.drop_duplicates(subset='Translation')

    # Function to clean and preprocess text
    def clean_text(text):
        # Remove mentions (e.g., @username)
        text = re.sub(r'@[\w]+', '', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()

        # Convert to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize text
        words = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    # Apply the clean_text function to the "Translation" column
    df['Cleaned Translation'] = df['Translation'].apply(clean_text)

    return df

## ............................................... ##
# Function to create a Word Cloud
@st.cache_data
def create_wordcloud(df):
    # Combine all text
    text = ' '.join(df['Cleaned Translation'])

    # Create a Word Cloud
    wordcloud = WordCloud(width=700, height=400, max_words=50).generate(text)

    # Convert the word cloud to an image
    wordcloud_image = wordcloud.to_image()

    # Display the Word Cloud using st.image
    st.write("word Cloud by Tweets")
    st.image(wordcloud_image, use_column_width=True)

## ............................................... ##
# IMPORTANT: Cache the conversion to prevent computation on every rerun
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

## ............................................... ##
# Set page configuration (Call this once and make changes as needed)
st.set_page_config(page_title='(Tweet) X Scrapper Dashboard',  layout='wide', page_icon='	📱')

## ............................................... ##
with st.container():
  # Define Streamlit app title and introduction
  st.title("(Tweet) X Scrapper Dashboard")
  st.write("Created by Bayhaqy")

# Sidebar content
st.sidebar.subheader("About the app")
st.sidebar.info("This app allows you to get data, analysis and prediction with the (Tweet) X Scrapper tool.")

url = "https://blogs.bayhaqy.my.id/2023/10/auth-token-twitter.html"
st.sidebar.markdown("check this [link](%s) for guides on how to get your own X Auth Token" % url)

st.sidebar.write("\n\n")
st.sidebar.markdown("**Please contact me if you have any questions**")
st.sidebar.write("\n\n")
st.sidebar.divider()
st.sidebar.markdown("© 2023 (Tweet) X Scrapper Dashboard")

## ............................................... ##
# HuggingFace API KEY input
X_Sources = os.environ.get("X_Sources")
X_Limit = int(os.environ.get("X_Limit"))
X_Auth = os.environ.get("X_Auth")
X_Model = os.environ.get("X_Model")

## ............................................... ##
# Initialize to install node.js
if st.sidebar.button("Check Node.js updated"):
    install_nodejs()

# Initialize model and tokenizer
model, tokenizer = get_models_and_tokenizers()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

## ............................................... ##
# Set up logging
logging.basicConfig(filename='tweet_harvest.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

## ............................................... ##
with st.container():    
    with st.expander("Start to scrapping"):
        # Input search parameters
        search_keyword = st.text_input("Enter search keyword", "Mitra Adiperkasa",)

        col1, col2 = st.columns(2)
        with col1:
          from_date = st.date_input('From Date :', pd.to_datetime('2023-01-01'))
          limit = st.number_input("Enter limit", min_value=1, value=10, max_value = X_Limit)
        with col2:
          to_date = st.date_input('To Date :', pd.to_datetime('2023-12-01'))
          delay = st.number_input("Enter delay in seconds", min_value=1, value=3)

        token = st.text_input("Enter your X Auth Token", X_Auth, type="password")

## ............................................... ##
        col1, col2 = st.columns(2)

        with col1:
          # Checkbox options for different processing steps
          include_translation = st.checkbox("Include Translation", value=False)
          include_sentiment_analysis = st.checkbox("Include Sentiment Analysis", value=False)
        with col2:
          include_sentiment_vader = st.checkbox("Include VADER Sentiment Analysis", value=False)
          include_sentiment_textblob = st.checkbox("Include TextBlob Sentiment Analysis", value=False)

## ............................................... ##
# Create a button to trigger tweet-harvest
with st.container():
    if st.button("Run it"):
        # Format the dates as "DD-MM-YYYY"
        from_date = from_date.strftime("%d-%m-%Y")
        to_date = to_date.strftime("%d-%m-%Y")
        
        filename = 'tweets_data.csv'

        run_X_scrapping(search_keyword,from_date,to_date,limit,delay,token,filename)

        df = selection_data(filename)

        # Conditionally apply translation function to the 'Translation' column
        if include_translation:
            df['Translation'] = df.apply(lambda row: translate_text((row['Tweet']), source=row['Detect Language'], target='en'), axis=1)
            df = preprocessing_data(df)
        
        # Conditionally apply sentiment analysis function to the 'Translation' column
        if include_sentiment_analysis:
            df[['Fake Check', 'Sentiment Distilbert']] = df['Translation'].apply(lambda text: pd.Series(analyze_sentiment_distilbert(text, model, tokenizer))).apply(lambda x: x.str.title())
        
        # Conditionally apply VADER sentiment analysis to the 'Translation' column
        if include_sentiment_vader:
            df['Sentiment VADER'] = df['Translation'].apply(analyze_sentiment_vader)
        
        # Conditionally apply TextBlob sentiment analysis to the 'Translation' column
        if include_sentiment_textblob:
            df['Sentiment TextBlob'] = df['Translation'].apply(analyze_sentiment_textblob)
        
        # Save the data to session_state
        st.session_state.data = df

## ............................................... ##
# Check if data is available
if st.session_state.data is not None:  
    with st.container():
        df_cache = st.session_state.data
        st.markdown("### Download Processed Data as CSV")
        st.write("Click the button below to download the processed data as a CSV file.")
        csv_data = convert_df(df_cache)

        # Create a downloadable link
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name='processed_data.csv',
            mime='text/csv',
        )

        with st.expander("See for Table"):
            ## ............................................... ##
            # Display processed data
            st.dataframe(df_cache)

        # Display processed data
        with st.expander("See for Exploratory Data Analysis"):
            ## ............................................... ##
            col1, col2 = st.columns(2)
            with col1:
                ## ............................................... ##
                # Create a new column with a count of 1 for each tweet
                df_date = pd.DataFrame(df_cache['Created Date'])
                df_date['Tweet Count'] = 1

                # Resample the data per second and calculate the count
                data_resampled = df_date.resample('S', on='Created Date')['Tweet Count'].count().reset_index()

                # Create a time series plot with custom styling
                fig = px.line(data_resampled, x='Created Date', y='Tweet Count', title='Tweet Counts Over Time')
                fig.update_xaxes(title_text='Time')
                fig.update_yaxes(title_text='Tweet Count')
                fig.update_layout(xaxis_rangeslider_visible=True)

                # Specify custom dimensions for the chart
                st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=700, height=400)

                ## ............................................... ##
                # Create wordcloud
                try:
                    create_wordcloud(df_cache)
                except Exception as e:
                    logging.error(f" Column Translation Not Available : {str(e)}")

                ## ............................................... ##

            with col2:
                ## ............................................... ##
                # Create a DataFrame to count the number of tweets by language
                language_counts = df_cache['Language'].value_counts().reset_index()
                language_counts.columns = ['Language', 'Tweet Count']

                # Create an attractive Plotly bar chart
                fig = px.bar(language_counts, x='Language', y='Tweet Count', text='Tweet Count', title='Total Tweet by Language')
                fig.update_xaxes(title_text='Language')
                fig.update_yaxes(title_text='Total Tweet')

                # Specify custom dimensions for the chart
                st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=700, height=400)

                ## ............................................... ##
                # Group by Sentiment columns and get the count
                try:
                    sentiment_counts = df_cache[['Sentiment Distilbert', 'Sentiment VADER', 'Sentiment TextBlob']].apply(lambda x: x.value_counts()).T

                    # Reset index to get Sentiment as a column
                    sentiment_counts = sentiment_counts.reset_index()

                    # Melt the DataFrame for easier plotting
                    sentiment_counts = pd.melt(sentiment_counts, id_vars='index', var_name='Sentiment', value_name='Count')

                    # Create the plot
                    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='index', barmode='group', title='Total Tweet per Sentiment')

                    # Specify custom dimensions for the chart
                    st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=700, height=400)

                except Exception as e:
                    logging.error(f" Generate Report error : {str(e)}")

        ## ............................................... ##
        # Display processed data
        with st.expander("See for Analysis with ydata-profiling"):
            st.write("Click the button below to download the processed data as a CSV file.")
            # Show dataset information
            pr = ProfileReport(df_cache)
            #st_profile_report(pr)
