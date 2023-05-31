# from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
# from dotenv import dotenv_values
import requests
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from statsmodels.graphics.gofplots import qqplot
import plotly.graph_objs as go

nltk.download('wordnet')
load_dotenv()

# Due to the data of the two contries having the same structure, the two dataframes shares the same functions.

# CONSTANTS USED THROUGH OUT THE 
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
URL = "https://api.worldnewsapi.com/search-news"

def request_news2023(sourceCountry):
    '''Retrive the news related to construction sector from November 2022 to December 2023'''
    data = []
    requestParams = {
        "earliest-publish-date": "2022-11-01T00:00:00Z",
        "latest-publish-date": "2023-04-30T23:59:00Z",
        "source-countries": sourceCountry,
        "number": 100,
        "api-key": NEWS_API_KEY,
        "text": "construction sector",
        "offset": 0,
        "language": "en"
    }
    # Python doesn't have a do-while loop. So the loop below emulates how a do-while loop executes. Do-while loop was chosen so that there is an assurance that the code was executed at least one time.
    while True:
        response = send_request(requestParams)
        data = data + response["news"]
        
        print("DATA LENGTH: ", len(data), response["available"])
        
        # https://www.scaler.com/topics/check-if-key-exists-in-dictionary-python/
        if("offset" in response):
            requestParams["offset"] += 100
        
        if len(data) >= response["available"]:
            break
    
    return data

# There are other libraries that can handle HTTP requests as well. One of which is urllib3. Due to the simplicity and ease of implementation, the researcher chose to use request package.
def send_request(params):
    '''This function sends a GET HTTP request to a specified URL'''
    response = requests.get(url = URL, params = params)
    data = response.json()
#     print(response.headers)
    return data

def preprocess_data(data):
    '''Function that removes the keys that are not needed and removes the duplicates in the data'''
    keysToRemove = ["summary", "text", "url", "image", "author", "language"]
    # Loop through all the array
    return [ dict(t) for t in set([tuple(d.items()) for d in list(map(lambda x: { k: v for k, v in x.items() if k not in keysToRemove }, data))])]

def save_to_csv(data, filename):
    # https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr
    # https://stackoverflow.com/questions/918154/relative-paths-in-python
    file_dir = os.path.realpath('...')
    print(file_dir)
    csv_folder = 'datasets'
    file_path = os.path.join(file_dir, csv_folder, filename)

    pd.DataFrame.from_dict(data).to_csv(file_path, index=False)
    
def remove_punctuation(text):
    # https://stackoverflow.com/questions/43038139/regex-removing-all-punctuation-but-leave-decimal-points-and-hyphenated-words
    pattern = r'[^a-zA-Z0-9_.-]|(?<!\d)\.(?!\d)|(?<!\w)-(?!\w)'
    return re.sub(pattern, '  ', text)

def lemmatize_text(text):
    # Textblob also has a lemmatizer library but NLTK lemmatizer was used since wordnet has also been downloaded. This will also save some memory (albeit small) but still an improvement for an overall performance.
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    stop = stopwords.words('english')
    return " ".join(x for x in text.split() if x not in stop)

def lower_case(text):
    return text.lower()

def count_words(x):
    return len(str(x).split())

def set_sentiment_label(x):
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "neutral"

def create_qqplot_current_month(data, month, country):
    qqplot_data = qqplot(data[data["month"] == month]["sentiment"], line='s').gca().lines
    
    fig = go.Figure()
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })


    fig['layout'].update({
        'title': f'Quantile-Quantile Plot for Construction-related News Headlines for {country} for the {month}',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Observable Values'
        },
        'showlegend': False,
        'width': 1000,
        'height': 700,
    })

    fig.show()
    
def create_inf_table(headers, data, title):
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers),
                cells=dict(values=data))
            ],
        layout = go.Layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "middle"
            },
            width = 900,
            height = 800
        )
    )
    fig.show()
    
def check_anderson_darling_result(result):
    significance_level = 0.05
    
    test_statistic = result.statistic
    critical_values = result.critical_values
    
    # Compare the test statistic with critical values
    for i, crit_value in enumerate(critical_values):
        if test_statistic > crit_value:
            return f"At significance level {significance_level}, reject the null hypothesis."
            break
            
    return f"At significance level {significance_level}, do not reject the null hypothesis."
            