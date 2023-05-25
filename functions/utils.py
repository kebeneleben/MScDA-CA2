# from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
# from dotenv import dotenv_values
import requests
import pandas as pd
from pandas.tseries.offsets import MonthEnd

load_dotenv()

# CONSTANTS USED THROUGH OUT THE 
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
URL = "https://api.worldnewsapi.com/search-news"

def request_news2023(sourceCountry):
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

def send_request(params):
    '''This function sends a GET HTTP request to a specified URL'''
    response = requests.get(url = URL, params = params)
    data = response.json()
#     print(response.headers)
    return data

def preprocess_data(data):
    '''Function that removes the keys that are not needed and removes the duplicates in the data'''
    keysToRemove = ["summary", "text", "url", "image", "author", "language"]
    return [ dict(t) for t in set([tuple(d.items()) for d in list(map(lambda x: { k: v for k, v in x.items() if k not in keysToRemove }, data))])]

def save_to_csv(data, filename):
    # https://stackoverflow.com/questions/48745333/using-pandas-how-do-i-save-an-exported-csv-file-to-a-folder-relative-to-the-scr
    # https://stackoverflow.com/questions/918154/relative-paths-in-python
    file_dir = os.path.realpath('...')
    print(file_dir)
    csv_folder = 'datasets'
    file_path = os.path.join(file_dir, csv_folder, filename)

    pd.DataFrame.from_dict(data).to_csv(file_path, index=False)