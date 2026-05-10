import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import streamlit as st

def authenticate_kaggle():
    # 1. Try Streamlit Secrets (Cloud) first, then local .env
    try:
        os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
        os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
    except:
        load_dotenv()

    api = KaggleApi()
    api.authenticate()
    return api

def download_data():
    api = authenticate_kaggle()
    dataset = 'rikdifos/credit-card-approval-prediction'
    path = './data'
    
    if not os.path.exists(path):
        os.makedirs(path)
        api.dataset_download_files(dataset, path=path, unzip=True)
    return path