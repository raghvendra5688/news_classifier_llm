# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Code to run a stream lit app hosting the models and running the news classifiers

# %%
# Install necessary libraries for the script to run
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns   
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Configure matplotlib
mpl.rcParams['figure.figsize'] = (12, 6)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['ytick.labelsize'] = 'medium'
warnings.simplefilter('ignore')
mpl.style.use('ggplot')


# %%
def load_model(model_name):
    # Load the saved huggingface model and test on the test dataset
    if model_name == "Google BERT":

        # Load google bert tokenizer
        model_id = "google-bert/bert-base-uncased"  # You can change this to any other model id
        # Load Tokenizer
        google_bert_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the model from the saved directory
        finetuned_model_id = "raghvendramall/bert-uncased-news-classifier"  # You can change this to any other model id
        google_bert_model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_id)

        # Return the model and tokenizer
        return google_bert_model, google_bert_tokenizer
    
    elif model_name == "Modern BERT":

        model_id = "answerdotai/ModernBERT-base"
        
        # Load Tokenizer
        modern_bert_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load the model from the saved directory
        finetuned_model_id = "raghvendramall/modernbert-news-classifier"  # You can change this to any other model id
        # Load the model
        modern_bert_model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_id)  

        # Return the model and tokenizer
        return modern_bert_model, modern_bert_tokenizer



# %%
def main():
    """News Classifier Using Google BERT and ModernBERT"""
    st.title("News Classifier Using Google BERT and ModernBERT")
    st.subheader("ML App with Streamlit")
    html_temp = """
        <div style="background-color:grey;padding:10px">
        <h1 style="color:white;text-align:center;">Streamlit News Classifier App </h1>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    news_text = st.text_area("Enter News Here", "Type Here")
    ml_models = ["Google BERT", "ModernBERT"]
    choice = st.selectbox("Select Model", ml_models)

    if st.button("Classify"):
        if choice == 'Google BERT' and news_text != "":
            st.info("Using Google BERT Model")
            st.text("Original Text::\n{}".format(news_text))
            # Load the model and tokenizer
            google_bert_model, google_bert_tokenizer = load_model("Google BERT")
            predictor = pipeline("text-classification", model=google_bert_model, tokenizer=google_bert_tokenizer)
            prediction = predictor(news_text)

            # Write prediction result to streamlit
            st.write("Predicted Label::\n{}".format(prediction[0]['label']))
            st.write("Predicted Score::\n{}".format(prediction[0]['score']))
            st.success("News classified successfully!")
        elif choice == 'ModernBERT' and news_text != "":
            st.info("Using ModernBERT Model")
            st.text("Original Text::\n{}".format(news_text))
            # Load the model and tokenizer
            modern_bert_model, modern_bert_tokenizer = load_model("Modern BERT")
            # Use the pipeline for text classification
            predictor = pipeline("text-classification", model=modern_bert_model, tokenizer=modern_bert_tokenizer)
            prediction = predictor(news_text)

            # Write prediction result to streamlit
            st.write("Predicted Label::\n{}".format(prediction[0]['label']))
            st.write("Predicted Score::\n{}".format(prediction[0]['score']))
            st.success("News classified successfully!")
        else:
            st.error("Please enter some text to classify or select a model.")

    # Display the sidebar
    st.sidebar.text("Built with Streamlit")
    # Display additional information
    st.sidebar.text("This app classifies news articles using pre-trained BERT models.")
    st.sidebar.text("Select a model and enter the news text to get predictions.")
    st.sidebar.text("Developed by \n [Dr. Raghvendra Mall]")
    st.sidebar.text("Version 1.0")
    st.sidebar.text("Contact: [raghvendra5688@gmail.com]")
    return


# %%
if __name__ == '__main__':
    main()

# %%
