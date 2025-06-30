from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
import subprocess
import importlib
import os
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xgboost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-review-data.pkl"

    def __init__(self):
        self._ensure_xgboost()

        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))

        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)

        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))

        self.data = pd.read_csv("dataset/sample30.csv")

        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _ensure_xgboost(self):
        try:
            import xgboost
        except ModuleNotFoundError:
            print("Installing xgboost at runtime...")
            subprocess.check_call(["pip", "install", "xgboost"])
            importlib.import_module("xgboost")

    def getRecommendationByUser(self, user):
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    def getSentimentRecommendations(self, user):
        if user in self.user_final_rating.index:
            recommendations = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)]

            X = self.vectorizer.transform(filtered_data["reviews_text_cleaned"].astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)

            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(
                lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count()
            )
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)

            sorted_products = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]

            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        else:
            print(f"User name {user} doesn't exist")
            return None

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    def preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub(r"\[\s*\w*\s*\]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\S*\d\S*", "", text)
        text = self.lemma_text(text)
        return text

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha() and word not in self.stop_words]
        return " ".join(words)

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(self.remove_stopword(text)))
        words = [
            self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(tag[1]))
            for tag in word_pos_tags
        ]
        return " ".join(words)
