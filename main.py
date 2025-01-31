import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import nltk
from keras.models import load_model
import numpy as np

model = load_model('model.h5')  # Use Keras to load the model
tf_idf_vector = joblib.load('tfidf.pkl')

stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def predict_sentiment(review):
    cleaned_review = re.sub('<.*?>', '', review)
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stem.stem(word) for word in filtered_review]
    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])
    sentiment_prediction = model.predict(tfidf_review)[0]
    sentiment_class = np.argmax(sentiment_prediction)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    return sentiment_labels[sentiment_class]
st.title('Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')

if st.button('Predict Sentiment'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:", predicted_sentiment)

# %%


# %%
