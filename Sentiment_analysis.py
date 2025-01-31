# %%
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense

# %%
df =pd.read_csv("twitter_training.csv")

# %%
df

# %%
df = df.drop(['2401','Borderlands'],axis=1)

# %%
df.columns

# %%
new_coloumns = ['sentiment','text']

# %%
df.columns=new_coloumns

# %%
df

# %%
df['sentiment'].unique()

# %%
#le = LabelEncoder()

# %%
#df['sentiment']=le.fit_transform(df['sentiment'])
#df['text']=le.fit_transform(df['text'])

# %%
df['text'] = df['text'].astype(str)

# %%
df['clean_text'] = df['text'].apply(lambda x: re.sub("<.*?>", "", x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', "", x))
df['clean_text'] = df['clean_text'].str.lower()

# %%
df['tokenize_text'] = df['clean_text'].apply(lambda x: word_tokenize(x))

# %%
stop_words = set(stopwords.words('english'))
df['filtered_text'] = df['tokenize_text'].apply(lambda x: [word for word in x if word not in stop_words])

# %%
stem = PorterStemmer()
df['stem_text'] = df['filtered_text'].apply(lambda x: [stem.stem(word) for word in x])

# %%
lemma = WordNetLemmatizer()
df['lemma_text'] = df['filtered_text'].apply(lambda x: [lemma.lemmatize(word) for word in x])

# %%
X = df['stem_text'].apply(lambda x: ' '.join(x))
y = df['sentiment']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# %%
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# %%
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# %%
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(4, activation="softmax")  # Output layer for 4 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
model.fit(X_train, y_train, epochs=10)

# %%
model.save('model.h5')
joblib.dump(tfidf, 'tfidf.pkl')

# %%
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import nltk
from keras.models import load_model
import numpy as np

# Load model and vectorizer
model = load_model('model.h5')  # Use Keras to load the model
tf_idf_vector = joblib.load('tfidf.pkl')

# Initialize NLP components
stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def predict_sentiment(review):
    # Preprocess the review
    cleaned_review = re.sub('<.*?>', '', review)
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stem.stem(word) for word in filtered_review]
    
    # Transform review to TF-IDF features
    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])
    
    # Predict sentiment
    sentiment_prediction = model.predict(tfidf_review)[0]
    
    # Determine the class with the highest probability
    sentiment_class = np.argmax(sentiment_prediction)
    
    # Define sentiment labels
    sentiment_labels = ["Negative", "Neutral", "Positive", "Irrelevant"]  # Update with actual labels
    return sentiment_labels[sentiment_class]

# Streamlit UI
st.title('Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')

if st.button('Predict Sentiment'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:", predicted_sentiment)


# %%
!ipynb-py-convert Untitled.ipynb Untitled.py

# %%
