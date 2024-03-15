import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string  # Added import statement for string module

# Load the saved vectorizer and naive bayes model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # converting to lowercase
    text = nltk.word_tokenize(text)  # tokenize

    # remove special characters and retaining alphabetic words
    text = [word for word in text if word.isalnum()]

    # removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # applying stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Saving streamlit code
st.title("Email spam classifier")
input_sms = st.text_area("Enter Message")

if st.button('Predict'):
    # Preprocess
    transform_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transform_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
