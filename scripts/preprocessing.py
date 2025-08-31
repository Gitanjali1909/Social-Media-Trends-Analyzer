import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):   
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)          
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub(r'\d+', '', text)                           
    text = " ".join([word for word in text.split() if word not in stop_words]) 
    return text
