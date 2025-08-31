import joblib
from scripts.preprocessing import clean_text

model = joblib.load("models/logreg_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

print(predict_sentiment("I love this app!"))
