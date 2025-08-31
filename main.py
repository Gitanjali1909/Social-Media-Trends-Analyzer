import pandas as pd
from scripts.preprocessing import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

dataset_choice = "reddit"

if dataset_choice == "youtube":
    df = pd.read_csv("data/youtubecomment.csv")
    df = df.rename(columns={"Comment": "Comment", "Sentiment": "Sentiment"})
elif dataset_choice == "reddit":
    df = pd.read_csv("data/Reddit_Data.csv")
    df = df.rename(columns={"clean_comment": "Comment", "category": "Sentiment"})
    df['Sentiment'] = df['Sentiment'].map({-1: "negative", 0: "neutral", 1: "positive"})

df['cleaned'] = df['Comment'].apply(clean_text)

X = df['cleaned']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

joblib.dump(model, "models/logreg_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

st.set_page_config(page_title="Social Media Sentiment Analyzer", page_icon="💬")
st.title("💬 Social Media Sentiment Analyzer")
st.write(f"Model trained on **{dataset_choice.upper()}** dataset")

user_input = st.text_area("Enter a comment/post:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec).max() * 100

        st.subheader("Prediction:")
        if prediction == "positive":
            st.success(f"😊 Positive ({prob:.2f}% confidence)")
        elif prediction == "negative":
            st.error(f"😠 Negative ({prob:.2f}% confidence)")
        else:
            st.info(f"😐 Neutral ({prob:.2f}% confidence)")