import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("dataset/bec_dataset.csv")

print(df.head())
print(df.columns)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectorized, y_train)

predictions = model.predict(X_test_vectorized)

print(classification_report(y_test, predictions))

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/bec_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved!")