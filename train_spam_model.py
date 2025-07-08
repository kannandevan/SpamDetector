import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


df =pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1','v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_counts, y_train)
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer trained and saved successfully!")