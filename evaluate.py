import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from vector import retriever

df = pd.read_csv("yelp_reviews.csv")
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf = LogisticRegression().fit(X_train_tfidf, y_train)
accuracy = clf.score(X_test_tfidf, y_test)
print(f"Sentiment Accuracy: {accuracy:.2f}")

def precision_at_k(retrieved_docs, question, k=5):
    relevant = sum(1 for doc in retrieved_docs if "pizza" in doc.page_content.lower() and question.lower().find("pizza") != -1)
    return relevant / k if k > 0 else 0

question = "What's the best pizza in town?"
reviews = retriever.invoke(question)
precision = precision_at_k(reviews, question)
print(f"Retrieval Precision@5: {precision:.2f}")