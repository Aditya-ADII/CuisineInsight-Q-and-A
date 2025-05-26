from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import joblib

df = pd.read_csv("yelp_reviews.csv")
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf = LogisticRegression().fit(X_train_tfidf, y_train)
accuracy = clf.score(X_test_tfidf, y_test)
print(f"Sentiment Accuracy: {accuracy:.2f}")

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
db_location = "./chroma_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        sentiment = "Positive" if clf.predict(vectorizer.transform([row["Review"]]))[0] == 1 else "Negative"
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": float(row["Rating"]), "date": row["Date"], "sentiment": sentiment},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        if i % 1000 == 0:
            print(f"Processed {i} documents")
    
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    # Upsert in batches of 5000
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"Upserted batch {i // batch_size + 1}")

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(clf, "sentiment_classifier.pkl")