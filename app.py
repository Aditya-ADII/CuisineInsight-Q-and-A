from flask import Flask, request, render_template
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import joblib
import torch
import re
import gc  # Added import

app = Flask(__name__)
model = OllamaLLM(model="llama3.2:3b")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
clf = joblib.load("sentiment_classifier.pkl")
template = """
You are an expert in answering questions about restaurants.
Reviews: {reviews}
Question: {question}
Include sentiment analysis.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def precision_at_k(retrieved_docs, question, k=5):
    food_keywords = {
        "pizza": ["pizza", "pizzeria", "slice", "crust", "toppings"],
        "mexican": ["mexican", "taco", "salsa", "fajita", "enchilada"],
        "italian": ["italian", "pasta", "spaghetti", "lasagna"],
        "soup": ["soup", "broth", "stew"],
        "vegetarian": ["vegetarian", "vegan", "veggie"],
        "spicy": ["spicy", "hot", "chili"],
        "salad": ["salad", "greens", "caesar"],
        "ambiance": ["ambiance", "atmosphere", "vibe", "setting"]
    }
    question_lower = question.lower()
    query_keywords = []
    for category, keywords in food_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            query_keywords.extend(keywords)
    query_keywords.extend(["best", "good", "perfect", "affordable"])
    relevant = 0
    for doc in retrieved_docs:
        content_lower = doc.page_content.lower()
        is_query_specific = any(keyword in content_lower for keyword in query_keywords)
        is_positive = doc.metadata.get("sentiment") == "Positive" and doc.metadata.get("rating", 0) >= 3.5
        if is_query_specific and is_positive:
            relevant += 1
        print(f"Review: {content_lower[:100]}... | Query-Specific: {is_query_specific} | Positive: {is_positive}")
    return relevant / k if k > 0 else 0

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"]
        reviews = retriever.invoke(question)
        sentiment = [clf.predict(vectorizer.transform([doc.page_content]))[0] for doc in reviews]
        sentiment_summary = "Mostly positive" if sum(sentiment) / len(sentiment) > 0.5 else "Mostly negative"
        precision = precision_at_k(reviews, question)
        result = chain.invoke({"reviews": reviews, "question": question})
        torch.cuda.empty_cache()
        gc.collect()
        return render_template("index.html", result=result, sentiment=sentiment_summary, precision=precision)
    return render_template("index.html", result=None, sentiment=None, precision=None)

if __name__ == "__main__":
    app.run(debug=True)