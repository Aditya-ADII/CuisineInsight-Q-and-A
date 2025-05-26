from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import torch
import re
import gc  # Added import

model = OllamaLLM(model="llama3.2:3b")
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

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question == "q":
        torch.cuda.empty_cache()
        gc.collect()
        break
    
    reviews = retriever.invoke(question)
    precision = precision_at_k(reviews, question)
    print(f"Retrieval Precision@5: {precision:.2f}")
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)

    torch.cuda.empty_cache()
    gc.collect()