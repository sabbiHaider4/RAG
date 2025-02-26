import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from fastapi import FastAPI, HTTPException
import openai
import os

app = FastAPI()

# Sample Texts for FAISS Indexing
texts = [
    "The capital of France is Paris.",
    "Python is a popular programming language for AI.",
    "RAG stands for Retrieval-Augmented Generation."
]

# Generate OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Create and Save FAISS Index
vector_store = FAISS.from_texts(texts, embeddings)
vector_store.save_local("faiss_index")

# ✅ Load FAISS Index Safely
vector_store = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

# ✅ OpenAI API Key (Set this in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/ask/")
def ask_rag(question: str):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(question, k=2)
    context = " ".join([doc.page_content for doc in docs])

    # Generate response using OpenAI
    prompt = f"""
            You must answer the question **ONLY** using the provided context.
            Context: {context}            
            Question: {question}
            If the answer is not in the context, say: "I don’t know based on the provided information."
            """
    response = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

    return {"answer": response.choices[0].message.content.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
