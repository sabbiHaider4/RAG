import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Initialize FastAPI app
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation with FastAPI", version="1.0")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key

# Initialize FAISS vector store
vector_store = None


# Load and process documents
def load_documents(file_path: str):
    if file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    return []


def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    # Verify that embeddings are being created correctly
    if not chunks:
        raise ValueError("Document chunks are empty. Check the input document format.")

    # Generate embeddings for the chunks
    chunk_texts = [chunk.page_content for chunk in chunks]  # Ensure we're using the actual content
    embeddings_list = embeddings.embed_documents(chunk_texts)

    # Ensure embeddings are correctly generated
    if not embeddings_list:
        raise ValueError("Failed to generate embeddings for document chunks.")

    return FAISS.from_documents(chunks, embeddings)



# Upload and Process File Endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global vector_store

    # Save uploaded file
    file_path = f"uploaded_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process documents
    documents = load_documents(file_path)
    if not documents:
        return {"error": "Unsupported file type or empty file"}

    try:
        vector_store = process_documents(documents)
    except Exception as e:
        return {"error": str(e)}

    return {"message": f"{file.filename} processed successfully"}


# Ask a Question Endpoint
@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    global vector_store

    if vector_store is None:
        return {"error": "No document uploaded. Please upload a file first."}

    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4")  # You can change this to gpt-3.5 if needed
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    response = qa_chain.run(query)
    return {"query": query, "answer": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)