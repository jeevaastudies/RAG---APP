import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def get_retriever(persist_directory: str = "../chroma_db", top_k: int = 4):
    """
    Connects to ChromaDB and returns a configured LangChain retriever.
    """
    # Initialize the same embedding model used during document ingestion
    embeddings = OpenAIEmbeddings()
    
    # Load the persisted Chroma database
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Create the retriever. This is where you tune the Top-K chunks.
    # search_kwargs={"k": top_k} dictates exactly how many documents to fetch.
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    return retriever