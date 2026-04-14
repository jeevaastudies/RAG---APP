import os
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from retriever import get_retriever

# Ensure your API key is set in your environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def build_qa_chain():
    """
    Builds the RetrievalQA chain, connecting the LLM to the database.
    """
    # 1. Connect to the LLM (OpenAI API)
    # Using ChatOpenAI here. If you prefer Claude, use ChatAnthropic.
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o") 
    
    # 2. Initialize the retriever (Tune Top-K here)
    retriever = get_retriever(top_k=5)
    
    # 3. Build the LangChain RetrievalQA chain
    # Setting return_source_documents=True is vital for testing relevance.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" injects all retrieved chunks into the prompt
        retriever=retriever,
        return_source_documents=True 
    )
    
    return qa_chain

# --- Testing Block ---
if __name__ == "__main__":
    print("Initializing the AI Brain...")
    chain = build_qa_chain()
    
    # Test that answers are accurate and relevant
    test_query = "What is the main objective of our system?"
    print(f"\nUser Query: {test_query}")
    
    # Execute the chain
    response = chain.invoke({"query": test_query})
    
    print("\n--- AI Response ---")
    print(response["result"])
    
    print("\n--- Context Retrieved (For Accuracy Testing) ---")
    for i, doc in enumerate(response["source_documents"], 1):
        # Print a snippet of the retrieved chunks to verify relevance
        print(f"\nChunk {i}: {doc.page_content[:200]}...")