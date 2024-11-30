import os
import numpy as np
import streamlit as st
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.retrievers import EnsembleRetriever, BM25Retriever
import sys
from FlagEmbedding import FlagReranker
from config_loader import load_config

# Function to set up environment variables for Azure OpenAI API
def setup_environment(api_version, api_key):
    """
    Sets up environment variables required for accessing the Azure OpenAI API.
    Parameters:
    - api_version (str): The API version to use.
    - api_key (str): The API key for authentication.
    """
    os.environ['OPENAI_API_VERSION'] = api_version
    os.environ['AZURE_OPENAI_API_KEY'] = api_key


# Function to initialize the models and retrievers
def models():
    """
    Initializes the language model, embeddings, and retrievers.
    Returns:
    - llm (AzureChatOpenAI): The initialized Azure ChatGPT model.
    - ensemble_retriever (EnsembleRetriever): An ensemble retriever combining dense and sparse retrieval methods.
    """
    config = load_config()  # Load configuration settings from config.yaml

    # Set up environment for Azure OpenAI API
    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], 
                      api_key=config['azure_openai']['azure_api_key'])

    # Initialize Azure LLM (ChatGPT)
    llm = AzureChatOpenAI(
        azure_deployment=config['azure_openai']['llm']['azure_deployment_llm'], 
        azure_endpoint=config['azure_openai']['azure_api_endpoint']
    )

    # Initialize embeddings for similarity search
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'], 
        azure_endpoint=config['azure_openai']['azure_api_endpoint'], 
        chunk_size=100
    )
    
    # Set up Chroma for dense retrieval
    db = Chroma(
        persist_directory="./chroma_db",  # Directory containing Chroma database
        embedding_function=embeddings    # Embedding function for similarity search
    )

    # Retrieve all documents from ChromaDB
    documents = db.similarity_search("*")  # Wildcard to retrieve all documents
    doc_texts = [doc.page_content for doc in documents]  # Extract text content from documents

    # Initialize BM25 retriever for sparse text-based retrieval
    bm25_retriever = BM25Retriever.from_texts(doc_texts)

    # Combine Chroma and BM25 retrievers into an ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[db.as_retriever(), bm25_retriever],  # List of retrievers
        weights=[0.6, 0.4]  # Weights for dense (Chroma) and sparse (BM25) retrievers
    )

    return llm, ensemble_retriever


# Function to initialize the reranker model
def initialize_reranker_model():
    """
    Initializes the FlagEmbedding reranker model.
    Returns:
    - FlagReranker: The initialized reranker model.
    """
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)  # Use a lightweight version of the model


# Function to process the query and generate a response
def process_query(query, llm, ensemble_retriever, reranker_model):
    """
    Processes a user query and generates a response using retrieval, reranking, and language generation.
    Parameters:
    - query (str): The user query.
    - llm (AzureChatOpenAI): The initialized Azure ChatGPT model.
    - ensemble_retriever (EnsembleRetriever): The hybrid retriever.
    - reranker_model (FlagReranker): The reranker model for ranking retrieved documents.
    Returns:
    - response (str): The generated response.
    """
    # Step 1: Retrieve relevant documents using the ensemble retriever
    relevant_docs = ensemble_retriever.get_relevant_documents(query)

    # Step 2: Extract text content from each retrieved document for reranking
    reranking_docs = [doc.page_content for doc in relevant_docs]
    
    # Prepare query-document pairs for reranking
    qd_list = [[query, doc] for doc in reranking_docs]
    
    # Step 3: Calculate scores for each document-query pair using the reranker model
    scores = reranker_model.compute_score(qd_list)

    # Step 4: Sort documents by their scores in descending order
    doc_score_pairs = sorted(zip(reranking_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [pair[0] for pair in doc_score_pairs]

    # Step 5: Construct the context using the top reranked documents
    context = "\n".join(reranked_docs)

    # Step 6: Define the prompt template for generating a response
    prompt = PromptTemplate(
        input_variables=["human_input", "context"],
        template="""You are an AI assistant. Given the following context and question, provide a clear and concise response.

        Context:
        {context}

        Question:
        {human_input}

        Answer:"""
    )

    # Step 7: Use LLM chain to generate a response
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({'human_input': query, 'context': context})
    return response['text'].strip()


# Main function
def main():
    """
    Main function to initialize models and process a query provided via the command line.
    """
    # Step 1: Initialize models and retrievers
    llm, ensemble_retriever = models()
    reranker_model = initialize_reranker_model()

    # Step 2: Process the query if provided as a command-line argument
    if len(sys.argv) > 1:
        query = sys.argv[1]  # Retrieve the query from the command line
        result = process_query(query, llm, ensemble_retriever, reranker_model)
        print(result)  # Print the generated response


# Entry point for the script
if __name__ == "__main__":
    main()
