import os
import streamlit as st
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.retrievers import EnsembleRetriever, BM25Retriever
import numpy as np
import sys
import json
from BCEmbedding import RerankerModel
from config_loader import load_config

# Function to set up environment variables for Azure OpenAI API
def setup_environment(api_version, api_key):
    """
    Sets up the environment variables required for Azure OpenAI API.
    Parameters:
    - api_version (str): Version of the Azure OpenAI API.
    - api_key (str): API key for authentication.
    """
    os.environ['OPENAI_API_VERSION'] = api_version
    os.environ['AZURE_OPENAI_API_KEY'] = api_key


# Function to initialize models and retrievers
def models():
    """
    Initializes the necessary models and retrievers for processing queries.
    Returns:
    - llm (AzureChatOpenAI): Azure ChatGPT model.
    - ensemble_retriever (EnsembleRetriever): Hybrid retriever combining dense and sparse methods.
    """

    config = load_config()  # Load configuration settings from config.yaml
    print(f'Configs: {config}')

    # Set up Azure OpenAI API environment
    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], 
                      api_key=config['azure_openai']['azure_api_key'])

    # Initialize the Azure OpenAI language model (LLM)
    llm = AzureChatOpenAI(
        azure_deployment=config['azure_openai']['llm']['azure_deployment_llm'], 
        azure_endpoint=config['azure_openai']['azure_api_endpoint']
    )

    # Initialize the embeddings model for dense similarity search
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'], 
        azure_endpoint=config['azure_openai']['azure_api_endpoint'], 
        chunk_size=100
    )

    # Load Chroma for dense retrieval and similarity search
    db = Chroma(
        persist_directory="./chroma_db",  # Path to the persistent Chroma database
        embedding_function=embeddings    # Embedding model for similarity search
    )

    # Retrieve all documents stored in ChromaDB
    documents = db.similarity_search("*")  # Retrieve all documents; "*" acts as a wildcard query
    doc_texts = [doc.page_content for doc in documents]  # Extract text content from documents

    # Initialize a BM25 retriever for sparse text-based retrieval
    bm25_retriever = BM25Retriever.from_texts(doc_texts)

    # Combine Chroma and BM25 retrievers into an ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[db.as_retriever(), bm25_retriever],  # List of retrievers to combine
        weights=[0.6, 0.4]  # Weights for dense (Chroma) and sparse (BM25) retrievers
    )

    return llm, ensemble_retriever


# Function to initialize the reranker model
def initialize_reranker_model():
    """
    Initializes the BCEmbedding reranker model for re-ranking retrieved passages.
    Returns:
    - RerankerModel: Initialized reranker model.
    """
    return RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")


# Function to process a query and generate a response
def process_query(query, llm, ensemble_retriever, reranker_model):
    """
    Processes a user query using hybrid retrieval, reranking, and LLM response generation.
    Parameters:
    - query (str): The user query.
    - llm (AzureChatOpenAI): Azure ChatGPT model.
    - ensemble_retriever (EnsembleRetriever): Hybrid retriever combining dense and sparse methods.
    - reranker_model (RerankerModel): Model for reranking retrieved documents.
    Returns:
    - response (str): The final generated response.
    """

    # Retrieve relevant documents using the ensemble retriever
    relevant_docs = ensemble_retriever.get_relevant_documents(query)

    # Extract document texts for reranking
    docs = [doc.page_content for doc in relevant_docs]

    # Use the reranker model to rerank the retrieved documents
    reranked_docs = reranker_model.rerank(query, docs)

    # Extract reranked passages for generating the context
    reranked_passages = reranked_docs.get("rerank_passages", [])

    # Combine reranked passages into a single context string
    context = "\n\n".join(reranked_passages)

    # Define a prompt template for generating a response
    prompt = PromptTemplate(
        input_variables=["human_input", "context"],
        template="""You are an AI assistant. Given the following context and question, provide a clear and concise response.

        Context:
        {context}

        Question:
        {human_input}

        Answer:"""
    )

    # Create a language model chain for generating a response
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate the response using the LLM chain
    response = chain.invoke({'human_input': query, 'context': context})
    return response['text'].strip()


# Main function
def main():
    """
    Main function for command-line query processing.
    Initializes models and processes a query if provided as a command-line argument.
    """
    
    # Initialize models and retrievers
    llm, ensemble_retriever = models()
    reranker_model = initialize_reranker_model()

    # Process query if provided as a command-line argument
    if len(sys.argv) > 1:
        query = sys.argv[1]  # Retrieve query from command-line arguments
        result = process_query(query, llm, ensemble_retriever, reranker_model)
        print(result)  # Print the generated response


# Entry point of the script
if __name__ == "__main__":
    main()
