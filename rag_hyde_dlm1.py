import os
import streamlit as st
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
# from langchain.embeddings import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from BCEmbedding import RerankerModel
import sys
from config_loader import load_config

# Function to set up environment variables
def setup_environment(api_version, api_key):
    """
    Sets up the necessary environment variables for Azure OpenAI API.
    Parameters:
    - api_version (str): The API version to use.
    - api_key (str): The API key for authentication.
    """
    os.environ['OPENAI_API_VERSION'] = api_version
    os.environ['AZURE_OPENAI_API_KEY'] = api_key

# Function to initialize the language model, embeddings, and Chroma database
def llm_and_db():
    """
    Initializes the language model, embeddings, and Chroma database.
    Returns:
    - llm (AzureChatOpenAI): Initialized Azure LLM.
    - embeddings (AzureOpenAIEmbeddings): Azure embeddings model for similarity search.
    - db (Chroma): Chroma database for dense retrieval.
    """

    config = load_config()  # Load configuration settings from config.yaml
    print(f'Configs: {config}')

    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], 
                      api_key=config['azure_openai']['azure_api_key'])

    # Initialize Azure LLM
    llm = AzureChatOpenAI(
        azure_deployment=config['azure_openai']['llm']['azure_deployment_llm'],
        azure_endpoint=config['azure_openai']['azure_api_endpoint']
    )
    
    # Initialize Azure embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'],
        azure_endpoint=config['azure_openai']['azure_api_endpoint'],
        chunk_size=100
    )

    # Initialize Chroma database for dense similarity retrieval
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return llm, embeddings, db

# HyDE Function to generate a hypothetical document
def generate_hypothetical_answer(query, llm):
    """
    Generates a hypothetical answer for the query to enhance retrieval performance.
    Parameters:
    - query (str): The user query.
    - llm (AzureChatOpenAI): The initialized Azure LLM.
    Returns:
    - hypothetical_answer (str): The generated hypothetical answer.
    """
    # Define a prompt for generating a hypothetical answer
    prompt = PromptTemplate(
        input_variables=["human_input"],
        template="You are an AI assistant. Generate a hypothetical answer based on the following query:{human_input}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate the hypothetical answer
    hypothetical_answer = chain.invoke({'human_input': query})['text']
    
    return hypothetical_answer.strip()

# Function to initialize the reranker model
def initialize_reranker_model():
    """
    Initializes the reranker model for document ranking.
    Returns:
    - RerankerModel: The initialized reranker model.
    """
    return RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")

# Function to process query and generate a response with HyDE
def process_query(query, llm, embeddings, db, reranker_model):
    """
    Processes a user query by generating a hypothetical answer (HyDE),
    retrieving relevant documents, reranking them, and generating a response.
    Parameters:
    - query (str): The user query.
    - llm (AzureChatOpenAI): The initialized Azure LLM.
    - embeddings (AzureOpenAIEmbeddings): The embeddings model for similarity search.
    - db (Chroma): Chroma database for document retrieval.
    - reranker_model (RerankerModel): Model to rerank retrieved documents.
    Returns:
    - response (str): The generated response.
    """
    # Generate a hypothetical answer for query enhancement
    hypothetical_answer = generate_hypothetical_answer(query, llm)
    
    # Convert the hypothetical answer into embeddings
    hypothetical_embedding = embeddings.embed_query(hypothetical_answer)
    
    # Retrieve similar documents using the hypothetical embedding
    retrieved_docs = db.similarity_search_by_vector(hypothetical_embedding)

    # Extract document content
    docs = [doc.page_content for doc in retrieved_docs]
    # Rerank the documents based on the query
    reranked_docs = reranker_model.rerank(query, docs)

    # Extract passages from the reranked documents
    reranked_passages = reranked_docs.get("rerank_passages", [])

    # Construct context by combining reranked passages
    context = "\n\n".join(reranked_passages)

    # Define a prompt template for final answer generation
    prompt = PromptTemplate(
        input_variables=["human_input", "context"],
        template="""You are an AI having a conversation with a human.
                    Given the following extracted parts of a long document and a question, create a final answer.

                    Context: {context}

                    Human: {human_input}
                    AI:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate the final response
    chain_res = chain.invoke({'human_input': query, 'context': context})
    return chain_res['text'].strip()
    

# Main function for script execution
def main():
    """
    Main function for automated script usage. Initializes LLM, embeddings,
    and processes a query passed via the command line.
    """
    # Initialize LLM, embeddings, and database
    llm, embeddings, db = llm_and_db()
    reranker_model = initialize_reranker_model()

    # Check if a query is provided as a command-line argument
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = process_query(query, llm, embeddings, db, reranker_model)
        print(result)

# Entry point for the script
if __name__ == "__main__":
    main()
