import os
import sys
import streamlit as st
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from FlagEmbedding import FlagReranker
from config_loader import load_config

# Function to set up the environment variables for Azure OpenAI
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
    Initializes the language model, embeddings, and Chroma database for document retrieval.
    Returns:
    - llm (AzureChatOpenAI): Azure LLM for generating responses.
    - embeddings (AzureOpenAIEmbeddings): Embeddings model for similarity searches.
    - db (Chroma): Chroma database for dense retrieval of documents.
    """

    # Load configuration settings from config.yaml
    config = load_config()
    print(f'Configs: {config}')

    setup_environment(
        api_version=config['azure_openai']['llm']['azure_api_version_llm'], 
        api_key=config['azure_openai']['azure_api_key']
    )

    # Initialize Azure language model
    llm = AzureChatOpenAI(
        azure_deployment=config['azure_openai']['llm']['azure_deployment_llm'],
        azure_endpoint=config['azure_openai']['azure_api_endpoint']
    )
    
    # Initialize Azure embeddings for similarity search
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'],
        azure_endpoint=config['azure_openai']['azure_api_endpoint'],
        chunk_size=100
    )

    # Initialize Chroma database
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return llm, embeddings, db

# Function to generate a hypothetical document for query enhancement (HyDE)
def generate_hypothetical_answer(query, llm):
    """
    Generates a hypothetical answer to improve query performance.
    Parameters:
    - query (str): The user query.
    - llm (AzureChatOpenAI): The language model used for generating the answer.
    Returns:
    - str: A hypothetical answer based on the query.
    """
    # Define the prompt for hypothetical answer generation
    prompt = PromptTemplate(
        input_variables=["human_input"],
        template="You are an AI assistant. Generate a hypothetical answer based on the following query: {human_input}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate the hypothetical answer
    hypothetical_answer = chain.invoke({'human_input': query})['text']
    
    return hypothetical_answer.strip()

# Function to initialize the reranker model
def initialize_reranker_model():
    """
    Initializes the FlagReranker model for reranking retrieved documents.
    Returns:
    - FlagReranker: The initialized reranker model.
    """
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# Function to process the query and generate a response
def process_query(query, llm, embeddings, db, reranker_model):
    """
    Processes a query by generating a hypothetical answer, retrieving documents,
    reranking them, and generating a response based on the context.
    Parameters:
    - query (str): The user query.
    - llm (AzureChatOpenAI): The language model for generating responses.
    - embeddings (AzureOpenAIEmbeddings): The embeddings model for similarity search.
    - db (Chroma): The Chroma database for document retrieval.
    - reranker_model (FlagReranker): The reranker model for sorting relevant documents.
    Returns:
    - str: The generated response.
    """
    # Step 1: Generate a hypothetical answer for query enhancement
    hypothetical_answer = generate_hypothetical_answer(query, llm)
    
    # Step 2: Generate embeddings for the hypothetical answer
    hypothetical_embedding = embeddings.embed_query(hypothetical_answer)
    
    # Step 3: Retrieve similar documents using embeddings
    retrieved_docs = db.similarity_search_by_vector(hypothetical_embedding)
    
    # Step 4: Prepare documents for reranking
    reranking_docs = [doc.page_content for doc in retrieved_docs]
    qd_list = [[query, doc] for doc in reranking_docs]
    
    # Step 5: Compute reranking scores and sort documents
    scores = reranker_model.compute_score(qd_list)
    doc_score_pairs = sorted(zip(reranking_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [pair[0] for pair in doc_score_pairs]

    # Step 6: Construct context from the top reranked documents
    context = "\n".join(reranked_docs)

    # Step 7: Define the prompt for response generation
    prompt = PromptTemplate(
        input_variables=["human_input", "context"],
        template="""You are an AI having a conversation with a human.
                    Given the following extracted parts of a long document and a question, create a final answer.

                    Context: {context}

                    Human: {human_input}
                    AI:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Step 8: Generate and return the response
    chain_res = chain.invoke({'human_input': query, 'context': context})
    return chain_res['text'].strip()

# Main function to handle script execution
def main():
    """
    Main function to initialize components and process a query passed as a command-line argument.
    """
    # Initialize LLM, embeddings, and database
    llm, embeddings, db = llm_and_db()
    reranker_model = initialize_reranker_model()

    # Check if a query is provided via command-line arguments
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = process_query(query, llm, embeddings, db, reranker_model)
        print(result)

# Entry point for the script
if __name__ == "__main__":
    main()
