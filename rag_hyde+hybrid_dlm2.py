import os
import sys
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from FlagEmbedding import FlagReranker
from config_loader import load_config

# Function to set up environment variables for Azure OpenAI API
def setup_environment(api_version, api_key):
    """
    Sets up the necessary environment variables for Azure OpenAI API.
    
    Parameters:
    - api_version (str): The version of the Azure OpenAI API to use.
    - api_key (str): The Azure OpenAI API key for authentication.
    """
    os.environ['OPENAI_API_VERSION'] = api_version
    os.environ['AZURE_OPENAI_API_KEY'] = api_key


# Function to initialize the language model (LLM), embeddings, and document database
def llm_and_db():
    """
    Initializes the Azure language model, embeddings for similarity search, and the Chroma document database.
    
    Returns:
    - llm (AzureChatOpenAI): The initialized language model.
    - embeddings (AzureOpenAIEmbeddings): The embeddings model for similarity search.
    - db (Chroma): The Chroma database for document retrieval.
    """

    config = load_config()  # Load configuration settings from config.yaml
    print(f'Configs: {config}')

    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], api_key=config['azure_openai']['azure_api_key'])

    # Initialize the Azure language model for generating responses
    llm = AzureChatOpenAI(azure_deployment=config['azure_openai']['llm']['azure_deployment_llm'], azure_endpoint=config['azure_openai']['azure_api_endpoint'])
    
    # Initialize the embeddings model to embed text for similarity search
    embeddings = AzureOpenAIEmbeddings(azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'], azure_endpoint=config['azure_openai']['azure_api_endpoint'], chunk_size=100)
    
    # Initialize the Chroma database to store documents
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    return llm, embeddings, db


# Function to generate a hypothetical answer to enhance query understanding (HyDE)
def generate_hypothetical_answer(query, llm):
    """
    Generates a hypothetical answer for a given query using the LLM to improve understanding.
    
    Parameters:
    - query (str): The input question to generate a hypothetical answer for.
    - llm (AzureChatOpenAI): The language model used to generate the answer.
    
    Returns:
    - str: A generated hypothetical answer based on the query.
    """
    prompt = PromptTemplate(input_variables=["human_input"], template="You are an AI assistant. Generate a hypothetical answer based on the following query: {human_input}")
    chain = LLMChain(llm=llm, prompt=prompt)
    hypothetical_answer = chain.invoke({'human_input': query})['text']
    return hypothetical_answer


# Function to create an ensemble retriever combining dense and sparse retrieval methods
def create_ensemble_retriever(db):
    """
    Combines dense retrieval from Chroma with sparse retrieval using BM25 into an ensemble retriever.
    
    Parameters:
    - db (Chroma): The Chroma database for dense document retrieval.
    
    Returns:
    - EnsembleRetriever: The ensemble retriever combining both dense and sparse retrieval methods.
    """
    # Retrieve documents from the Chroma database
    documents = db.similarity_search("*")
    doc_texts = [doc.page_content for doc in documents]

    # Initialize the BM25 retriever with document texts for sparse retrieval
    bm25_retriever = BM25Retriever.from_texts(doc_texts)

    # Combine dense and sparse retrievers into an ensemble
    return EnsembleRetriever(retrievers=[db.as_retriever(), bm25_retriever], weights=[0.6, 0.4])


# Function to initialize the reranker model for sorting documents based on relevance
def initialize_reranker_model():
    """
    Initializes the reranker model for ranking documents based on their relevance to the query.
    
    Returns:
    - FlagReranker: The initialized reranker model used for reranking the retrieved documents.
    """
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)


# Function to process a query, generate a response, and rank documents based on relevance
def process_query(query, ensemble_retriever, llm, embeddings, db, reranker_model):
    """
    Processes a user's query by generating a hypothetical answer, retrieving relevant documents,
    reranking them based on relevance, and then generating a final response using the language model.
    
    Parameters:
    - query (str): The user's query.
    - ensemble_retriever (EnsembleRetriever): The retriever combining dense and sparse retrieval methods.
    - llm (AzureChatOpenAI): The language model used for generating responses.
    - embeddings (AzureOpenAIEmbeddings): The embeddings model for similarity search.
    - db (Chroma): The Chroma database used for document retrieval.
    - reranker_model (FlagReranker): The model used for reranking retrieved documents.
    
    Returns:
    - str: The generated response based on the retrieved and reranked documents.
    """
    # Generate a hypothetical answer to enhance query understanding
    hypothetical_answer = generate_hypothetical_answer(query, llm)

    # Convert the hypothetical answer into embeddings for document retrieval
    hypothetical_embedding = embeddings.embed_query(hypothetical_answer)

    # Retrieve relevant documents using both dense and sparse retrieval methods
    dense_docs = db.similarity_search_by_vector(hypothetical_embedding)
    sparse_docs = ensemble_retriever.get_relevant_documents(query)

    # Combine and deduplicate retrieved documents
    all_docs = dense_docs + sparse_docs
    
    # Extract text content from the retrieved documents for reranking
    reranking_docs = [doc.page_content for doc in all_docs]
    
    # Prepare query-document pairs for reranking based on relevance
    qd_list = [[query, doc] for doc in reranking_docs]
    
    # Calculate relevance scores for reranking the documents
    scores = reranker_model.compute_score(qd_list)

    # Sort the documents by their relevance score in descending order
    doc_score_pairs = sorted(zip(reranking_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [pair[0] for pair in doc_score_pairs]

    # Construct the context from the reranked documents
    context = "\n".join(reranked_docs)

    # Define the prompt template to generate the final response
    prompt = PromptTemplate(input_variables=["human_input", "context"], template="""You are an AI assistant. Based on the following context and question, provide a clear and concise response.

    Context: {context}

    Question: {human_input}

    Answer:""")

    # Generate the final response using the LLM
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({'human_input': query, 'context': context})['text'].strip()


# Main function for command-line usage
def main():
    """
    Main function to initialize the necessary components and process a query passed as a command-line argument.
    """
    # Initialize the LLM, embeddings, and document database
    llm, embeddings, db = llm_and_db()
    # Create the ensemble retriever for hybrid document retrieval
    ensemble_retriever = create_ensemble_retriever(db)
    # Initialize the reranker model for reranking retrieved documents
    reranker_model = initialize_reranker_model()

    # Check if a query is passed via command line arguments
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = process_query(query, ensemble_retriever, llm, embeddings, db, reranker_model)
        print(result)  # Print the result for automation scripts to capture

# Entry point for the script
if __name__ == "__main__":
    main()
