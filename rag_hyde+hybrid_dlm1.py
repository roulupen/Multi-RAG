import os
import sys
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from BCEmbedding import RerankerModel
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


# Function to initialize the language model, embeddings, and database
def llm_and_db():
    """
    Initializes the Azure language model, embeddings, and Chroma database for document retrieval.
    
    Returns:
    - llm (AzureChatOpenAI): The initialized language model for generating responses.
    - embeddings (AzureOpenAIEmbeddings): The embeddings model used for similarity search.
    - db (Chroma): The Chroma database for document retrieval.
    """

    # Load configuration settings from config.yaml
    config = load_config()
    print(f'Configs: {config}')

    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], api_key=config['azure_openai']['azure_api_key'])

    # Initialize Azure language model
    llm = AzureChatOpenAI(azure_deployment=config['azure_openai']['llm']['azure_deployment_llm'], azure_endpoint=config['azure_openai']['azure_api_endpoint'])
    
    # Initialize embeddings for similarity search
    embeddings = AzureOpenAIEmbeddings(azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'], azure_endpoint=config['azure_openai']['azure_api_endpoint'], chunk_size=100)
    
    # Initialize Chroma database for document retrieval
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    return llm, embeddings, db


# Function to generate a hypothetical answer for query enhancement (HyDE)
def generate_hypothetical_answer(query, llm):
    """
    Generates a hypothetical answer based on the input query using the language model.
    
    Parameters:
    - query (str): The user's query to enhance.
    - llm (AzureChatOpenAI): The language model to generate the hypothetical answer.
    
    Returns:
    - str: A generated hypothetical answer based on the query.
    """
    prompt = PromptTemplate(input_variables=["human_input"], template="You are an AI assistant. Generate a hypothetical answer based on the following query: {human_input}")
    chain = LLMChain(llm=llm, prompt=prompt)
    hypothetical_answer = chain.invoke({'human_input': query})['text']
    return hypothetical_answer


# Function to create an ensemble retriever that combines dense and sparse retrievers
def create_ensemble_retriever(db):
    """
    Combines dense retrieval from Chroma and sparse retrieval using BM25 into an ensemble retriever.
    
    Parameters:
    - db (Chroma): The Chroma database for dense retrieval.
    
    Returns:
    - EnsembleRetriever: The ensemble retriever combining dense and sparse retrievals.
    """
    # Retrieve all documents from ChromaDB
    documents = db.similarity_search("*")
    doc_texts = [doc.page_content for doc in documents]

    # Initialize BM25 retriever with the document texts
    bm25_retriever = BM25Retriever.from_texts(doc_texts)

    # Combine dense and sparse retrievers into an ensemble
    return EnsembleRetriever(retrievers=[db.as_retriever(), bm25_retriever], weights=[0.6, 0.4])


# Function to initialize the reranker model for document reranking
def initialize_reranker_model():
    """
    Initializes the reranker model for reranking the retrieved documents.
    
    Returns:
    - RerankerModel: The initialized reranker model.
    """
    return RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")


# Function to process a query and generate a response
def process_query(query, ensemble_retriever, llm, embeddings, db, reranker_model):
    """
    Processes a user query by generating a hypothetical answer, retrieving relevant documents, 
    reranking them, and generating a final response using the language model.
    
    Parameters:
    - query (str): The user's query.
    - ensemble_retriever (EnsembleRetriever): The ensemble retriever for combining dense and sparse retrievals.
    - llm (AzureChatOpenAI): The language model for generating responses.
    - embeddings (AzureOpenAIEmbeddings): The embeddings model for similarity search.
    - db (Chroma): The Chroma database for document retrieval.
    - reranker_model (RerankerModel): The reranker model for sorting retrieved documents.
    
    Returns:
    - str: The generated response based on the retrieved and reranked documents.
    """
    # Generate hypothetical answer using HyDE to improve query understanding
    hypothetical_answer = generate_hypothetical_answer(query, llm)

    # Convert the hypothetical answer to embeddings for document retrieval
    hypothetical_embedding = embeddings.embed_query(hypothetical_answer)

    # Retrieve relevant documents using dense and sparse methods
    dense_docs = db.similarity_search_by_vector(hypothetical_embedding)
    sparse_docs = ensemble_retriever.get_relevant_documents(query)

    # Combine and deduplicate retrieved documents
    all_docs = dense_docs + sparse_docs
    
    # Get the content from the retrieved documents
    docs = [doc.page_content for doc in all_docs]

    # Rerank the retrieved documents
    reranked_docs = reranker_model.rerank(query, docs)

    # Extract the passages from the reranked documents
    reranked_passages = reranked_docs.get("rerank_passages", [])

    # Construct context from the top reranked passages
    context = "\n\n".join(reranked_passages)

    # Define the prompt template for generating the final answer
    prompt = PromptTemplate(input_variables=["human_input", "context"], template="""You are an AI assistant. Based on the following context and question, provide a clear and concise response.

    Context: {context}

    Question: {human_input}

    Answer:""")

    # Generate and return the final response
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({'human_input': query, 'context': context})['text'].strip()


# Main function to handle command-line usage
def main():
    """
    Main function to initialize components and process a query passed as a command-line argument.
    """
    # Initialize components
    llm, embeddings, db = llm_and_db()
    ensemble_retriever = create_ensemble_retriever(db)
    reranker_model = initialize_reranker_model()

    # Check if query is passed via command line
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = process_query(query, ensemble_retriever, llm, embeddings, db, reranker_model)
        print(result)  # Print the result for automation scripts

# Entry point for the script
if __name__ == "__main__":
    main()
