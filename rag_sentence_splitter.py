import os
import argparse
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
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


# Function to load documents from a specified directory
def load_documents(directory_path):
    """
    Loads documents from the specified directory and returns them as a list.
    
    Supports PDF, DOCX/DOC, and TXT file formats.

    Parameters:
    - directory_path (str): Path to the directory containing the documents.

    Returns:
    - documents_query (list): A list of documents loaded from the directory.
    """
    documents_query = []
    files = os.listdir(directory_path)  # List all files in the directory
    file_options = np.array(files)  # Convert to a numpy array for easy iteration

    # Loop through each file in the directory
    for file in file_options:
        file_path = os.path.join(directory_path, file)  # Full file path
        if file.endswith('.pdf'):  # Process PDF files
            loader = PyPDFLoader(file_path)
            documents_query.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):  # Process DOCX and DOC files
            loader = Docx2txtLoader(file_path)
            documents_query.extend(loader.load())
        elif file.endswith('.txt'):  # Process TXT files
            print('Processing text file:', file)
            try:
                loader = TextLoader(file_path)
                documents_query.extend(loader.load())
            except Exception as e:
                print(f"Error while processing text file {file}: {e}")

    return documents_query


# Function to split documents into sentence-based chunks
def sentence_split_documents(documents):
    """
    Splits documents into sentence-based chunks using SentenceTransformersTokenTextSplitter.

    Parameters:
    - documents (list): A list of documents to be split.

    Returns:
    - list: A list of documents split into sentence chunks.
    """
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
    return text_splitter.split_documents(documents)


# Function to create an embedding database from documents
def create_embedding_database(docs, azure_deployment, azure_endpoint, persist_directory="./chroma_db"):
    """
    Creates an embedding database from the given documents using Chroma.

    Parameters:
    - docs (list): A list of documents to create embeddings from.
    - azure_deployment (str): The deployment name for Azure OpenAI Embeddings.
    - azure_endpoint (str): The endpoint URL for Azure OpenAI API.
    - persist_directory (str): Directory to persist the Chroma database.

    Returns:
    - db (Chroma): The created Chroma database.
    """
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        chunk_size=16  # Set the chunk size for embedding generation
    )
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    return db


# Main function to load documents, split them, and create an embedding database
def main(directory_path):
    """
    Main function to initialize the environment, load documents from a directory,
    split them into sentence chunks, and create an embedding database.
    """
    config = load_config()  # Load configuration settings from config.yaml

    

    # Set up environment variables for Azure API
    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], api_key=config['azure_openai']['azure_api_key'])
    
    # Define the directory where the documents are stored
    #directory_path = 'docs'
    # Load documents from the specified directory
    documents_query = load_documents(directory_path)
    
    # Use sentence-based chunking to split documents
    docs = sentence_split_documents(documents_query)
    
    # Create the embedding database from the split documents
    db = create_embedding_database(
        docs,
        azure_deployment=config['azure_openai']['embedding']['azure_deployment_emb'],
        azure_endpoint=config['azure_openai']['azure_api_endpoint']
    )
    
    # Check if the database was created successfully
    if db:
        print("Database created successfully with sentence-based splitting.")
    else:
        print("Database creation failed.")


# Run the main function when the script is executed
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a directory path from the command line.")
    parser.add_argument("directory_path", type=str, help="Path to the directory to be processed.")

    # Parse the command-line argument
    args = parser.parse_args()

    main(args.directory_path)
