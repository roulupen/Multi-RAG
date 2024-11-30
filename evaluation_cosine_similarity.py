import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import AzureOpenAIEmbeddings
from config_loader import load_config

# Function to set up environment variables for Azure OpenAI API
def setup_environment(api_version, api_key, azure_endpoint):
    """
    Sets up the necessary environment variables for Azure OpenAI API.
    
    Parameters:
    - api_version (str): The version of the Azure OpenAI API to use.
    - api_key (str): The Azure OpenAI API key for authentication.
    - azure_endpoint (str): The Azure OpenAI endpoint for authentication.
    """
    os.environ['OPENAI_API_VERSION'] = api_version
    os.environ['AZURE_OPENAI_API_KEY'] = api_key
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint

def get_embeddings(texts):
    """
    Generate embeddings for a list of texts using Azure OpenAI embeddings.
    
    :param texts: List of strings to embed.
    :param embeddings: Instance of AzureOpenAIEmbeddings.
    :return: List of embeddings.
    """
    print("Generating embeddings...")
    # Initialize Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=32)

    return [embeddings.embed_query(text) for text in tqdm(texts, desc="Embedding Progress")]

def calculate_cosine_similarity(excel_path, output_path):
    """
    Calculate cosine similarity using Azure OpenAI embeddings (via langchain).
    
    :param excel_path: Path to input Excel file.
    :param output_path: Path to save output Excel file with cosine similarity columns
    """

    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Check for "ground truth" column
    if "ground truth" not in df.columns:
        raise ValueError("The 'ground truth' column is missing from the input file.")

    # Get response columns (excluding S.No, Query, and Ground truth)
    response_columns = [col for col in df.columns if col not in ["S.No", "query", "ground truth"]]

    # Generate embeddings for ground truth
    ground_truth_texts = df["ground truth"].fillna("").tolist()
    ground_truth_embeddings = get_embeddings(ground_truth_texts)

    # Add cosine similarity columns to the DataFrame
    print("Calculating cosine similarities for RAG-generated answers...")
    for response_col in tqdm(response_columns, desc="Processing RAG Techniques", unit="technique"):
        # Generate embeddings for the response column
        response_texts = df[response_col].fillna("").tolist()
        response_embeddings = get_embeddings(response_texts)

        # Compute cosine similarities row by row
        similarities = [
            cosine_similarity([ground_truth_embeddings[i]], [response_embeddings[i]])[0][0]
            for i in range(len(ground_truth_embeddings))
        ]

        # Add the similarity as a new column
        df[f"{response_col}_cs"] = similarities

    # Save the updated DataFrame to a new Excel file
    df.to_excel(output_path, index=False)
    print(f"Cosine similarity results saved to {output_path}")

def summarize_results(results_file, res_summary_output_path):

    results_df = pd.read_excel(results_file)

    mean_results_file = results_df.iloc[:, 21:].mean() # after 21 columns we have the similarity scores

    output_df = pd.DataFrame({
    'rag_method_name': mean_results_file.index,
    'cosine_similarity_mean': mean_results_file.values})

    # Save the output DataFrame to an Excel file
    output_df.to_excel(res_summary_output_path, index=False)
    print(f"Results summary saved to {res_summary_output_path}")


# Run the main function when the script is executed
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a directory path from the command line.")
    parser.add_argument("retriever_results_file", type=str, help="Path to the directory to be processed.")
 
    # Parse the command-line argument
    args = parser.parse_args()

    config = load_config()  # Load configuration settings from config.yaml
    print(f'Configs: {config}')

    # Set up environment variables for Azure API
    setup_environment(api_version=config['azure_openai']['llm']['azure_api_version_llm'], 
                    api_key=config['azure_openai']['azure_api_key'], azure_endpoint=config['azure_openai']['azure_api_endpoint'])

    # Calculate cosine similarity scores
    output_file = f"output/{os.path.basename(args.retriever_results_file).split('.')[0]}_cosine_similarity.xlsx"
    calculate_cosine_similarity(args.retriever_results_file, output_file)

    # Summarize cosine similarity results across all the methods
    res_summary_output_path = f"output/{os.path.basename(output_file).split('.')[0]}_summary.xlsx"
    summarize_results(output_file, res_summary_output_path)