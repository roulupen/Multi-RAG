import os
import argparse
import subprocess
import pandas as pd
from chromadb.config import Settings
from chromadb import Client
import warnings

# Suppress warnings from pandas or other libraries
warnings.filterwarnings('ignore')

# Define lists of splitter and retriever script file names
splitter_scripts = [
    "rag_recursive_splitter.py",     # Script for recursive splitting of input
    "rag_semantic_splitter.py",      # Script for semantic splitting of input
    "rag_sentence_splitter.py"       # Script for sentence-level splitting
]

retrieving_scripts = [
    "rag_hybrid_dlm1.py",            # Hybrid retriever script using DLM1
    "rag_hybrid_dlm2.py",            # Hybrid retriever script using DLM2
    "rag_hyde_dlm1.py",              # HYDE (Hypothetical Document Embeddings) retriever using DLM1
    "rag_hyde_dlm2.py",              # HYDE retriever using DLM2
    "rag_hyde+hybrid_dlm1.py",       # Combined HYDE + Hybrid retriever with DLM1
    "rag_hyde+hybrid_dlm2.py"        # Combined HYDE + Hybrid retriever with DLM2
]

def reset_database(persist_directory="./chroma_db"):
    """
    Clears the ChromaDB database to ensure clean data for each splitter-retriever combination.
    
    Parameters:
    - persist_directory (str): Directory where ChromaDB data is stored.
    """
    client = Client(Settings(
        persist_directory=persist_directory,
        allow_reset=True  # Enable resetting of the database
    ))
    client.reset()  # Clears all data in the ChromaDB database
    print("ChromaDB reset successfully.")

def process_queries_with_combinations(query_excel_path, docuemnt_directory="./docs/pdf"):
    """
    Processes queries using all combinations of splitter and retriever scripts, 
    and saves the results to an Excel file.

    Parameters:
    - query_excel_path (str): Path to the Excel file containing queries.
    """
    # Check if the Excel file exists
    if not os.path.exists(query_excel_path):
        print(f"Excel file not found: {query_excel_path}")
        return

    # Load queries from the specified Excel file
    queries_df = pd.read_excel(query_excel_path)

    # Create a copy of the original DataFrame to store results
    results_df = queries_df.copy()

    # Loop through each splitter script
    for splitter_script in splitter_scripts:
        if not os.path.exists(splitter_script):  # Check if the splitter script exists
            print(f"Splitter script not found: {splitter_script}")
            continue

        # Reset the ChromaDB before running the splitter script
        reset_database()

        print(f"Running splitter script: {splitter_script}")
        subprocess.run(["python", splitter_script, docuemnt_directory])  # Execute the splitter script

        # Loop through each retriever script
        for retriever_script in retrieving_scripts:
            if not os.path.exists(retriever_script):  # Check if the retriever script exists
                print(f"Retriever script not found: {retriever_script}")
                continue

            print(f"Running retriever script: {retriever_script} for each query...")

            # Create a column name to store responses from this combination
            column_name = f"{splitter_script.split('.')[0]}_{retriever_script.split('.')[0]}"

            # Store the responses for each query in this list
            responses = []
            for _, row in queries_df.iterrows():
                query = row['query']  # Extract the query from the DataFrame
                print(f"Processing query: {query} with {retriever_script}")

                # Run the retriever script with the query and capture the response
                response = subprocess.check_output(
                    ["python", retriever_script, query],
                    universal_newlines=True
                ).strip()

                # Append the response for the current query to the responses list
                responses.append(response)

            # Add the responses as a new column in the results DataFrame
            results_df[column_name] = responses

    # Save the final results to an Excel file
    output_file = f"output/retriever_results_combinations_{os.path.basename(query_excel_path)}.xlsx"
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process queries from an Excel file and use documents from a directory.")
    parser.add_argument("query_excel_path", type=str, help="Path to the Query Excel file containing queries.")
    parser.add_argument("documents_directory", type=str, help="Path to the directory containing documents.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Specify the path to the Excel file containing queries
    process_queries_with_combinations(args.query_excel_path, args.documents_directory)
